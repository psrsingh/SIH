"""
DepthWizard metric calibration.

The monocular model produces relative / scale-ambiguous depth.

For a GeoTIFF with a reference DEM:

    Relative Depth
          ↓
    Reference DEM
          ↓
    Linear regression
          ↓
    Metric elevation

Mathematically:

    Elevation = a * Depth + b

For PNG/JPG without geographic information, we generate
a relative DSM for visualization only.
"""

from pathlib import Path
from typing import Optional
import csv
import json

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================
# GEOTIFF INFORMATION
# ============================================================

def read_geotiff_info(path: Path) -> Optional[dict]:
    try:
        import rasterio

        with rasterio.open(path) as src:
            return {
                "crs": str(src.crs) if src.crs else None,
                "bounds": list(src.bounds),
                "resolution": list(src.res),
                "width": src.width,
                "height": src.height,
                "transform": list(src.transform),
                "count": src.count,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
            }

    except Exception as exc:
        logger.warning("GeoTIFF metadata error: %s", exc)
        return None


# ============================================================
# VALID SAMPLE EXTRACTION
# ============================================================

def _valid_samples(depth, elevation, max_samples=100000):
    d = np.asarray(depth, dtype=np.float64)
    z = np.asarray(elevation, dtype=np.float64)

    if d.shape != z.shape:
        raise ValueError("Depth and elevation arrays must have matching dimensions.")

    mask = np.isfinite(d) & np.isfinite(z)
    d = d[mask]
    z = z[mask]

    # Remove extreme DEM outliers.
    if len(z) > 20:
        low, high = np.percentile(z, [1, 99])
        keep = (z >= low) & (z <= high)
        d = d[keep]
        z = z[keep]

    # Limit number of regression samples.
    if len(d) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(d), max_samples, replace=False)
        d = d[indices]
        z = z[indices]

    return d, z


# ============================================================
# LINEAR REGRESSION
# ============================================================

def _fit_linear(x, y):
    """Fit y = a*x + b."""
    A = np.column_stack([x, np.ones_like(x)])
    coefficients, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coefficients

    prediction = a * x + b
    residual = prediction - y
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))

    return float(a), float(b), rmse, mae


def _fit_metrics(prediction, target):
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if residual.size == 0:
        return None
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    correlation = float(np.corrcoef(prediction, target)[0, 1]) if target.size > 1 else 0.0
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "rmse_m": float(np.sqrt(np.mean(residual ** 2))),
        "mae_m": float(np.mean(np.abs(residual))),
        "correlation": correlation,
        "sample_count": int(residual.size),
    }


# ============================================================
# CALIBRATE USING REFERENCE DEM
# ============================================================

def calibrate_with_reference_dem(rel_depth: np.ndarray, reference_elevation: np.ndarray):
    """
    Convert relative depth to metric elevation.

    Tests both:
        Elevation = a * Depth + b
    and:
        Elevation = a * (-Depth) + b
    because different monocular depth models may have different depth
    orientations (near=high value vs near=low value).
    """
    d, z = _valid_samples(rel_depth, reference_elevation)

    if len(d) < 20:
        raise ValueError(
            "Not enough valid overlapping pixels between the depth "
            "prediction and reference DEM."
        )
    if np.ptp(d) < 1e-8 or np.ptp(z) < 1e-8:
        raise ValueError("Reference DEM overlap must contain elevation and depth variation.")

    candidates = []

    a, b, rmse, mae = _fit_linear(d, z)
    candidates.append({"orientation": "direct", "scale_a": a, "offset_b": b,
                        "rmse_m": rmse, "mae_m": mae})

    a, b, rmse, mae = _fit_linear(-d, z)
    candidates.append({"orientation": "inverted", "scale_a": a, "offset_b": b,
                        "rmse_m": rmse, "mae_m": mae})

    best = min(candidates, key=lambda item: item["rmse_m"])

    if best["orientation"] == "direct":
        dsm = best["scale_a"] * rel_depth + best["offset_b"]
    else:
        dsm = best["scale_a"] * (-rel_depth) + best["offset_b"]

    dsm = np.asarray(dsm, dtype=np.float32)

    calibration_info = {
        "method": "reference_dem_linear_regression",
        "orientation": best["orientation"],
        "scale_a": best["scale_a"],
        "offset_b": best["offset_b"],
        "fit_rmse_m": best["rmse_m"],
        "fit_mae_m": best["mae_m"],
        "fit_correlation": _fit_metrics(
            best["scale_a"] * (d if best["orientation"] == "direct" else -d)
            + best["offset_b"], z
        )["correlation"],
        "sample_count": len(d),
        "absolute": True,
        "note": (
            "Relative monocular depth was converted to metric elevation "
            "using least-squares regression against the supplied "
            "reference DEM."
        ),
    }

    return dsm, calibration_info


def _calibrate_with_gcps(rel_depth, gcps):
    """Fit the same relative-depth model from ``x,y,elevation`` GCPs."""
    height, width = rel_depth.shape
    samples = []
    if not isinstance(gcps, (list, tuple)):
        raise ValueError("GCP data must be a list of points.")
    for point in gcps:
        try:
            x = int(round(float(point["x"])))
            y = int(round(float(point["y"])))
            elevation = float(point.get("elevation", point.get("z")))
        except (KeyError, TypeError, ValueError):
            continue
        # GCP coordinates are pixel coordinates: x is image column, y is row.
        if 0 <= x < width and 0 <= y < height and np.isfinite(elevation):
            samples.append((float(rel_depth[y, x]), elevation))
    if len(samples) < 3:
        raise ValueError("At least 3 valid GCPs are required for calibration.")

    d = np.asarray([item[0] for item in samples], dtype=np.float64)
    z = np.asarray([item[1] for item in samples], dtype=np.float64)
    if np.ptp(d) < 1e-8 or np.ptp(z) < 1e-8:
        raise ValueError("GCPs must contain variation in both depth and elevation.")
    candidates = []
    for orientation, values in (("direct", d), ("inverted", -d)):
        a, b, rmse, mae = _fit_linear(values, z)
        candidates.append((rmse, orientation, a, b, mae, values))
    _, orientation, a, b, mae, values = min(candidates, key=lambda item: item[0])
    prediction = a * values + b
    info = {
        "method": "gcp_linear_regression",
        "orientation": orientation,
        "scale_a": a,
        "offset_b": b,
        "fit_rmse_m": float(np.sqrt(np.mean((prediction - z) ** 2))),
        "fit_mae_m": mae,
        "fit_correlation": _fit_metrics(prediction, z)["correlation"],
        "sample_count": len(samples),
        "absolute": True,
    }
    signed_depth = rel_depth if orientation == "direct" else -rel_depth
    return (a * signed_depth + b).astype(np.float32), info


def _calibrate_with_local_dem(rel_depth, reference_elevation, tile_size=64):
    """Fit overlapping local tiles and blend their predictions by distance."""
    height, width = rel_depth.shape
    step = max(1, int(tile_size) // 2)
    prediction_sum = np.zeros_like(rel_depth, dtype=np.float64)
    weight_sum = np.zeros_like(rel_depth, dtype=np.float64)
    tile_errors = []
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            y1, x1 = min(y0 + tile_size, height), min(x0 + tile_size, width)
            d, z = _valid_samples(rel_depth[y0:y1, x0:x1], reference_elevation[y0:y1, x0:x1])
            if len(d) < 20 or np.ptp(d) < 1e-8:
                continue
            candidates = []
            for orientation, values in (("direct", d), ("inverted", -d)):
                a, b, rmse, mae = _fit_linear(values, z)
                candidates.append((rmse, orientation, a, b, mae))
            rmse, orientation, a, b, mae = min(candidates, key=lambda item: item[0])
            tile_depth = rel_depth[y0:y1, x0:x1]
            signed = tile_depth if orientation == "direct" else -tile_depth
            yy, xx = np.mgrid[y0:y1, x0:x1]
            cy, cx = (y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2
            weights = 1.0 / (1.0 + ((yy - cy) / max(tile_size, 1)) ** 2 + ((xx - cx) / max(tile_size, 1)) ** 2)
            prediction_sum[y0:y1, x0:x1] += weights * (a * signed + b)
            weight_sum[y0:y1, x0:x1] += weights
            tile_errors.append({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                                "rmse_m": rmse, "mae_m": mae, "sample_count": len(d)})
    coverage = weight_sum > 0
    if not tile_errors or np.count_nonzero(coverage) < rel_depth.size * 0.5:
        raise ValueError("Local calibration has insufficient valid tile coverage.")
    dsm = np.full_like(rel_depth, np.nan, dtype=np.float32)
    dsm[coverage] = (prediction_sum[coverage] / weight_sum[coverage]).astype(np.float32)
    valid_depth = rel_depth[coverage]
    valid_reference = reference_elevation[coverage]
    metrics = _fit_metrics(dsm[coverage], valid_reference)
    return dsm, {
        "method": "reference_dem_local_tile_regression",
        "tile_size_px": int(tile_size),
        "fit_rmse_m": metrics["rmse_m"],
        "fit_mae_m": metrics["mae_m"],
        "fit_correlation": metrics["correlation"],
        "sample_count": metrics["sample_count"],
        "tile_errors": tile_errors,
        "absolute": True,
    }


def load_gcps(path: Path):
    """Read GCPs from CSV or GeoJSON without adding a geometry dependency."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            points = list(csv.DictReader(handle))
            if not points:
                raise ValueError("GCP CSV is empty.")
            return points
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") == "FeatureCollection":
        points = []
        for feature in data.get("features", []):
            coords = feature.get("geometry", {}).get("coordinates", [])
            props = feature.get("properties", {})
            if len(coords) >= 2:
                points.append({"x": coords[0], "y": coords[1], "elevation": props.get("elevation", props.get("z"))})
        return points
    points = data if isinstance(data, list) else data.get("gcps", [])
    if not points:
        raise ValueError("GCP file contains no points.")
    return points


# ============================================================
# REPROJECT REFERENCE DEM ONTO THE OPTICAL IMAGE'S GRID
# ============================================================

def _read_reference_dem_on_target_grid(reference_path: Path, target_path: Path):
    """
    Reproject the reference DEM so it matches the optical GeoTIFF grid
    (the DEM and RGB image usually differ in CRS, resolution, dimensions
    and transform).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(target_path) as target:
        if not target.crs:
            raise ValueError("Target GeoTIFF has no CRS.")

        dst_shape = (target.height, target.width)
        dst_transform = target.transform
        dst_crs = target.crs

    with rasterio.open(reference_path) as src:
        if not src.crs:
            raise ValueError("Reference DEM has no CRS.")

        source = src.read(1).astype(np.float32)
        destination = np.full(dst_shape, np.nan, dtype=np.float32)

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return destination


# ============================================================
# MAIN CALIBRATION FUNCTION
# ============================================================

def calibrate_height(
    rel_depth: np.ndarray,
    min_height: float = 0.0,
    max_height: float = 50.0,
    geo_info: Optional[dict] = None,
    reference_dem_path: Optional[Path] = None,
    target_geotiff_path: Optional[Path] = None,
    gcps=None,
    local_tile_size: int = 64,
):
    """
    CASE 1 — PNG/JPG: relative depth -> relative DSM (visualization only).
    CASE 2 — GeoTIFF + reference DEM: relative depth + reference DEM ->
             linear regression -> absolute metric DSM.
    CASE 3 — GeoTIFF without a reference DEM: relative visualization DSM;
             this must NOT be treated as absolute elevation.
    """
    if gcps:
        return _calibrate_with_gcps(rel_depth, gcps)

    if not np.asarray(rel_depth).ndim == 2 or not np.isfinite(rel_depth).any():
        raise ValueError("Depth prediction contains no valid finite pixels.")
    if not np.isfinite(min_height) or not np.isfinite(max_height) or max_height <= min_height:
        raise ValueError("Height range must be finite and max_height must exceed min_height.")

    if reference_dem_path and target_geotiff_path and geo_info:
        reference = _read_reference_dem_on_target_grid(
            Path(reference_dem_path), Path(target_geotiff_path)
        )
        try:
            return _calibrate_with_local_dem(rel_depth, reference, local_tile_size)
        except ValueError:
            return calibrate_with_reference_dem(rel_depth, reference)

    d = rel_depth.astype(np.float32)
    finite = np.isfinite(d)
    d_min = np.min(d[finite])
    d_max = np.max(d[finite])
    d_norm = (d - d_min) / (d_max - d_min + 1e-8)
    dsm = min_height + d_norm * (max_height - min_height)

    info = {
        "method": "relative_minmax" if not geo_info else "relative_minmax_no_reference_dem",
        "scale_a": float(max_height - min_height),
        "offset_b": float(min_height),
        "absolute": False,
        "note": (
            "No reference DEM or GCP was supplied. The output is a "
            "relative DSM for visualization and is not an absolute "
            "survey-grade elevation product."
        ),
    }

    return dsm.astype(np.float32), info
