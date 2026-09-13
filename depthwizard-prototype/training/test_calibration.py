import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

import numpy as np
from calibration import (
    calibrate_height,
    calibrate_with_reference_dem,
    load_gcps,
    read_geotiff_info,
    _calibrate_with_local_dem,
)


def test_global_recovers_scale_and_offset():
    depth = np.arange(1600, dtype=np.float32).reshape(40, 40)
    elevation = 2.5 * depth + 13.0
    _, info = calibrate_with_reference_dem(depth, elevation)
    assert abs(info["scale_a"] - 2.5) < 1e-5
    assert abs(info["offset_b"] - 13.0) < 1e-5
    assert info["fit_rmse_m"] < 1e-4


def test_gcp_mode_has_priority():
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    gcps = [{"x": 0, "y": 0, "elevation": 5}, {"x": 5, "y": 5, "elevation": 60}, {"x": 9, "y": 9, "elevation": 104}]
    _, info = calibrate_height(depth, gcps=gcps)
    assert info["method"] == "gcp_linear_regression"
    assert info["fit_rmse_m"] < 1e-4
    assert info["absolute"] is True


def test_relative_mode_is_not_absolute():
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    _, info = calibrate_height(depth, min_height=0, max_height=10)
    assert info["absolute"] is False


def test_reference_dem_rejects_flat_overlap():
    depth = np.ones((10, 10), dtype=np.float32)
    elevation = np.ones((10, 10), dtype=np.float32)
    try:
        calibrate_with_reference_dem(depth, elevation)
    except ValueError as exc:
        assert "variation" in str(exc)
    else:
        raise AssertionError("flat reference DEM should be rejected")


def test_local_dem_tile_blend_recovers_scale(tmp_path):
    depth = np.tile(np.arange(64, dtype=np.float32), (64, 1))
    elevation = 3.0 * depth + 7.0
    dsm, info = _calibrate_with_local_dem(depth, elevation, tile_size=32)
    assert info["method"] == "reference_dem_local_tile_regression"
    assert info["absolute"] is True
    valid = np.isfinite(dsm)
    assert valid.mean() > 0.5
    assert np.nanmax(np.abs(dsm[valid] - elevation[valid])) < 1.0


def test_local_dem_reports_nodata_declared_flag():
    depth = np.tile(np.arange(64, dtype=np.float32), (64, 1))
    elevation = 3.0 * depth + 7.0
    _, info = _calibrate_with_local_dem(depth, elevation, tile_size=32, dem_nodata_declared=True)
    assert info["dem_nodata_declared"] is True
    _, info = _calibrate_with_local_dem(depth, elevation, tile_size=32, dem_nodata_declared=False)
    assert info["dem_nodata_declared"] is False


def test_reference_dem_undeclared_nodata_note_is_transparent():
    depth = np.arange(1600, dtype=np.float32).reshape(40, 40)
    elevation = 2.5 * depth + 13.0
    _, declared = calibrate_with_reference_dem(depth, elevation, dem_nodata_declared=True)
    _, undeclared = calibrate_with_reference_dem(depth, elevation, dem_nodata_declared=False)
    assert "nodata" not in declared["note"].lower()
    assert "nodata" in undeclared["note"].lower()


def test_gcp_and_reference_dem_precedence_is_explicit(tmp_path):
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    gcps = [{"x": 0, "y": 0, "elevation": 5}, {"x": 5, "y": 5, "elevation": 60}, {"x": 9, "y": 9, "elevation": 104}]
    _, info = calibrate_height(depth, gcps=gcps, reference_dem_path=tmp_path / "unused.tif")
    assert info["method"] == "gcp_linear_regression"
    assert "precedence" in info["note"].lower()


def test_geographic_crs_is_rejected():
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    try:
        calibrate_height(depth, geo_info={"is_geographic": True})
    except ValueError as exc:
        assert "geographic" in str(exc).lower()
    else:
        raise AssertionError("a geographic-CRS GeoTIFF should be rejected")


def _write_tiny_geotiff(path, crs):
    import rasterio
    from rasterio.transform import from_origin

    data = np.zeros((1, 8, 8), dtype="float32")
    transform = from_origin(0, 8, 1, 1)
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1,
                        dtype="float32", crs=crs, transform=transform) as dst:
        dst.write(data)


def test_read_geotiff_info_flags_geographic_crs(tmp_path):
    geo_path = tmp_path / "geographic.tif"
    _write_tiny_geotiff(geo_path, "EPSG:4326")
    info = read_geotiff_info(geo_path)
    assert info["is_geographic"] is True

    projected_path = tmp_path / "projected.tif"
    _write_tiny_geotiff(projected_path, "EPSG:32633")
    info = read_geotiff_info(projected_path)
    assert info["is_geographic"] is False


def test_load_gcps_geojson_converts_crs_coordinates_to_pixels(tmp_path):
    geo_path = tmp_path / "target.tif"
    _write_tiny_geotiff(geo_path, "EPSG:32633")
    geo_info = read_geotiff_info(geo_path)

    # Pixel (2, 3) under from_origin(0, 8, 1, 1) is at CRS (x=2, y=5).
    geojson_path = tmp_path / "gcps.geojson"
    geojson_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [2.0, 5.0]},
            "properties": {"elevation": 42.0},
        }],
    }))

    points = load_gcps(geojson_path, transform=geo_info["transform"])
    assert len(points) == 1
    assert abs(points[0]["x"] - 2.0) < 1e-6
    assert abs(points[0]["y"] - 3.0) < 1e-6
    assert points[0]["elevation"] == 42.0


def test_load_gcps_geojson_without_transform_keeps_raw_coordinates(tmp_path):
    geojson_path = tmp_path / "gcps.geojson"
    geojson_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [7.0, 9.0]},
            "properties": {"elevation": 11.0},
        }],
    }))

    points = load_gcps(geojson_path, transform=None)
    assert points[0]["x"] == 7.0
    assert points[0]["y"] == 9.0
