"""
DepthWizard FastAPI backend.

Pipeline:

RGB / GeoTIFF -> monocular depth -> relative depth -> (optional reference
DEM calibration) -> DSM -> (GeoTIFF DSM if georeferenced) -> 3D GLB terrain
"""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from depth_estimation import estimate_depth
from calibration import calibrate_height, load_gcps, read_geotiff_info
from mesh_builder import build_mesh_glb

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))
OUTPUT_DIR.mkdir(exist_ok=True)
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))
logger = logging.getLogger("depthwizard")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="DepthWizard Prototype API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _save_depth_preview(depth: np.ndarray, path: Path):
    d = depth.astype(np.float32)
    d_min = np.nanmin(d)
    d_max = np.nanmax(d)
    d = (d - d_min) / (d_max - d_min + 1e-8)

    r = np.clip(1.5 - np.abs(4 * d - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * d - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * d - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def _save_dsm_geotiff(dsm: np.ndarray, source_path: Path, output_path: Path,
                       calib_info: dict, is_absolute: bool):
    import rasterio

    with rasterio.open(source_path) as src:
        if not src.crs:
            raise ValueError("Source GeoTIFF has no CRS.")
        profile = src.profile.copy()
        profile.update(driver="GTiff", dtype="float32", count=1,
                        compress="deflate", predictor=3, nodata=np.nan)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(dsm.astype(np.float32), 1)
            # Tag the file honestly: a relative DSM carries CRS/transform
            # for convenience but is NOT survey-grade elevation, and a
            # downstream GIS user should be able to tell the difference
            # without re-reading this API's JSON response.
            dst.update_tags(
                depthwizard_calibration_method=calib_info.get("method", "unknown"),
                depthwizard_absolute_elevation=str(is_absolute),
            )


def _get_pixel_size(geo_info):
    if not geo_info or not geo_info.get("resolution"):
        return 1.0, 1.0
    rx, ry = geo_info["resolution"]
    return abs(float(rx)), abs(float(ry))


@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    reference_dem: UploadFile | None = File(None),
    gcp_file: UploadFile | None = File(None),
    min_height: float = Form(0.0),
    max_height: float = Form(50.0),
    mesh_resolution: int = Form(160),
    local_tile_size: int = Form(64),
):
    started = time.perf_counter()
    job_id = uuid.uuid4().hex[:10]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "input.png"
    suffix = Path(filename).suffix.lower() or ".png"
    if suffix not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        raise HTTPException(status_code=400, detail="Upload must be PNG, JPG, or GeoTIFF.")
    if mesh_resolution < 8 or mesh_resolution > 512:
        raise HTTPException(status_code=400, detail="mesh_resolution must be between 8 and 512.")
    if not np.isfinite(min_height) or not np.isfinite(max_height) or max_height <= min_height:
        raise HTTPException(status_code=400, detail="max_height must be finite and greater than min_height.")
    is_geotiff = suffix in (".tif", ".tiff")

    input_path = job_dir / f"input{suffix}"
    input_bytes = await file.read()
    if len(input_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_SIZE} bytes.")
    input_path.write_bytes(input_bytes)

    geo_info = None
    if is_geotiff:
        geo_info = read_geotiff_info(input_path)
        if not geo_info:
            return JSONResponse(status_code=400, content={
                "error": "The uploaded TIFF could not be read as a valid GeoTIFF."
            })
        if not geo_info.get("crs"):
            return JSONResponse(status_code=400, content={
                "error": "GeoTIFF does not contain a CRS."
            })

    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Unable to read image: {exc}"})

    try:
        rel_depth, method = await asyncio.to_thread(estimate_depth, img)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Depth estimation failed: {exc}"})

    reference_dem_path = None
    if reference_dem is not None:
        if not is_geotiff:
            return JSONResponse(status_code=400, content={
                "error": "Reference DEM calibration requires a GeoTIFF input."
            })
        ref_name = reference_dem.filename or "reference_dem.tif"
        ref_suffix = Path(ref_name).suffix.lower()
        if ref_suffix not in (".tif", ".tiff"):
            return JSONResponse(status_code=400, content={
                "error": "Reference DEM must be a GeoTIFF."
            })
        reference_dem_path = job_dir / f"reference_dem{ref_suffix}"
        reference_dem_path.write_bytes(await reference_dem.read())

    gcps = None
    if gcp_file is not None:
        gcp_name = gcp_file.filename or "gcps.csv"
        gcp_suffix = Path(gcp_name).suffix.lower()
        if gcp_suffix not in (".csv", ".json", ".geojson"):
            return JSONResponse(status_code=400, content={
                "error": "GCPs must be supplied as CSV, JSON, or GeoJSON."
            })
        gcp_path = job_dir / f"gcps{gcp_suffix}"
        gcp_path.write_bytes(await gcp_file.read())
        try:
            gcps = load_gcps(gcp_path)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": f"Unable to read GCPs: {exc}"})

    try:
        dsm, calib_info = calibrate_height(
            rel_depth,
            min_height=min_height,
            max_height=max_height,
            geo_info=geo_info,
            reference_dem_path=reference_dem_path,
            target_geotiff_path=input_path if reference_dem_path else None,
            gcps=gcps,
            local_tile_size=local_tile_size,
        )
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Calibration failed: {exc}"})

    _save_depth_preview(rel_depth, job_dir / "depth_preview.png")
    np.save(job_dir / "dsm.npy", dsm.astype(np.float32))

    absolute_dsm = bool(calib_info.get("absolute", False))
    synthetic_fallback = method == "fallback_synthetic"
    model_type = "synthetic-demo" if synthetic_fallback else (
        "GAMUS-finetuned" if method == "depth-anything-gamus" else "pretrained"
    )
    elevation_type = "absolute" if absolute_dsm else "relative"
    warning = (
        "This output is a relative DSM for visualization only. "
        "Absolute metric elevation requires a reference DEM or GCP calibration."
        if not absolute_dsm else
        "This output was calibrated to metric elevation using the supplied reference data."
    )

    dsm_url = None
    if is_geotiff and geo_info:
        dsm_path = job_dir / "dsm.tif"
        try:
            _save_dsm_geotiff(dsm, input_path, dsm_path, calib_info, absolute_dsm)
            dsm_url = f"/outputs/{job_id}/dsm.tif"
        except Exception as exc:
            return JSONResponse(status_code=500, content={
                "error": f"Failed to save DSM GeoTIFF: {exc}"
            })

    pixel_x, pixel_y = _get_pixel_size(geo_info)

    mesh_path = job_dir / "terrain.glb"
    try:
        await asyncio.to_thread(
            build_mesh_glb, img, dsm, mesh_path, resolution=mesh_resolution,
            pixel_size_x=pixel_x, pixel_size_y=pixel_y
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={
            "error": f"3D mesh generation failed: {exc}"
        })

    elapsed = time.perf_counter() - started
    logger.info("job=%s model=%s inference_method=%s calibration=%s elapsed=%.3fs",
                job_id, model_type, method, calib_info.get("method"), elapsed)
    return JSONResponse({
        "job_id": job_id,
        "filename": filename,
        "depth_method": method,
        "model_type": model_type,
        "synthetic_fallback": synthetic_fallback,
        "processing_time_s": elapsed,
        "is_georeferenced": is_geotiff,
        "absolute_dsm": absolute_dsm,
        "elevation_type": elevation_type,
        "scientific_note": warning,
        "geo_info": geo_info,
        "calibration": calib_info,
        "height_range_m": [float(np.nanmin(dsm)), float(np.nanmax(dsm))],
        "urls": {
            "mesh": f"/outputs/{job_id}/terrain.glb",
            "depth_preview": f"/outputs/{job_id}/depth_preview.png",
            "dsm": dsm_url,
        },
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "DepthWizard"}


app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
