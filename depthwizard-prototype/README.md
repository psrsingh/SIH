# DepthWizard — Hackathon MVP (SIH 26175)

A working, end-to-end demo of the pipeline described in the problem statement:
**single RGB image → monocular depth → calibrated DSM → interactive 3D flythrough.**

This is a hackathon MVP for turning one RGB aerial image into an inspectable
relative terrain surface, or a calibrated DSM when reference information is
available. Monocular depth is scale ambiguous.

```
RGB -> Monocular depth -> Calibration -> DSM -> 3D terrain
```

## What it does

1. **Upload** a PNG/JPG (non-georeferenced) or GeoTIFF (georeferenced) image.
2. **Elevation extraction** — runs a pretrained monocular depth model to get a
   relative depth map.
3. **Scale calibration** — two modes:
   - No reference data: linear min/max fit into a height range you specify
     (relative DSM, clearly labeled as such).
   - GeoTIFF + an uploaded reference DEM (e.g. SRTM): least-squares
     regression of relative depth against real reference elevations
     (tries both depth polarities and keeps whichever orientation fits
     better), producing an **absolute metric DSM** with reported RMSE/MAE.
4. **3D reconstruction + visualization** — builds a textured terrain mesh
   (`.glb`), scaled to real-world meters when GeoTIFF pixel resolution is
   known, and serves it into a Three.js viewer with an auto-fit
   orbit/flythrough camera, an elevation-ramp toggle, and a hover probe
   showing elevation and local slope.
5. **GeoTIFF DSM export** — when the input is georeferenced, the computed
   DSM is written back out as a GeoTIFF (same CRS/transform as the input),
   tagged with the calibration method used and whether it's absolute or
   relative, so a GIS user can tell at a glance without re-reading the API
   response.

## Architecture

```
backend/
  main.py              FastAPI app: /api/process endpoint + static hosting
  depth_estimation.py  Depth Anything -> MiDaS -> synthetic fallback chain
  calibration.py       relative depth -> DSM, GeoTIFF metadata reader
  mesh_builder.py       DSM + RGB -> textured .glb terrain mesh
  requirements.txt
frontend/
  index.html           control panel + viewport layout
  style.css
  app.js               upload flow, Three.js scene, raycasting probe
```

### Model selection and fallback

Hackathon demo day often means unreliable venue Wi-Fi. The backend tries, in
order: **Depth Anything (small)** → **MiDaS small** → a **synthetic pseudo-depth**
computed from luminance/edges. The API identifies it as
`model_type: synthetic-demo` and `synthetic_fallback: true`; evaluation never
uses this fallback.

## Setup

```bash
cd depthwizard-prototype
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python training/prepare_gamus.py --max-records 10
cd backend
uvicorn main:app --reload --port 8010
```

Open **http://localhost:8010** — the FastAPI app serves the frontend directly,
no separate frontend server needed.

Configuration is available in `backend/.env.example`: `DEPTHWIZARD_CHECKPOINT`,
`OUTPUT_DIR`, `MAX_UPLOAD_SIZE`, `MAX_IMAGE_PIXELS`, `ALLOWED_ORIGINS`,
`OUTPUT_RETENTION_HOURS`, `MAX_CONCURRENT_JOBS`, `LOG_LEVEL`, and the
training-only `GAMUS_CROISSANT_URL` / `GAMUS_MAX_RECORDS` /
`GAMUS_CACHE_DIR` / `GAMUS_LOCAL_ROOT`.

## Using it

1. Drop in an image (try a satellite/aerial crop — anything works for the demo).
2. Set an expected min/max height in meters for the scene (e.g. 0–60 m for a
   suburb, 0–300 m for hilly terrain). If you uploaded a GeoTIFF, an optional
   **reference DEM** dropzone appears — upload a low-res DEM (e.g. SRTM 30 m)
   covering the same area to get a real metric fit instead of the min/max
   guess.
3. Click **Generate terrain** — depth runs, the DSM is calibrated, the mesh
   builds, and the viewer loads it automatically.
4. Orbit/zoom to fly through the terrain. Toggle **Elevation ramp** to see a
   color-coded height map instead of the RGB texture. Hover to read off
   elevation and local slope. If a DSM GeoTIFF was produced, a download link
   appears in the result panel.

GCP CSV coordinates are pixel coordinates: `x` is the image column, `y` is the
image row, and `elevation` is the height value. They are not latitude/longitude.

## GAMUS dataset and training

The official Croissant endpoint is
`https://huggingface.co/api/datasets/earthflow/GAMUS/croissant`. Its current
metadata declares `default` records with `split`, `image` bytes, and an integer
`label`; it does not declare a 2D depth/elevation target. The loader reads
`records("default")`, filters train/validation/test by `split`, validates image
and target dimensions, and refuses to interpret a scalar label as depth. The
current endpoint's parquet operation also failed with an empty read during
verification, so no GAMUS checkpoint or benchmark is claimed.

The included `RSI-MMSegmentation-main` loader supports the original local
HDF5 release when its data is downloaded separately. The expected layout is:

```text
RSI-MMSegmentation-main/
   images/{train,val,test}/*IMG.h5
   classes/{train,val,test}/*CLS.h5
   heights/{train,val,test}/*AGL.h5
```

`AGL.h5` is used as the continuous elevation target; `CLS.h5` remains
available for semantic labels. Point `GAMUS_LOCAL_ROOT` at that directory or
pass `--data-root` to training.

When paired depth targets are available, the shared normalized cache is used by
both preparation and training:

```bash
python training/train_depth.py --epochs 5 --data-root RSI-MMSegmentation-main --output training/checkpoints --max-records 10 --cache-dir training/rsi_gamus_cache
python training/evaluate_depth.py --checkpoint training/checkpoints/best --data-root RSI-MMSegmentation-main --cache-dir training/rsi_gamus_cache
```

Training provides a seed, explicit splits, CUDA mixed precision, validation
loss/RMSE/MAE/correlation, resume support, `latest.pt`, and a best checkpoint
selected by validation RMSE. The pipeline has only been run as a smoke test
on the 25-image `GAMUS_SMOKE` sample so far (see `training/EVALUATION.md`)
— those metrics are not statistically meaningful and that checkpoint is not
loaded by the backend. A real before/after comparison needs a
training-sized paired dataset.

## Mapping to the evaluation criteria

| Milestone (from PS) | Where it lives |
|---|---|
| Elevation Extraction | `depth_estimation.py` |
| Scale Calibration | `calibration.py` |
| Visualization Layer | `mesh_builder.py` + `frontend/` |

## Honest limitations

- GAMUS fine-tuning has only been smoke-tested on a 25-image local sample
  (see `training/EVALUATION.md`) — not a statistically meaningful evaluation,
  and that checkpoint is **not** loaded by backend inference. The publicly
  published Croissant endpoint can't be used for this at all: it exposes an
  integer label rather than a paired depth raster, so the local
  RSI-MMSegmentation HDF5 mirror was used instead.
- The DEM-regression calibration is a **single global linear fit**
  (`elevation = a·depth + b`) over the whole scene. It reports RMSE/MAE for
  transparency, but a single linear fit will underperform in mixed
  urban/forested/hilly scenes — the PS explicitly calls out needing
  stability across those landscape types, which likely needs a more local
  or piecewise fit.
- Mesh resolution is capped for interactive frame rates; a production version
  would tile large scenes.

## Next steps toward the full problem statement

1. Fine-tune / adapt a monocular depth model when a paired GAMUS depth target
   is available.
2. Extend the DEM regression from one global linear fit to a
   locally-weighted or piecewise fit so accuracy holds up across urban,
   sparse, hilly, and forested landscapes, per the evaluation criteria.
3. Add a quantitative validation panel in the UI (RMSE / MAE / correlation
   vs. reference elevation) — the numbers are already computed in
   `calibration.py`, they just need to be surfaced visually.
4. Support GCP-based calibration (a handful of surveyed points) as an
   alternative to a full reference DEM, per the PS's "or GCPs" option.

## Verification

Run `pip install -r backend/requirements-dev.txt`, then `python -m pytest`,
`python -m pip check`, and the manual browser smoke
test: open the app, upload a PNG, click Generate terrain, confirm the depth
preview and GLB viewer, toggle Elevation ramp, and hover the mesh to inspect
elevation and slope. For GeoTIFF input, confirm CRS/transform are preserved
and the DSM download tag matches the displayed Relative or Absolute status.
