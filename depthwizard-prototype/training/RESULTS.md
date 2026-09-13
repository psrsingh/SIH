# DepthWizard upgrade results

## Execution status

The calibration and mesh changes were executed locally. GAMUS download, fine-tuning, and real test evaluation were not run in this environment because the optional `datasets` package is not installed and no GAMUS samples or LiDAR reference are present. The scripts fail with an actionable install message rather than inventing metrics.

## Calibration sanity check

Synthetic global regression recovered `elevation = 2.5 * depth + 13.0` with RMSE below `1e-4 m` and correlation `1.0`. The GCP branch uses the same regression and takes precedence over DEM calibration. Local tile fitting falls back to the global fit when valid tile coverage is insufficient.

These are unit/synthetic numbers, not GAMUS or LiDAR accuracy.

## Fine-tuning and before/after metrics

No honest before/after GAMUS RMSE, MAE, or correlation is reported yet. Run:

```text
pip install -r backend/requirements.txt
python training/prepare_gamus.py
python training/eda_gamus.py
python training/evaluate_depth.py --checkpoint LiheYoung/depth-anything-small-hf --output training/baseline.json
python training/train_depth.py --steps 1000
python training/evaluate_depth.py --checkpoint backend/models/depth-anything-gamus --output training/finetuned.json
```

`evaluate_depth.py` writes reusable JSON. Landscape breakdown is populated when the dataset exposes a landscape/category field; the current evaluator leaves it empty because the schema could not be inspected offline.

## Mesh timing

A small `16 x 16` export completed locally and produced a textured GLB with embedded `TextureVisuals` and UV coordinates. Representative production timing has not been measured here; benchmark with the target image sizes and resolutions before making a performance claim.

## Remaining risk

The training script assumes GAMUS exposes columns named `image`/`rgb` and `depth`/`target`, and reports a clear schema error otherwise. The model remains relative-depth; absolute quality still depends on reference DEM/GCP coverage. More compute and real category-balanced validation are required before claiming improvement over the pretrained baseline.

### Dependency note

`datasets` is the only new dependency: it is required to download/cache GAMUS and is intentionally optional for the offline inference demo.
