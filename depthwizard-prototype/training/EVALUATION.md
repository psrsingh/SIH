# DepthWizard Evaluation

## Current status

REAL FINE-TUNE, DEPLOYED — MODEST DATASET, MODEST GAINS

`backend/models/depth-anything-gamus/best` was fine-tuned on a 400-image
train / 80-image val subset of the actual `earthflow/GAMUS` dataset
(RSI-MMSegmentation HDF5 mirror, downloaded directly from the HF repo
files — the Croissant metadata endpoint still can't be used for this, see
below), not the 25-image `GAMUS_SMOKE` set. It is independently evaluated
on an 80-image held-out `test` split that the model never saw during
training or checkpoint selection.

| | Correlation | RMSE (raw units) | MAE (raw units) |
|---|---|---|---|
| Baseline (`LiheYoung/depth-anything-small-hf`) | 0.163 | 10.10 | 8.29 |
| Fine-tuned (`depth-anything-gamus/best`, epoch 2/5) | **0.613** | 12.13 | 10.23 |

(`training/baseline_subset.json`, `training/finetuned_subset.json`.)

Correlation nearly quadrupled — real evidence the fine-tune learned
aerial-relevant structure the pretrained model doesn't have. RMSE/MAE look
*worse*, not better: `train_depth.py`'s loss is scale/shift-invariant (it
fits per-sample scale+offset before scoring), so the model's raw output
magnitude is not tied to meters, and `evaluate_depth.py`'s RMSE/MAE compare
raw prediction to target with no such correction. Only correlation is a
fair before/after signal from these numbers. Downstream, `calibration.py`
re-fits scale/offset per scene from a reference DEM or GCPs (or a min/max
guess), so this is also the metric that matters for the actual product.

Training used `--crop-size 256 --batch-size 4` (down from the 384/2
defaults) purely for CPU wall-clock time; validation RMSE was still best at
epoch 2 of 5 and got worse after, i.e. the model overfits past that point
on this dataset size — the `best` checkpoint (selected by validation RMSE)
was used for the test evaluation above, not the final epoch.

`model.save_pretrained()` does not save the image processor config, so the
checkpoint directory needed `preprocessor_config.json` copied in from the
base model's HF cache before `transformers.pipeline(...)` could load it —
do this for any future checkpoint saved by `train_depth.py`, or inference
silently falls through to MiDaS/synthetic instead of erroring.

## Still true from the earlier smoke test

`training/smoke_checkpoint_25` / `smoke_checkpoint_small` (trained on the
25-image `GAMUS_SMOKE` sample) remain not statistically meaningful and are
not loaded by the backend. Superseded by the 400/80/80-image run above for
any real before/after claim.

## Remaining limitations

400 train images is still small by fine-tuning standards — 0.613
correlation is real but far from a ceiling. A larger subset (the full
dataset is ~8,724 images / ~98GB across images+classes+heights) and more
epochs with a val-aware LR schedule would likely improve further. The
official Croissant/HF-datasets endpoint (`GAMUS_CROISSANT_URL`) still
only exposes an image + integer label, not a depth target — the RSI-
MMSegmentation HDF5 files must be pulled directly from the dataset's repo
files (`huggingface_hub.hf_hub_download`), not through `datasets`/
`mlcroissant`.