# DepthWizard Evaluation

## Current status

NOT EVALUATED

No GAMUS fine-tuned checkpoint has been trained in this repository yet. Run
`python training/train_depth.py --epochs 5` after preparing a configured GAMUS
subset, then run `python training/evaluate_depth.py --checkpoint training/checkpoints/best`.
The evaluator writes `evaluation_results.json` only from the held-out `test`
split. Calibration fitting error is reported separately from independent test
metrics and is never called validation accuracy.