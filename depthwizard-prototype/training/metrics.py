import numpy as np


def regression_metrics(prediction, target, valid=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(prediction) & np.isfinite(target)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    prediction, target = prediction[mask], target[mask]
    correlation = float(np.corrcoef(prediction, target)[0, 1]) if len(prediction) > 1 else 0.0
    return {"rmse": float(np.sqrt(np.mean((prediction - target) ** 2))), "mae": float(np.mean(np.abs(prediction - target))),
            "correlation": correlation if np.isfinite(correlation) else 0.0, "count": int(len(prediction))}
