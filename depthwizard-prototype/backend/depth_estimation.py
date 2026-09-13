"""
DepthWizard monocular depth inference.

Priority:
1. Depth Anything Small through Hugging Face Transformers
2. MiDaS Small through torch.hub
3. Synthetic fallback for offline/demo operation only

Important:
The model produces RELATIVE depth.
It does NOT directly produce metric elevation.
Metric calibration is handled separately in calibration.py.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

_backend = None
_pipe = None


def _try_load_depth_anything():
    global _pipe, _backend

    try:
        from transformers import pipeline

        local_checkpoint = os.environ.get(
            "DEPTHWIZARD_CHECKPOINT",
            str(Path(__file__).resolve().parent / "models" / "depth-anything-gamus"),
        )
        model_name = local_checkpoint if Path(local_checkpoint).exists() else "LiheYoung/depth-anything-small-hf"
        print(f"[DepthWizard] Loading Depth Anything from {model_name}...")

        _pipe = pipeline(
            task="depth-estimation",
            model=model_name,
        )

        _backend = "depth-anything-gamus" if Path(model_name).exists() else "depth-anything-small-hf"

        print("[DepthWizard] Depth Anything loaded successfully.")

        return True

    except Exception as exc:
        print(f"[DepthWizard] Depth Anything unavailable: {exc}")
        return False


def _try_load_midas():
    global _pipe, _backend

    try:
        import torch

        print("[DepthWizard] Loading MiDaS Small...")

        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        model.eval()

        _pipe = (model, transform)
        _backend = "midas_small"

        print("[DepthWizard] MiDaS loaded successfully.")

        return True

    except Exception as exc:
        print(f"[DepthWizard] MiDaS unavailable: {exc}")
        return False


def _fallback_synthetic(img: Image.Image) -> np.ndarray:
    """
    Emergency visual-only fallback.

    This is NOT a real depth model.
    It only exists so the entire demo can still run
    when AI models cannot be downloaded.
    """
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    edge = np.sqrt(gx * gx + gy * gy)
    edge /= edge.max() + 1e-8

    pseudo = 0.6 * gray + 0.4 * edge
    return pseudo.astype(np.float32)


def estimate_depth(img: Image.Image):
    """
    Estimate relative depth.

    Returns:
        depth: H x W float32 numpy array, same size as the input image
        backend_name: name of the depth method used
    """
    global _backend, _pipe

    if _backend is None:
        if not _try_load_depth_anything():
            if not _try_load_midas():
                print("[DepthWizard] No depth model available.")
                print("[DepthWizard] Using synthetic fallback.")
                _backend = "fallback_synthetic"

    if _backend in ("depth-anything-small-hf", "depth-anything-gamus"):
        result = _pipe(img)
        depth = np.asarray(result["depth"], dtype=np.float32)

        # Guard against the pipeline returning a different size than the
        # source image (e.g. due to internal resizing) so downstream
        # calibration/mesh code can always assume depth.shape == img.size.
        if depth.shape != (img.height, img.width):
            depth = np.asarray(
                Image.fromarray(depth).resize(
                    (img.width, img.height), Image.Resampling.BICUBIC
                ),
                dtype=np.float32,
            )

        return depth, _backend

    if _backend == "midas_small":
        import torch

        model, transform = _pipe
        input_batch = transform(np.asarray(img))

        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(img.height, img.width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy().astype(np.float32), _backend

    return _fallback_synthetic(img), "fallback_synthetic"
