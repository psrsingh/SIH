"""Evaluate a Depth Anything checkpoint and emit reusable JSON metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from dataset import GamusDepthDataset
from metrics import regression_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", type=Path, default=Path("training/evaluation.json"))
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    if not args.checkpoint or (Path(args.checkpoint).suffix and not Path(args.checkpoint).exists()):
        result = {"status": "NOT EVALUATED", "reason": "No trained checkpoint is available."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    from transformers import AutoModelForDepthEstimation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForDepthEstimation.from_pretrained(args.checkpoint).to(device).eval()
    data = GamusDepthDataset("test", root=args.data_root, crop_size=args.crop_size, augment=False,
                             max_records=args.max_records, cache_dir=args.cache_dir)
    loader = DataLoader(data, batch_size=1)
    aggregate = []
    categories = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            image = batch["pixel_values"].to(device)
            image = (image - torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
            prediction = model(pixel_values=image).predicted_depth.unsqueeze(1)
            prediction = F.interpolate(prediction, size=batch["depth"].shape[-2:], mode="bilinear", align_corners=False).cpu().numpy()[0, 0]
            target = batch["depth"].numpy()[0, 0]
            metrics = regression_metrics(prediction, target, batch["valid"].numpy()[0, 0])
            aggregate.append(metrics)
            if "landscape" in batch:
                categories[batch["landscape"][0]].append(metrics)
    category_result = {
        name: {key: float(np.mean([item[key] for item in values])) for key in ("rmse", "mae", "correlation")}
        for name, values in categories.items()
    }
    result = {"status": "EVALUATED", "checkpoint": args.checkpoint, "split": "test", "aggregate": {
        key: float(np.mean([item[key] for item in aggregate])) for key in ("rmse", "mae", "correlation")
    }, "sample_count": len(aggregate), "by_landscape": category_result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
