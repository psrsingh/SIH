"""Report split counts, depth statistics, and any landscape metadata."""
import argparse
import json
from collections import Counter

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="earthflow/GAMUS")
    args = parser.parse_args()
    from datasets import load_dataset
    report = {}
    for split in ("train", "validation", "test"):
        data = load_dataset(args.dataset, split=split)
        depth_key = next((key for key in ("depth", "depth_image", "target", "label") if key in data.column_names), None)
        category_key = next((key for key in data.column_names if any(word in key.lower() for word in ("landscape", "scene", "category", "type"))), None)
        values = []
        categories = Counter()
        for row in data:
            if depth_key:
                value = row[depth_key]
                if hasattr(value, "convert"):
                    value = np.asarray(value, dtype=np.float32)
                elif isinstance(value, dict) and "bytes" in value:
                    from io import BytesIO
                    from PIL import Image
                    value = np.asarray(Image.open(BytesIO(value["bytes"])), dtype=np.float32)
                values.append(np.asarray(value).ravel())
            if category_key:
                categories[str(row[category_key])] += 1
        flat = np.concatenate(values) if values else np.array([], dtype=np.float32)
        report[split] = {"count": len(data), "depth_key": depth_key, "depth_min": float(np.nanmin(flat)) if flat.size else None,
                         "depth_max": float(np.nanmax(flat)) if flat.size else None,
                         "depth_p01": float(np.nanpercentile(flat, 1)) if flat.size else None,
                         "depth_p99": float(np.nanpercentile(flat, 99)) if flat.size else None,
                         "landscape_key": category_key, "landscape_counts": dict(categories)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
