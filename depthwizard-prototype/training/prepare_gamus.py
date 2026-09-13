"""Inspect Croissant GAMUS records and populate the shared normalized cache."""
import argparse
import json
from pathlib import Path

from dataset import GamusRecordAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("training/gamus_cache"))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    adapter = GamusRecordAdapter(max_records=args.max_records, cache_dir=args.output, local_root=args.data_root)
    summary = {}
    for split in ("train", "validation", "test"):
        records = adapter.records(split)
        samples = []
        for index, record in enumerate(records):
            rgb, depth, metadata = adapter.normalize_record(record)
            split_dir = args.output / split
            split_dir.mkdir(parents=True, exist_ok=True)
            import numpy as np
            path = split_dir / f"{index:06d}.npz"
            np.savez_compressed(path, rgb=np.asarray(rgb), depth=depth)
            samples.append({"path": str(path), "metadata": metadata})
        (args.output / split / "manifest.json").write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
        summary[split] = {"count": len(samples)}
    (args.output / "splits.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
