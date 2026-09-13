"""GAMUS loading through the official Croissant endpoint.

The endpoint is intentionally inspected at runtime because the dataset's
record names are part of its metadata contract, not a stable local guess.
Normalized samples are cached as NPZ files so training, validation, testing,
and the smoke test all consume the same representation.
"""
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_CROISSANT_URL = "https://huggingface.co/api/datasets/earthflow/GAMUS/croissant"


def _scalar(value):
    if isinstance(value, dict):
        for key in ("value", "data", "path", "url", "bytes"):
            if key in value:
                return value[key]
    return value


def _field_name(field):
    return str(getattr(field, "name", getattr(field, "id", ""))).lower()


def _record_items(record):
    if hasattr(record, "items"):
        return list(record.items())
    if hasattr(record, "to_dict"):
        return list(record.to_dict().items())
    return [(str(key), value) for key, value in vars(record).items() if not key.startswith("_")]


class GamusRecordAdapter:
    """Discover and validate the actual Croissant record fields."""

    def __init__(self, croissant_url=None, max_records=None, cache_dir=None, local_root=None):
        self.croissant_url = croissant_url or os.getenv("GAMUS_CROISSANT_URL", DEFAULT_CROISSANT_URL)
        self.max_records = int(max_records or os.getenv("GAMUS_MAX_RECORDS", "0")) or None
        self.cache_dir = Path(cache_dir or os.getenv("GAMUS_CACHE_DIR", "training/gamus_cache"))
        self.local_root = Path(local_root or os.getenv("GAMUS_LOCAL_ROOT", "")) if (local_root or os.getenv("GAMUS_LOCAL_ROOT")) else None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def records(self, split="default"):
        if self.local_root:
            return self._local_records(split)
        try:
            from mlcroissant import Dataset as CroissantDataset
        except ImportError as exc:
            raise RuntimeError("Install GAMUS dependencies with: pip install mlcroissant datasets") from exc
        dataset = CroissantDataset(jsonld=self.croissant_url)
        records = dataset.records("default")
        if split == "default":
            return records
        selected = []
        for record in records:
            items = dict(_record_items(record))
            record_split = items.get("split", items.get("default/split"))
            if record_split == split:
                selected.append(record)
                if self.max_records and len(selected) >= self.max_records:
                    break
        return selected

    def _local_records(self, split):
        """Read the RSI-MMSegmentation HDF5 layout and pair AGL targets."""
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Install h5py to read the RSI-MMSegmentation dataset.") from exc
        source_split = "val" if split == "validation" else split
        image_dir = self.local_root / "images" / source_split
        class_dir = self.local_root / "classes" / source_split
        height_dir = self.local_root / "heights" / source_split
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing RSI image split: {image_dir}")
        records = []
        for image_path in sorted(image_dir.glob("*.h5")):
            image_name = image_path.name
            if image_name.endswith("_IMG.h5"):
                stem = image_name[:-len("_IMG.h5")]
            elif image_name.endswith("_RGB.h5"):
                stem = image_name[:-len("_RGB.h5")]
            elif image_name.endswith("IMG.h5"):
                stem = image_name[:-len("IMG.h5")]
            else:
                stem = image_path.stem
            class_path = class_dir / f"{stem}CLS.h5"
            height_path = height_dir / f"{stem}AGL.h5"
            if image_name.endswith(("_IMG.h5", "_RGB.h5")):
                class_path = class_dir / f"{stem}_CLS.h5"
                height_path = height_dir / f"{stem}_AGL.h5"
            if not class_path.exists() or not height_path.exists():
                raise FileNotFoundError(f"Incomplete RSI sample {image_path.name}: expected paired CLS and AGL files")
            with h5py.File(image_path, "r") as handle:
                image = handle["image"][()]
            with h5py.File(height_path, "r") as handle:
                height = handle["image"][()]
            records.append({"image": image, "height": height, "sample": stem})
            if self.max_records and len(records) >= self.max_records:
                break
        if not records:
            raise ValueError(f"No RSI HDF5 samples found in {image_dir}")
        return records

    @staticmethod
    def _as_image(value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(BytesIO(value)).convert("RGB")
        value = _scalar(value)
        if isinstance(value, (str, Path)) and Path(value).exists():
            return Image.open(value).convert("RGB")
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] in (3, 4):
            return Image.fromarray(array[..., :3].astype(np.uint8)).convert("RGB")
        raise ValueError("record does not contain a readable RGB image")

    @staticmethod
    def _as_depth(value):
        if isinstance(value, Image.Image):
            return np.asarray(value, dtype=np.float32)
        if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
            return np.asarray(Image.open(BytesIO(value["bytes"])), dtype=np.float32)
        if isinstance(value, (bytes, bytearray)):
            return np.asarray(Image.open(BytesIO(value)), dtype=np.float32)
        value = _scalar(value)
        if isinstance(value, (str, Path)) and Path(value).exists():
            return np.asarray(Image.open(value), dtype=np.float32)
        depth = np.asarray(value, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[0] if depth.shape[0] == 1 else depth[..., 0]
        if depth.ndim != 2:
            raise ValueError("record does not contain a 2D depth/elevation target")
        return depth

    @classmethod
    def normalize_record(cls, record):
        items = _record_items(record)
        if not items:
            raise ValueError("empty GAMUS record")
        rgb = depth = None
        metadata = {}
        for name, value in items:
            key = str(name).lower()
            try:
                if rgb is None and any(token in key for token in ("rgb", "image", "photo", "input")):
                    rgb = cls._as_image(value)
                    continue
            except (TypeError, ValueError, OSError):
                pass
            try:
                if depth is None and any(token in key for token in ("depth", "elevation", "height", "target", "dem")):
                    depth = cls._as_depth(value)
                    continue
            except (TypeError, ValueError, OSError):
                pass
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[str(name)] = value
        if rgb is None or depth is None:
            raise KeyError(f"Could not map RGB/depth fields from record keys: {[name for name, _ in items]}")
        if rgb.width < 2 or rgb.height < 2 or depth.shape[0] < 2 or depth.shape[1] < 2:
            raise ValueError("GAMUS RGB and target dimensions must both be at least 2x2")
        if depth.shape != (rgb.height, rgb.width):
            depth = np.asarray(Image.fromarray(depth).resize(rgb.size, Image.Resampling.BILINEAR), dtype=np.float32)
        if not np.isfinite(depth).any():
            raise ValueError("GAMUS target contains no finite values")
        return rgb, depth, metadata


class GamusDepthDataset(Dataset):
    def __init__(self, split="train", root=None, crop_size=384, augment=None,
                 max_records=None, cache_dir=None, records=None):
        self.split = split
        self.crop_size = int(crop_size)
        self.augment = split == "train" if augment is None else augment
        self.adapter = GamusRecordAdapter(max_records=max_records, cache_dir=cache_dir, local_root=root)
        self.samples = self._load_samples(records)

    def _load_samples(self, records):
        split_dir = self.adapter.cache_dir / self.split
        manifest_path = split_dir / "manifest.json"
        if records is None and manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]
        records = self.adapter.records(self.split) if records is None else records
        split_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for index, record in enumerate(records):
            rgb, depth, metadata = self.adapter.normalize_record(record)
            sample_path = split_dir / f"{index:06d}.npz"
            np.savez_compressed(sample_path, rgb=np.asarray(rgb), depth=depth)
            samples.append({"path": str(sample_path), "metadata": metadata})
        manifest_path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
        return samples

    @staticmethod
    def _to_image(value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict) and "bytes" in value:
            from io import BytesIO
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        return Image.fromarray(np.asarray(value)).convert("RGB")

    @staticmethod
    def _to_depth(value):
        if isinstance(value, Image.Image):
            return np.asarray(value, dtype=np.float32)
        if isinstance(value, dict) and "bytes" in value:
            from io import BytesIO
            return np.asarray(Image.open(BytesIO(value["bytes"])), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        cached = np.load(sample["path"])
        image = Image.fromarray(cached["rgb"].astype(np.uint8), mode="RGB")
        depth = cached["depth"].astype(np.float32)
        image, depth = self._augment(image, depth)
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        valid = torch.isfinite(depth) & (depth > 0)
        result = {"pixel_values": image, "depth": depth, "valid": valid}
        landscape = next((value for key, value in sample.get("metadata", {}).items()
                          if any(word in key.lower() for word in ("landscape", "scene", "category", "type"))), None)
        if landscape is not None:
            result["landscape"] = str(landscape)
        return result

    def _augment(self, image, depth):
        import random
        h, w = depth.shape
        if self.crop_size and h >= self.crop_size and w >= self.crop_size:
            y = random.randint(0, h - self.crop_size) if self.augment else (h - self.crop_size) // 2
            x = random.randint(0, w - self.crop_size) if self.augment else (w - self.crop_size) // 2
            image = image.crop((x, y, x + self.crop_size, y + self.crop_size))
            depth = depth[y:y + self.crop_size, x:x + self.crop_size]
        if self.augment:
            if random.random() < 0.5:
                image, depth = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), np.fliplr(depth).copy()
            if random.random() < 0.5:
                image, depth = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), np.flipud(depth).copy()
            turns = random.randrange(4)
            if turns:
                image, depth = image.rotate(90 * turns), np.rot90(depth, turns).copy()
            from PIL import ImageEnhance
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        return image, depth
