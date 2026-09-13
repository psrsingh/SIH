import numpy as np
import h5py
from PIL import Image

from .dataset import GamusRecordAdapter
from .dataset import GamusDepthDataset


def test_gamus_record_smoke_mapping():
    rgb = Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8))
    target = np.arange(80, dtype=np.float32).reshape(8, 10)
    image, depth, metadata = GamusRecordAdapter.normalize_record({
        "rgb_image": rgb,
        "elevation": target,
        "scene_type": "smoke",
    })
    assert image.size == (10, 8)
    assert depth.shape == (8, 10)
    assert np.isfinite(depth).all()
    assert metadata["scene_type"] == "smoke"


def test_rsi_hdf5_pairs_agl_height(tmp_path):
    root = tmp_path / "RSI-MMSegmentation-main"
    for split in ("train", "val"):
        for name in ("images", "classes", "heights"):
            (root / name / split).mkdir(parents=True)
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    classes = np.ones((8, 10), dtype=np.uint8)
    height = np.arange(80, dtype=np.float32).reshape(8, 10)
    for folder, suffix, value in (("images", "IMG", image), ("classes", "CLS", classes), ("heights", "AGL", height)):
        with h5py.File(root / folder / "train" / f"sample{suffix}.h5", "w") as handle:
            handle.create_dataset("image", data=value)
    dataset = GamusDepthDataset("train", root=root, crop_size=8, cache_dir=tmp_path / "cache")
    sample = dataset[0]
    assert sample["pixel_values"].shape == (3, 8, 8)
    assert sample["depth"].shape == (1, 8, 8)
    assert sample["valid"].any()


def test_rsi_rgb_suffix_pairs_current_gamus_layout(tmp_path):
    root = tmp_path / "GAMUS"
    for name in ("images", "classes", "heights"):
        (root / name / "test").mkdir(parents=True)
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    classes = np.ones((8, 10), dtype=np.float32)
    height = np.arange(80, dtype=np.float32).reshape(8, 10)
    for folder, suffix, value in (("images", "RGB", image), ("classes", "CLS", classes), ("heights", "AGL", height)):
        with h5py.File(root / folder / "test" / f"sample_{suffix}.h5", "w") as handle:
            handle.create_dataset("image", data=value)
    dataset = GamusDepthDataset("test", root=root, crop_size=8, cache_dir=tmp_path / "cache")
    sample = dataset[0]
    assert sample["pixel_values"].shape == (3, 8, 8)
    assert sample["depth"].shape == (1, 8, 8)