import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

import numpy as np
from calibration import calibrate_height, calibrate_with_reference_dem


def test_global_recovers_scale_and_offset():
    depth = np.arange(1600, dtype=np.float32).reshape(40, 40)
    elevation = 2.5 * depth + 13.0
    _, info = calibrate_with_reference_dem(depth, elevation)
    assert abs(info["scale_a"] - 2.5) < 1e-5
    assert abs(info["offset_b"] - 13.0) < 1e-5
    assert info["fit_rmse_m"] < 1e-4


def test_gcp_mode_has_priority():
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    gcps = [{"x": 0, "y": 0, "elevation": 5}, {"x": 5, "y": 5, "elevation": 60}, {"x": 9, "y": 9, "elevation": 104}]
    _, info = calibrate_height(depth, gcps=gcps)
    assert info["method"] == "gcp_linear_regression"
    assert info["fit_rmse_m"] < 1e-4
    assert info["absolute"] is True


def test_relative_mode_is_not_absolute():
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    _, info = calibrate_height(depth, min_height=0, max_height=10)
    assert info["absolute"] is False


def test_reference_dem_rejects_flat_overlap():
    depth = np.ones((10, 10), dtype=np.float32)
    elevation = np.ones((10, 10), dtype=np.float32)
    try:
        calibrate_with_reference_dem(depth, elevation)
    except ValueError as exc:
        assert "variation" in str(exc)
    else:
        raise AssertionError("flat reference DEM should be rejected")
