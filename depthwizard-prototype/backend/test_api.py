import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
import main


def _fake_depth(image):
    return np.linspace(0, 1, image.width * image.height, dtype=np.float32).reshape(image.height, image.width), "depth-anything-small-hf"


def _small_png_bytes(size=24):
    image = Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _tiny_geotiff_bytes(crs="EPSG:32633", size=8):
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    data = np.zeros((1, size, size), dtype="float32")
    transform = from_origin(0, size, 1, 1)
    with MemoryFile() as memfile:
        with memfile.open(driver="GTiff", height=size, width=size, count=1,
                           dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(data)
        return memfile.read()


def test_health_and_invalid_upload():
    with TestClient(main.app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        response = client.post("/api/process", files={"file": ("bad.txt", b"no", "text/plain")})
        assert response.status_code == 400


def test_process_generates_preview_and_mesh(monkeypatch):
    monkeypatch.setattr(main, "estimate_depth", _fake_depth)
    with TestClient(main.app) as client:
        response = client.post("/api/process", files={"file": ("sample.png", _small_png_bytes(), "image/png")},
                               data={"min_height": "0", "max_height": "20", "mesh_resolution": "12"})
    assert response.status_code == 200
    result = response.json()
    assert result["absolute_dsm"] is False
    assert result["synthetic_fallback"] is False
    assert (main.OUTPUT_DIR / result["job_id"] / "depth_preview.png").exists()
    assert (main.OUTPUT_DIR / result["job_id"] / "terrain.glb").exists()


def test_corrupt_image_returns_400(monkeypatch):
    monkeypatch.setattr(main, "estimate_depth", _fake_depth)
    with TestClient(main.app) as client:
        response = client.post("/api/process", files={"file": ("bad.png", b"not a real png", "image/png")})
    assert response.status_code == 400


def test_main_upload_too_large_returns_413(monkeypatch):
    monkeypatch.setattr(main, "estimate_depth", _fake_depth)
    monkeypatch.setattr(main, "MAX_UPLOAD_SIZE", 10)
    with TestClient(main.app) as client:
        response = client.post("/api/process", files={"file": ("sample.png", _small_png_bytes(), "image/png")})
    assert response.status_code == 413


def test_reference_dem_too_large_returns_413(monkeypatch):
    monkeypatch.setattr(main, "estimate_depth", _fake_depth)
    geotiff_bytes = _tiny_geotiff_bytes()
    monkeypatch.setattr(main, "MAX_UPLOAD_SIZE", len(geotiff_bytes) + 10)
    oversized_ref = b"0" * (len(geotiff_bytes) + 1000)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/process",
            files={
                "file": ("sample.tif", geotiff_bytes, "image/tiff"),
                "reference_dem": ("ref.tif", oversized_ref, "image/tiff"),
            },
        )
    assert response.status_code == 413


def test_gcp_file_too_large_returns_413(monkeypatch):
    monkeypatch.setattr(main, "estimate_depth", _fake_depth)
    png_bytes = _small_png_bytes()
    monkeypatch.setattr(main, "MAX_UPLOAD_SIZE", len(png_bytes) + 100)
    oversized_gcps = ("x,y,elevation\n" + "1,1,1\n" * 500).encode()
    with TestClient(main.app) as client:
        response = client.post(
            "/api/process",
            files={
                "file": ("sample.png", png_bytes, "image/png"),
                "gcp_file": ("gcps.csv", oversized_gcps, "text/csv"),
            },
        )
    assert response.status_code == 413


def test_invalid_mesh_resolution_returns_400(monkeypatch):
    monkeypatch.setattr(main, "estimate_depth", _fake_depth)
    with TestClient(main.app) as client:
        response = client.post("/api/process", files={"file": ("sample.png", _small_png_bytes(), "image/png")},
                               data={"mesh_resolution": "4"})
    assert response.status_code == 400


def test_invalid_height_range_returns_400(monkeypatch):
    monkeypatch.setattr(main, "estimate_depth", _fake_depth)
    with TestClient(main.app) as client:
        response = client.post("/api/process", files={"file": ("sample.png", _small_png_bytes(), "image/png")},
                               data={"min_height": "20", "max_height": "5"})
    assert response.status_code == 400


def test_invalid_local_tile_size_returns_400(monkeypatch):
    monkeypatch.setattr(main, "estimate_depth", _fake_depth)
    with TestClient(main.app) as client:
        response = client.post("/api/process", files={"file": ("sample.png", _small_png_bytes(), "image/png")},
                               data={"local_tile_size": "1000"})
    assert response.status_code == 400


def test_geographic_crs_geotiff_returns_400(monkeypatch):
    monkeypatch.setattr(main, "estimate_depth", _fake_depth)
    geotiff_bytes = _tiny_geotiff_bytes(crs="EPSG:4326")
    with TestClient(main.app) as client:
        response = client.post("/api/process", files={"file": ("sample.tif", geotiff_bytes, "image/tiff")})
    assert response.status_code == 400
    assert "geographic" in response.json()["error"].lower()