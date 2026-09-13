import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
import main


def test_health_and_invalid_upload():
    with TestClient(main.app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        response = client.post("/api/process", files={"file": ("bad.txt", b"no", "text/plain")})
        assert response.status_code == 400


def test_process_generates_preview_and_mesh(monkeypatch):
    def fake_depth(image):
        return np.linspace(0, 1, image.width * image.height, dtype=np.float32).reshape(image.height, image.width), "depth-anything-small-hf"

    monkeypatch.setattr(main, "estimate_depth", fake_depth)
    image = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with TestClient(main.app) as client:
        response = client.post("/api/process", files={"file": ("sample.png", buffer.getvalue(), "image/png")},
                               data={"min_height": "0", "max_height": "20", "mesh_resolution": "12"})
    assert response.status_code == 200
    result = response.json()
    assert result["absolute_dsm"] is False
    assert result["synthetic_fallback"] is False
    assert (main.OUTPUT_DIR / result["job_id"] / "depth_preview.png").exists()
    assert (main.OUTPUT_DIR / result["job_id"] / "terrain.glb").exists()