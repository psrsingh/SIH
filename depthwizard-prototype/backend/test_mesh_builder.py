import struct
import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from mesh_builder import build_mesh_glb, _downsize_texture


def _read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    chunk_len = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + chunk_len])


def test_exported_mesh_has_color_attribute_for_elevation_ramp(tmp_path):
    # Regression test: mesh_builder used to set `visual.vertex_colors`,
    # which trimesh's glTF exporter silently ignores (it only reads
    # `visual.vertex_attributes["color"]`). That left the exported mesh
    # with no COLOR_0 accessor, so the frontend's elevation-ramp toggle
    # had nothing to switch to.
    img = Image.new("RGB", (16, 16), color=(120, 120, 120))
    dsm = np.linspace(0, 10, 16 * 16, dtype=np.float32).reshape(16, 16)
    out_path = tmp_path / "terrain.glb"

    build_mesh_glb(img, dsm, out_path, resolution=8)

    gltf = _read_glb_json(out_path)
    attributes = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert "COLOR_0" in attributes


def test_exported_material_is_not_metallic(tmp_path):
    # Regression test: the exported PBRMaterial never set metallicFactor,
    # which glTF defaults to 1.0 (fully metallic). With no environment map,
    # a fully metallic surface has almost no diffuse reflectance and
    # rendered as near-black in the viewer regardless of the depth model.
    img = Image.new("RGB", (16, 16), color=(120, 120, 120))
    dsm = np.linspace(0, 10, 16 * 16, dtype=np.float32).reshape(16, 16)
    out_path = tmp_path / "terrain.glb"

    build_mesh_glb(img, dsm, out_path, resolution=8)

    gltf = _read_glb_json(out_path)
    material = gltf["materials"][0]["pbrMetallicRoughness"]
    assert material.get("metallicFactor") == 0.0


def test_downsize_texture_caps_longest_edge():
    large = Image.new("RGB", (4000, 3000), color=(10, 20, 30))
    resized = _downsize_texture(large, max_edge=2048)
    assert max(resized.size) == 2048
    assert resized.size[0] / resized.size[1] == large.size[0] / large.size[1]

    small = Image.new("RGB", (512, 256), color=(1, 2, 3))
    assert _downsize_texture(small, max_edge=2048) is small


def test_build_mesh_glb_embeds_downsized_texture(tmp_path):
    img = Image.new("RGB", (4000, 3000), color=(200, 100, 50))
    dsm = np.linspace(0, 10, 16 * 16, dtype=np.float32).reshape(16, 16)
    out_path = tmp_path / "terrain.glb"

    build_mesh_glb(img, dsm, out_path, resolution=8)

    data = out_path.read_bytes()
    gltf = _read_glb_json(out_path)
    json_chunk_len = struct.unpack_from("<I", data, 12)[0]
    bin_chunk_start = 20 + json_chunk_len + 8
    buffer_view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    offset = bin_chunk_start + buffer_view["byteOffset"]
    embedded = Image.open(__import__("io").BytesIO(
        data[offset:offset + buffer_view["byteLength"]]
    ))
    assert max(embedded.size) <= 2048
