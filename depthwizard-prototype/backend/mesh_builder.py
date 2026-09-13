"""
DepthWizard 3D terrain mesh generation.

Converts DSM + RGB image into a textured GLB terrain mesh. For GeoTIFF
data, the horizontal mesh dimensions use the actual pixel resolution
when available, so the mesh's Y axis holds real elevation in meters and
X/Z hold real ground distance in meters.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def _clean_dsm(dsm):
    values = np.asarray(dsm, dtype=np.float32).copy()
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("DSM contains no finite elevation values.")
    fill = float(np.nanmedian(values))
    values[~finite] = fill
    padded = np.pad(values, 1, mode="edge")
    neighborhoods = np.stack([
        padded[y:y + values.shape[0], x:x + values.shape[1]]
        for y in range(3) for x in range(3)
    ])
    filtered = np.median(neighborhoods, axis=0)
    low, high = np.percentile(filtered, [0.5, 99.5])
    return np.clip(values, low, high).astype(np.float32)


MAX_TEXTURE_EDGE = 2048


def _downsize_texture(img: Image.Image, max_edge: int = MAX_TEXTURE_EDGE) -> Image.Image:
    """Cap the embedded glTF texture's longest edge.

    Uploads can be up to MAX_UPLOAD_SIZE (50MB); baking a multi-thousand-
    pixel source image into the GLB as-is produces needlessly huge files
    and risks exceeding common GPU MAX_TEXTURE_SIZE limits (many cap at
    4096/8192). UVs are already 0-1 normalized, so this needs no other
    changes to the mesh.
    """
    if max(img.size) <= max_edge:
        return img
    scale = max_edge / max(img.size)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def build_mesh_glb(
    img: Image.Image,
    dsm: np.ndarray,
    out_path: Path,
    resolution: int = 160,
    pixel_size_x: float = 1.0,
    pixel_size_y: float = 1.0,
):
    import trimesh

    dsm = _clean_dsm(dsm)
    h, w = dsm.shape

    # Prevent an excessively large mesh.
    res = max(8, min(int(resolution), h, w))

    ys = np.linspace(0, h - 1, res).astype(int)
    xs = np.linspace(0, w - 1, res).astype(int)
    grid_z = dsm[np.ix_(ys, xs)]

    # UVs map the original RGB image onto the terrain. The image is embedded
    # in the GLB by trimesh, avoiding color interpolation across vertices.
    uv = np.stack([
        np.tile(np.linspace(0.0, 1.0, res), res),
        np.repeat(1.0 - np.linspace(0.0, 1.0, res), res),
    ], axis=1)

    # Real-world horizontal dimensions (falls back to 1 unit/px when the
    # image isn't georeferenced, so a non-geo PNG still gets a sensible,
    # roughly isotropic mesh).
    width_m = max((w - 1) * abs(float(pixel_size_x)), 1.0)
    height_m = max((h - 1) * abs(float(pixel_size_y)), 1.0)

    x = np.linspace(-width_m / 2, width_m / 2, res)
    y = np.linspace(-height_m / 2, height_m / 2, res)
    xx, yy = np.meshgrid(x, y)

    # X = east/west, Y = elevation (meters), Z = north/south.
    vertices = np.stack([xx.ravel(), grid_z.ravel(), (-yy).ravel()], axis=-1)

    faces = []
    for r in range(res - 1):
        row0 = r * res
        row1 = (r + 1) * res
        for c in range(res - 1):
            i0, i1 = row0 + c, row0 + c + 1
            i2, i3 = row1 + c, row1 + c + 1
            # Wind CCW as seen from +Y (above) so face normals point up;
            # the frontend renders with default FrontSide culling, and the
            # opposite winding made every triangle invisible from the
            # camera's default overhead view.
            faces.append((i0, i1, i2))
            faces.append((i1, i3, i2))
    faces = np.asarray(faces, dtype=np.int32)

    texture_img = _downsize_texture(img.convert("RGB"))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_img, roughnessFactor=0.9, metallicFactor=0.0
    )
    vertex_colors = np.full((res * res, 4), 255, dtype=np.uint8)
    visual = trimesh.visual.texture.TextureVisuals(
        uv=uv, image=texture_img, material=material
    )
    # trimesh's glTF exporter only emits a COLOR_0 accessor when it finds
    # colors under vertex_attributes["color"] — a plain `.vertex_colors`
    # assignment on TextureVisuals is silently ignored at export time,
    # which would leave the frontend's elevation-ramp toggle with no
    # color attribute to switch to.
    visual.vertex_attributes["color"] = vertex_colors
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=faces,
        visual=visual,
        process=False,
    )

    mesh.export(out_path)
    return mesh
