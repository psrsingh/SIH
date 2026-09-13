import re
from pathlib import Path

FRONTEND_DIR = Path(__file__).parent


def _importmap():
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    match = re.search(r'"imports":\s*({.*?})\s*}', html, re.DOTALL)
    imports_body = match.group(1)
    entries = re.findall(r'"([^"]+)":\s*"([^"]+)"', imports_body)
    return dict(entries)


def _resolve(specifier, importmap):
    for prefix, target in importmap.items():
        if specifier == prefix or (prefix.endswith("/") and specifier.startswith(prefix)):
            return target + specifier[len(prefix):]
    return specifier


def test_app_js_imports_resolve_to_real_vendor_files():
    # Regression test: app.js's import specifiers must resolve to files that
    # actually exist under frontend/vendor. This previously silently broke
    # the whole 3D viewer (and the whole app.js module, since a failed
    # static import aborts the entire script) because the vendored
    # OrbitControls.js/GLTFLoader.js were flat in vendor/three/addons/
    # instead of under addons/controls/ and addons/loaders/ as imported.
    importmap = _importmap()
    app_js = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    specifiers = re.findall(r'from\s+"([^"]+)"', app_js)
    assert specifiers, "expected at least one import in app.js"
    for specifier in specifiers:
        resolved = _resolve(specifier, importmap)
        assert (FRONTEND_DIR / resolved).is_file(), f"{specifier} -> {resolved} does not exist"


def test_vendored_addon_relative_imports_resolve():
    # GLTFLoader.js imports BufferGeometryUtils.js with a relative path;
    # verify that also still resolves after any vendor directory reshuffle.
    gltf_loader = FRONTEND_DIR / "vendor/three/addons/loaders/GLTFLoader.js"
    text = gltf_loader.read_text(encoding="utf-8")
    for relative_specifier in re.findall(r'from\s+[\'"](\.\./[^\'"]+)[\'"]', text):
        resolved = (gltf_loader.parent / relative_specifier).resolve()
        assert resolved.is_file(), f"{relative_specifier} -> {resolved} does not exist"
