import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const API_BASE = "";

// ---------- DOM ----------
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const dropLabel = document.getElementById("dropLabel");
const previewImg = document.getElementById("previewImg");
const fileMeta = document.getElementById("fileMeta");

const demBlock = document.getElementById("demBlock");
const demInput = document.getElementById("demInput");
const demDropzone = document.getElementById("demDropzone");
const demDropLabel = document.getElementById("demDropLabel");
const demMeta = document.getElementById("demMeta");
const gcpInput = document.getElementById("gcpInput");
const gcpDropzone = document.getElementById("gcpDropzone");
const gcpDropLabel = document.getElementById("gcpDropLabel");
const gcpMeta = document.getElementById("gcpMeta");

const minHeight = document.getElementById("minHeight");
const maxHeight = document.getElementById("maxHeight");
const meshRes = document.getElementById("meshRes");
const meshResVal = document.getElementById("meshResVal");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");

const resultBlock = document.getElementById("resultBlock");
const rModel = document.getElementById("rModel");
const rMethod = document.getElementById("rMethod");
const rCalibration = document.getElementById("rCalibration");
const rGeo = document.getElementById("rGeo");
const rAbsolute = document.getElementById("rAbsolute");
const rRange = document.getElementById("rRange");
const rTime = document.getElementById("rTime");
const scientificWarning = document.getElementById("scientificWarning");
const depthPreview = document.getElementById("depthPreview");
const dsmLink = document.getElementById("dsmLink");
const validationBlock = document.getElementById("validationBlock");
const rRmse = document.getElementById("rRmse");
const rMae = document.getElementById("rMae");
const rCorrelation = document.getElementById("rCorrelation");
const tileErrors = document.getElementById("tileErrors");

const viewportHint = document.getElementById("viewportHint");
const probe = document.getElementById("probe");
const probeElev = document.getElementById("probeElev");
const probeSlope = document.getElementById("probeSlope");
const toolbar = document.getElementById("viewportToolbar");
const resetViewBtn = document.getElementById("resetViewBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingStep = document.getElementById("loadingStep");
const errorBanner = document.getElementById("errorBanner");
const errorBannerText = document.getElementById("errorBannerText");
const errorBannerDismiss = document.getElementById("errorBannerDismiss");

let selectedFile = null;
let selectedDem = null;
let selectedGcp = null;

// ---------- file inputs ----------
meshRes.addEventListener("input", () => (meshResVal.textContent = meshRes.value));

// A hidden <input type="file"> can't receive focus/Enter itself, so the
// wrapping dropzone label needs its own keyboard activation to be usable
// without a mouse (it's tabindex="0"/role="button" in the markup).
function makeKeyboardActivatable(el, onActivate) {
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onActivate();
    }
  });
}

dropzone.addEventListener("click", () => fileInput.click());
makeKeyboardActivatable(dropzone, () => fileInput.click());
["dragover", "dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => e.preventDefault())
);
dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

demDropzone.addEventListener("click", () => demInput.click());
makeKeyboardActivatable(demDropzone, () => demInput.click());
["dragover", "dragleave", "drop"].forEach((evt) =>
  demDropzone.addEventListener(evt, (e) => e.preventDefault())
);
demDropzone.addEventListener("drop", (e) => handleDem(e.dataTransfer.files[0]));
demInput.addEventListener("change", (e) => handleDem(e.target.files[0]));
gcpDropzone.addEventListener("click", () => gcpInput.click());
makeKeyboardActivatable(gcpDropzone, () => gcpInput.click());
gcpDropzone.addEventListener("drop", (e) => { e.preventDefault(); handleGcp(e.dataTransfer.files[0]); });
gcpInput.addEventListener("change", (e) => handleGcp(e.target.files[0]));

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  fileMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;

  const isGeoTiff = /\.tiff?$/i.test(file.name);
  demBlock.hidden = !isGeoTiff;
  if (!isGeoTiff) {
    selectedDem = null;
    demInput.value = "";
    demMeta.textContent = "No reference DEM selected.";
  }

  if (/\.(png|jpe?g)$/i.test(file.name)) {
    previewImg.src = URL.createObjectURL(file);
    previewImg.hidden = false;
    dropLabel.hidden = true;
  } else {
    previewImg.hidden = true;
    dropLabel.hidden = false;
    dropLabel.textContent = `${file.name} (GeoTIFF — no inline preview)`;
  }
}

function handleDem(file) {
  if (!file) return;
  if (!/\.tiff?$/i.test(file.name)) {
    demMeta.textContent = "Reference DEM must be a GeoTIFF (.tif/.tiff).";
    return;
  }
  selectedDem = file;
  demMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  demDropLabel.textContent = file.name;
}

function handleGcp(file) {
  if (!file) return;
  if (!/\.(csv|json|geojson)$/i.test(file.name)) {
    gcpMeta.textContent = "GCPs must be CSV, JSON, or GeoJSON.";
    return;
  }
  selectedGcp = file;
  gcpMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  gcpDropLabel.textContent = file.name;
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

function showLoading(step) {
  loadingStep.textContent = step;
  loadingOverlay.hidden = false;
}

function hideLoading() {
  loadingOverlay.hidden = true;
}

function showError(message) {
  errorBannerText.textContent = message;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
}

errorBannerDismiss.addEventListener("click", hideError);

const REQUEST_TIMEOUT_MS = 120_000;

// ---------- process ----------
let activeController = null;

runBtn.addEventListener("click", async () => {
  if (activeController) {
    activeController.abort("user-cancelled");
    return;
  }
  if (!selectedFile) {
    setStatus("select an image first", "error");
    return;
  }

  hideError();
  runBtn.textContent = "Cancel";
  runBtn.classList.add("run-btn-cancel");
  setStatus("Uploading", "busy");
  viewportHint.style.display = "none";
  showLoading("Uploading image…");

  const form = new FormData();
  form.append("file", selectedFile);
  if (selectedDem) form.append("reference_dem", selectedDem);
  if (selectedGcp) form.append("gcp_file", selectedGcp);
  form.append("min_height", minHeight.value);
  form.append("max_height", maxHeight.value);
  form.append("mesh_resolution", meshRes.value);

  const controller = new AbortController();
  activeController = controller;
  const timeoutId = setTimeout(() => controller.abort("timeout"), REQUEST_TIMEOUT_MS);

  try {
    setStatus("Running depth inference", "busy");
    showLoading("Running depth estimation and building the terrain mesh…");
    const res = await fetch(`${API_BASE}/api/process`, { method: "POST", body: form, signal: controller.signal });
    if (!res.ok) {
      let message = `Server returned ${res.status}.`;
      try {
        const body = await res.json();
        if (body?.error) message = body.error;
      } catch {
        // response wasn't JSON — fall back to the generic status message
      }
      const prefix = res.status === 413 ? "Upload too large: "
        : res.status === 400 ? "Invalid request: "
        : res.status >= 500 ? "Server error: "
        : "";
      throw new Error(prefix + message);
    }
    const data = await res.json();
    setStatus("Calibrating", "busy");
    showLoading("Loading the 3D viewer…");
    await onResult(data);
    setStatus("Complete", "done");
  } catch (err) {
    const cancelled = controller.signal.aborted && controller.signal.reason === "user-cancelled";
    const timedOut = controller.signal.aborted && controller.signal.reason === "timeout";
    console.error("[DepthWizard] Processing error:", err);
    if (cancelled) {
      setStatus("cancelled", "error");
    } else {
      setStatus("failed", "error");
      if (!err.isMeshLoadError) {
        showError(
          timedOut
            ? `Request timed out after ${REQUEST_TIMEOUT_MS / 1000}s. The server may still be processing — try a smaller image or lower mesh resolution.`
            : err.message || "Processing failed for an unknown reason."
        );
        viewportHint.textContent = "Processing failed. See the error above.";
        viewportHint.style.display = "flex";
      }
    }
  } finally {
    clearTimeout(timeoutId);
    activeController = null;
    runBtn.textContent = "Generate terrain";
    runBtn.classList.remove("run-btn-cancel");
    hideLoading();
  }
});

async function onResult(data) {
  resultBlock.hidden = false;
  rModel.textContent = data.model_type || "unknown";
  rMethod.textContent = data.depth_method;
  rCalibration.textContent = data.calibration?.method || "unknown";
  rGeo.textContent = data.is_georeferenced ? "yes" : "no";
  rAbsolute.textContent = data.absolute_dsm ? "Absolute" : "Relative Depth Terrain";
  rRange.textContent = `${data.height_range_m[0].toFixed(1)} – ${data.height_range_m[1].toFixed(1)} m`;
  rTime.textContent = `${(data.processing_time_s || 0).toFixed(2)} s`;
  if (data.scientific_note) {
    scientificWarning.hidden = false;
    scientificWarning.textContent = data.scientific_note;
  } else {
    scientificWarning.hidden = true;
    scientificWarning.textContent = "";
  }
  depthPreview.src = data.urls.depth_preview;

  if (data.urls.dsm) {
    dsmLink.href = data.urls.dsm;
    dsmLink.hidden = false;
  } else {
    dsmLink.hidden = true;
  }

  const calibration = data.calibration || {};
  const hasValidation = Number.isFinite(calibration.fit_rmse_m);
  validationBlock.hidden = !hasValidation;
  if (hasValidation) {
    rRmse.textContent = `${calibration.fit_rmse_m.toFixed(2)} m`;
    rMae.textContent = `${(calibration.fit_mae_m || 0).toFixed(2)} m`;
    rCorrelation.textContent = Number.isFinite(calibration.fit_correlation)
      ? calibration.fit_correlation.toFixed(3) : "—";
    const errors = calibration.tile_errors || [];
    tileErrors.textContent = errors.length
      ? `Local tiles: ${errors.length} · median RMSE ${median(errors.map((item) => item.rmse_m)).toFixed(2)} m`
      : "Global fit; no local tile error map available.";
  }

  viewportHint.style.display = "none";
  toolbar.hidden = false;
  probe.hidden = false;
  await loadMesh(data.urls.mesh);
}

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] || 0;
}

// ---------- Three.js viewer ----------
const canvas = document.getElementById("viewportCanvas");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1220);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100000);
camera.position.set(100, 100, 100);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x0b1220, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = true;
controls.enableZoom = true;
// Keep this a "flythrough" terrain viewer, not a free-roam camera — don't
// let users rotate under/through the mesh and end up looking at its back.
controls.maxPolarAngle = THREE.MathUtils.degToRad(89);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
// Soft sky/ground fill in addition to the two directional lights, so
// surfaces facing away from the sun aren't lit by flat ambient alone.
scene.add(new THREE.HemisphereLight(0xbdd6ff, 0x2b2013, 0.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(100, 200, 100);
scene.add(sun);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
fillLight.position.set(-100, 100, -100);
scene.add(fillLight);

let terrainMesh = null;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let currentMode = "texture";

function resizeRenderer() {
  const parent = canvas.parentElement;
  if (!parent) return;
  const width = parent.clientWidth || 800;
  const height = parent.clientHeight || 600;
  if (width <= 0 || height <= 0) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resizeRenderer);
// The viewport's size also changes when sidebar content toggles (e.g. the
// result panel appearing) without a window resize event firing, so watch
// the canvas's parent directly instead of polling getBoundingClientRect()
// every animation frame — that forced a layout recalc 60x/sec and could
// stall the main thread long enough to abort in-flight mesh downloads.
new ResizeObserver(resizeRenderer).observe(canvas.parentElement);
resizeRenderer();

function disposeTerrain(mesh) {
  mesh.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach((m) => m.dispose());
    }
  });
}

// Active camera tween state, driven from the animate() loop. Regenerating
// after the user has manually framed a shot used to snap the camera
// instantly to the new fit; this eases between the old and new views instead.
let cameraTween = null;
const CAMERA_TWEEN_MS = 600;

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// Auto-fit the camera to a mesh: it's in real-world meters (or a
// best-guess unit grid for non-georeferenced input), so a fixed camera
// position would put small or huge scenes out of view. Used both after a
// new mesh loads and from the "Reset view" button.
function fitCameraToObject(object, { animate = false } = {}) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxSize = Math.max(size.x, size.y, size.z, 1e-3);
  const distance = Math.max(maxSize * 1.4, 10);

  camera.near = Math.max(0.01, maxSize / 10000);
  camera.far = Math.max(1000, maxSize * 20);
  camera.updateProjectionMatrix();

  const endPosition = new THREE.Vector3(
    center.x + distance * 0.8,
    center.y + distance * 0.65,
    center.z + distance * 0.8
  );

  controls.minDistance = Math.max(maxSize * 0.02, 0.5);
  controls.maxDistance = Math.max(maxSize * 10, 100);

  sun.position.set(center.x + maxSize, center.y + maxSize * 2, center.z + maxSize);
  fillLight.position.set(center.x - maxSize, center.y + maxSize, center.z - maxSize);

  if (animate) {
    cameraTween = {
      startPosition: camera.position.clone(),
      startTarget: controls.target.clone(),
      endPosition,
      endTarget: center.clone(),
      startTime: performance.now(),
    };
  } else {
    cameraTween = null;
    camera.position.copy(endPosition);
    controls.target.copy(center);
    controls.update();
  }
}

function loadMesh(url) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        if (terrainMesh) {
          scene.remove(terrainMesh);
          disposeTerrain(terrainMesh);
        }
        const hadPreviousMesh = Boolean(terrainMesh);
        terrainMesh = gltf.scene;

        terrainMesh.traverse((child) => {
          if (!child.isMesh) return;
          // Always compute normals, independent of whether this mesh
          // happens to carry a vertex-color attribute — lighting shouldn't
          // be incidentally gated on an unrelated attribute's presence.
          child.geometry.computeVertexNormals();
          if (child.geometry.attributes.color) {
            child.material.vertexColors = true;
            child.userData.baseColors = child.geometry.attributes.color.array.slice();
            child.userData.baseMap = child.material.map || null;
          }
          child.material.needsUpdate = true;
        });

        scene.add(terrainMesh);
        fitCameraToObject(terrainMesh, { animate: hadPreviousMesh });

        applyMode(currentMode);
        resizeRenderer();
        viewportHint.style.display = "none";
        resolve();
      },
      undefined,
      (error) => {
        console.error("[DepthWizard] Failed to load GLB:", error);
        showError("The 3D terrain mesh could not be loaded. See the browser console for details.");
        viewportHint.textContent = "3D terrain could not be loaded.";
        viewportHint.style.display = "flex";
        reject(Object.assign(new Error("mesh load failed"), { isMeshLoadError: true }));
      }
    );
  });
}

function applyMode(mode) {
  currentMode = mode;
  if (!terrainMesh) return;
  terrainMesh.traverse((child) => {
    if (!child.isMesh) return;
    const geom = child.geometry;
    const colorAttr = geom.attributes.color;
    if (!colorAttr) return;

    if (mode === "texture") {
      colorAttr.array.set(child.userData.baseColors);
      colorAttr.needsUpdate = true;
      child.material.map = child.userData.baseMap;
      child.material.needsUpdate = true;
      return;
    }

    // Elevation ramp is a pure hypsometric tint — the RGB texture would
    // otherwise multiply into the vertex-color ramp and wash it out into
    // a noisy blend of both, rather than a clean color gradient.
    child.material.map = null;
    child.material.needsUpdate = true;

    const pos = geom.attributes.position;
    let minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i);
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const span = Math.max(maxY - minY, 1e-6);
    for (let i = 0; i < pos.count; i++) {
      const t = (pos.getY(i) - minY) / span;
      const [r, g, b] = elevationRamp(t);
      colorAttr.setXYZW(i, r * 255, g * 255, b * 255, 255);
    }
    colorAttr.needsUpdate = true;
  });
}

function elevationRamp(t) {
  const stops = [
    [0.0, 0.05, 0.35, 0.85],
    [0.35, 0.05, 0.75, 0.75],
    [0.65, 0.95, 0.7, 0.15],
    [1.0, 0.9, 0.15, 0.1],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0 || 1);
      return [r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f];
    }
  }
  return [0.9, 0.15, 0.1];
}

toolbar.querySelectorAll(".mode-btn[data-mode]").forEach((btn) => {
  btn.addEventListener("click", () => {
    toolbar.querySelectorAll(".mode-btn[data-mode]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    applyMode(btn.dataset.mode);
  });
});

resetViewBtn.addEventListener("click", () => {
  if (terrainMesh) fitCameraToObject(terrainMesh, { animate: true });
});

// probe: raycast under the pointer — the mesh's Y coordinate IS the
// elevation in meters, so no extra rescaling is needed here.
canvas.addEventListener("pointermove", (e) => {
  if (!terrainMesh) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  const hits = raycaster.intersectObject(terrainMesh, true);
  if (!hits.length) return;
  const hit = hits[0];

  const elevM = hit.point.y;
  let slopeDeg = 0;
  if (hit.face) {
    const normal = hit.face.normal.clone().transformDirection(terrainMesh.matrixWorld);
    const dot = THREE.MathUtils.clamp(normal.dot(new THREE.Vector3(0, 1, 0)), -1, 1);
    slopeDeg = THREE.MathUtils.radToDeg(Math.acos(dot));
  }
  probeElev.textContent = `${elevM.toFixed(1)} m`;
  probeSlope.textContent = `${slopeDeg.toFixed(1)}°`;
});

function updateCameraTween() {
  if (!cameraTween) return;
  const t = Math.min(1, (performance.now() - cameraTween.startTime) / CAMERA_TWEEN_MS);
  const eased = easeInOutCubic(t);
  camera.position.lerpVectors(cameraTween.startPosition, cameraTween.endPosition, eased);
  controls.target.lerpVectors(cameraTween.startTarget, cameraTween.endTarget, eased);
  if (t >= 1) cameraTween = null;
}

(function animate() {
  requestAnimationFrame(animate);
  updateCameraTween();
  controls.update();
  renderer.render(scene, camera);
})();
