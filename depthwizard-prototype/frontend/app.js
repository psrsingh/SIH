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

let selectedFile = null;
let selectedDem = null;
let selectedGcp = null;

// ---------- file inputs ----------
meshRes.addEventListener("input", () => (meshResVal.textContent = meshRes.value));

dropzone.addEventListener("click", () => fileInput.click());
["dragover", "dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => e.preventDefault())
);
dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

demDropzone.addEventListener("click", () => demInput.click());
["dragover", "dragleave", "drop"].forEach((evt) =>
  demDropzone.addEventListener(evt, (e) => e.preventDefault())
);
demDropzone.addEventListener("drop", (e) => handleDem(e.dataTransfer.files[0]));
demInput.addEventListener("change", (e) => handleDem(e.target.files[0]));
gcpDropzone.addEventListener("click", () => gcpInput.click());
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

// ---------- process ----------
runBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    setStatus("select an image first", "error");
    return;
  }
  runBtn.disabled = true;
  setStatus("Uploading", "busy");
  viewportHint.textContent = "Running depth estimation and building the mesh…";
  viewportHint.style.display = "flex";

  const form = new FormData();
  form.append("file", selectedFile);
  if (selectedDem) form.append("reference_dem", selectedDem);
  if (selectedGcp) form.append("gcp_file", selectedGcp);
  form.append("min_height", minHeight.value);
  form.append("max_height", maxHeight.value);
  form.append("mesh_resolution", meshRes.value);

  try {
    setStatus("Running depth inference", "busy");
    const res = await fetch(`${API_BASE}/api/process`, { method: "POST", body: form });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Server returned ${res.status}: ${errText}`);
    }
    const data = await res.json();
    setStatus("Calibrating", "busy");
    onResult(data);
    setStatus("Complete", "done");
  } catch (err) {
    console.error("[DepthWizard] Processing error:", err);
    setStatus("failed — see console", "error");
    viewportHint.textContent = "Processing failed. Check the browser console.";
    viewportHint.style.display = "flex";
  } finally {
    runBtn.disabled = false;
  }
});

function onResult(data) {
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
  loadMesh(data.urls.mesh);
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

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = true;
controls.enableZoom = true;

scene.add(new THREE.AmbientLight(0xffffff, 1.2));
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

function loadMesh(url) {
  const loader = new GLTFLoader();
  loader.load(
    url,
    (gltf) => {
      if (terrainMesh) {
        scene.remove(terrainMesh);
        disposeTerrain(terrainMesh);
      }
      terrainMesh = gltf.scene;

      terrainMesh.traverse((child) => {
        if (!child.isMesh) return;
        if (child.geometry.attributes.color) {
          child.material.vertexColors = true;
          child.geometry.computeVertexNormals();
          child.userData.baseColors = child.geometry.attributes.color.array.slice();
        }
        child.material.needsUpdate = true;
      });

      scene.add(terrainMesh);

      // Auto-fit the camera: the mesh is in real-world meters (or a
      // best-guess unit grid for non-georeferenced input), so a fixed
      // camera position would put small or huge scenes out of view.
      const box = new THREE.Box3().setFromObject(terrainMesh);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxSize = Math.max(size.x, size.y, size.z, 1e-3);
      const distance = Math.max(maxSize * 1.4, 10);

      camera.near = Math.max(0.01, maxSize / 10000);
      camera.far = Math.max(1000, maxSize * 20);
      camera.updateProjectionMatrix();
      camera.position.set(
        center.x + distance * 0.8,
        center.y + distance * 0.65,
        center.z + distance * 0.8
      );
      controls.target.copy(center);
      controls.minDistance = Math.max(maxSize * 0.02, 0.5);
      controls.maxDistance = Math.max(maxSize * 10, 100);
      controls.update();

      sun.position.set(center.x + maxSize, center.y + maxSize * 2, center.z + maxSize);
      fillLight.position.set(center.x - maxSize, center.y + maxSize, center.z - maxSize);

      applyMode(currentMode);
      resizeRenderer();
      viewportHint.style.display = "none";
    },
    undefined,
    (error) => {
      console.error("[DepthWizard] Failed to load GLB:", error);
      viewportHint.textContent = "3D terrain could not be loaded. Check the browser console.";
      viewportHint.style.display = "flex";
    }
  );
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
      return;
    }

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

toolbar.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    toolbar.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    applyMode(btn.dataset.mode);
  });
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

(function animate() {
  requestAnimationFrame(animate);
  resizeRenderer();
  controls.update();
  renderer.render(scene, camera);
})();
