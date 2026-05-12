/**
 * RF Connector AI — Browser ONNX Runtime Inference
 *
 * Architecture:
 *   image → YOLO detector (ONNX) → crop each bbox → classifier (ONNX) → results
 *
 * Both models run entirely client-side via onnxruntime-web.
 * No server, no uploads, works offline once models are cached.
 */

// --- Configuration -----------------------------------------------------------
const DETECTOR_URL = "models/detector.onnx";
const CLASSIFIER_URL = "models/classifier.onnx";
const VOCABS_URL = "models/classifier_vocabs.json";
const DET_SIZE = 640;
const CLS_SIZE = 384;
const CONF_THRESHOLD = 0.35;
const NMS_IOU_THRESHOLD = 0.45;

const FAMILY_NAMES = ["2.4MM", "2.92MM", "3.5MM"];
const DISPLAY_HEADS = ["family", "side_a_gender", "precision_family", "polarity", "mount_style", "orientation"];

// --- State -------------------------------------------------------------------
let detectorSession = null;
let classifierSession = null;
let vocabs = null;

// --- DOM Elements ------------------------------------------------------------
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const cameraBtn = document.getElementById("camera-btn");
const cameraModal = document.getElementById("camera-modal");
const cameraPreview = document.getElementById("camera-preview");
const captureBtn = document.getElementById("capture-btn");
const cancelCameraBtn = document.getElementById("cancel-camera-btn");
const resultsSection = document.getElementById("results-section");
const outputCanvas = document.getElementById("output-canvas");
const predictionsDiv = document.getElementById("predictions");
const resetBtn = document.getElementById("reset-btn");
const modelStatus = document.getElementById("model-status");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");

// --- Model Loading -----------------------------------------------------------
async function loadModels() {
  try {
    modelStatus.textContent = "Loading detector…";
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    detectorSession = await ort.InferenceSession.create(DETECTOR_URL, {
      executionProviders: ["wasm"],
    });
    modelStatus.textContent = "Loading classifier…";
    classifierSession = await ort.InferenceSession.create(CLASSIFIER_URL, {
      executionProviders: ["wasm"],
    });
    const resp = await fetch(VOCABS_URL);
    vocabs = await resp.json();
    modelStatus.textContent = "Ready";
    modelStatus.className = "status-badge ready";
  } catch (e) {
    console.error("Model load error:", e);
    modelStatus.textContent = "Model load failed — see console";
    modelStatus.className = "status-badge error";
  }
}

// --- Image Preprocessing -----------------------------------------------------
function preprocessForDetector(imageData, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  // Letterbox: scale to fit, pad with gray
  const scale = Math.min(size / imageData.width, size / imageData.height);
  const nw = Math.round(imageData.width * scale);
  const nh = Math.round(imageData.height * scale);
  const dx = Math.round((size - nw) / 2);
  const dy = Math.round((size - nh) / 2);
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(imageData, dx, dy, nw, nh);
  const pixels = ctx.getImageData(0, 0, size, size).data;
  const float32 = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    float32[i] = pixels[i * 4] / 255;                    // R
    float32[size * size + i] = pixels[i * 4 + 1] / 255;  // G
    float32[2 * size * size + i] = pixels[i * 4 + 2] / 255; // B
  }
  return { tensor: new ort.Tensor("float32", float32, [1, 3, size, size]), scale, dx, dy };
}

function preprocessForClassifier(canvas, size) {
  const resized = document.createElement("canvas");
  resized.width = size;
  resized.height = size;
  const ctx = resized.getContext("2d");
  ctx.drawImage(canvas, 0, 0, size, size);
  const pixels = ctx.getImageData(0, 0, size, size).data;
  const float32 = new Float32Array(3 * size * size);
  // ImageNet normalization
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  for (let i = 0; i < size * size; i++) {
    float32[i] = (pixels[i * 4] / 255 - mean[0]) / std[0];
    float32[size * size + i] = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1];
    float32[2 * size * size + i] = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor("float32", float32, [1, 3, size, size]);
}

// --- YOLO Postprocessing -----------------------------------------------------
function parseYoloOutput(output, scale, dx, dy, origW, origH) {
  // YOLO output shape: [1, 4+nc, num_boxes] transposed
  const data = output.data;
  const numBoxes = output.dims[2];
  const numClasses = output.dims[1] - 4;
  const boxes = [];

  for (let i = 0; i < numBoxes; i++) {
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];

    let bestClass = 0, bestScore = 0;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numBoxes + i];
      if (score > bestScore) { bestScore = score; bestClass = c; }
    }
    if (bestScore < CONF_THRESHOLD) continue;

    // Convert from letterbox coords to original image coords
    const x1 = ((cx - w / 2) - dx) / scale;
    const y1 = ((cy - h / 2) - dy) / scale;
    const x2 = ((cx + w / 2) - dx) / scale;
    const y2 = ((cy + h / 2) - dy) / scale;

    boxes.push({
      x1: Math.max(0, x1), y1: Math.max(0, y1),
      x2: Math.min(origW, x2), y2: Math.min(origH, y2),
      score: bestScore, classId: bestClass,
    });
  }

  // Simple NMS
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  const suppressed = new Set();
  for (let i = 0; i < boxes.length; i++) {
    if (suppressed.has(i)) continue;
    kept.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (suppressed.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > NMS_IOU_THRESHOLD) suppressed.add(j);
    }
  }
  return kept;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

// --- Classification ----------------------------------------------------------
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

async function classifyDetection(img, box) {
  // Crop detection from image
  const cropCanvas = document.createElement("canvas");
  const w = box.x2 - box.x1, h = box.y2 - box.y1;
  cropCanvas.width = Math.max(1, Math.round(w));
  cropCanvas.height = Math.max(1, Math.round(h));
  const ctx = cropCanvas.getContext("2d");
  ctx.drawImage(img, box.x1, box.y1, w, h, 0, 0, cropCanvas.width, cropCanvas.height);

  const tensor = preprocessForClassifier(cropCanvas, CLS_SIZE);
  const feeds = {};
  feeds[classifierSession.inputNames[0]] = tensor;
  const results = await classifierSession.run(feeds);

  const attributes = {};
  for (const headName of classifierSession.outputNames) {
    const logits = Array.from(results[headName].data);
    const probs = softmax(logits);
    const bestIdx = probs.indexOf(Math.max(...probs));
    const headVocab = vocabs[headName] || [];
    attributes[headName] = {
      label: headVocab[bestIdx] || `class_${bestIdx}`,
      confidence: probs[bestIdx],
    };
  }
  return attributes;
}

// --- Pipeline ----------------------------------------------------------------
async function runPipeline(img) {
  showLoading("Running detector…");

  // Detect
  const { tensor, scale, dx, dy } = preprocessForDetector(img, DET_SIZE);
  const detFeeds = {};
  detFeeds[detectorSession.inputNames[0]] = tensor;
  const detResults = await detectorSession.run(detFeeds);
  const detOutput = detResults[detectorSession.outputNames[0]];
  const boxes = parseYoloOutput(detOutput, scale, dx, dy, img.width, img.height);

  // Classify each detection
  showLoading(`Classifying ${boxes.length} detection(s)…`);
  const predictions = [];
  for (const box of boxes) {
    const attrs = await classifyDetection(img, box);
    predictions.push({ box, attrs, detClass: FAMILY_NAMES[box.classId] || "unknown" });
  }

  hideLoading();
  renderResults(img, predictions);
}

// --- Rendering ---------------------------------------------------------------
function renderResults(img, predictions) {
  dropZone.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  // Draw image with bounding boxes
  const ctx = outputCanvas.getContext("2d");
  outputCanvas.width = img.width;
  outputCanvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  const colors = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"];

  predictions.forEach((pred, i) => {
    const { box } = pred;
    const color = colors[i % colors.length];
    const w = box.x2 - box.x1, h = box.y2 - box.y1;

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, Math.min(img.width, img.height) * 0.004);
    ctx.strokeRect(box.x1, box.y1, w, h);

    // Label background
    const label = `${pred.attrs.family?.label || pred.detClass} ${(box.score * 100).toFixed(0)}%`;
    ctx.font = `bold ${Math.max(14, img.width * 0.025)}px Inter, sans-serif`;
    const metrics = ctx.measureText(label);
    const lh = Math.max(18, img.width * 0.03);
    ctx.fillStyle = color;
    ctx.fillRect(box.x1, box.y1 - lh - 4, metrics.width + 12, lh + 4);
    ctx.fillStyle = "white";
    ctx.fillText(label, box.x1 + 6, box.y1 - 6);
  });

  // Prediction cards
  predictionsDiv.innerHTML = "";
  if (predictions.length === 0) {
    predictionsDiv.innerHTML = `<div class="prediction-card"><p style="text-align:center;color:var(--text-dim)">No connectors detected. Try a clearer image.</p></div>`;
    return;
  }

  predictions.forEach((pred, i) => {
    const card = document.createElement("div");
    card.className = "prediction-card";

    const conf = pred.box.score;
    const confClass = conf > 0.8 ? "high" : conf > 0.5 ? "med" : "low";
    const familyLabel = pred.attrs.family?.label || pred.detClass;

    let attrsHtml = "";
    for (const head of DISPLAY_HEADS) {
      if (!pred.attrs[head]) continue;
      const val = pred.attrs[head];
      if (val.label === "not_applicable" || val.label === "unknown") continue;
      const displayName = head.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
      attrsHtml += `<div class="attr-item"><div class="attr-label">${displayName}</div><div class="attr-value">${val.label.replace(/_/g, " ")} <span style="color:var(--text-dim)">${(val.confidence*100).toFixed(0)}%</span></div></div>`;
    }

    card.innerHTML = `
      <div class="det-header">
        <span class="det-label">Detection ${i + 1}: ${familyLabel.replace(/_/g, " ")}</span>
        <span class="det-conf ${confClass}">${(conf * 100).toFixed(1)}%</span>
      </div>
      <div class="attr-grid">${attrsHtml}</div>`;
    predictionsDiv.appendChild(card);
  });
}

// --- UI Helpers --------------------------------------------------------------
function showLoading(text) {
  loadingText.textContent = text;
  loadingOverlay.classList.remove("hidden");
}
function hideLoading() { loadingOverlay.classList.add("hidden"); }

function handleImage(source) {
  if (!detectorSession || !classifierSession) {
    alert("Models are still loading. Please wait.");
    return;
  }
  const img = new Image();
  img.onload = () => runPipeline(img);
  if (source instanceof File) {
    img.src = URL.createObjectURL(source);
  } else if (typeof source === "string") {
    img.src = source;
  } else {
    // Canvas
    img.src = source.toDataURL("image/jpeg");
  }
}

// --- Event Handlers ----------------------------------------------------------
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleImage(e.target.files[0]);
});

dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) handleImage(e.dataTransfer.files[0]);
});

resetBtn.addEventListener("click", () => {
  resultsSection.classList.add("hidden");
  dropZone.classList.remove("hidden");
  predictionsDiv.innerHTML = "";
  fileInput.value = "";
});

// Camera
let cameraStream = null;
cameraBtn.addEventListener("click", async (e) => {
  e.stopPropagation();
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 } },
    });
    cameraPreview.srcObject = cameraStream;
    cameraModal.classList.remove("hidden");
  } catch (err) {
    alert("Camera access denied or not available.");
  }
});

captureBtn.addEventListener("click", () => {
  const canvas = document.createElement("canvas");
  canvas.width = cameraPreview.videoWidth;
  canvas.height = cameraPreview.videoHeight;
  canvas.getContext("2d").drawImage(cameraPreview, 0, 0);
  closeCam();
  handleImage(canvas);
});

cancelCameraBtn.addEventListener("click", closeCam);

function closeCam() {
  cameraModal.classList.add("hidden");
  if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
}

// --- Init --------------------------------------------------------------------
loadModels();
