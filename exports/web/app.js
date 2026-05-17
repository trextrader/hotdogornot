/**
 * RF Connector AI — Browser ONNX Runtime Inference
 *
 * Architecture:
 *   image → YOLO detector (ONNX, single-class "connector") → crop each bbox
 *         → flat 10-class classifier (ONNX) → connector type + confidence
 *
 * Both models run entirely client-side via onnxruntime-web.
 * No server, no uploads, works offline once models are cached.
 *
 * 2026-05-17: rewired from the legacy multi-head attribute model to the
 * flat 10-class EfficientNetV2-S classifier. The classifier ONNX bakes
 * ImageNet normalization into the graph (NormalizedClassifier wrapper),
 * so the JS feeds raw [0,1] pixels — do NOT re-apply mean/std here.
 */

// --- Configuration -----------------------------------------------------------
const DETECTOR_URL = "models/detector.onnx";
const CLASSIFIER_URL = "models/classifier.onnx";
const LABELS_URL = "models/classifier_labels.json";
const THRESHOLDS_URL = "thresholds.json";
const MODEL_BUNDLE_ID = "rev3_web_bundle";
const DET_SIZE = 640;
let CLS_SIZE = 384; // overridden from classifier_labels.json input_size
const NMS_IOU_THRESHOLD = 0.45;
const TOP_K = 3; // how many ranked guesses to show per detection

// --- State -------------------------------------------------------------------
let detectorSession = null;
let classifierSession = null;
let CLASS_NAMES = null; // flat 10-class label list (index == class id)
let THRESHOLDS = null; // validated Rev-3 thresholds bundle
let hardcaseStore = null;

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
const hardcaseConsent = document.getElementById("hardcase-consent");
const hardcaseExportBtn = document.getElementById("hardcase-export-btn");
const hardcaseStatus = document.getElementById("hardcase-status");

// --- Model Loading -----------------------------------------------------------
async function loadModels() {
  try {
    assertRev3HelpersLoaded();
    hardcaseStore = new window.AsemHardcase.IndexedDbHardCaseStore();
    initializeHardcaseControls();

    modelStatus.textContent = "Loading thresholds…";
    THRESHOLDS = await window.AsemThresholds.loadThresholds(THRESHOLDS_URL);

    modelStatus.textContent = "Loading labels…";
    const lab = await (await fetch(LABELS_URL)).json();
    CLASS_NAMES = lab.class_names;
    if (lab.input_size) CLS_SIZE = lab.input_size;
    if (!Array.isArray(CLASS_NAMES) || CLASS_NAMES.length === 0) {
      throw new Error("classifier_labels.json has no class_names");
    }
    if (CLASS_NAMES.length !== 10) {
      throw new Error("classifier_labels.json must contain the locked 10-class Rev-3 label set");
    }

    modelStatus.textContent = "Loading detector…";
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    detectorSession = await ort.InferenceSession.create(DETECTOR_URL, {
      executionProviders: ["wasm"],
    });
    modelStatus.textContent = "Loading classifier…";
    classifierSession = await ort.InferenceSession.create(CLASSIFIER_URL, {
      executionProviders: ["wasm"],
    });
    modelStatus.textContent = `Ready (${CLASS_NAMES.length} classes)`;
    modelStatus.className = "status-badge ready";
  } catch (e) {
    console.error("Model load error:", e);
    modelStatus.textContent = e.message || "Model load failed — see console";
    modelStatus.className = "status-badge error";
  }
}

function initializeHardcaseControls() {
  if (!hardcaseConsent || !hardcaseExportBtn || !hardcaseStatus || !hardcaseStore) return;
  hardcaseStore.hasConsent()
    .then((enabled) => {
      hardcaseConsent.checked = enabled;
      hardcaseStatus.textContent = enabled ? "Local hard-case logging enabled" : "";
    })
    .catch(() => {
      hardcaseStatus.textContent = "Local hard-case storage unavailable";
    });
  hardcaseConsent.addEventListener("change", async () => {
    try {
      await hardcaseStore.setConsent(hardcaseConsent.checked);
      hardcaseStatus.textContent = hardcaseConsent.checked ? "Local hard-case logging enabled" : "Local hard-case logging disabled";
    } catch (err) {
      hardcaseStatus.textContent = "Could not update local logging consent";
      hardcaseConsent.checked = false;
    }
  });
  hardcaseExportBtn.addEventListener("click", async () => {
    try {
      const jsonText = await hardcaseStore.exportJson();
      window.AsemHardcase.downloadJson(`asem_rev3_hardcases_${Date.now()}.json`, jsonText);
      hardcaseStatus.textContent = "Export JSON created";
    } catch (err) {
      hardcaseStatus.textContent = "Export unavailable in this WebView";
      console.warn("Hard-case export failed:", err);
    }
  });
}

function assertRev3HelpersLoaded() {
  const missing = [];
  if (!window.AsemThresholds) missing.push("thresholds");
  if (!window.AsemSupport) missing.push("support");
  if (!window.AsemQuality) missing.push("quality");
  if (!window.AsemDecision) missing.push("decision");
  if (!window.AsemHardcase) missing.push("hardcase");
  if (missing.length > 0) {
    throw new Error(`Rev-3 helper load failed: ${missing.join(", ")}`);
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
  // The classifier ONNX (NormalizedClassifier wrapper) bakes ImageNet
  // mean/std into the graph and expects raw [0,1] NCHW. Do NOT normalize
  // here — doing so double-normalizes and destroys accuracy.
  const resized = document.createElement("canvas");
  resized.width = size;
  resized.height = size;
  const ctx = resized.getContext("2d");
  ctx.drawImage(canvas, 0, 0, size, size);
  const pixels = ctx.getImageData(0, 0, size, size).data;
  const float32 = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    float32[i] = pixels[i * 4] / 255;                       // R
    float32[size * size + i] = pixels[i * 4 + 1] / 255;     // G
    float32[2 * size * size + i] = pixels[i * 4 + 2] / 255;  // B
  }
  return new ort.Tensor("float32", float32, [1, 3, size, size]);
}

// --- YOLO Postprocessing -----------------------------------------------------
function parseYoloOutput(output, scale, dx, dy, origW, origH, minScore) {
  // YOLO output shape: [1, 4+nc, num_boxes]. Detector is single-class
  // ("connector"), so nc=1 and the one class score is the box confidence.
  const data = output.data;
  const numBoxes = output.dims[2];
  const numClasses = output.dims[1] - 4;
  const boxes = [];

  for (let i = 0; i < numBoxes; i++) {
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];

    let bestScore = 0;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numBoxes + i];
      if (score > bestScore) bestScore = score;
    }
    if (bestScore < minScore) continue;

    // Convert from letterbox coords to original image coords
    const x1 = ((cx - w / 2) - dx) / scale;
    const y1 = ((cy - h / 2) - dy) / scale;
    const x2 = ((cx + w / 2) - dx) / scale;
    const y2 = ((cy + h / 2) - dy) / scale;

    boxes.push({
      x1: Math.max(0, x1), y1: Math.max(0, y1),
      x2: Math.min(origW, x2), y2: Math.min(origH, y2),
      score: bestScore,
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

async function classifyDetection(img, box) {
  // Crop detection from image
  const cropCanvas = document.createElement("canvas");
  const w = box.x2 - box.x1, h = box.y2 - box.y1;
  cropCanvas.width = Math.max(1, Math.round(w));
  cropCanvas.height = Math.max(1, Math.round(h));
  const ctx = cropCanvas.getContext("2d");
  ctx.drawImage(img, box.x1, box.y1, w, h, 0, 0, cropCanvas.width, cropCanvas.height);
  const roiImageData = ctx.getImageData(0, 0, cropCanvas.width, cropCanvas.height);
  const quality = window.AsemQuality.estimateQuality({
    frameWidth: img.width,
    frameHeight: img.height,
    roiImageData,
    bbox: box,
    thresholds: THRESHOLDS,
  });

  const tensor = preprocessForClassifier(cropCanvas, CLS_SIZE);
  const feeds = {};
  feeds[classifierSession.inputNames[0]] = tensor;
  const results = await classifierSession.run(feeds);

  // Flat single head: one logits tensor of length == CLASS_NAMES.length
  const logits = Array.from(results[classifierSession.outputNames[0]].data);
  const probabilities = window.AsemSupport.softmaxWithTemperature(logits, THRESHOLDS.calibration_T);
  const rankedFull = window.AsemDecision
    .topK(probabilities, CLASS_NAMES, CLASS_NAMES.length)
    .map((r) => ({ label: r.class_name, confidence: r.prob, index: r.index }));
  const support = window.AsemSupport.computeSupportScore({
    logits,
    thresholds: THRESHOLDS,
  });
  const decision = window.AsemDecision.decide({
    box_conf: box.score,
    quality,
    probabilities,
    labels: CLASS_NAMES,
    s_ood: support.s_ood,
    thresholds: THRESHOLDS,
    traceBase: { model_bundle_id: MODEL_BUNDLE_ID, thresholds_version: THRESHOLDS.thresholds_version },
  });
  const top = rankedFull[0];
  const second = rankedFull[1] || { label: "", confidence: 0 };
  const topAlternatives = rankedFull.slice(0, 5);
  const legacyOutput = {
    class: top.label,
    confidence: top.confidence,
    bbox: {
      x1: box.x1,
      y1: box.y1,
      x2: box.x2,
      y2: box.y2,
      score: box.score,
    },
    top_k: rankedFull.slice(0, TOP_K),
  };
  const trace = window.AsemHardcase.buildDecisionTrace({
    model_bundle_id: MODEL_BUNDLE_ID,
    thresholds_version: THRESHOLDS.thresholds_version,
    frame: {
      full_frame_saved: false,
      roi_saved: false,
      frame_width: img.width,
      frame_height: img.height,
      roi_bbox_xyxy: [box.x1, box.y1, box.x2, box.y2],
      box_conf: box.score,
    },
    quality,
    classification: {
      top1: top.label,
      top1_prob: top.confidence,
      top2: second.label,
      top2_prob: second.confidence,
      margin: top.confidence - second.confidence,
      topk: topAlternatives.map((r) => ({ class: r.label, prob: r.confidence })),
    },
    support: {
      s_ood: support.s_ood,
      method: support.method,
      unsupported_threshold: THRESHOLDS.unsupported,
    },
    decision,
    thresholds_snapshot: THRESHOLDS,
  });
  const asem_rev3 = {
    status: decision.status,
    reason: decision.reason,
    user_guidance: decision.user_guidance,
    top1: decision.top1 || { class_name: top.label, prob: top.confidence, index: top.index },
    top2: decision.top2 || { class_name: second.label, prob: second.confidence, index: second.index },
    margin: top.confidence - second.confidence,
    s_ood: support.s_ood,
    support_method: support.method,
    quality,
    thresholds_version: THRESHOLDS.thresholds_version,
    trace,
  };
  return {
    top,
    ranked: rankedFull.slice(0, TOP_K),
    logits,
    probabilities,
    quality,
    support,
    decision,
    legacy_output: legacyOutput,
    asem_rev3,
  };
}

// --- Pipeline ----------------------------------------------------------------
async function runPipeline(img) {
  try {
    showLoading("Running detector…");

    const { tensor, scale, dx, dy } = preprocessForDetector(img, DET_SIZE);
    const detFeeds = {};
    detFeeds[detectorSession.inputNames[0]] = tensor;
    const detResults = await detectorSession.run(detFeeds);
    const detOutput = detResults[detectorSession.outputNames[0]];
    const boxes = parseYoloOutput(detOutput, scale, dx, dy, img.width, img.height, THRESHOLDS.box_min);
    if (boxes.length === 0) {
      const decision = {
        status: "ABSTAIN",
        reason: "no_connector_found",
        user_guidance: window.AsemDecision.GUIDANCE.no_connector_found,
      };
      const trace = window.AsemHardcase.buildDecisionTrace({
        model_bundle_id: MODEL_BUNDLE_ID,
        thresholds_version: THRESHOLDS.thresholds_version,
        frame: {
          full_frame_saved: false,
          roi_saved: false,
          frame_width: img.width,
          frame_height: img.height,
          roi_bbox_xyxy: [0, 0, 0, 0],
          box_conf: 0,
        },
        quality: { dominant: "none" },
        classification: { topk: [] },
        support: { s_ood: 0, method: "energy", unsupported_threshold: THRESHOLDS.unsupported },
        decision,
        thresholds_snapshot: THRESHOLDS,
      });
      await maybeLogHardcase(trace, decision);
    }

    showLoading(`Classifying ${boxes.length} detection(s)…`);
    const predictions = [];
    for (const box of boxes) {
      const cls = await classifyDetection(img, box);
      predictions.push({ box, cls });
      await maybeLogHardcase(cls.asem_rev3.trace, cls.decision);
    }

    hideLoading();
    renderResults(img, predictions);
  } catch (err) {
    hideLoading();
    console.error("Inference error:", err);
    renderInferenceError(img, err);
  }
}

// --- Rendering ---------------------------------------------------------------
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function renderResults(img, predictions) {
  dropZone.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  const ctx = outputCanvas.getContext("2d");
  outputCanvas.width = img.width;
  outputCanvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  const colors = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"];

  predictions.forEach((pred, i) => {
    const { box, cls } = pred;
    const color = colors[i % colors.length];
    const w = box.x2 - box.x1, h = box.y2 - box.y1;

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, Math.min(img.width, img.height) * 0.004);
    ctx.strokeRect(box.x1, box.y1, w, h);

    const label = `${cls.top.label} ${(cls.top.confidence * 100).toFixed(0)}%`;
    ctx.font = `bold ${Math.max(14, img.width * 0.025)}px Inter, sans-serif`;
    const metrics = ctx.measureText(label);
    const lh = Math.max(18, img.width * 0.03);
    ctx.fillStyle = color;
    ctx.fillRect(box.x1, box.y1 - lh - 4, metrics.width + 12, lh + 4);
    ctx.fillStyle = "white";
    ctx.fillText(label, box.x1 + 6, box.y1 - 6);
  });

  predictionsDiv.innerHTML = "";
  if (predictions.length === 0) {
    predictionsDiv.innerHTML = `
      <div class="prediction-card">
        <div class="decision-banner abstain">
          <span class="decision-status">ABSTAIN</span>
          <span class="decision-reason">no_connector_found</span>
        </div>
        <p class="guidance">No connector was found clearly enough. Move closer and center the connector.</p>
      </div>`;
    return;
  }

  predictions.forEach((pred, i) => {
    const card = document.createElement("div");
    card.className = "prediction-card";

    const top = pred.cls.top;
    const conf = top.confidence;
    const asem = pred.cls.asem_rev3;
    const confClass = conf > 0.8 ? "high" : conf > 0.5 ? "med" : "low";
    const bannerClass = asem.status === "ACCEPT" ? "accept" : "abstain";

    let ranksHtml = "";
    for (const r of pred.cls.ranked) {
      ranksHtml += `<div class="attr-item"><div class="attr-label">${escapeHtml(r.label)}</div><div class="attr-value">${pct(r.confidence)}</div></div>`;
    }

    card.innerHTML = `
      <div class="decision-banner ${bannerClass}">
        <span class="decision-status">${escapeHtml(asem.status)}</span>
        <span class="decision-reason">${escapeHtml(asem.reason)}</span>
      </div>
      <div class="det-header">
        <span class="det-label">Detection ${i + 1}: ${escapeHtml(top.label)}</span>
        <span class="det-conf ${confClass}">${pct(conf)}</span>
      </div>
      <p class="guidance">${escapeHtml(asem.user_guidance)}</p>
      <div class="attr-grid">${ranksHtml}</div>
      <div class="attr-item" style="margin-top:8px">
        <div class="attr-label">Box confidence</div>
        <div class="attr-value">${pct(pred.box.score)}</div>
      </div>
      <div class="debug-grid">
        <div class="attr-item"><div class="attr-label">Margin</div><div class="attr-value">${pct(asem.margin)}</div></div>
        <div class="attr-item"><div class="attr-label">s_ood</div><div class="attr-value">${asem.s_ood.toFixed(3)}</div></div>
        <div class="attr-item"><div class="attr-label">Q_t dominant</div><div class="attr-value">${escapeHtml(asem.quality.dominant)}</div></div>
        <div class="attr-item"><div class="attr-label">Center res</div><div class="attr-value">${asem.quality.center_res.toFixed(3)}</div></div>
        <div class="attr-item"><div class="attr-label">Thresholds</div><div class="attr-value">${escapeHtml(asem.thresholds_version)}</div></div>
      </div>`;
    predictionsDiv.appendChild(card);
  });
}

function renderInferenceError(img, err) {
  dropZone.classList.add("hidden");
  resultsSection.classList.remove("hidden");
  const ctx = outputCanvas.getContext("2d");
  outputCanvas.width = img.width;
  outputCanvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  predictionsDiv.innerHTML = `
    <div class="prediction-card">
      <div class="decision-banner abstain">
        <span class="decision-status">ABSTAIN</span>
        <span class="decision-reason">rev3_startup_or_calibration_error</span>
      </div>
      <p class="guidance">${escapeHtml(err.message || "Rev-3 inference failed loudly; check thresholds and calibration.")}</p>
    </div>`;
}

// --- UI Helpers --------------------------------------------------------------
function showLoading(text) {
  loadingText.textContent = text;
  loadingOverlay.classList.remove("hidden");
}
function hideLoading() { loadingOverlay.classList.add("hidden"); }

async function maybeLogHardcase(trace, decision) {
  if (!hardcaseStore || !trace || !decision) return;
  if (!window.AsemHardcase.shouldLogHardcase(decision, THRESHOLDS)) return;
  try {
    const result = await hardcaseStore.addTrace(trace, null);
    if (result.stored && hardcaseStatus) {
      hardcaseStatus.textContent = "Hard case saved locally";
    }
  } catch (err) {
    if (hardcaseStatus) hardcaseStatus.textContent = "Hard-case save skipped";
    console.warn("Hard-case save skipped:", err);
  }
}

function handleImage(source) {
  if (!detectorSession || !classifierSession || !CLASS_NAMES || !THRESHOLDS) {
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
