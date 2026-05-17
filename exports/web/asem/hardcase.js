(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.AsemHardcase = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const TRACE_SCHEMA_VERSION = "asem_rev3_trace_v1";
  const DEFAULT_APP_VERSION = "rev3";
  const DEFAULT_DETECTOR_MODEL = "models/detector.onnx";
  const DEFAULT_CLASSIFIER_MODEL = "models/classifier.onnx";
  const FAILURE_TAGS = [
    "blur",
    "poor_center",
    "scale_ambiguous",
    "side_angle_needed",
    "family_confusion",
    "gender_confusion",
    "out_of_support",
    "detector_missed",
  ];

  function finiteOrZero(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : 0;
  }

  function bool(value) {
    return value === true;
  }

  function isoTimestamp(value) {
    if (typeof value === "string" && value.length > 0) return value;
    return new Date().toISOString();
  }

  function bboxArray(frame) {
    if (Array.isArray(frame && frame.roi_bbox_xyxy) && frame.roi_bbox_xyxy.length === 4) {
      return frame.roi_bbox_xyxy.map(finiteOrZero);
    }
    const box = frame && frame.bbox;
    if (box && typeof box === "object") {
      return [box.x1, box.y1, box.x2, box.y2].map(finiteOrZero);
    }
    return [0, 0, 0, 0];
  }

  function normalizeTopK(topk) {
    if (!Array.isArray(topk)) return [];
    return topk.map((item) => ({
      class: String(item.class || item.class_name || item.label || ""),
      prob: finiteOrZero(item.prob !== undefined ? item.prob : item.confidence),
    }));
  }

  function buildDecisionTrace(input) {
    const cfg = input || {};
    const frame = cfg.frame || {};
    const quality = cfg.quality || {};
    const classification = cfg.classification || {};
    const support = cfg.support || {};
    const decision = cfg.decision || {};
    const thresholds = cfg.thresholds_snapshot || cfg.thresholds || {};

    return {
      schema_version: TRACE_SCHEMA_VERSION,
      timestamp_utc: isoTimestamp(cfg.timestamp_utc),
      app_version: cfg.app_version || DEFAULT_APP_VERSION,
      model_bundle_id: cfg.model_bundle_id || "rev3_web_bundle",
      detector_model: cfg.detector_model || DEFAULT_DETECTOR_MODEL,
      classifier_model: cfg.classifier_model || DEFAULT_CLASSIFIER_MODEL,
      thresholds_version: cfg.thresholds_version || thresholds.thresholds_version || "",

      frame: {
        full_frame_saved: bool(frame.full_frame_saved),
        roi_saved: bool(frame.roi_saved),
        frame_width: finiteOrZero(frame.frame_width),
        frame_height: finiteOrZero(frame.frame_height),
        roi_bbox_xyxy: bboxArray(frame),
        box_conf: finiteOrZero(frame.box_conf),
      },

      quality: {
        blur_var: finiteOrZero(quality.blur_var),
        glare_frac: finiteOrZero(quality.glare_frac),
        roi_scale: finiteOrZero(quality.roi_scale),
        center_res: finiteOrZero(quality.center_res),
        oblique_proxy: finiteOrZero(quality.oblique_proxy),
        dominant_low_quality_reason: quality.dominant_low_quality_reason || quality.dominant || "none",
      },

      classification: {
        top1: classification.top1 || "",
        top1_prob: finiteOrZero(classification.top1_prob),
        top2: classification.top2 || "",
        top2_prob: finiteOrZero(classification.top2_prob),
        margin: finiteOrZero(classification.margin),
        topk: normalizeTopK(classification.topk),
      },

      support: {
        s_ood: finiteOrZero(support.s_ood),
        method: support.method || "energy",
        unsupported_threshold: finiteOrZero(support.unsupported_threshold),
      },

      decision: {
        status: decision.status || "ABSTAIN",
        reason: decision.reason || "ambiguous",
        user_guidance: decision.user_guidance || "",
      },

      thresholds_snapshot: thresholds,
    };
  }

  function validateDecisionTrace(trace) {
    const expectedKeys = [
      "schema_version",
      "timestamp_utc",
      "app_version",
      "model_bundle_id",
      "detector_model",
      "classifier_model",
      "thresholds_version",
      "frame",
      "quality",
      "classification",
      "support",
      "decision",
      "thresholds_snapshot",
    ];
    if (!trace || typeof trace !== "object" || Array.isArray(trace)) {
      throw new Error("trace must be an object");
    }
    const keys = Object.keys(trace);
    if (keys.length !== expectedKeys.length || expectedKeys.some((key, index) => keys[index] !== key)) {
      throw new Error("trace does not match asem_rev3_trace_v1 top-level schema");
    }
    if (trace.schema_version !== TRACE_SCHEMA_VERSION) {
      throw new Error("trace schema_version must be asem_rev3_trace_v1");
    }
    if (!Array.isArray(trace.frame.roi_bbox_xyxy) || trace.frame.roi_bbox_xyxy.length !== 4) {
      throw new Error("trace.frame.roi_bbox_xyxy must contain 4 numbers");
    }
    if (!Array.isArray(trace.classification.topk)) {
      throw new Error("trace.classification.topk must be an array");
    }
    return trace;
  }

  function normalizeFailureTag(tag) {
    if (tag === undefined || tag === null || tag === "") {
      return null;
    }
    const normalized = String(tag).trim().toLowerCase().replace(/[\s-]+/g, "_");
    if (!FAILURE_TAGS.includes(normalized)) {
      throw new Error(`Unknown hard-case failure tag: ${tag}`);
    }
    return normalized;
  }

  function randomDraw(randomFn) {
    if (typeof randomFn === "function") {
      return finiteOrZero(randomFn());
    }
    if (typeof randomFn === "number") {
      return finiteOrZero(randomFn);
    }
    return Math.random();
  }

  function shouldLogHardcase(decision, thresholds, randomFn) {
    if (!decision) return false;
    if (decision.status === "ABSTAIN") return true;
    if (decision.status !== "ACCEPT") return false;
    const rate = thresholds && typeof thresholds.hardcase_accept_sample_rate === "number"
      ? thresholds.hardcase_accept_sample_rate
      : 0;
    const draw = randomDraw(randomFn);
    return draw < Math.max(0, Math.min(1, rate));
  }

  function exportHardcasesJson(cases) {
    const records = Array.isArray(cases) ? cases : [];
    const traces = records.map((record) => {
      const trace = record && record.trace ? record.trace : record;
      return validateDecisionTrace(trace);
    });
    const payload = {
      schema_version: "asem_rev3_hardcase_export_v1",
      exported_at_utc: new Date().toISOString(),
      export_mode: "manual_local_json",
      count: traces.length,
      traces,
    };
    return JSON.stringify(payload, null, 2);
  }

  function createDownloadPayload(traces) {
    return exportHardcasesJson(traces);
  }

  async function saveHardcaseLocal(caseRecord) {
    const cfg = caseRecord || {};
    if (cfg.consent !== true) {
      return { stored: false, reason: "consent_required" };
    }

    const trace = cfg.trace || buildDecisionTrace(cfg);
    validateDecisionTrace(trace);
    const thresholds = cfg.thresholds || cfg.thresholds_snapshot || trace.thresholds_snapshot || {};
    if (!shouldLogHardcase(trace.decision, thresholds, cfg.randomFn)) {
      return { stored: false, reason: "sample_skipped" };
    }

    const failureTag = normalizeFailureTag(cfg.failure_tag !== undefined ? cfg.failure_tag : cfg.tag);
    const record = {
      trace,
      failure_tag: failureTag,
      saved_at_utc: new Date().toISOString(),
    };

    if (cfg.store && typeof cfg.store.addTrace === "function") {
      return cfg.store.addTrace(trace, failureTag);
    }
    if (Array.isArray(cfg.cases)) {
      cfg.cases.push(record);
    }
    return { stored: true, record };
  }

  function downloadJson(filename, jsonText, documentRef) {
    const doc = documentRef || (typeof document !== "undefined" ? document : null);
    if (!doc) {
      throw new Error("manual export download requires a browser document");
    }
    const blob = new Blob([jsonText], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = doc.createElement("a");
    link.href = url;
    link.download = filename;
    link.rel = "noopener";
    doc.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  class IndexedDbHardCaseStore {
    constructor(options) {
      const cfg = options || {};
      this.dbName = cfg.dbName || "asem_rev3_hardcases";
      this.storeName = cfg.storeName || "cases";
      this.metaStoreName = cfg.metaStoreName || "meta";
      this.indexedDB = cfg.indexedDB || (typeof indexedDB !== "undefined" ? indexedDB : null);
    }

    open() {
      if (!this.indexedDB) {
        return Promise.reject(new Error("IndexedDB is not available for local hard-case storage"));
      }
      return new Promise((resolve, reject) => {
        const req = this.indexedDB.open(this.dbName, 1);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(this.storeName)) {
            db.createObjectStore(this.storeName, { keyPath: "id", autoIncrement: true });
          }
          if (!db.objectStoreNames.contains(this.metaStoreName)) {
            db.createObjectStore(this.metaStoreName, { keyPath: "key" });
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }

    async setConsent(enabled) {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(this.metaStoreName, "readwrite");
        tx.objectStore(this.metaStoreName).put({ key: "consent", value: enabled === true });
        tx.oncomplete = () => {
          db.close();
          resolve(enabled === true);
        };
        tx.onerror = () => {
          db.close();
          reject(tx.error);
        };
      });
    }

    async hasConsent() {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(this.metaStoreName, "readonly");
        const req = tx.objectStore(this.metaStoreName).get("consent");
        req.onsuccess = () => resolve(req.result ? req.result.value === true : false);
        req.onerror = () => reject(req.error);
        tx.oncomplete = () => db.close();
      });
    }

    async addTrace(trace, tag) {
      if (!(await this.hasConsent())) {
        return { stored: false, reason: "consent_required" };
      }
      validateDecisionTrace(trace);
      const failureTag = normalizeFailureTag(tag);
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(this.storeName, "readwrite");
        const record = {
          trace,
          failure_tag: failureTag,
          stored_at_utc: new Date().toISOString(),
        };
        const req = tx.objectStore(this.storeName).add(record);
        req.onsuccess = () => resolve({ stored: true, id: req.result });
        req.onerror = () => reject(req.error);
        tx.oncomplete = () => db.close();
      });
    }

    async listTraces() {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(this.storeName, "readonly");
        const req = tx.objectStore(this.storeName).getAll();
        req.onsuccess = () => resolve((req.result || []).map((record) => record.trace));
        req.onerror = () => reject(req.error);
        tx.oncomplete = () => db.close();
      });
    }

    async exportJson() {
      const traces = await this.listTraces();
      return exportHardcasesJson(traces);
    }

    async clear() {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(this.storeName, "readwrite");
        tx.objectStore(this.storeName).clear();
        tx.oncomplete = () => {
          db.close();
          resolve(true);
        };
        tx.onerror = () => {
          db.close();
          reject(tx.error);
        };
      });
    }
  }

  return {
    TRACE_SCHEMA_VERSION,
    FAILURE_TAGS,
    buildDecisionTrace,
    validateDecisionTrace,
    shouldLogHardcase,
    shouldLogHardCase: shouldLogHardcase,
    normalizeFailureTag,
    saveHardcaseLocal,
    exportHardcasesJson,
    createDownloadPayload,
    downloadJson,
    IndexedDbHardCaseStore,
  };
});
