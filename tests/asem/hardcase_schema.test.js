const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  FAILURE_TAGS,
  TRACE_SCHEMA_VERSION,
  buildDecisionTrace,
  validateDecisionTrace,
  shouldLogHardcase,
  normalizeFailureTag,
  saveHardcaseLocal,
  exportHardcasesJson,
} = require("../../exports/web/asem/hardcase.js");

function sampleTrace(overrides = {}) {
  return buildDecisionTrace({
    timestamp_utc: "2026-05-17T00:00:00Z",
    model_bundle_id: "test_bundle",
    thresholds_version: "rev3_initial_2026_05_17",
    frame: {
      full_frame_saved: true,
      roi_saved: true,
      frame_width: 1280,
      frame_height: 720,
      roi_bbox_xyxy: [10, 20, 110, 120],
      box_conf: 0.9,
    },
    quality: {
      blur_var: 100,
      glare_frac: 0.01,
      roi_scale: 0.1,
      center_res: 0.9,
      oblique_proxy: 0.1,
      dominant: "none",
    },
    classification: {
      top1: "2.4mm-M",
      top1_prob: 0.91,
      top2: "2.4mm-F",
      top2_prob: 0.05,
      margin: 0.86,
      topk: [{ class: "2.4mm-M", prob: 0.91 }],
    },
    support: {
      s_ood: 0.1,
      method: "energy",
      unsupported_threshold: 0.6,
    },
    decision: {
      status: "ACCEPT",
      reason: "accepted",
      user_guidance: "Connector accepted.",
    },
    thresholds_snapshot: {
      thresholds_version: "rev3_initial_2026_05_17",
      unsupported: 0.6,
    },
    ...overrides,
  });
}

test("buildDecisionTrace emits exact asem_rev3_trace_v1 top-level schema", () => {
  const trace = sampleTrace();
  assert.equal(trace.schema_version, TRACE_SCHEMA_VERSION);
  assert.deepEqual(Object.keys(trace), [
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
  ]);
  assert.equal(trace.detector_model, "models/detector.onnx");
  assert.equal(trace.classifier_model, "models/classifier.onnx");
  assert.equal(validateDecisionTrace(trace), trace);
});

test("failure tag set matches Rev-3 contract", () => {
  assert.deepEqual(FAILURE_TAGS, [
    "blur",
    "poor_center",
    "scale_ambiguous",
    "side_angle_needed",
    "family_confusion",
    "gender_confusion",
    "out_of_support",
    "detector_missed",
  ]);
});

test("shouldLogHardcase logs all abstained and samples accepted cases", () => {
  assert.equal(shouldLogHardcase({ status: "ABSTAIN" }, { hardcase_accept_sample_rate: 0 }, () => 0.99), true);
  assert.equal(shouldLogHardcase({ status: "ACCEPT" }, { hardcase_accept_sample_rate: 0.05 }, () => 0.01), true);
  assert.equal(shouldLogHardcase({ status: "ACCEPT" }, { hardcase_accept_sample_rate: 0.05 }, () => 0.9), false);
});

test("saveHardcaseLocal requires consent and stores abstained cases locally", async () => {
  const cases = [];
  const trace = sampleTrace({
    decision: {
      status: "ABSTAIN",
      reason: "ambiguous",
      user_guidance: "The evidence is not strong enough to identify this connector safely.",
    },
  });

  assert.deepEqual(await saveHardcaseLocal({ consent: false, trace, cases }), {
    stored: false,
    reason: "consent_required",
  });

  const result = await saveHardcaseLocal({ consent: true, trace, cases, tag: "family_confusion" });
  assert.equal(result.stored, true);
  assert.equal(cases.length, 1);
  assert.equal(cases[0].failure_tag, "family_confusion");
});

test("saveHardcaseLocal samples accepted cases by hardcase_accept_sample_rate", async () => {
  const skipped = [];
  const stored = [];
  const trace = sampleTrace();

  assert.deepEqual(
    await saveHardcaseLocal({
      consent: true,
      trace,
      thresholds: { hardcase_accept_sample_rate: 0.05 },
      randomFn: () => 0.9,
      cases: skipped,
    }),
    { stored: false, reason: "sample_skipped" }
  );
  assert.equal(skipped.length, 0);

  const result = await saveHardcaseLocal({
    consent: true,
    trace,
    thresholds: { hardcase_accept_sample_rate: 0.05 },
    randomFn: () => 0.01,
    cases: stored,
  });
  assert.equal(result.stored, true);
  assert.equal(stored.length, 1);
});

test("normalizeFailureTag accepts exact tags and rejects invalid tags", () => {
  assert.equal(normalizeFailureTag("poor_center"), "poor_center");
  assert.equal(normalizeFailureTag("side-angle-needed"), "side_angle_needed");
  assert.equal(normalizeFailureTag(null), null);
  assert.throws(() => normalizeFailureTag("unsupported"), /Unknown hard-case failure tag/);
});

test("manual export bundle wraps traces without upload metadata", () => {
  const parsed = JSON.parse(exportHardcasesJson([{ trace: sampleTrace(), failure_tag: "family_confusion" }]));
  assert.equal(parsed.schema_version, "asem_rev3_hardcase_export_v1");
  assert.equal(parsed.export_mode, "manual_local_json");
  assert.equal(parsed.count, 1);
  assert.equal(parsed.traces[0].schema_version, TRACE_SCHEMA_VERSION);
});

test("hardcase module contains no upload code path", () => {
  const text = fs.readFileSync(path.join(__dirname, "../../exports/web/asem/hardcase.js"), "utf8");
  for (const token of ["fetch(", "XMLHttpRequest", "sendBeacon", ".open(\"POST\"", ".open('POST'"]) {
    assert.equal(text.includes(token), false, `${token} must not appear in hardcase.js`);
  }
});
