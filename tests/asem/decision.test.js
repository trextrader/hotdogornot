const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  topK,
  top2,
  requiredEvidenceVisible,
  decide,
} = require("../../exports/web/asem/decision.js");

const labels = [
  "1.85mm-M",
  "1.85mm-F",
  "2.4mm-M",
  "2.4mm-F",
  "2.92mm-M",
  "2.92mm-F",
  "3.5mm-M",
  "3.5mm-F",
  "SMA-F",
  "SMA-M",
];

const thresholds = {
  box_min: 0.25,
  accept: 0.85,
  margin: 0.2,
  unsupported: 0.6,
  q: {
    center_res_min: 0.85,
  },
};

function okQuality(overrides = {}) {
  return {
    blur_var: 100,
    glare_frac: 0,
    roi_scale: 0.2,
    center_res: 1,
    oblique_proxy: 0,
    q_low: false,
    dominant: "none",
    ...overrides,
  };
}

function baseInput(overrides = {}) {
  return {
    box_conf: 0.9,
    quality: okQuality(),
    probabilities: [0.9, 0.04, 0.02, 0.01, 0.01, 0.005, 0.005, 0.004, 0.003, 0.003],
    labels,
    s_ood: 0.1,
    thresholds,
    traceBase: { frame_id: "test" },
    ...overrides,
  };
}

test("decide returns no_connector_found when box_conf is below threshold", () => {
  const result = decide(baseInput({ box_conf: 0.1 }));
  assert.equal(result.status, "ABSTAIN");
  assert.equal(result.reason, "no_connector_found");
});

test("decide returns need_better_focus for blur, glare, and low_roi_scale", () => {
  for (const dominant of ["blur", "glare", "low_roi_scale"]) {
    const result = decide(baseInput({ quality: okQuality({ q_low: true, dominant }) }));
    assert.equal(result.status, "ABSTAIN");
    assert.equal(result.reason, "need_better_focus");
  }
});

test("decide returns need_second_angle for oblique and poor_face_visibility", () => {
  for (const dominant of ["oblique", "poor_face_visibility"]) {
    const result = decide(baseInput({ quality: okQuality({ q_low: true, dominant }) }));
    assert.equal(result.status, "ABSTAIN");
    assert.equal(result.reason, "need_second_angle");
  }
});

test("decide returns unsupported_connector when support score trips and quality is OK", () => {
  const result = decide(baseInput({ s_ood: 0.6 }));
  assert.equal(result.status, "ABSTAIN");
  assert.equal(result.reason, "unsupported_connector");
});

test("decide accepts only when box, quality, support, required-visible, probability, and margin pass", () => {
  const result = decide(baseInput());
  assert.equal(result.status, "ACCEPT");
  assert.equal(result.reason, "accepted");
  assert.equal(result.class_name, labels[0]);
  assert.equal(result.margin > thresholds.margin, true);
});

test("decide returns ambiguous when support and quality pass but probability or margin fails", () => {
  const result = decide(baseInput({
    probabilities: [0.5, 0.45, 0.02, 0.01, 0.01, 0.005, 0.002, 0.001, 0.001, 0.001],
  }));
  assert.equal(result.status, "ABSTAIN");
  assert.equal(result.reason, "ambiguous");
  assert.equal(result.alternatives.length, 5);
});

test("required-visible failure returns need_better_focus and never ambiguous", () => {
  const result = decide(baseInput({
    quality: okQuality({ center_res: 0.4 }),
    probabilities: [0.96, 0.01, 0.01, 0.005, 0.005, 0.003, 0.002, 0.002, 0.002, 0.001],
  }));
  assert.equal(requiredEvidenceVisible(labels[0], okQuality({ center_res: 0.4 }), thresholds), false);
  assert.equal(result.status, "ABSTAIN");
  assert.equal(result.reason, "need_better_focus");
  assert.notEqual(result.reason, "ambiguous");
});

test("top2 is stable and deterministic for ties by lower label index", () => {
  const tied = [0.4, 0.4, 0.1, 0.1, 0, 0, 0, 0, 0, 0];
  const [first, second] = top2(tied, labels);
  assert.equal(first.index, 0);
  assert.equal(second.index, 1);
});

test("topK defaults to five and sorts probability descending with index tie-break", () => {
  const probs = [0.2, 0.5, 0.5, 0.1, 0.3, 0.05, 0.05, 0.04, 0.03, 0.02];
  const ranked = topK(probs, labels);
  assert.equal(ranked.length, 5);
  assert.deepEqual(ranked.map((item) => item.index), [1, 2, 4, 0, 3]);
});

test("decision module and tests contain no out-of-scope connector labels", () => {
  const forbidden = [
    ["RP", "SMA"].join("-"),
    ["right", "angle"].join("-"),
    "bulk" + "head",
    ["cable", "end"].join("-"),
    ["board", "mount"].join("-"),
    "generic" + " SMA",
  ];
  const files = [
    path.join(__dirname, "../../exports/web/asem/decision.js"),
    __filename,
  ];
  const contents = files.map((file) => fs.readFileSync(file, "utf8")).join("\n");
  for (const token of forbidden) {
    assert.equal(contents.includes(token), false, `${token} must not appear in Rev-3 decision code/tests`);
  }
});
