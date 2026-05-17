const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");

const {
  validateThresholds,
  loadThresholds,
} = require("../../exports/web/asem/thresholds.js");

function validThresholds() {
  return JSON.parse(fs.readFileSync("exports/web/thresholds.json", "utf8"));
}

test("committed thresholds.json validates with Stage-3 calibrated percentiles", () => {
  const thresholds = validateThresholds(validThresholds());
  assert.equal(thresholds.schema_version, "asem_rev3_thresholds_v1");
  const { energy_in_support_p05: p05, energy_in_support_p95: p95 } =
    thresholds.support_calibration;
  assert.equal(Number.isFinite(p05), true, "p05 must be measured (non-null) post Stage-3");
  assert.equal(Number.isFinite(p95), true, "p95 must be measured (non-null) post Stage-3");
  assert.equal(p05 < p95, true, "p05 must be < p95");
});

test("validator still tolerates both-null percentiles (pre-Stage-3 bootstrap)", () => {
  const t = validThresholds();
  t.support_calibration = {
    method: "energy_minmax",
    energy_in_support_p05: null,
    energy_in_support_p95: null,
  };
  assert.equal(validateThresholds(t).support_calibration.energy_in_support_p05, null);
});

test("malformed threshold values fail validation loudly", () => {
  const thresholds = validThresholds();
  thresholds.calibration_T = 0;
  assert.throws(() => validateThresholds(thresholds), /calibration_T must be > 0/);
});

test("missing thresholds.json produces a visible startup-load error path", async () => {
  const fetchImpl = async () => ({ ok: false, status: 404, statusText: "Not Found" });
  await assert.rejects(() => loadThresholds("thresholds.json", fetchImpl), /Failed to load thresholds.json/);
});

test("malformed thresholds.json produces a visible startup-load error path", async () => {
  const fetchImpl = async () => ({
    ok: true,
    json: async () => {
      throw new Error("bad json");
    },
  });
  await assert.rejects(() => loadThresholds("thresholds.json", fetchImpl), /Malformed thresholds.json/);
});
