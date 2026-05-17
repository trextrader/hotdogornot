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

test("valid thresholds.json validates with null support percentiles before Stage 3", () => {
  const thresholds = validateThresholds(validThresholds());
  assert.equal(thresholds.schema_version, "asem_rev3_thresholds_v1");
  assert.equal(thresholds.support_calibration.energy_in_support_p05, null);
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
