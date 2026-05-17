const test = require("node:test");
const assert = require("node:assert/strict");

const {
  softmaxWithTemperature,
  logsumexp,
  energyScoreFromLogits,
  normalizeEnergyToSOod,
  sOodFromInSupportProbability,
  computeSupportScore,
} = require("../../exports/web/asem/support.js");

test("softmaxWithTemperature is stable and sums to one", () => {
  const probs = softmaxWithTemperature([1000, 1001, 1002], 2);
  assert.equal(probs.length, 3);
  assert.ok(Math.abs(probs.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.equal(probs[2] > probs[1], true);
});

test("softmaxWithTemperature rejects invalid temperature and empty logits", () => {
  assert.throws(() => softmaxWithTemperature([1, 2], 0), /calibration_T must be > 0/);
  assert.throws(() => softmaxWithTemperature([], 1), /must not be empty/);
});

test("logsumexp and energyScoreFromLogits use the Rev-3 formula", () => {
  const logits = [1, 2, 3];
  const lse = logsumexp(logits);
  const energy = energyScoreFromLogits(logits, 1);
  assert.ok(Math.abs(energy + lse) < 1e-12);
});

test("normalizeEnergyToSOod throws when support percentiles are null", () => {
  assert.throws(
    () => normalizeEnergyToSOod(0.5, {
      method: "energy_minmax",
      energy_in_support_p05: null,
      energy_in_support_p95: null,
    }),
    /energy_in_support_p05 must be a finite number/
  );
});

test("s_ood sign contract converts p_in_support=0.9 to unsupported risk 0.1", () => {
  const s_ood = sOodFromInSupportProbability(0.9);
  assert.ok(Math.abs(s_ood - 0.1) < 1e-12);
  assert.equal(s_ood >= 0.6, false);
});

test("computeSupportScore uses logit-energy without embedding dependency", () => {
  const result = computeSupportScore({
    logits: [3, 2, 1],
    thresholds: {
      calibration_T: 1,
      support_calibration: {
        method: "energy_minmax",
        energy_in_support_p05: -4,
        energy_in_support_p95: -1,
      },
    },
  });
  assert.equal(result.method, "energy");
  assert.equal(typeof result.energy, "number");
  assert.equal(typeof result.s_ood, "number");
});
