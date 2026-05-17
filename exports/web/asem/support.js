(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.AsemSupport = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function assertFiniteNumber(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`${name} must be a finite number`);
    }
  }

  function assertPositiveTemperature(T) {
    assertFiniteNumber(T, "calibration_T");
    if (T <= 0) {
      throw new Error("calibration_T must be > 0");
    }
  }

  function asFiniteArray(values, name) {
    if (!Array.isArray(values) && !(values instanceof Float32Array) && !(values instanceof Float64Array)) {
      throw new Error(`${name} must be an array of finite numbers`);
    }
    const arr = Array.from(values);
    if (arr.length === 0) {
      throw new Error(`${name} must not be empty`);
    }
    for (let i = 0; i < arr.length; i += 1) {
      assertFiniteNumber(arr[i], `${name}[${i}]`);
    }
    return arr;
  }

  function clamp01(value) {
    assertFiniteNumber(value, "value");
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
  }

  function logsumexp(values) {
    const arr = asFiniteArray(values, "values");
    const max = Math.max(...arr);
    let sum = 0;
    for (const value of arr) {
      sum += Math.exp(value - max);
    }
    return max + Math.log(sum);
  }

  function softmaxWithTemperature(logits, T) {
    assertPositiveTemperature(T);
    const arr = asFiniteArray(logits, "logits");
    const scaled = arr.map((value) => value / T);
    const lse = logsumexp(scaled);
    return scaled.map((value) => Math.exp(value - lse));
  }

  function energyScoreFromLogits(logits, T) {
    assertPositiveTemperature(T);
    const arr = asFiniteArray(logits, "logits");
    return -T * logsumexp(arr.map((value) => value / T));
  }

  function normalizeEnergyToSOod(energy, calibration) {
    assertFiniteNumber(energy, "energy");
    if (!calibration || typeof calibration !== "object") {
      throw new Error("support calibration is required");
    }
    if (calibration.method !== "energy_minmax") {
      throw new Error("support calibration method must be energy_minmax");
    }
    const p05 = calibration.energy_in_support_p05;
    const p95 = calibration.energy_in_support_p95;
    assertFiniteNumber(p05, "energy_in_support_p05");
    assertFiniteNumber(p95, "energy_in_support_p95");
    if (!(p05 < p95)) {
      throw new Error("support calibration requires energy_in_support_p05 < energy_in_support_p95");
    }
    return clamp01((energy - p05) / (p95 - p05));
  }

  function sOodFromInSupportProbability(pInSupport) {
    assertFiniteNumber(pInSupport, "p_in_support");
    if (pInSupport < 0 || pInSupport > 1) {
      throw new Error("p_in_support must be in [0,1]");
    }
    return 1 - pInSupport;
  }

  function computeSupportScore(input) {
    const cfg = input || {};
    const thresholds = cfg.thresholds || {};
    const calibration = cfg.calibration || thresholds.support_calibration;
    const T = cfg.T || thresholds.calibration_T;

    if (cfg.logits !== undefined && cfg.logits !== null) {
      const energy = energyScoreFromLogits(cfg.logits, T);
      return {
        s_ood: normalizeEnergyToSOod(energy, calibration),
        method: "energy",
        energy,
      };
    }

    if (cfg.p_in_support !== undefined && cfg.p_in_support !== null) {
      return {
        s_ood: sOodFromInSupportProbability(cfg.p_in_support),
        method: "p_in_support",
        energy: null,
      };
    }

    throw new Error("computeSupportScore requires logits for Rev-3 logit-energy scoring");
  }

  return {
    clamp01,
    logsumexp,
    softmaxWithTemperature,
    energyScoreFromLogits,
    normalizeEnergyToSOod,
    sOodFromInSupportProbability,
    computeSupportScore,
  };
});
