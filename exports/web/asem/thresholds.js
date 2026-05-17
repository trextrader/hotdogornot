(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.AsemThresholds = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const REQUIRED_NUMERIC = [
    "box_min",
    "accept",
    "margin",
    "unsupported",
    "hardcase_accept_sample_rate",
    "calibration_T",
  ];

  const REQUIRED_Q_NUMERIC = [
    "blur_var_min",
    "roi_scale_min",
    "glare_frac_max",
    "oblique_proxy_max",
    "center_res_min",
  ];

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function assertFiniteNumber(obj, key, path) {
    if (!isFiniteNumber(obj[key])) {
      throw new Error(`${path}.${key} must be a finite number`);
    }
  }

  function assertUnitInterval(obj, key, path) {
    assertFiniteNumber(obj, key, path);
    if (obj[key] < 0 || obj[key] > 1) {
      throw new Error(`${path}.${key} must be in [0,1]`);
    }
  }

  function validateThresholds(thresholds) {
    if (!thresholds || typeof thresholds !== "object" || Array.isArray(thresholds)) {
      throw new Error("thresholds.json must be a JSON object");
    }
    if (thresholds.schema_version !== "asem_rev3_thresholds_v1") {
      throw new Error("thresholds.json schema_version must be asem_rev3_thresholds_v1");
    }
    if (typeof thresholds.thresholds_version !== "string" || thresholds.thresholds_version.length === 0) {
      throw new Error("thresholds.json thresholds_version must be a non-empty string");
    }

    for (const key of REQUIRED_NUMERIC) {
      assertFiniteNumber(thresholds, key, "thresholds");
    }
    assertUnitInterval(thresholds, "box_min", "thresholds");
    assertUnitInterval(thresholds, "accept", "thresholds");
    assertUnitInterval(thresholds, "margin", "thresholds");
    assertUnitInterval(thresholds, "unsupported", "thresholds");
    assertUnitInterval(thresholds, "hardcase_accept_sample_rate", "thresholds");
    if (thresholds.calibration_T <= 0) {
      throw new Error("thresholds.calibration_T must be > 0");
    }

    if (!thresholds.q || typeof thresholds.q !== "object" || Array.isArray(thresholds.q)) {
      throw new Error("thresholds.q must be an object");
    }
    for (const key of REQUIRED_Q_NUMERIC) {
      assertFiniteNumber(thresholds.q, key, "thresholds.q");
    }

    const calibration = thresholds.support_calibration;
    if (!calibration || typeof calibration !== "object" || Array.isArray(calibration)) {
      throw new Error("thresholds.support_calibration must be an object");
    }
    if (calibration.method !== "energy_minmax") {
      throw new Error("thresholds.support_calibration.method must be energy_minmax");
    }
    const p05 = calibration.energy_in_support_p05;
    const p95 = calibration.energy_in_support_p95;
    const p05Set = p05 !== null && p05 !== undefined;
    const p95Set = p95 !== null && p95 !== undefined;
    if (p05Set && !isFiniteNumber(p05)) {
      throw new Error("thresholds.support_calibration.energy_in_support_p05 must be finite or null");
    }
    if (p95Set && !isFiniteNumber(p95)) {
      throw new Error("thresholds.support_calibration.energy_in_support_p95 must be finite or null");
    }
    if (p05Set !== p95Set) {
      throw new Error("support calibration percentiles must be both null or both finite");
    }
    if (p05Set && !(p05 < p95)) {
      throw new Error("support calibration requires energy_in_support_p05 < energy_in_support_p95");
    }

    return thresholds;
  }

  async function loadThresholds(url, fetchImpl) {
    const fetcher = fetchImpl || (typeof fetch !== "undefined" ? fetch : null);
    if (!fetcher) {
      throw new Error("No fetch implementation available for thresholds.json");
    }
    let response;
    try {
      response = await fetcher(url, { cache: "no-store" });
    } catch (err) {
      throw new Error(`Failed to load thresholds.json: ${err.message}`);
    }
    if (!response || !response.ok) {
      const status = response ? `${response.status} ${response.statusText || ""}`.trim() : "no response";
      throw new Error(`Failed to load thresholds.json: ${status}`);
    }
    let parsed;
    try {
      parsed = await response.json();
    } catch (err) {
      throw new Error(`Malformed thresholds.json: ${err.message}`);
    }
    try {
      return validateThresholds(parsed);
    } catch (err) {
      throw new Error(`Invalid thresholds.json: ${err.message}`);
    }
  }

  return {
    validateThresholds,
    loadThresholds,
  };
});
