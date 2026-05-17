(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.AsemDecision = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const GUIDANCE = {
    accepted: "Connector accepted.",
    no_connector_found: "No connector was found clearly enough. Move closer and center the connector.",
    need_better_focus: "Move closer, improve focus, reduce glare, and show the connector center clearly.",
    need_second_angle: "Capture a second angle with the connector face more visible.",
    unsupported_connector: "This connector is outside the supported Rev-3 class set.",
    ambiguous: "The evidence is not strong enough to identify this connector safely.",
  };

  function finite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`${name} must be a finite number`);
    }
    return value;
  }

  function validateThresholds(thresholds) {
    if (!thresholds || typeof thresholds !== "object") {
      throw new Error("thresholds are required");
    }
    for (const key of ["box_min", "accept", "margin", "unsupported"]) {
      finite(thresholds[key], `thresholds.${key}`);
    }
    if (!thresholds.q || typeof thresholds.q !== "object") {
      throw new Error("thresholds.q is required");
    }
    finite(thresholds.q.center_res_min, "thresholds.q.center_res_min");
  }

  function validateQuality(quality) {
    if (!quality || typeof quality !== "object") {
      throw new Error("quality is required");
    }
    if (typeof quality.q_low !== "boolean") {
      throw new Error("quality.q_low must be boolean");
    }
    if (typeof quality.dominant !== "string") {
      throw new Error("quality.dominant must be a string");
    }
    finite(quality.center_res, "quality.center_res");
  }

  function validateProbabilities(probabilities, labels) {
    if (!Array.isArray(probabilities) && !(probabilities instanceof Float32Array) && !(probabilities instanceof Float64Array)) {
      throw new Error("probabilities must be an array");
    }
    if (!Array.isArray(labels)) {
      throw new Error("labels must be an array");
    }
    if (probabilities.length === 0) {
      throw new Error("probabilities must not be empty");
    }
    if (probabilities.length !== labels.length) {
      throw new Error("probabilities and labels must have the same length");
    }
    for (let i = 0; i < probabilities.length; i += 1) {
      finite(probabilities[i], `probabilities[${i}]`);
      if (typeof labels[i] !== "string" || labels[i].length === 0) {
        throw new Error(`labels[${i}] must be a non-empty string`);
      }
    }
  }

  function validateDecisionInputs(box_conf, quality, probabilities, labels, s_ood, thresholds) {
    finite(box_conf, "box_conf");
    finite(s_ood, "s_ood");
    validateQuality(quality);
    validateProbabilities(probabilities, labels);
    validateThresholds(thresholds);
  }

  function topK(probabilities, labels, k = 5) {
    validateProbabilities(probabilities, labels);
    const limit = Math.max(0, Math.min(k, probabilities.length));
    return Array.from(probabilities, (prob, index) => ({
      index,
      class_name: labels[index],
      prob,
    }))
      .sort((a, b) => {
        if (b.prob !== a.prob) return b.prob - a.prob;
        return a.index - b.index;
      })
      .slice(0, limit);
  }

  function top2(probabilities, labels) {
    if (probabilities.length < 2) {
      throw new Error("top2 requires at least two probabilities");
    }
    return topK(probabilities, labels, 2);
  }

  function requiredEvidenceVisible(_className, quality, thresholds) {
    validateQuality(quality);
    validateThresholds(thresholds);
    return quality.center_res >= thresholds.q.center_res_min;
  }

  function qualityReason(quality) {
    if (["blur", "glare", "low_roi_scale", "low_center_res"].includes(quality.dominant)) {
      return "need_better_focus";
    }
    return "need_second_angle";
  }

  function abstain(reason, payload) {
    return {
      status: "ABSTAIN",
      reason,
      user_guidance: GUIDANCE[reason],
      ...payload,
    };
  }

  function accept(c1, c2, margin, payload) {
    return {
      status: "ACCEPT",
      reason: "accepted",
      user_guidance: GUIDANCE.accepted,
      class_name: c1.class_name,
      confidence: c1.prob,
      top1: c1,
      top2: c2,
      margin,
      ...payload,
    };
  }

  function withTrace(result, traceBase) {
    return {
      ...result,
      trace: {
        ...(traceBase || {}),
        decision: {
          status: result.status,
          reason: result.reason,
          user_guidance: result.user_guidance,
        },
      },
    };
  }

  function decide(input) {
    const cfg = input || {};
    const box_conf = cfg.box_conf;
    const quality = cfg.quality;
    const probabilities = cfg.probabilities;
    const labels = cfg.labels;
    const s_ood = cfg.s_ood;
    const thresholds = cfg.thresholds;
    const traceBase = cfg.traceBase || {};

    validateDecisionInputs(box_conf, quality, probabilities, labels, s_ood, thresholds);

    if (box_conf < thresholds.box_min) {
      return withTrace(abstain("no_connector_found", { alternatives: [] }), traceBase);
    }

    if (quality.q_low) {
      return withTrace(abstain(qualityReason(quality), { alternatives: [] }), traceBase);
    }

    if (s_ood >= thresholds.unsupported) {
      return withTrace(abstain("unsupported_connector", { alternatives: [] }), traceBase);
    }

    const [c1, c2] = top2(probabilities, labels);
    const margin = c1.prob - c2.prob;

    if (!requiredEvidenceVisible(c1.class_name, quality, thresholds)) {
      return withTrace(
        abstain("need_better_focus", {
          top1: c1,
          top2: c2,
          margin,
          alternatives: topK(probabilities, labels, 5),
        }),
        traceBase
      );
    }

    if (c1.prob >= thresholds.accept && margin >= thresholds.margin) {
      return withTrace(accept(c1, c2, margin, { alternatives: topK(probabilities, labels, 5) }), traceBase);
    }

    return withTrace(
      abstain("ambiguous", {
        top1: c1,
        top2: c2,
        margin,
        alternatives: topK(probabilities, labels, 5),
      }),
      traceBase
    );
  }

  return {
    GUIDANCE,
    topK,
    top2,
    requiredEvidenceVisible,
    decide,
  };
});
