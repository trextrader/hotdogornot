(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.AsemQuality = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const DOMINANT_PRIORITY = ["blur", "glare", "oblique", "low_roi_scale"];

  function finite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`${name} must be a finite number`);
    }
    return value;
  }

  function clamp01(value) {
    finite(value, "value");
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
  }

  function normalizeBbox(bbox) {
    if (!bbox || typeof bbox !== "object") {
      throw new Error("bbox is required");
    }
    const x1 = finite(bbox.x1, "bbox.x1");
    const y1 = finite(bbox.y1, "bbox.y1");
    const x2 = finite(bbox.x2, "bbox.x2");
    const y2 = finite(bbox.y2, "bbox.y2");
    return {
      x1,
      y1,
      x2,
      y2,
      width: Math.max(0, x2 - x1),
      height: Math.max(0, y2 - y1),
    };
  }

  function imageDataShape(roiImageData) {
    if (!roiImageData || typeof roiImageData !== "object") {
      throw new Error("roiImageData is required");
    }
    const width = finite(roiImageData.width, "roiImageData.width");
    const height = finite(roiImageData.height, "roiImageData.height");
    if (width <= 0 || height <= 0) {
      throw new Error("roiImageData dimensions must be positive");
    }
    if (!roiImageData.data || roiImageData.data.length < width * height * 4) {
      throw new Error("roiImageData.data must contain RGBA pixels");
    }
    return { width, height, data: roiImageData.data };
  }

  function grayscaleAt(data, idx) {
    return 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }

  function varianceOfLaplacian(roiImageData) {
    const { width, height, data } = imageDataShape(roiImageData);
    if (width < 3 || height < 3) return 0;

    const values = [];
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const c = (y * width + x) * 4;
        const left = (y * width + x - 1) * 4;
        const right = (y * width + x + 1) * 4;
        const up = ((y - 1) * width + x) * 4;
        const down = ((y + 1) * width + x) * 4;
        const lap =
          -4 * grayscaleAt(data, c) +
          grayscaleAt(data, left) +
          grayscaleAt(data, right) +
          grayscaleAt(data, up) +
          grayscaleAt(data, down);
        values.push(lap);
      }
    }
    if (values.length === 0) return 0;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    return variance;
  }

  function glareFraction(roiImageData) {
    const { width, height, data } = imageDataShape(roiImageData);
    let glare = 0;
    const pixels = width * height;
    for (let i = 0; i < pixels; i += 1) {
      const offset = i * 4;
      const r = data[offset];
      const g = data[offset + 1];
      const b = data[offset + 2];
      if (r >= 245 && g >= 245 && b >= 245) {
        glare += 1;
      }
    }
    return pixels ? glare / pixels : 0;
  }

  function roiScale(frameWidth, frameHeight, bbox) {
    finite(frameWidth, "frameWidth");
    finite(frameHeight, "frameHeight");
    if (frameWidth <= 0 || frameHeight <= 0) {
      throw new Error("frame dimensions must be positive");
    }
    const area = Math.max(0, bbox.width * bbox.height);
    return area / (frameWidth * frameHeight);
  }

  function obliqueProxy(bbox) {
    if (bbox.width <= 0 || bbox.height <= 0) return 1;
    const minSide = Math.min(bbox.width, bbox.height);
    const maxSide = Math.max(bbox.width, bbox.height);
    return clamp01(1 - minSide / maxSide);
  }

  function centerResolution(frameWidth, frameHeight, bbox, thresholds) {
    const frameArea = frameWidth * frameHeight;
    const minAcceptableSide = Math.sqrt(frameArea * thresholds.q.roi_scale_min);
    if (minAcceptableSide <= 0) return 0;
    return clamp01(Math.min(bbox.width, bbox.height) / minAcceptableSide);
  }

  function pickDominant(failures) {
    for (const reason of DOMINANT_PRIORITY) {
      if (failures.includes(reason)) return reason;
    }
    return "none";
  }

  function validateThresholds(thresholds) {
    if (!thresholds || !thresholds.q) {
      throw new Error("thresholds.q is required");
    }
    for (const key of ["blur_var_min", "roi_scale_min", "glare_frac_max", "oblique_proxy_max", "center_res_min"]) {
      finite(thresholds.q[key], `thresholds.q.${key}`);
    }
  }

  function estimateQuality(input) {
    const cfg = input || {};
    validateThresholds(cfg.thresholds);
    const bbox = normalizeBbox(cfg.bbox);
    const frameWidth = finite(cfg.frameWidth, "frameWidth");
    const frameHeight = finite(cfg.frameHeight, "frameHeight");

    const blur_var = varianceOfLaplacian(cfg.roiImageData);
    const glare_frac = glareFraction(cfg.roiImageData);
    const scale = roiScale(frameWidth, frameHeight, bbox);
    const center_res = centerResolution(frameWidth, frameHeight, bbox, cfg.thresholds);
    const oblique_proxy = obliqueProxy(bbox);

    const failures = [];
    if (blur_var < cfg.thresholds.q.blur_var_min) failures.push("blur");
    if (glare_frac > cfg.thresholds.q.glare_frac_max) failures.push("glare");
    if (oblique_proxy > cfg.thresholds.q.oblique_proxy_max) failures.push("oblique");
    if (scale < cfg.thresholds.q.roi_scale_min) failures.push("low_roi_scale");

    const dominant = pickDominant(failures);
    return {
      blur_var,
      glare_frac,
      roi_scale: scale,
      center_res,
      oblique_proxy,
      q_low: dominant !== "none",
      dominant,
    };
  }

  return {
    estimateQuality,
    varianceOfLaplacian,
    glareFraction,
    roiScale,
    obliqueProxy,
    centerResolution,
  };
});
