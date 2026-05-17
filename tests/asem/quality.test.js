const test = require("node:test");
const assert = require("node:assert/strict");

const {
  estimateQuality,
} = require("../../exports/web/asem/quality.js");

const thresholds = {
  q: {
    blur_var_min: 60,
    roi_scale_min: 0.08,
    glare_frac_max: 0.2,
    oblique_proxy_max: 0.65,
    center_res_min: 0.85,
  },
};

function imageData(width, height, pixelAt) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const [r, g, b, a = 255] = pixelAt(x, y);
      const offset = (y * width + x) * 4;
      data[offset] = r;
      data[offset + 1] = g;
      data[offset + 2] = b;
      data[offset + 3] = a;
    }
  }
  return { width, height, data };
}

function sharpRoi(width = 24, height = 24) {
  return imageData(width, height, (x, y) => {
    const value = (x + y) % 2 === 0 ? 80 : 180;
    return [value, value, value];
  });
}

function blurredRoi(width = 24, height = 24) {
  return imageData(width, height, () => [128, 128, 128]);
}

function glareRoi(width = 24, height = 24) {
  return imageData(width, height, (x, y) => {
    const value = (x + y) % 2 === 0 ? 255 : 0;
    return [value, value, value];
  });
}

function qualityFor(overrides = {}) {
  return estimateQuality({
    frameWidth: 100,
    frameHeight: 100,
    bbox: { x1: 20, y1: 20, x2: 70, y2: 70 },
    roiImageData: sharpRoi(),
    thresholds,
    ...overrides,
  });
}

test("estimateQuality returns the full Q_t struct for an acceptable ROI", () => {
  const q = qualityFor();
  assert.equal(typeof q.blur_var, "number");
  assert.equal(typeof q.glare_frac, "number");
  assert.equal(typeof q.roi_scale, "number");
  assert.equal(typeof q.center_res, "number");
  assert.equal(typeof q.oblique_proxy, "number");
  assert.equal(q.q_low, false);
  assert.equal(q.dominant, "none");
});

test("blur trips q_low with dominant blur", () => {
  const q = qualityFor({ roiImageData: blurredRoi() });
  assert.equal(q.q_low, true);
  assert.equal(q.dominant, "blur");
  assert.ok(q.blur_var < thresholds.q.blur_var_min);
});

test("glare trips q_low with dominant glare when blur is acceptable", () => {
  const q = qualityFor({ roiImageData: glareRoi() });
  assert.equal(q.q_low, true);
  assert.equal(q.dominant, "glare");
  assert.ok(q.glare_frac > thresholds.q.glare_frac_max);
  assert.ok(q.blur_var >= thresholds.q.blur_var_min);
});

test("small ROI trips q_low with dominant low_roi_scale", () => {
  const q = qualityFor({
    bbox: { x1: 10, y1: 10, x2: 20, y2: 20 },
  });
  assert.equal(q.q_low, true);
  assert.equal(q.dominant, "low_roi_scale");
  assert.ok(q.roi_scale < thresholds.q.roi_scale_min);
});

test("oblique geometry trips q_low with dominant oblique", () => {
  const q = qualityFor({
    bbox: { x1: 10, y1: 0, x2: 30, y2: 100 },
  });
  assert.equal(q.q_low, true);
  assert.equal(q.dominant, "oblique");
  assert.ok(q.oblique_proxy > thresholds.q.oblique_proxy_max);
});

test("low center_res is reported but does not drive q_low", () => {
  const q = qualityFor({
    bbox: { x1: 10, y1: 10, x2: 30, y2: 60 },
  });
  assert.ok(q.center_res < thresholds.q.center_res_min);
  assert.equal(q.roi_scale >= thresholds.q.roi_scale_min, true);
  assert.equal(q.oblique_proxy <= thresholds.q.oblique_proxy_max, true);
  assert.equal(q.q_low, false);
  assert.equal(q.dominant, "none");
});

test("dominant q_low priority is blur before glare, oblique, and low_roi_scale", () => {
  const q = qualityFor({
    bbox: { x1: 10, y1: 10, x2: 20, y2: 70 },
    roiImageData: imageData(24, 24, () => [255, 255, 255]),
  });
  assert.equal(q.q_low, true);
  assert.equal(q.dominant, "blur");
  assert.ok(q.glare_frac > thresholds.q.glare_frac_max);
  assert.ok(q.oblique_proxy > thresholds.q.oblique_proxy_max);
  assert.ok(q.roi_scale < thresholds.q.roi_scale_min);
});
