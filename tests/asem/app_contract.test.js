const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");

test("app.js preserves legacy prediction fields and adds asem_rev3 alongside", () => {
  const app = fs.readFileSync("exports/web/app.js", "utf8");
  for (const token of ["legacyOutput", "class:", "confidence:", "bbox:", "top_k:", "asem_rev3"]) {
    assert.equal(app.includes(token), true, `${token} missing from app.js`);
  }
});

test("app.js surfaces threshold load failures in the status badge", () => {
  const app = fs.readFileSync("exports/web/app.js", "utf8");
  assert.equal(app.includes("AsemThresholds.loadThresholds"), true);
  assert.equal(app.includes("modelStatus.textContent = e.message"), true);
});
