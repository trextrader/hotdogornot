const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

function walkFiles(dir, predicate, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkFiles(full, predicate, out);
    } else if (predicate(full)) {
      out.push(full);
    }
  }
  return out;
}

test("Rev-3 web, canonical diagram, and tests contain no out-of-scope connector labels", () => {
  const forbidden = [
    ["RP", "SMA"].join("-"),
    ["right", "angle"].join("-"),
    "bulk" + "head",
    ["cable", "end"].join("-"),
    ["board", "mount"].join("-"),
    "generic" + " SMA",
  ];
  const files = [
    ...walkFiles("exports/web", (file) => /\.(html|css|js|json|dot)$/i.test(file)),
    "docs/fulldetector_active_visual_interrogation_system.dot",
    ...walkFiles("tests/asem", (file) => /\.test\.js$/i.test(file)),
  ];
  const failures = [];
  for (const file of files) {
    const text = fs.readFileSync(file, "utf8");
    for (const token of forbidden) {
      if (text.includes(token)) {
        failures.push(`${file}: ${token}`);
      }
    }
  }
  assert.deepEqual(failures, []);
});

test("canonical diagram contains ONNX Runtime Web terms and no native inference terms", () => {
  const dot = fs.readFileSync("docs/fulldetector_active_visual_interrogation_system.dot", "utf8");
  for (const required of ["ONNX Runtime Web", "WASM / WebGL", "*.onnx", "Stage-0 Decision Controller"]) {
    assert.equal(dot.includes(required), true, `${required} missing from canonical diagram`);
  }
  for (const token of ["TFLite", "NNAPI", ".tflite"]) {
    assert.equal(dot.includes(token), false, `${token} must not appear in canonical Rev-3 diagram`);
  }
});
