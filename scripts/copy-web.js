/**
 * Copy the canonical web bundle into the Capacitor app.
 *
 * Source of truth:
 *   exports/web/
 *
 * Mobile destination:
 *   exports/mobile/www/
 *
 * This recursive copy intentionally covers Rev-3 assets such as:
 *   exports/web/thresholds.json
 *   exports/web/asem/**
 */
const fs = require("fs");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const srcRoot = path.join(repoRoot, "exports", "web");
const destRoot = path.join(repoRoot, "exports", "mobile", "www");

function assertInside(child, parent) {
  const rel = path.relative(parent, child);
  if (rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel))) return;
  throw new Error(`Refusing to copy outside ${parent}: ${child}`);
}

function copyRecursive(src, dest) {
  assertInside(src, srcRoot);
  assertInside(dest, destRoot);

  if (!fs.existsSync(src)) {
    throw new Error(`Source does not exist: ${src}`);
  }
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyRecursive(srcPath, destPath);
      continue;
    }
    fs.copyFileSync(srcPath, destPath);
    const sizeMB = (fs.statSync(destPath).size / 1024 / 1024).toFixed(1);
    const rel = path.relative(srcRoot, srcPath);
    console.log(`  ${rel} (${sizeMB} MB)`);
  }
}

console.log(`Copying web assets: ${srcRoot} -> ${destRoot}`);
copyRecursive(srcRoot, destRoot);
console.log("Done.");
