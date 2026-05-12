/**
 * Copy the web demo (HTML/CSS/JS + ONNX models) into the Capacitor www/ directory.
 * Run before `npx cap sync` to ensure native projects have the latest assets.
 */
const fs = require('fs');
const path = require('path');

const SRC = path.resolve(__dirname, '..', '..', 'web');
const DEST = path.resolve(__dirname, '..', 'www');

function copyRecursive(src, dest) {
  if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
      const sizeMB = (fs.statSync(destPath).size / 1024 / 1024).toFixed(1);
      console.log(`  ${entry.name} (${sizeMB} MB)`);
    }
  }
}

console.log(`Copying web assets: ${SRC} → ${DEST}`);
copyRecursive(SRC, DEST);
console.log('Done.');
