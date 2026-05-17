"""Crop embedder images to the detected connector (remove hand/background).

Runs the YOLO connector detector over data/labeled/embedder/<class>/*, takes the
best box, pads it, and writes the crop to data/labeled/embedder_cropped/<class>/
with the SAME filename. Originals are never modified. Images the detector cannot
box (>= CONF) are passed through uncropped and logged so they can be reviewed.

Letterbox preprocessing + inverse mapping mirror exports/web/app.js so crops are
geometrically correct.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

SRC = Path("data/labeled/embedder")
DST = Path("data/labeled/embedder_cropped")
MODEL = Path("exports/web/models/detector.onnx")
MANIFEST = Path("reports/label_audit/crop_manifest.csv")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMGSZ = 640
CONF = 0.25          # matches thresholds.json box_min
PAD_FRAC = 0.15      # expand box 15% each side so body/threads aren't clipped
LOW_CONF = 0.40      # flag for review below this


def letterbox(im: Image.Image):
    w, h = im.size
    s = min(IMGSZ / w, IMGSZ / h)
    nw, nh = round(w * s), round(h * s)
    dx, dy = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.Resampling.BILINEAR), (dx, dy))
    arr = np.asarray(canvas, np.float32) / 255.0
    return np.expand_dims(arr.transpose(2, 0, 1), 0), s, dx, dy


def best_box(out: np.ndarray, s: float, dx: int, dy: int, W: int, H: int):
    # out: [1, 4+nc, 8400] -> [8400, 4+nc]
    p = out[0].T
    xywh = p[:, :4]
    score = p[:, 4:].max(axis=1)
    i = int(score.argmax())
    if score[i] < CONF:
        return None, float(score[i])
    cx, cy, bw, bh = xywh[i]
    x1 = (cx - bw / 2 - dx) / s
    y1 = (cy - bh / 2 - dy) / s
    x2 = (cx + bw / 2 - dx) / s
    y2 = (cy + bh / 2 - dy) / s
    px, py = (x2 - x1) * PAD_FRAC, (y2 - y1) * PAD_FRAC
    x1 = max(0, int(round(x1 - px)))
    y1 = max(0, int(round(y1 - py)))
    x2 = min(W, int(round(x2 + px)))
    y2 = min(H, int(round(y2 + py)))
    if x2 <= x1 or y2 <= y1:
        return None, float(score[i])
    return (x1, y1, x2, y2), float(score[i])


def main() -> int:
    if not MODEL.exists():
        print(f"missing {MODEL}", file=sys.stderr)
        return 1
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    rows = [["path", "class", "detected", "score", "box", "action"]]
    n_crop = n_pass = 0

    folders = sorted(p for p in SRC.iterdir() if p.is_dir())
    for folder in folders:
        out_dir = DST / folder.name
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
        for ip in imgs:
            try:
                im = ImageOps.exif_transpose(Image.open(ip)).convert("RGB")
            except Exception as e:  # noqa: BLE001
                rows.append([ip.as_posix(), folder.name, "ERR", "", "", str(e)])
                continue
            W, H = im.size
            blob, s, dx, dy = letterbox(im)
            out = sess.run(None, {iname: blob})[0]
            box, score = best_box(out, s, dx, dy, W, H)
            dst = out_dir / ip.name
            if box is None:
                im.save(dst)  # passthrough uncropped
                n_pass += 1
                rows.append([ip.as_posix(), folder.name, "no", f"{score:.3f}",
                             "", "passthrough"])
            else:
                im.crop(box).save(dst)
                n_crop += 1
                flag = "cropped_lowconf" if score < LOW_CONF else "cropped"
                rows.append([ip.as_posix(), folder.name, "yes", f"{score:.3f}",
                             "|".join(map(str, box)), flag])
        print(f"  {folder.name}: {len(imgs)} imgs")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    low = sum(1 for r in rows[1:] if r[5] == "cropped_lowconf")
    print(f"Done. cropped={n_crop} (low-conf={low}) passthrough={n_pass} "
          f"-> {DST}/  manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
