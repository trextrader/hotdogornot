"""Per-folder contact sheets for the embedder M/F label audit.

Renders one montage PNG per class folder under data/labeled/embedder so a
domain expert can give a single verdict per folder (correct / wrong-gender /
mixed) instead of clicking 600+ images. Cells are index-numbered so a "mixed"
folder can be corrected per-image by index.

Pure PIL thumbnailing — no model inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

SRC = Path("data/labeled/embedder")
OUT = Path("reports/label_audit")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
THUMB = 220
COLS = 8
PAD = 6
HEADER = 64
LABEL_H = 18


def font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(folder: Path) -> Path | None:
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not imgs:
        print(f"  (skip, empty) {folder.name}")
        return None

    rows = (len(imgs) + COLS - 1) // COLS
    cell = THUMB + PAD
    W = COLS * cell + PAD
    H = HEADER + rows * (cell + LABEL_H) + PAD

    sheet = Image.new("RGB", (W, H), (18, 22, 33))
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 14), f"{folder.name}    n={len(imgs)}",
              fill=(234, 242, 255), font=font(34))

    for i, ip in enumerate(imgs):
        r, c = divmod(i, COLS)
        x = PAD + c * cell
        y = HEADER + r * (cell + LABEL_H)
        try:
            im = ImageOps.exif_transpose(Image.open(ip)).convert("RGB")
            im.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
            sheet.paste(im, (x + (THUMB - im.width) // 2,
                             y + (THUMB - im.height) // 2))
        except Exception as e:  # noqa: BLE001
            draw.rectangle([x, y, x + THUMB, y + THUMB], outline=(248, 113, 113))
            draw.text((x + 4, y + 4), f"ERR {e}", fill=(248, 113, 113), font=font(12))
        draw.text((x + 2, y + THUMB + 2), f"{i}: {ip.name[:26]}",
                  fill=(168, 199, 250), font=font(12))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{folder.name}.png"
    sheet.save(out)
    print(f"  {out}  ({len(imgs)} imgs, {rows} rows)")
    return out


def main() -> int:
    if not SRC.is_dir():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    folders = sorted(p for p in SRC.iterdir() if p.is_dir())
    print(f"Building contact sheets for {len(folders)} folders -> {OUT}/")
    for f in folders:
        contact_sheet(f)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
