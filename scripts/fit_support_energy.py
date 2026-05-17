"""Fit Rev-3 logit-energy support calibration percentiles.

This script is designed for Colab/Kaggle execution against the curated in-support
image set. It runs the current ONNX classifier over the locked 10-class dataset,
computes logit energy:

    energy = -T * logsumexp(logits / T)

and writes p05/p95 into exports/web/thresholds.json when --write-thresholds is set.

Colab/Kaggle run recipe:

    git clone <repo-url> hotdogornot
    cd hotdogornot
    python -m pip install onnxruntime pillow numpy
    python scripts/fit_support_energy.py --write-thresholds

For GPU-backed ONNX Runtime in a T4 notebook, install onnxruntime-gpu and add:

    --providers CUDAExecutionProvider CPUExecutionProvider

Do not run the full dataset calibration on the local Windows PC unless the project
owner explicitly asks for local dataset-scale inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOCKED_CLASSES = [
    "1.85mm-M",
    "1.85mm-F",
    "2.4mm-M",
    "2.4mm-F",
    "2.92mm-M",
    "2.92mm-F",
    "3.5mm-M",
    "3.5mm-F",
    "SMA-F",
    "SMA-M",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Rev-3 in-support logit-energy percentiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/labeled/embedder"))
    parser.add_argument("--model", type=Path, default=Path("exports/web/models/classifier.onnx"))
    parser.add_argument("--labels", type=Path, default=Path("exports/web/models/classifier_labels.json"))
    parser.add_argument("--thresholds", type=Path, default=Path("exports/web/thresholds.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--report-name", default="asem_rev3_support_energy_report.json")
    parser.add_argument("--write-thresholds", action="store_true")
    parser.add_argument("--unsupported", type=float, default=None)
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def validate_labels(labels: dict) -> tuple[list[str], int]:
    class_names = labels.get("class_names")
    if class_names != LOCKED_CLASSES:
        raise ValueError(
            "classifier_labels.json class_names must match the locked Rev-3 10-class order"
        )
    input_size = int(labels.get("input_size", 384))
    if input_size <= 0:
        raise ValueError("classifier_labels.json input_size must be positive")
    return class_names, input_size


def iter_image_paths(dataset: Path, class_names: Iterable[str], limit_per_class: int | None) -> list[Path]:
    paths: list[Path] = []
    for class_name in class_names:
        class_dir = dataset / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        class_paths = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit_per_class is not None:
            class_paths = class_paths[:limit_per_class]
        paths.extend(class_paths)
    return paths


def preprocess_image(path: Path, input_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + math.log(float(np.sum(np.exp(values - max_value))))


def energy_from_logits(logits: np.ndarray, temperature: float) -> float:
    if not np.isfinite(logits).all():
        raise ValueError("Classifier emitted non-finite logits")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("calibration_T must be finite and > 0")
    scaled = logits.astype(np.float64) / temperature
    return -temperature * logsumexp(scaled)


def normalize_energy(energy: float, p05: float, p95: float) -> float:
    if not p05 < p95:
        raise ValueError("p05 must be less than p95")
    return max(0.0, min(1.0, (energy - p05) / (p95 - p05)))


def create_session(model_path: Path, providers: list[str]):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required. In Colab/Kaggle run: python -m pip install onnxruntime pillow numpy"
        ) from exc

    available = set(ort.get_available_providers())
    selected = [p for p in providers if p in available]
    if not selected:
        raise RuntimeError(f"None of the requested providers are available: {providers}; available={sorted(available)}")
    return ort.InferenceSession(str(model_path), providers=selected), selected


def summarize(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def main() -> int:
    args = parse_args()
    labels = load_json(args.labels)
    class_names, input_size = validate_labels(labels)
    thresholds = load_json(args.thresholds)
    temperature = float(thresholds["calibration_T"])

    image_paths = iter_image_paths(args.dataset, class_names, args.limit_per_class)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No calibration images found under {args.dataset}")

    session, selected_providers = create_session(args.model, args.providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    records = []
    for path in image_paths:
        tensor = preprocess_image(path, input_size)
        output = session.run([output_name], {input_name: tensor})[0]
        logits = np.asarray(output, dtype=np.float64).reshape(-1)
        if logits.size != len(class_names):
            raise ValueError(f"Expected {len(class_names)} logits, got {logits.size} from {path}")
        energy = energy_from_logits(logits, temperature)
        records.append({"path": path.as_posix(), "energy": energy})

    energies = np.asarray([r["energy"] for r in records], dtype=np.float64)
    stats = summarize(energies)
    p05 = stats["p05"]
    p95 = stats["p95"]
    s_ood = np.asarray([normalize_energy(v, p05, p95) for v in energies], dtype=np.float64)
    unsupported = args.unsupported
    if unsupported is None:
        unsupported = float(thresholds.get("unsupported", 0.6))

    report = {
        "schema_version": "asem_rev3_support_energy_report_v1",
        "model": args.model.as_posix(),
        "dataset": args.dataset.as_posix(),
        "labels": args.labels.as_posix(),
        "thresholds": args.thresholds.as_posix(),
        "providers": selected_providers,
        "input_size": input_size,
        "calibration_T": temperature,
        "image_count": len(image_paths),
        "class_order": class_names,
        "energy": stats,
        "chosen_percentiles": {
            "energy_in_support_p05": p05,
            "energy_in_support_p95": p95,
        },
        "normalized_s_ood": {
            "min": float(np.min(s_ood)),
            "median": float(np.percentile(s_ood, 50)),
            "max": float(np.max(s_ood)),
            "unsupported_threshold": unsupported,
            "curated_fraction_at_or_above_unsupported": float(np.mean(s_ood >= unsupported)),
        },
        "notes": [
            "s_ood is unsupported risk: larger means more likely out of support.",
            "Percentiles are measured on the locked in-support curated set only.",
            "Field validation must ratify the unsupported threshold on real unsupported/stress captures.",
        ],
    }

    report_path = args.report_dir / args.report_name
    write_json(report_path, report)

    if args.write_thresholds:
        thresholds["support_calibration"] = {
            "method": "energy_minmax",
            "energy_in_support_p05": p05,
            "energy_in_support_p95": p95,
        }
        thresholds["unsupported"] = unsupported
        write_json(args.thresholds, thresholds)

    print(f"Wrote report: {report_path}")
    if args.write_thresholds:
        print(f"Updated thresholds: {args.thresholds}")
    else:
        print("Thresholds unchanged. Re-run with --write-thresholds to commit measured percentiles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
