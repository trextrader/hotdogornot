"""Crash-safety + keep-best: train() must write a rolling last checkpoint
and per-epoch metrics, and weights.pt must be the BEST val_acc epoch
(not the last)."""
import json
from pathlib import Path

from rfconnectorai.classifier.train import TrainConfig, train


def _tiny(root: Path, classes, n=4):
    from PIL import Image
    for ci, c in enumerate(classes):
        d = root / c
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (64, 64), (ci * 60 + i * 10, 0, 0)).save(d / f"{i}.jpg")


def test_checkpoint_and_keep_best(tmp_path):
    classes = ["a", "b"]
    data = tmp_path / "data"
    _tiny(data, classes)
    out = tmp_path / "out"

    cfg = TrainConfig(
        data_dir=data, out_dir=out, class_names=classes,
        epochs=3, batch_size=2, val_fraction=0.5,
        architecture="resnet18", input_size=64,
    )
    metrics = train(cfg)

    # rolling last + per-epoch metrics survive even if a run is killed
    assert (out / "weights_last.pt").exists()
    assert (out / "weights.pt").exists()
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["history"] == metrics["history"]
    assert len(saved["history"]) == 3

    # version.json records the BEST val_acc, not the last epoch's
    best = max(h["val_acc"] for h in metrics["history"])
    ver = json.loads((out / "version.json").read_text())
    assert ver["val_acc"] == best
    assert ver["val_acc"] >= metrics["history"][-1]["val_acc"]
