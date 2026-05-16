"""Phase 2 warm-start: train() must load init weights when given."""
import json
from pathlib import Path

import torch

from rfconnectorai.classifier.train import TrainConfig, build_model, train


def _make_tiny_dataset(root: Path, classes: list[str], n: int = 3):
    from PIL import Image
    for c in classes:
        d = root / c
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (64, 64), (i * 30, 0, 0)).save(d / f"{i}.jpg")


def test_finetune_loads_init_weights(tmp_path):
    classes = ["a", "b"]
    data = tmp_path / "data"
    _make_tiny_dataset(data, classes)

    init = build_model(num_classes=2, architecture="resnet18")
    init_path = tmp_path / "phase1.pt"
    torch.save(init.state_dict(), init_path)

    cfg = TrainConfig(
        data_dir=data, out_dir=tmp_path / "out", class_names=classes,
        epochs=1, batch_size=2, val_fraction=0.5,
        architecture="resnet18", input_size=64,
        init_weights=init_path, learning_rate=1e-5,
    )
    metrics = train(cfg)
    assert "history" in metrics
    assert (tmp_path / "out" / "weights.pt").exists()
