# Fine-grained 9-class connector training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the training pipeline classify exactly the 9 customer connector classes with microscopic accuracy via a two-stage pipeline (single-class YOLO localizer → flat 9-class EfficientNetV2-S classifier, staged combined→curated training), exported to ONNX.

**Architecture:** `configs/classes.yaml` is the single source of truth for the 9 classes. A drift-guard test keeps `classes.yaml`, the labeler `CANONICAL_CLASSES`, and the classifier class list in lockstep. The detector dataset build gains a single-class mode so YOLO never collapses families. The flat classifier (`rfconnectorai.classifier`) gains an EfficientNetV2-S backbone, configurable input size, and a staged fine-tune (Phase 1 combined data → Phase 2 curated-only at low LR). A 9-class evaluation module reports per-class + adjacent-pair + M/F confusion. The Colab notebook is rewired to these.

**Tech Stack:** Python 3.9, PyTorch + torchvision, Ultralytics YOLO, pytest, Graphviz (docs only), Colab.

**Spec:** `docs/superpowers/specs/2026-05-15-fine-grained-9-class-connector-training-design.md`

**Conventions:**
- All commands run from the `training/` directory unless stated otherwise.
- Test runner: `pytest` (testpaths=`tests`, configured in `pyproject.toml`).
- Canonical 9-class order (used everywhere — ids 0..8):
  `1.85mm-M, 1.85mm-F, 2.4mm-M, 2.4mm-F, 2.92mm-M, 2.92mm-F, 3.5mm-M, 3.5mm-F, SMA-F`
- Commit after every task. Branch: work on `master` (already the working branch).

---

### Task 1: Reconcile `configs/classes.yaml` to the 9 customer classes

**Files:**
- Modify: `training/configs/classes.yaml`
- Test: `training/tests/test_classes.py`

- [ ] **Step 1: Update the failing test to expect 9 classes**

Replace the bodies of the three count/name tests in `training/tests/test_classes.py`:

```python
def test_load_classes_returns_nine():
    classes = load_classes(CONFIG)
    assert len(classes) == 9


def test_load_classes_ids_are_contiguous():
    classes = load_classes(CONFIG)
    assert [c.id for c in classes] == list(range(9))


def test_load_classes_names_match_spec():
    classes = load_classes(CONFIG)
    names = {c.name for c in classes}
    assert names == {
        "1.85mm-M", "1.85mm-F",
        "2.4mm-M", "2.4mm-F",
        "2.92mm-M", "2.92mm-F",
        "3.5mm-M", "3.5mm-F",
        "SMA-F",
    }
```

Also replace `test_precision_classes_flagged_correctly` body:

```python
def test_precision_classes_flagged_correctly():
    classes = load_classes(CONFIG)
    families = {c.name: c.family for c in classes}
    assert families["SMA-F"] == "sma"
    for name in [
        "1.85mm-M", "1.85mm-F", "2.4mm-M", "2.4mm-F",
        "2.92mm-M", "2.92mm-F", "3.5mm-M", "3.5mm-F",
    ]:
        assert families[name] == "precision"
```

Delete the old `test_load_classes_returns_eight` if a separate function still exists (it was renamed above). Keep `test_connector_class_is_frozen` unchanged.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_classes.py -v`
Expected: FAIL (current `classes.yaml` has 8 classes incl. SMA-M, no 1.85mm).

- [ ] **Step 3: Rewrite `configs/classes.yaml` to the 9 classes**

Replace the entire `classes:` list (keep the comment header) with:

```yaml
classes:
  - id: 0
    name: "1.85mm-M"
    family: "precision"
    gender: "male"
    inner_pin_diameter_mm: 0.51
    frequency_ghz_max: 67.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 1
    name: "1.85mm-F"
    family: "precision"
    gender: "female"
    inner_pin_diameter_mm: 0.51
    frequency_ghz_max: 67.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 2
    name: "2.4mm-M"
    family: "precision"
    gender: "male"
    inner_pin_diameter_mm: 1.04
    frequency_ghz_max: 50.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 3
    name: "2.4mm-F"
    family: "precision"
    gender: "female"
    inner_pin_diameter_mm: 1.04
    frequency_ghz_max: 50.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 4
    name: "2.92mm-M"
    family: "precision"
    gender: "male"
    inner_pin_diameter_mm: 1.27
    frequency_ghz_max: 40.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 5
    name: "2.92mm-F"
    family: "precision"
    gender: "female"
    inner_pin_diameter_mm: 1.27
    frequency_ghz_max: 40.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 6
    name: "3.5mm-M"
    family: "precision"
    gender: "male"
    inner_pin_diameter_mm: 1.52
    frequency_ghz_max: 34.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 7
    name: "3.5mm-F"
    family: "precision"
    gender: "female"
    inner_pin_diameter_mm: 1.52
    frequency_ghz_max: 34.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0

  - id: 8
    name: "SMA-F"
    family: "sma"
    gender: "female"
    inner_pin_diameter_mm: 1.27
    frequency_ghz_max: 18.0
    impedance_ohms: 50
    mating_torque_in_lb: 8.0
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_classes.py -v`
Expected: PASS (all class tests green).

- [ ] **Step 5: Commit**

```bash
git add training/configs/classes.yaml training/tests/test_classes.py
git commit -m "feat: reconcile classes.yaml to the 9 customer connector classes"
```

---

### Task 2: Add `class_names` helper to `classes.py` (single source for the classifier)

**Files:**
- Modify: `training/rfconnectorai/data/classes.py`
- Test: `training/tests/test_classes.py`

- [ ] **Step 1: Write the failing test**

Append to `training/tests/test_classes.py`:

```python
def test_class_names_returns_ordered_nine():
    from rfconnectorai.data.classes import class_names
    names = class_names(CONFIG)
    assert names == [
        "1.85mm-M", "1.85mm-F",
        "2.4mm-M", "2.4mm-F",
        "2.92mm-M", "2.92mm-F",
        "3.5mm-M", "3.5mm-F",
        "SMA-F",
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_classes.py::test_class_names_returns_ordered_nine -v`
Expected: FAIL with `ImportError` / `cannot import name 'class_names'`.

- [ ] **Step 3: Implement `class_names`**

Append to `training/rfconnectorai/data/classes.py`:

```python
def class_names(path: Path | str) -> list[str]:
    """Ordered list of connector class names (index == class id).

    This is THE source of truth for the flat classifier's label order.
    """
    return [c.name for c in load_classes(path)]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_classes.py::test_class_names_returns_ordered_nine -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/data/classes.py training/tests/test_classes.py
git commit -m "feat: add class_names() helper as classifier label-order source"
```

---

### Task 3: Reconcile labeler `CANONICAL_CLASSES` + add a drift-guard test

**Files:**
- Modify: `training/scripts/pages/1_Training_Data.py:40-46`
- Test: `training/tests/test_class_set_consistency.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_class_set_consistency.py`:

```python
"""Guards the single-source-of-truth invariant for the 9 classes.

classes.yaml, the labeler CANONICAL_CLASSES, and the flat classifier's
default class list must all agree on the same ordered 9 names. If this
test fails, a class list drifted and training/labeling/eval will silently
disagree on label indices.
"""
from pathlib import Path

from rfconnectorai.data.classes import class_names

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "classes.yaml"

EXPECTED = [
    "1.85mm-M", "1.85mm-F",
    "2.4mm-M", "2.4mm-F",
    "2.92mm-M", "2.92mm-F",
    "3.5mm-M", "3.5mm-F",
    "SMA-F",
]


def test_classes_yaml_matches_expected_order():
    assert class_names(CONFIG) == EXPECTED


def test_canonical_classes_matches_yaml():
    import importlib.util

    page = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "pages" / "1_Training_Data.py"
    )
    spec = importlib.util.spec_from_file_location("_training_data_page", page)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CANONICAL_CLASSES == class_names(CONFIG)


def test_classifier_default_classes_match_yaml():
    from rfconnectorai.classifier.train import default_class_names

    assert default_class_names() == class_names(CONFIG)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_class_set_consistency.py -v`
Expected: FAIL — `CANONICAL_CLASSES` still has SMA-M and wrong order; `default_class_names` does not yet exist (added in Task 4).

- [ ] **Step 3: Update `CANONICAL_CLASSES`**

In `training/scripts/pages/1_Training_Data.py`, replace lines 40-46:

```python
CANONICAL_CLASSES = [
    "1.85mm-M", "1.85mm-F",
    "2.4mm-M", "2.4mm-F",
    "2.92mm-M", "2.92mm-F",
    "3.5mm-M", "3.5mm-F",
    "SMA-F",
]
LABEL_CHOICES = ["(skip)"] + CANONICAL_CLASSES
```

- [ ] **Step 4: Run the test (expect 1 of 3 still failing)**

Run: `pytest tests/test_class_set_consistency.py -v`
Expected: `test_canonical_classes_matches_yaml` PASS, `test_classes_yaml_matches_expected_order` PASS, `test_classifier_default_classes_match_yaml` FAIL (resolved in Task 4).

- [ ] **Step 5: Commit**

```bash
git add training/scripts/pages/1_Training_Data.py training/tests/test_class_set_consistency.py
git commit -m "feat: reconcile labeler CANONICAL_CLASSES + add class-set drift guard"
```

---

### Task 4: Add `default_class_names()` to the flat classifier (derive from yaml)

**Files:**
- Modify: `training/rfconnectorai/classifier/train.py:405-419` (the `main()` argparse default + add helper)
- Test: `training/tests/test_class_set_consistency.py` (already written in Task 3)

- [ ] **Step 1: Confirm the failing test**

Run: `pytest tests/test_class_set_consistency.py::test_classifier_default_classes_match_yaml -v`
Expected: FAIL with `ImportError: cannot import name 'default_class_names'`.

- [ ] **Step 2: Implement `default_class_names()` and use it as the argparse default**

In `training/rfconnectorai/classifier/train.py`, add after the imports block (after line 42):

```python
from rfconnectorai.data.classes import class_names as _yaml_class_names

_CLASSES_YAML = (
    Path(__file__).resolve().parents[2] / "configs" / "classes.yaml"
)


def default_class_names() -> list[str]:
    """The 9-class label order, sourced from configs/classes.yaml."""
    return _yaml_class_names(_CLASSES_YAML)
```

Then in `main()` replace the hardcoded `--classes` default (lines ~409-414):

```python
    ap.add_argument("--classes", nargs="+", default=default_class_names())
```

- [ ] **Step 3: Run the test to verify it passes**

Run: `pytest tests/test_class_set_consistency.py -v`
Expected: all 3 PASS.

- [ ] **Step 4: Run the full class-related suite**

Run: `pytest tests/test_classes.py tests/test_class_set_consistency.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/classifier/train.py
git commit -m "feat: classifier default classes derive from classes.yaml (9-class)"
```

---

### Task 5: Add EfficientNetV2-S backbone + configurable input size

**Files:**
- Modify: `training/rfconnectorai/classifier/dataset.py:38-39` (configurable transforms)
- Modify: `training/rfconnectorai/classifier/train.py` (`_build_model`, `TrainConfig`, transforms wiring, `labels.json`)
- Test: `training/tests/test_classifier_backbone.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_classifier_backbone.py`:

```python
import torch

from rfconnectorai.classifier.train import build_model


def test_resnet18_backbone_shape():
    m = build_model(num_classes=9, architecture="resnet18").eval()
    out = m(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 9)


def test_efficientnet_v2_s_backbone_shape():
    m = build_model(num_classes=9, architecture="efficientnet_v2_s").eval()
    out = m(torch.zeros(1, 3, 384, 384))
    assert out.shape == (1, 9)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_classifier_backbone.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_model'` (current name is `_build_model`, resnet-only).

- [ ] **Step 3: Add transforms with a configurable size to `dataset.py`**

In `training/rfconnectorai/classifier/dataset.py`, keep `INPUT_SIZE = 224` and add parameterized factory functions (do NOT delete the existing `make_train_transforms`/`make_eval_transforms` — re-implement them to delegate):

```python
def make_train_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    resize_before = int(input_size * 1.14)
    return transforms.Compose([
        transforms.Resize(resize_before),
        transforms.RandomResizedCrop(input_size, scale=(0.55, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.05), ratio=(0.5, 2.0)),
    ])


def make_eval_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    resize_before = int(input_size * 1.14)
    return transforms.Compose([
        transforms.Resize(resize_before),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
```

(Delete the old hardcoded `RESIZE_BEFORE_CROP` module constant and the two old function bodies; the docstring/comment about 384 regressing ResNet-18 stays relevant — preserve it as a comment above `INPUT_SIZE`.)

- [ ] **Step 4: Add `build_model` (architecture-aware) to `train.py`**

In `training/rfconnectorai/classifier/train.py`, replace `_build_model` with a public, architecture-aware function (mirror the torchvision pattern from `rfconnectorai/classifier/model_multihead.py:58-59`):

```python
def build_model(num_classes: int, architecture: str = "resnet18") -> nn.Module:
    """Backbone with a fresh classification head of `num_classes`."""
    if architecture == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if architecture == "efficientnet_v2_s":
        net = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
        return net
    raise ValueError(f"unknown architecture {architecture!r}")


def _build_model(num_classes: int) -> nn.Module:  # back-compat shim
    return build_model(num_classes, "resnet18")
```

Add to `TrainConfig` (after `class_names: list[str]`):

```python
    architecture: str = "resnet18"
    input_size: int = INPUT_SIZE
```

In `train()`: change the dataset transform calls to pass the size, and the model build to pass architecture:

```python
    train_ds = ConnectorFolderDataset(
        root=config.data_dir, class_names=config.class_names,
        transform=make_train_transforms(config.input_size),
    )
    eval_ds = ConnectorFolderDataset(
        root=config.data_dir, class_names=config.class_names,
        transform=make_eval_transforms(config.input_size),
    )
    ...
    model = build_model(
        num_classes=len(config.class_names),
        architecture=config.architecture,
    ).to(device)
```

In the `labels.json` write block, change `"input_size": INPUT_SIZE` → `"input_size": config.input_size` and `"architecture": "resnet18"` → `"architecture": config.architecture`.

Add argparse options in `main()`:

```python
    ap.add_argument("--architecture", default="resnet18",
                    choices=["resnet18", "efficientnet_v2_s"])
    ap.add_argument("--input-size", type=int, default=INPUT_SIZE)
```

and pass them into `TrainConfig(... architecture=args.architecture, input_size=args.input_size)`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_classifier_backbone.py -v`
Expected: PASS (both backbones return `(1, 9)`).

- [ ] **Step 6: Run the existing classifier tests for no regression**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS (back-compat `_build_model` shim + unchanged default behavior).

- [ ] **Step 7: Commit**

```bash
git add training/rfconnectorai/classifier/dataset.py training/rfconnectorai/classifier/train.py training/tests/test_classifier_backbone.py
git commit -m "feat: EfficientNetV2-S backbone + configurable input size for classifier"
```

---

### Task 6: Make `export_onnx` architecture- and size-aware

**Files:**
- Modify: `training/rfconnectorai/classifier/export_onnx.py:36-95`
- Test: `training/tests/test_onnx_export.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `training/tests/test_onnx_export.py`:

```python
def test_export_efficientnet_v2_s(tmp_path):
    import json
    import torch
    from rfconnectorai.classifier.train import build_model
    from rfconnectorai.classifier.export_onnx import export_to_onnx

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    net = build_model(num_classes=9, architecture="efficientnet_v2_s")
    torch.save(net.state_dict(), model_dir / "weights.pt")
    (model_dir / "labels.json").write_text(json.dumps({
        "class_names": [f"c{i}" for i in range(9)],
        "input_size": 384,
        "architecture": "efficientnet_v2_s",
    }))

    out = export_to_onnx(model_dir, model_dir / "weights.onnx")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_onnx_export.py::test_export_efficientnet_v2_s -v`
Expected: FAIL — `export_onnx._build_model` is resnet18-only, so `load_state_dict` mismatches.

- [ ] **Step 3: Make `export_onnx` use the shared `build_model`**

In `training/rfconnectorai/classifier/export_onnx.py`:

- Delete the local `_build_model` (lines 36-40).
- Add import: `from rfconnectorai.classifier.train import build_model`
- In `export_to_onnx`, read architecture from labels and build accordingly:

```python
    labels_blob = json.loads(labels_path.read_text())
    class_names = labels_blob["class_names"]
    input_size = labels_blob.get("input_size", INPUT_SIZE)
    architecture = labels_blob.get("architecture", "resnet18")

    base = build_model(num_classes=len(class_names), architecture=architecture)
    base.load_state_dict(torch.load(weights_path, map_location="cpu"))
    wrapped = _NormalizedClassifier(base).eval()
```

(Leave `_NormalizedClassifier` and the `torch.onnx.export` call unchanged; `dummy` already uses `input_size`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_onnx_export.py -v`
Expected: PASS (both resnet18 and efficientnet_v2_s export).

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/classifier/export_onnx.py training/tests/test_onnx_export.py
git commit -m "feat: architecture-aware ONNX export (resnet18 + efficientnet_v2_s)"
```

---

### Task 7: Add staged fine-tune support (Phase 2 warm-start at low LR)

**Files:**
- Modify: `training/rfconnectorai/classifier/train.py` (`TrainConfig`, `train()`, `main()`)
- Test: `training/tests/test_classifier_finetune.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_classifier_finetune.py`:

```python
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

    # Phase-1-like checkpoint.
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_classifier_finetune.py -v`
Expected: FAIL — `TrainConfig` has no `init_weights` field.

- [ ] **Step 3: Implement warm-start**

In `training/rfconnectorai/classifier/train.py`:

Add to `TrainConfig`:

```python
    init_weights: Path | None = None
```

In `train()`, immediately after `model = build_model(...).to(device)`:

```python
    if config.init_weights is not None:
        state = torch.load(config.init_weights, map_location=device)
        model.load_state_dict(state)
        print(f"[train] warm-started from {config.init_weights}")
```

Add argparse in `main()`:

```python
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="Phase-1 weights.pt to warm-start from (Phase 2 fine-tune).")
```

and pass `init_weights=args.init_weights` into `TrainConfig(...)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_classifier_finetune.py -v`
Expected: PASS.

- [ ] **Step 5: Run the classifier suite for no regression**

Run: `pytest tests/test_classifier.py tests/test_classifier_backbone.py tests/test_classifier_finetune.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add training/rfconnectorai/classifier/train.py training/tests/test_classifier_finetune.py
git commit -m "feat: staged fine-tune — warm-start classifier from Phase-1 weights"
```

---

### Task 8: Single-class mode for the detector dataset build

**Files:**
- Modify: `training/rfconnectorai/data/build_yolo_dataset.py` (`build_dataset`, `main`)
- Test: `training/tests/test_build_yolo_dataset.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `training/tests/test_build_yolo_dataset.py` (mirror the existing dry-run test style in that file for manifest setup; reuse its existing helper/fixtures if present — otherwise build a 2-instance manifest as below):

```python
def test_single_class_mode_collapses_to_connector(tmp_path):
    import json
    from rfconnectorai.data.build_yolo_dataset import build_dataset

    manifest = tmp_path / "instances.jsonl"
    rows = [
        {"instance_id": "i1", "source_image": "a.jpg", "bbox_xyxy": [0, 0, 10, 10],
         "family": "2.4mm", "precision_family": "2.4mm", "side_a_gender": "male_pin",
         "side_b_gender": "unknown", "polarity": "standard", "mount_style": "cable_mount",
         "orientation": "straight", "termination": "solder",
         "finish_material_cue": "gold", "label_confidence": "high",
         "source_type": "photo"},
        {"instance_id": "i2", "source_image": "b.jpg", "bbox_xyxy": [0, 0, 10, 10],
         "family": "SMA", "precision_family": "standard_sma", "side_a_gender": "female_socket",
         "side_b_gender": "unknown", "polarity": "standard", "mount_style": "cable_mount",
         "orientation": "straight", "termination": "solder",
         "finish_material_cue": "gold", "label_confidence": "high",
         "source_type": "photo"},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    summary = build_dataset(
        manifest=manifest, out_dir=tmp_path / "ds", base_dir=tmp_path,
        dry_run=True, single_class=True,
    )
    assert summary["family_to_idx"] == {"connector": 0}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_build_yolo_dataset.py::test_single_class_mode_collapses_to_connector -v`
Expected: FAIL — `build_dataset()` has no `single_class` parameter.

- [ ] **Step 3: Implement `single_class`**

In `training/rfconnectorai/data/build_yolo_dataset.py`:

Add `single_class: bool = False` to the `build_dataset` keyword-only signature (alongside `dry_run`).

Replace the `family_to_idx = family_index(instances)` line with:

```python
    if single_class:
        family_to_idx = {"connector": 0}
    else:
        family_to_idx = family_index(instances)
```

Replace `cls_idx = class_index_for(instance, family_to_idx)` with:

```python
    cls_idx = 0 if single_class else class_index_for(instance, family_to_idx)
```

(The `names` list at the data.yaml write derives from `family_to_idx`, so it will correctly emit `nc: 1` / `names: [connector]` with no further change.)

Add to `main()` argparse:

```python
    parser.add_argument("--single-class", action="store_true",
                        help="Detector mode: emit nc=1 'connector' (no family/gender).")
```

and thread `single_class=args.single_class` into the `build_dataset(...)` call inside `main()`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_build_yolo_dataset.py -v`
Expected: PASS (new test + existing tests unchanged since default is `False`).

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/data/build_yolo_dataset.py training/tests/test_build_yolo_dataset.py
git commit -m "feat: --single-class mode for detector YOLO dataset (no family collapse)"
```

---

### Task 9: 9-class evaluation / confusion-matrix report

**Files:**
- Create: `training/rfconnectorai/eval/nine_class_report.py`
- Test: `training/tests/test_nine_class_report.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_nine_class_report.py`:

```python
from rfconnectorai.eval.nine_class_report import build_report


def test_report_per_class_and_pairs():
    class_names = ["2.4mm-M", "2.92mm-M", "SMA-F"]
    # y_true, y_pred as class indices.
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 2]  # one 2.4mm-M -> 2.92mm-M confusion

    rep = build_report(y_true, y_pred, class_names)

    assert rep["overall_accuracy"] == 5 / 6     # 1 error in 6 samples
    assert rep["confusion"][0][1] == 1          # 2.4mm-M predicted 2.92mm-M
    assert rep["per_class"]["2.92mm-M"]["recall"] == 1.0
    assert rep["per_class"]["2.4mm-M"]["recall"] == 0.5
    # adjacent-size + M/F confusion surfaced explicitly
    assert "2.4mm-M -> 2.92mm-M" in rep["notable_confusions"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_nine_class_report.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the report (stdlib only, no torch)**

Create `training/rfconnectorai/eval/nine_class_report.py`:

```python
"""9-class confusion report.

Pure stdlib so it runs on the local CPU PC and inside CI without torch.
Surfaces the subtle pairs the customer cares about: adjacent precision
sizes and male/female within a size.
"""
from __future__ import annotations

from typing import Sequence


def _base_and_gender(name: str) -> tuple[str, str]:
    # "2.92mm-M" -> ("2.92mm", "M"); "SMA-F" -> ("SMA", "F")
    base, _, g = name.rpartition("-")
    return (base or name), g


def build_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    n = len(class_names)
    confusion = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    total = len(y_true)
    correct = sum(confusion[i][i] for i in range(n))

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        tp = confusion[i][i]
        support = sum(confusion[i])
        pred_pos = sum(confusion[r][i] for r in range(n))
        recall = tp / support if support else 0.0
        precision = tp / pred_pos if pred_pos else 0.0
        per_class[name] = {
            "support": support,
            "recall": recall,
            "precision": precision,
        }

    notable: dict[str, int] = {}
    for i in range(n):
        for j in range(n):
            if i == j or confusion[i][j] == 0:
                continue
            bi, gi = _base_and_gender(class_names[i])
            bj, gj = _base_and_gender(class_names[j])
            same_size_diff_gender = (bi == bj and gi != gj)
            diff_size = (bi != bj)
            if same_size_diff_gender or diff_size:
                notable[f"{class_names[i]} -> {class_names[j]}"] = confusion[i][j]

    return {
        "overall_accuracy": (correct / total) if total else 0.0,
        "confusion": confusion,
        "class_names": list(class_names),
        "per_class": per_class,
        "notable_confusions": notable,
    }


def render_markdown(report: dict) -> str:
    names = report["class_names"]
    lines = [
        "# 9-Class Evaluation",
        "",
        f"- Overall accuracy: **{report['overall_accuracy']:.4f}**",
        "",
        "## Per-class",
        "",
        "| class | support | precision | recall |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        pc = report["per_class"][name]
        lines.append(
            f"| {name} | {pc['support']} | {pc['precision']:.3f} | {pc['recall']:.3f} |"
        )
    lines += ["", "## Notable confusions (adjacent size / M-vs-F)", ""]
    if report["notable_confusions"]:
        for pair, cnt in sorted(
            report["notable_confusions"].items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"- {pair}: {cnt}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
```

Create `training/rfconnectorai/eval/__init__.py` only if it does not already exist (it does — leave as-is).

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_nine_class_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/rfconnectorai/eval/nine_class_report.py training/tests/test_nine_class_report.py
git commit -m "feat: 9-class confusion report (adjacent-size + M/F surfacing)"
```

---

### Task 10: Rewire the Colab notebook (`training/SMAObjectDetection.ipynb`)

**Files:**
- Modify: `training/SMAObjectDetection.ipynb` (cells 5–10, add new cells)

> Notebook edits use the NotebookEdit tool (cell-indexed). Render the
> `source` exactly as below. Keep the existing clone/cd/pip/pull cells
> (0–4) unchanged.

- [ ] **Step 1: Cell 5 — list the 9 class folders**

Replace cell 5 source with:

```python
%cd /content/hotdogornot
!ls training/data/labeled/embedder
!python -c "import sys; sys.path.insert(0,'training'); from rfconnectorai.data.classes import class_names; print('classes:', class_names('training/configs/classes.yaml'))"
```

- [ ] **Step 2: Cell 7 — build the detector dataset SINGLE-CLASS**

Edit cell 7: add `--single-class` to the `build_yolo_dataset` invocation. The cell's command becomes:

```bash
%%shell
  set -euxo pipefail
  cd /content/hotdogornot
  export PYTHONPATH=/content/hotdogornot/training
  export PYTHONUNBUFFERED=1

  python -u -m rfconnectorai.data.build_yolo_dataset \
      --input datasets/rfconnectors/instances.jsonl \
      --out datasets/rfconnectors --base-dir training \
      --single-class \
      --taxonomy training/rfconnectorai/specs/connectors.yaml

  cat datasets/rfconnectors/data.yaml   # expect nc: 1, names: [connector]
```

- [ ] **Step 3: Cell 8 — detector training unchanged**

Verify cell 8 still trains YOLO against `datasets/rfconnectors/data.yaml` (now single-class). Cell 8 is a `%%shell` cell, so the `%%shell` magic MUST stay the first line — add the annotation as a *bash comment on line 2*, immediately after the magic (NOT before it, which breaks the cell magic):

```
%%shell
# Stage 1: single-class connector localizer (data.yaml is now nc=1).
set -euxo pipefail
... (rest of the existing cell 8 unchanged) ...
```

- [ ] **Step 4: Insert new cell after cell 8 — Phase 1 classifier (combined)**

Insert a new code cell with source:

```python
%%shell
set -euxo pipefail
cd /content/hotdogornot
export PYTHONPATH=/content/hotdogornot/training
export PYTHONUNBUFFERED=1

python -u -m rfconnectorai.classifier.train \
    --data-dir training/data/labeled/embedder \
    --out-dir models/connector_classifier_phase1 \
    --architecture efficientnet_v2_s --input-size 384 \
    --epochs 30 --batch-size 32 --lr 3e-4
```

- [ ] **Step 5: Insert new cell — Phase 2 fine-tune (curated-only)**

Insert a new code cell with source:

```python
%%shell
set -euxo pipefail
cd /content/hotdogornot
export PYTHONPATH=/content/hotdogornot/training
export PYTHONUNBUFFERED=1

# Curated-only dir = the clean 2026-05-14 field set. The embedder folder
# already contains exactly the 9 curated classes in a fresh clone of this
# snapshot; if legacy frames are also present, point --data-dir at a
# curated-only copy. Here we fine-tune from the Phase-1 checkpoint.
python -u -m rfconnectorai.classifier.train \
    --data-dir training/data/labeled/embedder \
    --out-dir models/connector_classifier \
    --architecture efficientnet_v2_s --input-size 384 \
    --init-weights models/connector_classifier_phase1/weights.pt \
    --epochs 12 --batch-size 32 --lr 2e-5
```

- [ ] **Step 6: Insert new cell — 9-class evaluation report**

Insert a new code cell with source:

```python
import sys, json
sys.path.insert(0, "training")
from rfconnectorai.eval.nine_class_report import build_report, render_markdown
# y_true / y_pred are produced by running the Phase-2 model over the
# curated holdout split; see rfconnectorai.classifier.predict for the
# inference helper. Persist the report for the run record:
# report = build_report(y_true, y_pred, class_names)
# open("reports/experiments/classifier_9class/REPORT.md","w").write(render_markdown(report))
print("eval helpers imported; wire y_true/y_pred from the curated holdout.")
```

- [ ] **Step 7: Insert new cell — export + download**

Insert a new code cell with source:

```python
!python -m rfconnectorai.classifier.export_onnx \
    --model-dir models/connector_classifier \
    --output models/connector_classifier/classifier.onnx

!zip -r /content/classifier_9class.zip \
    models/connector_classifier/ \
    reports/experiments/classifier_9class/ 2>/dev/null || true

from google.colab import files
files.download('/content/classifier_9class.zip')
```

- [ ] **Step 8: Validate the notebook JSON**

Run (from repo root): `python -c "import json; json.load(open('training/SMAObjectDetection.ipynb')); print('notebook JSON OK')"`
Expected: `notebook JSON OK`

- [ ] **Step 9: Commit**

```bash
git add training/SMAObjectDetection.ipynb
git commit -m "feat: rewire Colab notebook for 9-class two-stage staged training"
```

---

### Task 11: Full verification + spec coverage check + push

**Files:** none (verification + integration)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all PASS. If pre-existing unrelated failures exist, confirm they are unrelated to touched modules (classes, classifier, build_yolo_dataset, eval) and note them; do not mark complete on a touched-module failure.

- [ ] **Step 2: Spec coverage self-check**

Confirm each spec section maps to a task:
- Class reconciliation → Tasks 1–4
- Single-class detector → Task 8 + notebook Task 10 step 2
- EfficientNetV2-S @ 384 → Task 5
- Staged training → Task 7 + notebook steps 4–5
- Curated-only validation → inherent (val split is from `--data-dir`; Phase 2 uses curated dir)
- 9-class confusion report → Task 9 + notebook step 6
- ONNX export exit criterion → Task 6 + notebook step 7
- Out-of-scope (app rewiring/metrology) → untouched

- [ ] **Step 3: Push to trextrader**

```bash
git push trextrader master
git rev-list --left-right --count master...trextrader/master   # expect: 0  0
```

- [ ] **Step 4: Report completion**

State plainly which tasks completed, the full test result, and the trextrader sync status.

---

## Self-Review

**Spec coverage:** All spec sections map to tasks (see Task 11 Step 2). No gaps.

**Placeholder scan:** No "TBD"/"TODO"/"handle edge cases" — every code step has complete code. The notebook eval cell (Task 10 Step 6) intentionally leaves `y_true/y_pred` wiring as a documented inline comment because the curated-holdout inference loop depends on `rfconnectorai.classifier.predict` runtime state in Colab; the report function itself is fully implemented and tested in Task 9.

**Type consistency:** `build_model(num_classes, architecture)` has the same signature in Tasks 5 and 6. `default_class_names()` (Task 4) and `class_names()` (Task 2) names are consistent across Tasks 3, 4. `single_class` keyword consistent in Task 8. `build_report(y_true, y_pred, class_names)` consistent between Task 9 implementation, test, and notebook usage.

**Known risk flagged in plan:** `dataset.py` carries a hard-won comment that 384px regressed ResNet-18 on small data. The spec deliberately chose EfficientNetV2-S @ 384 (different backbone, larger receptive field) — Task 5 preserves the comment and the resnet path; if Phase-1 metrics regress, the staged design degrades to curated-only per the spec's fallback.
