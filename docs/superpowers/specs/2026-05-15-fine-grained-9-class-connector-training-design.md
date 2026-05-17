# Design: Fine-grained 10-class connector training (training-side)

> Retitled 2026-05-17: class set was reconciled to **10** (SMA-M added 2026-05-16).
> The "9" references in the Problem section below are preserved as a historical
> snapshot of the class-list disagreement as it stood on 2026-05-15.

- Date: 2026-05-15
- Status: Approved design, pending implementation plan
- Scope: Training side only. No on-device app rewiring.

## Problem

The customer uses exactly **9 connector classes** and the model must discriminate
them and no others. Differences between classes are subtle — primarily thread
size/pitch/count and overall diameter — so the model must classify
microscopically; coarse family-level detection is insufficient.

The repo currently does the opposite, and three class lists disagree:

| Source | Classes |
|---|---|
| New curated customer data (`data/labeled/embedder/`) | **9**: 1.85mm M/F, 2.4mm M/F, 2.92mm M/F, 3.5mm M/F, SMA-F |
| `training/configs/classes.yaml` (embedder) | 8: SMA M+F, 3.5/2.92/2.4 M/F — no 1.85mm |
| `CANONICAL_CLASSES` in `training/scripts/pages/1_Training_Data.py` | 10: adds SMA-M + 1.85mm M/F |
| Last `build_yolo_dataset` run in the notebook | **3**: `2.4MM, 2.92MM, 3.5MM` — gender collapsed, SMA & 1.85mm dropped |

The notebook's instance→YOLO build derives `family_to_idx` from instances and
collapses everything to 3 size families with no gender and no SMA/1.85mm — the
exact opposite of the required behavior.

## The 9 classes (single source of truth)

```
1.85mm-M, 1.85mm-F, 2.4mm-M, 2.4mm-F, 2.92mm-M, 2.92mm-F, 3.5mm-M, 3.5mm-F, SMA-F
```

Note: **SMA-M is not in the customer set** (SMA-F only). 1.85mm M/F **are** in
the set.

## Architecture (two-stage, runtime shape unchanged)

- **Stage 1 — Detector:** YOLO stays a **single-class `connector` localizer**
  (`configs/detector.yaml`, already `nc: 1`, names `{0: connector}`). It only
  has to find the connector and draw a tight box. The instance→YOLO
  *multi-class* build is removed/disabled so it can no longer collapse or
  mislabel classes. The detector never sees family or gender.
- **Stage 2 — Classifier:** a **flat 9-class** image classifier operating on
  the tight detector crop, built on the existing
  `rfconnectorai.classifier` path (`train.py`, `dataset.py`,
  `label_encoding.py`, `export_onnx.py`). The crop is upscaled to ≥384 px so
  thread detail is resolvable (one-stage 640px whole-frame detection cannot
  resolve thread pitch on small connectors — this is why two-stage was chosen).
- **Flow:** frame → YOLO box → crop → 9-class classifier → label.

## Class-set reconciliation

Define the 9 classes once and make everything derive from it:

- Rewrite `training/configs/classes.yaml` to exactly the 9 (drop SMA-M, add
  1.85mm M/F). Preserve the existing per-class metadata schema (id, name,
  family, gender, inner_pin_diameter_mm, frequency_ghz_max, impedance_ohms,
  mating_torque_in_lb); fill metadata for the added 1.85mm classes.
- Update `CANONICAL_CLASSES` in `training/scripts/pages/1_Training_Data.py` to
  the same 9 (remove SMA-M).
- Classifier label encoding derives the vocab from `classes.yaml` — no
  hardcoded class list in the classifier.
- Add a test asserting `classes.yaml`, `CANONICAL_CLASSES`, and the classifier
  label encoding all agree on the same 9 (fails loud on future drift).

## Classifier model & training — staged (Approach C)

- **Backbone:** EfficientNetV2-S, pretrained, input 384 px on the crop
  (already used in the repo; good fine-grained capacity).
- **Phase 1 — representation:** train on the **combined** dataset — the full
  legacy `data/labeled/embedder` set (present in a fresh trextrader Colab
  clone) remapped to the 9-class vocab + the curated set. Classes that cannot
  be mapped to one of the 9 are **excluded**, never collapsed. Class-balanced
  sampling.
- **Phase 2 — calibration:** fine-tune at low learning rate on **curated-only**
  (the clean field images committed 2026-05-14, `photo_2026-05-14_*`).
- **Validation:** always a held-out split of **curated-only**. Accuracy is
  judged on the real field distribution; synthetic/legacy data never appears
  in validation.
- **Augmentation:** tuned to preserve thread cues — no aggressive
  blur/downscale that destroys thread pitch. Class weighting / oversampling for
  SMA-F (only 12 curated images) and other low-count classes.

Fallback: if Phase 1 (legacy) proves too noisy to help, the design degrades
gracefully to curated-only training (Approach A) by skipping Phase 1.

### Current curated counts (per class, on disk 2026-05-15)

```
1.85mm-F 36   1.85mm-M 28   2.4mm-F 10   2.4mm-M 16
2.92mm-F 21   2.92mm-M 16   3.5mm-F 28   3.5mm-M 29   SMA-F 12
```

## Notebook changes (`training/SMAObjectDetection.ipynb`)

- **Cell 5:** keep the data-presence check; list the 9 class folders.
- **Cell 7:** stop the multi-class instance→YOLO family build. Produce a
  single-class detector dataset (or point detector training at the existing
  `data/labeled/detector` path / `detector.yaml`). Net effect: detector dataset
  is `nc: 1`, `names: {0: connector}`.
- **Cell 8:** detector training unchanged (single-class YOLO).
- **New cells (in order):**
  1. Phase-1 classifier training (combined data, remapped to 9).
  2. Phase-2 fine-tune (curated-only, low LR).
  3. 9-class evaluation: per-class precision/recall + confusion matrix,
     explicitly surfacing subtle pairs (adjacent sizes; M vs F within a size).
  4. ONNX export via `classifier/export_onnx.py` + zip/download.
- All paths point at the reconciled config; no hardcoded family lists anywhere
  in the notebook.

## Evaluation & exit criteria

The work is complete when:

- `classes.yaml`, `CANONICAL_CLASSES`, classifier label encoding all agree on
  the 9 classes, enforced by a passing test.
- The detector dataset build is single-class (`nc: 1`); the family-collapse
  path is gone.
- A validated **9-class `classifier.onnx`** is exported.
- A confusion-matrix report on the curated holdout exists, reporting per-class
  metrics and the subtle-pair confusions (e.g. 2.4mm-M vs 2.92mm-M; M vs F
  within a size).

No numeric accuracy gate is fixed in this spec; the confusion-matrix report is
the deliverable that makes subtle-pair accuracy measurable. A target gate can
be set in a follow-up once a baseline number exists.

## Out of scope

- On-device Capacitor app rewiring to consume the flat 9-class ONNX (separate
  follow-up spec).
- The multi-head attribute classifier model.
- Measurement / metrology features.
- `origin` / `probably-on-fire` remote — push only to `trextrader`.

## Risks

- **Small curated set** (196 images, SMA-F only 12). Mitigated by staged
  training + class weighting + curated-only validation; degrades to
  curated-only if legacy hurts.
- **Legacy domain gap** (synthetic/video, dup-heavy). Mitigated by Phase-2
  curated-only calibration and curated-only validation.
- **Local vs Colab data divergence:** the full legacy set lives in trextrader
  git history, not the local working tree. The notebook runs in a fresh Colab
  clone where both sets are present; local runs would only have the curated
  set. Implementation must not assume local presence of the legacy set.
