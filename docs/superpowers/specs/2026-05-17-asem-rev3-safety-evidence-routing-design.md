# Design: ASEM Rev 3 — Safety & Evidence-Routing Release

- Date: 2026-05-17
- Status: Approved design, pending implementation plan
- Author of source proposal: Dr. T. Jerry Mahabub, Ph.D. (QMTF, Company Confidential)
- Formal long-form + math: `docs/Adaptive Evidence Mesh for Near-Zero-Risk Connector Identification.pdf`
  (Appendix B/B1/B2). This document is the **engineering spec scoped to rev 3**; the PDF is the
  formal architecture/derivation reference.
- Target architecture diagram: `docs/fulldetector_active_visual_interrogation_system.dot`
  (to be revised per §8 before it becomes the canonical README image).

---

## 1. Problem and corrected diagnosis

Rev 2 shipped a flat 10-class EfficientNetV2-S classifier behind a single-class YOLO11n
detector, on-device via Capacitor WebView + ONNX Runtime Web. On a real field capture
(a coax/F-type TV connector unscrewed from a hotel-room TV — a genuinely **out-of-support**
object) the app returned a hard, moderately-confident wrong answer instead of declining.

The corrected diagnosis (confirmed by the technical review of the proposal):

- The rev-2 miss was **not** primarily a "needs finer probe models" failure.
- It was a **support-set / forced-answer** failure: the flat path has no `unsupported`,
  no abstention, and no required-evidence gate, so it is structurally incapable of saying
  "this isn't one of my 10 classes" or "I can't safely decide from this capture."
- The Kaggle confusion report (Appendix A: 0.9732 over all 635 incl. training data) is
  optimistic; the reliable signal is the confusion *structure* (well-distributed, 2.4 mm
  bottleneck closed, gender clean, weakest = 1.85mm-F @ recall 0.915). In-distribution
  organization is good; **field decision policy is the gap.**

The class set is the locked customer set of **10**: `1.85mm-M, 1.85mm-F, 2.4mm-M,
2.4mm-F, 2.92mm-M, 2.92mm-F, 3.5mm-M, 3.5mm-F, SMA-F, SMA-M`. No other classes are in
scope; generic RP-SMA / right-angle / bulkhead / board-mount variants are **not** part of
this problem and must be removed from the canonical diagram (see §8).

## 2. Target architecture (ASEM) — summary

Adaptive Semantic Evidence Mesh: keep the flat detector+classifier as a fast **stage 0**,
and make hard captures progressively accumulate targeted evidence until the system can
either declare *nailed it* or responsibly route to `unsupported` / `need_second_angle` /
`need_scale_reference` / `ambiguous`. Evidence "points" are connector-semantic
(center contact, annular aperture, thread ring, body span, axial shoulder), spent
adaptively (expected information gain per cost), and gated by calibrated risk + a
required-visible-evidence check. Full staged pipeline and Bayesian fusion math: PDF
Appendix B/B2.

**We commit to ASEM as the target architecture.** This spec does *not* build all of it now.
It defines **rev 3 = the safety and evidence-routing release**, with rev 4–6 as the
data-backed capability roadmap.

## 3. Locked decisions

| # | Decision |
|---|----------|
| D1 | **Rev 3 scope** = Stage-0 flat model + calibrated abstention + OOD/support gate (embedding-distance/energy v1) + hard-case logging loop. No probes, no multi-head, no scale workflow, no native rewrite. |
| D2 | **Rev staging:** Rev 3 = safety/routing. Rev 4 = restored multi-head attribute refiner (existing scaffolding). Rev 5 = center + ring semantic probes (+ the native-vs-WebView gate, see D3). Rev 6 = thread/body-span/axial probes + guided recapture + scale-marker (ArUco) path. |
| D3 | **Runtime:** Rev 3 stays Capacitor WebView + ONNX Runtime Web (WASM/WebGL). Rev 3 has no per-frame probe budget, so there is **zero runtime risk**. "Native (TFLite/NNAPI) rewrite vs stay WebView" is an explicit **rev-5 gate**, decided on *measured* ORT-Web probe latency, not guessed now. All TFLite/NNAPI language is removed from the canonical diagram. |
| D4 | **Support gate v1** = calibrated embedding-distance / energy score over the *existing* trained classifier embedding (or energy over logits). Requires **no negative corpus to train**; trained on the full existing curated set. A corpus/classifier-trained support gate is a later rev, fed by the hard-case pool. |
| D5 | **OOD validation path:** real-world support-gate / field validation is done by **Chris in California against the actual customer connectors**. Workflow: models trained → APK built → Chris side-loads on Android → tests on real connectors → returns results + hard captures. Rev 3 does not block on any data-acquisition task. |
| D6 | **Acceptance policy:** one coherent stage-0 regime (§5). SPRT / log-odds (Λ_t) is reserved for the **sequential evidence loop** (rev 5+), *not* used as a redundant single-shot gate on the flat softmax. The PDF's strict τ_accept=0.995–0.999 is the *future full "nailed it"* target, **not** applied as-is to the rev-3 real-phone flat model. |
| D7 | **Success metric reframed.** Rev 3 success = **risk-on-accepted ↓, abstention ↑, field telemetry ↑**. NOT "more forced predictions correct." The spec states plainly: *rev 3 will likely abstain more often than rev 2; that is the intended correction to forced incorrect classification, not a regression.* Accepted-coverage ↑ is a rev-4/5 outcome, after hard-case data improves the gate/refiner/probes. |
| D8 | **`Q_t` capture-quality latent** is a first-class component (blur / glare / oblique angle / center resolution / crop scale / ROI quality). In rev 3 it drives abstention reasons (`need_better_focus` / `need_second_angle`). In rev 5+ it jointly down-weights all probe likelihoods so correlated bad-frame evidence cannot be naively multiplied into overconfident posteriors. |
| D9 | **Migration discipline:** the working flat APK path is preserved. `/predict`-style output stays backward-compatible; rev 3 adds new structured fields **alongside** the existing ones, never replacing them. |

## 4. Rev-3 components and data flow

Pipeline (all in `exports/web/`, the canonical git-tracked source; mirrored to mobile by
`scripts/copy-web.js` + `npx cap sync android`):

```
camera frame
  └─ detector (YOLO11n, single-class)         → bbox, box_conf
       └─ [Q_t capture-quality estimate on ROI: blur/glare/angle/center-res/scale]
            └─ flat classifier (EffNetV2-S)    → logits
                 └─ temperature calibration    → calibrated π(c)
                      └─ support score s_ood   → p(in_support) vs p(unsupported)
                           └─ Stage-0 Decision Controller (§5)
                                ├─ ACCEPT  → class + π + evidence/reason trace
                                └─ ABSTAIN → reason ∈ {no_connector_found, unsupported_connector,
                                              need_second_angle, need_better_focus, ambiguous}
                                                 └─ Hard-Case Logging (§6)
```

Components to build in rev 3:

1. **Temperature calibration.** Fit a single scalar `T` (temperature scaling, Guo et al.)
   on a held-out split; export `T` into the model bundle. Consumer applies
   `softmax(logits / T)`. No retrain of the backbone.
2. **Support score `s_ood`.** Calibrated open-set score with no negatives required:
   energy score over logits (`-T·logsumexp(logits/T)`) and/or Mahalanobis/cosine distance
   in the penultimate embedding. The classifier ONNX must expose either logits (already
   available) or the embedding; if only `_NormalizedClassifier` output is available, use
   the logit-energy variant. Threshold tuned on the curated set's score distribution;
   final threshold ratified by Chris's field results (D5).
3. **`Q_t` capture-quality estimator.** Lightweight, classical-CV-first (variance-of-Laplacian
   blur, glare/exposure, ROI scale fraction, bbox aspect/oblique proxy, center-region
   resolution). No new model in rev 3. Emits a small struct + an overall `q_low` flag.
4. **Stage-0 Decision Controller** in `exports/web/app.js` — the policy in §5. Pure
   function over (box_conf, Q_t, π, s_ood, thresholds). Emits decision + reason + trace.
5. **Hard-case logging loop** (§6).
6. **`thresholds.json`** bundle contract (§7) — all tunables externalized, no magic numbers
   in `app.js`.
7. **UI / output states** — render ACCEPT vs each ABSTAIN reason with actionable guidance
   ("move closer", "improve focus", "show the center", "this connector isn't supported").

## 5. Rev-3 acceptance policy (one coherent regime)

Stage-0 only. SPRT is **not** here (D6).

```
decide(box_conf, Q_t, π, s_ood, thr):
    if box_conf < thr.box_min:
        return ABSTAIN(no_connector_found)

    if Q_t.q_low:                      # blur / glare / oblique / center unreadable
        return ABSTAIN(need_better_focus | need_second_angle)   # pick by dominant Q_t cause

    if s_ood >= thr.unsupported:
        return ABSTAIN(unsupported_connector)

    c1, c2 = top2(π)
    if π[c1] >= thr.accept
       and (π[c1] - π[c2]) >= thr.margin
       and required_evidence_visible(c1, Q_t):     # rev-3: coarse — center-region readable
        return ACCEPT(c1, π, trace)

    return ABSTAIN(ambiguous, alternatives=topk(π))
```

- `required_evidence_visible` in rev 3 is **coarse**: it only asserts the center region of
  the ROI is resolvable enough (`Q_t.center_res` ≥ threshold) to make a gender call. The
  true per-class ReqVisible table (pin vs socket evidence) arrives with the center probe in
  rev 5. The gate is asymmetric by design: bias toward "not visible" — a false
  "visible=1" is worse than an abstention.
- `thr.accept` / `thr.margin` are **tuned on a real-phone validation pass for a chosen
  risk-on-accepted target**, not set to the PDF's strict 0.995/0.25. Coverage is allowed
  to float; low initial coverage is expected and acceptable (D7).
- Rationale for the order: cheapest/safest rejections first (no ROI → bad frame →
  out-of-support → ambiguous), so the model never "reasons" on a frame that can't support
  a decision.

SPRT / Λ_t = log(π_t(c\*)/π_t(c²)) with ε-stabilization, and the strict combined
acceptance stack, are specified in PDF Appendix B2 and become active only when there are
genuine **sequential** posterior updates (support → refiner → probes), i.e. rev 5+.

## 6. Hard-case logging loop (first-class, not telemetry)

Every capture that does **not** ACCEPT (and a sampled fraction that does) is a hard-case
candidate. This is the data flywheel for rev 4/5 — it is a component, not an afterthought.

- **Capture:** ROI crop + full frame thumbnail + decision trace (π, s_ood, Q_t struct,
  reason, thresholds snapshot, model/version ids, timestamp).
- **Consent + local storage:** opt-in; stored on-device only; explicit user control;
  no automatic upload. (Field testers like Chris export deliberately.)
- **Failure tag:** quick on-device tag picker — `blur`, `poor_center`, `scale_ambiguous`,
  `side_angle_needed`, `family_confusion`, `gender_confusion`, `out_of_support`,
  `detector_missed`.
- **Export → labeling:** a deterministic export bundle that the repo's existing
  specimen-aware dataset builder can ingest (attach as a *new sparse annotation file* on
  the instance manifest — do **not** fork a second training universe; the builder already
  guards specimen leakage).
- **Retraining:** the hard-case pool becomes the primary dataset for the rev-4 refiner
  recalibration and rev-5 probe training, grouped by specimen + capture session.

## 7. `thresholds.json` contract

All tunables externalized so field tuning needs no code change or rebuild beyond asset
swap. Initial values are *starting points to be tuned on real-phone data*, not truths:

```jsonc
{
  "box_min":        0.25,        // detector box confidence floor
  "accept":         0.85,        // tune to risk-on-accepted target on real-phone val
  "margin":         0.20,        // top1 - top2 minimum
  "unsupported":    0.60,        // s_ood threshold (raise to 0.80 once gate is corpus-trained)
  "q": {
    "blur_var_min": 60.0,        // variance-of-Laplacian floor
    "center_res_min": 0.85,      // resolvable-center fraction for a gender claim
    "roi_scale_min": 0.08,       // ROI area fraction of frame
    "glare_frac_max": 0.20
  },
  "hardcase_accept_sample_rate": 0.05,
  "calibration_T": 1.0           // overwritten by fitted temperature
}
```

`app.js` must contain **zero** hardcoded decision constants — all read from this file
(loaded next to `classifier_labels.json`).

## 8. Canonical diagram revision (deliverable)

Revise `docs/fulldetector_active_visual_interrogation_system.dot` (then render hi-res PNG +
SVG) with exactly these eight changes before it becomes the canonical README architecture
image:

1. Replace **all** TFLite / NNAPI / `*.tflite` nodes and labels with **ONNX Runtime Web /
   WASM / WebGL** (per D3).
2. Replace the generic SMA taxonomy examples (SMA male straight, RP-SMA, right-angle,
   bulkhead, cable-end, board-mount) with the **locked 10 precision classes**.
3. Add **OOD Negative Corpus** as a *future* dataset node (rev-3.x+; not required to train
   the v1 gate — annotate accordingly).
4. Add **Support Gate** immediately after the ROI crop, before the flat classifier
   consumes the decision.
5. Add **Hard-Case Pool** as a major feedback loop (capture → consent/local storage →
   failure tag → labeling → retraining), not a telemetry leaf.
6. Add **`no_connector_found` / `low_box_confidence`** as a first-class output state.
7. Move the probe stack (center/ring/thread/body-span/axial) into a clearly marked
   **"future staged modules (rev 5–6)"** cluster — not the rev-3 active path.
8. Add the **`Q_t` Capture-Quality Latent** node (blur / glare / oblique angle / center
   resolution / crop scale / ROI quality) feeding both the abstention router (rev 3) and,
   dashed/future, the probe-likelihood down-weighting (rev 5+).

Acceptance-logic nodes: collapse to the single coherent stage-0 regime; keep SPRT/Λ_t
nodes but visibly scoped to the future sequential-evidence loop.

## 9. Evaluation and the rev-3 gate

Three tiers (promotion gated on tiers 2–3, never tier 1):

| Tier | Content | Role |
|---|---|---|
| In-distribution curated | clean class-balanced specimen images | debug/ablation only |
| Real-phone (Chris, CA) | actual customer connectors, varied surface/light/angle | **main promotion gate** |
| Stress / OOD | hotel-TV / coax / cable-back / unsupported / blur / oblique | **safety gate** |

Metrics: keep detector mAP, per-class P/R/F1, top-k, ECE; **add** coverage@target-risk,
abstention rate (by reason), risk-on-accepted, OOD rejection accuracy (validated via
Chris), AURC / risk-coverage. Rev-3 ships when: on tiers 2–3, **risk-on-accepted is below
target and the hotel-TV-class capture routes to `unsupported`/abstain rather than a hard
answer** — even if accepted coverage is low.

## 10. Non-goals for rev 3 (explicit)

No semantic probe models. No multi-head refiner (rev 4). No scale/ArUco workflow. No
native rewrite. No new OOD training corpus. No change to the locked 10-class set. No
strict 0.995 acceptance applied to the live flat model. No backbone retrain (calibration
is post-hoc).

## 11. Stated limitations (no false comfort)

- The v1 support gate is an energy/distance heuristic — genuinely weak vs a
  corpus-trained open-set classifier. It is the *right rev-3 step* (fixes the demonstrated
  forced-answer bug with zero data dependency) but its quality ceiling is bounded until
  the hard-case/OOD corpus exists.
- Exact thread pitch/height/size-family is not always recoverable from one handheld
  blurred frame without scale. Rev 3's correct response is abstention
  (`need_second_angle` / future `need_scale_reference`), not false precision.
- Reproducibility action item carried into rev 3: pin the training notebook to a commit
  SHA and stop storing `...`-elided cell sources, so the bundle is auditable.
- Detector mis-box on cluttered/oblique scenes is real; `no_connector_found` /
  `low_box_confidence` is the first-class mitigation, and detector recall on stress-tier
  captures is a tracked metric.

## 12. References

- `docs/Adaptive Evidence Mesh for Near-Zero-Risk Connector Identification.pdf`
  — formal proposal, Appendix B (ASEM design), B1 (equation→repo-file map), B2 (formal math).
- `docs/fulldetector_active_visual_interrogation_system.dot` — target architecture diagram (pre-revision).
- `docs/superpowers/specs/2026-05-15-fine-grained-9-class-connector-training-design.md` — the training-side spec this builds on (retitled in-content to 10-class 2026-05-17; filename unchanged to preserve cross-references).
- Locked class-set constraint: see project memory `customer-connector-class-set`.
