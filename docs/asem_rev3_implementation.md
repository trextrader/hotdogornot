# ASEM Rev 3 Implementation Plan — Safety & Evidence-Routing Release

> **For agentic workers:** implement task-by-task from `docs/asem_rev3_tasks.md`
> (checkbox tracking). Both files share **one canonical stage map** (§3).

- Date: 2026-05-17
- Status: Implementation-ready (converted from source draft; gap-closers folded)
- Source design: `docs/superpowers/specs/2026-05-17-asem-rev3-safety-evidence-routing-design.md`
- Companion task list: `docs/asem_rev3_tasks.md`
- Runtime target: Capacitor WebView + ONNX Runtime Web (WASM/WebGL)

**Goal:** Stop the Rev-2 failure where an out-of-support or low-evidence capture
receives a hard in-support class answer. Add calibrated abstention, an OOD/support
gate, a capture-quality latent, and hard-case logging — without breaking the working
flat APK path.

**Rev-3 success = risk-on-accepted ↓, abstention ↑, hard-case telemetry ↑.**
Rev 3 will likely abstain more than Rev 2. That is the intended correction, **not** a
regression. Accepted coverage rises later (Rev 4/5) after hard-case data exists.

---

## 1. Non-goals (hard scope)

No semantic probes. No multi-head refiner. No scale/ArUco. No native TFLite/NNAPI
rewrite. No new OOD training corpus. No strict 0.995 "nailed it" threshold on the
live flat model. No backbone retrain. No new connector classes. No pin/socket
inference in Rev 3.

## 2. Binding architecture decisions

- **Runtime:** stays Capacitor WebView + ONNX Runtime Web (WASM/WebGL). `exports/web/`
  is canonical source; `scripts/copy-web.js` + `npx cap sync android` is the mobile
  sync path. Native-vs-WebView is a Rev-5 decision on measured probe latency.
- **Locked class set (read order from `exports/web/classifier_labels.json`, never
  hardcode):** `1.85mm-M, 1.85mm-F, 2.4mm-M, 2.4mm-F, 2.92mm-M, 2.92mm-F, 3.5mm-M,
  3.5mm-F, SMA-F, SMA-M`. No RP-SMA / right-angle / bulkhead / cable-end / board-mount
  / generic-SMA strings in any UI, diagram node, fixture, or branch.
- **`s_ood` contract:** larger = more likely unsupported. If code computes
  `p_in_support`, it MUST convert `s_ood = 1 - p_in_support`. Controller tests
  `if (s_ood >= thresholds.unsupported) ABSTAIN("unsupported_connector")`. Sign
  reversal = critical failure (rejects good, accepts bad).
- **Support gate v1:** logit-energy is the **mandatory** baseline
  `energy = -T * logsumexp(logits / T)`. Embedding-distance is optional and only if
  the existing ONNX already exposes the penultimate embedding. **Must not block on
  embedding export.**
- **Acceptance:** single Stage-0 policy (box → quality → support → coarse
  required-visible → confidence+margin → ambiguous). **SPRT/Λ_t is NOT in Rev 3** —
  it belongs to Rev 5+ sequential evidence.

## 3. Canonical stage map (shared by plan and tasks)

| Stage | Title |
|---|---|
| 0 | Preflight audit & baseline preservation (+ gap-closers: copy-web coverage, ESM/test loadability, real ONNX filenames) |
| 1 | `thresholds.json` + fail-loud loader |
| 2 | `support.js`: temperature calibration + logit-energy `s_ood` |
| 3 | **Support-gate calibration measurement** (curated set → energy percentiles → write `thresholds.json`) — *gap-closer, hard dependency for Stage 12* |
| 4 | `quality.js`: `Q_t` estimator |
| 5 | `decision.js`: Stage-0 controller (+ explicit required-visible vs `Q_t` relationship) |
| 6 | `hardcase.js`: trace + local opt-in logging + export (+ Android export path decision) |
| 7 | `app.js` wiring (preserve legacy fields) |
| 8 | Tests: 13 acceptance cases + test infra |
| 9 | Canonical diagram revision + render |
| 10 | Browser smoke test |
| 11 | Android sync + debug APK |
| 12 | Field-validation package (Chris, CA) |
| 13 | Reproducibility cleanup + docs update |

`docs/asem_rev3_tasks.md` uses these exact numbers. Update that file as each stage
completes.

## 4. Target file layout

```
exports/web/
  thresholds.json
  app.js                      (modified; legacy fields preserved)
  classifier_labels.json      (existing; label-order source of truth)
  models/detector.onnx        (existing — REAL filename, verify in Stage 0)
  models/classifier.onnx      (existing — REAL filename, verify in Stage 0)
  asem/
    support.js
    quality.js
    decision.js
    hardcase.js
scripts/
  copy-web.js                 (audit/extend in Stage 0 to cover asem/ + thresholds.json)
  fit_support_energy.py       (NEW — Stage 3 calibration measurement)
docs/
  fulldetector_active_visual_interrogation_system.{dot,svg,png}   (Stage 9)
  asem_rev3_implementation.md
  asem_rev3_tasks.md
tests/asem/
  decision.test.* support.test.* quality.test.* hardcase_schema.test.*
```
Preserve the repo's existing test-dir convention if one exists; keep equivalent
coverage.

---

## Stage 0 — Preflight audit & baseline preservation

**Goal:** confirm the existing flat path works and resolve all execution unknowns
before adding logic.

Required checks:
- Locate the **real** detector/classifier ONNX filenames (earlier APK inspection
  showed `detector.onnx` / `classifier.onnx`, *not* the idealized
  `yolo11n_*`/`effnetv2s_*`). Record the actual names — the trace schema (Stage 6)
  must use these, not aspirational ones.
- Confirm classifier output: raw **logits** preferred (needed for energy + temp
  scaling); normalized probabilities acceptable fallback (document it; energy then
  uses a documented logit-recovery or probability-energy variant).
- Confirm where prediction output and UI rendering happen in `app.js`; enumerate the
  legacy output fields (`class`, `confidence`, `bbox`, any `/predict`-style) — these
  are preserved verbatim.
- **Gap-closer A — `copy-web.js` coverage:** open `scripts/copy-web.js`. If it copies
  an explicit allowlist (not a recursive dir copy), extend it so
  `exports/web/asem/**` and `exports/web/thresholds.json` are copied into `www`.
  Without this the APK silently runs stale code.
- **Gap-closer B — module loadability:** determine how `exports/web` JS is loaded in
  the WebView (plain `<script>` vs ES modules). The `asem/*.js` modules must be
  authored so they are **both** browser-loadable in that scheme **and** importable by
  the chosen Node test runner. Decide the module format now (recommended: ESM with a
  thin browser entry) and record it.
- Confirm the current debug APK still builds before any change.

**Output:** no functional change. Commit message / note:
`Baseline verified: flat path preserved; real ONNX names <…>; copy-web covers asem/.`

**Acceptance:** app runs in browser; legacy UI displays; legacy fields documented;
real ONNX names recorded; `copy-web.js` proven to cover new paths; module/test format
decided; no Rev-3 logic active.

## Stage 1 — `thresholds.json` + fail-loud loader

**Create** `exports/web/thresholds.json`:
```jsonc
{
  "schema_version": "asem_rev3_thresholds_v1",
  "thresholds_version": "rev3_initial_2026_05_17",
  "box_min": 0.25,
  "accept": 0.85,
  "margin": 0.20,
  "unsupported": 0.60,
  "q": {
    "blur_var_min": 60.0,
    "roi_scale_min": 0.08,
    "glare_frac_max": 0.20,
    "oblique_proxy_max": 0.65,
    "center_res_min": 0.85
  },
  "support_calibration": {
    "method": "energy_minmax",
    "energy_in_support_p05": null,
    "energy_in_support_p95": null
  },
  "hardcase_accept_sample_rate": 0.05,
  "calibration_T": 1.0
}
```
> `support_calibration` percentiles start `null` and are **filled by Stage 3**. A
> `null` here MUST cause a visible error if the energy path needs them (no silent
> fallback into an arbitrary threshold).

Loader: load at startup; validate required top-level + `q` keys; all numerics
finite; `calibration_T > 0`; `hardcase_accept_sample_rate ∈ [0,1]`. Missing or
malformed → **visible startup error**, never silent defaults. No decision constant
hardcoded in `app.js`.

**Acceptance:** valid file loads; missing/malformed errors visibly; no hardcoded
`box_min/accept/margin/unsupported/q.*` in `app.js`.

## Stage 2 — `support.js`: calibration + logit-energy `s_ood`

Exports: `softmaxWithTemperature(logits, T)`, `logsumexp(values)`,
`energyScoreFromLogits(logits, T)`, `normalizeEnergyToSOod(energy, calibration)`,
`computeSupportScore({logits, probabilities, thresholds, calibration})`.

- Stable softmax (divide by T → subtract max → exp → normalize); throw on
  `T<=0`/non-finite/empty.
- `energy = -T * logsumexp(logits / T)`.
- `normalizeEnergyToSOod`: when `calibration.method==="energy_minmax"` with finite
  `p05<p95`, return `clamp01((energy - p05)/(p95 - p05))` (higher energy → higher
  `s_ood`). If percentiles are `null`/missing → **throw** (forces Stage 3 to run; no
  arbitrary threshold).
- `computeSupportScore` uses logit-energy when logits exist; never requires
  embedding; returns `{ s_ood, method:"energy", energy }`.

**Acceptance:** softmax sums ≈1; invalid T throws; sign contract holds and is
comment-documented + tested (`p_in_support=0.9 → s_ood≈0.1`); no embedding required;
missing percentiles throw (not silently pass).

## Stage 3 — Support-gate calibration measurement *(gap-closer; hard dependency)*

**Goal:** replace the `null` energy percentiles with real numbers measured from the
existing curated set, so the support gate — the primary Rev-3 fix for the hotel-TV
bug — actually discriminates.

**Create** `scripts/fit_support_energy.py`:
- Run the **current** classifier ONNX over the full curated set
  (`data/labeled/embedder/**`), collect per-image logit-energy
  `-T·logsumexp(logits/T)` using `calibration_T`.
- Compute the in-support energy distribution; write `energy_in_support_p05` and
  `energy_in_support_p95` into `exports/web/thresholds.json` (and pick a defensible
  starting `unsupported` operating point on the normalized score, documented).
- Emit a short report (energy histogram stats, chosen percentiles, resulting
  `s_ood` for the curated set min/median/max) into `reports/` so the choice is
  auditable.

This must complete and its values be committed **before Stage 12** (field package).
Until then the support gate is not field-meaningful.

**Acceptance:** `thresholds.json` percentiles are non-null and measured; report
committed; re-running the script is deterministic.

## Stage 4 — `quality.js`: `Q_t` estimator

`estimateQuality({frameWidth, frameHeight, roiImageData, bbox, thresholds})` →
```
{ blur_var, glare_frac, roi_scale, center_res, oblique_proxy,
  q_low: boolean, dominant: "none"|"blur"|"glare"|"low_roi_scale"|"oblique"|"poor_face_visibility"|"low_center_res" }
```
Metrics:
- `roi_scale = bbox_area / frame_area`; fail if `< q.roi_scale_min` → dominant
  `low_roi_scale` (NOT `low_center_res` — fixes the source-draft §4.1 mislabel).
- `blur_var` = variance-of-Laplacian (or close JS equiv); fail if `< q.blur_var_min`
  → `blur`.
- `glare_frac` = near-saturated px / total; fail if `> q.glare_frac_max` → `glare`.
- `oblique_proxy` = bbox-aspect / geometry proxy; fail if `> q.oblique_proxy_max` →
  `oblique` (or `poor_face_visibility`).
- `center_res` = coarse resolvability proxy
  (`clamp01(min(roi_w,roi_h)/required_roi_min_px)` or center-area sqrt proxy).
  **Computed here but it does NOT set `q_low`** — see Stage 5 gap-closer C.

**Gap-closer C (threshold relationship):** `q_low` is driven by **blur, glare,
roi_scale, oblique** only. `center_res` is evaluated **solely** by
`requiredEvidenceVisible` in Stage 5. This makes the required-visible branch (and
test 7) reachable instead of being shadowed by an earlier `q_low` catch.

`q.center_res_min` is **strictly the required-visible floor**, deliberately distinct
from the `q_low` quality gates. `quality.js` performs **no pin/socket inference**.

Dominant priority among the `q_low` causes: `blur > glare > oblique > low_roi_scale`.

**Acceptance:** bad blur/glare/small-ROI/oblique trip `q_low`; `center_res` is
reported but never trips `q_low`; no pin/socket logic.

## Stage 5 — `decision.js`: Stage-0 controller

Pure module (no DOM/camera/localStorage/ONNX). Exports `topK`, `top2`,
`requiredEvidenceVisible`, `decide`.

```js
export function decide({ box_conf, quality, probabilities, labels, s_ood, thresholds, traceBase = {} }) {
  validateDecisionInputs(box_conf, quality, probabilities, labels, s_ood, thresholds);

  if (box_conf < thresholds.box_min)
    return abstain("no_connector_found", ...);

  if (quality.q_low) {                                  // blur/glare/oblique/roi_scale only
    if (["blur","glare","low_roi_scale"].includes(quality.dominant))
      return abstain("need_better_focus", ...);
    return abstain("need_second_angle", ...);            // oblique / poor_face_visibility
  }

  if (s_ood >= thresholds.unsupported)
    return abstain("unsupported_connector", ...);

  const [c1, c2] = top2(probabilities, labels);

  if (!requiredEvidenceVisible(c1.class_name, quality, thresholds))
    return abstain("need_better_focus", ...);            // low center_res → focus/distance
                                                          // (NEVER ambiguous)
  const margin = c1.prob - c2.prob;
  if (c1.prob >= thresholds.accept && margin >= thresholds.margin)
    return accept(c1, c2, margin, ...);

  return abstain("ambiguous", { alternatives: topK(probabilities, labels, 5) }, ...);
}

export function requiredEvidenceVisible(_c1, quality, thresholds) {
  return quality.center_res >= thresholds.q.center_res_min;   // coarse only
}
```
- `requiredEvidenceVisible` ignores the class (coarse Rev-3 gate); **no pin/socket,
  no per-class logic, no probe heuristics.** Its failure → `need_better_focus`
  (center unreadable is a focus/distance problem), **never `ambiguous`** (test 7).
- `top2`/`topK` deterministic: higher prob first; equal prob → lower label index.
- Guidance map for `accepted / no_connector_found / need_better_focus /
  need_second_angle / unsupported_connector / ambiguous`.

**Acceptance:** the 13 tests pass; pure & unit-testable; no hardcoded thresholds; no
out-of-scope labels; required-visible failure never returns `ambiguous`.

## Stage 6 — `hardcase.js`: trace + local logging + export

`buildDecisionTrace(input)` produces **exactly** `asem_rev3_trace_v1` (schema in the
design doc §6.1) — using the **real** ONNX filenames from Stage 0, not idealized
ones. Failure tags: `blur, poor_center, scale_ambiguous, side_angle_needed,
family_confusion, gender_confusion, out_of_support, detector_missed`.

Logging: explicit opt-in; local-only; **no automatic upload**; log all abstained
(when consented) + sample accepted at `hardcase_accept_sample_rate`.

**Gap-closer D — Android export path:** a plain `<a download>` typically fails to
surface a file from an Android Capacitor WebView. Decide one:
1. Use the Capacitor Filesystem + Share plugin if already in the project (preferred —
   Chris gets a shareable file), **or**
2. Explicitly scope Rev-3 export as browser-only and document `adb pull` /
   `chrome://inspect` retrieval for Chris in the field package (Stage 12).
Record the decision in the trace/export README. Storage: IndexedDB (or existing
Capacitor FS) + manual export bundle (folder-like manifest or single JSON with data
URLs).

**Acceptance:** abstained logged; accepted sampled; export matches
`asem_rev3_trace_v1`; logging disable works; no upload code path exists; the chosen
Android export path is documented and demonstrably retrievable.

## Stage 7 — `app.js` wiring

Flow: `frame → detector(bbox,box_conf) → ROI → Q_t → classifier logits →
softmaxWithTemperature → computeSupportScore → decide() → render → maybe log`.

Preserve legacy `{class, confidence, bbox, …}` verbatim; add an `asem_rev3` object
alongside (`status, reason, user_guidance, top1, top2, margin, s_ood, quality,
thresholds_version, trace`). Render distinct UI for ACCEPT ("NAILED IT" + class +
confidence + margin) and each ABSTAIN reason with its guidance; debug panel shows
top-k, margin, `s_ood`, `Q_t`, `thresholds_version`.

**Acceptance:** browser runs; syncs to Android; legacy fields intact; `asem_rev3`
present; all states render; no out-of-scope strings.

## Stage 8 — Tests (13 acceptance + infra)

Confirm/select runner in Stage 0 (prefer `node:test` for pure ESM modules; else
Vitest/Jest, minimal deps). `npm` script runs them. Implement the **13** design
§9.1 cases verbatim, including: required-visible failure ≠ `ambiguous`;
deterministic `top2` ties; missing/malformed `thresholds.json` visible error;
`asem_rev3_trace_v1` conformance; no out-of-scope class anywhere
(`exports/web` + the `.dot` + tests); legacy fields preserved; `s_ood` sign
contract (`p_in_support=0.9 → s_ood≈0.1`, no `unsupported` trip).

**Acceptance:** all pass; single command; documented; non-flaky.

## Stage 9 — Canonical diagram revision

Revise `docs/fulldetector_active_visual_interrogation_system.dot` per design §8 (8
changes): ONNX-Web terminology (no TFLite/NNAPI), locked 10-class taxonomy, OOD
negative corpus as *future* node, Support Gate after ROI, Hard-Case Pool feedback
loop, `no_connector_found`/`low_box_confidence` first-class, probes in a *future Rev
5–6* cluster, `Q_t` latent node (solid → abstention router; dashed → future probe
down-weighting), SPRT scoped to the future loop. `fontname="Segoe UI"`; no
cluster-conflicting top-level `rank`. Render:
```
dot -Tsvg docs/fulldetector_active_visual_interrogation_system.dot -o docs/fulldetector_active_visual_interrogation_system.svg
dot -Tpng docs/fulldetector_active_visual_interrogation_system.dot -Gdpi=400 -o docs/fulldetector_active_visual_interrogation_system.png
```
**Acceptance:** renders without rankset/font warnings; no out-of-scope classes; no
TFLite/NNAPI; Rev-3 path visually distinct from future probes.

## Stage 10 — Browser smoke test

Startup loads thresholds/labels/ONNX with no console errors. Simulated/injected
inputs exercise every state: low `box_conf`→`no_connector_found`;
blur/glare/small-ROI→`need_better_focus`; oblique→`need_second_angle`; high `s_ood`
+ good quality→`unsupported_connector`; good all + high prob/margin→ACCEPT; good all
+ low margin→`ambiguous`. Hard-case consent + export validate.

## Stage 11 — Android sync + debug APK

`node scripts/copy-web.js` → `npx cap sync android` → build via the verified
PowerShell recipe (`$env:JAVA_HOME="C:\local\jdk21"; $env:ANDROID_HOME=
"C:\local\android-sdk"; .\gradlew.bat assembleDebug`). Smoke on device: models load,
detector+classifier run, Rev-3 routing active, **no forced hard class on
unsupported/stress when gates fail**, hard-case export retrievable per Stage-6
decision.

## Stage 12 — Field-validation package (Chris, CA)

**Depends on Stage 3 being complete (calibrated support gate).** Package: debug APK,
`thresholds.json`, `classifier_labels.json`, real model-bundle id, tester
instructions (incl. "Rev 3 abstains more on purpose"), hard-case export
instructions, supported-class list, expected-abstention list. Capture matrix:
supported ×10 (angles/light), stress (blur/glare/oblique/occlusion/far/cluttered),
unsupported (coax/F-type/BNC/N/UHF/RCA/random). Metrics: accepted/abstained counts,
accepted correct/wrong, abstention-by-reason, unsupported correctly-rejected /
wrongly-accepted, hardcase export count, crash/model-load-fail counts.

**Promotion gate:** risk-on-accepted < Rev 2; hotel-TV/coax-like → not a hard
supported class; bad captures → focus/angle guidance; hard cases exportable; no
legacy breakage; APK stable. *Not judged by accepted coverage.*

## Stage 13 — Reproducibility cleanup + docs

Pin the training notebook to a commit SHA; remove/replace `...`-elided cell sources;
record model-bundle id + data snapshot + export command. Update README/docs: Rev-3
summary, diagram image, ACCEPT-vs-ABSTAIN, supported classes, limitations, field
instructions, build/test commands, and the explicit
`s_ood` direction / `Q_t` / decision-priority / non-goals developer notes. Docs
contain no out-of-scope classes and no Rev-3 TFLite/NNAPI language.

---

## Critical prohibitions (for any implementer / Codex)

Do not: hardcode class labels or thresholds; add out-of-scope classes; infer
pin/socket in Rev 3; add TFLite/NNAPI/native paths; replace the working flat path;
remove or rename legacy fields; silently default missing thresholds; auto-upload
hard-case data; let `center_res` drive `q_low` (it is the required-visible gate
only); ship Stage 12 before Stage 3's measured percentiles are committed.

## Done definition

All true: `thresholds.json` validates; Stage-3 percentiles measured & committed;
`support.js`/`quality.js`/`decision.js`/`hardcase.js` exist and meet their
acceptance; `app.js` wired with legacy preserved + `asem_rev3` present; all UI
states render; 13 tests pass; diagram revised + SVG/PNG rendered; browser smoke
passes; Android debug APK builds; field package ready; docs state Rev-3 success =
risk-down / abstention-up / telemetry-up.
