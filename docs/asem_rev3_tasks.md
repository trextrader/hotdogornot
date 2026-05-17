# ASEM Rev 3 Task Checklist — Safety & Evidence-Routing Release

- Date: 2026-05-17
- Status: Ready for implementation
- Companion plan: `docs/asem_rev3_implementation.md`
- Source design: `docs/superpowers/specs/2026-05-17-asem-rev3-safety-evidence-routing-design.md`

Legend: `[ ]` not started · `[/]` in progress · `[x]` done · `[!]` blocked · `[~]` needs review

Stage numbers are the **canonical map** shared with the plan (plan §3). Update this
file as each item completes.

---

## Stage 0 — Preflight audit & baseline preservation

- [ ] Confirm canonical source `exports/web/`; sync via `scripts/copy-web.js` + `npx cap sync android`; runtime = Capacitor WebView + ONNX Runtime Web
- [ ] Record the **real** detector/classifier ONNX filenames on disk (expected `detector.onnx` / `classifier.onnx`) — used verbatim by the Stage-6 trace
- [ ] Confirm classifier output is logits (preferred) or probabilities (document fallback)
- [ ] Locate prediction + UI code in `app.js`; enumerate legacy fields (`class`, `confidence`, `bbox`, `/predict`-style)
- [ ] **Gap A:** audit `scripts/copy-web.js`; extend it so `exports/web/asem/**` + `thresholds.json` are copied into `www`
- [ ] **Gap B:** determine WebView JS load scheme; decide `asem/*.js` module format that is both browser-loadable and Node-test importable; record it
- [ ] Confirm current debug APK still builds (unchanged)
- [ ] Commit baseline note (no functional change)

## Stage 1 — thresholds.json + fail-loud loader

- [ ] Create `exports/web/thresholds.json` with the Stage-1 content (incl. `support_calibration` percentiles = `null`)
- [ ] Implement loader: validate top-level + `q` keys; finite numerics; `calibration_T>0`; `hardcase_accept_sample_rate∈[0,1]`
- [ ] Missing/malformed → visible startup error; **no silent defaults**
- [ ] Remove every hardcoded decision constant from `app.js` (`box_min/accept/margin/unsupported/q.*`)

## Stage 2 — support.js: calibration + logit-energy s_ood

- [ ] Create `exports/web/asem/support.js`
- [ ] `softmaxWithTemperature(logits,T)` — stable; throws on `T<=0`/non-finite/empty
- [ ] `logsumexp(values)` — max-subtraction stable
- [ ] `energyScoreFromLogits(logits,T)` = `-T*logsumexp(logits/T)`
- [ ] `normalizeEnergyToSOod(energy,calibration)` — minmax→`clamp01`; **throws if percentiles null/missing**
- [ ] `computeSupportScore(...)` — logit-energy baseline; never requires embedding; returns `{s_ood,method:"energy",energy}`
- [ ] Sign contract comment + behavior: larger `s_ood` = more unsupported

## Stage 3 — Support-gate calibration measurement (gap-closer, hard dependency)

- [ ] Create `scripts/fit_support_energy.py`
- [ ] Run current classifier ONNX over `data/labeled/embedder/**`; collect logit-energy per image
- [ ] Compute in-support energy `p05`/`p95`; choose documented `unsupported` operating point
- [ ] Write measured percentiles into `exports/web/thresholds.json` (non-null)
- [ ] Emit auditable report to `reports/` (energy stats, chosen percentiles, curated-set `s_ood` min/median/max)
- [ ] Deterministic re-run verified; values committed (**must precede Stage 12**)

## Stage 4 — quality.js: Q_t estimator

- [ ] Create `exports/web/asem/quality.js`
- [ ] `roi_scale = bbox_area/frame_area`; fail `< q.roi_scale_min` → dominant `low_roi_scale`
- [ ] `blur_var` (variance-of-Laplacian); fail `< q.blur_var_min` → `blur`
- [ ] `glare_frac` (near-saturated); fail `> q.glare_frac_max` → `glare`
- [ ] `oblique_proxy`; fail `> q.oblique_proxy_max` → `oblique`/`poor_face_visibility`
- [ ] `center_res` computed **but does NOT set `q_low`** (Gap C)
- [ ] `q_low` driven only by blur/glare/oblique/roi_scale; dominant priority `blur>glare>oblique>low_roi_scale`
- [ ] No pin/socket inference anywhere in `quality.js`
- [ ] `estimateQuality(...)` returns the full struct

## Stage 5 — decision.js: Stage-0 controller

- [ ] Create `exports/web/asem/decision.js` (pure: no DOM/camera/localStorage/ONNX)
- [ ] `topK`/`top2` deterministic (prob desc; tie → lower label index); validate lengths/finite
- [ ] `requiredEvidenceVisible(_c1,quality,thresholds)` = `center_res >= q.center_res_min`; **no class/pin/socket logic**
- [ ] `decide()` exact policy: box → q_low(focus/angle) → support → required-visible(→`need_better_focus`, never `ambiguous`) → accept(conf&margin) → `ambiguous`
- [ ] Guidance map for all 6 reasons
- [ ] ACCEPT/ABSTAIN payloads per design output contract; trace attached
- [ ] No hardcoded thresholds; no out-of-scope labels

## Stage 6 — hardcase.js: trace + local logging + export

- [ ] Create `exports/web/asem/hardcase.js`
- [ ] `buildDecisionTrace()` emits exactly `asem_rev3_trace_v1` using **real** ONNX filenames
- [ ] Failure tags: blur, poor_center, scale_ambiguous, side_angle_needed, family_confusion, gender_confusion, out_of_support, detector_missed
- [ ] Opt-in consent; local-only; disable works; **no auto-upload code path**
- [ ] Log all abstained (consented) + sample accepted at `hardcase_accept_sample_rate`
- [ ] **Gap D:** decide Android export path (Capacitor Filesystem/Share **or** documented browser-only + adb/inspect); document it
- [ ] Storage (IndexedDB / Capacitor FS) + manual export bundle (manifest + traces + images/data-URLs)

## Stage 7 — app.js wiring

- [ ] Import support/quality/decision/hardcase
- [ ] Startup: load+validate thresholds (block inference if invalid); load labels from `classifier_labels.json` (10, locked, order from file)
- [ ] Flow: detector → ROI → `Q_t` → classifier logits → temp softmax → `s_ood` → `decide()` → render → maybe log
- [ ] Preserve legacy fields verbatim; add `asem_rev3` object alongside
- [ ] Render distinct UI for ACCEPT + all 5 abstentions; debug panel: top-k, margin, `s_ood`, `Q_t`, `thresholds_version`
- [ ] No out-of-scope class strings

## Stage 8 — Tests (13 acceptance + infra)

- [ ] Select/confirm runner (prefer `node:test`); add npm script; document command
- [ ] T1 `no_connector_found` (box_conf<box_min)
- [ ] T2 `need_better_focus` (blur/glare/low center)
- [ ] T3 `need_second_angle` (oblique/face)
- [ ] T4 `unsupported_connector` (s_ood≥unsupported, Q_t OK)
- [ ] T5 ACCEPT only when all gates pass
- [ ] T6 `ambiguous` (support+quality OK, prob/margin fail)
- [ ] T7 required-visible failure → focus/angle, **never** `ambiguous`
- [ ] T8 deterministic `top2` ties
- [ ] T9 missing/malformed `thresholds.json` visible error (no silent defaults)
- [ ] T10 trace conforms to `asem_rev3_trace_v1`
- [ ] T11 no out-of-scope class in `exports/web` + `.dot` + tests
- [ ] T12 legacy `/predict`-style fields preserved
- [ ] T13 `s_ood` sign contract (`p_in_support=0.9 → s_ood≈0.1`, no `unsupported` trip)
- [ ] All pass; single command; non-flaky

## Stage 9 — Canonical diagram revision

- [ ] `fontname="Segoe UI"`; remove cluster-conflicting top-level `rank`
- [ ] Remove TFLite/NNAPI/`*.tflite`; add ONNX Runtime Web/WASM/WebGL/`*.onnx`
- [ ] Replace taxonomy examples with the locked 10 classes; remove RP-SMA/right-angle/bulkhead/cable-end/board-mount
- [ ] Add Rev-3 path nodes (camera→detector→ROI→`Q_t`→classifier→temp→support→Stage-0 controller→ACCEPT/ABSTAIN)
- [ ] Add 5 abstention outputs incl. `no_connector_found`/`low_box_confidence`
- [ ] Add Hard-Case Pool feedback loop (capture→consent/storage→tag→export→labeling→retrain)
- [ ] Probes → future Rev 5–6 cluster; SPRT/Λ_t scoped to future loop; dashed `Q_t`→future-probe edge
- [ ] Render SVG + PNG (dpi 400); no rankset/font warnings; no out-of-scope/no TFLite

## Stage 10 — Browser smoke test

- [ ] Startup: thresholds+labels+ONNX load, no console errors
- [ ] Each state reproduced via simulated/injected values (all 6 outcomes)
- [ ] UI states distinct; debug panel shows `Q_t`/`s_ood`/top-k/thresholds version
- [ ] Hard-case consent + export validate

## Stage 11 — Android sync + debug APK

- [ ] `node scripts/copy-web.js` → `npx cap sync android`
- [ ] Build via verified PowerShell recipe (`$env:JAVA_HOME=C:\local\jdk21`, `ANDROID_HOME=C:\local\android-sdk`, `.\gradlew.bat assembleDebug`)
- [ ] On-device smoke: models load; detector+classifier run; Rev-3 routing active; **no forced hard class** on unsupported/stress; hard-case export retrievable per Gap D
- [ ] Existing debug path preserved

## Stage 12 — Field-validation package (Chris, CA) — *requires Stage 3 done*

- [ ] Package: APK, thresholds.json, classifier_labels.json, model-bundle id, tester + export instructions, supported-class list, expected-abstention list
- [ ] Tester instructions state "Rev 3 abstains more on purpose (safety)"
- [ ] Capture matrix: supported ×10 (angles/light) · stress · unsupported (coax/F/BNC/N/UHF/RCA/random)
- [ ] Metrics sheet: accepted/abstained, accepted correct/wrong, abstention_by_reason, unsupported correctly/wrongly, hardcase exports, crash/model-load-fail
- [ ] Promotion gate evaluated (risk-on-accepted↓; unsupported not hard-answered; bad→guidance; exportable; no breakage; APK stable) — not judged by coverage

## Stage 13 — Reproducibility cleanup + docs

- [ ] Pin training notebook to commit SHA; remove `...`-elided cell sources; record model-bundle id + data snapshot + export command
- [ ] README/docs: Rev-3 summary, diagram image, ACCEPT-vs-ABSTAIN, supported classes, limitations, field + build/test commands
- [ ] Developer notes: `s_ood` direction, `Q_t`, decision priority, hard-case export, non-goals
- [ ] Docs contain no out-of-scope classes and no Rev-3 TFLite/NNAPI language

---

## Codex handoff prompt (use verbatim if handing off)

```
You are implementing ASEM Rev 3 for the hotdogornot repo.
Read and follow exactly:
- docs/superpowers/specs/2026-05-17-asem-rev3-safety-evidence-routing-design.md
- docs/asem_rev3_implementation.md
- docs/asem_rev3_tasks.md

Goal: implement the Rev-3 safety/evidence-routing release while preserving the
existing flat detector/classifier APK path.

Hard scope: Capacitor WebView + ONNX Runtime Web only. No TFLite/NNAPI/native
rewrite. No semantic probes. No multi-head refiner. No ArUco/scale. No new
connector classes. No pin/socket inference. Do not remove/rename legacy fields.

Required files: exports/web/thresholds.json; exports/web/asem/{support,quality,
decision,hardcase}.js; updated exports/web/app.js; scripts/fit_support_energy.py;
revised+rendered docs/fulldetector_active_visual_interrogation_system.{dot,svg,png}.

Decision behavior: no_connector_found (box_conf<box_min); need_better_focus
(blur/glare/low-center/small-ROI); need_second_angle (oblique/face);
unsupported_connector (quality OK & s_ood>=unsupported); ACCEPT (all gates pass);
ambiguous (quality/support OK but conf/margin fail).

Hard contracts: s_ood larger=more unsupported; thresholds only from
thresholds.json; missing/malformed thresholds must visibly error; labels from
classifier_labels.json; legacy fields preserved; hard-case logging local-only +
opt-in; no auto-upload; center_res does NOT drive q_low (required-visible only);
Stage 3 measured energy percentiles committed before Stage 12.

Add the 13 acceptance tests. Work in small commits per stage. Update
docs/asem_rev3_tasks.md as each item completes.
```

## Completion checklist

- [ ] thresholds.json validates · Stage-3 percentiles measured & committed
- [ ] support/quality/decision/hardcase modules meet acceptance
- [ ] app.js wired; legacy preserved; `asem_rev3` present; all UI states render
- [ ] 13 tests pass · diagram revised + SVG/PNG · browser smoke passes
- [ ] Android debug APK builds · field package ready
- [ ] Docs state Rev-3 success = risk-down / abstention-up / telemetry-up
