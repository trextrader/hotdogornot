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

- [x] Confirm canonical source `exports/web/`; sync via `scripts/copy-web.js` + `npx cap sync android`; runtime = Capacitor WebView + ONNX Runtime Web
  - Note: canonical source is `exports/web/`; Capacitor project lives in `exports/mobile/`; repo-root `scripts/copy-web.js` mirrors into `exports/mobile/www/`.
- [x] Record the **real** detector/classifier ONNX filenames on disk (expected `detector.onnx` / `classifier.onnx`) — used verbatim by the Stage-6 trace
  - Note: real filenames are `models/detector.onnx` and `models/classifier.onnx`.
- [x] Confirm classifier output is logits (preferred) or probabilities (document fallback)
  - Note: classifier ONNX output is `logits` with shape `[1,10]`.
- [x] Locate prediction + UI code in `app.js`; enumerate legacy fields (`class`, `confidence`, `bbox`, `/predict`-style)
  - Note: current browser output is UI-only; legacy prediction fields to preserve are class label/confidence/top-k plus bbox coordinates and detector box confidence from `renderResults()`/`classifyDetection()`.
- [x] **Gap A:** audit `scripts/copy-web.js`; extend it so `exports/web/asem/**` + `thresholds.json` are copied into `www`
  - Note: repo-root recursive copy script added and verified with `node scripts/copy-web.js`.
- [x] **Gap B:** determine WebView JS load scheme; decide `asem/*.js` module format that is both browser-loadable and Node-test importable; record it
  - Note: existing WebView uses plain scripts. Rev-3 `asem/*.js` will use a small UMD/CommonJS wrapper so app.js can read `window.Asem*` globals and Node tests can `require()` the same files.
- [~] Confirm current debug APK still builds (unchanged)
  - Needs review: final/debug APK build is left for the project owner per Android build constraint; Rev-3 prep will document the local command.
- [x] Commit baseline note (no functional change)
  - Note: local stage commits are being created. Push to `origin/master` is blocked by GitHub 403 for the current credentials; pushes to `trextrader/master` succeed.

## Stage 1 — thresholds.json + fail-loud loader

- [x] Create `exports/web/thresholds.json` with the Stage-1 content (incl. `support_calibration` percentiles = `null`)
- [x] Implement loader: validate top-level + `q` keys; finite numerics; `calibration_T>0`; `hardcase_accept_sample_rate∈[0,1]`
  - Note: loader lives in `exports/web/asem/thresholds.js`; support percentiles may be both `null` until Stage 3, but malformed/mismatched values fail.
- [x] Missing/malformed → visible startup error; **no silent defaults**
  - Note: `app.js` blocks startup on threshold load/validation failure and shows the error in the model status badge.
- [x] Remove every hardcoded decision constant from `app.js` (`box_min/accept/margin/unsupported/q.*`)
  - Note: detector filtering now uses validated `THRESHOLDS.box_min`; no Rev-3 decision constants are hardcoded in `app.js`.
  - Check: `CONF_THRESHOLD` was removed from `exports/web/app.js`; detector confidence gating comes from `thresholds.json`.

## Stage 2 — support.js: calibration + logit-energy s_ood

- [x] Create `exports/web/asem/support.js`
- [x] `softmaxWithTemperature(logits,T)` — stable; throws on `T<=0`/non-finite/empty
- [x] `logsumexp(values)` — max-subtraction stable
- [x] `energyScoreFromLogits(logits,T)` = `-T*logsumexp(logits/T)`
- [x] `normalizeEnergyToSOod(energy,calibration)` — minmax→`clamp01`; **throws if percentiles null/missing**
  - Check: direct Node invocation with `energy_in_support_p05/p95:null` throws before scoring proceeds.
- [x] `computeSupportScore(...)` — logit-energy baseline; never requires embedding; returns `{s_ood,method:"energy",energy}`
- [x] Sign contract comment + behavior: larger `s_ood` = more unsupported
  - Note: `sOodFromInSupportProbability(0.9)` returns `0.1`; decision code must reject only when `s_ood >= thresholds.unsupported`.

## Stage 3 — Support-gate calibration measurement (gap-closer, hard dependency)

- [x] Create `scripts/fit_support_energy.py`
  - Note: script is Colab/Kaggle-ready with ONNX Runtime provider selection, locked-label validation, deterministic sorted image traversal, threshold update option, and JSON report output.
- [~] Run current classifier ONNX over `data/labeled/embedder/**`; collect logit-energy per image
  - Needs review: full curated-set inference is a cloud/T4 calibration step per model-work constraint; do not run heavy dataset inference locally.
- [~] Compute in-support energy `p05`/`p95`; choose documented `unsupported` operating point
  - Needs review: requires cloud script output before non-null calibrated values can be accepted.
- [~] Write measured percentiles into `exports/web/thresholds.json` (non-null)
  - Needs review: `thresholds.json` intentionally still has `null` support percentiles until cloud calibration output is applied.
- [~] Emit auditable report to `reports/` (energy stats, chosen percentiles, curated-set `s_ood` min/median/max)
  - Needs review: report is produced by `python scripts/fit_support_energy.py --write-thresholds` in Colab/Kaggle.
- [~] Deterministic re-run verified; values committed (**must precede Stage 12**)
  - Needs review: script compilation/help path verified locally; deterministic full rerun waits on cloud calibration.

## Stage 4 — quality.js: Q_t estimator

- [x] Create `exports/web/asem/quality.js`
- [x] `roi_scale = bbox_area/frame_area`; fail `< q.roi_scale_min` → dominant `low_roi_scale`
- [x] `blur_var` (variance-of-Laplacian); fail `< q.blur_var_min` → `blur`
- [x] `glare_frac` (near-saturated); fail `> q.glare_frac_max` → `glare`
- [x] `oblique_proxy`; fail `> q.oblique_proxy_max` → `oblique`/`poor_face_visibility`
- [x] `center_res` computed **but does NOT set `q_low`** (Gap C)
  - Check: `node --test tests/asem/quality.test.js` verifies low `center_res` is reported while `q_low=false` when other quality gates pass.
- [x] `q_low` driven only by blur/glare/oblique/roi_scale; dominant priority `blur>glare>oblique>low_roi_scale`
- [x] No pin/socket inference anywhere in `quality.js`
- [x] `estimateQuality(...)` returns the full struct
  - Check: direct Node tests cover acceptable ROI, blur, glare, small ROI, oblique, low-center-resolution, and dominant-priority behavior.

## Stage 5 — decision.js: Stage-0 controller

- [x] Create `exports/web/asem/decision.js` (pure: no DOM/camera/localStorage/ONNX)
- [x] `topK`/`top2` deterministic (prob desc; tie → lower label index); validate lengths/finite
- [x] `requiredEvidenceVisible(_c1,quality,thresholds)` = `center_res >= q.center_res_min`; **no class/pin/socket logic**
- [x] `decide()` exact policy: box → q_low(focus/angle) → support → required-visible(→`need_better_focus`, never `ambiguous`) → accept(conf&margin) → `ambiguous`
  - Check: `node --test tests/asem/decision.test.js` covers all Stage-0 decision branches and required-visible failure.
- [x] Guidance map for all 6 reasons
- [x] ACCEPT/ABSTAIN payloads per design output contract; trace attached
- [x] No hardcoded thresholds; no out-of-scope labels
  - Check: decision tests scan `decision.js` and decision fixtures for out-of-scope labels and browser/ONNX dependencies.

## Stage 6 — hardcase.js: trace + local logging + export

- [x] Create `exports/web/asem/hardcase.js`
- [x] `buildDecisionTrace()` emits exactly `asem_rev3_trace_v1` using **real** ONNX filenames
  - Note: trace defaults to `models/detector.onnx` and `models/classifier.onnx` from Stage 0.
- [x] Failure tags: blur, poor_center, scale_ambiguous, side_angle_needed, family_confusion, gender_confusion, out_of_support, detector_missed
- [x] Opt-in consent; local-only; disable works; **no auto-upload code path**
  - Check: `tests/asem/hardcase_schema.test.js` covers consent-required local save and scans for network upload primitives.
- [x] Log all abstained (consented) + sample accepted at `hardcase_accept_sample_rate`
- [~] **Gap D:** decide Android export path (Capacitor Filesystem/Share **or** documented browser-only + adb/inspect); document it
  - Needs review: Android/native export via Capacitor Filesystem/Share is not implemented in Rev 3. Rev-3 hard-case export is browser/WebView JSON export only; field package must validate/download via WebView debugging, device file sharing, or adb retrieval.
- [x] Storage (IndexedDB / Capacitor FS) + manual export bundle (manifest + traces + images/data-URLs)
  - Note: browser IndexedDB store and manual local JSON export helper are implemented; no automatic upload path exists.

## Stage 7 — app.js wiring

- [x] Import support/quality/decision/hardcase
- [x] Startup: load+validate thresholds (block inference if invalid); load labels from `classifier_labels.json` (10, locked, order from file)
  - Check: headless Chrome loaded `index.html` from local HTTP server and reached `Ready (10 classes)`.
- [x] Flow: detector → ROI → `Q_t` → classifier logits → temp softmax → `s_ood` → `decide()` → render → maybe log
  - Note: support scoring is wired to fail loudly until Stage 3 writes non-null energy percentiles.
- [x] Preserve legacy fields verbatim; add `asem_rev3` object alongside
  - Note: legacy `top`/`ranked` UI fields remain, and `legacy_output` keeps `class`, `confidence`, `bbox`, and `top_k` alongside `asem_rev3`.
- [x] Render distinct UI for ACCEPT + all 5 abstentions; debug panel: top-k, margin, `s_ood`, `Q_t`, `thresholds_version`
- [x] No out-of-scope class strings
  - Check: app wiring uses labels from `classifier_labels.json`; no class labels are hardcoded in `app.js`.

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
