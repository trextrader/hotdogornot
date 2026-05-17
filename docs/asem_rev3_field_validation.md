# ASEM Rev 3 Field Validation Package

Status: prepared, pending cloud support-energy calibration output and owner-built APK.

## Bundle Inputs

- APK: owner-built locally after calibration values are committed.
- Web bundle source: `exports/web/`
- Thresholds: `exports/web/thresholds.json`
- Labels: `exports/web/models/classifier_labels.json`
- Model bundle id: `rev3_web_bundle`
- Detector model: `models/detector.onnx`
- Classifier model: `models/classifier.onnx`

## Required Build Commands

Run from the repo root and mobile project:

```powershell
node scripts/copy-web.js
Set-Location exports/mobile
npx cap sync android
Set-Location android
.\gradlew.bat assembleDebug
```

JDK 21 is expected on PATH. The project owner performs the final APK build.

## Calibration Gate

Before field validation, run support-energy calibration in Colab/Kaggle T4:

```powershell
python scripts/fit_support_energy.py --write-thresholds --providers CUDAExecutionProvider CPUExecutionProvider
```

The field package is not ready until `energy_in_support_p05` and
`energy_in_support_p95` in `exports/web/thresholds.json` are non-null and the
calibration report is committed under `reports/`.

## Supported Classes

1. `1.85mm-M`
2. `1.85mm-F`
3. `2.4mm-M`
4. `2.4mm-F`
5. `2.92mm-M`
6. `2.92mm-F`
7. `3.5mm-M`
8. `3.5mm-F`
9. `SMA-F`
10. `SMA-M`

Rev 3 abstains more on purpose. Success is lower risk on accepted answers,
more abstention when evidence is insufficient, and useful hard-case exports.

## Capture Matrix

- Supported: each of the 10 classes under varied angle, distance, and lighting.
- Stress: blur, glare, oblique angle, partial occlusion, far distance, clutter.
- Unsupported: coax/F-type, BNC, N, UHF, RCA, and random non-connector objects.

## Expected Abstentions

- `no_connector_found`: detector confidence below `thresholds.box_min`.
- `need_better_focus`: blur, glare, small ROI, or unreadable center region.
- `need_second_angle`: oblique capture or poor face visibility.
- `unsupported_connector`: good capture but `s_ood >= thresholds.unsupported`.
- `ambiguous`: quality/support OK but probability or margin is insufficient.

## Metrics Sheet Columns

- capture_id
- supported_class_expected
- accepted_or_abstained
- predicted_class
- accepted_correct
- abstention_reason
- top1_prob
- top2_prob
- margin
- s_ood
- q_dominant
- unsupported_correctly_rejected
- unsupported_wrongly_accepted
- hardcase_exported
- crash_or_model_load_failure
- notes

## Hard-Case Export

Rev 3 uses browser/WebView JSON export only. Native Capacitor Filesystem/Share
export is not implemented in Rev 3. If Android WebView download behavior does
not expose the file directly, retrieve the export through WebView debugging,
device file sharing, or adb and record the retrieval method in the returned
field notes.

## Promotion Gate

Promotion is not judged by coverage alone. Validate:

- risk on accepted answers is lower than Rev 2;
- unsupported/stress captures are not forced into a supported class;
- bad captures produce focus or angle guidance;
- hard cases are exportable;
- legacy output remains present;
- APK is stable on the test device.
