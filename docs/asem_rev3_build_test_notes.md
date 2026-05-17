# ASEM Rev 3 Build, Test, and Repro Notes

## Local Test Command

```powershell
npm test
```

This runs the Node test suite for Rev-3 thresholds, support scoring, Q_t,
decision routing, hard-case traces, app contracts, and out-of-scope scans.

## Browser Smoke Command

Serve `exports/web` locally, then open the page:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory exports/web
```

Expected startup state after thresholds, labels, and ONNX models load:

```text
Ready (10 classes)
```

## Android Sync and Owner Build

```powershell
node scripts/copy-web.js
Set-Location exports/mobile
npx cap sync android
Set-Location android
.\gradlew.bat assembleDebug
```

JDK 21 is installed and expected on PATH. The project owner performs the final
APK build and device smoke test.

## Developer Contracts

- `s_ood` is unsupported risk: larger means more likely unsupported.
- `Q_t.q_low` is driven by blur, glare, ROI scale, and oblique/face visibility.
- `center_res` is computed by `quality.js` but only gates
  `requiredEvidenceVisible()` in `decision.js`.
- Rev 3 does not infer pin versus socket.
- Rev 3 does not add connector classes.
- Rev 3 keeps Capacitor WebView plus ONNX Runtime Web.
- Hard-case logging is local-only, opt-in, and manually exported as JSON.
- Missing or malformed thresholds fail visibly; app code must not silently
  default decision thresholds.

## Model Bundle Record

- model_bundle_id: `rev3_web_bundle`
- detector: `exports/web/models/detector.onnx`
- classifier: `exports/web/models/classifier.onnx`
- labels: `exports/web/models/classifier_labels.json`
- thresholds: `exports/web/thresholds.json`

## Reproducibility Status

Support-energy calibration is prepared but not locally executed over the full
dataset. Run `scripts/fit_support_energy.py` in Colab/Kaggle T4 and commit the
resulting non-null percentiles plus report before field validation.

Training notebook commit pinning and cleanup of historical elided cells remain
a review item outside the local Rev-3 web wiring changes.
