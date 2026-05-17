# ASEM Rev 3 Hard-Case Export

Rev 3 hard-case logging is local-only and opt-in. The app stores consented traces in
IndexedDB and exports a manual JSON bundle with schema
`asem_rev3_hardcase_export_v1`; it does not upload traces automatically.

Android export decision for Rev 3: browser/WebView JSON export only. The current
Capacitor project includes Camera but not Filesystem or Share, so this release does
not add native export plugins. For field retrieval, use WebView debugging
(`chrome://inspect`) or `adb`/device file sharing after the manual export is created.

The trace records use schema `asem_rev3_trace_v1` and real model bundle names:
`models/detector.onnx` and `models/classifier.onnx`.

Run cloud calibration before field validation:

```powershell
python scripts/fit_support_energy.py --write-thresholds
```

Use Colab/Kaggle T4 or equivalent for the full curated-set run.
