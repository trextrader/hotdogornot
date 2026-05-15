# Mobile App + Model Exports

This directory contains two related things:

1. **Capacitor mobile app** (`android/`, `ios/`, `www/`, `dist/`) — the
   on-device application that runs the connector identification pipeline
   in a WebView using ONNX Runtime Web.
2. **Exported model artifacts** — ONNX/TFLite/Core ML files produced by
   `training/rfconnectorai/export/export_mobile.py` and consumed by the
   mobile app (and by the future server pipeline).

---

## Pre-built debug APK

A pre-built debug APK is checked into `dist/app-debug.apk` (~87 MB) for
fast on-device testing. Sideload it on an Android phone:

1. Copy `dist/app-debug.apk` to the phone (USB, Drive, email, etc.) or
   run `adb install -r dist/app-debug.apk` from this directory.
2. On the phone, allow "Install unknown apps" for the source you used.
3. Open **RF Connector AI** from the launcher.

> Note: This APK is unsigned (debug build) and packs the ONNX models as
> assets. For Play Store distribution we would switch to a release build,
> sign with a real keystore, and move the models out of `assets/` to a
> downloaded model store. See "Release builds" below.

---

## Build the Android APK from source

### Prerequisites

| Tool | Version tested | Notes |
|---|---|---|
| Node.js | 18+ | Only needed if you re-run `npm install` or `npx cap sync` |
| JDK | **21 LTS** | Capacitor 7.x **requires JDK 21**. JDK 17 will fail with `error: invalid source release: 21`. |
| Android SDK | platforms 34 + 35, build-tools 34.0.0 + 35.0.0, platform-tools | Must live in a **writable** location (not `Program Files`) |
| `ANDROID_HOME` env var | pointing at the SDK root | |
| `JAVA_HOME` env var | pointing at the JDK root | |

#### Why "writable SDK location" matters

If your SDK is in `C:\Program Files (x86)\Android\android-sdk` (the
classic Android Studio default), Gradle cannot write license-acceptance
markers and cannot auto-install missing build-tools. You will see:

```
Failed to install the following Android SDK packages as some licences
have not been accepted.
     build-tools;34.0.0 Android SDK Build-Tools 34
```

Fix: copy or relocate the SDK to a writable path, e.g.
`C:\local\android-sdk`, and point `local.properties` there.

### 1. Install JDK 21 (portable, no admin)

Pick any path on a writable drive — the examples below assume
`C:\local\jdk21`.

```powershell
# Windows PowerShell
$url = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.11%2B10/OpenJDK21U-jdk_x64_windows_hotspot_21.0.11_10.zip"
Invoke-WebRequest $url -OutFile C:\local\jdk21.zip
Expand-Archive C:\local\jdk21.zip -DestinationPath C:\local\jdk21_tmp
Move-Item C:\local\jdk21_tmp\jdk-21.* C:\local\jdk21
Remove-Item C:\local\jdk21.zip, C:\local\jdk21_tmp
C:\local\jdk21\bin\java.exe -version   # sanity check
```

```bash
# macOS / Linux (Homebrew)
brew install --cask temurin@21
/usr/libexec/java_home -v 21   # use as JAVA_HOME
```

### 2. Set up the Android SDK in a writable location

If you already have Android Studio, copy its SDK to a writable path:

```powershell
# Windows — copy the Android Studio SDK out of Program Files
Copy-Item "C:\Program Files (x86)\Android\android-sdk" C:\local\android-sdk -Recurse
```

If you don't have a SDK yet, download Google's command-line tools and
install the required components:

```bash
# bash on any platform; paths below assume Windows
export JAVA_HOME="C:/local/jdk21"
export PATH="$JAVA_HOME/bin:$PATH"
"C:/local/android-sdk/cmdline-tools/12.0/bin/sdkmanager.bat" \
    --sdk_root="C:/local/android-sdk" \
    "platform-tools" "platforms;android-35" \
    "build-tools;34.0.0" "build-tools;35.0.0"
```

### 3. Accept SDK licenses

```bash
yes | "C:/local/android-sdk/cmdline-tools/12.0/bin/sdkmanager.bat" \
    --sdk_root="C:/local/android-sdk" --licenses
```

You should see `All SDK package licenses accepted`. After this,
`C:/local/android-sdk/licenses/` will contain the license-acceptance
marker files.

### 4. Create `android/local.properties`

Capacitor's Android project reads the SDK path from `local.properties`
(which is `.gitignore`d on purpose). Create it as:

```properties
## Path to Android SDK.
sdk.dir=C\:\\local\\android-sdk
```

> Use double-backslashes on Windows. The leading `C\:` is also escaped.

### 5. Build the debug APK

From this directory (`exports/mobile/`):

```bash
cd android
export JAVA_HOME="C:/local/jdk21"
export ANDROID_HOME="C:/local/android-sdk"
export PATH="$JAVA_HOME/bin:$PATH"
./gradlew.bat assembleDebug    # Windows
./gradlew assembleDebug        # macOS / Linux
```

First run downloads `build-tools;34.0.0` (transitively requested by
Capacitor) plus Gradle plugin deps — expect 5-10 minutes. Incremental
rebuilds (e.g. after editing the manifest or a JS file) take ~30-60 s.

Output: `android/app/build/outputs/apk/debug/app-debug.apk` (~87 MB).

### 6. Install on a phone over USB

1. On the phone, enable **Developer options** (Settings → About phone →
   tap **Build number** 7 times).
2. Enable **USB debugging** in Developer options.
3. Plug in USB; on the phone tap **Allow** for the RSA prompt.

```bash
"C:/local/android-sdk/platform-tools/adb.exe" devices            # should list the phone
"C:/local/android-sdk/platform-tools/adb.exe" install -r \
    android/app/build/outputs/apk/debug/app-debug.apk
"C:/local/android-sdk/platform-tools/adb.exe" shell monkey \
    -p com.rfconnector.ai -c android.intent.category.LAUNCHER 1   # launch
```

### 7. View runtime logs

```bash
"C:/local/android-sdk/platform-tools/adb.exe" logcat \
    --pid=$(adb shell pidof com.rfconnector.ai) \
    Capacitor:V Capacitor/Console:V Capacitor/Plugin:V "*:S"
```

For Chrome DevTools inspection, with the app open, go to
`chrome://inspect/#devices` in desktop Chrome — `android:` declares
`webContentsDebuggingEnabled = true` in this project's
`capacitor.config.ts`.

---

## Build the iOS app

iOS native build requires **macOS** with Xcode 15+ and CocoaPods. From
this directory on a Mac:

```bash
npm install
npx cap sync ios
cd ios/App
pod install
open App.xcworkspace
```

In Xcode: select your team in Signing & Capabilities, plug in an iPhone,
hit **Run**.

For App Store / TestFlight distribution, use a paid Apple Developer
account, archive from Xcode, and upload via Transporter.

> Note: no Mac is required to edit the JS/HTML — only to compile the iOS
> binary. The same `www/` folder backs both Android and iOS.

---

## Permissions declared

`android/app/src/main/AndroidManifest.xml` declares:

- `android.permission.INTERNET` — for any future server fallback.
- `android.permission.CAMERA` — needed because the app uses
  `navigator.mediaDevices.getUserMedia()` for live camera preview.
- `<uses-feature android:name="android.hardware.camera" required="false">`
  — so the app installs on camera-less devices too.

Capacitor's `BridgeWebChromeClient` grants `getUserMedia` automatically
to the WebView when the app itself holds `CAMERA`. The Android system
permission prompt fires the first time the user taps **Use Camera**.

If you add microphone / mic input later, also declare
`RECORD_AUDIO` in the manifest.

---

## Release builds

This project currently ships a **debug** APK only. Before publishing:

1. Generate a release keystore and store credentials outside the repo.
2. Add `signingConfigs.release { ... }` to `android/app/build.gradle`.
3. Run `./gradlew assembleRelease` (or `bundleRelease` for an AAB).
4. Enable R8 minification — `minifyEnabled true` — and adjust
   `proguard-rules.pro` for ONNX Runtime Web.
5. Move the 87 MB of ONNX assets out of the bundle. Options:
   - Download from a CDN on first launch with a progress UI.
   - Switch to Play Asset Delivery (install-time/conditional asset
     packs) so the install size stays small.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `JAVA_HOME is not set` from `sdkmanager.bat` | Export `JAVA_HOME` before invoking it (Windows env-var prefix in bash does *not* propagate into batch scripts; set it explicitly). |
| `error: invalid source release: 21` | You are on JDK 17. Install JDK 21 and re-export `JAVA_HOME`. |
| `Failed to install build-tools;34.0.0 ... licences not accepted` | SDK is in a read-only path. Copy to a writable location and re-run `sdkmanager --licenses` there. |
| `Probably the SDK is read-only` warnings in build log | Same root cause — relocate SDK. |
| Camera button → "Camera access denied or not available" | Manifest is missing `CAMERA` permission, *or* the system prompt was denied. Reinstall and accept the prompt, or toggle in Settings → Apps → RF Connector AI → Permissions. |
| Phone not listed by `adb devices` | USB debugging off, RSA prompt not accepted, or the wrong USB cable (data vs charge-only). |
| Install fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | App with the same `applicationId` is installed under a different signing key — `adb uninstall com.rfconnector.ai` first. |

---

## Layout

```text
exports/mobile/
  package.json                       npm deps (Capacitor core + camera plugin)
  capacitor.config.ts                appId, webDir, plugin config
  www/                               web assets + ONNX models (auto-copied to native projects)
    index.html
    app.js
    style.css
    models/
      detector.onnx                  YOLO11n detector (~10 MB)
      classifier.onnx                EfficientNetV2-S multi-head (~77 MB)
      classifier_vocabs.json
  android/                           native Android project (Capacitor scaffold)
    app/src/main/AndroidManifest.xml
    app/build/outputs/apk/debug/app-debug.apk  (build output, gitignored)
    local.properties                 SDK path (gitignored, must be created locally)
  ios/                               native iOS project (Capacitor scaffold)
  dist/
    app-debug.apk                    pre-built debug APK for sideloading
  scripts/copy-web.js                helper to copy training web demo into www/
```

Every artifact filename embeds the `model_id` from
`training/rfconnectorai/models/registry.py` so it can be matched back to
the exact training run, dataset hash, and taxonomy hash.

---

## Model exports (training-side)

The ONNX/TFLite/Core ML files shipped in `www/models/` and any future
server-side model store are produced by
`training/rfconnectorai/export/export_mobile.py`.

Local PCs only run `--dry-run` (no torch/onnx/coremltools install
required). Real exports run in Kaggle / Colab where the ML toolchain is
available.

```bash
python -m rfconnectorai.export.export_mobile \
    --target detector:models/detector/best.pt:reports/experiments/<run>/model_record.json:onnx,tflite \
    --target classifier:models/multihead_classifier/best.pt:reports/experiments/<run>/model_record.json:onnx,coreml \
    --out exports/mobile \
    --dry-run
```

After a real cloud run, the manifest at
`exports/mobile/exports_manifest.json` lists every exported artifact
with:

- target name (`detector` / `classifier`),
- output format,
- source artifact path,
- `model_record.json` reference,
- model_id, architecture, dataset id, and taxonomy hash.

### Compatibility notes

- ONNX is always produced. ONNX Runtime is the most portable option for
  both server and mobile.
- TFLite/LiteRT requires a `tensorflow` or `ai-edge-torch` install in
  the cloud env.
- Core ML requires `coremltools` and is macOS-friendliest.
- Mobile latency / thermal benchmarks live in the per-device benchmark
  reports under `reports/experiments/<run>/latency_report.md`.

### Hard rule

Mobile exports must not silently drop heads, change attribute
vocabularies, or break the legacy `/predict` response. Every exported
artifact must match the head vocabulary recorded in
`reports/experiments/<run>/head_vocabs.json`.
