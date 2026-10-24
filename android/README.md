# Android offline-labeling app

Offline saccade/blink labeling on a phone, with on-device training of the
EyeLive U-Net and export of the labels + model for Google Colab. The plan
(roadmap + phase changes) is in `docs/EYELIVE_ROADMAP.md`; the implementation
follows the phased plan:

| phase | content |
|---|---|
| 0 | build `cpp/` + `cpp/app/` for Android arm64-v8a with the NDK, measure the trainer + HMM on a real phone before any UI work |
| 1 | the C++ `OfflineSession` facade (`offline_session/`) + the JNI bridge (`liboffline_session.so`) |
| 2 | the two-pass Kotlin labeling UI (tap + auto-scroll, then press-drag-lift), Canvas touch view, zoom/pan |
| 3 | background training worker (WorkManager), pre-labeler swap to the last trained model, convergence history |
| 4 | convergence flag in the UI (silent until the test set has enough events, bootstrap CI over the last rounds, "Converged" badge + alert) and the learning-curve metrics view (F1 / kappa per class vs reviewed seconds) |
| 5 | export: labels CSV (UnEye convention), X/Y CSV, MATLAB v5 `.mat`, trained model `.bin` + `.json`, via the Storage Access Framework and the system share sheet |

Nothing in `cpp/` or `cpp/app/` is modified: the contracts in
`cpp/app/include/eyelive/` are frozen and `docs/tasks/OWNERSHIP.json` assigns
module folders to owners. All new code lives in this folder.

## Files

| file | role |
|---|---|
| `CMakeLists.txt` | top-level project wrapping the frozen `cpp/` tree + `bench/` + `offline_session/` |
| `android.toolchain.cmake` | NDK toolchain file (arm64-v8a, android-24, static libc++) |
| `build_android.sh` | host build (no NDK) or arm64-v8a cross-build (with NDK path) |
| `bench/bench_offline.cpp` | benchmark harness: load CSV → cut 1 s trials → resample to 1 kHz → HMM pre-label → agreement vs file labels → ExpertLabels → train the U-Net → report seconds/epoch |
| `bench/CMakeLists.txt` | builds `bench_offline` against the frozen modules |
| `offline_session/` | the C++ `OfflineSession` facade (new folder, no frozen contract touched) + the JNI bridge |
| `app/` | the Android app (Kotlin UI + Gradle/NDK build) |
| `PHASE0_RESULTS.md` | measured numbers (host + phone) |

## App structure (`app/src/main/java/com/uneye/offline/`)

| file | role |
|---|---|
| `Native.kt` | the single JNI entry point (1:1 with `offline_session_jni.cpp`) |
| `Session.kt` | the labeling session state machine; Kotlin owns only the UI state, every label/edit decision lives in the C++ facade |
| `LabelingScreen.kt` | the two-pass labeling UI (pass 1: tap + auto-scroll; pass 2: press-drag-lift with focused event centering [-50 ms, +50 ms]) |
| `Kinematics.kt` | Savitzky-Golay cubic filter, derivative engine, robust noise estimation, and adaptive threshold detector |
| `TraceView.kt` | the Canvas touch view (traces, label band, marks, zoom/pan) |
| `TraceFiles.kt` | CSV + MATLAB v5 readers (`CsvParser`, `MatReader`) |
| `Exporter.kt` / `MatWriter.kt` | the export writers (labels CSV, X/Y CSV, `.mat`, model `.bin` + `.json`) |
| `ExportHelper.kt` | the SAF `CREATE_DOCUMENT` launcher + the `ACTION_SEND_MULTIPLE` share intent (FileProvider) |
| `TrainingScheduler.kt` / `TrainingWorker.kt` | the background training (WorkManager, charging/idle constraints) |
| `ConvergenceStore.kt` / `ConvergenceFlag.kt` | the convergence history + the flag (bootstrap CI over the last rounds) |
| `MetricsView.kt` | the learning-curve view (F1 / kappa per class vs reviewed seconds) |
| `MainActivity.kt` / `FileListDb.kt` / `UneyeApp.kt` | the file list (Room) + the load dialog + the application object |

## Download APK (Smartphone & Tablet)

Every commit with modifications to the Android app is automatically compiled and published:
- 📦 **[Download uneye-android.zip](https://github.com/jobellet/uneye/releases/download/android-latest/uneye-android.zip)** *(Recommended for smartphones: avoids direct `.apk` download blocks on mobile — download, extract the zip, and tap `uneye-android.apk` to install)*.
- 📱 **[Direct APK download: uneye-android.apk](https://github.com/jobellet/uneye/releases/download/android-latest/uneye-android.apk)** *(For direct download in mobile browser)*.
- 🔗 **[Latest release page](https://github.com/jobellet/uneye/releases/tag/android-latest)**

## Build

Host (first measurement, no phone needed):

```sh
sh android/build_android.sh
./build-host/bench/bench_offline data/dataset1/dataset1_1000hz_X_setA.csv \
    data/dataset1/dataset1_1000hz_Y_setA.csv data/dataset1/dataset1_1000hz_Labels_setA.csv \
    cpp/models/universal_hmm.txt 50 4 --rate 1000
```

Android app (needs the Android SDK + NDK 27.2.12479018):

```sh
cd android
./gradlew assembleDebug          # APK
./gradlew testDebugUnitTest      # JVM unit tests (convergence flag, exporter)
./gradlew connectedDebugAndroidTest   # instrumented tests on a device/emulator
```

Options: `EYELIVE_PORTABLE=ON` (no `-march=native`, required for ARM),
`EYELIVE_BUILD_GUI=OFF` (Dear ImGui needs desktop GL), no ONNX Runtime (the
native engine is dependency-free). The trainer tests
(`eyelive_train_test_speed`, `eyelive_train_test_trainer`,
`eyelive_detect_hmm_test_hmm`) are built by the same tree and can be pushed to
the phone the same way:

```sh
sh android/build_android.sh /path/to/android-ndk
adb push build-android/bench/bench_offline /data/local/tmp/
adb shell chmod +x /data/local/tmp/bench_offline
adb push data/dataset1/dataset1_1000hz_X_setA.csv /data/local/tmp/X.csv
adb push data/dataset1/dataset1_1000hz_Y_setA.csv /data/local/tmp/Y.csv
adb push data/dataset1/dataset1_1000hz_Labels_setA.csv /data/local/tmp/L.csv
adb shell /data/local/tmp/bench_offline /data/local/tmp/X.csv /data/local/tmp/Y.csv /data/local/tmp/L.csv x 50 4 --rate 1000
```

## Export formats (phase 5)

The Export button on the labeling screen writes, under
`Android/data/com.uneye.offline/files/exports/<recording>/` (or to any
provider via the Storage Access Framework):

| file | format |
|---|---|
| `<name>_L.csv` | labels, trials × time, 0 fixation / 1 saccade / 2 blink, `nan` = not reviewed (the UnEye convention: `np.loadtxt(L, delimiter=",")`, `np.isnan(Labels)` selects the labeled samples) |
| `<name>_X.csv`, `<name>_Y.csv` | positions in degrees, trials × time (`np.loadtxt(X)` / `np.loadtxt(Y)`) |
| `<name>.mat` | MATLAB v5, one file with `X`, `Y`, `Sacc` (`io.loadmat(L)['Sacc']`) |
| `<name>_config.json` | JSON sidecar with preprocessing filter parameters (`filter_type`, `window_ms`, `lambda`, `min_peak_vel_deg_s`, `min_amplitude_deg`) and a Python snippet to preprocess test traces identically with `scipy.signal.savgol_filter` |
| `<name>_model.bin` + `.json` | the last trained model (architecture + metadata), readable by the native engine and `uneye` |

Upload the CSVs or the `.mat` to Google Colab and train a uneye network with
`UnEye.py -x <name>_X.csv -y <name>_Y.csv -l <name>_L.csv -f 1000` (or the
`.mat` variants). The `.bin` + `.json` model can be loaded back into the app
as the pre-labeler or the training init (`init_model`).
