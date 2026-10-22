# Android offline-labeling app — phase 0 (build + benchmark)

Phase 0 of the plan (see the task list in `docs/EYELIVE_ROADMAP.md` revision 2
and the app roadmap): build `cpp/` + `cpp/app/` for Android arm64-v8a with the
NDK, and measure the trainer + HMM on a real phone **before any UI work**.

Nothing in `cpp/` or `cpp/app/` is modified: the contracts in
`cpp/app/include/eyelive/` are frozen and `docs/tasks/OWNERSHIP.json` assigns
module folders to owners. All new code lives in this folder.

## Files

| file | role |
|---|---|
| `CMakeLists.txt` | top-level project wrapping the frozen `cpp/` tree + `bench/` |
| `android.toolchain.cmake` | NDK toolchain file (arm64-v8a, android-24, static libc++) |
| `build_android.sh` | host build (no NDK) or arm64-v8a cross-build (with NDK path) |
| `bench/bench_offline.cpp` | benchmark harness: load CSV → cut 1 s trials → resample to 1 kHz → HMM pre-label → agreement vs file labels → ExpertLabels → train the U-Net → report seconds/epoch |
| `bench/CMakeLists.txt` | builds `bench_offline` against the frozen modules |
| `PHASE0_RESULTS.md` | measured numbers (host + phone) |

## Build

Host (first measurement, no phone needed):

```sh
sh android/build_android.sh
./build-host/bench/bench_offline data/dataset1/dataset1_1000hz_X_setA.csv \
    data/dataset1/dataset1_1000hz_Y_setA.csv data/dataset1/dataset1_1000hz_Labels_setA.csv \
    cpp/models/universal_hmm.txt 50 4 --rate 1000
```

Android (needs the NDK, e.g. `ndk;24.0.11862519` from
https://developer.android.com/ndk/downloads, unpacked):

```sh
sh android/build_android.sh /path/to/android-ndk
adb push build-android/bench/bench_offline /data/local/tmp/
adb shell chmod +x /data/local/tmp/bench_offline
adb push data/dataset1/dataset1_1000hz_X_setA.csv /data/local/tmp/X.csv
adb push data/dataset1/dataset1_1000hz_Y_setA.csv /data/local/tmp/Y.csv
adb push data/dataset1/dataset1_1000hz_Labels_setA.csv /data/local/tmp/L.csv
adb shell /data/local/tmp/bench_offline /data/local/tmp/X.csv /data/local/tmp/Y.csv /data/local/tmp/L.csv x 50 4 --rate 1000
```

Options: `EYELIVE_PORTABLE=ON` (no `-march=native`, required for ARM),
`EYELIVE_BUILD_GUI=OFF` (Dear ImGui needs desktop GL), no ONNX Runtime (the
native engine is dependency-free). The trainer tests
(`eyelive_train_test_speed`, `eyelive_train_test_trainer`,
`eyelive_detect_hmm_test_hmm`) are built by the same tree and can be pushed to
the phone the same way.
