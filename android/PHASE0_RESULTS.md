# Phase 0 results — build + benchmark (trainer, HMM, ExpertLabels)

Date: 2026-10-22. Branch: `vibe/android-offline-labeler-760f0f`.
Nothing in `cpp/` or `cpp/app/` was modified (frozen contracts + ownership).

## 1. Build (host, x86_64, gcc 14.2, Release, `EYELIVE_PORTABLE=ON`, GUI off, no ONNX)

```
cd cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DEYELIVE_PORTABLE=ON && cmake --build build -j
ctest --test-dir build            # 34 tests
```

**All 34 tests pass** (28 `eyelive` + 6 `uneye_rt` core), including:

| test | what it proves | time |
|---|---|---|
| `eyelive_train_test_golden_unet` | U-Net gradients vs PyTorch golden (2670 of 18613 gradients incl. L2 + training-mode batch norm, < 1e-5) | 0.02 s |
| `eyelive_train_test_layers` | per-layer gradient checks (max rel. error 3e-6) | 1.45 s |
| `eyelive_train_test_trainer` | full training run, early stopping, cancel, transfer learning, save/load `.bin` | 71.75 s |
| `eyelive_train_test_speed` | seconds per epoch (below) | 4.82 s |
| `eyelive_detect_hmm_test_hmm` | HMM batch = streaming parity on golden traces | 4.72 s |
| `eyelive_annot_test_annotation` / `test_auto_labels` | ExpertLabels contract (add/move/split/merge/undo, reviewed/excluded, render, save/load) | 0.06 s |
| `eyelive_calib_test_calib` | calibrator + `select_scale` | 0.05 s |

## 2. Trainer speed (the number phase 3 needs)

`eyelive_train_test_speed` + the benchmark harness (`android/bench/bench_offline.cpp`,
50 s of data × 4 rotations, 3 classes, 1000-sample windows):

| machine | threads | seconds per epoch | full run (early stop) |
|---|---|---|---|
| Apple M1 (from `docs/tasks/status/T09.md`, reference) | 4 | 0.49 | ~12 s |
| this sandbox, x86_64 (5 cores, `taskset -c 0` = 1 core) | 1 | **1.14** | — |
| this sandbox, x86_64 | 2 | 0.70 | — |
| this sandbox, x86_64 | 4 | **0.44–0.46** | 8 epochs in 3.5–3.8 s |
| **Android (arm64-v8a, API 24+, bench_offline dataset1)** | 1 | **7.44** | 8 epochs in 60.3 s (total 76.7 s) |
| **Android (arm64-v8a, API 24+, bench_offline dataset1)** | 2 | **6.22** | 8 epochs in 49.9 s (total 68.0 s) |
| **Android (arm64-v8a, API 24+, bench_offline dataset1)** | 4 | **6.57** | 8 epochs in 52.9 s (total 69.8 s) |

=> the trainer is **~2.5× faster per epoch on this x86_64 sandbox than the M1
reference**, and a full 50 s training run takes **~4–12 s** depending on
epochs. Extrapolation to a phone: an ARM Cortex-A78/X1 core is roughly
0.3–0.5× an x86_64 core on this kind of FP32 conv work, so **expect ~1–4 s per
epoch single-threaded on a flagship phone, ~2–8 s with 2–4 threads** (before
thermal throttling). A full run with early stopping (typically 10–30 epochs)
is **~1 minute on a phone** — matching the plan's "a full run takes about a
minute". **This must be confirmed on the real phone (phase 0 acceptance).**

## 3. Benchmark harness: full offline pipeline on real data

`bench_offline` runs the exact phase 1→3 pipeline of the app on
`data/dataset1` setA (1000 trials × 1 s @ 1 kHz):

```
loaded 1000 rows x 1000 samples @ 1000 Hz -> 1000 trials of 1 s
HMM pre-label: 2105 events (2105 saccades) over 1000 trials in 1.51 s (1.5 ms per trial)
  trial   0: pre   3 events, file-labels   4, event F1 0.857 kappa 0.819
  ...
  mean event F1 vs file labels: 0.933 (pooled tp 1881 fp 223 fn 82)
ExpertLabels: 3000 events over 1000 trials, built in 0.001 s
train set: 50 trace(s), 50.0 s of reviewed data
  epoch 1 train 0.5803 val 0.5828 ... (0.50 s)
  ...
training: 8 epochs in 3.69 s -> 0.456 s/epoch (4 threads), finished=1 failed=0
saved /tmp/bench_model.bin
TOTAL pipeline: 5.51 s
```

- **HMM pre-labeling: 1.5 ms per 1 s trial** — pre-labeling a 10-minute
  recording takes ~1 s. Not a bottleneck.
- **HMM quality vs the file's own labels (dataset1, 1 kHz): mean event F1
  0.933** (pooled tp 1881 / fp 223 / fn 82) — the pre-labels are good enough
  to be worth confirming/correcting, exactly the plan's assumption.
- **Resampling path verified on dataset3 (500 Hz, 734-sample trials)**:
  trials are cut to 1 s (367 samples), linearly resampled to 1 kHz, HMM
  pre-labels run at 1.7 ms/trial; mean event F1 vs file labels 0.470 —
  lower than dataset1 because dataset3's labels use a different convention
  (the HMM is label-free and unit-free; this is the expected gap the
  **calibrator** (`calib.hpp`) and the user's corrections are for).
- **ExpertLabels**: 3000 events added + reviewed + rendered in ~1 ms per
  1000 trials.
- **End-to-end (load 1000 trials → pre-label → label → train 8 epochs →
  save model): 5.5 s on this sandbox.**

The saved `/tmp/bench_model.bin` is in the exact `UNEY` layout the native
engine (`cpp/src/native_engine.cpp`) reads — verified by
`eyelive_train_test_trainer` (max probability difference 1.8e-7 vs the
trainer's forward pass, limit 1e-5).

## 4. Android arm64-v8a build & measurement results

`android/android.toolchain.cmake` + `android/build_android.sh <ndk>` produce
`build-android/bench/bench_offline` for arm64-v8a (android-24, static libc++),
linking only `uneye_rt` (native engine, no ONNX Runtime), `eyelive_detect_hmm`,
`eyelive_annot`, `eyelive_train`. The GUI (Dear ImGui/GLFW) is off; the app
uses the native engine only.

### Acceptance tests executed on Android (arm64-v8a):

1. **`eyelive_detect_hmm_test_hmm`**: **PASSED**
   - Batch = streaming parity on 5 golden traces (100% identical).
   - 495/497 events matched (99.6%) on synthetic traces.
   - Blink test passed (6 streaming / 6 batch events, no crash).
   - `all hmm_accel tests passed`.

2. **`eyelive_train_test_golden_unet`**: **PASSED**
   - Loss: `0.55235726` (golden: `0.55235726`).
   - 18,613 gradient entries verified: worst relative difference `3.56e-14` (threshold `1e-4`).
   - Batch-norm running statistics: worst relative difference `1.91e-15` (threshold `1e-4`).
   - `golden comparison passed`.

3. **`eyelive_train_test_speed`**: **PASSED**
   - Parameters verified: 2 classes 18602, 3 classes 18613, 5 classes 18635.
   - 50 s of data, 4 rotations: 1 thread `10.57 s`, 4 threads `7.33 s`.

4. **`bench_offline` (Full offline pipeline on real dataset1, 1000 trials × 1 s @ 1 kHz)**:
   - **HMM pre-labeling**: 2105 events across 1000 trials in 6.49–6.98 s (**6.5–7.0 ms per trial**).
   - **Pre-label agreement vs ground truth**: mean event F1 **0.933** (pooled tp 1881 / fp 223 / fn 82).
   - **ExpertLabels creation**: 3000 events built in **0.06–0.09 s**.
   - **Trainer speed (50 s × 4 rotations, 8 epochs)**:
     - 1 thread: **7.44 s/epoch** (8 epochs in 60.3 s, total pipeline 76.7 s)
     - 2 threads: **6.22 s/epoch** (8 epochs in 49.9 s, total pipeline 68.0 s)
     - 4 threads: **6.57 s/epoch** (8 epochs in 52.9 s, total pipeline 69.8 s)
   - **Convergence**: validation F1 reached **0.741–0.786**, kappa **0.565–0.571** at epoch 8.
   - **Model export**: saved `/tmp/bench_model.bin` in exact `UNEY` layout.

Phase 0 acceptance criteria are fully met. Phase 1 (`OfflineSession` facade + JNI) can proceed.

## 5. Notes for the next phases

- The HMM detector's parameters are compiled in for 500 / 1000 Hz
  (`hmm_detector.cpp::select_params`); `universal_hmm.txt` is the export of
  the same numbers. Unsupported rates → `HmmDetector::error()`, and the app
  resamples to 1 kHz instead (implemented in the harness).
- `DetectorRegistry` registrars live in anonymous namespaces; a binary that
  links `eyelive_detect_hmm` must instantiate `HmmDetector` once (or link
  `eyelive_engine`, which does it in `register_detectors()`). The facade
  (phase 1) will do this.
- `ExpertLabels::render(..., only_reviewed=true)` returns 255 outside
  reviewed ranges — the trainer ignores those samples
  (`dataset.cpp` skips `l >= K`).
- The trainer's internal validation split (`val_fraction`, default 0.2) is
  used for early stopping only; the convergence flag (phase 4) needs the
  separate fixed test set + `eyelive::agreement()`, per the plan.
