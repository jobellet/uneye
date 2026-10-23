# Project: real, on-device training in the web labeler (iPad M1, Android phone, laptop; no app to install)

Written 2026-10-23 for the AI agent that will do this work. Read `AGENT.md`, `docs/WEB_FORMATS.md`, `docs/SPEED_LIMIT_HANDOFF.md`, `cpp/app/AGENT.md` first.

## 1. The problem

The web labeler (`index.html`, `app.js`, `formats.js`; served by GitHub Pages at https://jobellet.github.io/uneye/) lets the owner label saccades, microsaccades and blinks
in two passes (tap marks, then press-drag refinement; the pre-labeler is an adaptive Engbert-Kliegl / HMM written in JavaScript). Its **"Train" button does not train anything**:
`runTrainingRound()` in `app.js` (about line 1470) only invents metrics ("Simulate training metrics progressing on the iPad M1": F1 0.88-0.92 + noise), and
`evaluateConvergenceFlag()` / `showMetricsDialog()` display them. There is no network in the page. The owner needs to be able to **label on a device and train there**, in the browser:

- an **iPad with an M1** (Safari; WebGPU is probably available, WASM SIMD yes),
- an **Android phone or an ordinary laptop with only a CPU** (Chrome), when that is all he has in hand,
- always through the **web site** (not the Android app of this repository, not an installed program), offline after the first load if possible.

Goal: after labeling a few minutes of eye traces (a few dozen to a few hundred labeled seconds), press Train, see real epochs, real validation/test F1 and kappa, get a model that
(1) replaces the pre-labeler on the not yet reviewed data, (2) can be saved and (3) can be used outside the page (C++ engine, PyTorch, `mocol_saccades/detect_saccades_mocol.py`).

## 2. What already exists (use it, do not rewrite it)

| what | where | notes |
|---|---|---|
| **C++ trainer without PyTorch** | `cpp/app/train/` (`include/eyelive/train/{layers,unet,dataset,model_io}.hpp`, `src/{trainer,dataset,model_io}.cpp`, about 580 lines), contract `cpp/app/include/eyelive/trainer.hpp` (`ITrainer`, `TrainConfig`, `TrainProgress`) | the original U'n'Eye topology (ks 5, mp 5), **18 602 parameters** (2 classes; 3 classes 18 613; 5 classes 18 635), Adam, MCLoss + L2, early stopping with learning-rate decay, rotation / noise / gain augmentation, windows of about 1 s, validation split, event F1 and kappa (`score_predictions`), `init_model` (transfer learning), cooperative `cancel()`, progress callback per epoch, `save_model()` in the `cpp/models/*.bin` layout. Gradient parity with PyTorch: `cpp/app/train/tests/golden_unet_grads.txt` |
| speed of that trainer (measured 2026-10-23, Intel i7-1068NG7, 4 cores) | `cpp/app/train/tests/test_speed.cpp` | 50 s of labeled data with 4 rotations (200 training windows of 1000 samples): **2.1 s per epoch on 1 thread, 0.76 s on 4 threads**. Training stops after tens of epochs: about 1-2 minutes on one core |
| inference | `cpp/src/native_engine.cpp` (dependency-free U-Net), `cpp/src/detector.cpp` (`StreamingDetector`), `cpp/include/uneye/*.hpp` | the same `.bin` models; the engine also runs the BiTCN (`bitcn.cpp`) |
| the two-pass UI, label store, convergence logic | `app.js`: `state.expertEvents`, `state.reviewedRanges`, `state.pointMarks`, `state.testTrials` (every 4th trial), `runPrelabeler`, `evaluateConvergenceFlag` (bootstrap 95% CI of the F1 gain over the last 3 rounds, SILENT until >= 100 saccades and >= 20 blinks in the test set), `totalReviewedSeconds`, `loadDataset`, `exportDataset` | **Data rules (from `android/app/README.md` design): point marks never train; only refined events (pass 2) and reviewed ranges are labels; an epoch counts when scanned AND all its marks are refined. The test trials are never trained on; the trainer's own validation F1 is biased and is never used for the convergence flag** |
| file formats | `formats.js` (read / write `.mat` `.npz` `.h5` with variable-length trials, labels, names) | already tested in Node and Chrome |
| PyTorch reference | `uneye/classifier.py` (`DNN.train`, `max_speed` = speed limit), `cpp/scripts/export_model.py` (PyTorch weights -> `.bin` / `.onnx`) | there is **no** `.bin` -> PyTorch direction yet |
| speed limit | 1300 deg/s on the network input everywhere (`uneye.functions.clip_speed`, `Config::max_speed_deg_s`, `TrainConfig::max_speed_deg_s`) | the web trainer must apply the same limit at training and at inference, after rotation augmentation, before the network; see `docs/SPEED_LIMIT_HANDOFF.md` |

## 3. Options, with a recommendation

The network is tiny (18.6 k parameters, 1000-sample windows, batches of 32): kernel-launch and memory-transfer overheads dominate on a GPU, so **a GPU is not what limits this**; correctness, determinism and
portability are. Evaluate in this order, and decide with measurements on real devices (section 6):

1. **Recommended - WebAssembly build of the existing C++ trainer and inference engine** (Emscripten, `-O3 -msimd128`; optional `-pthread`). Same numerics as EyeLive and Android, the model file is the same `.bin`,
   gradient goldens already exist, runs on any phone / tablet / laptop CPU, no GPU needed, fully offline, a few hundred KB. Expect 1.5-3x slower than native per core: a few seconds per epoch for 50 s of data on
   a modest phone, faster on an M1. Run it in a **Web Worker**, post progress messages per epoch, cancel with a flag. *Threads:* GitHub Pages cannot send the COOP / COEP headers that `SharedArrayBuffer` needs;
   use the `coi-serviceworker.js` trick (the service worker re-serves the page with the headers) for multi-threaded WASM, and make the single-thread build the default / fallback. Safari on iPad supports WASM SIMD (16.4+) and, when cross-origin isolated, threads.
2. **TensorFlow.js** (WebGPU backend on the M1 iPad where available, WebGL, and the WASM-XNNPACK backend on CPU-only devices). Needs the topology re-implemented (Conv2D with a (5, 2) kernel on a (T, 2) input, Conv1D, MaxPool1D, Conv1DTranspose, BatchNorm in training mode, softmax, the masked MCLoss with the "ignore" label, L2, Adam) and re-validated against `golden_unet_grads.txt`; BatchNorm training-mode statistics and transposed convolutions are the usual sources of silent mismatch. The GPU gain is probably small for this network. Worth it only if (a) the owner later wants a larger network, (b) the benchmark shows a clear win on the M1 iPad. Keep it behind the same worker interface as option 1 so the two can be compared on the same data.
3. **ONNX Runtime Web training** (`onnxruntime-web` training artifacts generated offline with `onnxruntime.training.artifacts.generate_artifacts` from the PyTorch model; WASM and WebGPU back-ends). Same topology as PyTorch by construction; heavier (several MB), the web training API is young. Candidate if option 1 hits a wall.
4. **Plain JavaScript (Float32Array) port of `layers.hpp`** in a worker: no dependency, small, but 5-10x slower than WASM; keep as the fallback for a browser without WASM SIMD.
5. **Hand-written WebGPU compute shaders**: best speed on the M1, most work; only for later use (batch pre-labeling of very long recordings).

Whatever is chosen: **one worker API** (`train.start(traces, config)`, `progress` events, `cancel()`, `predict(trace)`, `saveModel()`), so back-ends are interchangeable and testable against each other.

## 4. What to build

1. **Replace the simulation** in `runTrainingRound()` by real training (remove the fake metrics and the 600 ms timeout). Training set = the reviewed stretches of the non-test trials (labels: 0 fixation, 1 saccade, 2 blink; unreviewed samples = ignore label 255), with the settings of `TrainConfig` (rotation augmentation, validation split, early stopping). Test set = `state.testTrials`, scored with the same event F1 (a predicted run of >= 3 samples overlapping a human event is a hit) and Cohen's kappa as `cpp/app/train/src/dataset.cpp::score_predictions` / `foundation/compare.py::event_f1`; **per class** (saccade, blink) as the metrics dialog and the convergence flag expect. Feed these real numbers to `state.convergenceHistory`, `evaluateConvergenceFlag`, the metrics plots and the status line.
2. **Trials of different durations** (already supported by the page since 2026-10-23): windows are cut inside each trial; never across trials.
3. **Initial model**: start from the shipped pre-trained model (`cpp/models/combined.bin`, transfer learning via `init_model`) or from scratch (option in the load dialog or the Train dialog). Note that the shipped weights were trained without the speed limit.
4. **Scheduling**: train in the background after every N newly reviewed seconds (the Android design uses 20 s; make it a setting) and on "Train now"; show epoch, train / validation loss, validation event F1, ETA, a Cancel button; keep the screen awake during training (`navigator.wakeLock`) and handle iOS pausing background tabs (resume or restart cleanly).
5. **Use the model**: after a round, run it on the not yet reviewed parts (`apply_auto_labels`: expert events and reviewed ranges are never touched) and let the user choose it as the pre-labeler instead of the HMM. Always keep the previous model so a bad round can be undone.
6. **Persistence**: store the labels, the settings and the model in IndexedDB (autosave) so that closing the tab on an iPad loses nothing; "Download model" writes `uneye_model.bin` + `uneye_model.json` (the `cpp/models` layout: the same files the C++ engine, EyeLive and `mocol_saccades` use) and a PyTorch state dict (see 7).
7. **Python side**: add `cpp/scripts/bin_to_state_dict.py` (inverse of `export_model.py`, tested by a round trip) so that a model trained in the browser loads in `uneye_min.UNet` / `mocol_saccades/detect_saccades_mocol.py --weights`; document the preprocessing that goes with a model (Savitzky-Golay window, speed limit, rate) inside the `.json` and `uneye_config.json`.
8. **Offline**: a service worker caching `index.html`, `app.js`, `formats.js`, the `.wasm`, `combined.bin` (and h5wasm if HDF5 is used) so that the page works without a connection after the first visit. (Keep the `coi-serviceworker` and the cache worker as one worker.)
9. **Build**: `scripts/build_wasm.sh` (emsdk), the built `.wasm` / `.js` committed under `web/` or the repository root (GitHub Pages serves static files; keep them small), a short `docs/WEB_TRAINING.md`.

## 5. Constraints and pitfalls

- Static hosting only; no server, no custom headers; no installation; works from `https://jobellet.github.io/uneye/`.
- Do not touch the Android app or the EyeLive C++ behaviour (the C++ is shared): new code goes into a new target (`cpp/web/`) that links the existing sources unchanged. All `ctest` (34) must stay green.
- iPad Safari: memory limits (keep the WASM heap below a few hundred MB), no `SharedArrayBuffer` without cross-origin isolation, background tabs are suspended, `DecompressionStream` needs Safari 16.4+.
- Determinism: fixed seeds; the same labels and seed must give the same model on every device within float tolerance (this is how the back-ends are compared).
- Do not run training on the main thread; the labeling UI must stay at 60 fps while training.
- Do not commit data (`data/Andersson/`, `archive/`, weights `foundation/runs/*.pt` are git-ignored). Commit messages end with the co-author line required by the session (see `AGENT.md`).
- Report only numbers from scripts that were run.

## 6. Acceptance tests

1. **Parity**: with the same traces, labels, seed and settings, the browser trainer (worker, WASM) and the native `eyelive_train_test_trainer` give the same loss curve (to 1e-4 relative) and the same saved `.bin` (to 1e-4); the gradient goldens of `golden_unet_grads.txt` pass in the browser build; the PyTorch `uneye.DNN` reaches an equal validation F1 on `data/dataset1` set A (10 draws of 20 and 50 labeled trials as in `docs/BENCHMARK_SUMMARY.md` table 3).
2. **Speed table** (device x browser x back-end): seconds per epoch for 50 s of data with 4 rotations, and the time to a finished training (early stopping), on: an iPad M1 (Safari), an Android phone (Chrome), an ordinary laptop (Chrome, CPU only), and the Intel laptop used for the native reference (2.1 s / epoch, 1 thread). Budget: a complete training of 50 s of labels in under 5 minutes on the phone.
3. **End to end** with the 213 mocol trials of `labeling_pool/` (Google Drive: `Postdoc_Siegel/MS_CD/DATA/mocol/labeling/web/all_trials.npz`, 17 minutes of eye traces, trials of 3.5-7.3 s; the labels can be made by loading `mocol_saccades/labeling_set/` or by labeling a few trials): label, train, see the pre-labels change on the unreviewed trials, export `unEye_labeled.npz` and the model, load the model in `detect_saccades_mocol.py`, F1 against the held-out labeled trials >= the HMM pre-labeler.
4. **Robustness**: a trial with NaN stretches and values of thousands of degrees (blinks, tracker loss) must not produce NaN losses or a crash; the speed limit is applied.
5. Headless-browser tests (puppeteer-core with Chrome, as in `test_formats.js`'s environment) for the worker API, cancel, resume, IndexedDB restore; a manual checklist for iPad Safari and Android Chrome.

## 7. Prompt

> You are working in the repository `uneye` (https://github.com/jobellet/uneye, branch `master`; pull first). Read `docs/WEB_TRAINING_PROJECT.md` completely, then `AGENT.md`, `docs/WEB_FORMATS.md`,
> `docs/SPEED_LIMIT_HANDOFF.md`, `cpp/app/AGENT.md` and `cpp/app/train/`. The web labeler (`index.html`, `app.js`, `formats.js`, GitHub Pages) has a "Train" button that only simulates training
> (`runTrainingRound()` invents F1 scores). Your project: make it train a real U'n'Eye network, on the device, in the browser, with no app to install, on an iPad M1 (Safari, probably WebGPU) AND on an
> Android phone or an ordinary CPU-only laptop (Chrome). Start from the C++ trainer that already exists (`cpp/app/train`, 18.6 k parameters, 2.1 s per epoch on one core for 50 s of data):
> build it to WebAssembly (Emscripten, SIMD, optional threads with `coi-serviceworker` because GitHub Pages cannot set COOP/COEP) and run it in a Web Worker; compare it, on the same data, with a TensorFlow.js
> (WebGPU / WASM) implementation and, if needed, ONNX Runtime Web training, and choose with measurements on real devices (section 6 of the document). Replace the fake metrics by real ones (event F1 and kappa
> per class on the held-out test trials, the existing convergence flag fed with them), train on the reviewed, refined labels only (point marks never train), apply the 1300 deg/s speed limit, support trials of
> different durations, start from `cpp/models/combined.bin` or from scratch, use the trained model as the pre-labeler for the unreviewed data, autosave to IndexedDB, work offline after the first visit,
> and export the model as `.bin` + `.json` (add `cpp/scripts/bin_to_state_dict.py` and a round-trip test so that `mocol_saccades/detect_saccades_mocol.py --weights` can use it). Keep the C++ behaviour and
> the 34 ctest tests unchanged. Deliver: the code, the build script, a speed table on the devices, the parity tests of section 6, a `docs/WEB_TRAINING.md`, and commits (co-author line as in AGENT.md). Ask the owner
> only for things you cannot do yourself (a physical iPad or phone to test on). Report numbers only from scripts you ran.
