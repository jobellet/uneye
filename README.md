![U'n'Eye](logo.jpeg)

# U'n'Eye — tools to detect and label saccades, microsaccades and blinks

This repository is a **fork of [berenslab/uneye](https://github.com/berenslab/uneye)** that has grown far beyond it.

> **The original U'n'Eye** (Bellet, Bellet, Nienborg, Hafed & Berens 2019, *Human-level saccade detection performance using deep neural networks*, J. Neurophysiol.; [paper](https://journals.physiology.org/doi/full/10.1152/jn.00601.2018)) is a **proof of principle**: a Python package that trains a U-Net to detect saccades on a finished recording. **This repository is the practical side**: tools you can install on a tablet, a phone or a lab computer, to label your own data quickly, train a detector on it, and run it live during psychophysics experiments. The original package, its install instructions and its notebooks are kept in this repository (see [the original package](#6-the-original-python-package)) and documented in [`docs/README_UNEYE_ORIGINAL.md`](docs/README_UNEYE_ORIGINAL.md); for that alone, the [original repository](https://github.com/berenslab/uneye) is the reference.

> **Transparency (2026):** the real-time C++ engine, EyeLive, the web and Android labelers, the label-free and foundation studies and the documents in `docs/` were written with AI assistance. Every number quoted in `docs/` comes from a script of this repository that was run (`docs/BENCHMARK_SUMMARY.md`, `docs/ROADMAP.md`, `docs/safety/hazards.md`).

## Which tool do I want?

| I want to... | Use | Runs on |
|---|---|---|
| mark saccades / blinks by hand on my recordings, with a pen or fingers, and train a detector from my labels | [**Web labeler**](#1-web-labeler-ipad-tablet-phone-laptop) | iPad, Android tablet or phone, any laptop (browser, no install) |
| the same, offline in a phone app that trains in the background | [**Android labeler**](#2-android-labeler) | Android phone / tablet (arm64) |
| record from an eye tracker, detect events live, correct the labels, trigger my experiment | [**EyeLive**](#3-eyelive-real-time-recording-labeling-and-triggers) | Windows, macOS, Linux |
| run a causal detector sample by sample inside my own program | [**Real-time C++ engine**](#4-real-time-c-engine-and-command-line-tools) | any C++17 compiler; Windows / macOS / Linux / Android |
| the most accurate offline detector, or a detector that needs no labels | [**Python detectors**](#5-python-detectors-and-studies) | Python 3, PyTorch |
| the original 2019 package and notebooks | [**Original package**](#6-the-original-python-package) | Python 3 |

---

## 1. Web labeler (iPad, tablet, phone, laptop)

**Open <https://jobellet.github.io/uneye/>.** Nothing to install; everything runs in the browser and your data never leave the device.

### Install on a device
- **iPad / iPhone (Safari)**: open the page, wait for it to finish loading once, then *Share → Add to Home Screen*. It then starts like an app and works **offline**.
- **Android (Chrome)**: open the page, menu *⋮ → Install app / Add to Home screen*.
- **Laptop (Chrome, Edge, Safari, Firefox)**: just open the page; to work offline keep the tab or install it from the address bar.
- After a new version, reload the page twice (the service worker updates the offline copy).
- Self-hosting: serve this folder with any static server (`python3 -m http.server`) and open `index.html`.

### What you can do
- **Load** eye traces: CSV / TXT (two columns x y, or rows X / Y per trial), `.npz`, `.mat` (MATLAB v5, trials in a cell array, different durations allowed) and `.h5` (HDF5; the HDF5 reader is downloaded once, so be online the first time). Formats and layouts: [`docs/WEB_FORMATS.md`](docs/WEB_FORMATS.md).
- **Pass 1 — mark**: the trace is shown in windows whose width in ms you choose (default 500 ms, 10 % overlap between windows, then on to the next trial). Tap a saccade or a blink to mark it; tap again to remove. Automatic pre-labels (adaptive Engbert–Kliegl / HMM) are shown as suggestions.
- **Pass 2 — refine**: the view jumps from event to event (across trials, without a "next trial" button) with a ±50 ms frame; press-drag to set the onset and offset, or accept the pre-label.
- **Eraser**: drag between two times to delete every event, mark and pre-label in between.
- **1° scale bar** at the right of each position lane, so a microsaccade is told from a large saccade whatever the automatic zoom.
- **No accidental marks** after a two-finger zoom (600 ms refractory period).
- **Save / reload your labels** (`Save labels` / `Load labels`, a `.json` file) to continue later or on another device; export CSV, `.npz`, `.mat` or `.h5` with the labels (`-1` = not reviewed).
- **Train a real U-Net in the browser** (button *Train*): the C++ trainer compiled to WebAssembly (SIMD) runs in a worker on the iPad M1 or on an ordinary CPU, starts from the shipped pretrained model, reports epochs, loss and F1 on held-out trials, and can be cancelled; the model is saved on the device and downloadable (`Model`, a `.bin` the other tools read). Details, speeds and caveats: [`docs/WEB_TRAINING.md`](docs/WEB_TRAINING.md).
- **Only what you looked at is ever used for training**: a window counts only after you moved on from it in pass 1, and only if every mark in it has been refined; automatic pre-labels never count as truth; every 4th trial is held out to score the network. Behaviour specification: [`docs/LABELER_UX_SPEC.md`](docs/LABELER_UX_SPEC.md).

Files: `index.html`, `app.js`, `formats.js`, `train_worker.js`, `sw.js`, `web/uneye_train.{js,wasm}`, `models/combined.bin`. Rebuild the WebAssembly with `scripts/build_wasm.sh` (needs [emsdk](https://github.com/emscripten-core/emsdk)). Tests: `node test_formats.js`, `node test_labeler_flow.mjs` (headless Chrome, `puppeteer-core`), `node test_wasm_trainer.js`.

## 2. Android labeler

Offline app for a phone or tablet with the same two-pass labeling as the web labeler (windows with 10 % overlap and a width you set, event-to-event refinement across trials, eraser, 1° scale bar, refractory period after a pinch), plus **background training** of the on-device U-Net (WorkManager), a **convergence flag** (bootstrap confidence interval of the F1 gain over the last rounds) with learning curves, and export of labels CSV (UnEye convention), X / Y CSV, MATLAB v5 `.mat` and the trained model `.bin` / `.json` through the system share sheet.

### Install
There is no store release; you build the APK once:
1. Install [Android Studio](https://developer.android.com/studio) with the SDK and the NDK (arm64-v8a), JDK 17.
2. `cd android && ./gradlew :app:assembleDebug` (or *Run* in Android Studio with the phone connected by USB, developer mode and USB debugging on).
3. Or install the APK by hand: `adb install app/build/outputs/apk/debug/app-debug.apk`.
4. Copy your CSV (two files X / Y, trials × time) or a `.mat` (variables `X`, `Y`) to the phone, open the app, pick the file, the sampling rate and the units.

Host-side check without a phone: `cmake -S android -B build-host && cmake --build build-host --target test_offline_session && ctest --test-dir build-host -R test_offline_session`. Design, phases and measurements: [`android/README.md`](android/README.md), [`android/PHASE0_RESULTS.md`](android/PHASE0_RESULTS.md).

## 3. EyeLive: real-time recording, labeling and triggers

A small desktop program (a window `eyelive` plus the command-line twin `eyelive_cli`, no installer, no administrator rights) that **records an eye-tracker stream, labels saccades and blinks as they happen, lets you correct the labels in minutes, learns from your corrections and can trigger your experiment on events**. The state block of a running session is readable from MATLAB / Psychtoolbox or Python / PsychoPy.

### Install
1. Download the zip for your system from the *Releases* of the repository (built by `.github/workflows/eyelive-release.yml`: Windows, macOS, Linux).
2. Check the SHA-256, unblock / allow the unsigned program, and start it: step by step for each system in [`docs/eyelive/INSTALL_UNSIGNED.md`](docs/eyelive/INSTALL_UNSIGNED.md). Keep the `models/` folder next to the program.
3. First launch: a four-question wizard (tracker, signal, goal, name) creates a profile; then *Start recording*.

### What you can do
Simulator or file replay (no tracker needed to try it), UDP stream from the tracker computer (protocol in the manual, helper senders in `cpp/app/tools/`: Python, EyeLink template, PsychoPy and MATLAB frame feeders), several detectors swappable during a recording (label-free `hmm_accel`, the very fast threshold detector for triggers, the pretrained U'n'Eye networks, causal and bidirectional TCNs, your own), review and correction of labels, training on your corrections, prediction on other sessions, export (events CSV, BIDS TSV + JSON, MAT-file, HTML report with provenance). Manuals: [`docs/eyelive/MANUAL.md`](docs/eyelive/MANUAL.md) (English), [`docs/eyelive/MANUAL_FR.md`](docs/eyelive/MANUAL_FR.md) (français), [`docs/eyelive/EYELINK_DATA.md`](docs/eyelive/EYELINK_DATA.md), [`docs/eyelive/LAB_TEST.md`](docs/eyelive/LAB_TEST.md), [`docs/eyelive/WINDOWS_MATLAB_TEST.md`](docs/eyelive/WINDOWS_MATLAB_TEST.md).

### Build from source
`cmake -S cpp -B build -G Ninja && cmake --build build && ctest --test-dir build -L eyelive --output-on-failure` (see [`cpp/AGENT.md`](cpp/AGENT.md), [`cpp/app/AGENT.md`](cpp/app/AGENT.md)).

## 4. Real-time C++ engine and command-line tools

Dependency-free C++17 port of the networks that labels **sample by sample while the signal is recorded** (`StreamingDetector`: ring buffer, windowing, lookahead commit, event state machine), with a physics layer and a safety guard (`GazeEngine`, `SafetyGuard`: NORMAL / DEGRADED / SAFE) and ONNX / native `.bin` models. Command-line tools in `cpp/tools/`: `uneye_rt` (`--sim`, `--replay` a recording as a live stream and score it, `--stdin`), `gaze_replay`, `fault_injection`, `bench_latency`. Build: `cd cpp && cmake -B build && cmake --build build && ctest --test-dir build`. Usage, design, guarantees and what is *not* verified: [`cpp/README.md`](cpp/README.md), [`cpp/GAZE_ENGINE.md`](cpp/GAZE_ENGINE.md), [`docs/safety/hazards.md`](docs/safety/hazards.md), [`docs/ROADMAP.md`](docs/ROADMAP.md).

All detectors clip the input velocity at 1300 deg/s (physiological limit of macaque saccades) so that blinks and tracker loss cannot destabilise the network: [`docs/SPEED_LIMIT_HANDOFF.md`](docs/SPEED_LIMIT_HANDOFF.md).

## 5. Python detectors and studies

| Folder | What it is |
|---|---|
| [`foundation/`](foundation/AGENT.md) | the best offline detector so far, a 2-channel (vx, vy) bidirectional TCN (`sup_bitcn.py`), the benchmark on 5 datasets (event F1 and kappa, tables in [`docs/BENCHMARK_SUMMARY.md`](docs/BENCHMARK_SUMMARY.md)) and every self-supervised variant tried ([`docs/OVERNIGHT_REPORT.md`](docs/OVERNIGHT_REPORT.md)) |
| [`free_saccade/`](free_saccade/AGENT.md) | label-free detectors: Engbert–Kliegl, a universal minimum-duration HMM (F1 0.88), seeded contrastive embeddings |
| [`online/`](online/AGENT.md) | causal (forward-only) networks for online labeling, the model behind the live engine |
| [`training/`](training/AGENT.md), [`data/`](data/AGENT.md) | pretrained weights and the datasets of the original article |
| [`docs/`](docs/AGENT.md) | roadmaps, audits, safety documents, benchmark tables, slides |

Set-up (macOS / Linux): `python3 -m venv .venv && .venv/bin/pip install -r requirements_mac.txt` (or `requirements_lin.txt`, `requirements_wind.txt`), then run from the repository root, e.g. `.venv/bin/python foundation/sup_bitcn.py --inputs 2`. Evaluation metric everywhere: event F1 (a predicted run of at least 3 samples overlapping a human saccade) and Cohen's kappa.

## 6. The original Python package

`uneye` (PyTorch, `pip install ./ -r requirements_*.txt`) trains and applies the original U-Net on position traces; pretrained weights are in `training/`, the datasets in `data/`, and the examples in `UnEye.ipynb` (and the online-labeling notebooks `UnEye_online_labeling.ipynb`, `UnEye_online_training.ipynb`). Installation, Docker, command-line use and the module description are in [`docs/README_UNEYE_ORIGINAL.md`](docs/README_UNEYE_ORIGINAL.md). `saccade_labeler.py` is the older desktop (PySide6) labeling GUI.

## Citation

If you use U'n'Eye, please cite the original article: Bellet, M. E., Bellet, J., Nienborg, H., Hafed, Z. M., & Berens, P. (2019). *Human-level saccade detection performance using deep neural networks.* Journal of Neurophysiology, 121(2), 646–661.

## Repository map

`AGENT.md` at the root is the router for coding agents (what lives where, conventions, environment); every folder has its own `AGENT.md`.
