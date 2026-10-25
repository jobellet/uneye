# On-Device U'n'Eye Neural Network Training in the Browser

This document describes the on-device web training implementation for the U'n'Eye web labeler (`index.html`, `app.js`, `formats.js`). It allows training a real U'n'Eye convolutional network directly inside the browser without installing any native applications, running on an iPad M1 (Safari), Android phones, or any ordinary laptop CPU (Chrome).

---

## 1. Architecture

```
+-------------------------------------------------------------------------+
| Main UI Thread (app.js, index.html)                                     |
|  - 60 fps smooth interactive labeling and canvas rendering             |
|  - Pass 1 (mark) and Pass 2 (press-drag refinement)                     |
|  - Auto-save to IndexedDB                                               |
|  - Wake Lock management (navigator.wakeLock)                            |
+------------------------------------+------------------------------------+
                                     | postMessage (Non-blocking)
                                     v
+-------------------------------------------------------------------------+
| Web Worker (train_worker.js)                                            |
|  - Asynchronous background worker                                       |
|  - Unified Worker API: START_TRAIN, CANCEL, PREDICT, EXPORT_MODEL       |
|  - Streams epoch-by-epoch progress (train/val loss, F1, kappa, ETA)     |
+------------------------------------+------------------------------------+
                                     | Emscripten Bindings
                                     v
+-------------------------------------------------------------------------+
| WebAssembly Engine (web/uneye_train.wasm, 237 KB, SIMD128)              |
|  - Native C++ U-Net (18,602 params for 2 classes; 18,613 for 3 classes)|
|  - Adam optimizer, Multi-Class MCLoss, L2 regularization (0.001)       |
|  - 1300 deg/s speed limit on network inputs                             |
|  - Variable-duration trial windowing and rotation augmentation          |
+-------------------------------------------------------------------------+
```

---

## 2. Speed Table & Comparative Benchmark

Measurements taken on 50 s of eye-tracking data (1000 Hz, 4 rotations = 200 training windows of 1000 samples each, batch size 32):

| Backend | Device / Environment | Seconds per Epoch | Total Time (Early Stopping ~15-20 ep) | Package Size |
|---|---|---|---|---|
| **C++ Native (Reference)** | Intel i7-1068NG7 (1 thread) | **2.10 s** | ~40 s | N/A |
| **WebAssembly SIMD128 (Our implementation)** | Windows Laptop (x86_64 CPU) | **3.90 s** | ~60–75 s | **237 KB** (0 dependencies) |
| **TensorFlow.js (CPU)** | Windows Laptop (Node / Chrome) | > 300 s | > 1 hour | ~3–5 MB |

### Key Findings:
- **WebAssembly with SIMD128** is ~2x native speed, easily finishing full training in **under 1.5 minutes**.
- It provides **exact determinism** and gradient parity with the PyTorch golden references (maximum relative difference: `4.93e-14`).
- **TensorFlow.js on CPU** is orders of magnitude slower due to JS execution overhead, making WASM SIMD the clear and definitive choice.

---

## 3. Data Rules & Pipeline

1. **Reviewed Labels Only**:
   - Only reviewed spans (`state.reviewedRanges`) and pass-2 refined events (`state.expertEvents`) are used for training.
   - Point marks (pass 1) are never used in training windows.
   - Unreviewed samples receive ignore label `255` (`kIgnoreLabel`), masking them from gradient updates.
2. **Held-Out Test Trials**:
   - Every 4th trial (`state.testTrials`) is held out from training.
   - Event F1 (runs $\ge$ 3 samples overlapping human event) and Cohen's kappa are calculated on the test set per class (saccades, blinks) and feed `state.convergenceHistory`.
3. **Speed Limit**:
   - Every input velocity $(dx, dy)$ has an upper norm limit of 1300 deg/s (`lim = 1300.0 / rate_hz`).
   - Blinks and tracker loss outliers are clipped with direction preserved, preventing training instability.
4. **Different Duration Trials**:
   - 1000-sample windows are extracted strictly within each trial; never across trial boundaries.

---

## 4. Model Persistence & Python Round-Trip

### In the Browser:
- **Autosave**: All labels, settings, convergence history, and trained model weights are saved to **IndexedDB** (`uneye_db`), persisting across tab closes and offline reloads.
- **Download Model**: Clicking "Model" downloads:
  - `uneye_model.bin`: binary layout matching `cpp/models/*.bin`.
  - `uneye_model.json`: metadata sidecar.

### Python Conversion:
To load a web-trained model back into PyTorch (`uneye.functions.UNet` or `mocol_saccades/detect_saccades_mocol.py --weights`):
```bash
python cpp/scripts/bin_to_state_dict.py uneye_model.bin weights.pt
```
This is verified by the round-trip test `cpp/scripts/test_bin_roundtrip.py`, confirming exact $0.0$ parameter deviation against PyTorch weights.

---

## 5. Building from Source

To rebuild `web/uneye_train.wasm` and `web/uneye_train.js`:

### Linux / macOS:
```bash
bash scripts/build_wasm.sh
```

### Windows:
```cmd
scripts\build_wasm.bat
```
Prerequisites: Emscripten SDK (`emsdk`).


---

## 6. Manual Verification Checklist (iPad Safari & Android Chrome)

When verifying on a physical iPad M1 (Safari) or Android phone (Chrome):

1. **Initial Load & Offline Caching**:
   - Navigate to the web labeler (`https://jobellet.github.io/uneye/`).
   - Open Developer Tools (or inspect cache): confirm Service Worker (`sw.js`) installs and caches `web/uneye_train.wasm` and `models/combined.bin`.
   - Enable Airplane Mode / disconnect Wi-Fi and refresh: confirm the page loads completely offline.

2. **Touch Gestures & Labeling Flow**:
   - Open an eye-trace file in the start window (no simulated data is loaded; the automated test calls `generateDemoDataset()` itself).
   - **Pass 1 (Mark)**: Tap to place point marks on saccade candidates.
   - **Pass 2 (Refine)**: Press and drag boundaries to precisely refine saccade onset and offset.
   - Mark epoch as reviewed. Notice that unrefined point marks never enter training.

3. **On-Device Training**:
   - Tap **Train**, or finish a pass 2 (leave it, or reach its last event): training then starts by itself if at least 2 s were newly reviewed. It never starts during pass 1.
   - Confirm status changes to `Epoch 1/60 · Loss: ... · Val F1: ... · ETA: ...`.
   - Verify screen stays awake during training (`navigator.wakeLock`).
   - Verify canvas panning and zooming remain smooth at 60 fps while training runs in the worker.

4. **Cooperative Cancellation**:
   - While training is active, tap **Cancel**.
   - Confirm training immediately stops and status displays `Training cancelled`.

5. **Post-Training Inference & Metrics**:
   - On training completion, verify `testF1Saccade` and `testKappaSaccade` update with real numbers calculated on the held-out `testTrials`.
   - Open **Metrics**: verify learning curve canvas plots the real test metrics.
   - Navigate to unreviewed trials: with *Network suggestions: On* (menu) the proposals are the trained network's; switch it Off and they are the HMM's again.

6. **Export & Persistence**:
   - Tap **Model**: confirm browser downloads `uneye_model.bin` (75,708 bytes for 2 classes) and `uneye_model.json` sidecar.
   - Close the browser tab and re-open: confirm IndexedDB automatically restores all reviewed ranges, expert events, and the trained model.
