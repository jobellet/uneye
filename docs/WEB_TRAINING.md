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
