# U'n'Eye real-time (C++ / ONNX)

C++ port of the U'n'Eye inference pipeline that labels saccades and microsaccades
**while the eye signal is being recorded** (sample by sample), instead of on a finished recording.

```
gaze sample (x,y) ──► diff (dX,dY) ──► 200 ms ring buffer ──► U-Net ──► P(saccade) per sample
                                         every `hop` samples (default 1)      (ONNX Runtime or native C++)
                                                                          │
                       events (onset/offset)  ◄── min-duration / merge ◄── threshold, after `lookahead` samples
```

## Parts

| File | Purpose |
|---|---|
| `scripts/export_model.py` | PyTorch weights → `.onnx` (fixed window) + `.bin` (raw weights for the native engine). Checks ONNX vs PyTorch. |
| `src/native_engine.cpp` | Dependency-free C++ re-implementation of the U-Net (same numbers as PyTorch, ~1e-6). |
| `src/onnx_engine.cpp` | ONNX Runtime back-end (optional). |
| `src/detector.cpp` | `StreamingDetector`: the real-time part (ring buffer, windowing, lookahead commit, event state machine). |
| `include/uneye/simulator.hpp` | Synthetic eye tracker with ground-truth labels (main-sequence saccades + microsaccades, drift, noise). |
| `tools/uneye_rt.cpp` | CLI: `--sim`, `--replay` (recorded data as a live stream, scored), `--stdin` (live input). |
| `tests/test_parity.cpp` | C++ engines vs PyTorch reference outputs (`scripts/make_golden.py`). |
| `models/*.onnx, *.bin` | Exported pretrained models (`combined` = datasets 1+2+3, the most general; `andersson` is 5-class). |

## Build

```
cd cpp
# native engine only (no dependencies)
cmake -B build && cmake --build build -j
# with ONNX Runtime (unpack a release from github.com/microsoft/onnxruntime/releases)
cmake -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.20.1 && cmake --build build -j
ctest --test-dir build --output-on-failure
```

Re-export models (needs `torch`, `onnx`, `onnxruntime`, `scikit-image`, `scikit-learn`, `scipy`, `matplotlib`):

```
python3 scripts/export_model.py ../training/weights_1+2+3 models/combined --window 200
```
Use your own weights file the same way. `--window` must be a multiple of `mp**2` (25).

## Use

```
# 1. simulated eye tracker, scored against ground truth, with a latency sweep
build/uneye_rt --model models/combined.onnx --sim --duration 120 --sweep-lookahead

# 2. recorded data replayed as if live (add --realtime to pace at the sampling rate)
build/uneye_rt --model models/combined.bin --replay ../data/dataset1/dataset1_1000hz_X_setB.csv \
    ../data/dataset1/dataset1_1000hz_Y_setB.csv ../data/dataset1/dataset1_1000hz_Labels_setB.csv --lookahead 10

# 3. LIVE: any program that prints "x y" (degrees, one line per sample) can be piped in
your_eyetracker_reader | build/uneye_rt --model models/combined.bin --stdin
#   EVENT cls=1 onset=760 offset=794 dur_ms=35.0 known_at=824          (add --per-sample for LABEL lines)
```

### Plugging in a real tracker (e.g. EyeLink link data)

Link the library and call `push()` from the tracker's sample callback:

```cpp
uneye::Config cfg; cfg.fs = 1000; cfg.hop = 5; cfg.lookahead = 10;
uneye::StreamingDetector det(uneye::make_native_engine("models/combined.bin", 200), cfg);
det.on_onset = [](int64_t i){ /* saccade started at sample i (known ~10-20 ms later) */ };
det.on_event = [](const uneye::Event& e){ /* finished event */ };
while (tracker.next_sample(x, y)) det.push(x, y);   // x,y in degrees; NaN for blinks
```
If your samples are in pixels or raw units, set `cfg.input_scale` to convert to degrees. The models
were trained on 1000 Hz (and 500 Hz for `dataset3`) data; other rates are not resampled.

## Knobs that trade accuracy for latency

* `lookahead` (samples): a sample's label is final once this many newer samples exist. Bigger = more context = a bit more accurate, later.
* `window_ms` (default 200 ms = 200 samples at 1 kHz): the time bin the network sees; `window_samples(fs, ms)` converts it (multiple of 25 samples). The ONNX file has a fixed window, so re-export with `--window` when you change it or the sampling rate.
* `hop` (default 1): the newest 200 ms bin is evaluated on every new sample.
* **Fast labels** (`on_fast`, `on_fast_onset`, CLI `--fast`): result of the network for the newest sample right after it arrives, with no lookahead. The committed labels (`on_label`, `on_event`) arrive `lookahead` samples later and are more accurate.
* `provisional_p()`: probability of the newest sample with zero lookahead, for gaze-contingent triggers.

## Caveats

* The network is non-causal by design; this port feeds it the most recent window and zero-pads at the right edge. The sweeps in the results below measure how much this costs.
* Pretrained weights were trained on parts of the provided datasets, so scores on those recordings are optimistic. Use your own labeled recording to validate before relying on it.

## Results: can saccades and microsaccades be detected in real time?

Yes, with about 10–30 ms delay and almost no accuracy loss. Measured with `uneye_rt` (window 200, hop 5, 1000 Hz, `combined` model).
Real data = `data/dataset1` setB, 1000 trials × 1 s, human labels, 1921 events of which 1591 are microsaccades (<1°). Scores are sample-level for the saccade class.

| Setting | F1 | Cohen's κ | Event recall (micro) | Onset known after (median / p95) |
|---|---|---|---|---|
| Original offline Python, whole trial | 0.868 | 0.856 | – | – |
| Online, lookahead 60 ms | 0.868 | 0.856 | 0.980 (0.976) | 68 / 77 ms |
| Online, lookahead 25 ms | 0.854 | 0.841 | 0.978 (0.973) | 33 / 42 ms |
| Online, lookahead 10 ms | 0.847 | 0.834 | 0.972 (0.966) | 19 / 29 ms |
| Online, lookahead 0 | 0.751 | 0.732 | 0.956 (0.947) | 16 / 26 ms |

* "Onset known after" = time from the true saccade start until the detector reports that a saccade is running (it needs 6 ms of consistent labels, the `min_sacc_dur` of the original pipeline). Lookahead 0 is fast but gives many false alarms (precision drops to 0.73); **lookahead 10 is a good default**.
* Simulated tracker (`--sim`, 120 s, 63 % microsaccades, noise 0.01°): F1 0.86–0.91 for every lookahead, all large saccades found, onset known after 12–17 ms (lookahead 0–10).
* Compute: one network call takes ~0.12 ms (ONNX Runtime) or ~0.28 ms (native C++), once every 5 samples, on a normal CPU core. This is far below the 5 ms between calls at 1 kHz, so there is a large margin. Run with `--realtime` to test pacing on your machine.
* `ctest`: C++ engines match PyTorch to < 2e-5.
* The weights were partly trained on these datasets; expect somewhat lower numbers on a new recording. Not tested yet: a real live tracker, blinks, other sampling rates.

### Fast mode: evaluate the newest 200 ms bin on every sample

`uneye_rt --model models/combined.onnx --replay ... --fast --lookahead 10` (200 trials of dataset1 setB):

| Output | F1 | Event recall (micro) | Event precision | Onset known after (median / p95) |
|---|---|---|---|---|
| Fast (no lookahead, every sample) | 0.71 | 0.958 (0.948) | 0.63 | 10 / 20 ms |
| Committed (lookahead 10 ms) | 0.86 | 0.961 (0.951) | 0.79 | 17 / 25 ms |

* Fast onset = 3 consecutive saccade labels (`fast_confirm`). It is about 7 ms earlier than the committed one but has more false alarms and its onset time is ~8 ms late/early, because the newest samples sit at the edge of the bin. Use it for triggers, and the committed labels for analysis.
* One call per sample: 0.12 ms (ONNX) / 0.27 ms (native), p99 0.2 / 0.5 ms, within the 1 ms sample interval. A 10 s paced run (`--realtime`) finished in 10.03 s.

## Causal network (past bins only)

`models/causal.bin` is a forward-only network trained for online detection (see `online/README.md`). It runs as a stateful filter
(one step per sample, 53 µs) and its label is final at once. Use it exactly like the other models:
`uneye_rt --model models/causal.bin --replay ... --fast` or `--stdin`. On the last sample of a 200 ms bin it reaches F1 0.79 / 0.79 / 0.60
on datasets 1 / 2 / 3, against 0.68 / 0.66 / 0.59 for the original U-Net.

### Causal model trained with lookahead
A causal model trained with `lookahead_ms = L` outputs at time t the label of sample t-L (see `online/README_architectures.md`). Export it with
`online/export_causal.py` (the `.json` next to the `.bin` gives the delay in samples) and run `uneye_rt --model causal.bin --stdin --label-delay <samples>`.
Check on 200 replayed dataset-1 trials with a TCN trained with L = 10 ms: kappa 0.73 without the delay flag (wrong alignment), 0.87 with `--label-delay 10`.
