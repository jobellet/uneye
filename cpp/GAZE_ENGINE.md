# GazeEngine: a deterministic streaming gaze-state engine

One raw gaze sample in, at **every** sample out:

| output | what | how |
|---|---|---|
| state of the current bin | fixation / saccade / blink / invalid, a confidence, and **who decided** (guard, heuristic, network, veto, override) | fast causal TCN, gated and checked by the physics layer |
| revised labels | **three stages**: fast label (age 0); the bin 10 ms ago relabeled by a second TCN trained with 10 ms of lookahead; the bins that reach age 80 ms finalized by the original U'n'Eye run on the last 200 samples every 10 ms, blended 50/50 with the previous label. Blink margins are applied retroactively. `Output::revised_first/last` says which bins changed, `BinView::stage` (0/1/2) and `final` how far a bin has come | rolling buffer with revisions |
| segmentation of the last 100 bins | saccade / microsaccade events: onset, offset, amplitude, peak speed, direction, provisional or final | `snapshot()` |
| position 10 and 20 ms ahead | location + Laplace scale per axis (an 80 % interval is +-1.61 scale), flag if it comes from the ballistic tables | fixed tables (saccade in progress) or "stay + measured scatter" |
| health | Ok / Degraded (network not trusted -> heuristic fallback) / Failed (no valid input for 500 ms, unsupported rate) and reason flags | watchdog |

The neural networks are **one input among several**. A hard-coded physics layer (input guard, velocity-threshold detector with a robust noise estimate, fixed landing tables) can veto them ("no movement in the data -> no saccade") and replaces them when a watchdog finds them misbehaving.

```
raw sample -> [input guard] -> velocity, robust noise -> [velocity-threshold detector: weak / normal / strong evidence]
                 |                                  \
                 |             fast TCN (age 0) ------> [fusion + veto + override] -> label of the bin  -> ring buffer (512 bins)
                 |             TCN with 10 ms lookahead -> [fusion] -> revises the bin 10 ms ago ----^          |
                 |             U'n'Eye on the last 200 samples, every 10 ms -> [blend + same gate] -> finalizes the bins at age 80 ms
                 |                                                                                          snapshot(): last 100 bins + events
                 |             [network watchdog: raw output judged at all times: invalid / stuck / flicker / veto rate / missed clear saccades]
                 \-> blink flag, gaps, resets                        [forecast: ballistic tables if a saccade is in progress, else stay]
```

## Run-time rules (what `push()` and `snapshot()` do and never do)
* single owner thread; `noexcept`; **no heap allocation**, no I/O, no clock, no randomness, no recursion; all loops bounded by compile-time constants (ring 512 bins, noise window 200, at most 50 synthetic samples per gap);
* all state in fixed-size arrays; models and the engine are created once (`GazeEngine` constructor may allocate and throw; nothing after it);
* a sample feeder in another thread must use `SpscQueue` (lock-free, wait-free, fixed capacity); the engine itself has no locks and no shared state;
* identical input gives bit-identical output (no clock, no thread, no random; floating-point is deterministic for the same binary and CPU architecture).

## Guard rails
| situation | what happens |
|---|---|
| non-finite value, |x|,|y| > 90 deg, jump faster than 1000 deg/s, timestamp backwards | sample is **invalid**; flagged (`NonFinite`, `OutOfRange`, `TooFast`, `TimeBackwards`); networks are fed zeros so that their state stays aligned |
| 3+ invalid samples in a row | **blink**; the 30 ms before it are **revised** to blink, the 30 ms after it stay blink; 1-2 invalid samples are a glitch (state Invalid, no margin) |
| missing samples (timestamp gap) | synthetic invalid bins; more than 50 missing: full **reset** (`Reset` flag, indices restart at 0) |
| no sample for > 500 ms, or sampling rate not ~1 kHz | health **Failed** (networks and tables are for 1 kHz data) |
| the network says saccade but the data show no movement (weak evidence in +-4 ms) | **vetoed**: label is fixation (`NnVetoed`) |
| clear, very fast movement (strong evidence) and the network says fixation | **override** to saccade (`HeuristicOverride`); thresholds are strict, rarely active |
| the window network (U'n'Eye) outputs anything that is not a probability, anywhere in the window | the whole run is discarded; the previous labels stay (`Stats::window_invalid`) |
| the watchdog distrusts the networks (health Degraded) | the window network is **not applied** (a third network must not overrule the heuristic fallback); bins are only marked final |
| network output not finite / not a probability / bit-identical for 1000 samples / label flickering (> 0.1 flips per sample) / says saccade without movement too often / misses 4 clear fast movements in a row | network **untrusted**: labels come from the heuristic, health Degraded; the watchdog keeps judging the raw output, trust returns only after 1000 samples and clean behaviour |
| blink, invalid sample | no forecast (`NoForecast`) |

## Evidence (all numbers from this repository's tests; test set = set B, never used for any choice)
**Tuning provenance.** Detector and gate thresholds were chosen on 300 trials of set A (datasets 1+2) only; the forecast uncertainty was calibrated on the same trials (stay scale x0.8, ballistic scale x4.5 in the first 10 ms after onset and x1.8 later); the landing tables were fitted on set A with human onsets.

**Labels, 600 test trials (datasets 1+2, 1 kHz), sample-level kappa vs human labels**

| mode | age 0 (fast) | age 30 ms (TCN 10 ms) | age 95 ms (final, with U'n'Eye) | event F1 (age 0 / 30 / 95) |
|---|---|---|---|---|
| fused, three stages (production) | 0.763 | 0.842 | **0.865** | 0.885 / 0.870 / **0.906** |
| fused, two stages (no window network) | 0.763 | 0.842 | 0.841 | 0.885 / 0.870 / 0.870 |
| network only (no guards, two stages) | 0.771 | 0.847 | - | 0.891 / 0.867 / - |
| heuristic only (the fallback) | 0.465 | 0.518 | - | 0.727 / 0.723 / - |

The final stage raises kappa by 0.024, event F1 by 0.036 (recall 0.951, precision 0.864) and lowers false alarms from 31 to 23 per minute; the two networks make different mistakes, so their 50/50 average beats either alone (on the tuning trials: the window network alone at age 80 gave kappa 0.840, the blend 0.853, the 10 ms network alone 0.830). Age 80 ms and the 50/50 blend were chosen on 300 trials of set A only (ages 20, 30, 50, 60, 80 and blends 0, 0.3, 0.5, 0.7 tried). Final labels therefore exist 80-90 ms after the sample; the guards cost about 0.005-0.008 kappa when everything is fine and are what keeps the output usable when it is not (below). Alarm delay (detection of a saccade onset) is about 9 ms at age 0.

**Persistent network faults** (one uninterrupted stream of 300 trials, fault starts after 60 000 samples, scored on trials 100-299): output NaN, probabilities not summing to 1, always saccade, always fixation, stuck constant, random: **all end as the heuristic fallback** (kappa 0.465 at age 0, 0.543 final, health Degraded 100 %; re-checked with the window stage on for NaN, always-saccade, always-fixation and random), against kappa 0.767 / 0.864 without a fault. Dropped samples (2 %) or artifacts (0.2 % of samples) change sample-level kappa by at most 0.02.

**Forecast** (600 test trials, error = distance between predicted and true position, degrees; "stay" = current position)

| situation | horizon | RMSE stay | RMSE engine | 80 % interval covers |
|---|---|---|---|---|
| stable fixation | 10 ms | 0.094 | 0.094 | 0.82 |
| first 10 ms of a saccade | 10 ms | 0.520 | **0.435** | 0.56 (overconfident) |
| later in a saccade | 10 ms | 0.412 | **0.319** | 0.79 |
| later in a saccade | 20 ms | 0.596 | **0.504** | 0.80 |
| first 10 ms of a saccade | 20 ms | 0.903 | 0.893 | 0.52 (overconfident) |
| saccade about to start | 10 / 20 ms | 0.165 / 0.476 | same | 0.69 / 0.54 |

The beginning of a saccade is **not predictable** from the data before it starts, and only weakly in its first 10 ms; a ballistic table beats "stay" by 16-23 % once the movement is under way (it also beat a neural landing predictor in the same comparison).

**Micro- vs larger saccade** (decided from the predicted final amplitude, threshold 1 deg): 20-30 ms after the onset 89 % correct, 30-60 ms 95 % (majority class 66 % / 57 %); before about 15 ms it is no better than chance. For a "stimulus after a microsaccade" trigger the earliest *reliable* class decision is therefore about 20 ms after the onset; the bare detection is available after about 9 ms.

**Engineering checks** (`ctest`, `tests/test_gaze_engine.cpp`)

| claim | test | result |
|---|---|---|
| no heap allocation after construction | `tests/test_zero_heap.cpp`: hooks `operator new/delete` and glibc `malloc/calloc/realloc/free`; counts from the end of construction, **no warm-up**, over 100 000 pushes (+ snapshots) with NaN, absurd values, backwards time, gaps and a blink, all three networks (10 008 window-network runs inside the region); a control block checks that the hooks see deliberate allocations; the window network is also tested alone (1000 `infer()` calls from the first one). Before the fix the stand-alone window network made 15 allocations on its first call (the engine hid this with a warm-up run in its constructor); now all work buffers are sized in the network constructor | 0 allocations (also 0 with the older operator-new-only counter in `test_gaze`) |
| deterministic | two engines, 20 000 samples, bit-by-bit; and `reset()` replay | identical |
| survives hostile input | 400 000 pushes with NaN, inf, +-1e308, denormals, absurd / backwards / NaN timestamps, 400-sample dropout bursts, long gaps | no crash, every output within its invariants; run under **AddressSanitizer + UndefinedBehaviorSanitizer** with `_GLIBCXX_ASSERTIONS`: no report |
| thread-safe hand-off | reader thread -> `SpscQueue` -> engine thread, 30 000 samples | identical to the single-thread run; ThreadSanitizer run on the current code **not completed, result unknown** (an earlier code state was clean) |
| window stage | bins reach stage 2; a window network that outputs NaN changes no label and every run is counted invalid | as specified |
| guards | blink margin, glitch, out of range, backwards time, unsupported rate, NaN / always-saccade / no network on a motionless eye | all as specified (no saccade is ever declared on a motionless eye) |
| compile hygiene | `-Wall -Wextra -Werror` on the engine source, at -O3, and with the address+undefined sanitizers (thread sanitizer: see above) | clean (a GCC false-positive under ThreadSanitizer was removed by filling the output in place; the output was verified bit-identical afterwards) |

All test sections pass in the normal build; the AddressSanitizer + UndefinedBehaviorSanitizer run of `test_gaze` passed on the previous code state; the run of the new zero-heap test under the sanitizers is still pending. ThreadSanitizer on the current code is NOT confirmed. This is evidence from synthetic and replayed data on one machine, not a proof.

Timing (replay of 600 000 samples, `-O3 -march=native`): with two networks `push()` is median 0.12 ms, p99 0.24 ms, p99.99 about 0.35-0.5 ms. **With the window network every 10th push also runs it (about 0.3 ms): median 0.13 ms, p99 about 0.45 ms, p99.99 about 0.75 ms, i.e. only about 25 % headroom inside the 1 ms sample period at 1 kHz.** If that is too tight for the target machine, use a faster build of the window network (ONNX Runtime ran it in 0.12 ms), a larger `window_hop` (fewer, not shorter, spikes) or leave the window network out (`Models::window` null; labels are then final at 30 ms, kappa 0.841). Occasional single pushes of 1-55 ms were seen (1-2 per 150 000, different in every run) on the shared cloud machine used for testing; these come from the operating system (scheduling, page faults), not from the algorithm, which is O(1) per sample. A hard real-time guarantee needs a real-time kernel / pinned core / pre-faulted memory on the target machine; the engine does not provide that by itself.

## What this is NOT
* **Not certified.** "Biomedical grade" is a process (requirements, risk analysis, traceability, independent verification, e.g. IEC 62304 / ISO 14971), not a property of code. This engine follows good engineering practice and ships the tests above; it has no safety certification and must not be used where a wrong output can harm a person without that process.
* **1 kHz only.** The networks and the tables were trained / fitted on 1 kHz data (datasets 1 and 2); 500 Hz data (dataset 3) is refused (health Failed). Other rates need retrained models and refitted tables.
* The networks learned the human labelers' conventions; label noise limits what "agreement" can mean (the paper reports kappa 0.83 between two human coders).
* Smooth pursuit without saccades, PSOs and other classes are not modeled (they are "fixation" here). Blinks are detected by the **data pattern** (missing or impossible samples), not by a pupil signal: with a tracker that reports blinks/pupil loss, feed missing samples (NaN) in those moments.
* The forecast does not predict the *start* of a saccade, and its uncertainty is overconfident in the first 10 ms of a saccade.
* The fused labels need the **models in `models/causal.bin`, `models/tcn_l10.bin`** (full TCN, 131 ms receptive field) and, for the third stage, `models/combined.bin` (the original U'n'Eye trained on datasets 1+2+3). Replace them only with models trained with the same input convention.

## Use
```cpp
#include "uneye/gaze_engine.hpp"
using namespace uneye::gaze;
Models m; m.fast = uneye::make_causal_engine("models/causal.bin"); m.refine = uneye::make_causal_engine("models/tcn_l10.bin"); m.refine_delay = 10;
m.window = uneye::make_native_engine("models/combined.bin", 200);   // optional third stage: U'n'Eye finalizes bins at age 80 ms
GazeEngine eng(Config{}, std::move(m));          // init: may allocate / throw
Output out; Snapshot snap;
// per sample (single thread):
eng.push({t_us, x_deg, y_deg}, out);             // x, y in degrees; NaN for lost samples
if (out.state == State::Saccade) { /* ... */ }
if (out.forecast.at20.valid) { /* where the eye will be in 20 ms: out.forecast.at20.x, .y, +- 1.61 * .scale_x */ }
eng.snapshot(snap);                              // last 100 bins and the events, when you need them
```
Replay and fault injection: `gaze_replay` (`--window models/combined.bin`) (see `tools/gaze_replay.cpp`), scoring: `online/gaze_score.py`. Sanitizers: `cmake -B build_asan -DUNEYE_SANITIZE=address && cmake --build build_asan --target test_gaze && build_asan/test_gaze .` (same with `thread`).
Tables are regenerated with `scripts/fit_landing_tables.py` (set A only).
