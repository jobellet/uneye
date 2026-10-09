# ROADMAP and hand-over (read this first)

Written on 2026-10-09 by the cloud Claude Code session so that another session (the one running on the owner's own computer) can continue **to the end**
without the cloud session. `docs/SAFETY_DEMO_PLAN.md` holds the aim and the rules; this file holds the state, the insights, and a concrete plan per step.
The owner speaks French; documents and code comments are English; figures for the slides get **German** labels (see Step 7).

## Progress since this hand-over (local M1 session, 2026-10-09) — read before section 4
All steps 2-8 have a first complete version on `master`; numbers in each commit message and in `docs/safety/hazards.md`.
- Step 2: output-level equivalence of the causal TCN (`cpp/scripts/equivalence.py`, ctest `equivalence`). Layer-by-layer: NOT done.
- Step 3: `cpp/tools/bench_latency.cpp` (M1: median 78 us, 7 of 1e6 pushes > 1 ms). Step 4: `cpp/safety_guard.*`, ctest `safety_guard`.
- Step 5: `cpp/tools/fault_injection.cpp` (24 cases, guard: 17 with 0 dangerous outputs after the OOD noise limit was calibrated on set A
  to 16 deg/s by `cpp/scripts/calibrate_ood.py`; 4 clear failures listed in hazards.md).
- Step 6: `docs/safety/hazards.md`, `soup.md`. Step 7: `docs/slides/make_figures.py` (Fig. 1-5, German), `docs/EXPLAIN.md`, `docs/QA.md`.
- Step 8: `.github/workflows/ci.yml` (Release + ASan/UBSan), first run green on ubuntu-24.04 (commit 55e9d8e).
- The Mac has no Xcode tools: C++ is built with zig c++ (clang 22) from `.venv`, git is the one inside GitHub Desktop.
- Open: layer-by-layer parity, the 4 unsolved perturbations (bursts, x10, 500 Hz as 1 kHz), GAZE_ENGINE.md refresh, TSan run.

## 0. The two goals (unchanged)
1. A working online saccade/microsaccade detector: causal network + C++ streaming engine.
2. Evidence that research Python becomes C++ that is predictable, tested and safe even when the model is wrong or sees data it was never trained on.
   Deliverables: 5 figures, `docs/safety/*`, `docs/EXPLAIN.md`, `docs/QA.md`, minimal CI. The owner will present 4-5 figures on slides on Monday 2026-10-12.

**Rules (non-negotiable):** no invented numbers (every number comes from a script that was run; write machine, compiler, flags, seed next to it);
no certification / WCET / bit-exactness claims beyond what was measured and under which conditions; simple readable C++17 with WHY-comments; small commits;
one command per figure with fixed seeds; keep the ONNX engine and existing results; keep the README note that this work is AI-assisted.
**Report after each step** in 4 parts: (1) files changed, (2) what was measured (numbers + machine + flags), (3) what is NOT shown / NOT guaranteed, (4) what the owner should check.

## 1. Repository map
| path | what |
|---|---|
| `uneye/`, `training/` | original U'n'Eye Python package (U-Net 1-D, non-causal) and pretrained weights |
| `data/` | the 4 labeled datasets (sets A for training, B for test) |
| `online/` | causal (past-only) TCN, training/eval scripts, architecture study (`README_architectures.md`, notebook `UnEye_online_architectures.ipynb`) |
| `free_saccade/` | label-free (self-supervised) detection study + notebook `UnEye_label_free.ipynb` (`RESULTS.md`) |
| `cpp/` | C++17 engines: `native_engine.cpp` (U-Net, no dependency), `causal_engine.cpp` (streaming TCN, one step per sample), `gaze_engine.cpp` (the composite engine: networks + physics guard + watchdog + rolling buffer + forecast), `detector.cpp` (older window-based `StreamingDetector`), `onnx_engine.cpp` (ONNX Runtime back-end, optional) |
| `cpp/models/` | exported weights `.bin` (float32, magic `UNEY` U-Net, `UNCZ` causal TCN) and `.onnx`: `combined`, `dataset1-3`, `andersson`, `synthetic`, `causal` (L=0 TCN), `tcn_l10` (10 ms lookahead TCN) |
| `cpp/tests/` | `test_parity.cpp` (vs PyTorch golden files), `test_gaze_engine.cpp` (7 sections: no alloc, determinism, hostile input, SPSC queue, window stage...), `test_zero_heap.cpp` (new, Step 1) |
| `cpp/GAZE_ENGINE.md` | design, guarantees, evidence tables, limits of the composite engine |
| `docs/` | this roadmap, the plan |

Build and test (Linux, gcc 13): `cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 && (cd build && ctest)`; ~2 min (`gaze_engine` test is ~100 s).
Sanitizers: `-DUNEYE_SANITIZE=address` (ASan+UBSan) or `thread`. ONNX optional: `-DONNXRUNTIME_ROOT=...` (not needed for anything below).
Replay tool: `build/gaze_replay --x X.csv --y Y.csv --fast cpp/models/causal.bin --refine cpp/models/tcn_l10.bin --window cpp/models/combined.bin ...` (options: `--mode fused|nn|heur`, `--fault nan|const1|const0|stuck|random|badsum`, `--loss`, `--spike`, `--trials`, `--lambda*`, `--floor*`); `online/gaze_score.py` scores its output.

## 2. State of the work (what exists, measured, as of commit "GAZE_ENGINE.md: record ASan+UBSan result...")
Machine for all numbers below unless stated: cloud VM, Intel Xeon 2.1 GHz (FMA), 4 cores, gcc 13.3.0, `-O3 -march=native`.

**Detection results (research part, from earlier sessions, scripts in `online/`)**
- Causal TCN trained from scratch is the best causal model from ~50 labeled trials on (kappa 0.70 at 50, 0.74 at 100, 0.75 at 300). Lookahead L (label of sample t-L emitted at t) is the main accuracy lever: L=10 ms TCN reaches kappa 0.855 on the all-label test set. JEPA / S4D / GRU were dropped (no gain or unstable). Weak labels from Engbert-Kliegl help only at very small N.
- Composite engine (`gaze_engine.cpp`): fast TCN (age 0) + 10 ms TCN (revises bin t-10) + original U'n'Eye on 200 samples every 10 ms (finalizes bins at age 80, 50/50 blend); final labels kappa 0.865 on the untouched test set; with the window network every 10th push also runs it (~0.3 ms), leaving only ~25 % headroom in the 1 ms sample period on the cloud VM (see `cpp/GAZE_ENGINE.md`, "Timing"). Those timing numbers are from earlier runs on a shared machine and are **to be re-measured in Step 3**.
- Label-free study (`free_saccade/RESULTS.md`): classic detectors reach kappa 0.4-0.8 without labels (HMM best on 3 of 4 sets); label-free *scores* correlate only weakly with real kappa (mean within-dataset Spearman +0.21 for the combined score, +0.40 for the injection benchmark in the user's own run). The user's own Kaggle run so far was on demo data, not their dataset.

**Engineering state of `cpp/` (Step 0 audit, 2026-10-09)** - see also the audit section in `docs/SAFETY_DEMO_PLAN.md`
- Hot path: `GazeEngine::push` (`gaze_engine.cpp:422`) -> `process` (:171), `fuse` (:145), `run_window` every 10th sample (:302), `StepEngine::step` (`causal_engine.cpp:87`). `gaze_engine.cpp` has no `std::vector`, `new`, `malloc`, string or iostream. Models are created before the engine; the engine constructor may allocate/throw, nothing after it.
- **Step 1 DONE.** `NativeEngine` sizes all 14 work buffers in its constructor (before: 15 allocations on the first `infer()`; the engine hid it with a warm-up call). `tests/test_zero_heap.cpp` hooks `operator new/delete` + glibc `malloc` family (`__libc_malloc`...), counts from the end of construction with **no warm-up** over 100 000 pushes (NaN, 1e9 value, backwards time, gap, blink, 10 008 window-network runs), and has a control block that allocates on purpose so a "0" cannot come from a dead hook. Result: 0 allocations (also under ASan+UBSan, where only `operator new` is counted because the sanitizer owns malloc). Parity numbers unchanged (6.3e-7, 1.6e-6, 1.5e-5, 1.4e-6).
- Types: weights/inputs float32 everywhere (export `astype("<f4")`; PyTorch default float32). `gaze_engine.cpp` uses double for positions, timestamps, forecasts; narrowing double->float where the network input `dx, dy` is built. No float64 Python reference exists yet.
- Parity test (`test_parity.cpp`): one 200-sample window per model + one causal sequence vs golden text files (`%.9g`), absolute tolerance 1e-4 on **probabilities**, no relative/ULP, no layer-by-layer, no labels. Golden files are made by `cpp/scripts/make_golden.py` (U-Net) and `online/export_causal.py --golden` (causal).
- Existing guard logic is **inside** `gaze_engine.cpp` (not a separate file): input guard (`:425-444`: NaN/inf, |x|>90 deg, jump >1000 deg/s, backwards time, gaps with synthetic invalid bins, reset above 50 missing), blink margins, robust noise estimate, velocity-threshold detector (weak/normal/strong evidence), veto ("network says saccade but no movement") and override, network watchdog (non-probability output, stuck output, flicker EMA, veto-rate EMA, missed clear saccades EMA, hysteresis + 1000-sample hold), `Health Ok/Degraded/Failed`, `Source` enum (who decided a bin), flags bit mask. **Step 4 should formalize this as `cpp/safety_guard.*` (extract + extend), not start from zero.** Missing today: saturation / flat-signal check, measured (not configured) sampling interval, physiological *output* limits as a separate final check (max velocity, amplitude, duration, refractory time, no offset-without-onset / no overlap), a simple OOD score, a deadline watchdog, and the three-state NORMAL/DEGRADED/SAFE machine with "SAFE emits no events".
- Sanitizers: ASan+UBSan pass for `test_zero_heap`, `test_parity`, and (earlier code state) `test_gaze`. **ThreadSanitizer on the current code is NOT confirmed** (the run never finished). No CI exists yet.

## 3. Lessons and traps (save yourself the time)
- **macOS / Apple Clang:** `test_zero_heap.cpp` uses glibc `__libc_malloc` etc.; this does not exist on macOS. Make the malloc hook conditional (`#if defined(__GLIBC__)`) and fall back to counting `operator new/delete` only (same as the sanitizer build). `-march=native` on Apple Silicon = NEON, FMA always on; results can differ in the last bits from x86. Do not claim cross-machine bit-equality: **measure** it (Step 2).
- **Floating point:** GCC contracts `a*b+c` into FMA by default (`-ffp-contract=fast`) when the target has FMA; sums depend on operation order. For Step 2 compare builds with `-ffp-contract=off` and `-O0`/`-O3` to see what changes. State exactly under which conditions results are equal.
- **Never `pkill -f <script name>` from a shell command that contains that name** (it kills the shell). Kill by PID.
- Long runs (ctest ~2 min, ASan test ~3 min, TSan > 15 min) must run in the background; do not chain `sleep`.
- `gaze_engine.cpp` is compiled with `-Wall -Wextra -Werror` (must stay warning-free). `test_zero_heap.cpp` has a `#pragma GCC diagnostic ignored "-Wmismatched-new-delete"` on purpose (it is an allocator replacement).
- The score in `free_saccade/intrinsic.py` (ILS) is gameable and a weak ranker; do not use it for safety claims.
- Files under `/tmp` of the cloud session are gone; nothing needed lives only there. Golden files and model `.bin` files are in the repo.
- Trained causal weights: `online/weights_causal` (+ `online/runs`, gitignored). Re-exporting a model changes golden files: regenerate with the scripts above and re-run parity.

## 4. Plan for the remaining steps (Steps 2-8), with concrete guidance
Each step ends with the 4-part report. Commit small; push to `master` (the owner explicitly allows direct merge, no PR approval needed).

### Step 2 - Python vs C++ numerical equivalence (Figure 1)
- Build a **Python float64 reference** and a **float32 reference** of the *same* networks (`online/causal_net.py`, `uneye/functions.py::UNet`): `.double()` copy of the PyTorch model, hooks to capture every layer's output (conv -> ReLU -> BN -> residual for the TCN; the U-Net blocks).
- Add a C++ debug entry that dumps the same intermediate tensors (a test-only function; do not slow the hot path): e.g. `StepEngine::step_debug(...)` or a compile-time `UNEYE_TRACE` hook writing layer outputs into a caller-provided buffer.
- Inputs: not just one golden window: use many windows from the datasets (fixed seed), plus synthetic stress inputs (zeros, huge values, denormals).
- Metrics: max abs, max relative (guard against division by ~0), ULP difference (use `numpy.spacing`/integer reinterpretation) per layer and at the output, float32 and float64 separately. Label-level agreement: % identical labels after threshold 0.5 and number of differing events (use `online/metrics.py::match_events`); explain what "identical" means and under which conditions (compiler, flags, machine) equality is exact. Run at least `-O3 -march=native`, `-O3 -ffp-contract=off`, and (if possible) another machine (the owner's Mac) to show what changes.
- New ctest that fails above a **documented** tolerance (put the tolerance and its derivation in the test file and in the report: e.g. derived from the measured distribution with margin, not guessed).
- Figure 1: histogram (log scale) of errors Python-vs-C++, float32 vs float64. Script: `docs/slides/make_figures.py` (one script for all figures, one function per figure).

### Step 3 - Timing (Figure 2)
- New tool `cpp/tools/bench_latency.cpp`: pre-generate the input, warm up, then time each `push()` with `std::chrono::steady_clock` (clock calls outside the engine, not inside!) over >= 1e6 samples; save raw latencies to a binary file; report median, p99, p99.9, max. Record machine (`lscpu`), compiler, flags, whether CPU pinning/performance governor was used.
- Draw the 1 ms budget (1 kHz). State clearly: observed max is **not** a WCET, a PC/Mac is not the target device, OS jitter exists (earlier runs saw single pushes of 1-55 ms on the shared cloud VM). Report with and without the window network (every 10th push is slower: show the periodic spike pattern in the time series).
- Figure 2: latency histogram or time series with p50/p99/max and the budget line.

### Step 4 - Deterministic safety guard `cpp/safety_guard.hpp/.cpp` (most important)
Principle: simple, deterministic, independent of the network; the network can never override it; when unsure -> safe state. Understandable on one slide. No allocation, fixed cost per sample, `noexcept`, `-Wall -Wextra -Werror`.
- (a) Input plausibility: NaN/inf, physical range, **saturation** (value stuck at a rail / clipping count), **flat/stuck signal** (variance below a floor for N samples), timestamp gap and **measured** sampling interval vs nominal, spike beyond physical limit (reuse the existing checks, extract them).
- (b) Output plausibility (physiology) on every event the engine emits: max peak velocity (~1000 deg/s), amplitude and duration limits, minimum inter-event refractory time, onset/offset consistency (no offset without onset, no overlap). Violation -> event dropped + reason code; repeated violations -> state change.
- (c) Confidence / OOD: entropy or margin of the network output + running mean/variance of the input against the training range (store training range as constants exported with the model; say how it was computed). Keep it explainable in 30 s.
- (d) Deadline watchdog: the engine API gets a budget argument or a caller-supplied elapsed-time input (the engine itself must not read a clock); if the result is late the safe value is output.
- State machine NORMAL -> DEGRADED -> SAFE with documented transition rules and hysteresis (back to NORMAL only after N consecutive clean samples; SAFE -> DEGRADED -> NORMAL, never directly). DEGRADED: classical deterministic detector (the existing velocity-threshold/Engbert-Kliegl style code). SAFE: output "invalid", **emit no events, never guess**. Every decision carries a reason-code enum (`enum class Reason : uint8_t`) and is available for logging.
- Tests (`cpp/tests/test_safety_guard.cpp`): one unit test per check; transitions + hysteresis; network output ignored in SAFE (feed a network that says "saccade" at p=1 forever; the output must contain no event); determinism (bit-identical replay); zero-heap test extended to the guard.
- Keep the existing `Health`/`Source`/flags API stable or migrate it cleanly (update `gaze_replay`, `test_gaze_engine.cpp`, `GAZE_ENGINE.md`).

### Step 5 - OOD / fault injection (Figures 3 and 4)
- Script `cpp/tools/fault_injection` (C++ driver) or Python wrapper around `gaze_replay` with a fixed seed: perturbations = increasing noise, burst interference (analogy: electromagnetic disturbance), blink-like dropouts, NaN segments, saturation, slow drift, amplitude scaling x0.1 and x10, different sampling rate (e.g. 500 Hz data fed as 1 kHz and with correct timestamps), spikes. Compare `--mode nn` (network only) vs fused-with-guard.
- Metrics per perturbation and level: false events per minute, missed true events (use labeled data, set B), fraction of time in DEGRADED/SAFE, and **dangerous outputs** = events violating the physiological limits: must be **0 with the guard**. If the guard misses a case, **show it and say so**.
- Figure 3: false events/min per perturbation, network only vs with guard. Figure 4: one example time series (perturbed signal, raw network output, guard state, final output) with annotations where the guard switches.

### Step 6 - Architecture figure + mini traceability pack (Figure 5)
- Figure 5: Python research (training, evaluation) -> export -> C++ engine (zero-heap) -> safety guard -> output, with arrows for the three verification tests (parity, timing, fault injection) and their test IDs.
- `docs/safety/hazards.md`: table of 6-10 rows (hazard e.g. "wrong event during an interference burst", cause, guard measure, REQ-xx, TEST-xx linking to real test names, residual risk). Label it "illustrative, inspired by IEC 62304 / ISO 14971, not a compliance claim".
- `docs/safety/soup.md`: ONNX Runtime as SOUP: why the native engine exists (no hidden allocations: ORT allocates inside `Run`; version pinning; qualification effort), what would be needed to qualify it.

### Step 7 - Slides and explanation
- `docs/slides/make_figures.py`: all figures 16:9, large fonts, colour-blind-safe palette (e.g. Okabe-Ito), **German** labels and titles, PNG + SVG into `docs/slides/figs/`; one command.
- `docs/EXPLAIN.md`: per figure 3 simple German sentences the owner can say out loud + the one honest limit.
- `docs/QA.md`: 10 likely questions (zero-heap, float32 vs float64, why a guard, why not trust the network, what is SOUP, what was NOT done, ...) with short answers, German and English.

### Step 8 - Minimal CI
`.github/workflows/ci.yml`: cmake build + ctest on the native engine, one job normal Release, one with `-DUNEYE_SANITIZE=address` (ASan+UBSan); no ONNX; the long `gaze_engine` test may need a time limit or a shorter input option. TSan job optional (slow). README badge optional.

### Also open
- Refresh `cpp/GAZE_ENGINE.md` after Steps 2-4 with real numbers; complete the TSan run on the current code and record it only if it passes.
- Add to README: a short "AI-assisted" transparency note (not done yet).
- `free_saccade`: the user still has to run `UnEye_label_free.ipynb` with their Kaggle dataset attached (`eye_tracking_data`, `subject_h5_metadata.csv`, h5 `x` = (n_seq, L, 3) = x, y, dt). Not on the critical path for Monday.

## 5. Facts about the owner's situation
- Deadline: slides with 4-5 figures needed by Monday 2026-10-12 morning (interview in German). Priority order = step order; **Step 4 and Step 5 are the most valuable** for the story; if time is short do Steps 4-5-6-7 before polishing 2-3.
- The owner can train networks on an Apple M1 MacBook Air (the local session): if retraining is needed (e.g. a causal TCN trained with float64 reference or different export), use `online/train_causal.py` (see `online/README.md`); a `.py` version of the notebook training code is being prepared there.
- Everything the owner presents must be explainable line by line: prefer simple code over clever code.
