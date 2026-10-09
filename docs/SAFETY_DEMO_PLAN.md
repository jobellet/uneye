# Plan: from research Python to predictable, safe C++ (working document)

This repository started as the U'n'Eye saccade detector. The C++ part (`cpp/`) turns the research code into a streaming engine. This
plan describes the second goal and the order of work, so that everybody (and every Claude session) working here knows the aim.

**Aim.** Show, with measured evidence, how a Python research model becomes C++ code that is predictable, tested and *safe even when the model is
wrong or sees data it was never trained on*. Domain: online saccade detection from 1 kHz eye-tracker data (a signal-processing problem with
hard per-sample deadlines). The method (zero-heap hot path, numerical equivalence tests, deterministic guard around a network, fault injection)
is the point; the domain is the example.

**Transparency.** This work is AI-assisted (Claude Code). Every number in a figure or document must come from a script that was run, with machine,
compiler, flags and seed written down. Nothing here is a certification or a compliance claim; documents named "IEC 62304 / ISO 14971 inspired"
are illustrations only.

## Rules
- No invented numbers. Be honest about limits (no WCET claim, no bit-exactness claim beyond what is measured, and under which conditions).
- Simple, readable C++17, comments explain the WHY. Small commits, one topic each. Keep the ONNX engine and existing results.
- One command per figure, fixed seeds.

## Steps (stop and report after each)
0. **Audit** (done, see below).
1. Zero-heap hot path, provable by a test that hooks `operator new/delete` and `malloc`.
2. Python vs C++ numerical equivalence (layer by layer, abs / rel / ULP, float32 and float64) + a ctest with a documented tolerance. Figure 1.
3. Per-sample latency (median, p99, p99.9, max over >= 1e6 samples) against the 1 ms budget. Figure 2. Not a WCET.
4. **Deterministic safety guard** (`cpp/safety_guard.*`): input plausibility, output physiology limits, simple confidence / OOD score, deadline watchdog,
   NORMAL -> DEGRADED -> SAFE with hysteresis, classical fallback in DEGRADED, no events in SAFE, reason codes for every decision.
5. OOD / fault injection: network only vs network + guard; dangerous outputs must be zero with the guard; failures are shown, not hidden. Figures 3, 4.
6. Architecture figure (Figure 5), `docs/safety/hazards.md`, `docs/safety/soup.md` (ONNX Runtime as SOUP).
7. Slide figures (German labels, PNG + SVG, one script), `docs/EXPLAIN.md`, `docs/QA.md`.
8. Minimal CI: cmake + ctest with ASan + UBSan, no ONNX.

> **Update 2026-10-09:** the work now continues in a session on the owner's own computer; the state, lessons and a detailed plan per step are in `docs/ROADMAP.md` (read it first). Step 1 is done.

## Division of work between sessions (superseded by ROADMAP.md)
- Cloud session: `cpp/` (engine, guard, tests, CI), `docs/`.
- Mac session (Apple M1, can train networks): Python training scripts (`online/`), retraining of the causal networks, `.py` versions of the notebook
  training code. Please do not edit `cpp/src` or `cpp/tests` without telling the other session; commit small and rebase often.

## Step 0 audit (read-only, measured 2026-10-09; machine Intel Xeon 2.1 GHz with FMA, gcc 13.3.0, `-O3 -march=native`)
- Per-sample path: `GazeEngine::push` (`cpp/src/gaze_engine.cpp:422`) -> `process` (:171), `fuse` (:145), `run_window` every 10th sample (:302), and
  `StepEngine::step` of the causal TCNs (`cpp/src/causal_engine.cpp:87`). `gaze_engine.cpp` uses no `std::vector`, `new`, `malloc`, string or iostream.
  The causal TCN allocates all history in the constructor.
- **Exception:** the native window network (`cpp/src/native_engine.cpp:133`, `infer`) calls `resize/assign` on 14 member vectors on every call; the first
  call allocates, later calls reuse capacity. The existing allocation test needs a 3000-sample warm-up and hooks only `operator new` / `new[]`
  (`cpp/tests/test_gaze_engine.cpp:17-20, 82-92`). The ONNX back-end allocates inside ONNX Runtime. -> to fix in Step 1.
- Types: network weights and inputs are float32 (export `astype("<f4")`, PyTorch default); `gaze_engine.cpp` uses double for positions, timestamps and
  forecasts; the narrowing double -> float happens where the network input `dx, dy` is built. No float64 Python reference exists yet.
- Parity test (`cpp/tests/test_parity.cpp`): compares raw softmax probabilities of one 200-sample window per model (and one causal sequence) with golden
  text files; absolute tolerance 1e-4 only, no relative or ULP check, no layer-by-layer check, no labels. Measured max |diff|: combined 6.3e-7, dataset1 1.6e-6,
  andersson 1.5e-5, causal stream 1.4e-6. `-march=native` + FMA contraction means other machines or flags can give different last bits; cross-machine
  equality is not tested (the determinism test compares two engines in one process).
- Input handling: NaN/inf, range, speed jumps, backwards time, gaps (synthetic invalid bins, reset above 50 missing) are handled (`gaze_engine.cpp:425-444`).
  The sampling rate is only checked against the configured `fs_hz` (:96), not measured. No saturation or flat-signal check yet.
- CI: none. Sanitizers: CMake option `UNEYE_SANITIZE`, run by hand; ASan + UBSan passed on `test_gaze`; the ThreadSanitizer run on the final code did
  not finish, so `cpp/GAZE_ENGINE.md` ("TSan clean") is not verified for the current code. `test_parity` has not been run under a sanitizer.
- Timing numbers in `cpp/GAZE_ENGINE.md` and the "53 us per sample" in PR #2 come from earlier runs on a shared cloud machine and have not been re-measured.
