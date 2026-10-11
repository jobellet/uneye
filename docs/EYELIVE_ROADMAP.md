# EyeLive — roadmap and design (to read BEFORE any more code is written)

Status: **design for review** (revision 2 of 2026-10-11 at the end of this file supersedes the parts it names: triggers, training workflow, classes, annotation). Nothing of the application is committed yet. What exists on disk (uncommitted): the downloaded GUI libraries (`cpp/third_party/`, outside git, `sh cpp/third_party/fetch.sh`),
a window that opens with Dear ImGui + GLFW + OpenGL through the zig toolchain (`cpp/app/hello_gui.cpp`), the HMM parameter export (`cpp/scripts/export_hmm.py`, `cpp/models/universal_hmm.txt`) and a first draft of the signal-processing core (`cpp/app/include/eyelive/dsp.hpp`,
does not compile yet: one type error). Decisions already taken with the owner: **Dear ImGui + GLFW**, **input = simple binary UDP + file replay** (EyeLink adapter later).

## 1. What the application must do (owner's requirements, 2026-10-11)
1. Take a stream of eye positions from a tracker (EyeLink-like), **record it in real time** to disk.
2. **Pre-allocate** memory / disk before the experiment: measure the sampling rate, ask the duration, compute the size, reserve it, so that nothing can run out during the experiment.
3. **Detect saccades / microsaccades online, as early and as precisely as possible**; start without any label (unsupervised), then improve with the expert.
4. The expert can **annotate / correct** detected events in the GUI; the detector **adapts to the expert's convention** (the way he or she reports saccades).
5. Several modes in separate tabs: a purely unsupervised mode (HMM) for experts who do not want to annotate, up to a mode trained on the expert's annotations.
6. **Triggers**: output a signal when a (micro)saccade is detected, or N ms after it (psychology experiments, gaze-contingent displays), with measured latency.
7. **Future position prediction** (where the saccade will land) for psychophysics and VR (foveated rendering): keep in mind, try to beat the state of the art.
8. Everything in C++, fast, runs on the lab computer (EyeLink labs are often on Windows: keep the core portable).

## 2. What already exists in the repository and will be reused
| asset | where | use |
|---|---|---|
| label-free **universal minimum-duration HMM** (no label, fitted on archive/) | `foundation/universal_hmm.py`, parameters exported to `cpp/models/universal_hmm.txt` | mode 1, works from the first second; benchmark mean F1 0.88 / kappa 0.66 |
| **acceleration-zero boundaries** on top of the HMM | `foundation/accel_boundaries.py` | mode 1b; mean 0.89 / 0.70 label-free, Andersson kappa 0.33 -> 0.60 (rule "first inflection" for datasets that exclude post-saccadic oscillations, "last zero" for those that include them) |
| **calibrator** (amplitude floor, onset / offset shift, merge gap; offset rule) fitted by grid search on a few annotated trials | `foundation/conventions.py` | mode 2: learns the expert's convention in milliseconds; on dataset 1, 10 annotated trials give F1 0.94 / kappa 0.83 (U'n'Eye trained on 50: 0.94 / 0.83) |
| **BiTCN** (non-causal, 2 channels vx, vy) and its C++ port | `cpp/src/bitcn.cpp` | accuracy mode with delay (sees +-257 samples: 257 ms of latency), used for review / post-hoc, not for triggers |
| **causal TCN streaming engine**, `GazeEngine` (zero heap, physics layer, `SafetyGuard`, ballistic landing tables for 10 / 20 ms ahead) | `cpp/src/causal_engine.cpp`, `gaze_engine.cpp`, `safety_guard.cpp` | fast online label, input guard (invalid / out of range / gaps / blink), first position prediction |
| synthetic gaze generator with labels | `cpp/include/uneye/simulator.hpp` | demo source, tests, latency tests |
| FiLM-conditioned BiTCN (16 numbers fitted on the target) | `foundation/film_bitcn.py` | research result: very good on d3 / d4 with 3-10 trials; needs gradient through the network (see open question Q3) |

Honest limits of the evidence (from `docs/BENCHMARK_SUMMARY.md`): the label-free chain is **below U'n'Eye trained in the dataset** on Andersson (0.79 / 0.60 against 0.88 / 0.82) and on d3 / d4 in kappa; microsaccade datasets (d1, d3) give label-free F1 0.85-0.93 / kappa 0.62-0.80. These are benchmark numbers on recorded 1 s trials, **not** measurements of the streaming system: latency, false-trigger rate and behaviour on a real tracker stream must be measured (milestones M2 / M3).

## 3. Architecture

```
 tracker ---UDP/TCP/file-->  [Source thread] --Sample--> [Recorder: preallocated mmap file, SoA] --count (atomic)-->
                                                              |                                       \
                                          [Analysis thread, hop 5 ms]                                  [GUI thread: reads, never writes]
                                  features -> detector(s) -> refine -> calibrator -> Event queue (lock-free) --> GUI event list / overlays
                                                              |
                                                       [Trigger thread: timer heap, steady clock] --> sinks (UDP, serial/TTL, file, stdout, callback)
```

### 3.1 Components (all C++17, core without GUI dependency; one binary per front-end)
| module | responsibility | notes |
|---|---|---|
| `types.hpp` | `Sample {t, x, y, flags}`, `Event {t_on, t_off, amp, peak_speed, dir, kind, provisional, final}` | degrees; pixels converted at the source with a configured geometry |
| `recorder` | **memory-mapped preallocated file** `session.eyelive`: header (magic, version, rate, capacity, count) + arrays t (f64), x, y (f32), flags / label / expert label (u8) | the file IS the buffer: data survive a crash, no copy, the GUI maps it too. Capacity = rate x duration x margin; `ftruncate` + `fcntl(F_PREALLOCATE)` / `posix_fallocate` reserve the disk, pages are touched at start (`mlock` best effort). 19 bytes / sample: 1 kHz x 1 h = 68 MB, 2 kHz x 2 h = 273 MB. Export to CSV / NumPy at the end |
| `rate` | measures the sampling rate from the first ~500 ms (median dt, jitter, gaps) and proposes the nearest standard rate (60, 120, 250, 500, 1000, 2000 Hz) | feeds the pre-allocation dialog and the detector parameters (the HMM has per-rate parameters, 1000 and 500 Hz exported) |
| `source` | `ISource`: `UdpSource` (binary packets, see 3.3), `FileSource` (CSV replay at real time or x speed), `SimSource` (synthetic) | TCP and the EyeLink Data Link adapter later; a pylink sender script is provided |
| `dsp` | fill, 5-point velocity, detrended velocity, robust noise, HMM score, Viterbi, Savitzky-Golay, acceleration-zero refinement | header-only; **parity test against the Python reference on recorded traces** (golden file) |
| `detector` | modes: `Fast` (velocity threshold, provisional events), `Hmm` (+ refinement), `CausalTcn`, `BiTcnDelayed`; all produce `Event`s | streaming wrapper: fixed-lag decisions, incremental features (no re-computation of the running median) |
| `calibrator` | parameters (amplitude floor, onset shift, offset shift, merge gap, offset rule first / last) fitted on expert events | grid search in C++, < 1 s; also gives the agreement (event F1, kappa) between the detector and the expert |
| `annotation` | expert events: add / delete / move boundaries, undo, save with the session; metrics against the detector | the expert edits a *time range*, not a whole recording |
| `trigger` | rules `{event kind, amplitude range, anchor (onset-provisional / onset-final / offset), delay ms, sink, payload, refractory}` + timer thread + latency log | see 4 for what is physically possible |
| `predict` | future position: ballistic landing tables from `GazeEngine` first, then the research track (section 6) | shows predicted landing point and its interval |
| `engine` | owns everything above, thread orchestration, statistics (latencies, drops, CPU) | `eyelive_cli` runs it headless (tests, scripting, servers) |
| `gui_main` | ImGui front-end, 7 tabs, `--shot file.ppm` renders frames to an image (the dev machine has no usable display, so the GUI is checked through images) | |

### 3.2 Threads and data flow
* Source thread: blocking `recvfrom`, timestamps each packet with the monotonic clock on arrival, writes the sample into the recorder (single writer), publishes `count` with release semantics.
* Analysis thread: wakes every hop (5 ms) or on new data, reads the recorder (acquire), updates incremental features, runs the detector on the recent window, pushes finalized events to an SPSC queue. No allocation after start.
* Trigger thread: min-heap of (fire time, action); sleeps with `clock_nanosleep` / `sleep_until`, a short spin for the last 100 us; every fire logs requested vs actual time.
* GUI thread: only reads (recorder, event snapshot, stats); commands (start / stop / change mode) go through a small command queue.
* Clock model: sample times come from the tracker (or the arrival clock if absent); a running estimate of `host_time - tracker_time` maps event times to the host steady clock (needed for "N ms after the saccade" triggers). Dropped / late packets are counted and shown.

### 3.3 Input protocol v1 (UDP, little endian)
`uint32 magic = 0x31455945 ("EYE1")`, `uint32 n`, then `n` x `{double t_seconds, float x_deg, float y_deg, uint32 flags}` (20 bytes each; flag bit 0 = valid, bit 1 = blink). A text mode (`t x y [valid]` per line) for debugging. Several samples per packet are allowed (EyeLink sends batches). Pixel input: a geometry message (screen size, distance, resolution) converts to degrees.

## 4. Latency budget (targets to MEASURE, not claims)
| stage | what | expected | where it matters |
|---|---|---|---|
| network + tracker | tracker-dependent (EyeLink 1000: ~1.5-3 ms sample latency) | outside our control | all |
| velocity | 5-point difference needs +2 samples; minimum event length 3 samples | ~3-5 ms after onset | provisional events |
| **provisional detection** | threshold crossing on the robust velocity | **~3-6 ms after onset** | "trigger on microsaccade onset", "10 ms after the onset" |
| HMM decision | fixed-lag Viterbi, lag 20-40 ms | 20-40 ms after the event | final label |
| refinement | smoothed acceleration needs +8 ms, search margin 25 ms | ~45-65 ms after the offset | final onset / offset values, amplitude |
| BiTCN | +-257 samples | ~260 ms | review, not triggers |

What this means for triggers: anchored on the **onset** they can use the provisional event (a few ms, with a higher false-alarm rate that must be measured and can be shown / controlled); a delay of 10 ms after the onset is feasible; anything needing the **amplitude** (is it a microsaccade, < 1 deg?) or the **offset** has to wait for the final event (tens of ms) unless the amplitude is estimated from the first few ms of the velocity profile (research item). The GUI must display, per rule, the *measured* detection latency and trigger jitter.

## 5. Training from the expert (what "learns from the annotations" means here)
| mode | trained on | what is learned | cost | evidence |
|---|---|---|---|---|
| 1 Unsupervised | nothing (HMM fitted on archive/; optional re-fit of its two Gaussians on the session, no label) | detects, rule first / last chosen by the user | none | label-free 0.89 / 0.70 mean |
| 2 Calibrated | 3-10 annotated trials / ~30 annotated saccades | 5 convention parameters | < 1 s, pure C++ | d1: 0.94 / 0.83 with 10 trials |
| 3 Adapted network | the same annotations | 16-number embedding of a conditioned BiTCN, or the last layers | needs gradients (Q3) | d3 / d4: 0.91-0.95 / 0.81-0.88 with 3-10 trials; d2 / Andersson not solved |
| 4 Zero-shot network | nothing | pre-trained on other datasets | none | zero-shot below U'n'Eye on d2, d3, Andersson |
The U-Net style training "while the data come" is delivered in two steps: **calibrator first (mode 2)**, a real network adaptation later (mode 3), because training a convolutional network from scratch in C++ on a handful of seconds is not what the benchmark evidence supports (pre-training and fine-tuning gave at best marginal gains, `docs/BENCHMARK_SUMMARY.md` section 3).

## 6. Position prediction (research track, separate from the application)
Existing: ballistic landing tables in `GazeEngine` (10 / 20 ms ahead, interval from Laplace scale). To do: (1) literature search for the state of the art in saccade landing-position prediction for gaze-contingent rendering / foveated VR (I do not trust my memory of the exact references and numbers: they must be looked up and the metric defined: error in degrees at the saccade end, as a function of the time since onset); (2) build the benchmark on our labeled data (predict the landing position from the first k ms after onset, compare with the true offset position; baselines: main-sequence extrapolation, the current tables, a small causal network); (3) integrate the best predictor in `predict`. Success criterion to be agreed before any model: the metric, the datasets, and the number to beat.

## 7. GUI layout (final target)
```
+------------------------------------------------------------------------------------------------------+
| EyeLive   [source: UDP :5005  | 1000 Hz measured | 0 drops]   rec 00:12:41 / 01:00:00   mem 68 MB (preallocated)   latency p50 6 ms |
+------------+-----------------------------------------------------------------------------------------+
| Tabs       |  x, y (deg)  scrolling 10 s  ============================================================ |
|  Acquire   |  detected events shaded: blue = provisional, green = final, orange = expert               |
|  Live      |  speed (deg/s) with threshold line ----------------------------------------------------- |
|  Detection |  predicted landing point (cross) while a saccade is in progress                           |
|  Annotate  +------------------------------------+----------------------------------------------------+
|  Triggers  |  2-D gaze map (last 2 s)           |  event list: t, amp, peak v, dur, kind, source     |
|  Predict   |                                    |  click = jump / delete / edit boundaries          |
|  Log       |                                    |  agreement with the expert: F1 0.94 / kappa 0.83  |
+------------+------------------------------------+----------------------------------------------------+
| status: mode Hmm + first-inflection | trigger rules: 2 active | last trigger +10.2 ms after onset (requested 10.0)  |
+------------------------------------------------------------------------------------------------------+
```
* **Acquire**: source choice and parameters, measured rate, duration, memory / disk estimate, session path, Start / Stop, storage check. The pre-allocation is done at Start and cannot fail silently.
* **Live**: the plots above; nothing editable.
* **Detection**: mode (1 / 1b / 2 / 3 / 4), parameters (noise floor, first / last rule, amplitude floor), with the live agreement against the expert events.
* **Annotate**: pause / scrub the recording, drag to add a saccade, click to delete or move a boundary, "accept all suggestions in this range", undo; "fit to my annotations" button (mode 2).
* **Triggers**: rule table, test button, sink configuration, measured latency / jitter histogram.
* **Predict**: landing prediction, error against the realized landing once the saccade ends.
* **Log**: stream health (drops, gaps, blinks), guard flags, CPU.

## 8. Milestones, with acceptance tests
| # | content | acceptance (automatic test unless noted) |
|---|---|---|
| M0 | `dsp` core + parity | C++ events equal the Python HMM / refined events on 5 recorded traces (golden file) |
| M1 | recorder (mmap, preallocation, rate estimation), sources, UDP loopback, `eyelive_cli`, replay tool | write 1e6 samples, kill the process, reload them; rate estimated within 0.5 %; UDP loopback without loss at 2 kHz |
| M2 | streaming engine: incremental features, fixed-lag HMM, refinement, events | streaming events agree with batch events (> 99 % of events matched, boundary shift <= 2 samples); CPU < 5 % of one core at 2 kHz |
| M3 | trigger engine + latency measurement | on a simulated 1 kHz stream with known onsets: provisional latency, jitter and false-alarm rate reported; "+10 ms" rule fires within the stated tolerance |
| M4 | GUI shell: Acquire + Live, `--shot` mode | frame images checked by eye; 60 fps with 10 s of 2 kHz data |
| M5 | Annotate + calibrator + agreement metrics + Detection tab | annotating 10 trials of a recorded file reproduces the benchmark gain (F1 / kappa of the Python calibrator within 0.02) |
| M6 | Predict tab (existing tables), then the research track | metric and baseline numbers published in `docs/` before any model |
| M7 | Windows / Linux builds (sockets, mmap, timers abstracted), packaging, EyeLink adapter | builds in CI; manual test with a real tracker (needs the owner's hardware) |
Realistic target for the first working session: **M0-M4**, M5 if time allows. M6 research and M7 hardware tests are separate days.

## 9. Open questions (to decide before coding further)
* **Q1 Platform**: resolved by the owner's constraint (a single executable that installs anywhere): Windows + macOS (+ Linux) from the start. The platform-specific parts (file mapping, sockets, timers, serial) sit behind one small interface, built and tested NATIVELY on CI runners of each OS (section 11).
* **Q2 Trigger hardware**: UDP to another process is easy; TTL needs a device (USB-serial Arduino, LabJack, parallel port, DAQ). Which one does the lab use? Proposal: UDP + serial byte first, a plugin interface for the rest.
* **Q3 Where the network adaptation (mode 3) runs**: option (b), a Python helper, is now excluded (a bundled Python runtime is the classic cause of antivirus alerts and breaks the single-executable goal). Remaining: (a) pure C++ (backward pass of the BiTCN for a 16-number embedding or the last layer: several days) or (c) skip it and ship modes 1, 1b, 2 and 4. Proposal: (c) now, (a) only if mode 2 proves insufficient on the owner's own data.
* **Q4 Microsaccade definition**: amplitude threshold (commonly < 1 deg, some use 0.5) and duration limit. Configurable, default 1 deg; the owner's lab convention decides.
* **Q5 Geometry**: does the tracker stream degrees or pixels / raw units? Needed for velocities in deg/s (the HMM is unit-free but the amplitude floor and the trigger amplitude range are not).
* **Q6 Classes**: saccade / microsaccade first; blink from the guard; PSO and pursuit later (the multi-class benchmark is the weak point, Andersson).
* **Q7 Expert workflow**: annotate only a start-up block (calibration phase of 1-2 minutes) or continuously during the experiment? Changes the Annotate tab (offline review vs. live correction).

## 10. Risks
1. Online false alarms of provisional detection on noisy real streams (microsaccades are small): measure on d1 / d3 microsaccade data with the simulator-injected noise; give the user a visible threshold.
2. Streaming vs batch differences (robust noise estimated on a sliding window, fixed-lag Viterbi): covered by the M2 test.
3. Timer accuracy on a non-real-time OS (macOS / Windows): measure the jitter, report it in the GUI, never claim sub-millisecond.
4. Scope: "a U-Net that trains itself" is a research result, not a feature; modes 1-2 are what the evidence supports today.
5. The GUI cannot be viewed on the development machine (no display): `--shot` images and the owner's checks replace screenshots.

## 11. Distribution, antivirus and footprint (owner's constraints, 2026-10-11)
Goal: one executable that runs on any lab machine without antivirus complaints, as fast and light as possible. C++ with static libraries is the right base for it (no interpreter, no runtime to install, no packer). What actually decides the antivirus outcome, and what we do:
| cause of alerts | decision |
|---|---|
| **unsigned binary / no reputation** (by far the main cause) | sign every release: Windows code-signing certificate (an OV certificate, or Azure Trusted Signing; EV removes SmartScreen warnings faster), macOS Developer ID + notarization (the Apple account costs a yearly fee; without notarization Gatekeeper blocks the app). Cannot be done by the code alone: needs the owner's accounts / certificates. Without signing nobody can promise "no complaint". |
| **packers, self-extracting archives, bundled scripting runtimes** (PyInstaller, UPX, Electron) | none of them. Plain native executable, no compression, no embedded interpreter, no dynamic code loading. This is also why the Python helper is dropped (Q3) |
| **suspicious behaviour** (hooks, injection, privilege escalation, hidden persistence) | none. Standard APIs only: sockets, files, timers, serial. No admin rights needed (portable: the exe runs from any folder, data in a user folder). Do NOT use the real-time priority class (needs admin); use high priority + `timeBeginPeriod(1)` on Windows |
| **network listening** | the UDP port is opened only when the user starts a session, bound to localhost by default (the Windows firewall prompt on a non-local bind is normal and documented) |
| missing runtime DLLs | static C++ runtime (`/MT`, `-static-libstdc++`), static GLFW and ImGui; the only dynamic dependencies are system libraries present on every machine (opengl32 / gdi32 / user32 on Windows, system frameworks on macOS) |
| trust by inspection | open source, reproducible build in CI, published SHA-256, submission of each release to Microsoft Defender's analysis portal, a one-page "what this program does" |
Footprint targets (all MEASURED in the tests, reported in the Log tab): executable < 5 MB; CPU < 2 % of one core at 1 kHz with the window open, < 0.5 % headless; resident memory < 50 MB + the preallocated session (the session file is memory-mapped: pages are committed, not copied); the GUI renders on demand (events, new data at 30 Hz) instead of redrawing continuously. OpenGL 3.2 is available on all current machines but is deprecated by Apple; a Direct3D 11 backend for Windows and Metal for macOS are optional later switches (ImGui provides both backends) if a lab machine has a bad OpenGL driver.
Cross-building: the zig toolchain used here can also produce a Windows executable from this Mac, but I cannot run or test it here; Windows and Linux builds are therefore produced and tested by GitHub Actions runners (to set up in M7, one-line additions to the existing CI), and the first Windows executable must be tried by the owner on a real machine.
Why not the alternatives: Qt (bigger, DLLs, LGPL obligations), Electron / web UI (heavy, frequent antivirus alerts), Python GUI (needs an interpreter bundle), Rust (no benefit here, the existing engines are C++).

## 12. Revision 2 (owner's additions, 2026-10-11) — supersedes sections 3.1 `trigger`, 5, 7 (Annotate / Triggers tabs), 8 and 9 where they differ

### 12.1 Workflow: pre-label -> review -> train (in C++) -> deploy
```
 live stream --> [pre-labeler chosen in the GUI] --> pre-labels on the first S seconds (default 50 s, user-adjustable)
                                                         |
                       [Review]  the expert corrects EVERYTHING: presence, onset, offset, class of each event
                                                         |
                       [Train]   U'n'Eye network trained in C++ on the reviewed seconds (rotation augmentation ON by default, can be switched off)
                                                         |
                       [Deploy]  the trained network becomes a detector available in the Detection tab (online, with its decision delay)
```
* **Pre-labeler = a selectable option** (registry of detectors with one common interface `prelabel(trace) -> events with classes`): (1) label-free HMM + acceleration-zero boundaries (works with nothing installed, 2 classes + blink from the guard), (2) **U'n'Eye pre-trained on the owner's datasets** (the shipped weights: `weights_1+2+3` binary, `weights_Andersson` 5 classes, dataset-specific weights), (3) causal / non-causal TCN (`causal.bin`, BiTCN zero-shot), (4) a network previously trained by the user in an earlier session (the same file format), (5) none (annotate from scratch). Each option states its classes and its latency. Evidence on what to expect from each: `docs/BENCHMARK_SUMMARY.md`; **only saccade-versus-rest was ever scored, per-class quality (PSO, pursuit, blink) is NOT measured yet** and must be before the multi-class options are advertised.
* **Review = full event editing**: toggle an event (keep / delete), **drag onset and offset**, change its class, split, merge, add an event by dragging on the timeline, undo / redo, "accept all in this range", jump to the next uncertain event (low confidence or disagreement between two pre-labelers), mark ranges as "do not use for training". A counter shows the reviewed seconds out of the target.
* **Training data amount**: **50 s by default**, a numeric field to raise it (the benchmark curve saturates around 50-300 one-second trials: U'n'Eye F1 0.94-0.96, so more helps little but is allowed; the estimated training time is displayed). 20 % of the reviewed material is held out automatically for early stopping and for the displayed validation score.
* **Augmentation**: random rotations of the position trace **on by default**, a checkbox to disable it (also exposed: gain and added noise, off by default). Rotation matters because the network sees x and y velocities; without it a short session may not cover directions.
* **Start from random weights or from a pre-trained network** (choice in the Train tab); the simulation of section 12.5 decides which default is better.

### 12.2 Classes: up to 5, user-defined
* Per-sample label `uint8` 0..K-1 with K <= 5; class 0 is the background (fixation); names and colours are editable; defaults: fixation, saccade, post-saccadic oscillation, smooth pursuit, blink. Classes are mutually exclusive per sample; an event overlapping another is resolved by the order of the class list (drawn on top).
* **Microsaccade is not a class by default but a property** (amplitude below a threshold, default 1 deg, configurable): triggers can filter on it; a lab that wants it as a class can define it.
* The trained network has K outputs (softmax, as the original U'n'Eye) and K is fixed at training time. Only the saccade-versus-rest result is currently backed by measurements.

### 12.3 Trainer in C++ (no PyTorch)
Scope: the original U'n'Eye topology (a U-Net of about 10-40 channels, two max-poolings, batch normalisation, softmax / cross-entropy over K classes), forward pass already in `cpp/src/native_engine.cpp`; to write: backward passes (1-D convolutions, transposed convolution, max-pool, batch-norm in training mode, ReLU, concatenation, softmax-cross-entropy), Adam, learning-rate decay, early stopping on the held-out part, rotation / noise / gain augmentation, deterministic seeds, a worker thread with a progress callback (loss, validation F1), cancel button. **Verification**: gradient of every layer against finite differences and against PyTorch autograd on golden tensors; then one training run on a benchmark subset must reproduce the PyTorch result within noise. Estimated cost per epoch from the network size: a few seconds single-threaded for ~50 s of data with 4 rotations (to be MEASURED; PyTorch on CPU needed 40-80 s for 50 one-second trials with early stopping). The owner's budget is 20 minutes on a CPU.
Inference of the trained network online: the U-Net looks at a window of about 200 ms, so its labels are final after a delay of half a window; triggers keep using the fast provisional detection (section 4) and the network refines the labels and the boundaries.

### 12.4 Triggers: a file / shared memory readable in real time (replaces the DAC-style sinks)
* The session file (memory-mapped) starts with a **state block** (fixed layout, versioned, documented): sequence counter (seqlock), time of the last sample, gaze x / y, speed, current state (class), last event (class, onset, offset, amplitude, peak speed, provisional / final), event counter, predicted landing point and its interval, trigger-rule flags, health. A reader copies the block between two reads of the sequence counter and retries if it changed.
* **MATLAB / Psychtoolbox**: `memmapfile` reads the block with no compilation or installation (an optional MEX reader later, which I cannot compile or test here). **PsychoPy / Python**: `numpy.memmap` / `mmap` on the same file. **Any other tool**: the format is plain binary and documented. A UDP broadcast of the same state remains available for a second computer.
* "Delay N ms after the saccade" is then done by the experiment code (a loop reading the block), the application only guarantees what it publishes and the measured latency of publication. Rule-based output (fire a flag N ms after an event) stays optional.
* No driver, no hardware, no extra permission: the simplest thing for antivirus and for locked-down lab computers.

### 12.5 What is being measured before building (running now)
`foundation/workflow_sim.py`: ~50 s of a session pre-labeled by the label-free HMM + rule, the expert simulated with the human labels (C1 pseudo-labels as they come; C2 false positives deleted; C3 missed saccades added; C4 both; C5 full human labels = reference), U'n'Eye (the original network, CPU) trained on each and tested on held-out trials, 5 datasets x 3 draws; plus the pre-trained start (weights `weights_1+2+3`) on the two datasets it never saw (d4, Andersson). It tells which correction matters most (so the Review tab can prioritise it) and whether the pre-trained start is worth offering by default. Boundary corrections and classes are NOT covered by this simulation (the expert is a label oracle with the benchmark's own boundaries): the boundary effect is estimated separately from the calibrator results (kappa gains of 0.05-0.3 from fixing the onset / offset convention).

### 12.6 Milestones (replaces section 8)
| # | content | acceptance |
|---|---|---|
| M0 | `dsp` core + parity with Python | events equal on 5 recorded traces |
| M1 | recorder (mmap, preallocation, state block), rate estimation, sources, UDP loopback, `eyelive_cli`, replay | crash-safe reload; rate within 0.5 %; no UDP loss at 2 kHz; MATLAB `memmapfile` and Python readers on the state block (reader examples + a test) |
| M2 | streaming engine + pre-labeler registry (HMM + rule first; U'n'Eye / TCN engines wired through the same interface) | streaming = batch within 2 samples on > 99 % of events; latency measured |
| M3 | GUI shell (Acquire, Live, Detection), `--shot` image mode | images checked; 60 fps; CPU target |
| M4 | **Review tab** (presence, onset, offset, class, undo, 5 classes, training-set counter) | unit tests of the edit model; editing session replayed from a script |
| M5 | **C++ trainer** with gradient checks, augmentation toggle, held-out early stopping, Train tab | gradients vs PyTorch; reproduces the Python result within noise; 50 s trained in < 20 min on the owner's CPU class |
| M6 | deploy: trained network as a detector (online with delay), model files saved / loaded | same events as the Python network on golden traces |
| M7 | multi-class evaluation (PSO / pursuit / blink per-class scores) and defaults | numbers in `docs/` before the options are advertised |
| M8 | prediction research track, Windows / Linux CI builds, signing | see sections 6 and 11 |

### 12.7 Owner's answers (2026-10-11) and what they imply
* **Review at start-up AND during the experiment.** Two modes of the same Review tab. *Start-up block*: the first S seconds are pre-labeled and reviewed before training. *Continuous*: a review queue shows the events as they stream (accept / modify / reject, uncertain ones first), the corrected material is added to the training set, and the network is **retrained in the background** (worker thread with a capped number of cores and a lower priority so that acquisition and publication latency are not touched) and **hot-swapped** atomically at a safe moment (between trials or on request). Every event stores which detector / model version produced it (provenance) and the original pre-label is kept next to the corrected one, so that the analysis can be redone later.
* **Stream in degrees OR pixels (or any unit).** Raw positions are always stored as received; the session header stores the scale factor in use (deg per unit, exact if the user gives pixels-per-degree / screen geometry, otherwise estimated) so that nothing is lost. A network trained in degrees is applied to `c x position`; a network trained by the user on his or her own stream carries its own `c`.
* **Autonomous scale selection (works, see 12.8).** Instead of a physical conversion, the program chooses the scale `c` for which the degree-trained network and the unit-free HMM agree best on the session itself (about 50 s are enough), and shows the result with its stability; the user can override with the geometry. A main-sequence plausibility panel (duration, amplitude, peak-velocity / amplitude, saccade rate, fixation durations) is displayed as an extra health check, not used as the estimator (see 12.8).

### 12.8 Evidence for the unit handling (`foundation/scale_check.py`, `foundation/scale_select.py`, results in `foundation/runs/night/`)
* **Tolerance**: U'n'Eye (weights_1+2+3) on positions multiplied by a wrong factor: for the microsaccade datasets (d1-d4) a factor between 0.7 and 1.5 costs little (event F1 0.87-0.93), 2x costs 0.05-0.15, 4x or 0.25x destroys it (F1 0.4-0.78). A conversion good to +-30-50 % is enough.
* **The best scale is not the physical one**: on Andersson (natural viewing, large saccades) the same weights give F1 0.54 at the true scale and 0.82 with the positions divided by 4: the network works best when the amplitudes resemble its training range. This is what the autonomous selection finds.
* **The classical main-sequence slope does NOT give the scale**: with the textbook duration slope (2.2 ms/deg) the estimated scale of data whose true scale is 1 comes out as 2.1 (Andersson) and 4.7-7.3 (microsaccade datasets), because the duration-amplitude relation is far from linear for small saccades (slope 11-16 ms/deg, intercept 11-21 ms) and its spread is +-20-50 % with 80 events. Peak velocity over amplitude is scale-invariant (typically 60-180 per second here) and cannot give a scale. Main-sequence quantities therefore serve as a plausibility check only.
* **Agreement-based selection**: the scale maximising the event agreement between the network and the unit-free HMM (+ acceleration-zero rule), on 3 random subsets of 50 trials: d1 chooses 1.4 / 1.0 / 1.0, d2 1.0 / 1.0 / 1.4, d3 0.7 x3, d4 1.4 / 1.4 / 1.0, Andersson 0.25 x3 (stable); human-label F1 / kappa at the chosen scale: 0.90 / 0.86, 0.93 / 0.87, 0.91 / 0.79, 0.87 / 0.80, 0.82 / 0.62 against 0.90 / 0.86, 0.93 / 0.87, 0.93 / 0.82, 0.90 / 0.83, 0.54 / 0.34 at c = 1. Within 0.03 F1 of the true scale on the microsaccade data and equal to the best scale in hindsight on Andersson.
* **Limits**: tested on degree data multiplied by factors in [0.25, 4] around the truth; real pixel streams span a much wider range (about 10-60 pixels per degree, i.e. c = 0.017-0.1 relative to degrees), so the search grid must be logarithmic over about two decades (about 16 values at a ratio of 1.4, cheap) and must be tested on pixel-scaled data before it is advertised; the criterion presupposes that the HMM works on the session (a very noisy or non-saccadic stream would make both detectors agree on nothing: the agreement score is shown and a low value is reported instead of a silent choice).

### 12.9 Questions still open
1. Defaults of the 5 classes and whether microsaccade should also be offered as a class (proposal: property, not class).
2. Signing certificates (Windows) and an Apple developer account for distribution (section 11).
3. Whether a lab still needs a TTL output (the state block of 12.4 replaces it for MATLAB / PsychoPy).
4. Whether the continuous-review mode may use several CPU cores for retraining during an experiment (proposal: at most one, lower priority, configurable).

### 12.10 Result of the workflow simulation (`foundation/workflow_sim.py`, table in `docs/BENCHMARK_SUMMARY.md` section 3e) and what it changes
Mean over the 5 datasets (F1 / kappa): pseudo-labels as they come 0.84 / 0.70; + false positives deleted 0.90 / 0.74; + missed saccades added 0.91 / 0.75; both 0.91 / 0.76; full human labels 0.90 / 0.82. Per dataset: d1 and d4 gain mostly from deleting false positives (d4: 0.71 -> 0.95), d3 from adding missed saccades (0.83 -> 0.89), d2 and Andersson already start high. Consequences for the Review tab and the trainer:
* Priorities of the review UI: (1) one-click deletion of false positives with the most doubtful ones first, (2) add missed events, (3) **boundary editing**, which is what the kappa needs (the part of the gain this simulation does NOT measure: 0.76 -> 0.82 is the gap left by the convention of the boundaries, 0.11 on d2).
* The trained network improves on the pseudo-labeler by itself on Andersson (0.79 / 0.60 -> 0.86 / 0.72 with no correction): training is worth offering even without review.
* The start from pre-trained weights must be offered and chosen on the validation score: large gain on d4 (human labels 0.83 / 0.82 -> 0.96 / 0.90), none on Andersson.
* Cost: each training on 50 s took 5-80 s with PyTorch on a CPU; the C++ trainer has to match it (milestone M5).
* Limits: oracle expert, 3 draws, binary saccade-versus-rest only, and one unexplained anomaly (d4, full human labels give a lower F1 than corrected pseudo-labels).
