# EyeLive work packages for asynchronous coding agents (Jules)

The EyeLive application (`docs/EYELIVE_ROADMAP.md`) is split into independent work packages so that several agents can build it at the same time without touching the same files. **To start an agent, give it one line:**

> Execute the instructions in `docs/tasks/T04.md` (replace `T04` by the task).

Each task file is complete: goal, what to read, the paths the agent owns, what to implement, acceptance tests with numbers, build / test commands, definition of done. The agent must read this README first.

## Rules that apply to every task
1. **Ownership.** You may change only the paths listed under "You own" in your task file (the same list is in `docs/tasks/OWNERSHIP.json`, enforced by CI: a pull request from a branch named `jules/Txx-<name>` that touches any other file fails the `ownership` check). Never edit another task's files, the contract headers, `cpp/app/CMakeLists.txt`, `docs/tasks/*.md`, or the base workflow.
2. **Contracts first.** The interfaces are the headers in `cpp/app/include/eyelive/` (`types`, `trace`, `detector`, `source`, `recorder`, `state_block`, `annotation`, `trainer`, `calib`, `trigger`, `engine`, `gui_tab`, `dsp`). They are frozen. Code against them; test with the in-memory doubles (`VectorTrace`, your own mocks). A module never includes another module's implementation, only contract headers and the public headers its task file names. If a contract is wrong or incomplete, **stop and open a GitHub issue labelled `contract`** (title `Txx needs: ...`): the integrator changes the contract in main and tells you to rebase. Do not work around it.
3. **One module = one directory** under `cpp/app/<module>/` with `src/`, `tests/`, `include/eyelive/`. The module's `CMakeLists.txt` is a single line (`eyelive_add_module(<name>)`); sources and tests are found by glob (`src/**/*.cpp`, `tests/test_*.cpp`): **adding a file never requires editing a build file.** Every test is registered automatically, labelled `eyelive` and `<module>`, and receives the path of the `cpp/` directory as `argv[1]` (to find goldens and models).
4. **Language and style.** C++17, no exceptions across module interfaces, no global mutable state, warnings clean with `-Wall -Wextra` on gcc and clang (CI: Linux gcc and macOS clang), no third-party library beyond what the task names. Data hot paths do not allocate after start. Match the surrounding comment style (short, factual, English).
5. **Numbers, not adjectives.** Acceptance tests have thresholds; tests print the measured values; the status file (`docs/tasks/status/Txx.md`) reports them and lists what is not done. Do not claim a result you did not measure. Do not weaken a threshold to make a test pass: report the measured value and why.
6. **What is not in git.** `data/Andersson/`, `archive/` and `foundation/runs/*.pt` are not in the repository and a cloud agent cannot use them. Every test must run from files committed to the repository (goldens are committed). The datasets `data/dataset1..4` and the models in `cpp/models/` are in git.
7. **Pull requests.** Branch `jules/Txx-<short-name>`, title `Txx: <title>`, body = the status file's content, small commits, no reformatting of files you do not own, no unrelated changes. Tasks with two parts (T09) open two pull requests in order.
8. **Environment.** `sh cpp/app/ci/jules_setup.sh` (Debian / Ubuntu; add `gui` for the GUI tasks) installs the toolchain; build with `cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build`; run `ctest --test-dir build -L <module> --output-on-failure` and then `ctest --test-dir build -L eyelive` (nothing else may break).
9. **Honest limits.** Windows and MATLAB code that cannot be tested in the agent's environment must say so in the status file; macOS-specific code likewise.

## Phases and dependencies
| task | title | phase | depends on | size |
|---|---|---|---|---|
| [T01](T01.md) | Recorder: preallocated memory-mapped session file | 1 | — | M |
| [T02](T02.md) | Sources: UDP (protocol v1), file replay, simulator, rate estimator | 1 | — | M |
| [T03](T03.md) | State block writer and readers for MATLAB / PsychoPy / Python | 1 | — | M |
| [T04](T04.md) | Detector: streaming label-free HMM with acceleration-zero boundaries | 1 | — | L |
| [T05](T05.md) | Detector: fast causal threshold detector with provisional events | 1 | — | M |
| [T06](T06.md) | Detectors: neural adapters (U'n'Eye, causal TCN, BiTCN, user models) | 1 | — | L |
| [T07](T07.md) | Annotation model: expert events, undo / redo, 5 classes, agreement metrics | 1 | — | M |
| [T08](T08.md) | Calibrator and automatic unit-scale selection | 1 | — | M |
| [T09](T09.md) | Trainer: U'n'Eye training in C++ without PyTorch | 1 | — | XL |
| [T12](T12.md) | GUI shell: window, render loop, tab registry, mock engine, offscreen `--shot` | 1 | — | M |
| [T13](T13.md) | GUI tab: Acquire | 2 | T12 | M |
| [T14](T14.md) | GUI tab: Live (plots, events, gaze map) and the plot widget | 2 | T12 | M |
| [T15](T15.md) | GUI tab: Review (event editing, 5 classes, undo / redo) and the timeline widget | 2 | T12, T07 | M |
| [T16](T16.md) | GUI tabs: Detection and Train | 2 | T12 | M |
| [T17](T17.md) | GUI tabs: Triggers, Predict and Log | 2 | T12 | M |
| [T18](T18.md) | Engine: orchestration of recorder, source, detectors, state block, triggers, trainer | 2 | T01, T02, T03, T04, T05, T07, T19 | XL |
| [T19](T19.md) | Trigger engine: rules, delays, UDP / log sinks, latency log | 1 | — | M |
| [T20](T20.md) | CLI and end-to-end tests | 3 | T18 | S |
| [T21](T21.md) | CI / release pipeline, packaging, signing documentation | 1 | — | M |
| [T22](T22.md) | User manual | 3 | T13, T14, T15, T16, T17 | S |

* **Phase 0 (integrator, done before any agent starts):** contracts, the batch DSP with its Python parity test (`cpp/app/dsp`), the CMake skeleton, the ownership checker, the base CI workflow `eyelive.yml`, this task system.
* **Phase 1 (start together, no dependency between them):** T01, T02, T03, T04, T05, T06, T07, T08, T09, T12, T19, T21. All depend only on the contract headers. T09 is the longest (two pull requests); start it first.
* **Phase 2 (start when the listed tasks are merged):** T13-T17 (GUI tabs, need the shell T12; T15 also needs T07), T18 (engine, needs T01-T05, T07, T19).
* **Phase 3:** T20 (CLI, needs T18), T22 (manual, needs the tabs).
* **Integrator (not for agents):** reviewing and merging, contract changes, cross-module integration and end-to-end validation, the research tracks (landing-position prediction, per-class evaluation, scale selection on pixel-scaled data), the signing accounts.

## Ownership map (directories)
```
cpp/app/include/eyelive/   contracts                 integrator
cpp/app/dsp/               batch DSP + parity test   integrator (done)
cpp/app/recorder/          T01        cpp/app/source/   T02 (+ tools/send_udp.py)
cpp/app/state/, readers/   T03        cpp/app/detect_hmm/ T04     cpp/app/detect_fast/ T05     cpp/app/detect_net/ T06
cpp/app/annot/             T07        cpp/app/calib/    T08       cpp/app/train/ T09           cpp/app/trigger/ T19
cpp/app/engine/            T18        cpp/app/cli/      T20
cpp/app/gui/               T12 shell (CMake, shell_*.cpp, mock_engine.hpp); tabs: T13 acquire, T14 live + plot widget, T15 review + timeline widget, T16 detection + train, T17 triggers + predict + log
packaging, release CI      T21        docs/eyelive/     T22
```

## Continuous integration (`.github/workflows/eyelive.yml`, integrator-owned)
* `ownership`: on pull requests from `jules/Txx-*` branches, `python3 cpp/app/ci/check_ownership.py <branch> <base>`.
* `build-test`: Linux (gcc) and macOS (clang), Release, all eyelive tests (`ctest -L eyelive`).
* `sanitizers`: Linux AddressSanitizer and ThreadSanitizer on all eyelive tests.
* `gui`: Linux build with `-DEYELIVE_BUILD_GUI=ON` and the offscreen `--shot` smoke test (skipped until T12 exists).
* `eyelive-release.yml` (T21): Windows / macOS / Linux release builds, static runtime, checksums.

## Contract change process
Open an issue labelled `contract` describing the exact signature needed and why. The integrator edits the header in main (one pull request, tests updated), comments on the issue, and the agent rebases. Contract changes never come from a task pull request (CI would reject them).

## What the integrator checks before merging a task
Acceptance numbers in the status file against the task file; the diff touches only owned paths; tests do not depend on files outside the repository; no weakened thresholds; the module builds on both compilers; for GUI tasks, the attached `--shot` images. A task is merged only when CI is green.
