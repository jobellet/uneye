# cpp/app/ — EyeLive application (real-time acquisition, detection, review, training, triggers)
Design: `../../docs/EYELIVE_ROADMAP.md`. Work packages for agents: `../../docs/tasks/README.md`. One directory per module, contracts in `include/eyelive/` (frozen). Build: `cmake -S cpp -B build -G Ninja && cmake --build build`, tests `ctest --test-dir build -L eyelive`.
| dir | content | task |
|---|---|---|
| `include/eyelive/` | contract headers (types, trace, dsp, detector, source, recorder, state_block, annotation, trainer, calib, trigger, engine, gui_tab) | integrator |
| `dsp/` | batch DSP parity test against Python (`dsp.hpp` is in `include/`) | integrator |
| `recorder/` `source/` `state/` `readers/` | session file, sources + simulator, state block + MATLAB / Python readers | T01 T02 T03 |
| `detect_hmm/` `detect_fast/` `detect_net/` | HMM + acceleration boundaries, fast threshold detector, neural adapters | T04 T05 T06 |
| `annot/` `calib/` `train/` `trigger/` | expert model, calibrator + scale selection, C++ trainer, trigger engine | T07 T08 T09 T19 |
| `export/` `setup/` | export (CSV, BIDS-style, MAT), session report, provenance journal, SHA-256; device profiles, trigger-rule parser, checks, health light | after T22 |
| `engine/` `cli/` `gui/` | orchestration, headless front-end, Dear ImGui front-end (third-party libraries in `../third_party`, fetched by `fetch.sh`) | T18 T20 T12-T17 |
| `ci/` | `check_ownership.py`, `jules_setup.sh` | integrator |

`cli/tools/eyelive_tool.cpp` (info, export, redetect, check, profiles) is built with the CLI; `gui/src/ui_logic.cpp` (modes, wizard, controller) has no ImGui and is tested without a window (`gui/tests/test_ui_logic.cpp`). Engine extensions after T22: candidates, landing network, frame mailbox, session notes (docs/CANDIDATE_TARGETS.md, docs/LANDING_NET.md, docs/FRAME_MAILBOX.md).
