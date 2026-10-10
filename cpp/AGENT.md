# cpp/ — real-time C++ engine, safety guard, models
Read `README.md` (usage), `GAZE_ENGINE.md` (design, guarantees, what is NOT verified), `../docs/safety/hazards.md` and `../docs/ROADMAP.md`. Build: see the Environment section of `../AGENT.md`; `ctest --test-dir build` runs 6 tests (parity, zero_heap, gaze_engine, safety_guard, equivalence, bitcn).
| Folder | Content |
|---|---|
| `include/uneye/` | public headers (engines, `gaze_engine.hpp`, `safety_guard.hpp`, `bitcn.hpp`, ...) — see `include/uneye/AGENT.md` |
| `src/` | implementations; `gaze_engine.cpp` and `safety_guard.cpp` compile with `-Wall -Wextra -Werror` — see `src/AGENT.md` |
| `tests/` | ctest executables + golden files (PyTorch references) — see `tests/AGENT.md` |
| `tools/` | CLIs: `uneye_rt`, `gaze_replay`, `fault_injection`, `bench_latency`, `causal_dump` |
| `scripts/` | Python: export models, golden files, calibration, equivalence check — see `scripts/AGENT.md` |
| `models/` | exported networks: `*.bin` (native engine) and `*.onnx`; `bitcn_sup_bitcn_v2.*` = best 2-channel BiTCN; `causal.*` = streaming TCN; `combined/dataset*/andersson` = original U-Nets |
Safety stack: network -> `GazeEngine` (physics layer, zero heap) -> `SafetyGuard` (NORMAL/DEGRADED/SAFE). Two model families: causal streaming (`causal_engine`, online) and non-causal BiTCN (`bitcn`, offline / short delay).
