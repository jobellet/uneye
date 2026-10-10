# AGENT.md — repository router (read this first, then ONLY the folder you need)

Fork of U'n'Eye (Bellet et al. 2019, saccade detection with a U-Net) extended with a label-free study, a causal/online network, a real-time C++ engine with a safety guard,
and an eye-movement "foundation" study. Owner speaks French; code, docs and comments are English. Every number quoted in `docs/` comes from a script in this repo.

## Where to go
| You want to... | Go to |
|---|---|
| use / train the original U'n'Eye (Python package) | `uneye/AGENT.md`, `training/AGENT.md`, `README.md` |
| the current best detector and all recent experiments (BiTCN, HMM, SSL, self-training) | `foundation/AGENT.md` |
| label-free detectors (classic velocity thresholds, HMM, seeded CEBRA) | `free_saccade/AGENT.md` |
| causal (online, forward-only) networks, training and notebooks | `online/AGENT.md` |
| C++ real-time engine, safety guard, ONNX/.bin models, tests, CI | `cpp/AGENT.md` |
| datasets and their formats | `data/AGENT.md` |
| reports, roadmap, safety documents, slides | `docs/AGENT.md` |
| the original 2019 article's analyses (notebooks) | `analysis scripts/AGENT.md` |

## Current state (2026-10-10)
- Best detector: 2-channel (vx, vy) bidirectional TCN, `foundation/sup_bitcn.py --inputs 2`: mean event F1 0.90 / kappa 0.81 on the 5 benchmarks (U'n'Eye 0.85 / 0.75). C++ port with parity test: `cpp/src/bitcn.cpp`.
- Label-free best: universal HMM (F1 0.88 / kappa 0.66) and its noisy-student TCN (0.83 / 0.67). Self-supervised encoders (JEPA, MAE, HuBERT, TS2Vec, DINO) did NOT beat plain velocity inputs: `docs/OVERNIGHT_REPORT.md`.
- Hand-over documents: `docs/ROADMAP.md` (safety demo, steps 2-8, done), `docs/FOUNDATION_ROADMAP.md` (foundation study, findings), `docs/OVERNIGHT_REPORT.md` (auto-generated comparison).

## Conventions that are easy to get wrong
- Evaluation metric everywhere: event F1 (predicted run >= 3 samples overlapping a human saccade) AND Cohen's kappa (sample-wise, saccade vs rest), both from `foundation/compare.py::event_f1`. Test sets are set B ("test"); set A ("train") is for training/validation. Never select anything on test.
- Inputs are velocities only (unit-free, noise-normalised), never absolute position. Everything is resampled to 1 kHz in `foundation/data.py`.
- Dataset 4 set B: positions have 3000 trials, labels 3300 rows; the FIRST 3000 label rows are right (already handled in the loader).
- Gitignored (do not look for them in git): `archive/` (unlabeled recordings), `data/Andersson/`, `.venv/`, `foundation/runs/*.pt|*.log`, big traces. Do not push data.

## Environment (macOS M1, no Xcode command line tools)
- Python: `.venv/bin/python` (uv). Training uses PyTorch `mps` (`foundation/train_lodo.py::DEV`); `uneye.DNN` trains on CPU only.
- C++: `export PATH=$PWD/.venv/bin:$PATH` (cmake, ninja, zig as compiler), then `cd cpp && cmake -B build && cmake --build build && ctest --test-dir build`.
- Git/push: GitHub Desktop's bundled git (`/Applications/GitHub Desktop.app/Contents/Resources/app/git/bin/git`); credentials are entered by the user, never by an agent.
- One GPU job at a time. Commit messages end with the co-author line required by the session.
