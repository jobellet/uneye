# online/ — causal (forward-only) networks for online labeling
Question answered: which network that only sees the past can label saccades at the latest sample? Notebook `UnEye_online_architectures.ipynb`, results `results_v2.csv|json`, docs `README.md`, `README_architectures.md`.
| File | Role |
|---|---|
| `causal_net.py` | the causal TCN used by the C++ streaming engine (`final_causal.*`) |
| `archs.py` | alternative causal backbones; `experiment.py`, `run_architectures.py`, `select_variant.py` (choice on validation trials only) |
| `train_causal.py`, `run_online_training.py` | training (same logic as `uneye.DNN.train`) |
| `evaluate_online.py`, `metrics.py`, `viz.py`, `gaze_score.py` | evaluation vs U'n'Eye on the last sample of a 200 ms bin; online metrics; figures; scoring of `cpp/build/gaze_replay` |
| `export_causal.py` | exports `.bin` (magic UNCZ) + `.onnx` for `cpp/` |
| `weak_labels.py`, `forecast.py` | free labels from a velocity threshold; gaze-position forecasting (exploratory) |
Large outputs (`runs/`, `final_causal.*`, `results_v2.*`) are gitignored in part; check `.gitignore` before expecting them.
