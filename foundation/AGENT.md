# foundation/ — eye-movement "foundation model" study and the best detector

Start from `docs/FOUNDATION_ROADMAP.md` (findings) and `docs/OVERNIGHT_REPORT.md` (table of every candidate). Run everything from the repo root with `.venv/bin/python foundation/<file>.py`.

## Core (use these)
| File | Role |
|---|---|
| `data.py` | the loader: `FD.load(name, "train"|"test")`, names `d1 d2 d3 d4 andersson`; everything at 1 kHz; labels 1 = saccade |
| `compare.py` | the scorer `event_f1(pred, S)` (F1, kappa, recall, precision) + reference predictors (EK, HMM, causal TCN, U'n'Eye `pred_unet`) |
| `dino1d.py` | input features: `feats2` (vx, vy only, 2 ch, the current best input), `feats` (8 ch), `training_pool` (archive + unlabeled splits) |
| `bitcn.py` | `BiTCN` (bidirectional dilated TCN, receptive field 515), `score_fn_factory` (TTA), `val_direct`; module switch `BT.FEATS` |
| `sup_bitcn.py` | supervised training on d1+d2+d3 (`--inputs 2|8`), EMA, early stopping; writes `runs/<name>.pt` (`{"net","ema"}`) and `runs/night/<name>*.json` |
| `selftrain.py` | label-free noisy-student TCN from the universal HMM pseudo-labels |
| `universal_hmm.py`, `hmm_score.py` | label-free minimum-duration HMM fitted on `archive/` only; HMM on any 1-D score |
| `export_bitcn.py` | BiTCN -> ONNX + `.bin` + golden file for the C++ engine (`cpp/`) |
| `paper_figs.py`, `paper_plots.py`, `uneye_curves.py` | the 2019 article's analyses (performance vs number of labeled trials, between-subject generalization) with BiTCN and retrained U'n'Eye -> `docs/figs_paper/` |

## Self-supervised / exploratory candidates (all negative results, kept for reference)
`ssl_vit.py` (I-JEPA / MAE / HuBERT 1-D), `ts2vec.py`, `dino1d.py` + `dino_eval.py` (DINO), `ssl_encoder.py` (CEBRA-time), `ssl_probe.py` (validation protocol: leave-one-dataset-out linear probe), `eval_ckpt.py`, `label_eff.py`, `prior_hyperplane.py`, `diagnose_embedding.py`, `hmm_improvements.py`, `hmm_on_foundation.py`, `model.py` + `train_lodo.py` + `calibrate.py` (first causal foundation model).

## Overnight machinery
`night.py` (queue driver, auto commit/push), `night_eval.py` (common evaluation + JSON schema), `night_report.py` (builds `docs/OVERNIGHT_REPORT.md` and `docs/figs_night/`), `sweep_jepa.sh`.

## Data written here
`runs/` — checkpoints (`*.pt`, gitignored), caches (`pool.npz`, `pseudo_hmm.npy`), `runs/night/<id>.json` (results, tracked), `runs/night/logs/` (training logs).
Rule: add a new candidate by writing a script that ends with `night_eval.eval_direct(...)` / `eval_features(...)` + `night_eval.save(...)`; the report picks it up automatically.
