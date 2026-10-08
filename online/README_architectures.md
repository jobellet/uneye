# Which causal network, trained how, for the project?

Notebook: [`UnEye_online_architectures.ipynb`](UnEye_online_architectures.ipynb) (Colab / Kaggle ready; `QUICK = False` for the real run).
Code: `archs.py` (backbones), `experiment.py` (data, training, teacher-student, plan runner), `metrics.py`, `viz.py` (figures), `weak_labels.py`, `export_causal.py`.
Regenerate the notebook with `python3 make_notebook.py`.

## What version 1 showed (360 runs: 4 backbones x 5 strategies x 6 N x 3 seeds, Kaggle 2 x T4)
Pooled kappa on set B, threshold 0.5, mean of 3 seeds:

| labeled trials | TCN scratch | TCN weak-label pretrain | TCN-lite scratch | GRU scratch | S4D best |
|---|---|---|---|---|---|
| 5 | 0.26 (one seed failed) | 0.52 | 0.36 | 0.56 | 0.33 |
| 10 | 0.63 | 0.64 | 0.57 | 0.60 | 0.55 |
| 20 | 0.67 | 0.66 | 0.63 | 0.66 | 0.59 |
| 50 | 0.70 | 0.69 | 0.65 | 0.66 | 0.53 |
| 100 | 0.74 | 0.71 | 0.67 | 0.68 | 0.53 |
| 300 | **0.75** | 0.74 | 0.69 | 0.68 | 0.46 |

* **Causal TCN from scratch** is the best model from about 50 labeled trials on and still improves with more labels (0.77 with ~2350 labels). With 20 labels it already beats the original U'n'Eye evaluated on the last 200 ms bin (0.64, trained with all labels).
* **JEPA / raw-signal self-supervised pretraining gave no gain** from 20 labels on and got *worse* with more labels (TCN 0.68 at N=20 to 0.58 at N=300, a sign that the fine-tuning recipe, not only the idea, was wrong); frozen-encoder probes plateaued at 0.52-0.58. They were removed.
* **S4D** was unstable (0.46-0.59, worse with more labels); **GRU** is stable but plateaus at 0.66-0.68; **TCN-lite** is 0.04-0.06 below TCN with 2.6x fewer parameters.
* Free pseudo-labels from the classic Engbert-Kliegl detector (`weak_ft`) were as good as JEPA at small N and cost seconds.

## Version 2: finalists, plus three new questions
Candidates: **TCN** (main), **TCN-lite** ("cheap"), **GRU** (one-point reference). Strategies: `scratch`, `weak_ft`, and new:
* **`distill`**: the original non-causal U'n'Eye is trained on the N labels, labels all set-A recordings (soft probabilities), and the causal network learns from those labels plus the true ones. Tests whether unlabeled recordings help through a teacher that sees the future.
* **Lookahead L**: the network output at time t is the label of sample t-L (still real time, L ms late; the network gets L ms of future context). Everything else equal, the gap to the offline model is probably mostly this missing future. Quick check (TCN from scratch, 300 labels, one run): kappa 0.73 without delay, **0.87 with 10 ms**, reproduced by the C++ engine (`uneye_rt --label-delay 10`).
* **The right baseline**: the original U'n'Eye trained on the **same N labeled trials**, offline and on the last sample of a 200 ms bin (the teacher is trained anyway).

All methods use the same labeled subset for a given (N, seed), so the notebook reports **paired differences** (is a strategy better than scratch in the same seeds?). Metrics: kappa, MCC, event F1, |onset|/|offset| error (ms, as in Bellet et al.), alarm delay (includes L), false alarms per minute.
Figures follow the paper: learning curves (Fig 7A), per-metric boxplots with a reference line (Fig 4), example traces with label bars (Fig 3), main-sequence plots of hits / false alarms / misses (Fig 5).
The last section trains the chosen design on all labels and exports it to the C++ engine (full TCN only), including the label delay.

Not changed from v1 on purpose: model selection by validation **loss** (with 3 validation trials at N=10 a validation kappa is too noisy), L2 1e-4, positive-class weight 3, rotation augmentation, early stopping not before epoch 15. Weak labels now ignore blink/dropout samples (v1 produced overflow warnings from non-finite eye positions).

## Version 2 results (Kaggle 2 x T4, 7.7 min for 170 student runs + 20 teachers, 5 seeds)
Pooled kappa on set B (threshold 0.5), mean over 5 seeds. Seeds share the labeled subset across methods (paired).

| labeled trials | U'n'Eye online (last bin) | **TCN scratch** | TCN distill | TCN weak-label | TCN-lite scratch | GRU scratch | U'n'Eye offline (target) |
|---|---|---|---|---|---|---|---|
| 10 | 0.56 | 0.62 | **0.66** | 0.63 | 0.49 | – | 0.73 |
| 30 | 0.58 | 0.67 | 0.65 | 0.67 | 0.64 | 0.65 | 0.78 |
| 100 | 0.59 | **0.72** | 0.71 | 0.72 | 0.67 | 0.68 | 0.82 |
| 300 | 0.56 | **0.74** | 0.70 | 0.74 | 0.68 | – | 0.81 |

Lookahead (TCN from scratch; `alarm delay` includes L):

| L (ms) | kappa, N=30 | kappa, N=100 | event F1, N=100 | |onset error| (ms) | alarm delay (ms) | false alarms / min |
|---|---|---|---|---|---|---|
| 0 | 0.64 | 0.72 | 0.83 | 7.5 | 9.7 | 43 |
| 5 | 0.71 | 0.81 | 0.84 | 3.0 | 10.1 | 39 |
| 10 | 0.78 | 0.82 | 0.83 | 1.9 | 12.0 | 35 |
| 20 | 0.76 | 0.83 | 0.86 | 1.9 | 20.9 | 27 |
| 40 | 0.76 | 0.83 | 0.87 | 2.4 | 41.1 | 25 |

* **A few ms of lookahead is the main lever**: 5-10 ms bring the causal TCN to the level of the offline U'n'Eye trained on the same labels (0.82 at N=100) and its onset error (1.9 ms against 2.2 ms). Beyond about 10-20 ms there is no further gain.
* The causal TCN beats the original U'n'Eye evaluated online (last bin) at every N, including N=10, and the gap grows with N (the online U'n'Eye does not improve with more labels).
* **Distillation** helps only with very few labels: at N=10 +0.05 kappa and +0.13 event F1 (5/5 seeds) and about a third of the false alarms; at N>=100 it is not better, and at N=300 it is worse in kappa (-0.03, 0/5 seeds) although event F1 stays slightly higher. **Weak-label pretraining** gives no consistent gain for the TCN.
* **TCN-lite** is 0.05-0.06 below TCN at every N; **GRU** is at the TCN-lite level.
* **False alarms** are still high without lookahead (28-53 per minute against 5-8 for the offline U'n'Eye); lookahead and distillation both reduce them. Misses are mostly small, slow movements (amplitude 0.1-0.5 deg, 20-60 deg/s), and the false alarms lie on the main sequence at amplitudes below 0.3 deg.
* One final model trained once is high-variance, mostly on the small dataset 3 (same design: kappa 0.61 in one run, 0.78 in another), so `train_final` now trains several seeds and keeps the lowest validation loss.

## Speed
Data live on the GPU (`experiment.Bank`), no per-step host sync, and all independent runs are spread over all GPUs with several worker processes per GPU (`run_plan(..., devices=default_devices(2))`). Finished runs are stored in `results_v2.json`; `resume=True` skips them. (On a 4-core CPU box the same small grid went from 272 s to 111 s; the GPU gain has not been measured here.)

## Honest status
v2 was tested here only at small scale (QUICK notebook run on CPU, a 6-run sanity grid at N=30). The 170-run grid is for Colab/Kaggle. One training run per (design, N, seed); differences below ~0.02 kappa should not be over-interpreted.
