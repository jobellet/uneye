# Causal architectures + self-supervised pretraining (JEPA) for online detection with few labels

Notebook: [`UnEye_online_architectures.ipynb`](UnEye_online_architectures.ipynb) (Colab / Kaggle ready; set `QUICK = False` for the real run).
Code: `archs.py` (backbones), `jepa.py` (SSL), `weak_labels.py`, `experiment.py` (data, training, grid), `metrics.py`.
Regenerate the notebook with `python3 make_notebook.py`.

## Run on Colab / Kaggle
1. Open the notebook (Colab: *File → Open notebook → GitHub*, paste `jobellet/uneye`, branch `claude/pipeline-cpp-onyx-isignal-krrw75`; Kaggle: import the file and switch *Internet* on).
2. Turn on a GPU (optional; the S4D and GRU models are much faster on GPU, the TCNs run fine on CPU).
3. The first cell clones the repository (the data is in it) and installs the few missing packages. Run all cells.

## State of the art checked (October 2026) and what it led to
* **JEPA for sensor time series** is new and promising but not settled: HAR-JEPA for wearable IMU data ([arXiv 2607.16350](https://arxiv.org/abs/2607.16350)), CHARM ([2605.31580](https://arxiv.org/abs/2605.31580)), ECG/I-JEPA variants ([2607.01145](https://arxiv.org/abs/2607.01145)), automotive monitoring ([2602.09985](https://arxiv.org/abs/2602.09985)), and an analysis that finds *mixed* evidence for forecasting gains ([2609.31680](https://arxiv.org/abs/2609.31680)). I found **no published use of JEPA / self-supervised pretraining for saccade detection** (closest: semi-supervised event-camera tracking, SaccadeX, WACV 2026), so here it is a hypothesis to test, not a known win.
* **State-space models** (S4D, LRU, Mamba) have a fixed-size recurrent state, so the cost per new sample does not depend on the history: a good fit for streaming. The official Mamba kernels need a GPU; S4D is simple, runs on CPU and has an exact recurrent form (`S4DLayer.step`, checked against the FFT form to 5e-6).
* **Metrics**: see the notebook (section 1).

## The architectures
| name | idea | params | receptive field |
|---|---|---|---|
| `tcn` | causal dilated convs + residual (previous PR) | 43 k | 131 samples |
| `tcn_lite` | depthwise-separable version, ~3x fewer MACs | 17 k | 131 samples |
| `s4d` | 4 diagonal state-space layers (16 complex states / channel), GLU mixing | 32 k | unlimited, decaying; O(1) state per step |
| `gru` | causal conv stem + 2-layer GRU | 29 k | unlimited |

All take the same input as U'n'Eye (velocity dX, dY, plus speed) and are checked to be strictly causal.

## Self-supervision without labels
* **`jepa`**: context encoder sees a corrupted signal (masked blocks, noise, random rotation); an EMA target encoder sees the clean signal; small predictors forecast the target *embedding* at t, t+5, t+10, t+20. Stop-gradient + EMA + a variance term prevent collapse (monitor `emb_std`).
* **`recon`** (ablation): same, but predicts the raw future velocity instead of a latent. Tests whether the latent target is what helps.
* **`weak`**: supervised pretraining on pseudo-labels from the classic Engbert-Kliegl velocity-threshold detector – free, noisy labels.
* All pretraining uses only set-A recordings (optionally dataset 4), never labels and never set B.
* Downstream: fine-tune on N human-labeled trials (all weights, or only the head for a linear probe).

## Honest status
The code and the notebook are tested here only at small scale (CPU). See the pilot results below for what was actually measured; the full grid (all four backbones, 3 seeds, N up to 300) is meant to be run on Colab/Kaggle.
