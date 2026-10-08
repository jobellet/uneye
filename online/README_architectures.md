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

## Pilot result (CPU, `tcn` backbone only, 2 seeds, short pretraining) – read before the big run

Pooled kappa on set B (threshold 0.5), mean of 2 seeds. Pretraining: 1200 steps (~3 min) on the 2350 set-A recordings without labels.
Original U'n'Eye on the last sample of a 200 ms bin (trained with all ~2350 labeled recordings): kappa 0.637.

| labeled trials | scratch | JEPA + fine-tune | JEPA, frozen (linear probe) | raw-signal SSL + fine-tune | weak labels + fine-tune |
|---|---|---|---|---|---|
| 5 | 0.27 (0.00 / 0.53) | 0.50 (0.48 / 0.52) | 0.45 | 0.47 | **0.54** (0.48 / 0.60) |
| 20 | 0.66 | **0.68** | 0.54 | 0.68 | 0.65 |
| 100 | **0.73** | 0.60 (0.66 with tuned threshold) | 0.58 | 0.57 (0.67 tuned) | 0.72 |

What this does and does not show:
* **With 5 labeled trials, any pretraining makes training reliable** (one of the two scratch runs never left the "no saccade" solution; kappa 0.00). With 5 trials a kappa of about 0.5 is reached, and the original U-Net with all labels gets 0.64.
* **From 20 trials on, pretraining gives no clear gain** over training from scratch (+0.02, inside the seed noise), and at 100 trials fine-tuning a JEPA encoder was *worse* at the default threshold (0.60 vs 0.73; the gap shrinks to 0.66 vs 0.73 with a tuned threshold, so part of it is calibration).
* **JEPA (latent target) was not better than the simple raw-signal predictor**, and the free weak-label pretraining (Engbert-Kliegl pseudo-labels, 22 s) was as good as or better than JEPA (165 s) at 5 and 100 labels. In this pilot the JEPA idea is therefore **not yet supported** for this task; it may need much longer pretraining, more unlabeled data (dataset 4), a lower fine-tuning learning rate, or gradual unfreezing. These are the first things to try in the notebook.
* The pilot is small: 2 seeds, one backbone, short pretraining. Differences below about 0.05 are noise. The first version of the pilot had an early-stopping flaw that made the scratch baseline artificially bad (kappa 0 at N=100); it is fixed (`min_epochs`), and any similar protocol problem would produce the same kind of false "JEPA wins".
* S4D, GRU and light TCN are implemented and tested for causality and speed but **not yet trained in this study**.

## Speed (version 2 of the experiment code)
The runs are tiny (43 k parameters, batches of 32 x 700 samples), so a GPU is *launch-bound*, not compute-bound: the CPU thread that launches the kernels is at 100 % while the GPU waits. What was changed:
* **Data on the device.** `experiment.Bank` keeps all trials on the GPU and draws a batch with a few gather operations (before: a Python loop over trials and a host-to-device copy at every step).
* **Vectorised masking** in `jepa.corrupt` (before: a Python double loop = hundreds of tiny kernels per step) and a **fused EMA update** (`torch._foreach_*`).
* **No host synchronisation per step**: JEPA loss statistics stay on the device and are read once per epoch; `cudnn.benchmark` is on.
* **Several jobs at once**: `run_grid(..., devices=default_devices(2))` starts one worker process per GPU x 2 (never more than the CPU cores). All independent runs (pretraining per backbone, then every strategy x N x seed) are spread over all GPUs, longest first. The old code used one process and one GPU.
* **Resume**: finished runs are in `results_grid.json`; `resume=True` skips them.

Measured on a 4-core CPU box (no GPU available here): the same small grid took 272 s in one process and 111 s with 4 worker processes (2.4x). The GPU gain was **not measured** here; expect more than on CPU, because the second GPU is used and each GPU is shared by 2 processes, but check the `elapsed` time printed per run.
Remaining single-thread work per run: the event-level metrics (Python loops over trials) at the end of each run.

## Honest status
The code and the notebook are tested here only at small scale (CPU). The full grid (four backbones, 3 seeds, N up to 300, longer pretraining) is meant to be run on Colab/Kaggle.
