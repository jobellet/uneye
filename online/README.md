# Causal (past-only) network for online detection

The original U-Net looks at both sides of a sample (convolutions with symmetric padding, a decoder that
upsamples). In a live bin its newest samples have no future, so they are labelled poorly.
`causal_net.py` is a **forward-only** network: the output at time *t* uses only samples ≤ *t*.

* Input: eye velocity dX, dY per sample (as in U'n'Eye) + speed.
* Stem causal conv (k=5) + 6 residual blocks of causal dilated convs (k=3, dilations 1,2,4,8,16,32), ReLU → BatchNorm like the original, 1×1 softmax head.
* 48 channels, 43 k parameters, receptive field 131 samples (131 ms at 1 kHz). Fits inside a 200 ms bin.
* Can run as a stateful filter: one cheap step per new sample, no window re-evaluation (`cpp/src/causal_engine.cpp`).

## Train (about 5–30 min on 4 CPU cores, no GPU needed)

```
cd online
python3 train_causal.py --channels 48 --pos-weight 3 --dilations 1,2,4,8,16,32 --out weights_causal
python3 export_causal.py weights_causal ../cpp/models/causal      # .bin (C++ streaming), .onnx, golden test
python3 evaluate_online.py                                        # test on set B, compared with the original U-Net
```

Same logic as `uneye.DNN.train`: set A of datasets 1,2,3 pooled and shuffled (seed 1), 30 trials held out for
validation / early stopping, rotation augmentation, MCLoss + L2 0.001, Adam 1e-3, lr halved when validation gets worse
(best weights restored), stop after more than 3 bad epochs. Differences: minibatches of 64 trials, and `--pos-weight`
(saccade class weight in the loss). The variant was chosen with `select_variant.py` on **validation trials only**:

| Variant | params | RF | val F1 (thr 0.5) | val F1 (tuned thr) |
|---|---|---|---|---|
| 24 ch, weight 1 | 9.5k | 67 | 0.668 | 0.728 |
| 24 ch, weight 3 | 9.5k | 67 | 0.693 | 0.735 |
| 48 ch, weight 3 | 36k | 67 | 0.722 | 0.726 |
| **48 ch, weight 3, dilation up to 32** | 43k | 131 | **0.737** | **0.745** |

## Test: last sample of a 200 ms bin

Protocol (both networks): at every time *t* the network sees the 200 ms bin ending at *t*; the label of *t* is the output (no lookahead, threshold 0.5, no post-processing).
Test = set B of datasets 1, 2, 3 (300 random trials each, seed 10, like `analysis scripts/evaluation_networks.ipynb`).
Original U-Net = `training/weights_1+2+3` (trained on the same set A data). "U-Net offline" uses the whole trial including the future and is only a reference.
Event metrics: prediction runs of at least 3 samples; "alarm delay" = time from true onset until 3 consecutive saccade labels.

| Set | Model | F1 | κ | Precision | Recall | Event recall | Event precision | Alarm delay (median / p95 ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | U-Net offline (reference) | 0.865 | 0.853 | 0.890 | 0.841 | 0.979 | 0.828 | – |
| 1 | U-Net, last bin | 0.680 | 0.657 | 0.858 | 0.564 | 0.968 | 0.556 | 11 / 20 |
| 1 | **Causal, last bin** | **0.789** | **0.771** | 0.874 | 0.719 | 0.977 | 0.788 | 8 / 17 |
| 2 | U-Net offline (reference) | 0.881 | 0.870 | 0.904 | 0.859 | 0.942 | 0.938 | – |
| 2 | U-Net, last bin | 0.659 | 0.632 | 0.783 | 0.568 | 0.881 | 0.716 | 13 / 22 |
| 2 | **Causal, last bin** | **0.787** | **0.770** | 0.910 | 0.693 | 0.925 | 0.910 | 10 / 19 |
| 3 (500 Hz) | U-Net offline (reference) | 0.828 | 0.821 | 0.843 | 0.814 | 0.910 | 0.956 | – |
| 3 (500 Hz) | U-Net, last bin | 0.590 | 0.576 | 0.669 | 0.528 | 0.910 | 0.799 | 12 / 18 |
| 3 (500 Hz) | Causal, last bin | 0.602 | 0.586 | 0.610 | 0.594 | 0.799 | 0.767 | 10 / 16 |

* The causal net is clearly better than the U-Net on its last bin for the 1 kHz data (+0.11 and +0.13 F1) and about equal for dataset 3. At 500 Hz a 200 ms bin has 100 samples, less than the 131-sample receptive field, so the net is run on the bin only (strict protocol).
* Both are far below the offline U-Net (0.83–0.88): a sample at the start of a saccade cannot be recognised before the movement is visible. This is a limit of any online method, not of the implementation. Detection of the event itself is fast (alarm after about 8–10 ms median).
* With a threshold tuned on the validation trials (0.4) the numbers change by ≤ 0.01 F1 (see `results_online.json` after running the script).
* Cost with the C++ streaming engine: 53 µs median, 82 µs p99 per sample (ONNX Runtime U-Net on a 200 ms bin: 120 µs / 178 µs) and the result is final immediately.

C++ check on 200 replayed trials of dataset1 setB (`uneye_rt --model models/causal.bin --replay ... --fast`):
F1 0.81 (U-Net fast mode 0.71), event precision 0.81 (0.63), onset known after 8 ms median (10 ms).

## Caveats
* Only one training run per variant (seed 1); no confidence intervals. Differences below about 0.02 F1 should not be over-interpreted.
* The thresholds and the variant were chosen on 30 validation trials per dataset, then the test set was scored once for the chosen variant.
* Dataset 4 and the 5-class Andersson setting are not covered.
