# eyelive_cli

Headless front-end of the EyeLive engine: records a stream (or a replayed file, or a simulation) into a session file, labels it with a detector, fires triggers, and can pre-label and train. Everything the GUI does at acquisition time, without a window.

```sh
# 1. record 3 s of a simulated eye as fast as possible and print the events found by the label-free HMM
eyelive_cli --sim --duration 3 --session /tmp/a.eyelive

# 2. replay a recorded trace (CSV "t,x,y[,valid]") at real time, with the fast detector, and write the batch events to a file
eyelive_cli --replay trace.csv --detector fast_threshold --session /tmp/b.eyelive --events-out events.csv

# 3. listen to a tracker streaming UDP packets (protocol v1: binary "EYE1" or text lines "t x y [valid]") on port 5005 for one hour
eyelive_cli --udp 5005 --duration 3600 --session /data/subject1.eyelive --scale auto

# 4. fire a UDP message 10 ms after every saccade onset, from a rules file
eyelive_cli --udp 5005 --triggers rules.txt --duration 600

# 5. pre-label the first 50 s with the detector and train a U'n'Eye on them (no review: for tests and benchmarks)
eyelive_cli --sim --duration 60 --prelabel-seconds 50 --train --session /tmp/c.eyelive
```

A rules file has one rule per line (`#` starts a comment):

```
name=sacc;cls=1;anchor=provisional;delay_ms=10;sink=udp:127.0.0.1:6000;payload=SACCADE {t} {amp}
name=micro;cls=1;anchor=final;amp_max=1.0;sink=log:/tmp/micro.log;payload=MICRO {t} {amp}
```

Output: one line per final event (`event <onset s> <offset s> class=<c> amp=<deg> peak=<deg/s>`), then a summary line (samples, rate, dropped packets, gaps, events, latency, scale). Exit code 0 on success, 1 when the session cannot start or no data was recorded, 2 for a bad command line. The session file can be reopened with `Recorder::open_readonly` and read live from MATLAB / Python (see `cpp/app/readers/README.md`).

## uneye_train: training, cross-validation and prediction from the command line

`tools/uneye_train.cpp` trains a U'n'Eye network with the native C++ trainer (`eyelive::make_trainer`, the recipe of the web / Android labelers), without
Python or PyTorch. Data are the labelers' CSV export (U'n'Eye convention): `X.csv`, `Y.csv` = one trial per row (trials may have different lengths,
empty / `nan` = invalid sample), `L.csv` = labels per trial (0 fixation, 1 saccade, 2 blink; -1 / empty / `nan` = not reviewed, never trained on or scored;
labels >= `--classes` are ignored). One process is one training, so the folds of a cross-validation run in parallel, one CPU each.

```sh
# train on everything, starting from the shipped network, with the labelers' 11-ms Savitzky-Golay display filter applied to the input
uneye_train train --x X.csv --y Y.csv --labels L.csv --init models/combined.bin --savgol 11 --out mine.bin
# 5-fold cross-validation grouped by session (G.csv: one group per trial), fold i in its own process (e.g. a SLURM array task)
uneye_train cv --x X.csv --y Y.csv --labels L.csv --groups G.csv --folds 5 --fold $i --init models/combined.bin --savgol 11 --threads 1 \
               --metrics fold$i.json --out fold$i.bin
# apply a model: one row of per-sample classes per trial (--probs: p(saccade))
uneye_train predict --model mine.bin --x X.csv --y Y.csv --savgol 11 --out pred.csv
```

`cv` writes the held-out scores of its fold as JSON: event F1 of the saccades (a predicted run of >= 3 samples overlapping a labelled saccade), event
precision and recall, Cohen's kappa sample by sample (saccade vs rest), on the reviewed samples of the held-out trials only, plus the confusion counts
(`tp fp fn tn`, `n_true`, `n_pred`, matched events) so the folds can be pooled exactly. Prediction: argmax of the softmax (2 classes: p > 0.5),
then U'n'Eye's post-processing (saccades closer than `--merge-ms` 5 merged, shorter than `--min-dur-ms` 5 removed). `--savgol W` must
be the same at training and prediction (the web labeler does the same with its display filter since 2026-10-25).

Recipe: by default the labelers' recipe (`TrainConfig::resample_each_epoch`, `--batch 32`): whole traces held out for validation, training windows cut
again every epoch around the labelled samples with random offsets, random rotations and mirrors; early stopping on the validation loss of the held-out
traces (learning rate halved and best weights restored when it does not improve, stop after `--patience` such epochs, best weights returned).
`--classic` restores the original U'n'Eye recipe. Cross-validation on 71 mocol trials (345 saccades, 5 session folds, from `combined.bin`,
`--savgol 11`, scored with the same scorer and post-processing as the PyTorch pipeline): event F1 0.943, kappa 0.851 (classic 0.912 / 0.833; PyTorch
0.930-0.934 / 0.840-0.844); 40-100 epochs, about 10 min per fold on one CPU. Build: `cmake --build build --target uneye_train` (or, without CMake:
`g++ -O3 -std=c++17 -I cpp/app/include -I cpp/app/train/include cpp/app/cli/tools/uneye_train.cpp cpp/app/train/src/{trainer,dataset,model_io}.cpp -lpthread`).
