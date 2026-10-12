# Assisted annotation: simulation on ground truth

Tool: `eyelive_autolearn_sim` (cpp/app/cli/tools). The real engine runs on 60 s of dataset 1 (116 saccades, 1000 Hz).
The dataset labels play the expert, 5 s window by window. Every adopted model is scored on 60 s of held-out trials (123 saccades) against the ground truth.

Teacher (label-free HMM) on the held-out set: **F1 0.944, kappa 0.805** (tp 118, fp 9, fn 5).

## Mode `oracle` (expert corrects every window), 1 CPU thread, ~1.5 s per epoch

| reviewed (s) | model | epochs | train time (s) | test F1 | test kappa | next-window F1 before review |
|---|---|---|---|---|---|---|
| 5 | #1 | 48 | 72.1 | 0.915 | 0.801 | 0.897 (15 ev.) |
| 10 | #2 | 73 | 109.8 | 0.888 | 0.795 | 0.480 (7) |
| 15 | #3 | 150 | 224.7 | 0.919 | 0.808 | 1.000 (8) |
| 20 | #4 | 60 | 90.1 | 0.918 | 0.811 | 1.000 (10) |
| 25 | #5 | 85 | 128.3 | 0.925 | 0.797 | 0.923 (6) |
| 30 | #6 | 27 | 41.3 | 0.919 | 0.803 | 1.000 (8) |
| 35 | #7 | 67 | 105.6 | 0.937 | 0.808 | 1.000 (11) |
| 40 | #8 | 59 | 90.9 | 0.921 | 0.815 | 0.923 (7) |
| 45 | #9 | 29 | 44.0 | 0.924 | 0.802 | 1.000 (15) |
| 50 | #10 | 107 | 161.9 | 0.922 | 0.843 | 0.917 (11) |
| 55 | #11 | 75 | 112.9 | 0.939 | 0.856 | 0.857 (6) |
| 60 | #13 | 42 | 67.8 | 0.944 | 0.846 | 1.000 (12) |

Reading: event F1 reaches the teacher's level (0.944) only at the end; kappa (sample-level agreement, boundary precision) ends clearly higher (0.846 vs 0.805), so the student learns the expert's boundaries better than the HMM, not more events. Training is CPU-bound (CPU/wall 1.0 on one thread; about 4x faster with 4 threads).

## Mode `accept` (no correction: the student learns from the teacher's own labels)

4 threads, same data. Test F1 / kappa per model (reviewed 5 s ... 60 s):

| model | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11 | #13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.890 | 0.875 | 0.888 | 0.857 | 0.900 | 0.882 | 0.843 | 0.858 | 0.847 | 0.865 | 0.851 | 0.848 |
| kappa | 0.791 | 0.744 | 0.751 | 0.704 | 0.658 | 0.656 | 0.643 | 0.665 | 0.662 | 0.660 | 0.643 | 0.657 |

Training took 14-75 s per model (CPU/wall 3.3 with 4 threads).

**Answer: without correction the student does not beat the teacher on the ground truth.** It ends at F1 0.848 / kappa 0.657, below the teacher (0.944 / 0.805), and it gets worse as it trains on its own labels (kappa 0.79 -> 0.66). With expert correction (oracle mode) it reaches F1 0.944 and kappa 0.846, i.e. above the teacher in kappa. The gain therefore comes from the corrections, not from distillation.

Caveats: one dataset, one run per mode, a single 60 s training stream, an ideal (error-free) expert in oracle mode.
