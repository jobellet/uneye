# `archive/` — unlabeled gaze recordings (local only, not in git)

`archive/` (3.4 GB) is in `.gitignore`. It is the multi-source dataset that `free_saccade/` expects (the "Kaggle h5 layout").
Measured on 2026-10-09 by reading the files (no labels anywhere).

## Layout
```
archive/
  subject_h5_metadata.csv                      filename, dataset, subject, num_sequences, gaze_x_mean, gaze_x_std, dt_mean
  datasets_by_subject/datasets_by_subject/*.h5 6083 files, one per subject / session
  predict_saccade_endpoints/
    training_set_high_disp/high_disp_snippets.h5   one group per source: traces (N, T, 2) float64, attrs fs, unit='raw', n_samples
    read_training_set.py, train_predictive_coding.py
```
Each subject file: one dataset `x`, float32, shape `(n_sequences, 1024, 3)` = `(x, y, dt)`; root attrs `dataset`, `subject`, `num_sequences`.
Read with `free_saccade.data.H5Source("archive/datasets_by_subject/datasets_by_subject", "archive/subject_h5_metadata.csv")`
(`GROUP_COL = "dataset"`).

## Per source

| dataset | files | sequences | rate from `dt` | usable for saccade detection? |
|---|---|---|---|---|
| VEDB | 517 | 164 075 | 120 Hz | yes, low rate (microsaccades not resolvable) |
| EMTeC | 107 | 420 290 | ~2 kHz (see dt below) | yes, best rate, ~62 h |
| GazeCom | 54 | 8 687 | 250 Hz | yes |
| 360EM | 13 | 2 905 | 125 Hz | yes, low rate |
| GazeBase | 1 | 1 654 | 1 kHz | yes (small) |
| Lund2013 | 110 | 1 295 | `dt` = 1.0 (wrong unit) | yes, true rate 500 Hz (`high_disp_snippets.h5` attr) |
| DUT-OMRON | 5 168 | 10 302 | ~30 Hz, 63 % dt = 0 | no: image-saliency fixation data, not raw samples |
| EGTEA | 86 | 5 889 | 30 Hz | no: video rate, saccades shorter than a sample |
| OASST-ETC | 25 | 227 | `dt` 0.37-1.0 | no / unclear |
| VQA-MHUG | 2 | 327 | — | no: x, y are all NaN |

## Pitfalls (check before trusting any number)
1. **Positions are normalized, not degrees.** Per-file std ≈ 1, values clipped to ±6 (6 % of VEDB samples sit at the clip).
   Thresholds in deg/s (Engbert-Kliegl is relative, fine; I-VT, main sequence, injection amplitudes are not) are meaningless unless
   rescaled. `UNIT_SCALE` cannot recover degrees: the scale differs per file.
2. **`dt` is inconsistent.** EMTeC alternates 0 and 0.001 (2 kHz timestamps rounded to ms): `data.resample` takes the median of
   `dt > 0` = 1 ms and plays it at half speed. Lund2013 / OASST-ETC have `dt` ≈ 1 → treated as ms → wrong rate.
   Samples with `dt = 0` have distinct positions, so EMTeC really is ~2 kHz (`dt_mean` 0.52 ms); `high_disp_snippets.h5` says 1000 Hz for
   EMTeC, probably after downsampling. For Lund2013 use 500 Hz (snippet attr). Safest: a fixed rate per source instead of `dt`.
3. **Padding.** Sequences are padded to 1024 by repeating the last sample (DUT-OMRON, VEDB): flat tails look like perfect fixation.
4. Sources are very unbalanced in files (DUT-OMRON = 85 % of files) and in hours (VEDB, EMTeC dominate): sample per source.
