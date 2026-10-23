# Speed limit of the network input: what changed, and how to re-evaluate U'n'Eye

Written 2026-10-23 (Claude Code, on the owner's request). Read this, then follow "To do".

## What changed

Blinks, tracker losses and digitising glitches gave huge eye velocities that reached the network (linear input) at up to a thousand times
the size of a saccade, and the network learned badly. Every network input now has a **speed limit of 1300 deg/s**: the first difference of the eye
position (dx, dy) is scaled by `limit / |v|` where the speed |v| exceeds it (the direction is kept; NaN stays NaN; the limit is in deg/s, so it holds at any rate).
1300 deg/s is the upper end of the peak speed of saccades in rhesus monkeys (1000-1300; humans about 700, saturating above 15-20 deg of amplitude): sources
Wikipedia "Saccade" (Fuchs 1967) and the Radboud course page `perception-and-action-37169a.pages.science.ru.nl/content/12_the_saccadic_system_pulsestep.html`.

| where | change |
|---|---|
| `uneye/functions.py` | new `clip_speed(dx, dy, max_speed=1300.0, fs=1000.0)` |
| `uneye/classifier.py` | `DNN(..., max_speed=1300.0)`; applied right after the NaN / Inf handling, before the network, in `train` (training and validation data), `predict`, `test`, `crossvalidate` (7 places). `None` or `0` = no limit |
| `online/common.py` | `velocity(X, Y, inf_correction=1.5, fs=1000.0, max_speed=1300.0)`; callers (`online/*.py`, `foundation/compare.py`) now pass `fs` |
| `foundation/model.py` | the old per-axis `np.clip(v, -2000, 2000)` is replaced by the norm limit at `MAX_SPEED = 1300` deg/s |
| `free_saccade/detectors.py`, `foundation/dino1d.py` | `limit_speed(v, max_speed)`; `feats`, `feats2`, `feats2c` take `max_speed=None` (OFF by default: the BiTCN input is arcsinh of v / noise and the unlabeled archive is not in degrees) |
| `cpp/include/uneye/detector.hpp`, `cpp/src/detector.cpp` | `Config::max_speed_deg_s = 1300.0` (0 = off); applied in `StreamingDetector::push` after NaN / Inf, for the windowed U-Net engines and the causal TCN alike |
| `cpp/src/bitcn.cpp`, `cpp/include/uneye/bitcn.hpp` | `features2(pos, T, fs, out, max_speed = 1300.0)` and `detect(..., max_speed = 1300.0)`: the limit is applied to (vx, vy) before the noise normalisation, as `feats2(..., max_speed=1300)` in Python |
| `cpp/tools/uneye_rt.cpp` | option `--max-speed DEG_S` (0 = off) |
| `cpp/app/include/eyelive/trainer.hpp`, `cpp/app/train/src/dataset.cpp` | the C++ trainer (EyeLive, `ITrainer`) builds its own (dX, dY) windows: `TrainConfig::max_speed_deg_s = 1300` is applied there too, after rotation and noise, before the network (it was missing in the first version of this change; the 4 train tests pass) |
| `app.js` (web labeler) | the velocities of the pre-labeler, the thresholds and the display are limited to 1300 deg/s |
| `foundation/eval_speed_limit.py` | new: old weights evaluated with the limit off / 1300 / 2000 / 2500 deg/s (writes `docs/figs_data/uneye_speed_limit_inference_only.json`) |
| `foundation/make_summary.py`, `docs/BENCHMARK_SUMMARY.md` | all cells that come from a U'n'Eye network are NA (`UNEYE_PENDING = True`) with a note at the top |

Not done / not verified: the Android sources are not in the working tree on the machine where this was written (the Android app builds the `cpp/` tree, so it gets the
C++ default); the research scripts of the self-supervised models (`foundation/unet_*.py`, `ssl_models`, `weak_labels`) and the label-free detectors were left unchanged (their inputs are
arcsinh-compressed or median-normalised; the unlabeled archive is not in degrees).

## What was measured (Intel Mac, CPU, PyTorch 2.2.2; 2026-10-23)

1. **The 34 C++ tests pass** (`ctest`), with the limit on by default: the goldens of parity, equivalence, bitcn did not change.
2. **Old U'n'Eye weights (trained without the limit) evaluated with the limit** (event F1 / kappa, the 300-trial subsets of `docs/BENCHMARK_SUMMARY.md`; `off` reproduces the table):

| dataset | weights | off | 1300 | 2000 | 2500 | samples > 1300 deg/s |
|---|---|---|---|---|---|---|
| d1 | 1+2+3 | 0.900 / 0.849 | 0.891 / 0.845 | 0.892 / 0.845 | 0.892 / 0.845 | 105 |
| d2 | 1+2+3 | 0.937 / 0.883 | same | same | same | 0 |
| d3 | 1+2+3 | 0.930 / 0.821 | same | same | same | 0 |
| d4 | 1+2+3 | 0.920 / 0.850 | 0.918 / 0.850 | 0.918 / 0.850 | 0.918 / 0.850 | 8 |
| Andersson | Andersson (5 classes) | 0.881 / 0.822 | 0.870 / 0.812 | 0.875 / 0.815 | 0.876 / 0.815 | 97 |
| Andersson | 1+2+3 | 0.535 / 0.337 | 0.543 / 0.336 | 0.542 / 0.337 | 0.543 / 0.337 | 97 |

   Raising the limit to 2500 does not recover the loss: the old weights learned to react to the unlimited outliers; only a network trained with the limit can be judged.
3. **C++ live detector** (`uneye_rt --model models/combined.bin --replay ... --lookahead 25`, committed labels, 300-trial subsets of set B; sample F1 / kappa, event precision),
   limit 0 -> 1300: d1 0.844 / 0.829 (precision 0.859) -> 0.839 / 0.825 (0.845); d2 0.861 / 0.850 unchanged; d3 (500 Hz) 0.801 / 0.793 unchanged; d4 0.853 / 0.848 (0.883) -> 0.853 / 0.847 (0.880).
   Latencies unchanged. Same old weights, so the same remark as above.
4. Raw 1-sample velocities of the labeled saccades of d1, d2, d3 (sets A and B; d4 not measured) are far below the limit (median peak 50-90 deg/s, p99.9 at most 566 deg/s; one saccade of d1 set B at 1237): the limit
   only touches outliers there. On a noisy tracker the raw 1-sample speed of a real 6 deg saccade reaches 1200-2000 deg/s (mocol data): there the signal must be smoothed
   (Savitzky-Golay 15 ms) BEFORE the network, as MS_CD does, or the limit must be higher (about 2500 deg/s).

## To do (on the machine that has the weights, the GPU and the archive)

1. `git pull`. Environment as in `AGENT.md`. `ctest --test-dir cpp/build` must stay green (34 tests).
2. **Retrain U'n'Eye with the limit** (the default of `uneye.DNN` is now `max_speed=1300`), with the SAME data and settings as the shipped weights, so that only the limit differs:
   `weights_1+2+3` (d1+d2+d3, set A), `weights_dataset1`, `weights_dataset2`, `weights_dataset3`, `weights_Andersson` (5 classes, `foundation/data.py::load_andersson("train")`), and `weights_synthetic`
   if it is used. Save them under new names first (for example `weights_1+2+3_limit1300`) and keep the old ones until the comparison is done. Check what the old
   training used (rates of d3 / Andersson 500 Hz: `sampfreq`), the notebook `analysis scripts/` and the paper (Bellet et al. 2019) if the recipe is not in the repository.
3. **Re-run what produced the U'n'Eye cells**: `python foundation/compare.py` (the "U'n'Eye, offline" rows; `foundation/runs/baselines_f1.json`), the night results `ref_uneye.json` and
   `ref_uneye_andersson_own.json` (`foundation/runs/night/`), `docs/figs_paper/results_uneye.json` (table 3, few labels: U'n'Eye retrained on N labeled trials of d1, 10 draws), and
   `foundation/workflow_sim.py` (table 3e, U'n'Eye trained on pseudo / expert labels, columns C1-C5, P1, P4, P5). `foundation/eval_speed_limit.py` shows the expected size of the change (about -0.01 in F1 on d1 / Andersson with the OLD weights).
4. Decide on the limit with the retrained network: run `foundation/eval_speed_limit.py` logic for limits off / 1300 / 2000 / 2500 deg/s on the NEW weights. If 1300 is not as good as off on d1 or
   Andersson, take the lowest value that is (the C++ default and `max_speed` follow it; Andersson's large saccades are the test: 97 samples above 1300 deg/s in its test subset).
5. **Re-export the models to C++** (`cpp/scripts/export_model.py ../../training/weights_1+2+3_limit1300 models/combined --window 200`, same for the others), regenerate the goldens (`cpp/scripts/make_golden.py`,
   `export_unet_golden.py`, `export_agreement_golden.py`, `export_calib_golden.py` as needed), `ctest` again, and re-run `uneye_rt --replay` on the four subsets (`--max-speed 0` and `1300`).
6. **BiTCN**: its inputs are unchanged by default (`max_speed=None` in Python; the C++ applies 1300 deg/s). Check that the C++ BiTCN gives the same event F1 on d1-d4 and Andersson with
   `max_speed` 0 and 1300 (`uneye::detect`); if it differs on Andersson, train the BiTCN with `feats2(..., max_speed=1300)` too (`foundation/sup_bitcn.py --inputs 2`) so that training and the
   C++ engine agree, and re-export (`foundation/export_bitcn.py`).
7. In `foundation/make_summary.py` set `UNEYE_PENDING = False`, run it, update the verdict texts that quote U'n'Eye numbers (tables 1, 3, 3b, 3e and the notes), and delete the
   note at the top of `docs/BENCHMARK_SUMMARY.md` or turn it into a result.

## Prompt for the other agent

> The repository `uneye` changed on 2026-10-23: every neural-network input now has a speed limit of 1300 deg/s (`uneye/functions.py::clip_speed`, `uneye.DNN(max_speed=1300)`,
> `online/common.py::velocity(fs=, max_speed=)`, C++ `Config::max_speed_deg_s` and `features2(..., max_speed)`, the web labeler). Read `docs/SPEED_LIMIT_HANDOFF.md` first: it lists the
> changes, what was measured (old weights evaluated with the limit: F1 -0.009 on d1, -0.002 on d4, -0.011 on Andersson with its own weights, unchanged on d2 and d3; the 34 C++ tests pass), and the
> steps. All U'n'Eye cells of `docs/BENCHMARK_SUMMARY.md` are NA because the shipped U'n'Eye weights were trained without the limit. Your job: (1) retrain U'n'Eye with the limit on the same data and
> settings as the shipped weights (`weights_1+2+3`, `weights_dataset1|2|3`, `weights_Andersson`), saving under new names; (2) re-run `foundation/compare.py`, the experiments of tables 3 and 3e
> and `foundation/eval_speed_limit.py`; choose the limit with the retrained networks (lowest value that does not lose F1 / kappa against no limit on d1 and Andersson); (3) re-export the models to C++
> (`cpp/scripts/export_model.py`), regenerate the goldens, rebuild and run `ctest` (34 tests) and `uneye_rt --replay` with `--max-speed 0` and `1300`; (4) check the C++ BiTCN with and without the limit;
> (5) set `UNEYE_PENDING = False` in `foundation/make_summary.py`, regenerate `docs/BENCHMARK_SUMMARY.md`, update the verdict texts that quote U'n'Eye numbers, and commit. Do not push data
> (`data/Andersson/`, `archive/` and the weights of `foundation/runs/*.pt` are git-ignored). Report numbers from scripts that were run, as in `AGENT.md`.
