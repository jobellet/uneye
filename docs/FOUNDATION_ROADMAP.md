# Foundation model for eye movements: state and roadmap (hand-over, 2026-10-09)

Written by the Claude Code session on the owner's MacBook Air M1 so that any other session can continue. Read with
`docs/ROADMAP.md` (C++ safety demo, done) and `free_saccade/RESULTS.md` (label-free experiments). Owner speaks French.

## The owner's goal
A **foundation model** for eye-movement segmentation that works on ANY new dataset (any tracker, rate, unit, task), trained once on
several datasets with human ground truth; on a new dataset it should detect saccades (and blinks, fixations, pursuit; an
arbitrary number of classes) by itself, or after very few user clicks. End product: **everything in C++**, fast, self-contained
(model exported to ONNX / the native `.bin` format), with a **GUI** that receives data online (simulated by replaying the
datasets), shows segments with the algorithm's prediction **hidden**, lets the user click events; the algorithm aligns to the
user's convention (onset / offset criteria differ between experts: same event F1, different Cohen's kappa) and asks for the next
segment where it is least sure, until it agrees with the user on N segments in a row. Only use the parts of CEBRA that are needed.

Bar to beat (owner's rule): **event F1 at least equal to U'n'Eye**, otherwise the approach is not good enough.

## Data (all loaded by `foundation/data.py`, 1 kHz)
- datasets 1-4 of Bellet et al. 2019 (`data/`, saccade vs not), set A = train, set B = test.
- Andersson et al. 2017 / Lund 2013 multi-class benchmark: `data/Andersson/` (downloaded from
  github.com/richardandersson/EyeMovementDetectorEvaluation, GPL-3.0, **not in git**; re-download with the file list of that repo,
  folder `annotated_data/`). 500 Hz, classes fixation, saccade, PSO, pursuit, blink. Train = trials coded by RA only; test = the
  34 two-coder trials of the article (RA vs MN gives the human ceiling: 5-class kappa 0.81). **PSOs can be ignored** (owner).
- `archive/` (local only, 3.4 GB): many unlabeled recordings (VEDB, EMTeC, GazeCom, ...), see `docs/ARCHIVE_DATA.md`.

## What was tested (numbers measured on the M1, details in the commit messages and result files)
| approach | human labels of the target | result on dataset 1 (kappa / event F1) | verdict |
|---|---|---|---|
| U'n'Eye supervised (training/weights_1+2+3), offline | all set A | **0.856 / 0.899** | the bar |
| HMM, unsupervised (free_saccade) | 0 | 0.78 / 0.92 (500 Hz protocol) | best label-free baseline so far |
| seeded CEBRA, multi-session alignment (free_saccade) | 0-1 | 0.72-0.75 / ~0.91 (500 Hz) | does not beat HMM except dataset 2 |
| foundation model (`foundation/`), zero-shot, never saw d1 | 0 | 0.733 / 0.869 | promising, below the bar |
| foundation + calibration, full fine-tune | 100 trials | 0.836 / 0.868 | below the bar |
| foundation + calibration, full fine-tune | 1000 trials | 0.842 / 0.878 | **still below U'n'Eye** |
| foundation + calibration, head only (prototypes + last layer) | 1-1000 trials | 0.76-0.81 / 0.81-0.84 | not enough alone |

Zero-shot foundation on dataset 2 (pursuit, never seen): 0.767 / 0.864 (better than every label-free method). The full
leave-one-dataset-out run was stopped after d1 and d2 (owner: not worth it until the F1 bar is met).
**Likely reason the foundation model stays below U'n'Eye: it is causal with only 20 ms lookahead, U'n'Eye sees the whole trial.**
First test to do: the same training with a NON-causal encoder (see "Architecture decision" below).

Key insights so far:
- Inputs must be velocities only (as U'n'Eye; the start position of a saccade is a confound). Checked: a constant offset changes
  nothing (`free_saccade/cebra_seed.py::feats`, float64).
- **Velocity minus its 100 ms running median** ("detrended") is what makes microsaccades during pursuit findable, including those
  AGAINST the pursuit whose raw speed drops: on dataset 2, 310 such saccades, raw Engbert-Kliegl finds 22 %, detrended 100 %.
- In a purely self-supervised embedding (CEBRA-Time, no hint) saccades do NOT form their own cluster (k-means / HDBSCAN fail),
  but a linear probe separates them with AUC 0.989 (input features: 0.904): the information is there, a direction is missing.
- One human saccade alone never orients the embedding; with label-free seeds it helps on 3 of 4 datasets.
- Classes that only one dataset has (PSO, pursuit, blink) cannot be learnt zero-shot when that dataset is held out (Andersson
  held out: 5-class kappa 0.14): new classes must come from the user's clicks (prototype head).

## Architecture decision (owner, 2026-10-09): an ensemble of predictors, the foundation encoder need NOT be causal
- The self-supervised / foundation encoder can be a **non-causal U-Net-shaped network** (past AND future context around each
  sample): it is trained once offline, and precise detection of saccade onset / offset needs the future. Causality only matters
  for the real-time path.
- The end product is a **"Swiss army knife" of complementary predictors** that share the data stream, each doing what it is best at:
  1. a **causal** network (the existing `online/causal_net.py` TCN, C++ `StepEngine`): immediate label, no lookahead;
  2. a **non-causal U-Net / foundation encoder**: precise onset / offset once the future is available (the offline GUI, or the
     delayed stage of the real-time engine);
  3. a **physics-based** predictor (velocity threshold, main sequence, the landing tables of `cpp/src/gaze_engine.cpp`): robust,
     explainable, and the fallback the safety guard uses;
  4. **forecasting** of the eye position in the next bins (already explored: Laplace heads at +5/+10/+20 ms and landing-point
     prediction of ongoing saccades, `online/forecast.py`, used in `cpp/src/gaze_engine.cpp`).
  The same representation should serve forecasting AND precise detection (past + future window around a saccade).
- This is already partly built: `GazeEngine` fuses a causal TCN (age 0), a 10 ms-lookahead TCN, the original U'n'Eye window
  network (finalises bins at 80 ms), a physics layer with veto / override, and a forecaster, behind the safety guard
  (`cpp/GAZE_ENGINE.md`). What is missing is replacing / complementing the U'n'Eye stage by the **dataset-independent foundation
  encoder with a prototype head**, and the user-click calibration loop.
- Consequence for the experiments: (a) rerun `foundation/train_lodo.py` + `foundation/calibrate.py` with a non-causal encoder
  (set `LOOKAHEAD` to the full receptive field, or use a U-Net like `uneye/functions.py::UNet`); the bar is event F1 >= U'n'Eye
  (0.899 on dataset 1); (b) the physiological-prior hyperplane idea below uses the same non-causal encoder; (c) compare each
  predictor alone and fused, offline (precision of onset / offset in ms) and in the stream (latency).

## NEW IDEA to test next (owner, 2026-10-09): self-supervised embedding + a hyperplane chosen by physiological priors
No labels and no detector seeds on the new dataset; only general knowledge of eye movements.
1. **Encoder**: one shared CEBRA-Time-style encoder (positives = time neighbours, InfoNCE on the unit sphere, velocity +
   detrended-velocity inputs), trained once on many datasets: 1-4, Andersson and `archive/` (hundreds of hours, no labels).
2. **Blinks first, by rule**: long (about 100 ms or more), magnitude far larger than saccades, often lost signal -> removed before
   the hyperplane.
3. **Per new dataset, search the hyperplane (w, b) of the embedding** that maximises a label-free score of the segmentation it
   produces: saccade durations mostly 5-50 ms; inter-saccade intervals mostly 100-300 ms (almost never < 50 ms); ballistic main
   sequence (rank correlation amplitude-peak velocity and amplitude-duration high). Ignore PSOs. Start from candidate directions
   (detrended speed, principal axes of the embedding) and refine with a simple gradient-free search (random search, CMA-ES).
4. **The decisive test**: over many candidate hyperplanes, does the prior score rank them like the human kappa / event F1
   (Spearman per dataset)? Earlier, a similar label-free score (`free_saccade/intrinsic.py`, ILS) was a WEAK ranker (it rewards
   detectors that keep only large clean saccades and is gameable), so this must be measured, not assumed. What differs here: one
   embedding, a search over directions, and the interval prior punishes split saccades.
5. Compare with: HMM (label-free bar), foundation zero-shot, foundation + few clicks, U'n'Eye (supervised bar).

## Agreed plan after that (owner approved)
1. Foundation encoder (Python, M1 GPU), leave-one-dataset-out, vs U'n'Eye. **Status: started, below the bar; next = lookahead /
   bidirectional variant, then the prior-hyperplane idea above.**
2. Export to C++: encoder as `.bin` (extend the `UNCZ` causal-TCN format to output the embedding) + ONNX; prototype head with
   online update in C++ (cosine to class prototypes; clicks move prototypes; new class = new prototype); equivalence test vs
   Python as in `cpp/tests/test_equivalence.cpp`.
3. Streaming simulation with a **simulated user** (clicks taken from the human labels): curve "agreement vs number of clicks" on a
   held-out dataset. This measures the whole loop without a human.
4. GUI (Dear ImGui proposed): masked prediction, user clicks onset / offset / class, active choice of the next segment, stop when
   N segments agree. The repository already has two labeling GUIs (`saccade_labeler.py`, web `app.js`).

## How to run (M1: no Xcode tools; Python in `.venv` via uv, C++ via `.venv/zigbin` clang)
```
python foundation/train_lodo.py --targets d1,d2        # leave-one-dataset-out (about 6 min per training on the M1)
python foundation/calibrate.py --target d1             # calibration curve vs U'n'Eye (needs foundation/runs/lodo_d1.pt)
python free_saccade/benchmark_cebra.py                 # label-free baselines incl. HMM
```
`foundation/runs/*.pt` are local only (not in git); retrain with `train_lodo.py`.
