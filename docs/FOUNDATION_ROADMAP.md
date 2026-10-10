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

## BEST SO FAR (strict protocol requested by the owner 2026-10-09): detector fitted ONLY on archive/, tested on U'n'Eye's benchmarks, F1 AND kappa
Goal restated by the owner: equal or beat U'n'Eye in event F1 AND Cohen's kappa; ideally a model trained only on the extra datasets and tested on the
same benchmarks. `foundation/universal_hmm.py` (reads only archive/ for fitting; benchmarks = test only, no fit, no label, not even their unlabeled
train split): the minimum-duration HMM of `free_saccade/detectors.py` (2 Gaussian states, saccade = chain of 6 states) on the unit-free score
log10(1 + detrended speed / robust noise of the window), one HMM per sampling rate (1 kHz and 500 Hz, each benchmark at its native rate),
Viterbi-trained on 2000 archive windows (EMTeC, GazeBase, GazeCom, Lund2013). The "boundary shift" widens / narrows every detected event by `pre`
samples at the onset and `post` at the offset (the expert's convention): 2 integers chosen on N human-labeled TRIALS of the target's train split by
maximising kappa on them (mean of 3 draws of the trials; test split B). Same scorer for all (foundation/compare.py), 1 kHz.
| dataset | universal HMM, 0 label (F1 / kappa) | + boundary shift from 3 trials | + from 10 trials | + from 30 trials | U'n'Eye supervised (F1 / kappa) |
|---|---|---|---|---|---|
| d1 | 0.912 / 0.717 | 0.931 / 0.815 | 0.931 / 0.815 | 0.929 / 0.817 | 0.899 / 0.856 |
| d2 (pursuit) | 0.908 / 0.752 | 0.909 / 0.780 | 0.910 / 0.832 | 0.911 / 0.831 | 0.941 / 0.874 |
| d3 | 0.854 / 0.671 | 0.854 / 0.703 | 0.854 / 0.720 | 0.854 / 0.731 | 0.929 / 0.815 |
| d4 | 0.932 / 0.813 | 0.941 / 0.833 | 0.939 / 0.843 | 0.939 / 0.844 | 0.924 / 0.850 |
| andersson (saccade vs rest) | 0.795 / 0.335 | 0.807 / 0.342 | 0.805 / 0.351 | 0.805 / 0.351 | 0.885 / 0.813 (`weights_Andersson`, 5 classes) |
- F1: above U'n'Eye on d1 and d4, 0.03 below on d2, 0.07 below on d3; kappa: 0.02-0.14 below, except d4 (0.843 vs 0.850, equal within noise).
- Most of the kappa gain comes from the boundary convention (2 integers, 3-10 labeled trials suffice; ONE trial is unreliable: it made d3 collapse in a
  first run). The ceiling of a pure boundary shift (chosen on the test split) is 0.82 / 0.83 / 0.74 / 0.86 (d1-d4): the rest of the kappa gap to U'n'Eye is
  not a constant boundary offset.
- U'n'Eye was trained in-domain (d1-d3); the universal HMM never saw them. No confidence intervals yet (d3 test = 53 trials).
- Andersson: kappa 0.34 whatever the shift: not understood yet (blinks, pursuit segments, PSO labeled as a separate class: to look at first).
- The HMM decoding is offline (Viterbi on a whole trial); for the C++ stream use a fixed-lag decoder.
Verdict so far: the label-free universal HMM + 2-integer boundary calibration is the strongest approach tested and needs no neural network. The learned
encoders did not add anything yet. Ideas: per-event boundary rule (fraction of the peak speed) learned from a few clicks instead of a constant shift; an
emission model better than one Gaussian per state; per-state duration laws (semi-Markov); the 5-class case (blink, pursuit, PSO) for Andersson.

### The three improvements of the universal HMM (`foundation/hmm_improvements.py`), tested because the owner suspects a local maximum: they confirm it
(The owner also notes that an HMM, Sheynikhovich et al., was already compared in the 2019 article: good F1 on some datasets, that is all.)
| change | effect on kappa (universal HMM alone: d1 0.717, d2 0.752, d3 0.671, d4 0.813) | effect on F1 |
|---|---|---|
| boundaries at a fraction of each event's peak speed (2 fractions chosen on 10 labeled trials) | WORSE than the constant shift: 0.740 / 0.676 / 0.608 / 0.772 (constant shift: 0.815 / 0.832 / 0.720 / 0.843) | d1 0.935, d4 0.948 (+0.02) |
| explicit duration law for the saccade state (semi-Markov; pmf from the events found in archive/ or lognormal prior, no label) | none: +-0.005 | +-0.01 |
Andersson diagnosis: 97 % of the human saccade samples are found, but 41 % of the samples called saccade are labeled fixation (20 % of all fixation samples),
12 % PSO, 16 % pursuit, 4 % blink. Ignoring PSO and blinks in the scoring: kappa 0.34 -> 0.40 only. The annotators of this benchmark labeled almost no
small saccades (5th percentile of the human amplitudes 0.56 deg; detected events with no human counterpart: median 0.36 deg): keeping only events >= 0.5 deg
raises F1 0.795 -> 0.854, kappa 0.335 -> 0.364. So part of the F1 gap is a convention on WHICH events count (an amplitude floor), not a boundary one.

### DINO for eye traces (`foundation/dino1d.py`, owner's request: "the attention map of DINO, but for eye traces")
Self-distillation (student / EMA teacher, 2 global crops of 336 samples + 6 local crops of 84, centring + sharpening, 512 prototypes) of a small transformer
on patches of 4 samples of the unit-free velocity features, trained ONLY on archive/; question: does the attention of the [CLS] token to the patches
highlight the saccades by itself? Scored on the benchmarks by the AUC of the attention map against the human labels (no label in training); results below
when finished.

## Results of the first comparison of predictors alone, event F1 only (owner's rule), test split at 1 kHz
Scorer `foundation/compare.py` (same for everyone: a predicted run of >= 3 samples overlapping a human saccade = hit; every predictor runs at the
native rate of the dataset). Supervised columns were trained on set A of datasets 1, 2, 3 (so in-domain for d1-d3, unseen for d4, andersson).
| dataset | Engbert-Kliegl | EK on detrended velocity | HMM (0 label) | causal TCN | causal TCN +10 ms | U'n'Eye (offline) |
|---|---|---|---|---|---|---|
| d1 | 0.705 | 0.697 | **0.887** | 0.877 | 0.860 | **0.899** |
| d2 (pursuit) | 0.608 | 0.901 | 0.831 | 0.918 | 0.903 | **0.941** |
| d3 | 0.806 | 0.793 | 0.900 | 0.903 | **0.941** | 0.929 |
| d4 | 0.763 | 0.760 | 0.689 | 0.878 | 0.899 | **0.924** |
| andersson | 0.614 | 0.568 | 0.821 | 0.339 | 0.340 | 0.546 (0.885 with `weights_Andersson`, 5 classes) |
- The label-free HMM almost equals U'n'Eye on d1 and d3; it fails on d4. The supervised causal TCN with 10 ms lookahead equals U'n'Eye on d3.
- Dataset 4, set B: the label file has 3300 rows but only 3000 trials of positions; the FIRST 3000 label rows are the right ones
  (Engbert-Kliegl event F1 0.76 vs 0.05 = chance with the last 3000). `foundation/data.py` handles it.

### Self-supervised non-causal encoder + one hyperplane chosen by physiological priors (the owner's idea): first implementation does NOT work
Code: `foundation/ssl_encoder.py` (CEBRA-Time loss, own ~15-line InfoNCE, dilated symmetric conv net, 121 ms receptive field, trained on
datasets 1-4, Andersson and 4 sources of archive/ with NO label, 8 min on the M1), `foundation/prior_hyperplane.py` (prior score and search),
`foundation/diagnose_embedding.py`. Prior score fixed beforehand (docstring of prior_hyperplane.py).
| dataset | event F1 of the prior-chosen hyperplane (test) | linear readout fitted on the labels (test) | random hyperplanes (median) | Spearman(prior score, human F1) |
|---|---|---|---|---|
| d1 | 0.54 | 0.24 | 0.10 | -0.03 |
| d2 | 0.40 | 0.29 | 0.11 | **+0.60** |
| d3 | 0.45 | 0.53 | 0.05 | +0.18 |
| d4 | 0.01 | 0.19 | 0.05 | -0.15 |
| andersson | 0.37 | 0.38 | 0.07 | -0.12 |
Diagnosis (`diagnose_embedding.py`, single threshold chosen WITH the set-B labels, so optimistic bounds), sample AUC / event F1:
embedding 0.90/0.61 (d1), 0.89/0.46 (d2), 0.98/0.85 (d3), 0.94/0.58 (d4), 0.92/0.53 (andersson); plain input features 0.88/0.81, 0.89/0.90,
0.82/0.73, 0.92/0.82, 0.94/0.68. => the embedding separates saccades at the SAMPLE level as well as the speed feature (better on d3, the
microsaccades) but gives fragmented EVENTS; a threshold on the normalised (detrended) speed alone beats it at the event level on 4 of 5 datasets.
Retrained with a hard negative from the same recording (>= 15 samples away; `--hard`, closer to CEBRA's sampling): no improvement
(event F1 0.56, 0.50, 0.81, 0.54, 0.54): this was NOT the main cause.
Ideas not yet tried: (1) decode with duration priors (semi-Markov / HMM with a minimum-duration chain, as `free_saccade/detectors.py::HMM`
already does on speed and reaches 0.89 / 0.90 on d1 / d3) on a LEARNED score instead of searching a hyperplane; (2) larger time offset for the
positives (+-10-20 samples) so that the embedding is smooth over a saccade; (3) use the priors only to choose a threshold on the normalised
speed (1-D, cheap), then test whether the prior score ranks thresholds like the human F1; (4) the real `cebra` package as a reference.
The owner's decision: keep only what is useful (own loss + small conv encoder + prototype head), no dependency on the `cebra` toolbox.

### The minimum-duration HMM (`free_saccade/detectors.py::HMM`) applied to different 1-D scores (`foundation/hmm_score.py`, `hmm_on_foundation.py`)
HMM = 2 hidden states (fixation / saccade), Gaussian emission on the score, saccade state = a chain of 6 states (a saccade lasts >= 6 samples),
fitted WITHOUT labels on the target's unlabeled train split (Viterbi training), then decoding the test split. Event F1 on the test split:
| dataset | log speed + HMM | log detrended speed + HMM | linear score on 6 input features, fitted on the labels of the OTHER 4 datasets, + HMM | same on the embedding + HMM | embedding, label-free direction + HMM |
|---|---|---|---|---|---|
| d1 | 0.881 | 0.874 | **0.893** | 0.316 | 0.190 |
| d2 | 0.822 | 0.516 | **0.824** | 0.153 | 0.300 |
| d3 | 0.697 | 0.681 | 0.566 | 0.315 | 0.226 |
| d4 | 0.658 | 0.643 | **0.842** | 0.186 | 0.106 |
| andersson | 0.753 | 0.762 | 0.629 | 0.322 | 0.207 |
- The HMM is what removes the fragmentation: the same feature score thresholded at 0.5 gives 0.09-0.25, decoded by the HMM 0.84-0.89 (d1, d2, d4).
  A linear score on 6 input features learned on OTHER datasets + HMM reaches U'n'Eye's level on d1 (0.893 vs 0.899) with a tiny model; d4 0.842 vs 0.924.
- The self-supervised embedding adds NOTHING in this form (0.1-0.3), supervised or not, with or without the HMM.
- HMM on the logit of the supervised foundation network (never saw the target): d1 0.869 -> 0.535 (worse), d2 0.864 -> 0.905 (better). A Gaussian
  emission fitted on a saturated logit is a poor model. Next: a hybrid NN-HMM, i.e. use the network's posterior as the emission likelihood
  (scaled likelihood p(class | x) / p(class)) instead of fitting Gaussians, keep the minimum-duration chain (and give each state a duration law).

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

## Overnight 2026-10-09/10 (details: docs/OVERNIGHT_REPORT.md)
17 steps, 0 failures. Mean event F1 / kappa over the 5 benchmarks (same test subsets):
- Supervised BiTCN (d1+d2+d3 labels, EMA+TTA, `sup_bitcn_b_tta`): 0.89 / 0.81, above U'n'Eye 0.85 / 0.75; weak point Andersson (0.75 / 0.59).
- Label-free noisy student of the universal HMM (`selftrain_b_r1_tta`): 0.83 / 0.67 (kappa above U'n'Eye mean, F1 slightly below); universal HMM alone 0.88 / 0.66.
- Self-supervised representations (JEPA a/b/d/e, MAE a/b, HuBERT, TS2Vec a/b) + probe + HMM: 0.46-0.68 F1, none beats plain input channels (0.75 / 0.51); several equal or below the untrained control (mae_b equals the control: best checkpoint was step 0). Only ts2vec_a is notable with 20 labeled trials on d1/d2 (0.84 / 0.71). Novelty-curve detection from TS2Vec is poor (F1 <= 0.33).
- Conclusion: for the C++ tool, ship the bidirectional TCN (supervised or HMM-distilled), not a SSL foundation encoder; SSL representation learning adds nothing measurable here.

## Few-label regime (article analysis, dataset 1: N labeled trials of set B -> test on 300 trials of set A) — benchmark NOT beaten
Mean of 3 draws, F1 / kappa. U'n'Eye retrained with `uneye.DNN` (its own early stopping), BiTCN 2 channels with a fixed 1500 steps and no validation:
| N | U'n'Eye | BiTCN (vx, vy) |
|---|---|---|
| 10 | 0.89 / 0.76 | 0.77 / 0.79 |
| 20 | 0.92 / 0.81 | 0.82 / 0.83 |
| 50 | 0.95 / 0.84 | 0.84 / 0.85 |
| 100 | 0.95 / 0.86 | 0.86 / 0.86 |
| 200 | 0.95 / 0.86 | 0.91 / 0.87 |
| 300 | 0.96 / 0.87 | 0.90 / 0.87 |
With few labels U'n'Eye has the better event F1 (kappa is equal); the BiTCN only matches it from ~200 trials on kappa and does not reach its F1 at N<=300. The BiTCN's advantage (0.90 / 0.81 vs 0.85 / 0.75) is in the zero-shot / pooled-training setting, not in the few-label regime.
Decision (owner): no further runs on this benchmark (U'n'Eye was not retrained for N=1000 nor for the between-subject analysis on dataset 4; that figure shows the BiTCN only). Next: try another approach for the few-label regime rather than tuning this one.

## Zero-shot (leave-one-dataset-out) — benchmark NOT beaten (2026-10-10, `foundation/lodo_bitcn.py`, results `foundation/runs/night/lodo_bitcn.json`)
2-channel BiTCN trained on the train splits of the OTHER four datasets only (early stopping on 10 % of those), tested on the common test subset of the held-out dataset (threshold, TTA). Reference: U'n'Eye trained in the dataset (d1-d3: weights 1+2+3, d4: unseen by U'n'Eye too, Andersson: own weights). Event F1 / kappa:
| held-out | BiTCN, never saw the dataset | U'n'Eye |
|---|---|---|
| d1 | 0.94 / 0.79 | 0.90 / 0.85 |
| d2 | 0.75 / 0.54 | 0.94 / 0.88 |
| d3 | 0.84 / 0.74 | 0.93 / 0.82 |
| d4 | 0.95 / 0.85 | 0.92 / 0.85 |
| Andersson | 0.81 / 0.58 | 0.89 / 0.81 (general weights: 0.55 / 0.33) |
| mean | 0.86 / 0.70 | 0.92 / 0.84 |
It wins only on d1 (F1) and d4 (F1, kappa equal), loses clearly on d2 (microsaccades during pursuit), d3 and Andersson. A detector with no dataset-specific training does not match U'n'Eye trained on the dataset; ~50 s of labeled data (see few-label section) remain the better deal. Idea dropped unless a new approach for d2/d3 appears (what is missing: pursuit and very small microsaccades are absent from the other datasets' conventions).

## Pre-trained wider U-Net (masked auto-encoder), few labels — marginal (2026-10-10, `foundation/unet_ssl.py`)
U'n'Eye topology x3 channels, input vx, vy + mask channel, pre-trained to reconstruct masked spans of the velocity (30 % of 500-sample windows, spans 8-40) on 22 887 unlabeled windows: archive sources with rate >= 200 Hz (EMTeC, GazeBase, Lund2013, GazeCom; 360EM, VEDB, DUT-OMRON, EGTEA excluded) + unlabeled train splits of the 5 benchmarks; windows with > 1 % missing samples or tracker spikes removed. Fine-tuned (all layers) on N labeled trials of d1 set B (own validation split, early stopping), test 300 trials of set A, mean of 3 draws, event F1 / kappa:
| N | pre-trained U-Net | same U-Net from scratch | U'n'Eye retrained |
|---|---|---|---|
| 10 | 0.911 / 0.799 | 0.897 / 0.786 | 0.894 / 0.756 |
| 20 | 0.922 / 0.829 | 0.910 / 0.815 | 0.921 / 0.810 |
| 50 | 0.925 / 0.842 | 0.930 / 0.823 | 0.945 / 0.844 |
Pre-training beats U'n'Eye at N=10 (+0.017 F1, +0.04 kappa), ties at N=20, and loses in F1 at N=50 (0.925 vs 0.945). Its gain over the same network from scratch is ~0.01 F1 at N=10-20 and zero at N=50; most of the N=10 advantage comes from the wider architecture + our augmentations, not from the pre-training. Three draws only (spread ~0.01-0.02): not a significant win. Pre-training progress figures: `docs/figs_unet/`.

### Update — 10 draws (same trials for the three methods; supersedes the 3-draw table above)
Mean ± std over 10 draws of N labeled trials, event F1 / kappa:
| N | pre-trained U-Net | same U-Net from scratch | U'n'Eye retrained |
|---|---|---|---|
| 10 | 0.846±0.162 / 0.785±0.073 | 0.871±0.049 / 0.774±0.055 | 0.826±0.107 / 0.672±0.135 |
| 20 | 0.898±0.018 / 0.815±0.011 | 0.888±0.017 / 0.810±0.011 | 0.897±0.024 / 0.775±0.048 |
| 50 | 0.921±0.008 / 0.842±0.008 | 0.915±0.017 / 0.823±0.013 | 0.941±0.014 / 0.830±0.022 |
Paired F1 difference pre-trained minus U'n'Eye: N=10 +0.021 (se 0.035, 8/10 wins), N=20 +0.001 (se 0.006), N=50 -0.020 (se 0.005, 2/10 wins). The N=10 "win" of the 3-draw table was luck (one draw is a failure for the pre-trained net, F1 spread 0.16). Verdict: kappa is better than U'n'Eye at every N (+0.11, +0.04, +0.01), event F1 is tied at N<=20 and worse at N=50. Pre-training itself adds ~0.01 over the same network from scratch: the benchmark is NOT beaten in F1. The pre-training idea is parked.
