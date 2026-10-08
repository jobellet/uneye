"""Builds UnEye_online_architectures.ipynb, version 2 (run: python3 make_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip("\n")))
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

M("""
# Online saccade / microsaccade detection: which causal network, trained how?  (v2: finalists only)

**Goal.** Not a benchmark of every architecture: pick the design to use in the project.
Version 1 (360 runs) already settled several things:

| finding in v1 | consequence here |
|---|---|
| Causal **TCN trained from scratch** is the best model from ~50 labeled trials on (kappa 0.70 at 50, 0.74 at 100, 0.75 at 300) | main candidate |
| **TCN-lite** is 0.04-0.06 below, but 2.6x smaller; **GRU** is stable but plateaus at 0.66-0.68; **S4D** was unstable (0.46-0.59) | keep TCN-lite as the "cheap" candidate, GRU as a one-point reference, drop S4D |
| **JEPA** and the raw-signal predictor gave no gain from 20 labels on, and got *worse* with more labels; frozen-encoder probes plateaued at 0.52-0.58 | dropped |
| Weak labels from the classic Engbert-Kliegl detector (free) were as good as JEPA at small N | kept (`weak_ft`) |
| The online-vs-offline gap (kappa 0.77 vs 0.85) is probably mostly the missing future | new: **allow a few ms of delay** (`lookahead`) |
| The original U'n'Eye was only compared when trained on *all* labels | new: the original U'n'Eye trained on the **same N labels**, offline and online |

**What is compared** (everything is evaluated on set B of datasets 1-3, never used for any choice; N labeled trials come from set A, 20 % of them used for early stopping):

* `scratch`: causal network, random start.
* `weak_ft`: first trained on free Engbert-Kliegl pseudo-labels of all set-A recordings, then fine-tuned on the N labels.
* `distill` (new): the original non-causal U'n'Eye is trained on the N labels, labels all set-A recordings, and the causal network learns from those labels plus the true ones. Tests whether unlabeled data helps through a teacher that sees the future.
* **lookahead L** (new): the network output at time t is the label of sample t-L. Still real time, but L ms late, and the network gets L ms of future context.

Set `QUICK = True` for a smoke test (a few minutes). `False` runs about 170 student runs + 20 U'n'Eye teachers.
""")

C("""
import os, sys, subprocess
QUICK = True            # <- set to False for the real experiment
REPO, BRANCH = "https://github.com/jobellet/uneye", "claude/pipeline-cpp-onyx-isignal-krrw75"   # use "master" once merged
if not os.path.exists("../data/dataset1"):          # running on Colab / Kaggle: get the code and the data
    if not os.path.exists("uneye"):
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO], check=True)
    else:                                            # already cloned in this session: get the latest code, keep results_grid.json
        subprocess.run(["git", "-C", "uneye", "pull", "-q", "origin", BRANCH], check=False)
    os.chdir("uneye/online")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "scikit-image", "scipy", "pandas", "matplotlib"], check=True)
sys.path.insert(0, os.getcwd())
import time, json, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import archs, experiment as E, metrics as Mx, viz
viz.style()
print("device:", E.DEVICE, "| GPUs:", torch.cuda.device_count(), "| CPU cores:", os.cpu_count(), "| torch", torch.__version__)
if QUICK: print("*** QUICK MODE: smoke test only, the numbers below are meaningless. Set QUICK = False for the real run. ***")
""")

M("## 1. Data and the three candidate networks")
C("""
data = E.Data(n_test=40 if QUICK else 300)
for s in data.sets:
    V, L, fs = data.tr[s]; print(f"set {s}: {V.shape[0]} train recordings, {fs} Hz, saccade fraction {L.mean():.3f};  test: {data.te[s][0].shape[0]} recordings")
rows = []
for name in ("tcn", "tcn_lite", "gru"):
    net = archs.build(name)
    rows.append(dict(model=name, params=E.count_params(net), receptive_field=net.backbone.receptive_field or "unlimited (state)",
                     ms_per_200ms_window_cpu_eager=round(E.latency_ms(net), 2)))
pd.DataFrame(rows)
""")
M("For nets of this kind the multiply-adds per new sample are roughly the number of parameters (the C++ streaming engine needs about 53 us per sample for the 43 k-parameter TCN). Eager PyTorch timings are dominated by overhead and do not show the difference; the final section exports the chosen TCN to the C++ engine.")

M("## 2. Experiment plan")
C("""
cfg = E.QUICK if QUICK else E.FULL
SEEDS = (0,) if QUICK else (0, 1, 2, 3, 4)                 # the seeds share the same labeled subset for every method: paired comparison
N_MAIN = (10,) if QUICK else (10, 30, 100, 300)             # labeled trials (about 1 s each)
LOOKAHEADS = (0, 10) if QUICK else (0, 5, 10, 20, 40)      # ms of allowed delay
N_LOOK = (10,) if QUICK else (30, 100)

plan  = E.make_plan(("tcn",) if QUICK else ("tcn", "tcn_lite"), E.STUDENTS, N_MAIN, SEEDS)                 # main comparison (L = 0)
plan += [] if QUICK else E.make_plan(("gru",), ("scratch",), (30, 100), SEEDS)                              # one-point reference
plan += E.make_plan(("tcn",), ("scratch",) if not QUICK else ("scratch",), N_LOOK, SEEDS, [l for l in LOOKAHEADS if l > 0])   # delay budget
print(len(plan), "student runs +", len(N_MAIN) * len(SEEDS), "U'n'Eye teachers (also the 'original U'n'Eye trained on N labels' baselines)")
""")
C("""
WORKERS_PER_GPU = 2     # the models are tiny: one process cannot fill a GPU (it is launch-bound). All GPUs are used (Kaggle: 2 x T4).
                        # Set to 1 to run everything in this process, one job at a time.
DEVICES = E.default_devices(WORKERS_PER_GPU) if WORKERS_PER_GPU > 1 else None
t0 = time.time()
rows = E.run_plan(data, plan, cfg=cfg, save="results_v2.json", devices=DEVICES, resume=True)   # finished runs are kept: you can interrupt and restart
df = pd.DataFrame(rows); df.to_csv("results_v2.csv", index=False)
print(f"total {(time.time()-t0)/60:.1f} min, {len(df)} result rows")
""")

M("""
## 3. Results
### 3.1 How many labels are needed?  (cf. Fig 7A of the paper)
Median over seeds, band = inter-quartile range, dots = single seeds. Dashed: the **original** U'n'Eye trained on the same N labels.
""")
C("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
cands = [("tcn", "scratch"), ("tcn", "weak_ft"), ("tcn", "distill")]
viz.learning_curves(df, "kappa@0.5", cands, ax=axes[0]); axes[0].set_title("TCN: agreement with the human labels")
viz.learning_curves(df, "ev_f1@0.5", cands, ax=axes[1]); axes[1].set_title("TCN: event F1")
viz.learning_curves(df, "kappa@0.5", [("tcn", "scratch"), ("tcn_lite", "scratch"), ("gru", "scratch")], refs=(("unet", "lastbin"),), ax=axes[2]); axes[2].set_title("backbones (from scratch)")
plt.tight_layout(); plt.show()
""")

M("""
### 3.2 Is a strategy really better than scratch?  Paired differences
All methods use the same labeled subset for a given (N, seed), so the difference is computed per seed. Differences are only meaningful if they are consistent across seeds.
""")
C("""
d0 = df[np.isclose(df.lookahead_ms, 0) & (df.backbone != "unet")]
sc = d0[d0.strategy == "scratch"].set_index(["backbone", "n_labels", "seed"])
out = []
for (bb, st, n), g in d0[d0.strategy != "scratch"].groupby(["backbone", "strategy", "n_labels"]):
    g = g.set_index(["backbone", "n_labels", "seed"]); common = g.index.intersection(sc.index)
    for m in ("kappa@0.5", "ev_f1@0.5"):
        diff = (g.loc[common, m] - sc.loc[common, m]).values
        out.append(dict(backbone=bb, strategy=st, N=n, metric=m.split("@")[0], mean_gain_vs_scratch=diff.mean(), sem=diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan,
                        seeds_better=f"{(diff > 0).sum()}/{len(diff)}"))
pd.DataFrame(out).round(3).sort_values(["metric", "backbone", "strategy", "N"])
""")

M("""
### 3.3 Candidates side by side (cf. Fig 4)
Choose N (what you can afford to label). One box = the seeds. Red line = original U'n'Eye **offline** trained on the same N labels (it sees the future, so it is the target to approach, not a fair opponent).
""")
C("""
N_SHOW = 10 if QUICK else 100
fig = viz.boxes(df, N_SHOW, [("unet", "lastbin"), ("tcn", "scratch"), ("tcn", "weak_ft"), ("tcn", "distill"), ("tcn_lite", "scratch")] if not QUICK else
                [("unet", "lastbin"), ("tcn", "scratch"), ("tcn", "weak_ft"), ("tcn", "distill")])
plt.show()
cols = ["kappa@0.5", "mcc@0.5", "ev_f1@0.5", "onset_err_ms@0.5", "offset_err_ms@0.5", "alarm_delay_ms@0.5", "false_alarms_per_min@0.5"]
df[np.isclose(df.lookahead_ms, 0)].groupby(["backbone", "strategy", "n_labels"])[cols].median().round(3)
""")

M("""
### 3.4 Delay budget: what do a few ms of lookahead buy?
The network output at time t is trained to describe sample t-L. `alarm delay` includes L.
""")
C("""
for n in N_LOOK:
    fig = viz.delay_curve(df, "tcn", "scratch", n); plt.show()
""")

M("""
### 3.5 Decision helper
Set your constraints; the table lists the designs that satisfy them, best first. Delay includes the lookahead.
""")
C("""
N_DECIDE = N_LOOK[-1]
MAX_ALARM_DELAY_MS = 20      # time from the true onset until the detector can raise an alarm (median)
MAX_FALSE_ALARMS_PER_MIN = 60
c = df[(df.n_labels == N_DECIDE) & (df.backbone != "unet")].groupby(["backbone", "strategy", "lookahead_ms"]).agg(
    kappa=("kappa@0.5", "median"), kappa_sd=("kappa@0.5", "std"), ev_f1=("ev_f1@0.5", "median"), delay_ms=("alarm_delay_ms@0.5", "median"),
    false_alarms_per_min=("false_alarms_per_min@0.5", "median"), onset_err_ms=("onset_err_ms@0.5", "median"), params=("params", "first"), runs=("seed", "count")).round(3)
c["meets_constraints"] = (c.delay_ms <= MAX_ALARM_DELAY_MS) & (c.false_alarms_per_min <= MAX_FALSE_ALARMS_PER_MIN)
print(f"N = {N_DECIDE} labeled trials, delay <= {MAX_ALARM_DELAY_MS} ms, false alarms <= {MAX_FALSE_ALARMS_PER_MIN}/min")
c.sort_values(["meets_constraints", "kappa"], ascending=False)
""")

M("""
## 4. Final model for the project
Pick the design from the tables above, train it on **all** human labels, look at it (traces, errors) and export it.
""")
C("""
BACKBONE, STRATEGY, LOOKAHEAD_MS = "tcn", "scratch", 10      # <- from section 3 (the C++ engine supports the full 'tcn')
t0 = time.time()
final, res = E.train_final(data, BACKBONE, STRATEGY, LOOKAHEAD_MS, n="all", cfg=cfg)
print(f"trained in {time.time()-t0:.0f}s")
pd.DataFrame(res["0.5"]["per_set"]).T[["kappa", "mcc", "f1", "ev_recall", "ev_precision", "ev_f1", "onset_err_ms", "offset_err_ms", "alarm_delay_ms", "false_alarms_per_min"]].round(3)
""")
C("""
# example traces (cf. Fig 3): human labels vs the final model vs the original U'n'Eye evaluated online (last 200 ms bin)
orig = E.UNetTeacher().to(E.DEVICE); orig.net.load_state_dict(torch.load("../training/weights_1+2+3", weights_only=False, map_location=E.DEVICE))
preds = {"final model": {s: viz.align_pred(E.predict(final, data.te[s][0]), E.shift_samples(LOOKAHEAD_MS, data.te[s][2])) for s in data.sets},
         "original U'n'Eye, online": {s: E.predict_lastbin(orig, data.te[s][0], data.te[s][2]) for s in data.sets}}
fig = viz.trace_gallery(data, preds); plt.show()
""")
C("""
# where are the errors?  (cf. Fig 5)
fig = viz.main_sequence(data, preds["final model"]); plt.show()
""")
C("""
# export for the C++ pipeline (full TCN only) and print the command to run it
if BACKBONE == "tcn":
    import export_causal as X
    k = E.shift_samples(LOOKAHEAD_MS, 1000.0)
    X.export_saccadenet(final.cpu().eval(), "final_causal", lookahead_samples=k)
    torch.save(final.state_dict(), "final_model.pt")
    print(f"\\nwritten: final_causal.bin / .onnx / .json, final_model.pt\\nrun live:  uneye_rt --model final_causal.bin --stdin --label-delay {k}")
else:
    print("the C++ engine currently supports the full TCN; use the ONNX export for other backbones")
""")

M("""
## Appendix: which metric?
For online use every time bin is one decision. Saccades are rare (4-9 % of samples), so:
* **Cohen's kappa** is the chance-corrected agreement with the human labeler (the number used in the U'n'Eye paper). **MCC** uses all four cells of the confusion matrix and is recommended for imbalanced classes; the two are close here.
* **F1** ignores true negatives. Note that the paper's F1 (0.96) is an **event-level** F1 (was each saccade found?) - the comparable number here is `ev_f1`, not the sample-level `f1`.
* None of them measures **timing**: use `onset_err_ms`, `offset_err_ms` (|difference| to the human labels, as in the paper), `alarm_delay_ms` and `false_alarms_per_min`.
""")
C("""
rng = np.random.RandomState(0)
def demo(prev, rec, fpr, n=200000):
    t = rng.rand(n) < prev
    p = np.where(t, rng.rand(n) < rec, rng.rand(n) < fpr)
    return Mx.sample_metrics(*Mx.confusion(p, t))
print(f"{'prevalence':>10} {'recall':>7} {'FP rate':>8} | {'F1':>6} {'kappa':>6} {'MCC':>6}")
for prev, rec, fpr in [(0.08, .8, .01), (0.08, .8, .03), (0.04, .8, .01), (0.5, .8, .01), (0.08, .98, .20)]:
    m = demo(prev, rec, fpr); print(f"{prev:10.2f} {rec:7.2f} {fpr:8.2f} | {m['f1']:6.3f} {m['kappa']:6.3f} {m['mcc']:6.3f}")
""")

nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "UnEye_online_architectures.ipynb")
print("written", len(nb.cells), "cells")
