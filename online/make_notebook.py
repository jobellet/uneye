"""Builds UnEye_online_architectures.ipynb (run: python3 make_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip("\n")))
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

M("""
# Online saccade / microsaccade detection: causal architectures, self-supervised pretraining (JEPA), few labels

**Question.** Can a network that only sees the *past* (needed for online use) reach high agreement with human labels,
and can *self-supervised* pretraining on unlabeled recordings reduce the number of human-labeled trials needed?

**What this notebook does**
1. Compares four causal backbones (TCN, light TCN, S4D state-space, GRU) – size, receptive field, CPU latency.
2. Pretrains them **without labels**: a causal **JEPA** (predict the latent future/masked past of an EMA target encoder),
   a raw-signal predictive baseline, and a classic-detector weak-label baseline (Engbert-Kliegl).
3. Fine-tunes with **N = 5, 20, 100 … labeled trials** and compares with training from scratch and with the original U'n'Eye
   (evaluated on the last sample of a 200 ms time bin).
4. Reports **Cohen's kappa, MCC, F1, PR-AUC** (sample level) and **event F1, alarm delay, false alarms / min** (event level).

Protocol: train = set A of datasets 1-3, test = set B (300 trials per dataset, never used for any choice).
With a budget of N labeled trials, 20 % of them are used for early stopping and threshold tuning.
Set `QUICK = True` for a 2-minute smoke test, `False` for the real run (about 1-3 h on a Colab GPU depending on the grid).
""")

C("""
import os, sys, subprocess
QUICK = True            # <- set to False for the real experiment
REPO, BRANCH = "https://github.com/jobellet/uneye", "claude/pipeline-cpp-onyx-isignal-krrw75"   # after the PR is merged use "master"
if not os.path.exists("../data/dataset1"):          # running on Colab / Kaggle: get the code and the data
    if not os.path.exists("uneye"):
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO], check=True)
    os.chdir("uneye/online")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "scikit-image", "scipy", "pandas", "matplotlib"], check=True)
sys.path.insert(0, os.getcwd())
import time, json, copy, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import archs, jepa, experiment as E, metrics as Mx
print("device:", E.DEVICE, "| torch", torch.__version__)
""")

M("""
## 1. Which metric? F1 vs Cohen's kappa vs MCC

For online use every time bin is one decision. Saccades are rare (4-9 % of samples here), so:

* **F1** ignores true negatives and is not chance-corrected; it depends on which class you call "positive".
* **Cohen's kappa** corrects for the agreement expected by chance from the class frequencies. This is the natural "agreement with the human labeler" number, and the one used in the U'n'Eye paper.
  With rare positives kappa and F1 are close (see below) – so kappa is *more principled* but rarely changes the ranking of models.
* **MCC** (Matthews correlation) uses all four cells of the confusion matrix, is symmetric in the two classes and is recommended in the literature on imbalanced classification (Chicco & Jurman 2020; Chicco, Warrens & Jurman 2021); it is close to kappa but behaves better when the predicted and true class frequencies differ. 
* None of them measures **timing**. A prediction that is off by 2 samples at every boundary has perfect events but a lower sample-level score, while a detector that is late by 10 ms can have a good kappa but be useless for gaze-contingent control.

**Recommendation used here:** headline = kappa (agreement) and MCC, always with the *event-level* F1, **alarm delay** (ms from true onset to detection) and **false alarms per minute**. PR-AUC gives a threshold-free check.
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
# boundary jitter: perfect events, +-2 samples at both edges
T = np.zeros((1, 1000)); T[0, 200:230] = 1; T[0, 600:640] = 1
P = np.zeros_like(T); P[0, 202:232] = 1; P[0, 598:638] = 1
print("\\nshifted by 2 samples -> sample-level kappa %.3f, but event recall/precision = %.2f/%.2f" % (
    Mx.sample_metrics(*Mx.confusion(P > .5, T > 0))["kappa"], *[Mx.evaluate_probs(P, T, 1000, with_ap=False)[k] for k in ("ev_recall", "ev_precision")]))
""")

M("## 2. Architectures (all strictly causal)")
C("""
rows = []
for name in archs.BACKBONES:
    net = archs.build(name)
    v = torch.randn(2, 2, 300) * 0.05; v2 = v.clone(); v2[:, :, 200:] += 1; net.eval()
    leak = (net(v)[:, :, :200] - net(v2)[:, :, :200]).abs().max().item()
    rows.append(dict(model=name, params=E.count_params(net), receptive_field=net.backbone.receptive_field or "unlimited (state)",
                     ms_per_200ms_window_cpu=round(E.latency_ms(net), 2), future_leak=f"{leak:.0e}"))
pd.DataFrame(rows)
""")
M("""
`future_leak` must be ~0 (changing the future does not change past outputs). Latency is PyTorch eager mode, one CPU thread, one 200 ms window; real deployment uses ONNX Runtime / the streaming C++ engine (`cpp/`) which is faster. The S4D and GRU models can also run as a **recurrent filter** (constant cost per sample, `S4DLayer.step`).
""")

M("""
## 3. Data
Unlabeled pretraining uses **all set-A recordings without their labels**. Set `EXTRA_UNLABELED = True` to add dataset 4 (3300 more recordings, no labels used).
""")
C("""
EXTRA_UNLABELED = False
data = E.Data(extra_unlabeled=EXTRA_UNLABELED)
for s in data.sets:
    V, L, fs = data.tr[s]; print(f"set {s}: {V.shape[0]} train recordings, {fs} Hz, saccade fraction {L.mean():.3f};  test: {data.te[s][0].shape[0]} recordings")
""")

M("""
## 4. Self-supervised pretraining (JEPA) – look at it before using it

The context encoder sees a *corrupted* velocity signal (masked blocks, noise, random rotation), the EMA target encoder the clean one;
predictors forecast the target **embedding** at t+0, +5, +10, +20 samples. Check that it did not collapse: the embedding spread (`emb_std`) must stay clearly above 0 and the loss must go down.
""")
C("""
t0 = time.time()
cfg = E.QUICK if QUICK else E.FULL
net = archs.build("tcn")
_, hist = E.pretrain_jepa(net.backbone, data, cfg["pre_epochs"], cfg["pre_steps"])
h = pd.DataFrame(hist)
fig, ax = plt.subplots(1, 2, figsize=(9, 3)); h["pred"].plot(ax=ax[0], title="JEPA prediction loss"); h["emb_std"].plot(ax=ax[1], title="embedding std (collapse check)")
for a in ax: a.set_xlabel("epoch"); a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show(); print(f"{time.time()-t0:.0f}s")
""")

M("""
## 5. Label-efficiency experiment
Strategies (all end with the same fine-tuning on N human-labeled trials):

| name | pretraining without human labels | fine-tuning |
|---|---|---|
| `scratch` | none | all weights |
| `jepa_ft` | causal JEPA (latent prediction) | all weights |
| `jepa_probe` | causal JEPA | **only the 1x1 head** (frozen encoder: "linear probe") |
| `recon_ft` | predict raw future velocity (ablation: is the latent target needed?) | all weights |
| `weak_ft` | supervised on Engbert-Kliegl pseudo-labels (free, noisy) | all weights |

Edit the config, run, and read the table. One seed is fast but noisy: use `SEEDS = (0, 1, 2)` for conclusions.
""")
C("""
BACKBONES  = ("tcn", "tcn_lite", "s4d", "gru") if not QUICK else ("tcn",)
STRATEGIES = ("scratch", "jepa_ft", "jepa_probe", "recon_ft", "weak_ft") if not QUICK else ("scratch", "jepa_ft")
N_LABELS   = (5, 10, 20, 50, 100, 300) if not QUICK else (5,)
SEEDS      = (0, 1, 2) if not QUICK else (0,)
t0 = time.time()
rows, hist = E.run_grid(data, BACKBONES, STRATEGIES, N_LABELS, SEEDS, cfg=cfg, save="results_grid.json")
df = pd.DataFrame(rows); df.to_csv("results_grid.csv", index=False)
print(f"total {(time.time()-t0)/60:.1f} min")
""")

M("## 6. Reference: original U'n'Eye on the last sample of a 200 ms bin (trained on ALL labels of set A)")
C("""
ref, ref_sets = E.unet_lastbin(data)
pd.Series(ref).round(3).to_frame("U-Net last bin")
""")

M("## 7. Results")
C("""
metric = "kappa@0.5"            # try "mcc@0.5", "f1@0.5", "ev_f1@0.5", "kappa@tuned"
g = df.groupby(["backbone", "strategy", "n_labels"])[metric].agg(["mean", "std"]).reset_index()
fig, axes = plt.subplots(1, len(BACKBONES), figsize=(4.2 * len(BACKBONES), 3.6), sharey=True, squeeze=False)
for ax, bb in zip(axes[0], BACKBONES):
    for st in STRATEGIES:
        d = g[(g.backbone == bb) & (g.strategy == st)]
        ax.errorbar(d.n_labels, d["mean"], d["std"].fillna(0), marker="o", capsize=2, label=st)
    ax.axhline(ref["kappa" if metric.startswith("kappa") else metric.split("@")[0]], ls="--", c="gray", label="U-Net (all labels)")
    ax.set_xscale("log"); ax.set_title(bb); ax.set_xlabel("labeled trials"); ax.spines[["top", "right"]].set_visible(False)
axes[0][0].set_ylabel(metric); axes[0][-1].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")
C("""
cols = ["kappa@0.5", "kappa@tuned", "mcc@0.5", "f1@0.5", "pr_auc@0.5", "ev_f1@0.5", "alarm_delay_ms@0.5", "false_alarms_per_min@0.5"]
table = df.groupby(["backbone", "strategy", "n_labels"])[cols].mean().round(3)
table
""")
C("""
# best strategy per backbone and N (pooled kappa)
best = df.groupby(["backbone", "n_labels", "strategy"])["kappa@0.5"].mean().reset_index()
best = best.loc[best.groupby(["backbone", "n_labels"])["kappa@0.5"].idxmax()]
print(best.to_string(index=False))
# gain of JEPA pretraining over training from scratch
p = df.pivot_table(index=["backbone", "n_labels"], columns="strategy", values="kappa@0.5")
if "jepa_ft" in p and "scratch" in p: print("\\nJEPA fine-tune minus scratch (kappa):\\n", (p["jepa_ft"] - p["scratch"]).round(3).to_string())
""")

M("""
## 8. How to read this
* If `jepa_ft` > `scratch` at small N and the gap closes at large N, pretraining helps: that is the label saving. Check against `recon_ft` (does the *latent* target matter?) and `weak_ft` (does a free classic detector give the same benefit?).
* `jepa_probe` close to `jepa_ft` means the representation already separates saccades (linear probe); far below means it has to be adapted.
* Compare **alarm delay** and **false alarms per minute**, not only kappa. For gaze-contingent experiments a +5 ms delay can matter more than +0.01 kappa.
* Differences below ~0.02 kappa with 1 seed are noise.
* Next step for the winner: export (`export_causal.py` for the TCN, or ONNX) and run it in the C++ pipeline (`cpp/`).
""")

C("""
# optional: ONNX export of a trained model from the grid (TCN / GRU variants; S4D uses complex FFT ops that ONNX cannot export)
net = archs.build("tcn_lite").cpu().eval()
try:
    torch.onnx.export(net, torch.zeros(1, 2, 200), "saccadenet_example.onnx", input_names=["dxy"], output_names=["prob"],
                      opset_version=17, dynamo=False, dynamic_axes={"dxy": {0: "batch", 2: "time"}, "prob": {0: "batch", 2: "time"}})
    print("exported saccadenet_example.onnx (random weights - export your fine-tuned model the same way)")
except Exception as e:
    print("export failed:", e)
""")

nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "UnEye_online_architectures.ipynb")
print("written", len(nb.cells), "cells")
