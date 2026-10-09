# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# # Plug-and-play saccade / microsaccade detection WITHOUT human labels
#
# **Question.** DINO / JEPA / V-JEPA give "semantic segmentation for free" on images. Can the same idea, a self-supervised embedding of every time step of a gaze recording, replace a U'n'Eye that needs expert labels? And how do we know, on recordings that have **no labels**?
#
# **How it is tested** (everything runs on *your* Kaggle h5 data: `x` of shape `(n_sequences, L, 3) = (x, y, dt)` + `subject_h5_metadata.csv`):
#
# 1. **Candidates**, none uses a human label:
#    * classic detectors: Engbert-Kliegl at several thresholds, adaptive I-VT (Nystrom & Holmqvist), an HMM with a minimum-duration chain, and their **consensus**;
#    * self-supervised embeddings, one per time step: **dense DINO**, **1-D JEPA** (masked latent prediction), **dense contrastive** (InfoNCE; with and without the "neighbouring time steps are positives" persistence prior), each followed by k-means; *which clusters are saccades* is decided without labels (best intrinsic score);
#    * **physics-regularised self-training**: a small network trained on the consensus pseudo-labels (kept only where the detectors agree and the event obeys the main sequence), from scratch or on top of the DINO encoder, optionally 2 rounds (noisy student).
# 2. **Label-free tests** (on a held-out part of your data):
#    * *persistence*: a saccade label is not cut by gaps shorter than 20 ms and there is no 1-3 sample flicker;
#    * *main sequence*: amplitude vs peak speed and amplitude vs duration of the labeled events;
#    * *coverage / precision*: fast movement is labeled; labeled events really contain movement above the noise;
#    * *stereotypy* of the speed profile and straightness of the path;
#    * *invariance*: same answer after rotating/mirroring the gaze, playing the recording backwards, adding noise, shifting the time origin;
#    * **injection benchmark**: synthetic saccades (main sequence, minimum-jerk profile) are added to the quietest real windows of your data, so onset/offset are known **on your own noise**.
# 3. **Sanity check of the tests themselves** (the only place with human labels): the repository's labeled recordings are scored with the *same* label-free tests, and we check whether the score ranks methods like the real Cohen's kappa does. If it does not, the tests cannot be trusted to pick a winner, and the notebook says so.
#
# Set `QUICK = True` for a smoke test (minutes). `False` is the real run.

# %%
def display(*objs):                       # notebook display() -> plain print in a script
    for o in objs: print(o.to_string() if hasattr(o, "to_string") else o)
import os, sys, subprocess, glob, time
import matplotlib; matplotlib.use("Agg")      # headless
QUICK = "--full" not in sys.argv   # smoke test by default; pass --full for the real run
REPO, BRANCH = "https://github.com/jobellet/uneye", "master"
def sh(cmd, t=180):
    print("$", " ".join(cmd), flush=True)
    try: subprocess.run(cmd, check=False, timeout=t)
    except subprocess.TimeoutExpired: print("   (timeout after", t, "s, continuing)", flush=True)
if os.path.basename(os.getcwd()) == "free_saccade": os.chdir("..")   # run from the repository
if not os.path.exists("data/dataset1"):             # Colab / Kaggle: get the code (and the labeled recordings used for the sanity check)
    if not os.path.exists("uneye"): sh(["git", "clone", "-q", "-b", BRANCH, REPO])      # needs Internet ON in the Kaggle notebook settings
    else: sh(["git", "-C", "uneye", "pull", "-q", "origin", BRANCH])
    if not os.path.exists("uneye"): raise SystemExit("git clone failed: switch Internet ON (Kaggle: Settings > Internet) and run this cell again")
    os.chdir("uneye")

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "online"))
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
_n=[0]
def _show(*a,**k):
    os.makedirs("figures_run", exist_ok=True); _n[0]+=1; plt.savefig(f"figures_run/{os.path.basename(__file__)[:-3]}_{_n[0]:02d}.png", dpi=110); plt.close("all")
plt.show = _show
from free_saccade import data as FD, detectors as D, ssl_models as S, intrinsic as I, suite, viz
torch.set_num_threads(max(os.cpu_count() or 1, 1))
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", dev, "| CPU cores:", os.cpu_count(), "| torch", torch.__version__)
if QUICK: print("*** QUICK MODE: smoke test only, the numbers are not meaningful. Set QUICK = False for the real run. ***")

# %% [markdown]
# ## 1. Your data
#
# The cell looks for `subject_h5_metadata.csv` under `/kaggle/input` (attach your dataset to the notebook). If it finds nothing it builds a **demo** dataset in the same layout from the repository recordings (labels removed), so everything runs anywhere.
#
# Edit `UNIT_SCALE` if the positions are not in degrees (the audit table below shows the position range and speed noise of every source: a wrong unit shows immediately as absurd speeds).

# %%
UNIT_SCALE = 1.0          # multiply positions by this to get degrees
GROUP_COL = None          # metadata column naming the source dataset; None = guess (dataset/source/study column, else file name prefix)
def find_meta():
    for pat in ("/kaggle/input/**/subject_h5_metadata.csv", "/kaggle/input/**/*metadata*.csv", "/kaggle/input/**/*.csv"):
        for f in glob.glob(pat, recursive=True):
            try:
                if "filename" in pd.read_csv(f, nrows=2).columns: return f
            except Exception: pass
    return None
ARCHIVE_META = "archive/subject_h5_metadata.csv"          # local multi-source data (docs/ARCHIVE_DATA.md), not in git
use_archive = "--demo" not in sys.argv and os.path.exists(ARCHIVE_META)
meta = ARCHIVE_META if use_archive else find_meta()
if meta is None:
    print("!!! NO KAGGLE DATASET FOUND -> the results below are for the DEMO data (repository recordings), NOT for your data. !!!")
    print("Files under /kaggle/input:", glob.glob("/kaggle/input/*") or "none  (use 'Add Input' in the Kaggle notebook to attach your dataset)")
    meta = FD.write_demo_h5("data", "demo_gaze", per_file=40 if QUICK else 150, files_per_set=3 if QUICK else 6)
SRC_DIR = os.path.dirname(meta)
if use_archive:                                       # fixed rate per source, unusable sources skipped, any unit (no absolute limit)
    src = FD.H5Source("archive/datasets_by_subject/datasets_by_subject", meta, "dataset", 1.0, FD.ARCHIVE_FS, limit=None, clip=FD.ARCHIVE_CLIP)
else:
    src = FD.H5Source(SRC_DIR, meta, GROUP_COL, UNIT_SCALE)
print("metadata:", meta, "| source column:", src.group_from, "|", len(src.meta), "files")
audit = src.audit()
display(audit.round(3))

# %% [markdown]
# ## 2. Windows
# Every source is resampled to **500 Hz** (so "20 ms" is 10 samples everywhere) and cut into windows of 256 samples (0.5 s; sources whose sequences are shorter are reported with a warning). Windows with more than 20 % invalid samples (blinks, dropouts) are skipped; the invalid samples that remain stay `NaN`, nothing is filled with fake data. Half of the windows (disjoint) are used to **train**, the other half only to **evaluate**.
# `labeled` holds the repository's human-labeled recordings, used only in section 4.

# %%
rng = np.random.RandomState(0)
N_PER = 60 if QUICK else 600
WIN = 256                                  # 0.5 s at 500 Hz: also fits the short sequences
pool_all = src.sample(2 * N_PER, WIN, rng)
pool_all, UNIT_SCALES = FD.normalize_units(pool_all)     # unit-free: every source rescaled to the same speed noise (pseudo-degrees)
print("unit scale per source (x positions):", {k: round(v, 3) for k, v in UNIT_SCALES.items()})
perm = rng.permutation(len(pool_all)); half = len(pool_all) // 2
train, test = pool_all.subset(np.sort(perm[:half])), pool_all.subset(np.sort(perm[half:]))
labeled = FD.load_labeled("data", n=40 if QUICK else 150)
labeled = {k: FD.normalize_units(w)[0] for k, w in labeled.items()}   # same front end as the unlabeled data
print(len(train), "train windows,", len(test), "test windows; sources:", dict(zip(*np.unique(test.group, return_counts=True))))
print("labeled sanity-check sets:", {k: v.pos.shape for k, v in labeled.items()})

# %% [markdown]
# ## 3. Classic detectors (no training)
# Each is a function `Windows -> bool labels`. They are also the **teachers** of the self-training in section 6.

# %%
t0 = time.time()
hmm = D.HMM(train.fs).fit(train)                                      # fitted once on the training windows, frozen afterwards
print(f"HMM: mu={np.round(hmm.mu, 2)} sd={np.round(hmm.sd, 2)} p_on={hmm.p_on:.4f} p_stay={hmm.p_stay:.2f}  ({time.time() - t0:.0f} s)")
DETECT = {
    "EK lam=3":  lambda w: D.ek(w.pos, w.fs, lam=3.0),
    "EK lam=6":  lambda w: D.ek(w.pos, w.fs, lam=6.0),
    "EK lam=12": lambda w: D.ek(w.pos, w.fs, lam=12.0),
    "I-VT adaptive": D.ivt_adaptive,
    "HMM": hmm.predict,
}
DETECT["consensus (EK, I-VT, HMM)"] = lambda w: D.consensus(w, [DETECT["EK lam=6"](w), DETECT["I-VT adaptive"](w), DETECT["HMM"](w)])[0]

# %% [markdown]
# ## 4. Do the label-free tests rank methods like the truth?
# The same tests are run on the human-labeled recordings, for the classic detectors, and compared with the real Cohen's kappa. **Read this before trusting any ranking below.** The human labels themselves are scored too (row `human`).

# %%
rows, labs = [], {}
for name, f in DETECT.items():
    r, l = suite.evaluate(name, f, test, labeled, n_invariance=30 if QUICK else 80, n_inject=40 if QUICK else 120)
    rows.append(r); labs[name] = l
    print(f"{name:28s} ILS {r['ILS']:.2f}  persistence {r['persistence']:.2f}  main-seq {r['main_sequence']:.2f}  invariance {r['invariance']:.2f}  injection-kappa {r['inj_kappa']:.2f}   | real kappa {r['kappa']:.2f}")
vt = suite.validation_table(rows, list(labeled))
fig, (rho, rho_in) = viz.validation_scatter(vt); plt.show()
print(f"Spearman(ILS, real kappa) over {len(vt)} (method, dataset) pairs: pooled {rho:.2f}, mean within a dataset {rho_in:.2f}")
print("Within a dataset is what matters: the question is which method to pick for THIS recording setup.")
hum = {nm: I.score(w.pos, w.labels, w.fs)["ILS"] for nm, w in labeled.items()}
print("ILS of the HUMAN labels:", {k: round(v, 2) for k, v in hum.items()})

# %% [markdown]
# ## 5. Self-supervised embeddings (DINO / JEPA / contrastive)
# Input features are **rotation invariant** (speed relative to the window's own noise, acceleration, curvature), because the direction of a saccade says nothing about being one. Views of the same window differ by rotation, gain, small time warp and noise. The encoder is a dilated 1-D convolution stack, one embedding per time step.
#
# Then k-means on the embeddings (`K=6`) and the **cut** between fixation-like and saccade-like clusters (clusters ordered by their mean speed) is chosen by the best intrinsic score, never by a label.

# %%
STEPS = 150 if QUICK else 1500
SSL_CONFIGS = {"DINO": dict(method="dino"), "JEPA": dict(method="jepa"),
               "contrastive": dict(method="infonce", tpos=0), "contrastive+persistence": dict(method="infonce", tpos=6)}
ssl_models, cdet = {}, {}
fit_win = suite.subset(train, 200 if QUICK else 600)     # the intrinsic score on these windows chooses which clusters are saccades
score_fn = lambda lab: I.score(fit_win.pos, lab, fit_win.fs, groups=fit_win.group)["ILS"]
for name, cfg in SSL_CONFIGS.items():
    t0 = time.time(); cfg = dict(cfg); tpos = cfg.pop("tpos", 0)
    m = S.SSL(device=dev, **cfg).train(train, steps=STEPS, tpos=tpos, verbose=False)
    ssl_models[name] = m
    cdet[name] = S.ClusterDetector(m, k=6).fit(fit_win, score_fn=score_fn)
    DETECT[name] = cdet[name].predict
    print(f"{name}: trained in {time.time() - t0:.0f} s, saccade clusters = {len(cdet[name].order) - cdet[name].cut} of 6, loss {np.mean(m.hist[:20]):.3f} -> {np.mean(m.hist[-20:]):.3f}")
fig, axs = plt.subplots(1, len(ssl_models), figsize=(4 * len(ssl_models), 3))
for ax, (n_, m) in zip(np.atleast_1d(axs), ssl_models.items()): ax.plot(m.hist); ax.set_title(n_ + " loss"); ax.set_xlabel("step")
plt.tight_layout(); plt.show()

# %%
for name in SSL_CONFIGS:
    r, l = suite.evaluate(name, DETECT[name], test, labeled, n_invariance=30 if QUICK else 80, n_inject=40 if QUICK else 120)
    rows.append(r); labs[name] = l
    print(f"{name:28s} ILS {r['ILS']:.2f}  persistence {r['persistence']:.2f}  main-seq {r['main_sequence']:.2f}  invariance {r['invariance']:.2f}  injection-kappa {r['inj_kappa']:.2f}   | real kappa {r['kappa']:.2f}")
m = ssl_models["DINO"]; e = m.embed(suite.subset(test, 40)); fig = viz.embedding_plot(e, suite.subset(test, 40)); plt.show()


# %% [markdown]
# ## 6. Physics-regularised self-training (noisy student)
# Pseudo-labels = **where EK, I-VT and the HMM all agree** (saccade) or all say fixation with a quiet signal (fixation); every other sample, every sample near invalid data and the edges of saccades get weight 0. Events that violate the **main sequence** (peak speed more than 3x away from the pooled fit for their amplitude) are removed from the pseudo-labels. A small network learns from them with augmented inputs; it can therefore correct the edges and recover movements the three teachers hesitated about. `2 rounds` = the confident predictions of round 1 become the labels of round 2.

# %%
def main_sequence_filter(win, lab, w):
    amp, vp, du = I.main_sequence_table(win.pos, lab, win.fs)
    ok = (amp > 0.05) & (vp > 0)
    if ok.sum() < 20: return w
    a, b = np.polyfit(np.log10(amp[ok]), np.log10(vp[ok]), 1)
    w = w.copy()
    for e in D.events_of(win.pos, lab, win.fs):
        pred = 10 ** (a * np.log10(max(e["amp"], 0.05)) + b)
        if not (pred / 3 < e["vpeak"] < pred * 3): w[e["win"], e["on"]:e["off"] + 1] = 0.0
    return w
t0 = time.time()
votes = [D.ek(train.pos, train.fs), D.ivt_adaptive(train), hmm.predict(train)]
pl, pw = D.consensus(train, votes); pw = main_sequence_filter(train, pl, pw)
print(f"pseudo-labels: {pl[pw > 0].mean():.3f} saccade fraction among {pw.mean():.2f} of the samples  ({time.time() - t0:.0f} s)")
ST_STEPS = 150 if QUICK else 1200
students = {"self-train (scratch)": dict(), "self-train (scratch, 2 rounds)": dict(rounds=2),
            "self-train (frozen DINO encoder)": dict(enc="DINO", freeze=True), "self-train (finetuned DINO encoder)": dict(enc="DINO")}
for name, cfg in students.items():
    cfg = dict(cfg); enc = cfg.pop("enc", None)
    net = S.self_train(train, pl, pw, train.fs, init_encoder=(ssl_models[enc].enc if enc else None), steps=ST_STEPS, device=dev, verbose=False, **cfg)
    DETECT[name] = S.StudentDetector(net).predict
    r, l = suite.evaluate(name, DETECT[name], test, labeled, n_invariance=30 if QUICK else 80, n_inject=40 if QUICK else 120)
    rows.append(r); labs[name] = l
    print(f"{name:36s} ILS {r['ILS']:.2f}  persistence {r['persistence']:.2f}  main-seq {r['main_sequence']:.2f}  invariance {r['invariance']:.2f}  injection-kappa {r['inj_kappa']:.2f}   | real kappa {r['kappa']:.2f}")

# %% [markdown]
# ## 7. Results
# **Columns.** `ILS` = mean of persistence, main sequence, coverage, precision, stereotypy (all label-free, on your held-out windows). `invariance` = mean agreement (kappa) under rotation / time reversal / noise / shift. `inj_kappa` = kappa on the injection benchmark (truth known, your noise). `kappa` = real kappa against human labels, **reference only** (repository recordings).

# %%
df = suite.to_frame(rows)
cols = ["ILS", "persistence", "main_sequence", "coverage", "precision", "stereotypy", "invariance", "inj_kappa", "inj_ev_recall", "events_per_min", "flips_per_s", "kappa", "ev_f1"]
display(df[cols].round(3).sort_values("inj_kappa", ascending=False))
viz.scores_heatmap(df, title="label-free tests (1 = best)"); plt.show()

# %%
# agreement between the label-free ranking and the truth, over ALL methods
from scipy.stats import spearmanr
print("Spearman with the real kappa (human labels, reference):")
for c in ("ILS", "inj_kappa", "invariance", "persistence", "main_sequence"):
    print(f"  {c:14s} {spearmanr(df[c], df['kappa'], nan_policy='omit')[0]:+.2f}")

# %%
show = ["EK lam=6", "HMM", "consensus (EK, I-VT, HMM)", "DINO", "contrastive+persistence", "self-train (scratch)"]
show = [s for s in show if s in labs]
fig = viz.trace_gallery(test, {k: labs[k] for k in show}, n=6, title="your unlabeled recordings: busiest windows"); plt.show()
fig = viz.main_sequence_plot(test, {k: labs[k] for k in show}); plt.show()
fig = viz.duration_plot(test, {k: labs[k] for k in show}); plt.show()

# %%
# the same view on the human-labeled recordings (one dataset): are the labels where a human put them?
nm = list(labeled)[0]; w = labeled[nm]
lab_w = {k: DETECT[k](w) for k in show}
fig = viz.trace_gallery(w, lab_w, n=5, human=w.labels, title=f"labeled dataset {nm} (black band = human)"); plt.show()

# %% [markdown]
# ## 8. Save
# The label arrays of every method on the held-out windows and the score table are written to disk, so the best method can be exported (next step: distil it into the causal TCN of the C++ engine).

# %%
os.makedirs("free_results", exist_ok=True)
df.to_csv("free_results/scores.csv")
np.savez_compressed("free_results/labels_test.npz", pos=test.pos, group=test.group, **{k.replace(" ", "_"): v for k, v in labs.items()})
print("saved to", os.path.abspath("free_results"))

# %% [markdown]
# ## How to read the outcome
# * If a self-supervised method has a high `inj_kappa`, high `invariance` **and** its `ILS`/`kappa` agree with the classic baselines, it is a plug-and-play candidate.
# * In a local check on the 4 labeled repository sets (see `RESULTS.md`) the label-free ILS was a weak ranker (mean within-dataset Spearman +0.21 with the real kappa; coverage +0.43; main sequence and precision were even negative). Trust `inj_kappa` and `coverage` more than `ILS`.
# * The label-free score is a **failure detector** (a method that labels noise, flickers, or finds nothing gets a low score), not a fine-grained ranker: look at the correlation printed in section 4 before ranking two methods with a small score difference.
# * Differences between sources (rows in the audit) matter: run section 7 per source if one source has a very different noise or sampling rate.
# * The honest ceiling is a **few** human labels, used only to check the final choice (or the cut), not for training. The U'n'Eye notebook of this repository shows how few are needed when a method like `weak_ft` is used.
