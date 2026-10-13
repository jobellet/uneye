#!/usr/bin/env python3
"""Is the beginning of a saccade enough to find saccades that land at the same place?
For every dataset and every subject (when the dataset has the information: dataset 4 and Andersson; datasets 1-3 have no subject column, so each is ONE group),
each human-labelled saccade is described by
  trajectory = p(onset + k) - p(onset - 1), k = 0 .. 9 ms (native sampling: 10 samples at 1 kHz, 5 at 500 Hz), x and y  (a vector of 2 x K numbers)
  landing    = p(offset) - p(onset - 1)   (2-D saccade vector)
Within a group, the nearest neighbour of saccade i in trajectory space (Euclidean) is j. Reported: the percentage of saccades for which j is ALSO the nearest neighbour of i in
landing space ('top-1'); the percentage for which j is among the 3 / 5 nearest in landing space; the median rank percentile of j in the landing ordering of i (0 = best, 50 = chance);
and the chance level of each (k / (n - 1)). Saccades: amplitude >= 0.5 deg, at least K samples long, finite positions from onset - 1 to offset + 1.
Groups with fewer than 30 saccades are skipped; groups above 3000 are subsampled (seed 0).  -> foundation/runs/nn_trajectory_landing.{csv,png}
"""
import glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from foundation import data as FD
MS = 10.0
def saccades(X, Y, L, fs):
    """X, Y, L: (n_trials, T) -> list of (trial, traj (2K,), landing (2,))"""
    K = max(int(round(MS * fs / 1000.0)), 2); out = []
    for i in range(len(X)):
        dd = np.diff(np.r_[0, (L[i] > 0).astype(int), 0])
        for on, off in zip(np.where(dd == 1)[0], np.where(dd == -1)[0] - 1):
            if on < 1 or off - on + 1 < K or off + 1 >= X.shape[1]: continue
            seg = slice(on - 1, off + 2)
            if not (np.isfinite(X[i, seg]).all() and np.isfinite(Y[i, seg]).all()): continue
            p0 = np.array([X[i, on - 1], Y[i, on - 1]]); land = np.array([X[i, off], Y[i, off]]) - p0
            if np.linalg.norm(land) < 0.5: continue
            traj = np.r_[X[i, on:on + K] - p0[0], Y[i, on:on + K] - p0[1]]
            out.append((i, traj, land))
    return out
groups = {}                                    # (dataset, subject) -> (traj list, landing list)
def add(ds, subj, sacc):
    t, l = groups.setdefault((ds, str(subj)), ([], []))
    for _, a, b in sacc: t.append(a); l.append(b)
for ds in ("d1", "d2", "d3", "d4"):
    d, f, fs = FD.REPO[ds]; subj = None
    for which in ("A", "B"):
        ld = lambda k: np.loadtxt(os.path.join(FD.ROOT, d, f"{f}_{k}_set{which}.csv"), delimiter=",")
        X, Y, L = ld("X"), ld("Y"), ld("Labels"); n = min(len(X), len(Y), len(L)); X, Y, L = X[:n], Y[:n], L[:n]
        if ds == "d4": subj = np.loadtxt(os.path.join(FD.ROOT, d, f"Subject_nb_set{which}.csv"), delimiter=",").astype(int)[:n]
        s = saccades(X, Y, L, fs)
        if subj is None: add(ds, "all", s)
        else:
            for u in np.unique(subj):
                m = np.where(subj == u)[0]; idx = {t: k for k, t in enumerate(m)}
                add(ds, f"s{u}", saccades(X[m], Y[m], L[m], fs))
art = os.path.join(FD.ROOT, "Andersson", "annotated_data", "data used in the article"); orig = os.path.join(FD.ROOT, "Andersson", "annotated_data", "originally uploaded data")
files = sorted(glob.glob(os.path.join(art, "*", "*_RA.mat"))); done = {os.path.basename(p)[:-7] for p in files}
files += sorted(p for p in glob.glob(os.path.join(orig, "*", "*_RA.mat")) if os.path.basename(p)[:-7] not in done)
for p in files:
    xy, l = FD._read_andersson(p); xy[(xy == 0).all(1)] = np.nan
    add("andersson", os.path.basename(p).split("_")[0], saccades(xy[None, :, 0], xy[None, :, 1], (l[None] == 1).astype(int), 500.0))
rng = np.random.RandomState(0); rows = []
print(f"{'dataset':10s}{'subject':>8s}{'n':>6s} | {'top-1':>7s}{'chance':>8s} | {'top-3':>7s}{'top-5':>7s}{'chance5':>8s} | median rank pct")
for (ds, sj), (T, Lnd) in sorted(groups.items()):
    T, Lnd = np.array(T), np.array(Lnd); n = len(T)
    if n < 30: continue
    if n > 3000: sel = rng.choice(n, 3000, replace=False); T, Lnd = T[sel], Lnd[sel]; n = 3000
    dT = np.linalg.norm(T[:, None] - T[None], axis=2); dL = np.linalg.norm(Lnd[:, None] - Lnd[None], axis=2); np.fill_diagonal(dT, np.inf); np.fill_diagonal(dL, np.inf)
    nn = dT.argmin(1); order = np.argsort(dL, 1)                       # landing neighbours of each saccade, nearest first (self is last)
    rank = np.array([np.where(order[i] == nn[i])[0][0] for i in range(n)])   # 0 = nearest in landing space
    top = [100 * np.mean(rank < k) for k in (1, 3, 5)]; med = 100 * np.median(rank) / (n - 2)
    ch = [100 * k / (n - 1) for k in (1, 3, 5)]
    rows.append((ds, sj, n, *top, *ch, med)); print(f"{ds:10s}{sj:>8s}{n:6d} | {top[0]:6.1f}%{ch[0]:7.2f}% | {top[1]:6.1f}%{top[2]:6.1f}%{ch[2]:7.2f}% | {med:5.1f}")
import csv
with open(os.path.join(ROOT, "foundation", "runs", "nn_trajectory_landing.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "subject", "n_saccades", "top1_pct", "top3_pct", "top5_pct", "chance1_pct", "chance3_pct", "chance5_pct", "median_rank_pct"]); w.writerows(rows)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2)); lab = [f"{r[0]}\n{r[1]}" for r in rows]; x = np.arange(len(rows))
ax[0].bar(x, [r[3] for r in rows], label="trajectory NN = landing NN (top-1)"); ax[0].plot(x, [r[6] for r in rows], "k_", ms=10, label="chance"); ax[0].set_ylabel("% of saccades"); ax[0].legend(fontsize=7)
ax[1].bar(x, [r[9] for r in rows], color="C1"); ax[1].axhline(50, color="k", ls="--", lw=1); ax[1].set_ylabel("median rank percentile of the trajectory NN\nin landing space (0 best, 50 chance)")
for a in ax: a.set_xticks(x); a.set_xticklabels(lab, fontsize=6, rotation=90)
fig.suptitle("First 10 ms of the trajectory -> nearest neighbour in landing vector"); fig.tight_layout(); fig.savefig(os.path.join(ROOT, "foundation", "runs", "nn_trajectory_landing.png"), dpi=120)
