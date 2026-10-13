#!/usr/bin/env python3
"""Does a non-linear landing estimator beat the fixed linear tables of the gaze engine? Andersson: fit on the 'train' split (RA only), score on the
'test' split (two-coder trials, RA labels). Causal features at time tau after the onset (human onset): tau, d (distance moved), v (mean speed of the
last 3 samples), vmax, time since the speed peak, distance at the peak, speed 5 ms ago, v/vmax. Target: displacement still to come along the current
direction. Models: linear tables (as in C++, per 2 ms bin) vs gradient boosting (sklearn HistGradientBoosting, L1 loss) vs 'stops here'.
Output: foundation/runs/landing_nonlinear_andersson.png"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); import data as FD
FS = 1000.0
def feats(S):
    pos = S.pos.astype(float); lab = S.lab; rows = []
    for i in range(len(pos)):
        X, Y = pos[i, :, 0], pos[i, :, 1]; dd = np.diff(np.r_[0, (lab[i] == 1).astype(int), 0])
        for sid, (on, off) in enumerate(zip(np.where(dd == 1)[0], np.where(dd == -1)[0] - 1)):
            if on < 6 or off - on < 6 or off + 5 >= len(X) or not (np.isfinite(X[on-6:off+6]).all() and np.isfinite(Y[on-6:off+6]).all()): continue
            p0 = np.array([X[on-1], Y[on-1]]); land = np.array([X[off], Y[off]]); amp = np.linalg.norm(land - p0)
            if amp < 0.5: continue
            vs = []; vmax = 0; tpk = 0; dpk = 0
            for t in range(on, min(off, on + 59) + 1):
                p = np.array([X[t], Y[t]]); disp = p - p0; d = np.linalg.norm(disp)
                v = np.mean([np.hypot(X[k] - X[k-1], Y[k] - Y[k-1]) for k in (t-2, t-1, t)]) * FS / 1000; vs.append(v)
                if v > vmax: vmax, tpk, dpk = v, t - on, d
                if d <= 0.02: continue
                tau = t - on; v5 = vs[tau - 5] if tau >= 5 else 0.0
                rows.append((tau, amp, d, v, vmax, tau - tpk, dpk, v5, v / max(vmax, 1e-6), (land - p) @ (disp / d), i * 1000 + sid))
    return np.array(rows)
A = feats(FD.load("andersson", "train")); B = feats(FD.load("andersson", "test"))
print("train rows", len(A), "test rows", len(B))
cols = [0, 2, 3, 4, 5, 6, 7, 8]
def linear_fit(R):
    W = np.zeros((30, 3))
    for b in range(30):
        m = (R[:, 0] // 2 == b) if b < 29 else (R[:, 0] // 2 >= 29); F = R[m][:, 2:5]; W[b] = np.linalg.solve(F.T @ F + 1e-3 * np.eye(3), F.T @ R[m, 9])
    return W
def linear_pred(R, W): return np.array([max(W[min(int(r[0] / 2), 29)] @ r[2:5], 0) for r in R])
W = linear_fit(A); p_lin = linear_pred(B, W)
g = HGB(loss="absolute_error", max_iter=300, learning_rate=0.06, max_leaf_nodes=24, min_samples_leaf=40, random_state=0).fit(A[:, cols], A[:, 9])
p_gb = np.maximum(g.predict(B[:, cols]), 0)
e = {"stops here (0)": np.abs(B[:, 9]), "linear tables": np.abs(p_lin - B[:, 9]), "gradient boosting": np.abs(p_gb - B[:, 9])}
TAUS = np.arange(0, 60, 4); AMP = [0, 4, 7, 10, 15, 100]; lab_a = ["<4°", "4-7°", "7-10°", "10-15°", ">15°"]
fig, ax = plt.subplots(1, 5, figsize=(20, 3.8), sharey=True)
for a in range(5):
    for nm, er in e.items():
        y = [er[(B[:, 1] >= AMP[a]) & (B[:, 1] < AMP[a+1]) & (B[:, 0] >= t) & (B[:, 0] < t + 4)].mean() for t in TAUS]
        ax[a].plot(TAUS + 2, y, label=nm)
    ax[a].set_title("amplitude " + lab_a[a]); ax[a].set_xlabel("time since onset (ms)"); ax[a].grid(alpha=.3)
ax[0].set_ylabel("mean error of the remaining displacement (deg)"); ax[0].legend()
fig.suptitle("Landing estimate on Andersson test: linear tables vs gradient boosting (both fit on Andersson train)"); fig.tight_layout()
out = os.path.join(here, "runs", "landing_nonlinear_andersson.png"); fig.savefig(out, dpi=120); print("written", out)
for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)):
    m = (B[:, 0] >= lo) & (B[:, 0] < hi); print(f"tau {lo:2d}-{hi:2d} ms  " + "  ".join(f"{k}: {v[m].mean():.2f}" for k, v in e.items()))
