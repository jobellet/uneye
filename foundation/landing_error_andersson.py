#!/usr/bin/env python3
"""Landing-position estimate error of the C++ gaze engine's saccade tables (cpp/include/uneye/gaze_tables.hpp), on the Andersson test trials
(coder RA, out of sample: the tables were fit on datasets 1+2). At each time tau after a saccade's onset the engine predicts the landing point
= current position + remaining displacement a*d + b*v + c*vmax (along the current direction). Error = distance to the position at the offset
labelled by the human coder. Onset and offset are the human ones, so the plot isolates the estimator (not the detector).
Output: foundation/runs/landing_error_andersson.png and .csv"""
import os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
sys.path.insert(0, here)
import data as FD

txt = open(os.path.join(root, "cpp", "include", "uneye", "gaze_tables.hpp")).read()
blk = re.search(r"kLandCoef\[[^=]*=\s*\{(.*?)\n\};", txt, re.S).group(1)
coef = np.array([[float(v) for v in m.split(",")] for m in re.findall(r"\{([-\d.e+, ]+)\}", blk)]).reshape(3, 30, 3)[0]   # target 0: until the end
S = FD.load("andersson", "test"); pos = S.pos.astype(float); lab = S.lab; FS = 1000.0
TAUS = np.arange(0, 60, 2)                      # ms after onset
AMP_EDGES = [0, 2, 4, 7, 10, 15, 100]
rows = []                                       # (tau, amp_true, pred_err, abs_amp_err)
for i in range(len(pos)):
    X, Y = pos[i, :, 0], pos[i, :, 1]
    dd = np.diff(np.r_[0, (lab[i] == 1).astype(int), 0])
    for on, off in zip(np.where(dd == 1)[0], np.where(dd == -1)[0] - 1):
        if on < 2 or off - on < 6 or off + 5 >= len(X): continue
        seg = slice(on - 3, off + 6)
        if not (np.isfinite(X[seg]).all() and np.isfinite(Y[seg]).all()): continue
        p0 = np.array([X[on - 1], Y[on - 1]]); land = np.array([X[off], Y[off]]); amp = np.linalg.norm(land - p0)
        if amp < 0.5: continue
        vmax = 0.0
        for t in range(on, off + 1):
            tau = (t - on) * 1000.0 / FS
            if tau >= 60: break
            p = np.array([X[t], Y[t]]); disp = p - p0; d = np.linalg.norm(disp)
            v = np.mean([np.hypot(X[k] - X[k - 1], Y[k] - Y[k - 1]) for k in (t - 2, t - 1, t)]) * FS / 1000.0
            vmax = max(vmax, v)
            if d > 0.02: u = disp / d
            else:
                sv = np.array([X[t] - X[t - 3], Y[t] - Y[t - 3]]); n = np.linalg.norm(sv)
                if n < 1e-9: continue
                u = sv / n
            r = max(coef[min(int(tau / 2.0), 29)] @ np.array([d, v, vmax]), 0.0)
            rows.append((tau, amp, np.linalg.norm(p + r * u - land), abs(d + r - amp)))
R = np.array(rows); print("saccades used:", len({round(a, 6) for a in R[:, 1]}), " rows:", len(R))
nb = len(AMP_EDGES) - 1
M = np.full((nb, len(TAUS)), np.nan); N = np.zeros((nb, len(TAUS)), int); Mr = M.copy()
for a in range(nb):
    for k, t0 in enumerate(TAUS):
        m = (R[:, 1] >= AMP_EDGES[a]) & (R[:, 1] < AMP_EDGES[a + 1]) & (R[:, 0] >= t0) & (R[:, 0] < t0 + 2)
        N[a, k] = m.sum()
        if m.sum() >= 5: M[a, k] = R[m, 2].mean(); Mr[a, k] = (R[m, 2] / R[m, 1]).mean() * 100
labels = [f"{AMP_EDGES[a]}-{AMP_EDGES[a+1] if AMP_EDGES[a+1] < 100 else ''}°" if AMP_EDGES[a+1] < 100 else f">{AMP_EDGES[a]}°" for a in range(nb)]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
im = ax[0].imshow(M, origin="lower", aspect="auto", cmap="viridis", extent=[0, 60, -0.5, nb - 0.5]); ax[0].set_yticks(range(nb)); ax[0].set_yticklabels(labels)
ax[0].set_xlabel("time since saccade onset (ms)"); ax[0].set_ylabel("true amplitude"); ax[0].set_title("mean landing-point error (deg)"); fig.colorbar(im, ax=ax[0])
for a in range(nb): ax[1].plot(TAUS + 1, M[a], label=labels[a])
ax[1].set_xlabel("time since saccade onset (ms)"); ax[1].set_ylabel("mean landing-point error (deg)"); ax[1].legend(title="amplitude"); ax[1].grid(alpha=.3); ax[1].set_title("absolute error")
for a in range(nb): ax[2].plot(TAUS + 1, Mr[a], label=labels[a])
ax[2].set_xlabel("time since saccade onset (ms)"); ax[2].set_ylabel("error / amplitude (%)"); ax[2].set_ylim(0, 100); ax[2].grid(alpha=.3); ax[2].set_title("relative error")
fig.suptitle("Gaze-engine landing estimate vs time and amplitude: Andersson test set (RA labels, out of sample)")
fig.tight_layout(); out = os.path.join(here, "runs", "landing_error_andersson.png"); fig.savefig(out, dpi=130)
np.savetxt(out[:-4] + ".csv", np.c_[TAUS + 1, M.T], delimiter=",", header="tau_ms," + ",".join(labels), comments="")
print("written", out); print(np.round(M[:, ::3], 2)); print(N[:, ::3])
