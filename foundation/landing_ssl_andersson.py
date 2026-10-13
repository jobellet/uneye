#!/usr/bin/env python3
"""Does the masked-auto-encoder U-Net (foundation/runs/unet_pre.pt, pre-trained to fill hidden velocity spans) estimate where a saccade lands better than the
linear landing tables of the gaze engine?  Andersson test trials (RA labels; the pre-training pool holds no Andersson test position).
For a saccade with human onset `on` and a delay tau: positions are visible up to on+tau-1, the span [on+tau, on+tau+L) (L = 40, the longest span seen in
pre-training) is hidden (velocity channels blanked, mask channel = 1), everything AFTER the span stays visible (favourable to the network: a causal estimator
does not have it). Only saccades that end inside the hidden span are scored, so the whole remainder is hidden. The network's velocity in the span is
integrated from the last visible position; error = distance between that position at the end of the span and the true one (the same for the competitors:
'stays here' and the linear tables fit on Andersson train). -> foundation/runs/landing_ssl_andersson.png
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
from free_saccade import detectors as D
from foundation import data as FD, unet_ssl as U
from foundation.dino1d import feats2
from foundation.train_lodo import DEV
exec(open(os.path.join(ROOT, "foundation", "landing_error_nonlinear.py")).read().split("A = feats(")[0].split("FS = 1000.0")[1].join(["FS = 1000.0\n", ""])) if False else None
LEN, PRE, TAUS = 40, 200, (0, 3, 6, 10, 15, 20, 30)
S = FD.load("andersson", "test"); pos = S.pos.astype(np.float64); lab = S.lab
net = U.WUNet(nout=2).to(DEV); net.load_state_dict(torch.load(os.path.join(U.RUNS, "unet_pre.pt"), map_location=DEV, weights_only=False)); net.eval()
# linear tables fit on Andersson train (same features as landing_error_nonlinear.py)
def moves(Sx):
    p_, l_ = Sx.pos.astype(float), Sx.lab; R = []
    for i in range(len(p_)):
        X, Y = p_[i, :, 0], p_[i, :, 1]; dd = np.diff(np.r_[0, (l_[i] == 1).astype(int), 0])
        for on, off in zip(np.where(dd == 1)[0], np.where(dd == -1)[0] - 1):
            if on < 6 or off - on < 6 or off + 5 >= len(X) or not (np.isfinite(X[on-6:off+6]).all() and np.isfinite(Y[on-6:off+6]).all()): continue
            p0 = np.array([X[on-1], Y[on-1]])
            if np.linalg.norm(np.array([X[off], Y[off]]) - p0) < 0.5: continue
            vmax = 0
            for t in range(on, min(off, on + 59) + 1):
                p = np.array([X[t], Y[t]]); dsp = p - p0; d = np.linalg.norm(dsp); v = np.mean([np.hypot(X[k] - X[k-1], Y[k] - Y[k-1]) for k in (t-2, t-1, t)]) * 1.0; vmax = max(vmax, v)
                if d > 0.02: R.append((t - on, d, v, vmax, 0.0, i, on, off))
    return np.array(R)
tr = Sx = FD.load("andersson", "train"); A = []; p_, l_ = tr.pos.astype(float), tr.lab
for i in range(len(p_)):
    X, Y = p_[i, :, 0], p_[i, :, 1]; dd = np.diff(np.r_[0, (l_[i] == 1).astype(int), 0])
    for on, off in zip(np.where(dd == 1)[0], np.where(dd == -1)[0] - 1):
        if on < 6 or off - on < 6 or off + 5 >= len(X) or not (np.isfinite(X[on-6:off+6]).all() and np.isfinite(Y[on-6:off+6]).all()): continue
        p0 = np.array([X[on-1], Y[on-1]]); land = np.array([X[off], Y[off]])
        if np.linalg.norm(land - p0) < 0.5: continue
        vmax = 0
        for t in range(on, min(off, on + 59) + 1):
            p = np.array([X[t], Y[t]]); dsp = p - p0; d = np.linalg.norm(dsp); v = np.mean([np.hypot(X[k] - X[k-1], Y[k] - Y[k-1]) for k in (t-2, t-1, t)]); vmax = max(vmax, v)
            if d > 0.02: A.append((t - on, d, v, vmax, (land - p) @ (dsp / d)))
A = np.array(A); W = np.zeros((30, 3))
for b in range(30):
    m = (A[:, 0] // 2 == b) if b < 29 else (A[:, 0] // 2 >= 29); Fm = A[m][:, 1:4]; W[b] = np.linalg.solve(Fm.T @ Fm + 1e-3 * np.eye(3), Fm.T @ A[m, 4])
res = {t: {"stay": [], "linear": [], "ssl": [], "ssl_c": [], "amp": []} for t in TAUS}
for i in range(len(pos)):
    X, Y = pos[i, :, 0], pos[i, :, 1]; dd = np.diff(np.r_[0, (lab[i] == 1).astype(int), 0])
    for on, off in zip(np.where(dd == 1)[0], np.where(dd == -1)[0] - 1):
        s0 = on - PRE
        if s0 < 0 or s0 + 500 > len(X) or off - on < 6: continue
        w = pos[i, s0:s0 + 500]
        if not np.isfinite(w).all(): continue
        p0 = w[PRE - 1]; amp = np.linalg.norm(w[off - s0] - p0)
        if amp < 0.5: continue
        for tau in TAUS:
            a, b = PRE + tau, PRE + tau + LEN                       # hidden span [a, b)
            if off - s0 > b - 5 or tau < 1 or off - s0 < a + 3: continue
            vis = w[:a]; d_vec = w[a - 1] - p0; d = np.linalg.norm(d_vec)
            if d < 0.02: continue
            u = d_vec / d; vs = np.hypot(*np.diff(w[:a], axis=0).T)[-3:].mean() * 1.0; vm = np.hypot(*np.diff(w[PRE - 1:a], axis=0).T).max()
            rem = max(W[min(int(tau / 2), 29)] @ np.array([d, np.mean([np.hypot(*(w[a - 1 - k] - w[a - 2 - k])) for k in range(3)]), vm]), 0)
            truth = w[b - 1]
            m = np.zeros((1, 500), bool); m[0, a:b] = True
            f = feats2(w[None], 1000.0); v, _ = D.velocity(w[None], 1000.0); sg = np.maximum(D.robust_sigma(v).mean(2, keepdims=True), 1e-9)
            x = torch.as_tensor(np.concatenate([f * (1 - m[..., None]), m[..., None]], 2).astype(np.float32), device=DEV)
            with torch.no_grad(): out = net(x).transpose(1, 2).cpu().numpy()
            vel = np.sinh(out) * sg                                                  # deg/s
            wc = w[:b][None]; mc = np.zeros((1, b), bool); mc[0, a:b] = True; fc = feats2(wc, 1000.0); vc, _ = D.velocity(wc, 1000.0); sc = np.maximum(D.robust_sigma(vc).mean(2, keepdims=True), 1e-9)   # causal variant: the window ends with the hidden span
            xc = torch.as_tensor(np.concatenate([fc * (1 - mc[..., None]), mc[..., None]], 2).astype(np.float32), device=DEV); xp, T0 = U.pad25(xc)
            with torch.no_grad(): oc = net(xp).transpose(1, 2).cpu().numpy()[:, :T0]
            pendc = w[a - 1] + (np.sinh(oc) * sc)[0, a:b].sum(0) / 1000.0
            res[tau]["ssl_c"].append(np.linalg.norm(pendc - w[b - 1]))
            pend = w[a - 1] + vel[0, a:b].sum(0) / 1000.0
            res[tau]["stay"].append(np.linalg.norm(w[a - 1] - truth)); res[tau]["linear"].append(np.linalg.norm(w[a - 1] + rem * u - truth)); res[tau]["ssl"].append(np.linalg.norm(pend - truth)); res[tau]["amp"].append(amp)
print("tau  n  | mean error of the position at the end of the hidden span (deg): stays here / linear tables / masked auto-encoder")
rows = []
for tau in TAUS:
    r = {k: np.array(v) for k, v in res[tau].items()}
    if len(r["amp"]) < 5: continue
    rows.append((tau, len(r["amp"]), r["stay"].mean(), r["linear"].mean(), r["ssl"].mean(), r["ssl_c"].mean()))
    print(f"{tau:3d} {len(r['amp']):4d} | {r['stay'].mean():.2f} / {r['linear'].mean():.2f} / {r['ssl'].mean():.2f} / causal-window {r['ssl_c'].mean():.2f}   median {np.median(r['stay']):.2f} / {np.median(r['linear']):.2f} / {np.median(r['ssl']):.2f}")
R = np.array(rows); plt.figure(figsize=(6, 4))
for j, nm in ((2, "stays here"), (3, "linear tables"), (4, "masked auto-encoder (sees the future after the span)"), (5, "masked auto-encoder, window ends with the span (causal)")): plt.plot(R[:, 0], R[:, j], "o-", label=nm)
plt.xlabel("time since onset when the hidden span starts (ms)"); plt.ylabel("mean landing error (deg)"); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.title("Andersson test: saccades ending inside the hidden 40 ms"); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "foundation", "runs", "landing_ssl_andersson.png"), dpi=120)
