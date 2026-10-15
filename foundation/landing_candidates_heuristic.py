#!/usr/bin/env python3
"""The simple candidate-target selector planned for the C++ engine, tuned on GazeBase RAN (train split) and scored on the test split, with K = 2, 4, 8 candidates (one true).
At time tau after the onset: chord u = (cur - p0) / d, table amplitude A = d + r0 (the engine's fixed tables, cpp/include/uneye/gaze_tables.hpp, features d, v, vmax).
For each candidate c_k (position relative to p0): angle theta_k between c_k and the chord, amplitude |c_k|.
  score_k = w_k * exp(-theta_k^2 / (2 s_th^2)) * exp(-(ln|c_k| - ln A)^2 / (2 s_a^2)),   s_th = th0 + th1 * exp(-tau / th_tau) ... (here: s_th(tau) = a / (1 + d / d0))
posterior p_k = score_k / sum(score);  if max p_k >= P_MIN -> landing = candidate k (p0 + c_k), else landing = table landing (cur + r0 u).
Grid search of (s_th0, s_a, P_MIN) on the train split.   -> prints the table and the best parameters (to put in cpp/app/engine/include/eyelive/candidates.hpp)
"""
import itertools, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import foundation.landing_train as L
import foundation.landing_gazebase as G
txt = open(os.path.join(ROOT, "cpp", "include", "uneye", "gaze_tables.hpp")).read()
blk = re.search(r"kLandCoef\[[^=]*=\s*\{(.*?)\n\};", txt, re.S).group(1)
COEF = np.array([[float(v) for v in m.split(",")] for m in re.findall(r"\{([-\d.e+, ]+)\}", blk)]).reshape(3, 30, 3)[0]
z = np.load(os.path.join(ROOT, "data", "GazeBase", "ran_saccades.npz")); subj = z["subj"] % 1000; us = np.unique(subj); rng = np.random.RandomState(0); rng.shuffle(us)
nt, nv = int(0.8 * len(us)), int(0.1 * len(us)); sets = {"train": us[:nt], "test": us[nt + nv:]}; cap = {"train": 15000, "test": 8000}
def prep(k):
    idx = np.sort(rng.choice(np.where(np.isin(subj, sets[k]))[0], min(int(np.isin(subj, sets[k]).sum()), cap[k]), replace=False)); sp = G.build(z, idx)
    P0 = z["P"][idx][:, G.PRE - 1]; absT = z["tv"][idx] + P0; n = len(idx); r2 = np.random.RandomState(7); C = np.zeros((n, 8, 2)); C[:, 0] = z["tv"][idx]
    for j in range(1, 8): o = r2.randint(0, n, n); o = np.where(o == np.arange(n), (o + 1) % n, o); C[:, j] = absT[o] - P0
    i, on, off, t = sp.R.T; tp = t + L.PAD; p0 = sp.P[i, on + L.PAD - 1]; cur = sp.P[i, tp]; land = sp.P[i, off + L.PAD]
    # features as in the C++ forecast(): d, v (mean speed of the last 3 samples, deg/ms), vmax since the onset
    spd = np.hypot(*(sp.P[:, 1:] - sp.P[:, :-1]).transpose(2, 0, 1)); spd = np.concatenate([np.zeros((len(spd), 1)), spd], 1)       # (n, T): speed at sample s (deg/ms at 1 kHz)
    v = np.mean([spd[i, tp - k_] for k_ in range(3)], 0)
    vmax = np.array([spd[ii, oo + L.PAD:tt + 1].max() for ii, oo, tt in zip(i, on, tp)])
    d = np.linalg.norm(cur - p0, axis=1); u = (cur - p0) / np.maximum(d, 1e-9)[:, None]; tau = t - on
    r0 = np.maximum((COEF[np.minimum(tau // 2, 29)] * np.c_[d, v, vmax]).sum(1), 0)
    return dict(i=i, tau=tau, p0=p0, cur=cur, land=land, d=d, u=u, r0=r0, C=C[i])
def select(D, K, s_th0, s_a, d0, pmin):
    C = D["C"][:, :K]; chord = D["u"]; cn = np.maximum(np.linalg.norm(C, axis=2), 1e-6)
    cosv = np.clip((C * chord[:, None, :]).sum(2) / cn, -1, 1); th = np.arccos(cosv); A = np.maximum(D["d"] + D["r0"], 0.3)
    s_th = s_th0 / (1.0 + D["d"] / d0)
    sc = np.exp(-th ** 2 / (2 * s_th[:, None] ** 2)) * np.exp(-(np.log(cn) - np.log(A)[:, None]) ** 2 / (2 * s_a ** 2)) + 1e-12
    p = sc / sc.sum(1, keepdims=True); k = p.argmax(1); conf = p.max(1)
    tab = D["cur"] + D["r0"][:, None] * D["u"]; cand = D["p0"] + C[np.arange(len(k)), k]
    pred = np.where((conf >= pmin)[:, None], cand, tab)
    return pred, k, conf
tr, te = prep("train"), prep("test")
m_tr = (tr["tau"] >= 3); best = None
for s_th0, s_a, d0, pmin in itertools.product((0.12, 0.2, 0.3), (1.0, 1.5, 3.0), (0.5, 2.0, 6.0), (0.0, 0.5)):
    e = np.mean([np.linalg.norm(select(tr, K, s_th0, s_a, d0, pmin)[0] - tr["land"], axis=1)[m_tr].mean() for K in (2, 4, 8)])
    if best is None or e < best[0]: best = (e, (s_th0, s_a, d0, pmin))
print("best parameters (s_th0, s_a, d0, pmin):", best[1], " train mean error", round(best[0], 3))
bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]
print(f"{'':22s}" + "".join(f"{lo:>2d}-{hi:<4d}" for lo, hi in bins) + "  all    true chosen")
tab = np.linalg.norm(te["cur"] + te["r0"][:, None] * te["u"] - te["land"], axis=1)
rows = [("tables only", tab, None)]
for K in (1, 2, 4, 8):
    pred, k, conf = select(te, K, *best[1]); rows.append((f"candidates K={K}", np.linalg.norm(pred - te["land"], axis=1), float(np.mean(k == 0)) if K > 1 else None))
for name, e, acc in rows:
    print(f"{name:22s}" + "".join(f"{e[(te['tau'] >= lo) & (te['tau'] < hi)].mean():7.2f}" for lo, hi in bins) + f"  {e.mean():.2f}   " + (f"{acc * 100:.1f} %" if acc is not None else ""))
