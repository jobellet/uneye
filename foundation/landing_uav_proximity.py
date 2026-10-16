#!/usr/bin/env python3
"""PQFT saliency (the engine's map, the fastest of the models with the best NSS) weighted by a Gaussian centred on the eye position BEFORE the saccade (p0): candidate weight = saliency^g * exp(-|c - p0|^2 / (2 s^2)),
candidates = a dense 32 x 18 grid over the frame. Selection as in the engine (direction / amplitude consistency with the saccade so far, candidates.hpp) with these weights as priors, or the
arg-max of the weighted map alone. Errors (deg) at tau = 10 / 20 ms on the EyeTrackUAV2 / DTB70 saccades of foundation/landing_uav_saliency2.py; s is tuned on the even-numbered saccades
and reported on the odd-numbered ones.   -> foundation/runs/landing_uav_proximity.json"""
import glob, json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepgaze_kit"))
from uav_common import *
from foundation.saliency_models import Model
GX, GY = np.meshgrid((np.arange(32) + 0.5) * 1280 / 32, (np.arange(18) + 0.5) * 720 / 18); GRID = np.stack(f2deg(GX.ravel(), GY.ravel()), 1)
sacc = detect_saccades(); print("saccades", len(sacc), flush=True)
need = {}
for si, f0, *_ in sacc: need.setdefault(si, set()).add(f0)
W = {}                                                      # (si, f0) -> saliency at the grid points
for si, seq in enumerate(SEQS):
    mod = Model("pqft"); files = sorted(glob.glob(os.path.join(ROOT, "data", "DTB70", seq, "img", "*.jpg")))
    for k, f in enumerate(files):
        m = mod(np.asarray(Image.open(f).convert("RGB")))
        if k in need[si]: h, w = m.shape; W[(si, k)] = m[np.clip((GY.ravel() / 720 * h).astype(int), 0, h - 1), np.clip((GX.ravel() / 1280 * w).astype(int), 0, w - 1)]
    print(seq, flush=True)
SIG = [1, 2, 3, 5, 8, 12, 1e9]; GAM = [0, 1, 2]                       # gamma = 0: no saliency (proximity only)
def run(idx, s, g, mode):
    e = {10: [], 20: []}
    for i in idx:
        si, f0, p0, land, feats = sacc[i]; wgt = (W[(si, f0)] + 1e-3) ** g * np.exp(-np.sum((GRID - p0) ** 2, 1) / (2 * s * s)) + 1e-9
        for tau, (cur, rem, tb) in feats.items():
            if mode == "kin": j = pick(p0, cur, rem, GRID, wgt); est = rule(p0, GRID[j], tb, tau) if j >= 0 else tb
            else: est = rule(p0, GRID[int(np.argmax(wgt))], tb, tau)
            e[tau].append(np.linalg.norm(est - land))
    return np.array([np.mean(e[10]), np.mean(e[20])])
even = list(range(0, len(sacc), 2)); odd = list(range(1, len(sacc), 2)); out = {"saccades": len(sacc), "tables": {str(t): float(np.mean([np.linalg.norm(s[4][t][2] - s[3]) for s in sacc if t in s[4]])) for t in (10, 20)}, "results": []}
print("tables", out["tables"])
for mode in ("kin", "argmax"):
    for g in GAM:
        tr = [(s, run(even, s, g, mode)) for s in SIG]; s_best = min(tr, key=lambda t: t[1].sum())[0]; te = run(odd, s_best, g, mode)
        row = {"mode": mode, "gamma": g, "sigma_deg": s_best if s_best < 1e8 else None, "test_error": {"10": float(te[0]), "20": float(te[1])}, "train_by_sigma": {str(s if s < 1e8 else "inf"): [float(v) for v in e] for s, e in tr}}
        out["results"].append(row); print(mode, "gamma", g, "sigma", row["sigma_deg"], "test (odd saccades):", np.round(te, 2), flush=True)
json.dump(out, open(os.path.join(ROOT, "foundation", "runs", "landing_uav_proximity.json"), "w"), indent=1)
