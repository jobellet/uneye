#!/usr/bin/env python3
"""A learned prior on the saccade vector instead of a Gaussian: P(dx, dy) = smoothed 2-D histogram (1 deg bins) of (landing - eye position before the saccade) of the TRAIN participants
(even subject numbers), applied to a dense 32 x 18 grid of candidates (weight = P(c - p0)^a, optionally x PQFT saliency), with the engine's selector (kinematic consistency) or as an arg-max;
errors at tau = 10 / 20 ms on the TEST participants (odd subject numbers).   -> foundation/runs/landing_uav_prior.json"""
import glob, json, os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepgaze_kit"))
from uav_common import *
from foundation.saliency_models import Model
GX, GY = np.meshgrid((np.arange(32) + 0.5) * 1280 / 32, (np.arange(18) + 0.5) * 720 / 18); GRID = np.stack(f2deg(GX.ravel(), GY.ravel()), 1)
sacc = detect_saccades(with_subject=True); tr = [i for i, s in enumerate(sacc) if s[5] % 2 == 0]; te = [i for i, s in enumerate(sacc) if s[5] % 2 == 1]; print("saccades", len(sacc), "train", len(tr), "test", len(te), flush=True)
EX, EY = 25.0, 15.0; dxy = np.array([sacc[i][3] - sacc[i][2] for i in tr]); H2, _, _ = np.histogram2d(dxy[:, 0], dxy[:, 1], bins=[int(2 * EX), int(2 * EY)], range=[[-EX, EX], [-EY, EY]]); H2 = gaussian_filter(H2, 1.0) + 1e-3; H2 /= H2.sum()
def prior(v):
    i = np.clip(((v[:, 0] + EX)).astype(int), 0, H2.shape[0] - 1); j = np.clip(((v[:, 1] + EY)).astype(int), 0, H2.shape[1] - 1); return H2[i, j] * H2.size
amp = np.linalg.norm(dxy, axis=1); print("train landing vectors: median amplitude %.1f deg, share rightwards %.2f, horizontal-dominant %.2f" % (np.median(amp), (dxy[:, 0] > 0).mean(), (np.abs(dxy[:, 0]) > np.abs(dxy[:, 1])).mean()), flush=True)
need = {}
for s in sacc: need.setdefault(s[0], set()).add(s[1])
W = {}
for si, seq in enumerate(SEQS):
    mod = Model("pqft")
    for k, f in enumerate(sorted(glob.glob(os.path.join(ROOT, "data", "DTB70", seq, "img", "*.jpg")))):
        m = mod(np.asarray(Image.open(f).convert("RGB")))
        if k in need[si]: h, w = m.shape; W[(si, k)] = m[np.clip((GY.ravel() / 720 * h).astype(int), 0, h - 1), np.clip((GX.ravel() / 1280 * w).astype(int), 0, w - 1)]
def run(idx, a, g, mode):
    e = {10: [], 20: []}
    for i in idx:
        si, f0, p0, land, feats, _ = sacc[i]; wgt = prior(GRID - p0) ** a * (W[(si, f0)] + 1e-3) ** g + 1e-9
        for tau, (cur, rem, tb) in feats.items():
            if mode == "kin": j = pick(p0, cur, rem, GRID, wgt); est = rule(p0, GRID[j], tb, tau) if j >= 0 else tb
            else: est = rule(p0, GRID[int(np.argmax(wgt))], tb, tau)
            e[tau].append(np.linalg.norm(est - land))
    return np.array([np.mean(e[10]), np.mean(e[20])])
out = {"train": len(tr), "test": len(te), "tables": {str(t): float(np.mean([np.linalg.norm(sacc[i][4][t][2] - sacc[i][3]) for i in te if t in sacc[i][4]])) for t in (10, 20)}, "results": []}
print("tables (test participants)", out["tables"], "flat grid", np.round(run(te, 0, 0, "kin"), 2), flush=True)
for mode in ("kin", "argmax"):
    for g in (0, 1):
        for a in (0.5, 1.0, 2.0):
            r = run(te, a, g, mode); out["results"].append({"mode": mode, "saliency_gamma": g, "prior_exponent": a, "error": {"10": float(r[0]), "20": float(r[1])}}); print(mode, "saliency gamma", g, "prior exponent", a, np.round(r, 2), flush=True)
json.dump(out, open(os.path.join(ROOT, "foundation", "runs", "landing_uav_prior.json"), "w"), indent=1)
