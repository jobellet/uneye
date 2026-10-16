#!/usr/bin/env python3
"""Several saliency models on real videos (EyeTrackUAV2 gaze + DTB70 frames, same saccades and geometry as foundation/landing_uav_saliency.py).
Per model: NSS at the landing point (saliency at the landing point, in standard deviations of the map; 0 = chance), share of landings within 1 / 2 deg of one of the 16 peaks,
and landing error at tau = 10 / 20 ms with (a) the 16 peaks as candidates (forced choice), (b) a dense grid of 32 x 18 candidates weighted by the map (prior), (c) the same grid with
flat weights (control: no saliency at all), and the tables alone.   -> foundation/runs/landing_uav_saliency2.json
"""
import glob, json, os, sys, time
import numpy as np
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from foundation.landing_tex_saliency import pick, rule, COEF
from foundation.landing_uav_saliency import SEQS, f2deg, peaks
from foundation.saliency_models import Model
MODELS = ["pqft", "sr", "itti", "itti_dyn", "motion", "center"]
GX, GY = np.meshgrid((np.arange(32) + 0.5) * 1280 / 32, (np.arange(18) + 0.5) * 720 / 18); GRID = np.stack(f2deg(GX.ravel(), GY.ravel()), 1)
def sample(m, fx, fy):   # map value at frame pixel (fx, fy)
    h, w = m.shape; return m[min(max(int(fy / 720 * h), 0), h - 1), min(max(int(fx / 1280 * w), 0), w - 1)]
def deg2f(xd, yd):
    from foundation.landing_uav_saliency import MMX, MMY, D
    return np.tan(np.radians(xd)) * D / MMX + 960 - 320, 540 - np.tan(np.radians(yd)) * D / MMY - 180
# saccades (same detection as landing_uav_saliency.py)
sacc = []
for si, seq in enumerate(SEQS):
    nf = len(glob.glob(os.path.join(ROOT, "data", "DTB70", seq, "img", "*.jpg")))
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "EyeTrackUAV2", "RAW_FV", seq, "*.csv"))):
        a = np.loadtxt(fn, delimiter=","); fr = a[:, 0].astype(int); x, y = f2deg(a[:, 11], a[:, 12]); bad = ~np.isfinite(a[:, 11]) | (a[:, 11] <= 0) | (a[:, 12] <= 0); x[bad] = np.nan; y[bad] = np.nan
        p = np.c_[x, y]; sp = np.full(len(p), np.nan); sp[2:-2] = np.hypot(*((p[4:] + p[3:-1] - p[1:-3] - p[:-4]) / 6.0).T) * 1000; sp1 = np.r_[0, np.hypot(*np.diff(p, axis=0).T)]
        fast = np.nan_to_num(sp) > 40; dd = np.diff(np.r_[0, fast.astype(int), 0])
        for on, off in zip(np.where(dd == 1)[0], np.where(dd == -1)[0] - 1):
            if off - on + 1 < 10 or on < 30 or off + 10 >= len(p) or np.isnan(p[on - 30:off + 10]).any() or np.nanmax(sp[on:off + 1]) < 80: continue
            p0, land = p[on - 1], p[off]
            if np.linalg.norm(land - p0) < 1.0: continue
            feats = {}
            for tau in (10, 20):
                t = on + tau
                if t > off + 5: continue
                cur = p[t]; d = np.linalg.norm(cur - p0)
                if d < 0.02: continue
                u = (cur - p0) / d; v = sp1[t - 2:t + 1].mean(); vmax = sp1[on:t + 1].max(); rem = max(float(COEF[min(tau // 2, 29)] @ np.array([d, v, vmax])), 0.0); feats[tau] = (cur, rem, cur + rem * u)
            sacc.append((si, min(max(fr[on] - 1, 0), nf - 1), p0, land, feats))
print("saccades", len(sacc), flush=True)
out = {"saccades": len(sacc), "models": {}}
tab = {tau: np.mean([np.linalg.norm(s[4][tau][2] - s[3]) for s in sacc if tau in s[4]]) for tau in (10, 20)}
flat = {tau: [] for tau in (10, 20)}
for s in sacc:
    for tau, (cur, rem, tb) in s[4].items():
        j = pick(s[2], cur, rem, GRID, np.ones(len(GRID))); flat[tau].append(np.linalg.norm(rule(s[2], GRID[j], tb, tau) - s[3]))
out["tables"] = {str(t): float(tab[t]) for t in (10, 20)}; out["flat grid"] = {str(t): float(np.mean(flat[t])) for t in (10, 20)}
print("tables", out["tables"], " flat grid", out["flat grid"], flush=True)
for mname in MODELS:
    t0 = time.time(); maps = []
    for seq in SEQS:
        mod = Model(mname); ms = []
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "DTB70", seq, "img", "*.jpg"))): ms.append(mod(np.asarray(Image.open(f).convert("RGB"))))
        maps.append(ms)
    tmap = (time.time() - t0) / sum(len(m) for m in maps)
    nss, c1, c2, ep, eg = [], [], [], {10: [], 20: []}, {10: [], 20: []}
    for si, f0, p0, land, feats in sacc:
        m = maps[si][f0]; lx, ly = deg2f(*land); z = (sample(m, lx, ly) - m.mean()) / (m.std() + 1e-12); nss.append(z)
        xs, ys = peaks(m); h, w = m.shape; P = np.stack(f2deg((xs + 0.5) * 1280 / w, (ys + 0.5) * 720 / h), 1); dmin = np.linalg.norm(P - land, axis=1).min(); c1.append(dmin < 1); c2.append(dmin < 2)
        wg = np.array([sample(m, gx, gy) for gx, gy in zip(GX.ravel(), GY.ravel())]) + 1e-3
        for tau, (cur, rem, tb) in feats.items():
            j = pick(p0, cur, rem, P, np.ones(len(P))); ep[tau].append(np.linalg.norm(rule(p0, P[j], tb, tau) - land))
            j = pick(p0, cur, rem, GRID, wg); eg[tau].append(np.linalg.norm(rule(p0, GRID[j], tb, tau) - land))
    out["models"][mname] = {"ms_per_frame": round(1000 * tmap, 1), "NSS": float(np.mean(nss)), "peak_within_1deg": float(np.mean(c1)), "peak_within_2deg": float(np.mean(c2)),
                            "peaks_error": {str(t): float(np.mean(ep[t])) for t in (10, 20)}, "weighted_grid_error": {str(t): float(np.mean(eg[t])) for t in (10, 20)}}
    print(mname, json.dumps(out["models"][mname]), flush=True)
json.dump(out, open(os.path.join(ROOT, "foundation", "runs", "landing_uav_saliency2.json"), "w"), indent=1)
