#!/usr/bin/env python3
"""Does compensating the camera motion matter more when the camera moves?  NSS at the landing point of the frame-difference map with and without camera-motion compensation,
split by terciles of the global motion between the onset frame and the previous one (estimated translation, pixels of the 1280 x 720 frame). Same saccades as landing_uav_saliency2.py.
-> foundation/runs/uav_camera_motion_nss.json"""
import glob, os, sys, json
import numpy as np
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from foundation.saliency_models import global_motion, compensated_diff
from foundation.landing_tex_saliency import COEF
from foundation.landing_uav_saliency import SEQS, f2deg, MMX, MMY, D
def deg2f(xd, yd): return np.tan(np.radians(xd)) * D / MMX + 960 - 320, 540 - np.tan(np.radians(yd)) * D / MMY - 180
def sample(m, fx, fy):
    h, w = m.shape; return m[min(max(int(fy / 720 * h), 0), h - 1), min(max(int(fx / 1280 * w), 0), w - 1)]
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
need = {}
for si, f0, p0, land, feats in sacc: need.setdefault(si, set()).add(f0)
res = {}                                                   # (si, f0) -> (translation magnitude at full res, map plain, map compensated)
for si, seq in enumerate(SEQS):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "DTB70", seq, "img", "*.jpg"))); g = lambda f: zoom(np.asarray(Image.open(f).convert("L"), float) / 255, (180 / 720, 320 / 1280), order=1)
    for f0 in sorted(need[si]):
        if f0 < 1: continue
        p, c = g(files[f0 - 1]), g(files[f0]); Mc, A = compensated_diff(p, c); M = np.abs(c - p); mag = float(np.hypot(*(A[:, 2] * 4)) )
        res[(si, f0)] = (mag, gaussian_filter(zoom(M, (64 / 180, 64 / 320), order=1), 3.0), gaussian_filter(zoom(Mc, (64 / 180, 64 / 320), order=1), 3.0))
    print(seq, len(res), flush=True)
rows = []
for si, f0, p0, land, feats in sacc:
    if (si, f0) not in res: continue
    mag, m1, m2 = res[(si, f0)]; lx, ly = deg2f(*land); z = lambda m: (sample(m, lx, ly) - m.mean()) / (m.std() + 1e-12); rows.append((mag, z(m1), z(m2)))
R = np.array(rows); q = np.quantile(R[:, 0], [1 / 3, 2 / 3]); out = {"n": len(R), "translation_px_terciles": [float(v) for v in q], "all": {"plain": float(R[:, 1].mean()), "compensated": float(R[:, 2].mean())}}
for name, m in (("small camera motion", R[:, 0] <= q[0]), ("medium", (R[:, 0] > q[0]) & (R[:, 0] <= q[1])), ("large", R[:, 0] > q[1])):
    out[name] = {"n": int(m.sum()), "mean_translation_px": float(R[m, 0].mean()), "plain": float(R[m, 1].mean()), "compensated": float(R[m, 2].mean())}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(ROOT, "foundation", "runs", "uav_camera_motion_nss.json"), "w"), indent=1)
