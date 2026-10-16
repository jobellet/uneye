"""Shared pieces of the UAV tests: saccade detection on EyeTrackUAV2 (same rules as foundation/landing_uav_saliency.py) and the frame <-> degrees geometry."""
import glob, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from foundation.landing_tex_saliency import COEF, pick, rule
SEQS = ["Basketball", "Girl1", "Girl2", "ManRunning1", "ManRunning2", "Soccer1", "Soccer2", "StreetBasketball1", "Walking"]
MMX, MMY, D = 527.04 / 1920, 296.46 / 1080, 1000.0
def f2deg(x, y):   # frame pixel -> degrees (screen centre = 0, y up); the 1280 x 720 video is centred on the 1920 x 1080 screen (offset 320, 180)
    return np.degrees(np.arctan((np.asarray(x, float) + 320 - 960) * MMX / D)), np.degrees(np.arctan((540 - (np.asarray(y, float) + 180)) * MMY / D))
def peaks(m, N=16, rel=0.3, rad=2):
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(m, size=2 * rad + 1); ys, xs = np.where((m == mx) & (m >= rel * m.max())); o = np.argsort(-m[ys, xs])[:N]; return xs[o], ys[o]
def deg2f(xd, yd): return np.tan(np.radians(xd)) * D / MMX + 960 - 320, 540 - np.tan(np.radians(yd)) * D / MMY - 180
def detect_saccades(with_subject=False):
    """list of (seq index, onset frame index (0-based), p0 deg, landing deg, {tau: (cur, remaining by the tables, tables' landing)})"""
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
                row = (si, min(max(fr[on] - 1, 0), nf - 1), p0, land, feats)
                sacc.append(row + (int(re.search(r"subj_(\d+)", fn).group(1)),) if with_subject else row)
    return sacc
