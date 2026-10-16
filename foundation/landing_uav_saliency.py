#!/usr/bin/env python3
"""Saliency candidates on real videos: EyeTrackUAV2 gaze (1 kHz, EyeLink 1000 Plus, remote mode, 30 participants free viewing) on the 9 DTB70 sequences it uses,
frames from DTB70 (data/DTB70/<seq>/img, 1280 x 720, 30 fps; groundtruth_rect.txt = the tracked object per frame). Display (measured from the two coordinate files of the dataset):
the video is shown at native size centred on a 1920 x 1080 screen of 527.04 x 296.46 mm at 1 m -> 0.0157 deg per pixel.
Saccades: dominant eye, speed (5-sample central difference) > 40 deg/s for >= 10 ms, peak >= 80 deg/s, amplitude >= 1 deg, no missing sample from 30 ms before to 10 ms after.
Candidates at the onset frame: PQFT saliency peaks (N = 16; motion channel from the previous frame), the tracked object's centre, both, or 16 random points. Selector, amplitude rule and
tables as in the C++ engine (candidates.hpp). Error of the landing estimate at tau = 10 and 20 ms after the detected onset.   -> foundation/runs/landing_uav_saliency.json
"""
import glob, json, os, re, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from foundation.landing_tex_saliency import pick, rule, COEF
SEQS = ["Basketball", "Girl1", "Girl2", "ManRunning1", "ManRunning2", "Soccer1", "Soccer2", "StreetBasketball1", "Walking"]
MMX, MMY, D = 527.04 / 1920, 296.46 / 1080, 1000.0
def f2deg(x, y):   # frame pixel -> degrees (screen centre = 0, y up)
    return np.degrees(np.arctan((np.asarray(x, float) + 320 - 960) * MMX / D)), np.degrees(np.arctan((540 - (np.asarray(y, float) + 180)) * MMY / D))
def pqft(rgb, prev, n=64, sigma=3.0):
    g = rgb.astype(np.float64) / 255.0; r, gg, b = g[..., 0], g[..., 1], g[..., 2]; I = (r + gg + b) / 3
    M = np.zeros_like(I) if prev is None else np.abs(I - prev)
    F1 = np.fft.fft2(M + 1j * (r - gg)); F2 = np.fft.fft2((b - 0.5 * (r + gg)) + 1j * I); mod = np.sqrt(np.abs(F1) ** 2 + np.abs(F2) ** 2) + 1e-9
    s = gaussian_filter(np.abs(np.fft.ifft2(F1 / mod)) ** 2 + np.abs(np.fft.ifft2(F2 / mod)) ** 2, sigma, mode="nearest"); return s / s.max(), I
def peaks(m, N=16, rel=0.3, rad=2):
    mx = maximum_filter(m, size=2 * rad + 1); ys, xs = np.where((m == mx) & (m >= rel * m.max())); o = np.argsort(-m[ys, xs])[:N]; return xs[o], ys[o]
rng = np.random.RandomState(0); res = {k: {10: [], 20: []} for k in ("tables", "saliency", "object", "object + saliency", "random 16")}; cover = {k: [] for k in ("saliency", "object", "random 16")}; nsac = 0
for seq in SEQS:
    files = sorted(glob.glob(os.path.join(ROOT, "data", "DTB70", seq, "img", "*.jpg"))); gt = np.loadtxt(os.path.join(ROOT, "data", "DTB70", seq, "groundtruth_rect.txt"), delimiter=",")
    sal, obj, prev = [], [], None
    for k, f in enumerate(files):
        im = np.asarray(Image.open(f).convert("RGB").resize((64, 64), Image.BOX)); m, prev = pqft(im, prev); xs, ys = peaks(m)
        sal.append(np.stack(f2deg((xs + 0.5) * 1280 / 64, (ys + 0.5) * 720 / 64), 1)); x, y, w, h = gt[min(k, len(gt) - 1)]; obj.append(np.array(f2deg(x + w / 2, y + h / 2)))
    for fn in sorted(glob.glob(os.path.join(ROOT, "data", "EyeTrackUAV2", "RAW_FV", seq, "*.csv"))):
        a = np.loadtxt(fn, delimiter=","); fr = a[:, 0].astype(int); x, y = f2deg(a[:, 11], a[:, 12]); bad = ~np.isfinite(a[:, 11]) | (a[:, 11] <= 0) | (a[:, 12] <= 0)
        x[bad] = np.nan; y[bad] = np.nan; p = np.c_[x, y]; sp = np.full(len(p), np.nan); sp[2:-2] = np.hypot(*((p[4:] + p[3:-1] - p[1:-3] - p[:-4]) / 6.0).T) * 1000   # deg/s
        fast = np.nan_to_num(sp) > 40; dd = np.diff(np.r_[0, fast.astype(int), 0]); ons, offs = np.where(dd == 1)[0], np.where(dd == -1)[0] - 1
        for on, off in zip(ons, offs):
            if off - on + 1 < 10 or on < 30 or off + 10 >= len(p) or np.isnan(p[on - 30:off + 10]).any() or np.nanmax(sp[on:off + 1]) < 80: continue
            p0, land = p[on - 1], p[off]
            if np.linalg.norm(land - p0) < 1.0: continue
            f0 = min(max(fr[on] - 1, 0), len(files) - 1); nsac += 1; sp1 = np.hypot(*np.diff(p, axis=0).T); sp1 = np.r_[0, sp1]
            S = sal[f0]; O = obj[f0][None]; R = np.stack(f2deg(rng.uniform(0, 1280, 16), rng.uniform(0, 720, 16)), 1)
            cover["saliency"].append(np.linalg.norm(S - land, axis=1).min()); cover["object"].append(np.linalg.norm(O - land, axis=1).min()); cover["random 16"].append(np.linalg.norm(R - land, axis=1).min())
            for tau in (10, 20):
                t = on + tau
                if t > off + 5: continue
                cur = p[t]; d = np.linalg.norm(cur - p0)
                if d < 0.02: continue
                u = (cur - p0) / d; v = sp1[t - 2:t + 1].mean(); vmax = sp1[on:t + 1].max(); rem = max(float(COEF[min(tau // 2, 29)] @ np.array([d, v, vmax])), 0.0); tab = cur + rem * u
                res["tables"][tau].append(np.linalg.norm(tab - land))
                for name, C in (("saliency", S), ("object", O), ("object + saliency", np.r_[O, S]), ("random 16", R)):
                    j = pick(p0, cur, rem, C, np.ones(len(C))); res[name][tau].append(np.linalg.norm((rule(p0, C[j], tab, tau) if j >= 0 else tab) - land))
    print(seq, "saccades so far", nsac, flush=True)
out = {"saccades": nsac, "error_deg": {k: {str(t): float(np.mean(v[t])) for t in (10, 20)} for k, v in res.items()},
       "coverage_1deg": {k: float(np.mean(np.array(v) < 1.0)) for k, v in cover.items()}, "coverage_2deg": {k: float(np.mean(np.array(v) < 2.0)) for k, v in cover.items()}}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(ROOT, "foundation", "runs", "landing_uav_saliency.json"), "w"), indent=1)
