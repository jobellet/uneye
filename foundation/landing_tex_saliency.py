#!/usr/bin/env python3
"""Do saliency candidates help the landing estimate? GazeBase TEX task (reading a poem, 1 kHz, EyeLink 1000; the same page is shown to everybody for a given round and session,
screenshots in data/GazeBase/stim/TEXScreenshots). Candidates = the top-N peaks of the PQFT saliency map of the page (same algorithm as cpp/app/engine/include/eyelive/saliency.hpp,
numpy version; 64 x 64 map), converted to degrees with the display geometry of Pixel2DVAConv.txt. Saccades: EyeLink parser (lab == 2), >= 10 ms, amplitude >= 0.5 deg.
At tau = 10 and 20 ms after the onset, landing estimate from: the tables alone; the candidate selector (cpp/app/engine/include/eyelive/candidates.hpp, same parameters and amplitude rule);
the same selector with N RANDOM points over the whole screen; with N random points inside the text block. Also the share of saccades with a candidate within 1 deg of the landing point.
  python foundation/landing_tex_saliency.py [--rounds 3] [--per-file 150]
"""
import argparse, io, os, re, sys, zipfile
import numpy as np, pandas as pd
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, DW, DH, VOFF, DIST = 1680, 1050, 474.0, 297.0, 36.0, 550.0
def px2deg(xp, yp):
    x = np.degrees(np.arctan((xp * DW / W - DW / 2) / DIST)); y = np.degrees(np.arctan(((H - yp) * DH / H - (DH / 2 + VOFF)) / DIST)); return x, y
def pqft(img, n=64, sigma=3.0):
    g = np.asarray(img.convert("RGB").resize((n, n), Image.BOX), np.float64) / 255.0; r, gg, b = g[..., 0], g[..., 1], g[..., 2]
    I = (r + gg + b) / 3; RG = r - gg; BY = b - 0.5 * (r + gg); M = np.zeros_like(I)
    F1 = np.fft.fft2(M + 1j * RG); F2 = np.fft.fft2(BY + 1j * I); mod = np.sqrt(np.abs(F1) ** 2 + np.abs(F2) ** 2) + 1e-9
    s = np.abs(np.fft.ifft2(F1 / mod)) ** 2 + np.abs(np.fft.ifft2(F2 / mod)) ** 2
    from scipy.ndimage import gaussian_filter
    s = gaussian_filter(s, sigma, mode="nearest"); return s / s.max()
def peaks(m, N, rel=0.3, rad=2):
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(m, size=2 * rad + 1); ys, xs = np.where((m == mx) & (m >= rel * m.max())); v = m[ys, xs]; o = np.argsort(-v)[:N]; return xs[o], ys[o], v[o]
def cand_deg(xs, ys, n=64):
    return np.stack(px2deg((xs + 0.5) * W / n, (ys + 0.5) * H / n), 1)
# selector (candidates.hpp)
S_TH0, D0, S_A, A_MIN = 0.12, 0.5, 1.0, 0.3
def pick(p0, cur, rem, C, w):
    d = np.linalg.norm(cur - p0)
    if d < 1e-9 or len(C) == 0: return -1
    u = (cur - p0) / d; A = max(d + max(rem, 0.0), A_MIN); cv = C - p0; cn = np.maximum(np.linalg.norm(cv, axis=1), 1e-6)
    th = np.arccos(np.clip(cv @ u / cn, -1, 1)); sth = S_TH0 / (1 + d / D0)
    sc = w * np.exp(-th ** 2 / (2 * sth ** 2)) * np.exp(-(np.log(cn) - np.log(A)) ** 2 / (2 * S_A ** 2)) + 1e-12
    return int(sc.argmax())
def rule(p0, c, tab, ms):
    g = p0 + 0.95 * (c - p0); w = 1.0 if ms < 30 else (0.65 if ms < 45 else 0.43); return tab + w * (g - tab)
txt = open(os.path.join(ROOT, "cpp", "include", "uneye", "gaze_tables.hpp")).read()
blk = re.search(r"kLandCoef\[[^=]*=\s*\{(.*?)\n\};", txt, re.S).group(1)
COEF = np.array([[float(v) for v in m.split(",")] for m in re.findall(r"\{([-\d.e+, ]+)\}", blk)]).reshape(3, 30, 3)[0]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--per-file", type=int, default=150); ap.add_argument("--N", default="8,16,32"); a = ap.parse_args()
    Ns = [int(v) for v in a.N.split(",")]; rng = np.random.RandomState(0); stim = os.path.join(ROOT, "data", "GazeBase", "stim", "TEXScreenshots"); maps, boxes, cands = {}, {}, {}
    for r in range(1, 10):
        for s in (1, 2):
            im = Image.open(os.path.join(stim, f"TEX_R{r}S{s}_bg.png")); m = pqft(im); maps[(r, s)] = m
            gray = np.asarray(im.convert("L")); ys, xs = np.where(gray > 60); x0, x1 = px2deg(np.array([xs.min(), xs.max()], float), np.array([ys.max(), ys.min()], float)); boxes[(r, s)] = (x0[0], x0[1], x1[0], x1[1])   # xmin, xmax, ymin, ymax in deg
            cands[(r, s)] = {N: cand_deg(*peaks(m, N)[:2]) for N in Ns}
    print("candidates per page (N=16):", len(cands[(1, 1)][16]), "box deg", np.round(boxes[(1, 1)], 1))
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "GazeBase", "GazeBase_v2_0.zip")); names = [n for n in z.namelist() if n.endswith(".zip") and "Subject_" in n and int(re.search(r"Round_(\d+)", n).group(1)) <= a.rounds]
    rows = []                                                   # per saccade at tau: (tau, N, method, error, covered)
    for ni, name in enumerate(names):
        rnd = int(re.search(r"Round_(\d+)", name).group(1)); inner = zipfile.ZipFile(io.BytesIO(z.read(name)))
        for fn in inner.namelist():
            if not fn.endswith("_TEX.csv"): continue
            ses = int(re.search(r"_S(\d)_TEX", fn).group(1)); d = pd.read_csv(inner.open(fn)); x, y, val, lab = (d[c].values for c in ("x", "y", "val", "lab")); n = len(x)
            sac = np.diff(np.r_[0, (lab == 2).astype(np.int8), 0]); ons = np.where(sac == 1)[0]; offs = np.where(sac == -1)[0] - 1; take = np.arange(len(ons)); rng.shuffle(take)
            spd = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]; cnt = 0
            for k in take:
                on, off = ons[k], offs[k]
                if off - on + 1 < 25 or on < 5 or off + 5 >= n or (val[on - 1:off + 2] != 0).any() or not np.isfinite(x[on - 1:off + 2]).all(): continue
                p0 = np.array([x[on - 1], y[on - 1]]); land = np.array([x[off], y[off]])
                if np.linalg.norm(land - p0) < 0.5: continue
                cnt += 1
                if cnt > a.per_file: break
                for tau in (10, 20):
                    t = on + tau; cur = np.array([x[t], y[t]]); v = spd[t - 2:t + 1].mean(); vmax = spd[on:t + 1].max(); dd = np.linalg.norm(cur - p0)
                    if dd < 0.02: continue
                    u = (cur - p0) / dd; rem = max(float(COEF[min(tau // 2, 29)] @ np.array([dd, v, vmax])), 0.0); tab = cur + rem * u
                    rows.append((tau, 0, "tables", np.linalg.norm(tab - land), np.nan, 0.0, np.linalg.norm(tab - land), max(dd + rem, 0.3)))
                    bx0, bx1, by0, by1 = boxes[(rnd, ses)]
                    for N in Ns:
                        sets = {"saliency": cands[(rnd, ses)][N], "random screen": np.c_[rng.uniform(-HX, HX, N), rng.uniform(-HY, HY, N)] if False else np.c_[rng.uniform(px2deg(0, 0)[0], px2deg(W, 0)[0], N), rng.uniform(px2deg(0, H)[1], px2deg(0, 0)[1], N)],
                                "random text box": np.c_[rng.uniform(bx0, bx1, N), rng.uniform(by0, by1, N)]}
                        for mname, C in sets.items():
                            j = pick(p0, cur, rem, C, np.ones(len(C))); est = rule(p0, C[j], tab, tau) if j >= 0 else tab
                            rows.append((tau, N, mname, np.linalg.norm(est - land), float(np.linalg.norm(C - land, axis=1).min() < 1.0), np.linalg.norm(est - tab), np.linalg.norm(tab - land), max(dd + rem, 0.3)))
        if ni % 60 == 0: print(ni, len(rows), flush=True)
    df = pd.DataFrame(rows, columns=["tau", "N", "method", "err", "covered", "gap", "err_tab", "A"]); print("saccade estimates:", len(df[df.method == "tables"]) // 2)
    for tau in (10, 20):
        base = df[(df.tau == tau) & (df.method == "tables")].err.mean(); print(f"\ntau = {tau} ms: tables alone {base:.3f} deg")
        for N in Ns:
            print(f"  N = {N:2d}: " + "   ".join(f"{m} {df[(df.tau == tau) & (df.N == N) & (df.method == m)].err.mean():.3f} (coverage < 1 deg: {100 * df[(df.tau == tau) & (df.N == N) & (df.method == m)].covered.mean():.0f} %)" for m in ("saliency", "random text box", "random screen")))
    df.to_csv(os.path.join(ROOT, "foundation", "runs", "landing_tex_saliency.csv"), index=False)
