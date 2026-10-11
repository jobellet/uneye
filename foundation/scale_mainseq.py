#!/usr/bin/env python3
"""Pixel-to-degree scale from the duration-amplitude relation LEARNED from the labeled datasets (owner's idea). The textbook slope (2.2 ms/deg) failed (docs/EYELIVE_ROADMAP.md 12.8): the relation is not linear for small
saccades. Here the reference curves (median log duration and log peak speed per log-amplitude bin) come from the saccades of the OTHER datasets (leave-one-dataset-out), either from their human labels (R1) or from the events of
the label-free detector (R2, same measurement as at run time). For an unknown scale c (degrees per unit) the events of the session (A_unit, D, V_unit) are compared with the curve at (c A_unit): the c with the smallest median
log-distance is chosen. D alone, or D + peak speed (the speed / amplitude ratio is scale-free, but the speed itself is not).  Error factor = c_hat / c_true (1 = exact), for 20, 40, 80 events (about 50 s) and all events.
python foundation/scale_mainseq.py -> night/scale_mainseq.json
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from free_saccade import detectors as D
from foundation import compare as CP, night_eval as NE, universal_hmm as UH, accel_boundaries as AB
from foundation.hmm_score import ScoreHMM, Scores

uni = {}
for fs in (1000.0, 500.0): f, v = UH.unit_free_score(UH.archive_windows(fs), "detrended", fs); uni[fs] = ScoreHMM(fs).fit(Scores(f, v, fs))

def measure(p, runs, fs):
    out = []
    for s, e in runs:
        if s < 2 or e > len(p) - 3 or not np.isfinite(p[s - 1:e + 2]).all(): continue
        a = float(np.hypot(*(p[e + 1] - p[s - 1]))); d = (e - s + 1) * 1000.0 / fs; v = float(np.hypot(*(np.gradient(p[s:e + 1], axis=0) * fs).T).max())
        if 0.1 <= a <= 25 and 3 <= d <= 150 and v > 0: out.append((a, d, v))
    return out

def events(S, k, source):
    """-> (n_events, 3) array of (amplitude deg, duration ms, peak speed deg/s); source 'detector' (native rate, HMM + a-priori rule) or 'human' (labels at 1 kHz)"""
    rows = []
    if source == "human":
        for i in range(len(S)): p, _ = D.fill(S.pos[i:i + 1].astype(np.float64)); rows += measure(p[0], D.runs_1d(S.lab[i] == 1), 1000.0)
    else:
        fs = CP.NATIVE[k]; pos = CP.native_pos(S).astype(np.float64); f, v = UH.unit_free_score(CP.native_pos(S), "detrended", fs); hmm = uni[fs].predict(Scores(f, v, fs))
        m = AB.refine(hmm, pos, fs, "both", 15.0, offset_rule="first" if k == "andersson" else "last")
        for i in range(len(pos)): p, _ = D.fill(pos[i:i + 1]); rows += measure(p[0], D.runs_1d(m[i]), fs)
    return np.array(rows)

def curve(ev, nb=0.15, lo=-1.0, hi=1.4, minn=12):
    la = np.log10(ev[:, 0]); edges = np.arange(lo, hi + 1e-9, nb); xs, yd, yv = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (la >= a) & (la < b)
        if m.sum() >= minn: xs.append(0.5 * (a + b)); yd.append(np.median(np.log10(ev[m, 1]))); yv.append(np.median(np.log10(ev[m, 2])))
    return np.array(xs), np.array(yd), np.array(yv)

def estimate(ev, ref, use_speed, grid=np.logspace(-2.7, 0.7, 200)):
    xs, yd, yv = ref; best, bc = 1e9, 1.0; la = np.log10(ev[:, 0]); ld = np.log10(ev[:, 1]); lv = np.log10(ev[:, 2])
    for c in grid:
        x = la + np.log10(c)
        if np.mean((x >= xs[0] - 0.1) & (x <= xs[-1] + 0.1)) < 0.5: continue                              # candidate puts most events outside the reference range
        r = np.abs(ld - np.interp(x, xs, yd)); r = r + (0.5 * np.abs(lv + np.log10(c) - np.interp(x, xs, yv)) if use_speed else 0)
        sc = np.median(r)
        if sc < best: best, bc = sc, c
    return bc

res = {}; data = NE.data(); names = list(data)
ev_det = {k: events(data[k][1], k, "detector") for k in names}                      # target events: test subset, detector
ref_src = {src: {k: events(data[k][0], k, src) for k in names} for src in ("human", "detector")}   # references: train subsets of the other datasets
print("events per dataset (detector / human reference):", {k: (len(ev_det[k]), len(ref_src['human'][k])) for k in names}, flush=True)
rng = np.random.RandomState(0)
for k in names:
    res[k] = {}
    for src in ("human", "detector"):
        ref_ev = np.concatenate([ref_src[src][o] for o in names if o != k]); ref = curve(ref_ev)
        for use_speed in (False, True):
            tag = f"{src}_{'D+V' if use_speed else 'D'}"; out = {}
            out["all"] = float(estimate(ev_det[k], ref, use_speed))
            for n in (20, 40, 80):
                vals = [estimate(ev_det[k][rng.choice(len(ev_det[k]), min(n, len(ev_det[k])), replace=True)], ref, use_speed) for _ in range(60)]
                out[f"n{n}"] = [float(np.median(vals)), float(np.percentile(vals, 10)), float(np.percentile(vals, 90))]
            res[k][tag] = out
    print(f"{k:10s} " + " | ".join(f"{t}: all {res[k][t]['all']:.2f}, n=40 {res[k][t]['n40'][0]:.2f} [{res[k][t]['n40'][1]:.2f}-{res[k][t]['n40'][2]:.2f}]" for t in ("human_D", "human_D+V", "detector_D", "detector_D+V")), flush=True)
json.dump(res, open(os.path.join(NE.NIGHT, "scale_mainseq.json"), "w"), indent=1)
