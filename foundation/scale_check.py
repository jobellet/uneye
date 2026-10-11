#!/usr/bin/env python3
"""Pixels or degrees? (owner: the stream may be in pixels, a degree-trained network must then be used after an approximate conversion estimated from the data themselves.)
A. Tolerance: the original U'n'Eye (weights_1+2+3, trained in degrees) is run on test positions multiplied by a factor c (a wrong unit conversion): event F1 / kappa against c (c = 1: correct).
B. Estimation: from the events of the label-free detector (universal HMM, unit-free, so it works whatever the unit), the amplitude A (displacement across the event) and the duration D (ms) of every event are measured;
   the main sequence says D = d0 + k * A_deg (saccades, d0 ~ 21 ms, k ~ 2.2 ms/deg), so with an unknown scale s (degrees per data unit) the robust slope m of D on A_unit gives s = m / k. We test it on data whose true scale is known
   (s = 1, positions in degrees): the estimate s_hat per dataset and its spread (bootstrap) for 20, 40, 80 events (about 50 s of a session). The peak-velocity / amplitude ratio is scale-invariant and cannot give s: it is reported
   only as a sanity check of the detector (typical 20-60 per second).      python foundation/scale_check.py -> night/scale_check.json
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from scipy.stats import theilslopes
from free_saccade import detectors as D
from foundation import compare as CP, night_eval as NE, universal_hmm as UH, data as FD
from foundation.hmm_score import ScoreHMM, Scores
res = {"tolerance": {}, "estimate": {}}; d = NE.data(); FACTORS = (0.25, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 4.0)
print("A. U'n'Eye (degrees) on positions x c: F1 / kappa")
for k, (A, B) in d.items():
    n = min(len(B), 100); Bs = FD.Set(B.name, B.pos[:n], B.lab[:n], B.coarse); S_te = type("S", (), dict(lab=Bs.lab))(); res["tolerance"][k] = {}
    for c in FACTORS:
        Sc = FD.Set(B.name, (Bs.pos * c).astype(np.float32), Bs.lab, Bs.coarse); r = CP.event_f1(CP.pred_unet(Sc, "weights_1+2+3"), S_te); res["tolerance"][k][str(c)] = [float(r["ev_f1"]), float(r["kappa"])]
    print(k, " ".join(f"x{c}: {v[0]:.2f}/{v[1]:.2f}" for c, v in res["tolerance"][k].items()), flush=True)
print("B. scale estimated from the main sequence (true scale = 1 deg per unit): median [10-90 %] of s_hat over bootstrap draws of n events")
uni = {}
for fs in (1000.0, 500.0): f, v = UH.unit_free_score(UH.archive_windows(fs), "detrended", fs); uni[fs] = ScoreHMM(fs).fit(Scores(f, v, fs))
KREF = 2.2
for k, (A, B) in d.items():
    fs = CP.NATIVE[k]; pos = CP.native_pos(B).astype(np.float64); f, v = UH.unit_free_score(CP.native_pos(B), "detrended", fs); hmm = uni[fs].predict(Scores(f, v, fs)); amps, durs, vpk = [], [], []
    for i in range(len(pos)):
        p, _ = D.fill(pos[i:i + 1]); p = p[0]; sp = np.hypot(*(np.gradient(p, axis=0) * fs).T)
        for s, e in D.runs_1d(hmm[i]):
            if s < 2 or e > len(p) - 3: continue
            a = float(np.hypot(*(p[e + 1] - p[s - 1]))); dur = (e - s + 1) * 1000.0 / fs
            if 0.1 <= a <= 25 and 3 <= dur <= 150: amps.append(a); durs.append(dur); vpk.append(float(sp[s:e + 1].max()))
    amps, durs, vpk = map(np.array, (amps, durs, vpk)); res["estimate"][k] = dict(n_events=int(len(amps)), amp_median=float(np.median(amps)), amp_p10_p90=[float(np.percentile(amps, 10)), float(np.percentile(amps, 90))], dur_median=float(np.median(durs)),
                                                                                 v_over_a_median=float(np.median(vpk / amps)))
    m, b0, lo, hi = theilslopes(durs, amps); res["estimate"][k].update(slope_ms_per_deg=float(m), intercept_ms=float(b0))
    rng = np.random.RandomState(0)
    for ne in (20, 40, 80):
        sh = []
        for _ in range(200):
            idx = rng.choice(len(amps), min(ne, len(amps)), replace=True)
            if np.ptp(amps[idx]) < 1e-6: continue
            sh.append(theilslopes(durs[idx], amps[idx])[0] / KREF)
        res["estimate"][k][f"s_hat_n{ne}"] = [float(np.median(sh)), float(np.percentile(sh, 10)), float(np.percentile(sh, 90))]
    e = res["estimate"][k]; print(f"{k:10s} events {e['n_events']}, A median {e['amp_median']:.2f} deg (10-90 %: {e['amp_p10_p90'][0]:.2f}-{e['amp_p10_p90'][1]:.2f}), D median {e['dur_median']:.0f} ms, V/A {e['v_over_a_median']:.0f} /s, slope {e['slope_ms_per_deg']:.1f} ms/deg, intercept {e['intercept_ms']:.1f} ms | "
          + " | ".join(f"n={n}: {e[f's_hat_n{n}'][0]:.2f} [{e[f's_hat_n{n}'][1]:.2f}-{e[f's_hat_n{n}'][2]:.2f}]" for n in (20, 40, 80)), flush=True)
json.dump(res, open(os.path.join(NE.NIGHT, "scale_check.json"), "w"), indent=1)
