#!/usr/bin/env python3
"""Exports for the C++ EyeLive application (cpp/app):
 (1) cpp/models/universal_hmm.txt   the label-free universal minimum-duration HMM (fitted on archive/ only, foundation/universal_hmm.py), one line per sampling rate:  fs d mu0 mu1 sd0 sd1 p_on p_stay
 (2) cpp/app/tests/golden_hmm_accel.txt   reference traces with the events of the HMM and the events refined by the acceleration-zero rule (foundation/accel_boundaries.py), for tests/test_detect_parity.cpp
python cpp/scripts/export_hmm.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from free_saccade import detectors as D
from foundation import compare as CP, night_eval as NE, universal_hmm as UH, accel_boundaries as AB
from foundation.hmm_score import ScoreHMM, Scores

uni = {}
for fs in (1000.0, 500.0):
    f, v = UH.unit_free_score(UH.archive_windows(fs), "detrended", fs); uni[fs] = ScoreHMM(fs).fit(Scores(f, v, fs))
with open(os.path.join(ROOT, "cpp", "models", "universal_hmm.txt"), "w") as o:
    o.write("# fs d mu0 mu1 sd0 sd1 p_on p_stay   (score = log10(1 + |detrended velocity| / robust noise); state 0 fixation, states 1..d the saccade chain)\n")
    for fs, h in uni.items(): o.write(f"{fs:g} {h.d} {h.mu[0]:.9g} {h.mu[1]:.9g} {h.sd[0]:.9g} {h.sd[1]:.9g} {h.p_on:.9g} {h.p_stay:.9g}\n")
print(open(os.path.join(ROOT, "cpp", "models", "universal_hmm.txt")).read())

d = NE.data(); out = []
for name, k, idx in (("d1_1000Hz", "d1", [0, 1, 2]), ("andersson_500Hz", "andersson", [0, 1])):
    B = d[k][1]; fs = CP.NATIVE[k]
    for i in idx:
        pos = CP.native_pos(B)[i:i + 1].astype(np.float64); T = pos.shape[1]
        f, v = UH.unit_free_score(pos.astype(np.float32), "detrended", fs); hmm = uni[fs].predict(Scores(f, v, fs))
        first = AB.refine(hmm, pos, fs, "both", 15.0, offset_rule="first"); last = AB.refine(hmm, pos, fs, "both", 15.0, offset_rule="last")
        ev = lambda m: [(int(a), int(b)) for a, b in D.runs_1d(m[0])]
        p = np.where(np.isfinite(pos[0]), pos[0], np.nan)
        out.append((name, i, fs, p, ev(hmm), ev(first), ev(last)))
with open(os.path.join(ROOT, "cpp", "app", "tests", "golden_hmm_accel.txt"), "w") as o:
    o.write(f"{len(out)}\n")
    for name, i, fs, p, a, b, c in out:
        o.write(f"{name}_{i} {fs:g} {len(p)}\n"); o.write(" ".join("nan" if not np.isfinite(x) else f"{x:.9g}" for x in p.reshape(-1)) + "\n")
        for evs in (a, b, c): o.write(f"{len(evs)} " + " ".join(f"{s} {e}" for s, e in evs) + "\n")
        print(name, i, "samples", len(p), "events hmm/first/last", len(a), len(b), len(c))
