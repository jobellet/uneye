#!/usr/bin/env python3
"""Choosing the unit conversion WITHOUT labels, from the session itself (owner's idea: the detection quality is judged autonomously). The data are used as they are (degrees here, a stand-in for
an unknown unit): for every candidate scale c (positions x c) the degree-trained network (weights_1+2+3) is run and its events are compared with the label-free pseudo-labels of the unit-free
HMM + acceleration-zero rule on the SAME trials (event F1 / kappa between the two detectors, no human label). The c with the best agreement is chosen. We then report what the human-label score would have been
at the chosen c, at c = 1 and at the best c in hindsight.  Data: night_eval test subsets (100 trials), 3 independent random subsets of 50 trials to see the stability of the choice.
python foundation/scale_select.py -> night/scale_select.json
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from foundation import compare as CP, night_eval as NE, universal_hmm as UH, accel_boundaries as AB, data as FD
from foundation.hmm_score import ScoreHMM, Scores
FACT = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0); out = {}; uni = {}
for fs in (1000.0, 500.0): f, v = UH.unit_free_score(UH.archive_windows(fs), "detrended", fs); uni[fs] = ScoreHMM(fs).fit(Scores(f, v, fs))
def pseudo(Ssel, k):
    fs = CP.NATIVE[k]; pos = CP.native_pos(Ssel).astype(np.float64); f, v = UH.unit_free_score(CP.native_pos(Ssel), "detrended", fs); hmm = uni[fs].predict(Scores(f, v, fs))
    return CP.to_1khz(AB.refine(hmm, pos, fs, "both", 15.0, offset_rule="first" if k == "andersson" else "last"), Ssel)
for k, (A, B) in NE.data().items():
    n = min(len(B), 100); Bs = FD.Set(B.name, B.pos[:n], B.lab[:n], B.coarse); ps = pseudo(Bs, k); S_hmm = type("S", (), dict(lab=ps.astype(np.int8)))(); S_hum = type("S", (), dict(lab=Bs.lab))(); preds = {}
    for c in FACT: preds[c] = CP.pred_unet(FD.Set(B.name, (Bs.pos * c).astype(np.float32), Bs.lab, Bs.coarse), "weights_1+2+3")
    hum = {c: CP.event_f1(preds[c], S_hum) for c in FACT}; best_h = max(FACT, key=lambda c: hum[c]["ev_f1"]); res = dict(human={str(c): [float(hum[c]["ev_f1"]), float(hum[c]["kappa"])] for c in FACT}, chosen=[])
    for rep in range(3):
        idx = np.random.RandomState(rep).permutation(n)[:50]; sub = lambda m: m[idx]
        S_sub = type("S", (), dict(lab=ps[idx].astype(np.int8)))(); agree = {c: CP.event_f1(preds[c][idx], S_sub) for c in FACT}; sc = {c: 0.5 * (agree[c]["ev_f1"] + (0 if np.isnan(agree[c]["kappa"]) else agree[c]["kappa"])) for c in FACT}
        ch = max(FACT, key=sc.get); res["chosen"].append(ch)
    res["human_at_chosen"] = [float(np.mean([hum[c]["ev_f1"] for c in res["chosen"]])), float(np.mean([hum[c]["kappa"] for c in res["chosen"]]))]; res["best_c_hindsight"] = best_h; out[k] = res
    print(f"{k:10s} chosen c (3 subsets of 50 trials) {res['chosen']} | human F1/kappa at chosen {res['human_at_chosen'][0]:.2f}/{res['human_at_chosen'][1]:.2f} | at c=1 {hum[1.0]['ev_f1']:.2f}/{hum[1.0]['kappa']:.2f} | best in hindsight c={best_h} {hum[best_h]['ev_f1']:.2f}/{hum[best_h]['kappa']:.2f}", flush=True)
json.dump(out, open(os.path.join(NE.NIGHT, "scale_select.json"), "w"), indent=1)
