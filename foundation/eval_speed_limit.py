"""Effect of the speed limit of the network input (online/common.py::velocity, uneye.functions.clip_speed) on the ORIGINAL U'n'Eye weights.

The shipped weights (training/weights_*) were trained WITHOUT a limit. This evaluates them with the limit off and at 1300, 2000, 2500 deg/s on the
test subsets of docs/BENCHMARK_SUMMARY.md (300 trials of set B of d1-d4, 60 trials of the two-coder Andersson set; same scorer, foundation/compare.py).
It is NOT a re-evaluation of the new pipeline: that needs the network retrained with the limit (see the note in docs/BENCHMARK_SUMMARY.md).
Writes docs/figs_data/uneye_speed_limit_inference_only.json.     python foundation/eval_speed_limit.py
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "online")]
import numpy as np
from foundation import compare as CP, night_eval as NE, data as FD
import common

LIMITS = (("off", 0.0), ("1300", 1300.0), ("2000", 2000.0), ("2500", 2500.0))
JOBS = [("d1", "weights_1+2+3", 2), ("d2", "weights_1+2+3", 2), ("d3", "weights_1+2+3", 2), ("d4", "weights_1+2+3", 2),
        ("andersson", "weights_1+2+3", 2), ("andersson", "weights_Andersson", 5)]
out = {}
for ds, w, classes in JOBS:
    if ds == "andersson" and not os.path.isdir(os.path.join(ROOT, "data", "Andersson")): continue
    B = NE.subset(FD.load(ds, "test"), 300 if ds != "andersson" else 60, 5)
    pos = CP.native_pos(B).astype(np.float64); fs = CP.NATIVE[ds]
    v = np.hypot(np.diff(pos[..., 0], axis=1), np.diff(pos[..., 1], axis=1)) * fs
    above = {str(c): int(np.nansum(v > c)) for c in (1000, 1300, 2000, 2500)}
    for lab, ms in LIMITS:
        common.velocity.__defaults__ = (1.5, 1000.0, ms)
        r = CP.event_f1(CP.pred_unet(B, w, classes=classes), B)
        out[f"{ds}|{w}|{lab}"] = dict(f1=round(float(r["ev_f1"]), 4), kappa=round(float(r["kappa"]), 4), recall=round(float(r["ev_recall"]), 4),
                                      precision=round(float(r["ev_precision"]), 4), n_trials=len(B.pos), samples_above_deg_s=above)
        print(ds, w, lab, out[f"{ds}|{w}|{lab}"], flush=True)
os.makedirs(os.path.join(ROOT, "docs", "figs_data"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "docs", "figs_data", "uneye_speed_limit_inference_only.json"), "w"), indent=1)
