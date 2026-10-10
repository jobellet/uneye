#!/usr/bin/env python3
"""Does the unit / normalisation of the data matter? The unlabeled archive/ is z-scored (per file, clipped at +-6, positions not in degrees) and the benchmarks are in degrees, and the pre-training
pools mix both.  (a) invariance of a detector to the transformations that separate the two worlds: an isotropic gain + offset (should change nothing), a PER-AXIS z-score of every trial (x and y scaled by
different factors) and a dataset-level z-score with clipping at +-6 (clipped samples invalid, as in free_saccade/data.py);  (b) what the network actually SEES per source: statistics of the 2-channel input
(arcsinh(v / noise)) of the pre-training pool by source (noise-to-saccade ratio, anisotropy of the two components).   python foundation/unit_check.py -> night/unit_check.json
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from free_saccade import detectors as D
from foundation import bitcn as BT, data as FD, compare as CP, night_eval as NE, conventions as CV, unet_ssl as US
from foundation.dino1d import feats2
BT.FEATS = feats2; out = {}
d = NE.data()
print("(a) LODO BiTCN (trained on other datasets, in degrees) scored on transformed test positions: F1 / kappa")
for k in ("d1", "d4", "andersson"):
    B = d[k][1]; fn = BT.score_fn_factory(CV._net(k), tta=False); pos = B.pos.astype(np.float64); res = {}
    def score(p): s = fn(p.astype(np.float32)); r = CP.event_f1((s > 0) & np.isfinite(p).all(2), B); return [float(r["ev_f1"]), float(r["kappa"])]
    res["degrees (reference)"] = score(pos)
    res["isotropic gain x37 + offset"] = score(pos * 37.3 + np.array([123.0, -45.0]))
    pos = np.where(np.isfinite(pos), pos, np.nan); m = np.nanmean(pos, 1, keepdims=True); sd = np.nanstd(pos, 1, keepdims=True) + 1e-9; res["per-trial per-axis z-score (anisotropic)"] = score((pos - m) / sd)
    pos = np.where(np.isfinite(pos), pos, np.nan)                                    # the positions of some datasets contain +-inf (treated as invalid everywhere else)
    gm = np.nanmean(pos.reshape(-1, 2), 0); gs = np.nanstd(pos.reshape(-1, 2), 0); z = (pos - gm) / gs; z[np.abs(z) >= 5.99] = np.nan; res["dataset z-score, clipped at +-6 (clipped = invalid)"] = score(z)
    out[k] = res; print(k, {a: [round(x, 3) for x in b] for a, b in res.items()}, flush=True)
print("(b) statistics of the network input by source in the pre-training pool (windows of 500 samples at 1 kHz)")
z = np.load(US.POOL); pos, grp = z["pos"].astype(np.float64), z["grp"]; st = {}
for g in np.unique(grp):
    P = pos[grp == g][:600]; f = feats2(P, 1000.0); v, _ = D.velocity(P, 1000.0); sx = np.nanmedian(np.abs(v[..., 0]), 1); sy = np.nanmedian(np.abs(v[..., 1]), 1)
    af = np.abs(f); st[g] = dict(median_abs_input=float(np.nanmedian(af)), p99_abs_input=float(np.nanpercentile(af, 99)), frac_above_3=float(np.nanmean(af > 3)), noise_anisotropy_x_over_y=float(np.nanmedian(sx / np.maximum(sy, 1e-12))), n=int(len(P)))
    print(f"{g:10s}", {a: round(b, 3) for a, b in st[g].items()}, flush=True)
out["pool_input_stats"] = st; json.dump(out, open(os.path.join(NE.NIGHT, "unit_check.json"), "w"), indent=1)
