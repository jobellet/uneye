#!/usr/bin/env python3
"""Evaluate scanpath predictions (predictions.json from run_scanpaths_gpu.py, or --fake: the PQFT peaks, to test this script) as candidate targets for the landing estimate, on the EyeTrackUAV2 saccades
whose onset frame is within +-STEP/2 frames of a predicted frame (the prediction of the nearest frame is used: a static-image scanpath has no time axis).
Per source: share of landings within 1 / 2 deg of a predicted fixation, and the landing error at tau = 10 / 20 ms with the predicted fixations as candidates, against the tables alone, 8 random
points, 8 centre-weighted random points and the first 3 fixations only.   python foundation/deepgaze_kit/evaluate_deepgaze.py predictions.json | --fake"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uav_common import *
from PIL import Image
if sys.argv[1] == "--fake":
    from foundation.saliency_models import Model
    man = json.load(open(os.path.join(ROOT, "data", "DeepGazeKit", "manifest.json"))); pred = {}
    for m in man:
        mm = Model("pqft")(np.asarray(Image.open(os.path.join(ROOT, "data", "DeepGazeKit", "frames", m["file"])).convert("RGB")));         xs, ys = peaks(mm, 8); pred[m["file"]] = {"seq": m["seq"], "frame_index": m["frame_index"], "fixations_0_100": [[int((x + .5) / 64 * 100), int((y + .5) / 64 * 100)] for x, y in zip(xs, ys)]}
else: pred = json.load(open(sys.argv[1]))
by_seq = {}
for v in pred.values(): by_seq.setdefault(v["seq"], []).append((v["frame_index"], v["fixations_0_100"]))
for k in by_seq: by_seq[k].sort(key=lambda t: t[0])
step = int(np.median(np.diff([f for f, _ in by_seq[SEQS[0]]]))); sacc = detect_saccades(); rng = np.random.RandomState(0)
def to_deg(fx): fx = np.array(fx, float); return np.stack(f2deg(fx[:, 0] / 100 * 1280, fx[:, 1] / 100 * 720), 1) if len(fx) else np.zeros((0, 2))
res = {k: {10: [], 20: []} for k in ("tables", "scanpath (all)", "scanpath (first 3)", "random 8", "centre-weighted random 8")}; cov = {k: [] for k in ("scanpath (all)", "random 8")}; n = 0
for si, f0, p0, land, feats in sacc:
    lst = by_seq.get(SEQS[si]); 
    if not lst: continue
    j = int(np.argmin([abs(f - f0) for f, _ in lst]))
    if abs(lst[j][0] - f0) > step // 2 or len(lst[j][1]) < 3: continue
    P = to_deg(lst[j][1]); R = np.stack(f2deg(rng.uniform(0, 1280, 8), rng.uniform(0, 720, 8)), 1); Cw = np.stack(f2deg(np.clip(rng.normal(640, 220, 8), 0, 1280), np.clip(rng.normal(360, 130, 8), 0, 720)), 1); n += 1
    cov["scanpath (all)"].append(np.linalg.norm(P - land, axis=1).min()); cov["random 8"].append(np.linalg.norm(R - land, axis=1).min())
    for tau, (cur, rem, tb) in feats.items():
        res["tables"][tau].append(np.linalg.norm(tb - land))
        for name, C in (("scanpath (all)", P), ("scanpath (first 3)", P[:3]), ("random 8", R), ("centre-weighted random 8", Cw)):
            k = pick(p0, cur, rem, C, np.ones(len(C))); res[name][tau].append(np.linalg.norm((rule(p0, C[k], tb, tau) if k >= 0 else tb) - land))
print("saccades evaluated:", n, " (frames every", step, "frames)")
print("within 1 / 2 deg of a candidate:", {k: (round(float(np.mean(np.array(v) < 1)) * 100), round(float(np.mean(np.array(v) < 2)) * 100)) for k, v in cov.items()})
for k, v in res.items(): print(f"  {k:26s} tau 10 ms: {np.mean(v[10]):.2f} deg   tau 20 ms: {np.mean(v[20]):.2f} deg")
