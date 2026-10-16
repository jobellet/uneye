#!/usr/bin/env python3
"""Figure: 5 frames evenly spaced over 1 second of one DTB70 video, the median eye position of one EyeTrackUAV2 subject between each frame and the next (free viewing, dominant eye),
and the saliency maps of the six models of foundation/saliency_models.py for each of these frames.  python foundation/figure_uav_saliency.py [seq] [subject] [start_frame]
-> docs/figs_saliency/uav_saliency_<seq>_subj<k>.png"""
import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from foundation.saliency_models import Model
seq = sys.argv[1] if len(sys.argv) > 1 else "Basketball"; subj = int(sys.argv[2]) if len(sys.argv) > 2 else 10
files = sorted(glob.glob(os.path.join(ROOT, "data", "DTB70", seq, "img", "*.jpg"))); a = np.loadtxt(os.path.join(ROOT, "data", "EyeTrackUAV2", "RAW_FV", seq, f"DTB70_{seq}_subj_{subj}_FV_OrigRes_DomEye.csv"), delimiter=",")
fr, x, y = a[:, 0].astype(int), a[:, 11], a[:, 12]; ok = np.isfinite(x) & (x > 0) & (y > 0)
if len(sys.argv) > 3: f0 = int(sys.argv[3])
else:                                                          # the 1 s window (30 frames) with the largest total eye displacement
    best = (-1, 40)
    for s in range(40, len(files) - 70, 5):
        m = ok & (fr >= s) & (fr < s + 30); path = np.nansum(np.hypot(np.diff(x[m]), np.diff(y[m]))) if m.sum() > 100 else 0
        if path > best[0]: best = (path, s)
    f0 = best[1]
idx = [f0, f0 + 7, f0 + 15, f0 + 22, f0 + 30]; edges = idx + [idx[-1] + 8]       # frame k ... frame k+1 ; the last frame is followed by 0.25 s
names = ["pqft", "sr", "itti", "itti_dyn", "motion", "center"]; titles = {"pqft": "PQFT (engine)", "sr": "spectral residual", "itti": "Itti-Koch 1998", "itti_dyn": "Itti dynamic", "motion": "frame difference", "center": "centre bias"}
maps = {n: [] for n in names}
for n in names:
    mod = Model(n)
    for k in idx:
        mod(np.asarray(Image.open(files[k - 2]).convert("RGB")))              # previous frame (for the dynamic channels)
        maps[n].append(mod(np.asarray(Image.open(files[k - 1]).convert("RGB"))))
fig, ax = plt.subplots(7, 5, figsize=(20, 16.6), gridspec_kw=dict(wspace=0.015, hspace=0.015))
med = []
for c, k in enumerate(idx):
    img = np.asarray(Image.open(files[k - 1]).convert("RGB")); m = ok & (fr >= edges[c]) & (fr < edges[c + 1]); mx, my = np.median(x[m]), np.median(y[m]); med.append((mx, my))
    for r in range(7):
        A = ax[r, c]; A.set_xticks([]); A.set_yticks([])
        if r == 0: A.imshow(img); A.plot(x[m], y[m], ".", color="yellow", ms=2, alpha=.7)
        else:
            n = names[r - 1]; A.imshow(np.asarray(Image.fromarray(img).convert("L")), cmap="gray"); A.imshow(zoom(maps[n][c], (720 / maps[n][c].shape[0], 1280 / maps[n][c].shape[1]), order=1), cmap="jet", alpha=.55, extent=(0, 1280, 720, 0), vmin=0, vmax=1)
        A.plot([mx], [my], "o", mfc="none", mec="white", mew=2.5, ms=16); A.plot([mx], [my], "o", mfc="none", mec="red", mew=1.2, ms=16); A.set_xlim(0, 1280); A.set_ylim(720, 0)
        if c == 0: A.set_ylabel("frames + gaze" if r == 0 else titles[names[r - 1]], fontsize=13)
    ax[0, c].set_title(f"frame {k} (t = {(k - f0) / 30:.2f} s)\nmedian of the next {(edges[c + 1] - edges[c]) / 30 * 1000:.0f} ms", fontsize=11)
fig.suptitle(f"DTB70 '{seq}', EyeTrackUAV2 subject {subj} (free viewing): yellow = gaze samples until the next frame, circle = their median; rows 2-7 = saliency maps (jet, 0-1) of the same frames", fontsize=14, y=0.93)
os.makedirs(os.path.join(ROOT, "docs", "figs_saliency"), exist_ok=True); out = os.path.join(ROOT, "docs", "figs_saliency", f"uav_saliency_{seq}_subj{subj}.png"); fig.savefig(out, dpi=75, bbox_inches="tight"); print(out, "window from frame", f0, "medians", np.round(med, 0).tolist())
