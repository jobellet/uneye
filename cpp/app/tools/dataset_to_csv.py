#!/usr/bin/env python3
"""Converts a labeled recording of the repository (data/datasetN/datasetN_<rate>hz_X_setA.csv + _Y_) into the CSV that EyeLive replays: t,x,y,valid.
Trials are concatenated one after the other (positions are continued so that there is no artificial jump). Example:
    python3 cpp/app/tools/dataset_to_csv.py data/dataset1/dataset1_1000hz_X_setA.csv --seconds 60 --rate 1000 --out demo.csv
Replay it in real time: eyelive_cli --replay demo.csv --speed 1 ...   or in the window: Acquire tab > File replay > speed 1."""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("x_csv", help="..._X_setA.csv (rows = trials, columns = samples, degrees); the _Y_ file is found next to it")
ap.add_argument("--rate", type=float, default=1000.0)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--out", default="demo.csv")
a = ap.parse_args()
X = np.loadtxt(a.x_csv, delimiter=",", ndmin=2)
Y = np.loadtxt(a.x_csv.replace("_X_", "_Y_"), delimiter=",", ndmin=2)
need = int(a.seconds * a.rate)
xs, ys, off_x, off_y = [], [], 0.0, 0.0
for tx, ty in zip(X, Y):
    ok = np.isfinite(tx) & np.isfinite(ty)
    if ok.sum() < 10:
        continue
    first = np.argmax(ok)
    dx, dy = off_x - tx[first], off_y - ty[first]                       # continue from the last position
    xs.append(tx + dx); ys.append(ty + dy)
    last = len(tx) - 1 - np.argmax(ok[::-1]); off_x, off_y = tx[last] + dx, ty[last] + dy
    if sum(len(v) for v in xs) >= need:
        break
x = np.concatenate(xs)[:need]; y = np.concatenate(ys)[:need]
t = np.arange(len(x)) / a.rate
with open(a.out, "w") as f:
    f.write("t,x,y,valid\n")
    for ti, xi, yi in zip(t, x, y):
        v = np.isfinite(xi) and np.isfinite(yi)
        f.write(f"{ti:.4f},{xi:.4f},{yi:.4f},{1 if v else 0}\n" if v else f"{ti:.4f},0,0,0\n")
print(f"{a.out}: {len(x)} samples = {len(x) / a.rate:.1f} s at {a.rate:g} Hz")
