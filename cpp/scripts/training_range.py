#!/usr/bin/env python3
"""Training range of the input, for the out-of-distribution (OOD) check of the safety guard (cpp/include/uneye/safety_guard.hpp).

The guard estimates the fixation noise of the incoming signal the same way as the engine: robust standard deviation (Engbert-Kliegl,
sqrt(median(v^2) - median(v)^2)) of the 3-sample mean velocity over the last 200 samples, per axis, in deg/s. Here the same quantity is
computed on every 200-sample window of the TRAINING data (set A of datasets 1 and 2, 1 kHz, the rate the engine supports) and the
percentiles are printed. The guard calls the input "outside the training range" when the noise leaves [p0.5 / 2, p99.5 * 2].

Run (from online/, where the data paths are relative): python ../cpp/scripts/training_range.py   (deterministic)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "online"))
from common import load  # noqa: E402


def sigma_windows(X, Y, fs, win=200):
    out = []
    for x, y in zip(X, Y):
        dx, dy = np.diff(x), np.diff(y)
        k = np.ones(3) / 3.0
        vx, vy = np.convolve(dx, k, "valid") * fs, np.convolve(dy, k, "valid") * fs
        for s in range(0, len(vx) - win + 1, win):
            a, b = vx[s:s + win], vy[s:s + win]
            if not (np.isfinite(a).all() and np.isfinite(b).all()): continue
            sd = [np.sqrt(max(np.median(v ** 2) - np.median(v) ** 2, 1.0)) for v in (a, b)]   # same 1 deg/s floor as the engine
            out.append(sd)
    return np.array(out)


def main():
    rows = []
    for s in ("1", "2"):
        X, Y, L, fs = load(s, "A")
        assert fs == 1000, fs
        sg = sigma_windows(X, Y, fs); rows.append(sg)
        print(f"dataset {s}: {len(sg)} windows, sigma x/y median {np.median(sg, 0).round(2)} deg/s")
    sg = np.concatenate(rows).max(1)                                       # the guard uses the larger of the two axes
    p = np.percentile(sg, [0.5, 50, 99.5])
    print(f"pooled ({len(sg)} windows): sigma p0.5 {p[0]:.2f}  median {p[1]:.2f}  p99.5 {p[2]:.2f} deg/s")
    print(f"guard constants: kTrainSigmaLo = {p[0]:.2f}, kTrainSigmaHi = {p[2]:.2f}")


if __name__ == "__main__":
    main()
