#!/usr/bin/env python3
"""Reads the MAT-file written by test_export.cpp with scipy and checks the values (used by the test when scipy is available).  python3 check_mat.py file.mat"""
import sys
import numpy as np
from scipy.io import loadmat
d = loadmat(sys.argv[1])
t, x, y, valid, label, expert, ev = (d[k] for k in ("t", "x", "y", "valid", "label", "expert", "events"))
assert t.shape == (2000, 1) and x.shape == (2000, 1) and ev.shape == (43, 9), (t.shape, ev.shape)
assert abs(float(d["rate_hz"][0, 0]) - 1000.0) < 1e-9 and abs(float(d["deg_per_unit"][0, 0]) - 0.5) < 1e-12
assert abs(t[0, 0] - 10.0) < 1e-9 and abs(t[1999, 0] - 11.999) < 1e-9
assert abs(x[100, 0] - 100 * 0.01) < 1e-5 and abs(y[100, 0] + 100 * 0.02) < 1e-5
assert valid[5, 0] == 0.0 and valid[6, 0] == 1.0 and set(np.unique(label)) <= {0.0, 1.0, 2.0}
assert ev[0, 0] == 101 and ev[0, 1] == 120 and ev[0, 4] == 1 and abs(ev[0, 5] - 4.0) < 1e-5 and abs(ev[0, 6] - 500 * 4.0 ** 0.8) < 1e-2, ev[0]
assert ev[2, 4] == 2 and ev[1, 8] == 1.0           # a blink, and an expert event
print("scipy reads the MAT-file: OK")
