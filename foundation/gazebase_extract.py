#!/usr/bin/env python3
"""Extract the target-step saccades of GazeBase v2.0 (task RAN, random saccades, 1 kHz, EyeLink 1000; data/GazeBase/GazeBase_v2_0.zip, not in git) into one npz.
A target jump happens every second (columns xT, yT). For each jump, the FIRST EyeLink-parser saccade (lab == 2 run) that starts 50-450 ms after the jump is kept if it lasts >= 10 ms,
has amplitude >= 0.5 deg, and all positions from 100 ms before its onset to 20 ms after its offset are finite and valid (val == 0).
Stored per saccade: P (n, 250, 2) positions with the onset at index 100 (rows after the offset repeat nothing: NaN), on = 100, off (index of the last saccade sample), tv = target - start (2),
subject, round, session, fix (the previous fixation position error: eye at jump time minus the OLD target, deg).   -> data/GazeBase/ran_saccades.npz
  python foundation/gazebase_extract.py
"""
import io, os, re, sys, zipfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ZIP = os.path.join(ROOT, "data", "GazeBase", "GazeBase_v2_0.zip"); OUT = os.path.join(ROOT, "data", "GazeBase", "ran_saccades.npz")
PRE, T = 100, 250
P, OFF, TV, SUBJ, RND, SES, N0 = [], [], [], [], [], [], 0
z = zipfile.ZipFile(ZIP); subs = [n for n in z.namelist() if n.endswith(".zip") and "Subject_" in n]; print(len(subs), "subject archives", flush=True)
for si, name in enumerate(subs):
    rnd = int(re.search(r"Round_(\d+)", name).group(1)); subj = int(re.search(r"Subject_(\d+)", name).group(1))
    inner = zipfile.ZipFile(io.BytesIO(z.read(name)))
    for fn in inner.namelist():
        if not fn.endswith("_RAN.csv"): continue
        ses = int(re.search(r"_S(\d)_RAN", fn).group(1)); d = pd.read_csv(inner.open(fn))
        x, y, xT, yT, val, lab = (d[c].values for c in ("x", "y", "xT", "yT", "val", "lab")); n = len(x)
        jump = np.where((np.diff(xT) != 0) | (np.diff(yT) != 0))[0] + 1
        sac = np.r_[0, (lab == 2).astype(np.int8), 0]; dd = np.diff(sac); ons = np.where(dd == 1)[0]; offs = np.where(dd == -1)[0] - 1
        for j in jump:
            k = np.where((ons >= j + 50) & (ons <= j + 450))[0]
            if len(k) == 0: continue
            on, off = ons[k[0]], offs[k[0]]
            if off - on + 1 < 10 or on < PRE or off + 20 >= n or on + T - PRE > n + 0: continue
            seg = slice(on - PRE, off + 21)
            if not (np.isfinite(x[seg]).all() and np.isfinite(y[seg]).all()) or (val[seg] != 0).any(): continue
            p0 = np.array([x[on - 1], y[on - 1]]); land = np.array([x[off], y[off]])
            if np.linalg.norm(land - p0) < 0.5 or off - on + 1 > T - PRE - 20: continue
            w = np.full((T, 2), np.nan, np.float32); m = min(T, n - (on - PRE)); w[:m, 0] = x[on - PRE:on - PRE + m]; w[:m, 1] = y[on - PRE:on - PRE + m]
            w[off - on + PRE + 21:] = np.nan
            P.append(w); OFF.append(off - on + PRE); TV.append([xT[on] - p0[0], yT[on] - p0[1]]); SUBJ.append(subj); RND.append(rnd); SES.append(ses)
    if si % 50 == 0: print(si, len(P), flush=True)
np.savez_compressed(OUT, P=np.array(P, np.float32), off=np.array(OFF, np.int16), tv=np.array(TV, np.float32), subj=np.array(SUBJ), rnd=np.array(RND, np.int8), ses=np.array(SES, np.int8))
print("saccades", len(P), "subjects", len(set(SUBJ)))
