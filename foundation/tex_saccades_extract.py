#!/usr/bin/env python3
"""Cache of the reading saccades of GazeBase TEX (rounds 1-3, up to 150 saccades per file, EyeLink parser, >= 25 ms, >= 0.5 deg, valid data), with what the landing estimate
needs at tau = 10 and 20 ms: p0 (position before the onset), cur, v, vmax (deg / ms), landing, round, session, participant, and the previous saccade vector (dx, dy).
-> data/GazeBase/tex_saccades.npz   python foundation/tex_saccades_extract.py
"""
import io, os, re, sys, zipfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); rng = np.random.RandomState(0)
z = zipfile.ZipFile(os.path.join(ROOT, "data", "GazeBase", "GazeBase_v2_0.zip")); names = [n for n in z.namelist() if n.endswith(".zip") and "Subject_" in n and int(re.search(r"Round_(\d+)", n).group(1)) <= 3]
R = {k: [] for k in ("p0", "land", "cur10", "v10", "vmax10", "cur20", "v20", "vmax20", "rnd", "ses", "subj", "prev", "dur")}
for ni, name in enumerate(names):
    rnd = int(re.search(r"Round_(\d+)", name).group(1)); subj = int(re.search(r"Subject_(\d+)", name).group(1)) % 1000; inner = zipfile.ZipFile(io.BytesIO(z.read(name)))
    for fn in inner.namelist():
        if not fn.endswith("_TEX.csv"): continue
        ses = int(re.search(r"_S(\d)_TEX", fn).group(1)); d = pd.read_csv(inner.open(fn)); x, y, val, lab = (d[c].values for c in ("x", "y", "val", "lab")); n = len(x)
        sac = np.diff(np.r_[0, (lab == 2).astype(np.int8), 0]); ons = np.where(sac == 1)[0]; offs = np.where(sac == -1)[0] - 1; spd = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
        prev = np.zeros(2); cnt = 0
        for k in range(len(ons)):
            on, off = ons[k], offs[k]
            if off - on + 1 < 25 or on < 5 or off + 5 >= n or (val[on - 1:off + 2] != 0).any() or not np.isfinite(x[on - 1:off + 2]).all(): prev = np.zeros(2); continue
            p0 = np.array([x[on - 1], y[on - 1]]); land = np.array([x[off], y[off]])
            if np.linalg.norm(land - p0) < 0.5: continue
            f = {}
            for tau in (10, 20):
                t = on + tau; f[tau] = (np.array([x[t], y[t]]), spd[t - 2:t + 1].mean(), spd[on:t + 1].max())
            R["p0"].append(p0); R["land"].append(land); R["cur10"].append(f[10][0]); R["v10"].append(f[10][1]); R["vmax10"].append(f[10][2]); R["cur20"].append(f[20][0]); R["v20"].append(f[20][1]); R["vmax20"].append(f[20][2])
            R["rnd"].append(rnd); R["ses"].append(ses); R["subj"].append(subj); R["prev"].append(prev.copy()); R["dur"].append(off - on + 1); prev = land - p0; cnt += 1
            if cnt >= 150: break
    if ni % 80 == 0: print(ni, len(R["p0"]), flush=True)
np.savez_compressed(os.path.join(ROOT, "data", "GazeBase", "tex_saccades.npz"), **{k: np.array(v) for k, v in R.items()})
print("saccades", len(R["p0"]), "participants", len(set(R["subj"])))
