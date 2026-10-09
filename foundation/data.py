"""One loader for all labeled datasets, at one sampling rate (1 kHz, the rate of the C++ engine).

Datasets (Bellet et al. 2019 + the multi-class benchmark of Andersson et al. 2017, data/Andersson/, not in git, GPL-3.0):
  d1, d2, d4   1 kHz, labels saccade / not saccade
  d3           500 Hz, labels saccade / not saccade
  andersson    500 Hz, labels fixation, saccade, PSO, pursuit, blink ("other" -> ignored)
Classes of the model (K = 5): 0 fixation, 1 saccade, 2 PSO (post-saccadic oscillation), 3 pursuit, 4 blink.
A binary dataset labels 1 = saccade and 0 = "any non-saccade class" (COARSE): its loss uses p(saccade) vs 1 - p(saccade), so
it does not force a convention it does not have (e.g. whether a PSO belongs to the saccade).
Label value -1 = no label (ignored). Positions are in degrees; the model sees only velocities.
"""
import glob, os
import numpy as np
import scipy.io as sio
from math import atan2, degrees

FS = 1000.0
CLASSES = ("fixation", "saccade", "PSO", "pursuit", "blink")
K = len(CLASSES)
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPO = {"d1": ("dataset1", "dataset1_1000hz", 1000.0), "d2": ("dataset2", "dataset2_1000hz", 1000.0),
        "d3": ("dataset3", "dataset3_500hz", 500.0), "d4": ("dataset4", "dataset4_1000hz", 1000.0)}
ALL = ("d1", "d2", "d3", "d4", "andersson")


class Set:
    """trials of one dataset at FS: pos (n, T, 2) float32 deg (NaN = invalid), lab (n, T) int8, coarse (True: binary labels)"""
    def __init__(self, name, pos, lab, coarse, extra=None):
        self.name, self.pos, self.lab, self.coarse, self.extra = name, pos, lab, coarse, extra or {}
    def __len__(self): return len(self.pos)


def upsample(x, fs_in, nearest=False):
    """(n, T, ...) at fs_in -> at FS (linear for positions, nearest for labels)"""
    if fs_in == FS: return x
    T = x.shape[1]; t_in = np.arange(T) / fs_in; t = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / FS)
    if nearest:
        return x[:, np.clip(np.round(t * fs_in).astype(int), 0, T - 1)]
    out = np.empty((x.shape[0], len(t)) + x.shape[2:], np.float32)
    for i in range(x.shape[0]):
        for a in range(x.shape[2]):
            v = x[i, :, a]; ok = np.isfinite(v)
            out[i, :, a] = np.interp(t, t_in[ok], v[ok]) if ok.sum() > 1 else np.nan
            bad = ~ok[np.clip(np.round(t * fs_in).astype(int), 0, T - 1)]; out[i, bad, a] = np.nan
    return out


def load_repo(name, which="B", n=None):
    d, f, fs = REPO[name]
    ld = lambda k: np.loadtxt(os.path.join(ROOT, d, f"{f}_{k}_set{which}.csv"), delimiter=",")
    X, Y, L = ld("X"), ld("Y"), ld("Labels")
    if n: X, Y, L = X[:n], Y[:n], L[:n]
    n_pos = min(len(X), len(Y), len(L))                     # dataset 4 set B: 3000 trials of positions but 3300 rows of labels;
    X, Y, L = X[:n_pos], Y[:n_pos], L[:n_pos]               # the FIRST 3000 label rows are the right ones (Engbert-Kliegl event F1
    pos = np.stack([X, Y], 2).astype(np.float32)            # 0.76 with them, 0.05 = chance with the last 3000)
    lab = (L > 0).astype(np.int8)
    return Set(name, upsample(pos, fs), upsample(lab, fs, nearest=True), coarse=True)


def _read_andersson(path):
    d = sio.loadmat(path)["ETdata"][0][0]
    screen, res, dist = np.ravel(d[1]), np.ravel(d[2]), float(np.ravel(d[3])[0])
    deg_x = degrees(atan2(0.5 * screen[0], dist)) / (0.5 * res[0]); deg_y = degrees(atan2(0.5 * screen[1], dist)) / (0.5 * res[1])
    x, y, lab = d[0][:, 3] * deg_x, d[0][:, 4] * deg_y, d[0][:, 5].astype(int)
    m = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}                                        # file codes 1..6 -> model classes; 6 "other" -> -1
    return np.stack([x, y], 1).astype(np.float32), np.array([m.get(v, -1) for v in lab], np.int8)


def load_andersson(split="test", chunk=2000, coder="RA"):
    """split 'train': trials coded by RA only (originally uploaded data, minus the article's two-coder trials);
       split 'test': the two-coder trials of the article (labels of `coder`). Cut into chunks of `chunk` samples at 500 Hz (as in
       the 2019 notebook), then upsampled to 1 kHz. Blinks are NaN in the raw data or labeled 5: both kept as they are."""
    art = os.path.join(ROOT, "Andersson", "annotated_data", "data used in the article")
    orig = os.path.join(ROOT, "Andersson", "annotated_data", "originally uploaded data")
    test_ids = {os.path.basename(p)[:-7] for p in glob.glob(os.path.join(art, "*", "*_MN.mat"))}
    if split == "test":
        files = sorted(glob.glob(os.path.join(art, "*", f"*_{coder}.mat")))
    else:
        files = sorted(p for p in glob.glob(os.path.join(orig, "*", "*_RA.mat")) if os.path.basename(p)[:-7] not in test_ids)
    pos, lab = [], []
    for p in files:
        xy, l = _read_andersson(p)
        xy[(xy == 0).all(1)] = np.nan                                        # (0, 0) = lost signal in these files
        for s in range(0, len(l) - chunk + 1, chunk):
            pos.append(xy[s:s + chunk]); lab.append(l[s:s + chunk])
        if len(l) % chunk > chunk // 4:                                       # keep a long enough last piece, padded with "no label"
            r = len(l) % chunk; P = np.full((chunk, 2), np.nan, np.float32); L = np.full(chunk, -1, np.int8)
            P[:r] = xy[-r:]; L[:r] = l[-r:]; pos.append(P); lab.append(L)
    pos, lab = np.stack(pos), np.stack(lab)
    return Set("andersson", upsample(pos, 500.0), upsample(lab, 500.0, nearest=True), coarse=False, extra={"files": len(files)})


def load(name, which):
    """which = 'train' (set A / RA-only trials) or 'test' (set B / two-coder trials)"""
    if name == "andersson": return load_andersson("train" if which == "train" else "test")
    return load_repo(name, "A" if which == "train" else "B")
