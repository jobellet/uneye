#!/usr/bin/env python3
"""Snapshots of every data source (unlabeled archive/ sources and the five labeled benchmarks) to judge by eye which ones are good enough for pre-training.
One PNG per source in docs/figs_data/: 8 random 1-s windows (fixed seed, NOT cherry-picked), position x / y on top, speed below (log scale hint: arcsinh), human
saccade labels shaded for the benchmarks. Title = measured rate, missing fraction, speed noise (robust sigma of speed, in the source's own units / s), clipped and flat windows.
Also writes docs/figs_data/summary.md.   python foundation/data_snapshots.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from free_saccade import data as FS
from foundation import data as FD, compare as CP
OUT = os.path.join(ROOT, "docs", "figs_data"); os.makedirs(OUT, exist_ok=True)
NOTE = {"EMTeC": "reading, ~2 kHz", "GazeBase": "1 kHz, 1 file only", "Lund2013": "500 Hz (images/video/dots)", "GazeCom": "250 Hz, free viewing videos", "360EM": "125 Hz, 360-degree video", "VEDB": "120 Hz, mobile glasses, daily life",
        "DUT-OMRON": "30 Hz image fixations (excluded)", "EGTEA": "30 Hz egocentric video (excluded)"}
UNIT = {"archive": "z-scored (clipped at +-6)", "bench": "degrees"}


def stats(w, fs):
    """w (n, T, 2) windows"""
    ok = np.isfinite(w).all(2); miss = float(1 - ok.mean()); v = np.hypot(*np.diff(np.nan_to_num(w), axis=1).transpose(2, 0, 1)) * fs
    sig = [1.4826 * np.median(np.abs(v[i][ok[i, 1:]] - np.median(v[i][ok[i, 1:]]))) for i in range(len(w)) if ok[i].sum() > 10]
    flat = float(np.mean([(np.nanstd(np.diff(w[i], axis=0)) < 1e-9) for i in range(len(w))]))
    clip = float(np.mean(np.abs(np.nan_to_num(w)) >= 5.99)); return dict(missing=miss, speed_noise=float(np.median(sig)) if sig else np.nan, flat_windows=flat, clipped=clip)


def draw(name, w, fs, lab, unit, info, extra=""):
    fig, ax = plt.subplots(4, 4, figsize=(16, 8.5), gridspec_kw=dict(height_ratios=[2, 1, 2, 1]), sharex=False); t = np.arange(w.shape[1]) / fs
    for i in range(8):
        r, c = 2 * (i // 4), i % 4; a, b = ax[r, c], ax[r + 1, c]
        a.plot(t, w[i, :, 0], lw=.8, color="#337ab7"); a.plot(t, w[i, :, 1], lw=.8, color="#d9534f"); b.plot(t[1:], np.arcsinh(np.hypot(*np.diff(w[i], axis=0).T) * fs), lw=.7, color="k")
        if lab is not None:
            for s in (a, b): s.fill_between(t, 0, 1, where=lab[i] == 1, color="#f0ad4e", alpha=.35, transform=s.get_xaxis_transform())
        a.set_xticks([]); a.tick_params(labelsize=7); b.tick_params(labelsize=7); a.set_title(f"window {i + 1}", fontsize=8)
        if c == 0: a.set_ylabel(f"position ({'deg' if unit == 'bench' else 'z'})", fontsize=7); b.set_ylabel("asinh speed", fontsize=7)
    fig.suptitle(f"{name} — {info}  |  rate {fs:.0f} Hz, missing {100 * extra['missing']:.1f} %, speed noise {extra['speed_noise']:.3g} /s, flat windows {100 * extra['flat_windows']:.0f} %, clipped {100 * extra['clipped']:.1f} %   (blue x, red y; orange = human saccade)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    rows = []; fs_table = dict(FS.ARCHIVE_FS, **{"DUT-OMRON": 30.0, "EGTEA": 30.0})
    src = FS.H5Source(os.path.join(ROOT, "archive", "datasets_by_subject", "datasets_by_subject"), os.path.join(ROOT, "archive", "subject_h5_metadata.csv"), "dataset", 1.0, fs_table, limit=None, clip=FS.ARCHIVE_CLIP)
    for g in ["EMTeC", "GazeBase", "Lund2013", "GazeCom", "360EM", "VEDB", "DUT-OMRON", "EGTEA"]:
        fs = fs_table[g]; W = src.sample(40, min(int(fs), 1000), np.random.RandomState(0), fs=fs, groups=(g,), max_missing=0.5)
        if len(W.pos) < 8: print("not enough windows for", g, len(W.pos)); continue
        st = stats(W.pos, fs); draw(g, W.pos[:8], fs, None, "archive", NOTE[g], st); rows.append((g, "archive (unlabeled)", fs, NOTE[g], st)); print(g, st, flush=True)
    for k in FD.ALL:
        S = FD.load(k, "test"); r = int(round(FD.FS / CP.NATIVE[k])); fs = CP.NATIVE[k]; rng = np.random.RandomState(1); idx = rng.permutation(len(S))[:40]
        pos = S.pos[idx][:, ::r]; lab = S.lab[idx][:, ::r]; T = int(fs); pos, lab = pos[:, :T], lab[:, :T]
        st = stats(pos, fs); st['clipped'] = float('nan'); draw(k, pos[:8], fs, lab[:8], "bench", "labeled benchmark (set B)", st); rows.append((k, "benchmark (labeled)", fs, "labeled benchmark", st)); print(k, st, flush=True)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("| source | type | rate Hz | note | missing % | speed noise /s | flat windows % | clipped % |\n|---|---|---|---|---|---|---|---|\n")
        for g, ty, fs, nt, st in rows: f.write(f"| {g} | {ty} | {fs:.0f} | {nt} | {100 * st['missing']:.1f} | {st['speed_noise']:.3g} | {100 * st['flat_windows']:.0f} | {100 * st['clipped']:.1f} |\n")
