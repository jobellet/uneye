"""Glue: run one detector through all label-free tests (and, for the sanity check only, against human labels)."""
import os, sys
import numpy as np
import pandas as pd
from . import detectors as D, intrinsic as I

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "online"))
import metrics as M                                    # noqa: E402  (kappa, event F1 of the U'n'Eye study)


def subset(win, n, seed=0):
    idx = np.random.RandomState(seed).permutation(len(win))[:n]
    return win.subset(np.sort(idx))


def evaluate(name, detect, pool, labeled=None, n_invariance=60, n_inject=120, seed=0):
    """detect(Windows) -> bool labels (n, T).
    Label-free columns (computed on `pool`, unlabeled windows of YOUR data): ILS and its parts, invariance, inj_* (injection benchmark).
    Reference columns (only if `labeled` is given): kappa / ev_f1 against human labels of the repository recordings, per dataset + mean."""
    lab = detect(pool)
    row = {"method": name}
    s = I.score(pool.pos, lab, pool.fs, groups=pool.group)
    row.update({k: s[k] for k in ("ILS", "persistence", "main_sequence", "coverage", "precision", "stereotypy", "straightness", "events_per_min", "flips_per_s")})
    row.update(I.invariance(detect, subset(pool, n_invariance, seed), seed))
    row.update(I.bench_injection(detect, subset(pool, n_inject * 3, seed), seed=seed))
    if labeled:
        ks, fs_ = [], []
        for nm, w in labeled.items():
            l = detect(w); m = M.evaluate_probs(l.astype(np.float32), w.labels.astype(np.float32), w.fs, thr=0.5, min_event=2, with_ap=False)
            row[f"kappa_{nm}"] = m["kappa"]; ks.append(m["kappa"]); fs_.append(m["ev_f1"])
            row[f"ILS_{nm}"] = I.score(w.pos, l, w.fs)["ILS"]
        row["kappa"] = float(np.mean(ks)); row["ev_f1"] = float(np.mean(fs_))
    return row, lab


def to_frame(rows):
    return pd.DataFrame(rows).set_index("method")


def validation_table(rows, names):
    """long table (method, dataset, ILS, kappa) from the per-dataset columns, to check that the label-free score ranks like the truth"""
    out = []
    for r in rows:
        for nm in names:
            if f"kappa_{nm}" in r: out.append(dict(method=r["method"], dataset=nm, ILS=r[f"ILS_{nm}"], kappa=r[f"kappa_{nm}"]))
    return pd.DataFrame(out)
