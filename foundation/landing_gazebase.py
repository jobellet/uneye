#!/usr/bin/env python3
"""Causal landing estimate on GazeBase RAN (1 kHz, EyeLink 1000, target steps up to ~30 deg, 300+ subjects), with and without the KNOWN TARGET as an input.
Data: data/GazeBase/ran_saccades.npz (foundation/gazebase_extract.py). Participants (id % 1000) are split 80 / 10 / 10 into train / validation / test (no subject in two sets); at most
40 000 / 3 000 / 8 000 saccades. Rows: one per ms of tau = 0..59 after the onset of the parser-labelled saccade. Baselines on the same rows: stays here, linear tables (refit),
'goes to the target' (remaining = target - current position). Networks: tcn, gru (as in landing_train.py), with / without the target channels (tv, |tv|).
  python foundation/landing_gazebase.py [--models tcn,gru] [--steps 6000]
"""
import argparse, json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import foundation.landing_train as L
from foundation import unet_ssl as U
PRE, T = 100, 250
def history(z, K=8):
    """per saccade of the whole file: the K previous saccades of the same recording (participant, session), most recent first: (dx, dy)/10, amplitude/10, peak speed (deg/ms), duration/50, valid; + 3 summaries"""
    P, off = z["P"], z["off"].astype(int); n = len(P); key = z["subj"].astype(np.int64) * 10 + z["ses"]; vec = np.zeros((n, 2)); amp = np.zeros(n); pk = np.zeros(n); dur = (off - PRE + 1) / 50.0
    for i in range(n):
        seg = P[i, PRE - 1:off[i] + 1]; vec[i] = seg[-1] - seg[0]; amp[i] = np.linalg.norm(vec[i]); pk[i] = np.nanmax(np.hypot(*np.diff(seg, axis=0).T)) if len(seg) > 1 else 0
    H = np.zeros((n, K * 6 + 3), np.float32)
    for i in range(n):
        j = i - 1; c = 0; sa = sr = 0.0
        while j >= 0 and key[j] == key[i] and c < K:
            H[i, c * 6:(c + 1) * 6] = [vec[j, 0] / 10, vec[j, 1] / 10, amp[j] / 10, pk[j], dur[j], 1.0]; c += 1; j -= 1
        if c: H[i, K * 6] = H[i, [k * 6 + 2 for k in range(c)]].mean(); H[i, K * 6 + 1] = np.mean([H[i, k * 6 + 3] / max(H[i, k * 6 + 2] * 10, 0.3) for k in range(c)]); H[i, K * 6 + 2] = c / K
    return H
def build(z, sel, HIST=None):
    P, off, tv = z["P"][sel], z["off"][sel].astype(int), z["tv"][sel]; n = len(P)
    sp = L.Split.__new__(L.Split); sp.HIST = HIST[sel] if HIST is not None else None; sp.P = np.full((n, T + L.PAD, 2), np.nan); sp.P[:, L.PAD:] = P; sp.TV = tv.astype(np.float64); R = []
    for i in range(n):
        p0 = P[i, PRE - 1]; ts = np.arange(PRE, min(off[i], PRE + 59) + 1); d = np.linalg.norm(P[i, ts] - p0, axis=1)
        for t in ts[d > 0.02]: R.append((i, PRE, off[i], t))
    sp.R = np.array(R, np.int64); return sp
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="tcn,gru"); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--ctx", action="store_true"); ap.add_argument("--hist", action="store_true"); ap.add_argument("--no-target-only", action="store_true"); ap.add_argument("--tag", default=""); a = ap.parse_args()
    L.CTX, L.HIST = a.ctx, a.hist; HH = history(z_all := np.load(os.path.join(ROOT, "data", "GazeBase", "ran_saccades.npz"))) if a.hist else None
    z = np.load(os.path.join(ROOT, "data", "GazeBase", "ran_saccades.npz")); subj = z["subj"] % 1000; us = np.unique(subj); rng = np.random.RandomState(0); rng.shuffle(us)
    nt, nv = int(0.8 * len(us)), int(0.1 * len(us)); sets = {"train": us[:nt], "val": us[nt:nt + nv], "test": us[nt + nv:]}; cap = {"train": 40000, "val": 3000, "test": 8000}; S = {}
    for k, u in sets.items():
        idx = np.where(np.isin(subj, u))[0]; idx = rng.choice(idx, min(len(idx), cap[k]), replace=False); S[k] = build(z, np.sort(idx), HH)
    print("saccades train/val/test", {k: len(v.P) for k, v in S.items()}, "rows", {k: len(v.R) for k, v in S.items()}, flush=True)
    te = S["test"]; tau = te.R[:, 3] - te.R[:, 1]; i, on, off, t = te.R.T
    amp = np.linalg.norm(te.P[i, off + L.PAD] - te.P[i, on + L.PAD - 1], axis=1)
    lin, stay = L.linear_tables(S["train"], te); cur = te.P[i, t + L.PAD]; p0 = te.P[i, on + L.PAD - 1]
    res = {"stays here": stay, "linear tables": lin, "goes to the target": np.linalg.norm(te.TV[i] + p0 - te.P[i, off + L.PAD], axis=1) * 0 + np.linalg.norm((te.TV[i] + p0) - te.P[i, off + L.PAD], axis=1)}
    info = {}
    for tgt in ((False,) if a.no_target_only else (False, True)):
        L.TGT = tgt; L.NIN = 7 + (3 if tgt else 0) + (8 if a.ctx else 0) + (51 if a.hist else 0)
        for name in a.models.split(","):
            net, sec = L.train(name, S["train"], S["val"], a.steps); key = f"{name}{' +target' if tgt else ''}"; res[key] = L.evaluate(net, name, te); info[key] = round(sec)
            torch.save({"state": net.state_dict(), "name": name, "tgt": tgt}, os.path.join(U.RUNS, f"landing_gazebase_{name}{'_tgt' if tgt else ''}{a.tag}.pt")); print(f"== {key}: {res[key].mean():.3f} deg ({sec:.0f} s)", flush=True)
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]
    table = {k: {f"{lo}-{hi}": float(v[(tau >= lo) & (tau < hi)].mean()) for lo, hi in bins} | {"all": float(v.mean())} for k, v in res.items()}
    print(f"\nGazeBase RAN test: mean landing error (deg) by time since onset (ms); {len(te.P)} saccades of {len(sets['test'])} held-out subjects")
    print(f"{'':20s}" + "".join(f"{lo:>2d}-{hi:<4d}" for lo, hi in bins) + "  all")
    for k, r in table.items(): print(f"{k:20s}" + "".join(f"{r[f'{lo}-{hi}']:7.2f}" for lo, hi in bins) + f"  {r['all']:.2f}")
    AMP = [0, 5, 10, 20, 100]; print("\nby amplitude (all tau 5-20 ms):")
    for lo, hi in zip(AMP[:-1], AMP[1:]):
        m = (amp >= lo) & (amp < hi) & (tau >= 5) & (tau < 20); print(f"  amp {lo}-{hi} deg (n rows {m.sum()}): " + "  ".join(f"{k} {v[m].mean():.2f}" for k, v in res.items()))
    json.dump({"table": table, "train_s": info, "n_saccades": {k: len(v.P) for k, v in S.items()}}, open(os.path.join(U.RUNS, f"landing_gazebase{a.tag}.json"), "w"), indent=1)
