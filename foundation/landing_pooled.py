#!/usr/bin/env python3
"""One network for both settings: GazeBase RAN (target steps, 1 kHz EyeLink, large saccades) + EyeTrackUAV2 (drone videos, free viewing, small saccades), no rotation, absolute position + direction (ctx).
Models trained on GazeBase only, on the drone data only, and on both (the 2 169 drone training saccades repeated 10 times), tested on the test participants of each setting. MLP (64 samples, the one exported to C++)
and GRU (reference).   -> foundation/runs/landing_pooled.json, checkpoints foundation/runs/landing_pooled_<model>_<train set>.pt"""
import json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online")); sys.path.insert(0, os.path.join(ROOT, "foundation", "deepgaze_kit"))
import foundation.landing_train as L, foundation.landing_gazebase as G, foundation.landing_uav_nets as UN
from foundation import unet_ssl as U
def concat(parts, reps):
    sp = L.Split.__new__(L.Split); sp.HIST = None; sp.P = np.concatenate([p.P for p in parts]); R = []; off = 0
    for p, r in zip(parts, reps):
        for _ in range(r): q = p.R.copy(); q[:, 0] += off; R.append(q)
        off += len(p.P)
    sp.R = np.concatenate(R); return sp
if __name__ == "__main__":
    z = np.load(os.path.join(ROOT, "data", "GazeBase", "ran_saccades.npz")); subj = z["subj"] % 1000; us = np.unique(subj); rng = np.random.RandomState(0); rng.shuffle(us)
    nt, nv = int(0.8 * len(us)), int(0.1 * len(us)); sets = {"train": us[:nt], "val": us[nt:nt + nv], "test": us[nt + nv:]}; cap = {"train": 40000, "val": 3000, "test": 8000}; GB = {}
    for k, u in sets.items(): idx = np.sort(rng.choice(np.where(np.isin(subj, u))[0], min(int(np.isin(subj, u).sum()), cap[k]), replace=False)); GB[k] = G.build(z, idx)
    P, off, sj = UN.collect(); r2 = np.random.RandomState(0); trn = np.where(sj % 2 == 0)[0]; tst = np.where(sj % 2 == 1)[0]; r2.shuffle(trn); nv = len(trn) // 10
    DR = {"train": UN.build(P, off, trn[nv:]), "val": UN.build(P, off, trn[:nv]), "test": UN.build(P, off, tst)}
    L.CTX = True; L.NIN = 15; bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]; out = {}
    tests = {"GazeBase RAN": GB["test"], "drone videos": DR["test"]}; taus = {k: v.R[:, 3] - v.R[:, 1] for k, v in tests.items()}
    for k, sp in tests.items():
        lin, stay = L.linear_tables(GB["train"] if k == "GazeBase RAN" else DR["train"], sp); out[f"tables, {k}"] = {"all": float(lin.mean()), **{f"{lo}-{hi}": float(lin[(taus[k] >= lo) & (taus[k] < hi)].mean()) for lo, hi in bins}}
    trains = {"GazeBase only": (GB["train"], GB["val"]), "drone only": (DR["train"], DR["val"]), "both": (concat([GB["train"], DR["train"]], [1, 10]), concat([GB["val"], DR["val"]], [1, 5]))}
    for model in ("mlp", "gru"):
        for tn, (tr, va) in trains.items():
            net, sec = L.train(model, tr, va, 6000); torch.save({"state": net.state_dict(), "name": model}, os.path.join(U.RUNS, f"landing_pooled_{model}_{tn.replace(' ', '_')}.pt"))
            for k, sp in tests.items():
                e = L.evaluate(net, model, sp); out[f"{model}, trained on {tn}, tested on {k}"] = {"all": float(e.mean()), **{f"{lo}-{hi}": float(e[(taus[k] >= lo) & (taus[k] < hi)].mean()) for lo, hi in bins}, "train_s": round(sec)}
                print(f"{model:4s} {tn:14s} -> {k:13s} {e.mean():.3f} deg   (tables {out['tables, ' + k]['all']:.3f})", flush=True)
    json.dump(out, open(os.path.join(ROOT, "foundation", "runs", "landing_pooled.json"), "w"), indent=1)
