#!/usr/bin/env python3
"""GazeBase RAN with SEVERAL candidate targets, only one being the real one (the application knows where the possible targets are, not which one the eye will choose).
Candidates: the true target plus up to 7 distractors, drawn among the target positions of other saccades of the same split (so they follow the same screen distribution).
One network per architecture is trained with K (number of candidates, true one included) drawn from {1, 2, 4, 8} per row and tested at each K; the K valid candidates are given
sorted by the cosine between their direction and the current chord (cx, cy, |c|, cos, valid per slot). Compared with: no target (landing_gazebase.py), the heuristic
'go to the candidate best aligned with the current chord', and the number of rows where the network's landing is closer to the true target than to any distractor.
  python foundation/landing_gazebase_cand.py [--models tcn,gru] [--steps 6000]
"""
import argparse, json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import foundation.landing_train as L
import foundation.landing_gazebase as G
from foundation import unet_ssl as U
KMAX = 8
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="tcn,gru"); ap.add_argument("--steps", type=int, default=6000); a = ap.parse_args()
    z = np.load(os.path.join(ROOT, "data", "GazeBase", "ran_saccades.npz")); subj = z["subj"] % 1000; us = np.unique(subj); rng = np.random.RandomState(0); rng.shuffle(us)
    nt, nv = int(0.8 * len(us)), int(0.1 * len(us)); sets = {"train": us[:nt], "val": us[nt:nt + nv], "test": us[nt + nv:]}; cap = {"train": 40000, "val": 3000, "test": 8000}; S = {}
    for k, u in sets.items():
        idx = np.sort(rng.choice(np.where(np.isin(subj, u))[0], min(int(np.isin(subj, u).sum()), cap[k]), replace=False)); sp = G.build(z, idx)
        P0 = z["P"][idx][:, G.PRE - 1]; absT = z["tv"][idx] + P0; n = len(idx); r2 = np.random.RandomState(7)
        C = np.zeros((n, KMAX, 2)); C[:, 0] = z["tv"][idx]
        for j in range(1, KMAX):
            o = r2.randint(0, n, n); o = np.where(o == np.arange(n), (o + 1) % n, o); C[:, j] = absT[o] - P0
        sp.CAND = C; S[k] = sp
    L.CAND = KMAX; L.NIN = 7 + KMAX * 5
    te = S["test"]; i, on, off, t = te.R.T; tau = t - on; cur = te.P[i, t + L.PAD]; p0 = te.P[i, on + L.PAD - 1]; land = te.P[i, off + L.PAD]
    res, acc, info = {}, {}, {}
    for name in a.models.split(","):
        net, sec = L.train(name, S["train"], S["val"], a.steps); info[name] = round(sec)
        torch.save({"state": net.state_dict(), "name": name, "cand": KMAX}, os.path.join(U.RUNS, f"landing_gazebase_cand_{name}.pt"))
        for K in (1, 2, 4, 8):
            te.KFIX = K; net.eval(); out = []
            with torch.no_grad():
                for s in range(0, len(te.R), 512):
                    b = te.batch(np.arange(s, min(s + 512, len(te.R)))); o = net(L.seq_input(te, b)).cpu().numpy()[:, :2]; out.append(o)
            pred = cur + np.concatenate(out); err = np.linalg.norm(pred - land, axis=1); res[f"{name} K={K}"] = err
            ctrue = p0 + te.CAND[i, 0]; dist_true = np.linalg.norm(pred - ctrue, axis=1)
            if K > 1: acc[f"{name} K={K}"] = float(np.mean([dist_true[m] < np.min([np.linalg.norm(pred[m] - (p0[m] + te.CAND[i[m], j]), axis=1) for j in range(1, K)], axis=0) for m in [np.arange(len(i))]][0]))
            print(f"== {name} K={K}: {err.mean():.3f} deg" + (f"  true candidate chosen {acc[f'{name} K={K}']*100:.1f} %" if K > 1 else ""), flush=True)
    # heuristic: go to the candidate best aligned with the current chord (valid candidates = first K)
    ch = cur - p0; chn = np.maximum(np.linalg.norm(ch, axis=1), 1e-6)
    for K in (1, 2, 4, 8):
        cs = np.stack([(te.CAND[i, j] * ch).sum(1) / (np.maximum(np.linalg.norm(te.CAND[i, j], axis=1), 1e-6) * chn) for j in range(K)], 1); pick = cs.argmax(1)
        tgt = p0 + te.CAND[i, pick]; res[f"aligned heuristic K={K}"] = np.linalg.norm(tgt - land, axis=1); acc[f"aligned heuristic K={K}"] = float(np.mean(pick == 0))
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]
    table = {k: {f"{lo}-{hi}": float(v[(tau >= lo) & (tau < hi)].mean()) for lo, hi in bins} | {"all": float(v.mean())} for k, v in res.items()}
    print(f"\nGazeBase RAN with K candidate targets (one true): mean landing error (deg) by time since onset (ms)")
    print(f"{'':26s}" + "".join(f"{lo:>2d}-{hi:<4d}" for lo, hi in bins) + "  all   true chosen")
    for k, r in table.items(): print(f"{k:26s}" + "".join(f"{r[f'{lo}-{hi}']:7.2f}" for lo, hi in bins) + f"  {r['all']:.2f}   " + (f"{acc[k]*100:5.1f} %" if k in acc else ""))
    json.dump({"table": table, "true_chosen": acc, "train_s": info}, open(os.path.join(U.RUNS, "landing_gazebase_cand.json"), "w"), indent=1)
