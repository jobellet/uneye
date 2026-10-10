#!/usr/bin/env python3
"""The two analyses of the 2019 article (analysis scripts/analyses.ipynb, evaluation_networks.ipynb) redone with the 2-channel (vx, vy) bidirectional TCN:
 A. performance vs the number of human-labeled trials: train on N trials of dataset 1 set B (N = 10 ... 300, several draws and seeds), test on set A.
 B. generalization between subjects: dataset 4, train on the 33 trials of ONE subject of set A (or 3 per subject = all subjects), test on each subject of set B.
Fixed number of steps, no early stopping and no validation set (as in the article: only the labeled trials given are used). Threshold at 0 on the logit.
Results are cached in docs/figs_paper/results.json (resumable).   python foundation/paper_figs.py [--only A|B] [--steps 1500]
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn.functional as F
from foundation import bitcn as BT, data as FD, compare as CP
from foundation.dino1d import feats2
from foundation.train_lodo import augment, DEV
BT.FEATS = feats2
OUT = os.path.join(ROOT, "docs", "figs_paper"); os.makedirs(OUT, exist_ok=True); RES = os.path.join(OUT, "results.json")


def train(pos, lab, steps, seed, base=480, batch=32, lr=1e-3):
    rng = np.random.RandomState(seed); torch.manual_seed(seed); net = BT.BiTCN(nin=2).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    base = min(base, pos.shape[1])
    for it in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 100) * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.98 + 0.02)
        P, L = [], []
        for _ in range(batch):
            i = rng.randint(len(pos)); s0 = rng.randint(0, pos.shape[1] - base + 1); P.append(pos[i, s0:s0 + base]); L.append(lab[i, s0:s0 + base] == 1)
        aug = augment(np.stack(P), rng, True); x = torch.as_tensor(feats2(aug, 1000.0), device=DEV); y = torch.as_tensor(np.stack(L).astype(np.float32), device=DEV)
        w = torch.as_tensor(np.isfinite(aug).all(2), device=DEV).float(); net.train()
        loss = (F.binary_cross_entropy_with_logits(net(x), y, pos_weight=torch.tensor(1.5, device=DEV), reduction="none") * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step()
    return net


def test(net, pos, lab):
    s = BT.score_fn_factory(net)(pos); S = type("S", (), dict(lab=lab))(); r = CP.event_f1((s > 0) & np.isfinite(pos).all(2), S); return r["ev_f1"], r["kappa"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="AB"); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--reps", type=int, default=3); a = ap.parse_args()
    res = json.load(open(RES)) if os.path.exists(RES) else {"A": {}, "B": {}}; save = lambda: json.dump(res, open(RES, "w"), indent=1)
    if "A" in a.only:
        tr, te = FD.load("d1", "test"), FD.load("d1", "train"); tp, tl = te.pos[:300], te.lab[:300]         # train on set B, test on 300 trials of set A (the article's direction)
        for N in (10, 20, 50, 100, 200, 300, 1000):
            for r in range(a.reps):
                key = f"{N}_{r}"
                if key in res["A"]: continue
                t0 = time.time(); sel = np.random.RandomState(100 * N + r).permutation(len(tr))[:N]
                res["A"][key] = test(train(tr.pos[sel], tr.lab[sel], a.steps, r), tp, tl); save(); print(f"[A] N={N} draw {r}: F1 {res['A'][key][0]:.3f} kappa {res['A'][key][1]:.3f} ({time.time() - t0:.0f} s)", flush=True)
    if "B" in a.only:
        trA, teB = FD.load("d4", "train"), FD.load("d4", "test")
        subA = np.loadtxt(os.path.join(ROOT, "data", "dataset4", "Subject_nb_setA.csv"), delimiter=",")[:len(trA)]; subB = np.loadtxt(os.path.join(ROOT, "data", "dataset4", "Subject_nb_setB.csv"), delimiter=",")[:len(teB)]
        subjects = sorted(np.unique(subA)); rng = np.random.RandomState(0)
        for tr_s in subjects + ["all"]:
            if str(tr_s) in res["B"]: continue
            if tr_s == "all": idx = np.concatenate([rng.permutation(np.where(subA == s)[0])[:33] for s in subjects])
            else: idx = rng.permutation(np.where(subA == tr_s)[0])[:33]
            t0 = time.time(); net = train(trA.pos[idx], trA.lab[idx], a.steps, 0)
            res["B"][str(tr_s)] = [test(net, teB.pos[subB == s], teB.lab[subB == s]) for s in subjects]; save()
            print(f"[B] train {tr_s}: mean test F1 {np.mean([v[0] for v in res['B'][str(tr_s)]]):.3f} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
