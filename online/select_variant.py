#!/usr/bin/env python3
"""Pick the causal variant using the VALIDATION trials only (30 per dataset, same split as train_causal.py).
Score = sample-level F1 of the last bin at the validation-tuned threshold. Test set B is not used."""
import sys, glob, numpy as np, torch
from causal_net import CausalTCN
from common import load, velocity

np.random.seed(1)
val = []
for s in "123":
    X, Y, L, fs = load(s, "A"); p = np.random.permutation(X.shape[0])[:30]
    val.append((velocity(X[p], Y[p]), L[p] > 0))

def f1(P, T, thr):
    p = P > thr; tp = (p & T).sum(); return 2 * tp / max(2 * tp + (p & ~T).sum() + (~p & T).sum(), 1)

for path in sorted(glob.glob("runs/v*_*")):
    if path.endswith(".log"): continue
    ck = torch.load(path, weights_only=False)
    net = CausalTCN(2, ck["channels"], dilations=ck.get("dilations", (1, 2, 4, 8, 16))); net.load_state_dict(ck["state"]); net.eval()
    with torch.no_grad(): P = np.concatenate([net(torch.from_numpy(v))[:, 1].numpy().ravel() for v, _ in val])
    T = np.concatenate([t.ravel() for _, t in val])
    best = max((f1(P, T, th), th) for th in np.arange(0.1, 0.91, 0.05))
    print(f"{path:28s} params {sum(q.numel() for q in net.parameters()):6d} RF {net.receptive_field:4d}  val F1@0.5 {f1(P,T,0.5):.3f}  val F1@tuned {best[0]:.3f} (thr {best[1]:.2f})")
