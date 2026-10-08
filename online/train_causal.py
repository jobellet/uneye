#!/usr/bin/env python3
"""Train the causal network with the same logic as uneye.DNN.train:
 - training data: set A of the chosen datasets (pooled, shuffled with seed 1)
 - 30 trials of it are held out for validation / early stopping
 - rotation augmentation (4 extra rotations), MCLoss, L2 penalty 0.001, Adam lr 1e-3
 - lr halved when validation gets worse (best weights restored), stop after >3 bad epochs
Differences: minibatches of --batch trials instead of 10 giant batches per epoch (CPU friendly).
Test data (set B) is never touched here.
"""
import argparse, math, time, json
import numpy as np, torch
from causal_net import CausalTCN
from common import load, velocity, mc_loss

ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="1,2,3")
ap.add_argument("--out", default="weights_causal")
ap.add_argument("--channels", type=int, default=24)
ap.add_argument("--max-iter", type=int, default=100)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--l2", type=float, default=1e-3)
ap.add_argument("--val-samples", type=int, default=30)
ap.add_argument("--time-limit-min", type=float, default=1e9)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
np.random.seed(a.seed); torch.manual_seed(a.seed)
torch.set_num_threads(4)

def rotate(X, Y, t):
    return X * math.cos(np.pi * t) + Y * math.sin(np.pi * t), -X * math.sin(np.pi * t) + Y * math.cos(np.pi * t)

groups_tr, groups_val = [], []
for s in a.sets.split(","):
    X, Y, L, fs = load(s, "A")
    p = np.random.permutation(X.shape[0])           # shuffle, then first trials = validation (as in DNN.train)
    X, Y, L = X[p], Y[p], L[p]
    nv = a.val_samples
    groups_val.append((velocity(X[:nv], Y[:nv]), (L[:nv] > 0).astype(np.float32)))
    Xt, Yt, Lt = X[nv:], Y[nv:], L[nv:]
    xs, ys = [Xt], [Yt]
    for t in np.arange(0.25, 2, 0.5):               # rotation augmentation
        x2, y2 = rotate(Xt, Yt, t); xs.append(x2); ys.append(y2)
    V = np.concatenate([velocity(x, y) for x, y in zip(xs, ys)])
    Ltr = np.tile((Lt > 0).astype(np.float32), (len(xs), 1))
    groups_tr.append((V, Ltr))
    print("set", s, "train", V.shape[0], "val", nv, "saccade fraction %.3f" % Ltr.mean())

def onehot(L):  # (n,T) -> (n,2,T)
    return torch.from_numpy(np.stack([1 - L, L], 1))

net = CausalTCN(2, a.channels)
for m in net.modules():  # weights_init of uneye/functions.py
    if isinstance(m, torch.nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
print("params", sum(p.numel() for p in net.parameters()), "receptive field", net.receptive_field, "samples")
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
Vval = [(torch.from_numpy(v), onehot(l)) for v, l in groups_val]

def val_loss():
    net.eval()
    with torch.no_grad():
        n = sum(v.shape[0] for v, _ in Vval)
        return float(sum(mc_loss(net(v), l) * v.shape[0] for v, l in Vval) / n + a.l2 * sum((p ** 2).sum() for p in net.parameters()))

best, best_w, bad, hist, t0 = None, None, 0, [], time.time()
for epoch in range(1, a.max_iter + 1):
    net.train()
    batches = []
    for g, (V, L) in enumerate(groups_tr):
        idx = np.random.permutation(V.shape[0])
        batches += [(g, idx[i:i + a.batch]) for i in range(0, len(idx), a.batch)]
    np.random.shuffle(batches)
    tl = []
    for g, idx in batches:
        V, L = groups_tr[g]
        out = net(torch.from_numpy(V[idx]))
        loss = mc_loss(out, onehot(L[idx])) + a.l2 * sum((p ** 2).sum() for p in net.parameters())
        opt.zero_grad(); loss.backward(); opt.step(); tl.append(float(loss.detach()))
    vl = val_loss(); hist.append((float(np.mean(tl)), vl))
    if best is None or vl < best:
        best, bad = vl, 0
        best_w = {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        net.load_state_dict(best_w)
        for pg in opt.param_groups: pg["lr"] *= 0.5
    print(f"epoch {epoch:3d} train {hist[-1][0]:.5f} val {vl:.5f} best {best:.5f} bad {bad} lr {opt.param_groups[0]['lr']:.1e} [{(time.time()-t0)/60:.1f} min]", flush=True)
    if bad > 3:
        print("early stopping"); break
    if (time.time() - t0) / 60 > a.time_limit_min:
        print("time limit reached"); break
net.load_state_dict(best_w)
torch.save({"state": net.state_dict(), "channels": a.channels, "hist": hist}, a.out)
print("saved", a.out)
