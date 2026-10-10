#!/usr/bin/env python3
"""Dataset-conditioned BiTCN with FiLM modulation (Perez et al. 2018; adapters of Rebuffi et al. 2017), suggestion 5 of the Antigravity review.
A single bidirectional TCN (2 channels vx, vy, as foundation/lodo_bitcn.py) is trained on the labeled train splits of the four datasets other than the held-out one; each block output y is modulated as
h = h + (1 + W_g e) * y + W_b e, where e is a 16-dimensional learned embedding of the dataset (one per training dataset). On the held-out dataset NOTHING is trained except this 16-dimensional vector:
e is fitted by gradient descent on N labeled trials of the target (N = 0: the mean of the training embeddings), the backbone stays frozen.
Protocol as foundation/conventions.py calib: dataset 1 = adapt on draws of set B, test on set A[:300]; the other datasets adapt on their train subset (night_eval.data()) and test on the common test subset.
Early stopping of the backbone on 10 % held-out trials of the four training datasets (never the held-out dataset).   python foundation/film_bitcn.py [--only d1,d2] -> night/film_bitcn.json
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from foundation import data as FD, compare as CP, night_eval as NE
from foundation.dino1d import feats2
from foundation.train_lodo import augment, DEV
BASE, ED, NIGHT = 480, 16, NE.NIGHT


class FiLMBiTCN(nn.Module):
    def __init__(self, nin=2, ch=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128), ed=ED):
        super().__init__(); self.stem = nn.Conv1d(nin, ch, 5, padding=2); self.stem_bn = nn.BatchNorm1d(ch)
        self.convs = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dilations]); self.bns = nn.ModuleList([nn.BatchNorm1d(ch) for _ in dilations])
        self.wg = nn.ModuleList([nn.Linear(ed, ch) for _ in dilations]); self.wb = nn.ModuleList([nn.Linear(ed, ch) for _ in dilations]); self.head = nn.Conv1d(ch, 1, 1)
        for m in list(self.wg) + list(self.wb): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)              # starts as the plain BiTCN
    def forward(self, x, e):                                    # x (B, T, 2); e (B, ED) or (ED,) -> (B, T) logit
        if e.dim() == 1: e = e[None].expand(len(x), -1)
        h = self.stem_bn(F.relu(self.stem(x.transpose(1, 2))))
        for conv, bn, wg, wb in zip(self.convs, self.bns, self.wg, self.wb):
            y = bn(F.relu(conv(h))); h = h + (1.0 + wg(e))[:, :, None] * y + wb(e)[:, :, None]
        return self.head(h)[:, 0]


def rotate(p, th, mirror):
    c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)); q = np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]], 2)
    return np.stack([q[..., 0], -q[..., 1]], 2) if mirror else q


@torch.no_grad()
def score(net, e, pos, tta=True, bs=20):
    net.eval(); out = []; tr = [(th, mi) for th in ((0, 90, 180, 270) if tta else (0,)) for mi in ((False, True) if tta else (False,))]
    for i in range(0, len(pos), bs):
        p = pos[i:i + bs].astype(np.float64); acc = 0.0
        for th, mi in tr: acc = acc + net(torch.as_tensor(feats2(rotate(p, th, mi), 1000.0), device=DEV), e).float().cpu().numpy()
        out.append(acc / len(tr))
    o = np.concatenate(out); o[~np.isfinite(pos).all(2)] = 0.0; return o


def train_backbone(held, steps=4000, max_minutes=10, lr=1e-3, seed=0):
    rng = np.random.RandomState(seed); torch.manual_seed(seed); names = [k for k in FD.ALL if k != held]; train, val = {}, {}
    for k in names:
        S = FD.load(k, "train"); perm = np.random.RandomState(2).permutation(len(S)); nv = max(len(S) // 10, 20)
        val[k] = FD.Set(k, S.pos[perm[:nv]], S.lab[perm[:nv]], S.coarse); train[k] = (S.pos[perm[nv:]], S.lab[perm[nv:]])
    net = FiLMBiTCN().to(DEV); E = nn.Parameter(0.1 * torch.randn(len(names), ED, device=DEV)); opt = torch.optim.AdamW(list(net.parameters()) + [E], lr=lr, weight_decay=1e-2); path = os.path.join(ROOT, "foundation", "runs", f"film_{held}.pt"); os.makedirs(os.path.dirname(path), exist_ok=True)
    def val_f1():
        r = []
        for j, k in enumerate(names):
            S = val[k]; s = score(net, E[j].detach(), S.pos[:60], tta=False); r.append(CP.event_f1((s > 0) & np.isfinite(S.pos[:60]).all(2), type("S", (), dict(lab=S.lab[:60]))())["ev_f1"])
        return float(np.mean(r))
    best, bad, t0 = -1.0, 0, time.time()
    for it in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 200) * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.98 + 0.02)
        P, L, J = [], [], []
        for _ in range(64):
            j = rng.randint(len(names)); pos, lab = train[names[j]]; i = rng.randint(len(pos)); s0 = rng.randint(0, pos.shape[1] - BASE + 1); P.append(pos[i, s0:s0 + BASE]); L.append(lab[i, s0:s0 + BASE]); J.append(j)
        aug = augment(np.stack(P), rng, True); lab = np.stack(L); x = torch.as_tensor(feats2(aug, 1000.0), device=DEV); y = torch.as_tensor((lab == 1).astype(np.float32), device=DEV)
        w = torch.as_tensor(np.isfinite(aug).all(2) & (lab >= 0), device=DEV).float(); net.train()
        loss = (F.binary_cross_entropy_with_logits(net(x, E[torch.as_tensor(J, device=DEV)]), y, pos_weight=torch.tensor(1.5, device=DEV), reduction="none") * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step()
        if (it + 1) % 500 == 0:
            f1 = val_f1(); improved = f1 > best + 1e-3
            if improved: best, bad = f1, 0; torch.save({"net": net.state_dict(), "E": E.detach().cpu()}, path)
            else: bad += 1
            print(f"[film {held}] step {it + 1}/{steps} loss {float(loss.detach()):.4f} | val F1 {f1:.3f} (best {best:.3f}, patience {bad}/3) | {time.time() - t0:.0f} s", flush=True)
            if bad >= 3 or (time.time() - t0) / 60 > max_minutes: break
    ck = torch.load(path, weights_only=False, map_location=DEV); net.load_state_dict(ck["net"]); net.eval(); return net, ck["E"].to(DEV)


def adapt(net, e0, pos, lab, seed, steps=150, lr=0.05):
    """fit ONLY the 16-dimensional embedding on N labeled trials (backbone frozen, BN in eval mode)"""
    rng = np.random.RandomState(seed); e = nn.Parameter(e0.clone()); opt = torch.optim.Adam([e], lr=lr); net.eval(); b = min(BASE, pos.shape[1])
    for p_ in net.parameters(): p_.requires_grad_(False)                       # only e is trained
    for it in range(steps):
        P, L = [], []
        for _ in range(32): i = rng.randint(len(pos)); s0 = rng.randint(0, pos.shape[1] - b + 1); P.append(pos[i, s0:s0 + b]); L.append(lab[i, s0:s0 + b])
        aug = augment(np.stack(P), rng, True); lab_ = np.stack(L); x = torch.as_tensor(feats2(aug, 1000.0), device=DEV); y = torch.as_tensor((lab_ == 1).astype(np.float32), device=DEV); w = torch.as_tensor(np.isfinite(aug).all(2) & (lab_ >= 0), device=DEV).float()
        loss = (F.binary_cross_entropy_with_logits(net(x, e), y, pos_weight=torch.tensor(1.5, device=DEV), reduction="none") * w).sum() / w.sum(); opt.zero_grad(); loss.backward(); opt.step()
    return e.detach()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); a = ap.parse_args(); OUT = os.path.join(NIGHT, "film_bitcn.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}; d = NE.data()
    for k in FD.ALL:
        if a.only and k not in a.only.split(","): continue
        if f"{k}_0_0" in res and all(f"{k}_{N}_0" in res for N in (3, 10)): continue
        net, E = train_backbone(k); e0 = E.mean(0)
        if k == "d1": setB, setA = FD.load("d1", "test"), FD.load("d1", "train"); calP, calY, teP, teY, plan, draws = setB.pos, setB.lab, setA.pos[:300], setA.lab[:300], (0, 3, 5, 10, 20, 50), 10
        else: A, B = d[k]; calP, calY, teP, teY, plan, draws = A.pos, A.lab, B.pos, B.lab, (0, 3, 10, 30 if k != "andersson" else 20), 5
        S = type("S", (), dict(lab=teY))()
        for N in plan:
            for r in range(1 if N == 0 else draws):
                key = f"{k}_{N}_{r}"
                if key in res: continue
                e = e0 if N == 0 else adapt(net, e0, *(lambda sel: (calP[sel], calY[sel]))(np.random.RandomState(100 * N + r).permutation(len(calP))[:N]), seed=r)
                s = score(net, e, teP); m = CP.event_f1((s > 0) & np.isfinite(teP).all(2), S); res[key] = [m["ev_f1"], m["kappa"]]; json.dump(res, open(OUT, "w"), indent=1, default=float)
                print(f"[film] {key}: F1 {m['ev_f1']:.3f} kappa {m['kappa']:.3f}", flush=True)


if __name__ == "__main__":
    main()
