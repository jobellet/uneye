#!/usr/bin/env python3
"""Multi-scale input for the pre-trained wider U-Net (foundation/unet_ssl.py): instead of the instantaneous velocity only, the position differences p[n]-p[n-k] for k = 1, 2, 4, 8, 16
(x and y: 10 channels, arcsinh of the difference divided by a robust noise level) + a mask flag. The pre-training reconstructs ALL ten channels where they were hidden.
No-cheating masking: a span of POSITION samples is hidden, and every channel value whose computation window [n-k, n] touches a hidden position is blanked too, so a large-lag
difference that straddles a hidden span cannot reveal the displacement it contains; the noise level (sigma) is estimated on the visible values only. Loss only on blanked values.
Fine-tuning and test protocol identical to foundation/unet_ssl.py (dataset 1, N labeled trials of set B, 300 test trials of set A, same draws), results in night/unet_ms_finetune.json.
  python foundation/unet_ms.py pretrain [--minutes 25] | finetune [--reps 10]
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn.functional as F
from free_saccade import detectors as D
from foundation import data as FD, compare as CP
from foundation.unet_ssl import WUNet, pad25, RUNS, POOL, W
from foundation.train_lodo import augment, DEV
LAGS = (1, 2, 4, 8, 16); NCH = 2 * len(LAGS)


def feats_ms(pos, m=None):
    """pos (n, T, 2) (NaN = invalid), m (n, T) bool hidden positions or None -> input (n, T, 11), target (n, T, 10), loss mask (n, T, 10) = hidden & computable"""
    p, valid = D.fill(np.asarray(pos, np.float64)); n, T, _ = p.shape; X = np.zeros((n, T, NCH), np.float32); Y = np.zeros_like(X); L = np.zeros((n, T, NCH), bool)
    C = np.concatenate([np.zeros((n, 1), int), np.cumsum(m, 1)], 1) if m is not None else None
    for j, k in enumerate(LAGS):
        d = np.zeros((n, T, 2)); d[:, k:] = p[:, k:] - p[:, :-k]; ok = np.zeros((n, T), bool); ok[:, k:] = valid[:, k:] & valid[:, :-k]
        blank = np.zeros((n, T), bool)
        if m is not None: blank[:, k:] = (C[:, k + 1:T + 1] - C[:, :T - k]) > 0           # masked samples in [t-k, t]
        vis = ok & ~blank; sig = np.ones(n)
        for i in range(n):
            if vis[i].sum() > 10:
                v = d[i][vis[i]]; sig[i] = max(np.mean(np.sqrt(np.maximum(np.median(v ** 2, 0) - np.median(v, 0) ** 2, 1e-12))), 1e-9)
        f = np.arcsinh(d / sig[:, None, None]); sl = slice(2 * j, 2 * j + 2)
        X[:, :, sl] = (f * vis[..., None]); Y[:, :, sl] = f; L[:, :, sl] = (blank & ok)[..., None]
    flag = (m if m is not None else np.zeros((n, T), bool))[..., None].astype(np.float32)
    return np.concatenate([X, flag], 2), Y, L


def make_masks(n, T, rng, ratio=0.25):
    m = np.zeros((n, T), bool)
    for i in range(n):
        while m[i].mean() < ratio: Ln = rng.randint(8, 41); s = rng.randint(0, T - Ln); m[i, s:s + Ln] = True
    return m


def pretrain(minutes, lr=1e-3, bs=96, seed=0):
    z = np.load(POOL); pos = z["pos"]; rng = np.random.RandomState(seed); torch.manual_seed(seed); net = WUNet(nin=NCH + 1, nout=NCH).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    perm = rng.permutation(len(pos)); val, tr = pos[perm[:300]], pos[perm[300:]]; t0 = time.time(); best, bad, hist, it = 1e9, 0, [], 0
    def batch(p, r):
        a = augment(p, r, False); m = make_masks(len(a), a.shape[1], r); X, Y, L = feats_ms(a, m); return [torch.as_tensor(v, device=DEV) for v in (X, Y, L)]
    vX, vY, vL = batch(val, np.random.RandomState(7)); lossf = lambda o, Y, L: (F.smooth_l1_loss(o, Y, reduction="none") * L).sum() / L.sum().clamp(min=1)
    while True:
        frac = min((time.time() - t0) / (60 * minutes), 1.0)
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 200) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.98 + 0.02)
        X, Y, L = batch(tr[rng.randint(0, len(tr), bs)], rng); net.train(); loss = lossf(net(X).transpose(1, 2), Y, L.float()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step(); hist.append(float(loss.detach())); it += 1
        if it % 250 == 0:
            net.eval()
            with torch.no_grad(): vl = float(lossf(net(vX).transpose(1, 2), vY, vL.float()))
            improved = vl < best - 1e-4
            if improved: best, bad = vl, 0; torch.save(net.state_dict(), os.path.join(RUNS, "unet_ms_pre.pt"))
            else: bad += 1
            print(f"[ms pretrain] step {it} train {np.mean(hist[-250:]):.4f} val {vl:.4f} (best {best:.4f}{' *saved*' if improved else ''}) {time.time() - t0:.0f} s", flush=True)
            if frac >= 1.0 or bad >= 6: break
    print("[ms pretrain] done, best val", best)


def inp(pos): return feats_ms(pos, None)[0]


def finetune_one(sel_pos, sel_lab, init, seed, steps=600, lr=3e-4, bs=32):
    rng = np.random.RandomState(seed); torch.manual_seed(seed); net = WUNet(nin=NCH + 1, nout=1).to(DEV)
    if init:
        sd = {k: v for k, v in torch.load(os.path.join(RUNS, "unet_ms_pre.pt"), weights_only=False).items() if not k.startswith("head")}; net.load_state_dict(sd, strict=False)
    n = len(sel_pos); nv = max(n // 5, 2); perm = rng.permutation(n); vi, ti = perm[:nv], perm[nv:]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2); best, bad, state = 1e9, 0, None
    vp, vl = sel_pos[vi], sel_lab[vi] == 1; vX = torch.as_tensor(inp(vp), device=DEV); vY = torch.as_tensor(vl.astype(np.float32), device=DEV); vV = torch.as_tensor(np.isfinite(vp).all(2), device=DEV).float()
    for it in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 30) * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.9 + 0.1)
        P, Lb = [], []
        for _ in range(bs):
            i = ti[rng.randint(len(ti))]; s0 = rng.randint(0, sel_pos.shape[1] - W + 1); P.append(sel_pos[i, s0:s0 + W]); Lb.append(sel_lab[i, s0:s0 + W] == 1)
        a = augment(np.stack(P), rng, True); x = torch.as_tensor(inp(a), device=DEV); y = torch.as_tensor(np.stack(Lb).astype(np.float32), device=DEV); w = torch.as_tensor(np.isfinite(a).all(2), device=DEV).float()
        net.train(); loss = (F.binary_cross_entropy_with_logits(net(x)[:, 0], y, pos_weight=torch.tensor(1.5, device=DEV), reduction="none") * w).sum() / w.sum(); opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 25 == 0:
            net.eval()
            with torch.no_grad(): xx, T0 = pad25(vX); lo = net(xx)[:, 0, :T0]; vloss = float((F.binary_cross_entropy_with_logits(lo, vY, reduction="none") * vV).sum() / vV.sum())
            if vloss < best - 1e-4: best, bad = vloss, 0; state = {k: v.clone() for k, v in net.state_dict().items()}
            else: bad += 1
            if bad >= 8: break
    net.load_state_dict(state); return net


@torch.no_grad()
def score(net, pos, bs=20):
    net.eval(); out = []
    for i in range(0, len(pos), bs):
        x, T0 = pad25(torch.as_tensor(inp(pos[i:i + bs]), device=DEV)); out.append(net(x)[:, 0, :T0].cpu().numpy())
    return np.concatenate(out)


def finetune(reps):
    OUT = os.path.join(RUNS, "night", "unet_ms_finetune.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tr, te = FD.load("d1", "test"), FD.load("d1", "train"); tp, tl = te.pos[:300], te.lab[:300]; S = type("S", (), dict(lab=tl))()
    for N in (10, 20, 50):
        for r in range(reps):
            sel = np.random.RandomState(100 * N + r).permutation(len(tr))[:N]
            for init in (True, False):
                key = f"{'pre' if init else 'scratch'}_{N}_{r}"
                if key in res: continue
                t0 = time.time(); net = finetune_one(tr.pos[sel], tr.lab[sel], init, r); s = score(net, tp); e = CP.event_f1((s > 0) & np.isfinite(tp).all(2), S)
                res[key] = [e["ev_f1"], e["kappa"]]; json.dump(res, open(OUT, "w"), indent=1); print(f"[ms finetune] {key}: F1 {e['ev_f1']:.3f} kappa {e['kappa']:.3f} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["pretrain", "finetune", "selftest"]); ap.add_argument("--minutes", type=float, default=25); ap.add_argument("--reps", type=int, default=10); a = ap.parse_args()
    if a.cmd == "selftest":
        z = np.load(POOL)["pos"][:6].astype(np.float64); m = make_masks(6, W, np.random.RandomState(0)); X, Y, L = feats_ms(z, m)
        # no-leak check: changing the hidden positions arbitrarily must not change ANY visible input value
        z2 = z.copy(); z2[m] += np.random.RandomState(1).normal(0, 5.0, z2[m].shape); X2, _, _ = feats_ms(z2, m); vis = np.repeat(~L[:, :, ::2], 2, 2)
        d = np.abs(X[:, :, :NCH] - X2[:, :, :NCH]); print("max change of visible inputs when the hidden positions are altered:", float(d[vis].max()), "| blank fraction per lag:", [round(float(L[..., 2 * j].mean()), 2) for j in range(5)])
    else: {"pretrain": lambda: pretrain(a.minutes), "finetune": lambda: finetune(a.reps)}[a.cmd]()
