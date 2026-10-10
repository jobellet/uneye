#!/usr/bin/env python3
"""Pre-training tasks matched to saccade physics (suggestion of the Antigravity review), on the U'n'Eye-topology wider U-Net of foundation/unet_ssl.py. Multi-task, no label:
 (1) masked velocity reconstruction (as unet_ssl: 30 % of the window in spans of 8-40 samples; the input velocities are zeroed there + a mask flag),
 (2) masked RELATIVE POSITION: the position inside every hidden span minus the last visible position, arcsinh(. / per-sample noise scale)  (the pre-to-post step displacement),
 (3) masked STRAIGHTNESS at lags 4 and 16 (net displacement / path length of the last lag samples: ~1 for ballistic motion, ~1/sqrt(lag) for diffusion / noise),
 (4) a global "main-sequence" discriminator: in half of the windows that contain a detected saccade, the SMOOTH part of its position excursion is multiplied by g in [0.35, 0.6] or [1.7, 3.0] (the rest of the
     trace is shifted to stay continuous; the high-frequency sensor noise is untouched): same duration, different velocity and amplitude, which violates the amplitude - duration - peak-velocity relation;
     the network says real / altered. No resampling (an interpolated trace would be smoother than a real one and give a trivial cue).
Then the usual few-label protocol (dataset 1: N labeled trials of set B, test on 300 trials of set A, same draws as unet_ssl.py; the from-scratch and the plain masked-velocity rows are those of unet_ssl).
  python foundation/unet_phys.py pretrain [--minutes 25] | finetune [--reps 10]  ->  runs/unet_phys_pre.pt, night/unet_phys_finetune.json
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import binary_dilation
from free_saccade import detectors as D
from foundation import data as FD, compare as CP
from foundation.dino1d import feats2
from foundation.unet_ssl import WUNet, RUNS, POOL, W, finetune_one, score
from foundation.train_lodo import augment, DEV
FS = 1000.0; NOUT = 6; CK = "unet_phys_pre.pt"


class PhysUNet(WUNet):
    def __init__(self):
        super().__init__(nin=3, nout=NOUT); self.disc = nn.Linear(self.head.in_channels, 1)
    def forward(self, x, both=False):
        h0 = self.c0(x.transpose(1, 2)); c1 = self.c1(h0); p1 = self.p(c1); c2 = self.c2(p1); c3 = self.c3(self.p(c2))
        c4 = self.c4(torch.cat([p1, self.up1(c3)], 1)); c5 = self.c5(torch.cat([c1, self.up2(c4)], 1)); c6 = self.c6(c5)
        return (self.head(c6), self.disc(c6.mean(2))[:, 0]) if both else self.head(c6)


def straightness(p, lag):
    """p (B, T, 2) -> (B, T) net displacement over the last `lag` samples / path length over them; 0 for t < lag"""
    step = np.hypot(*np.diff(p, axis=1).transpose(2, 0, 1)); cs = np.concatenate([np.zeros((len(p), 1)), np.cumsum(step, 1)], 1)          # cs[t] = path from sample 0 to t
    T = p.shape[1]; out = np.zeros((len(p), T)); path = cs[:, lag:] - cs[:, :-lag]; disp = np.hypot(*(p[:, lag:] - p[:, :-lag]).transpose(2, 0, 1)); out[:, lag:] = disp / np.maximum(path, 1e-9); return out


def alter(p, rng):
    """p (B, T, 2) filled positions -> (altered positions, label real (B,), has_event (B,)): half of the windows with a detected saccade get the SMOOTH part of its excursion multiplied by g
    (p' = p + (g - 1) * e(t), e = smoothed position minus the onset position; constant after the offset). The sensor noise (the high-frequency part) is left untouched, so noise level is no cue."""
    from scipy.ndimage import gaussian_filter1d
    lab = D.ek(p, FS, lam=6.0); out = p.copy(); real = np.ones(len(p), np.float32); has = np.zeros(len(p), bool)
    for i in range(len(p)):
        runs = [r for r in D.runs_1d(lab[i]) if r[1] - r[0] + 1 >= 8]
        if not runs: continue
        s, e = max(runs, key=lambda r: r[1] - r[0]); has[i] = True
        if rng.rand() < 0.5:
            g = rng.uniform(0.35, 0.6) if rng.rand() < 0.5 else rng.uniform(1.7, 3.0); real[i] = 0.0; sm = gaussian_filter1d(p[i], 2.0, axis=0, mode="nearest"); ex = sm[s:e + 1] - sm[s]
            out[i, s:e + 1] = p[i, s:e + 1] + (g - 1.0) * ex; out[i, e + 1:] = p[i, e + 1:] + (g - 1.0) * ex[-1]
    return out, real, has


def vel_inputs(a, m):
    """a (B, T, 2) filled positions, m (B, T) hidden bool -> (visible input (B, T, 3), arcsinh velocity (B, T, 2), noise scale (B, 1, 1)). The noise scale is estimated on VISIBLE samples only."""
    v, _ = D.velocity(a, FS); vis = ~m; vv = np.where(vis[..., None], v, np.nan)
    sg = np.maximum(np.sqrt(np.maximum(np.nanmedian(vv ** 2, 1) - np.nanmedian(vv, 1) ** 2, 1e-12)).mean(1), 1e-9)[:, None, None]
    f = np.arcsinh(v / sg); return np.concatenate([f * vis[..., None], m[..., None]], 2).astype(np.float32), f, sg


def make_mask(B, T, rng):
    m = np.zeros((B, T), bool)
    for i in range(B):
        while m[i].mean() < 0.3: L = rng.randint(8, 41); s = rng.randint(3, T - L - 3); m[i, s:s + L] = True
    return m, binary_dilation(m, structure=np.ones((1, 5), bool))      # (spans, blanked region) -- the 5-point velocity at t uses positions t-2..t+2: blank 2 more samples on each side so that no visible input depends on a hidden position


def make_batch(pos, rng):
    a = augment(pos, rng, False); a, _ = D.fill(a); a, real, has = alter(a, rng); B, T, _ = a.shape
    _, m = make_mask(B, T, rng); inp, f, sg = vel_inputs(a, m); rel = np.zeros((B, T, 2))
    for i in range(B):
        for s, e in D.runs_1d(m[i]): rel[i, s:e + 1] = a[i, s:e + 1] - a[i, s - 1]
    rel_n = np.arcsinh(rel / (sg / FS)); st4, st16 = straightness(a, 4), straightness(a, 16)
    tgt = np.concatenate([f, rel_n, st4[..., None], st16[..., None]], 2).astype(np.float32)
    lw = np.ones((B, T, NOUT), np.float32); lw[:, :16, 5] = 0; lw[:, :4, 4] = 0                                            # straightness undefined before `lag` samples
    return [torch.as_tensor(x, device=DEV) for x in (inp, tgt, m, lw, real, has)]


def losses(net, batch):
    inp, tgt, m, lw, real, has = batch; dense, glob = net(inp, both=True); mm = m[..., None].float() * lw
    dense_loss = (F.smooth_l1_loss(dense.transpose(1, 2), tgt, reduction="none") * mm).sum() / mm.sum().clamp(min=1)
    hm = has.float(); disc = (F.binary_cross_entropy_with_logits(glob, real, reduction="none") * hm).sum() / hm.sum().clamp(min=1)
    return dense_loss, disc


def pretrain(minutes, lr=1e-3, bs=96, seed=0):
    z = np.load(POOL); pos = z["pos"]; rng = np.random.RandomState(seed); torch.manual_seed(seed); net = PhysUNet().to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    perm = rng.permutation(len(pos)); val, tr = pos[perm[:300]], pos[perm[300:]]; vb = make_batch(val, np.random.RandomState(7)); t0 = time.time(); best, bad, hist, it = 1e9, 0, [], 0
    while True:
        frac = min((time.time() - t0) / (60 * minutes), 1.0)
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 200) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.98 + 0.02)
        batch = make_batch(tr[rng.randint(0, len(tr), bs)], rng); net.train(); dl, dc = losses(net, batch); loss = dl + 0.5 * dc
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step(); hist.append((float(dl.detach()), float(dc.detach()))); it += 1
        if it % 250 == 0:
            net.eval()
            with torch.no_grad(): vdl, vdc = losses(net, vb)
            vl = float(vdl + 0.5 * vdc); improved = vl < best - 1e-4
            if improved: best, bad = vl, 0; torch.save(net.state_dict(), os.path.join(RUNS, CK))
            else: bad += 1
            h = np.mean(hist[-250:], 0); print(f"[phys pretrain] step {it} train dense {h[0]:.4f} disc {h[1]:.3f} | val dense {float(vdl):.4f} disc {float(vdc):.3f} (best total {best:.4f}{' *saved*' if improved else ''}) {time.time() - t0:.0f} s", flush=True)
            if frac >= 1.0 or bad >= 6: break
    print("[phys pretrain] done, best val total", best)


def finetune(reps):
    OUT = os.path.join(RUNS, "night", "unet_phys_finetune.json"); os.makedirs(os.path.dirname(OUT), exist_ok=True); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tr, te = FD.load("d1", "test"), FD.load("d1", "train"); tp, tl = te.pos[:300], te.lab[:300]; S = type("S", (), dict(lab=tl))()
    for N in (10, 20, 50):
        for r in range(reps):
            key = f"phys_{N}_{r}"
            if key in res: continue
            sel = np.random.RandomState(100 * N + r).permutation(len(tr))[:N]; t0 = time.time(); net = finetune_one(tr.pos[sel], tr.lab[sel], True, r, ckpt=CK); s = score(net, tp); e = CP.event_f1((s > 0) & np.isfinite(tp).all(2), S)
            res[key] = [e["ev_f1"], e["kappa"]]; json.dump(res, open(OUT, "w"), indent=1); print(f"[phys finetune] {key}: F1 {e['ev_f1']:.3f} kappa {e['kappa']:.3f} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["pretrain", "finetune", "selftest"]); ap.add_argument("--minutes", type=float, default=25); ap.add_argument("--reps", type=int, default=10); a = ap.parse_args()
    if a.cmd == "selftest":
        z = np.load(POOL)["pos"][:48]; b = make_batch(z, np.random.RandomState(0)); inp, tgt, m, lw, real, has = [x.cpu().numpy() for x in b]
        print("shapes", inp.shape, tgt.shape, "| real fraction among windows with an event %.2f" % (real[has > 0].mean() if has.any() else -1), "| windows with an event %d / %d" % (has.sum(), len(has)))
        print("target ranges (min, max) per channel:", np.round(tgt.min((0, 1)), 2).tolist(), np.round(tgt.max((0, 1)), 2).tolist())
        net = PhysUNet().to(DEV); d, c = losses(net, b); print("untrained losses dense %.3f disc %.3f" % (float(d.detach()), float(c.detach())))
        # no-leak check on the REAL input function: alter the hidden positions arbitrarily, no visible input value may change
        pf, _ = D.fill(z[:12].astype(np.float64)); m0, mk = make_mask(12, pf.shape[1], np.random.RandomState(1)); i1, _, _ = vel_inputs(pf, mk); pf2 = pf.copy(); pf2[m0] += np.random.RandomState(2).normal(0, 3.0, pf2[m0].shape); i2, _, _ = vel_inputs(pf2, mk)
        print("max change of the visible inputs when the hidden positions are altered:", float(np.abs(i1 - i2)[~mk].max()), "| hidden fraction %.2f" % mk.mean())
    else: {"pretrain": lambda: pretrain(a.minutes), "finetune": lambda: finetune(a.reps)}[a.cmd]()
