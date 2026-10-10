#!/usr/bin/env python3
"""Pre-trained wider U-Net (the U'n'Eye topology, x3 channels, 3 input channels) -> fine-tuned for saccade segmentation with few labels.
Question (owner, 2026-10-10): U'n'Eye saturates with data and generalises poorly; does a bigger U-Net PRE-TRAINED as a masked auto-encoder on lots of unlabeled recordings
beat the retrained U'n'Eye when only N = 10 / 20 / 50 labeled trials are available?  Success criterion: dataset 1, test on 300 set-A trials, event F1 above U'n'Eye
(0.894 / 0.921 / 0.945 for N = 10 / 20 / 50, docs/figs_paper/results_uneye.json); otherwise the idea is dropped.
Pre-training data (no label): archive sources with a rate >= 200 Hz (EMTeC, GazeBase, Lund2013, GazeCom; NOT 360EM, VEDB, DUT-OMRON, EGTEA) + the unlabeled train splits of the five
benchmarks; windows of 500 samples at 1 kHz with at most 1 % missing samples and no tracker spikes (|v| > 60 sigma). Input: vx, vy (feats2) + a mask channel.
Objective: masked span reconstruction of the arcsinh velocity (30 % of the window, spans of 8-40 samples; Huber loss on the masked samples).
  python foundation/unet_ssl.py pool | pretrain [--minutes 25] | finetune [--reps 3]
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from foundation import data as FD, compare as CP
from foundation.dino1d import feats2
from foundation.train_lodo import augment, DEV
RUNS = os.path.join(ROOT, "foundation", "runs"); POOL = os.path.join(RUNS, "unet_pool.npz"); W = 500


def build_pool(n_arc=3000, n_bench=3000, seed=0):
    from free_saccade import data as FS
    rng = np.random.RandomState(seed); P, G = [], []
    src = FS.H5Source(os.path.join(ROOT, "archive", "datasets_by_subject", "datasets_by_subject"), os.path.join(ROOT, "archive", "subject_h5_metadata.csv"), "dataset", 1.0, FS.ARCHIVE_FS, limit=None, clip=FS.ARCHIVE_CLIP)
    for g in ("EMTeC", "GazeBase", "Lund2013", "GazeCom"):
        w = src.sample(n_arc, W, rng, fs=1000.0, groups=(g,), max_missing=0.01).pos; P += list(w); G += [g] * len(w); print(g, len(w), flush=True)
    for k in FD.ALL:
        S = FD.load(k, "test" if k == "d1" else "train"); n, T = S.pos.shape[:2]; got = 0; tries = 0   # labels never read; dataset 1: set B only (set A is the few-label TEST set of finetune(): an audit found its unlabeled positions were in the pool)
        while got < n_bench and tries < 50 * n_bench:
            tries += 1; i, s0 = rng.randint(n), rng.randint(0, T - W + 1); w = S.pos[i, s0:s0 + W]
            if (~np.isfinite(w).all(1)).mean() > 0.01: continue
            P.append(w); G.append(k); got += 1
        print(k, got, flush=True)
    P = np.stack(P).astype(np.float32); G = np.array(G); keep = []
    for i in range(len(P)):                                                                    # drop tracker spikes, infinities and flat windows
        w = P[i].astype(np.float64)
        if not np.isfinite(w).all() and (~np.isfinite(w).all(1)).mean() > 0.01: keep.append(False); continue
        f = feats2(w[None], 1000.0)[0]; keep.append(bool(np.isfinite(f).all() and (np.abs(np.sinh(f)) > 60).mean() < 0.01 and np.abs(f).max() > 0))
    keep = np.array(keep); print("kept", keep.sum(), "of", len(P), {g: int(keep[G == g].sum()) for g in np.unique(G)}); np.savez(POOL, pos=P[keep], grp=G[keep])


class WUNet(nn.Module):
    """the U'n'Eye U-Net (kernel 5, two max-poolings of 5, skip connections) with channels x w; 3 inputs (vx, vy, mask); head = reconstruction (2) or segmentation (1)"""
    def __init__(self, nin=3, w=3, nout=2, ks=5, mp=5):
        super().__init__(); c1, c2 = 10 * w, 20 * w; pd = (ks - 1) // 2
        blk = lambda i, o: nn.Sequential(nn.Conv1d(i, o, ks, padding=pd), nn.ReLU(True), nn.BatchNorm1d(o))
        up = lambda: nn.Sequential(nn.ConvTranspose1d(c2, c2, mp, stride=mp), nn.ReLU(True), nn.BatchNorm1d(c2))
        self.c0, self.c1, self.p = blk(nin, c1), blk(c1, c2), nn.MaxPool1d(mp); self.c2, self.c3 = blk(c2, c2), blk(c2, c2)
        self.up1, self.c4, self.up2, self.c5, self.c6 = up(), blk(2 * c2, c2), up(), blk(2 * c2, c2), blk(c2, c1); self.head = nn.Conv1d(c1, nout, 1)
    def forward(self, x):                                       # (B, T, nin), T multiple of 25 -> (B, nout, T)
        h0 = self.c0(x.transpose(1, 2)); c1 = self.c1(h0); p1 = self.p(c1); c2 = self.c2(p1); c3 = self.c3(self.p(c2))
        c4 = self.c4(torch.cat([p1, self.up1(c3)], 1)); c5 = self.c5(torch.cat([c1, self.up2(c4)], 1)); return self.head(self.c6(c5))


def pad25(x):
    T = x.shape[1]; r = (-T) % 25; return (F.pad(x.transpose(1, 2), (0, r), mode="replicate").transpose(1, 2) if r else x), T


def make_input(pos, mask=None):
    f = feats2(pos, 1000.0); m = np.zeros(f.shape[:2] + (1,), np.float32) if mask is None else mask[..., None].astype(np.float32)
    if mask is not None: f = f * (1 - m)
    return np.concatenate([f, m], 2)


def pretrain(minutes, lr=1e-3, bs=128, seed=0, snap=False):
    """snap=True: no early stopping, saves foundation/runs/unet_snaps/step_<n>.pt at fixed steps (for foundation/unet_recon.py) and never touches unet_pre.pt"""
    z = np.load(POOL); pos = z["pos"]; rng = np.random.RandomState(seed); torch.manual_seed(seed); net = WUNet(nout=2).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    perm = rng.permutation(len(pos)); nv = 300; val = pos[perm[:nv]]; tr = pos[perm[nv:]]; t0 = time.time(); best, bad, hist, it = 1e9, 0, [], 0
    def batch(p, r):
        a = augment(p, r, False); f = feats2(a, 1000.0); m = np.zeros(f.shape[:2], bool)
        for i in range(len(m)):
            while m[i].mean() < 0.3:
                L = r.randint(8, 41); s = r.randint(0, W - L); m[i, s:s + L] = True
        x = np.concatenate([f * (1 - m[..., None]), m[..., None]], 2).astype(np.float32); return torch.as_tensor(x, device=DEV), torch.as_tensor(f, device=DEV), torch.as_tensor(m, device=DEV)
    vx, vf, vm = batch(val, np.random.RandomState(7)); SN = (0, 250, 500, 1000, 2000, 3000)
    if snap: os.makedirs(os.path.join(RUNS, 'unet_snaps'), exist_ok=True); torch.save(net.state_dict(), os.path.join(RUNS, 'unet_snaps', 'step_000000.pt'))
    while True:
        frac = min((time.time() - t0) / (60 * minutes), 1.0)
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 200) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.98 + 0.02)
        x, f, m = batch(tr[rng.randint(0, len(tr), bs)], rng); net.train(); out = net(x).transpose(1, 2)
        loss = (F.smooth_l1_loss(out, f, reduction="none").mean(2) * m).sum() / m.sum(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step(); hist.append(float(loss.detach())); it += 1
        if snap and it in SN: torch.save(net.state_dict(), os.path.join(RUNS, 'unet_snaps', f'step_{it:06d}.pt'))
        if snap and it >= SN[-1]: break
        if it % 250 == 0 and not snap:
            net.eval()
            with torch.no_grad(): o = net(vx).transpose(1, 2); vl = float((F.smooth_l1_loss(o, vf, reduction="none").mean(2) * vm).sum() / vm.sum())
            improved = vl < best - 1e-4
            if improved: best, bad = vl, 0; torch.save(net.state_dict(), os.path.join(RUNS, "unet_pre.pt"))
            else: bad += 1
            print(f"[pretrain] step {it} train {np.mean(hist[-250:]):.4f} val {vl:.4f} (best {best:.4f}{' *saved*' if improved else ''}) {time.time() - t0:.0f} s", flush=True)
            if frac >= 1.0 or bad >= 6: break
    print("[pretrain] done, best val", best)


def finetune_one(sel_pos, sel_lab, init, seed, steps=600, lr=3e-4, bs=32):
    rng = np.random.RandomState(seed); torch.manual_seed(seed); net = WUNet(nout=1).to(DEV)
    if init:
        sd = {k: v for k, v in torch.load(os.path.join(RUNS, "unet_pre.pt"), weights_only=False).items() if not k.startswith("head")}; net.load_state_dict(sd, strict=False)
    n = len(sel_pos); nv = max(n // 5, 2); perm = rng.permutation(n); vi, ti = perm[:nv], perm[nv:]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2); best, bad, state = 1e9, 0, {k: v.clone() for k, v in net.state_dict().items()}
    vp, vl = sel_pos[vi], sel_lab[vi] == 1; vX = torch.as_tensor(make_input(vp), device=DEV); vY = torch.as_tensor(vl.astype(np.float32), device=DEV); vV = torch.as_tensor(np.isfinite(vp).all(2), device=DEV).float()
    for it in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 30) * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.9 + 0.1)
        P, L = [], []
        for _ in range(bs):
            i = ti[rng.randint(len(ti))]; s0 = rng.randint(0, sel_pos.shape[1] - W + 1); P.append(sel_pos[i, s0:s0 + W]); L.append(sel_lab[i, s0:s0 + W] == 1)
        a = augment(np.stack(P), rng, True); x = torch.as_tensor(make_input(a), device=DEV); y = torch.as_tensor(np.stack(L).astype(np.float32), device=DEV); w = torch.as_tensor(np.isfinite(a).all(2), device=DEV).float()
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
        x, T0 = pad25(torch.as_tensor(make_input(pos[i:i + bs]), device=DEV)); out.append(net(x)[:, 0, :T0].cpu().numpy())
    return np.concatenate(out)


def finetune(reps):
    OUT = os.path.join(RUNS, "night", "unet_ssl_finetune.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tr, te = FD.load("d1", "test"), FD.load("d1", "train"); tp, tl = te.pos[:300], te.lab[:300]; S = type("S", (), dict(lab=tl))()
    for N in (10, 20, 50):
        for r in range(reps):
            sel = np.random.RandomState(100 * N + r).permutation(len(tr))[:N]
            for init in (True, False):
                key = f"{'pre' if init else 'scratch'}_{N}_{r}"
                if key in res: continue
                t0 = time.time(); net = finetune_one(tr.pos[sel], tr.lab[sel], init, r); s = score(net, tp); e = CP.event_f1((s > 0) & np.isfinite(tp).all(2), S)
                res[key] = [e["ev_f1"], e["kappa"]]; json.dump(res, open(OUT, "w"), indent=1); print(f"[finetune] {key}: F1 {e['ev_f1']:.3f} kappa {e['kappa']:.3f} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["pool", "pretrain", "finetune"]); ap.add_argument("--minutes", type=float, default=25); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--snap", action="store_true"); a = ap.parse_args()
    {"pool": build_pool, "pretrain": lambda: pretrain(a.minutes, snap=a.snap), "finetune": lambda: finetune(a.reps)}[a.cmd]()
