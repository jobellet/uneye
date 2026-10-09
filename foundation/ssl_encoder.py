#!/usr/bin/env python3
"""Self-supervised, NON-causal eye-movement encoder (CEBRA-Time style: Schneider, Lee & Mathis, Nature 2023). No label is used.

Trained once on many traces: set A of datasets 1-4, the Andersson trials, and unlabeled recordings of archive/ (EMTeC, GazeBase,
GazeCom, Lund2013), all at 1 kHz. Inputs (free_saccade/cebra_seed.py::feats): velocity-only, unit-free (each window divided by its own
robust noise), including the velocity minus a 100 ms running median. Encoder: dilated symmetric convolutions (past AND future
context, 121 ms receptive field) -> 32-d embedding per sample on the unit hypersphere. Loss: InfoNCE with temperature 0.1;
reference = a time step, positive = a time step within +-2 samples in another augmented view (rotation, gain, noise, simulated
lower sampling rate, added pursuit), negatives = random time steps of the other windows of the batch.

--hard adds, for every reference, a negative from the same recording >= 15 samples away (CEBRA time contrast within a session).
Run from the repository root:  python foundation/ssl_encoder.py [--steps 3000] [--hard] [--out foundation/runs/ssl_encoder.pt]
"""
import argparse, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from foundation import data as FD
from foundation.train_lodo import augment, DEV
from free_saccade import data as FS
from free_saccade.cebra_seed import feats
from free_saccade.ssl_models import DenseEncoder

T_WIN, TAU, TPOS, DIM, DEPTH, CH = 320, 0.1, 2, 32, 8, 64


class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = DenseEncoder(6, CH, DEPTH); self.proj = nn.Conv1d(CH, DIM, 1)
    def forward(self, f):                                   # (B, T, 6) -> (B, DIM, T) unit norm
        return F.normalize(self.proj(self.enc(f)), dim=1)


def training_windows(rng, per_dataset=600, per_archive=600):
    """(N, T_WIN, 2) float32 positions (deg or arbitrary units: the features are unit-free), group names"""
    P, G = [], []
    for k in FD.ALL:
        S = FD.load(k, "train"); n, T = S.pos.shape[:2]; got = 0
        while got < per_dataset:
            i, s = rng.randint(n), rng.randint(0, T - T_WIN + 1); w = S.pos[i, s:s + T_WIN]
            if np.isnan(w).any(1).mean() > 0.2: continue
            P.append(w); G.append(k); got += 1
    src = FS.H5Source(os.path.join(ROOT, "archive", "datasets_by_subject", "datasets_by_subject"), os.path.join(ROOT, "archive", "subject_h5_metadata.csv"),
                      "dataset", 1.0, FS.ARCHIVE_FS, limit=None, clip=FS.ARCHIVE_CLIP)
    W = src.sample(per_archive, T_WIN, rng, fs=1000.0, groups=("EMTeC", "GazeBase", "GazeCom", "Lund2013"))
    P += list(W.pos); G += list(W.group)
    return np.stack(P).astype(np.float32), np.array(G)


def train(P, steps, batch=48, seed=0, hard=False):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    net = Encoder().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.04)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    hist = []; t0 = time.time()
    for it in range(steps):
        net.train()
        p = P[rng.randint(0, len(P), batch)]
        fa = torch.as_tensor(feats(augment(p, rng, True), 1000.0), device=DEV); fb = torch.as_tensor(feats(augment(p, rng, True), 1000.0), device=DEV)
        za, zb = net(fa), net(fb)                                             # (B, D, T)
        nref = 8
        w = np.repeat(np.arange(batch), nref); t = rng.randint(TPOS, T_WIN - TPOS, batch * nref)
        r = za[w, :, t]; q = zb[w, :, t + rng.randint(-TPOS, TPOS + 1, len(w))]
        nw, nt = rng.randint(0, batch, len(w)), rng.randint(0, T_WIN, len(w))
        neg = zb[nw, :, nt]
        if hard:                                              # + one HARD negative per reference: the same recording, 15 or more samples away.
            th = (t + rng.randint(15, T_WIN - 15, len(w))) % T_WIN   # Without it the encoder can win by encoding which window a sample
            neg = torch.cat([neg, zb[w, :, th]])              # comes from (noise level...) instead of the local dynamics.
        pos_d = (r * q).sum(1) / TAU; neg_d = r @ neg.T / TAU
        c = neg_d.max(1).values.detach()
        loss = -(pos_d - c).mean() + torch.logsumexp(neg_d - c[:, None], 1).mean()          # CEBRA info_nce
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        hist.append(float(loss.detach()))
        if (it + 1) % 250 == 0: print(f"step {it + 1}/{steps} loss {np.mean(hist[-250:]):.3f}  ({time.time() - t0:.0f} s)", flush=True)
    return net, hist


@torch.no_grad()
def embed(net, pos, bs=16):
    """(n, T, 2) positions at 1 kHz -> (n, T, DIM) float32 embeddings (whole trials, any length)"""
    net.eval(); out = []
    for i in range(0, len(pos), bs):
        f = torch.as_tensor(feats(pos[i:i + bs], 1000.0), device=DEV)
        out.append(net(f).transpose(1, 2).cpu().numpy())
    return np.concatenate(out)


def load_encoder(path=None):
    path = path or os.path.join(ROOT, "foundation", "runs", "ssl_encoder.pt")
    net = Encoder(); net.load_state_dict(torch.load(path, weights_only=False)["state"]); return net.to(DEV).eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--hard", action="store_true"); ap.add_argument("--out", default=os.path.join(ROOT, "foundation", "runs", "ssl_encoder.pt"))
    a = ap.parse_args()
    rng = np.random.RandomState(0); P, G = training_windows(rng, 100 if a.steps < 500 else 600, 100 if a.steps < 500 else 600)
    print(f"{len(P)} training windows of {T_WIN} samples at 1 kHz from:", {g: int((G == g).sum()) for g in np.unique(G)}, f"| device {DEV}", flush=True)
    net, hist = train(P, a.steps, hard=a.hard)
    torch.save({"state": net.state_dict(), "steps": a.steps, "groups": sorted(set(G.tolist())), "hard_negatives": a.hard}, a.out); print("wrote", a.out, "| loss", np.mean(hist[:50]), "->", np.mean(hist[-50:]))
