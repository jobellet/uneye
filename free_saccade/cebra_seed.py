"""Seeded contrastive embedding for saccade segmentation without human labels (after CEBRA, Schneider, Lee & Mathis, Nature 2023).

Idea:
  1. Seeds without any human label: Engbert-Kliegl with a STRICT threshold gives a few obvious saccade samples (the cores of large,
     fast movements); very quiet samples far from any loose detection are fixation seeds. Everything else (saccade edges, small
     saccades, microsaccades, pursuit) is unlabeled.
  2. A dense temporal encoder (one embedding per time step, on the unit hypersphere) is trained with the CEBRA InfoNCE loss and a
     hybrid positive distribution: a positive of a reference time step is either a time neighbour (|dt| <= tpos, other augmented
     view: CEBRA-Time) or, for a seed, another seed of the same class (CEBRA-Behaviour with the seed class as discrete variable).
     Negatives are drawn from all time steps, balanced over {saccade seed, fixation seed, unlabeled}.
  3. Labeling = kNN in the embedding: every time step takes the majority class of its k nearest seeds (cosine similarity), the way
     CEBRA decodes with kNN. The small saccades the strict threshold missed are found by similarity to the seeds.
No human label is used anywhere in seeding, training, hyper-parameters or labeling; human labels only score the result.

Version 2 (after the first quick run): velocities are DETRENDED (minus a 100 ms running median per axis) before seeding, and the
detrended speed is a 6th input channel. Reason: during smooth pursuit (dataset 2) the raw speed is far above the fixation noise,
so the strict threshold marked 46 % of all samples as "saccade seeds" (a label-free red flag: no recording is 46 % saccades).
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, median_filter

from . import detectors as D
from .ssl_models import DenseEncoder, features, augment_pos

# fixed before any result was seen
LAM_SEED, LAM_LOOSE, QUIET_SIGMA, FAR_MS = 10.0, 4.0, 2.0, 10.0
DIM, DEPTH, TAU, TPOS, STEPS, BATCH, T_WIN, K_NN = 32, 6, 0.1, 2, 1500, 64, 128, 25


def detrended_velocity(pos, fs):
    """velocity minus its 100 ms running median per axis: removes smooth pursuit / drift, keeps the short saccades (20-50 ms)"""
    v, valid = D.velocity(pos, fs)
    w = int(0.1 * fs) | 1
    return v - median_filter(v, size=(1, w, 1), mode="nearest"), valid


def _runs(mask, fs, min_ms=6.0):
    return D.cleanup(mask, fs, min_ms=min_ms, merge_ms=0)


def seeds(pos, fs):
    """(n, T) int8: 1 saccade seed, 0 fixation seed, -1 unlabeled. Uses only the signal (Engbert-Kliegl on detrended velocity)."""
    v, valid = detrended_velocity(pos, fs)
    sg = D.robust_sigma(v)                                                    # (n, 1, 2)
    e = ((v / sg) ** 2).sum(2)                                                # squared normalised speed (ellipse units)
    bad = ~valid | D.invalid_zone(valid, fs)
    strict = _runs((e > LAM_SEED ** 2) & ~bad, fs)                             # cores of clear saccades only
    loose = _runs((e > LAM_LOOSE ** 2) & ~bad, fs)
    m = max(int(round(FAR_MS * fs / 1000)), 1)
    near = binary_dilation(loose, structure=np.ones((1, 2 * m + 1), bool))
    quiet = (e < QUIET_SIGMA ** 2) & ~near & ~bad
    s = np.full(pos.shape[:2], -1, np.int8)
    s[quiet] = 0; s[strict] = 1
    return s


def feats(pos, fs):
    """the 5 rotation-invariant features of ssl_models + the detrended speed (relative to its own robust noise)"""
    f = features(pos, fs)
    v, valid = detrended_velocity(pos, fs)
    sg = np.maximum(D.robust_sigma(v).mean(2, keepdims=True), 1.0)
    d = np.arcsinh(np.hypot(v[..., 0], v[..., 1])[..., None] / sg)
    d[~valid] = 0.0
    return np.concatenate([f, d.astype(np.float32)], 2)


class SeededCEBRA:
    def __init__(self, device=None, seed=0, hybrid=True):
        torch.manual_seed(seed); self.rng = np.random.RandomState(seed)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.enc = DenseEncoder(6, 64, DEPTH).to(self.dev)
        self.proj = torch.nn.Conv1d(64, DIM, 1).to(self.dev)
        self.hybrid = hybrid                                                   # False = CEBRA-Time only (ablation: seeds used for labeling only)
        self.hist = []

    def _z(self, f):
        return F.normalize(self.proj(self.enc(f)), dim=1)                     # (B, D, T) on the unit sphere

    def train(self, win, seed_lab, steps=STEPS):
        params = list(self.enc.parameters()) + list(self.proj.parameters())
        opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=0.04)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
        n, Tfull = seed_lab.shape
        has_s = np.where((seed_lab == 1).any(1))[0]
        self.enc.train()
        for it in range(steps):
            # half the windows contain a saccade seed, so that every batch has some (seeds are rare)
            idx = np.r_[self.rng.choice(has_s, BATCH // 2), self.rng.randint(0, n, BATCH - BATCH // 2)] if len(has_s) else self.rng.randint(0, n, BATCH)
            s0 = self.rng.randint(0, Tfull - T_WIN + 1)
            p = win.pos[idx, s0:s0 + T_WIN]; lab = seed_lab[idx, s0:s0 + T_WIN]
            za = self._z(torch.as_tensor(feats(augment_pos(p, self.rng, win.fs), win.fs), device=self.dev))
            zb = self._z(torch.as_tensor(feats(augment_pos(p, self.rng, win.fs), win.fs), device=self.dev))
            # reference time steps, balanced over the three seed groups (away from the window edges)
            ref = []
            for g in (1, 0, -1):
                w, t = np.where(lab[:, TPOS:T_WIN - TPOS] == g); t = t + TPOS
                if len(w): k = self.rng.randint(0, len(w), 128); ref.append(np.stack([w[k], t[k], np.full(128, g)], 1))
            ref = np.concatenate(ref)
            # positives: time neighbour in the other view, or (hybrid, for seeds, half of the time) another seed of the same class
            dt = self.rng.randint(-TPOS, TPOS + 1, len(ref))
            pw, pt = ref[:, 0].copy(), ref[:, 1] + dt
            if self.hybrid:
                for g in (1, 0):
                    sel = np.where((ref[:, 2] == g) & (self.rng.rand(len(ref)) < 0.5))[0]
                    w, t = np.where(lab == g)
                    if len(sel) and len(w): k = self.rng.randint(0, len(w), len(sel)); pw[sel], pt[sel] = w[k], t[k]
            r = za[ref[:, 0], :, ref[:, 1]]; q = zb[pw, :, pt]                     # (N, D)
            # negatives: random time steps of the batch, other view (CEBRA: empirical distribution)
            nw, nt = self.rng.randint(0, len(idx), len(ref)), self.rng.randint(0, T_WIN, len(ref))
            neg = zb[nw, :, nt]
            pos_d = (r * q).sum(1) / TAU; neg_d = r @ neg.T / TAU
            c = neg_d.max(1).values.detach()
            loss = -(pos_d - c).mean() + torch.logsumexp(neg_d - c[:, None], 1).mean()   # CEBRA info_nce
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            self.hist.append(float(loss.detach()))
        return self

    @torch.no_grad()
    def embed(self, win, bs=128):
        self.enc.eval(); out = []
        for i in range(0, len(win), bs):
            out.append(self._z(torch.as_tensor(feats(win.pos[i:i + bs], win.fs), device=self.dev)).transpose(1, 2).cpu())
        return torch.cat(out)                                                  # (n, T, D) CPU tensor

    @torch.no_grad()
    def label(self, win, seed_lab, k=K_NN, bank_size=4000):
        """kNN (cosine) to a class-balanced bank of this recording's own seeds -> p(saccade), labels"""
        z = self.embed(win); n, T, Dm = z.shape
        bank, cls = [], []
        for g in (1, 0):
            w, t = np.where(seed_lab == g)
            if len(w) == 0: continue
            kk = self.rng.choice(len(w), min(bank_size // 2, len(w)), replace=False)
            bank.append(z[w[kk], t[kk]]); cls.append(np.full(len(kk), g))
        bank = torch.cat(bank).to(self.dev); cls = torch.as_tensor(np.concatenate(cls), device=self.dev, dtype=torch.float32)
        zz = z.reshape(-1, Dm); p = torch.empty(zz.shape[0])
        for i in range(0, zz.shape[0], 20000):
            sim = zz[i:i + 20000].to(self.dev) @ bank.T
            top = sim.topk(min(k, bank.shape[0]), 1).indices
            p[i:i + 20000] = cls[top].mean(1).cpu()
        p = p.reshape(n, T).numpy()
        lab = p > 0.5
        v, valid = D.velocity(win.pos, win.fs)
        lab &= valid & ~D.invalid_zone(valid, win.fs)
        return p, D.cleanup(lab, win.fs, min_ms=6.0, merge_ms=0)
