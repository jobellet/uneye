"""Self-supervised dense embeddings for gaze windows, and the step from embedding to saccade label WITHOUT human labels.

Input features are rotation invariant (the direction of a saccade says nothing about being a saccade):
    ch0 asinh(speed / sigma)                       how fast, relative to the noise of THIS window
    ch1 asinh(d speed/dt / (sigma * fs/10))        acceleration along the movement
    ch2 cos(turn angle) * gate, ch3 sin(turn angle) * gate   curvature of the path (gate -> 0 when there is no movement)
    ch4 valid
Encoder: non-causal dilated 1-D convolutions (all time steps get an embedding that sees +-context).

Objectives (`method`):
    dino     dense DINO: EMA teacher, K prototypes, centering + sharpening, student predicts the teacher's cluster of the SAME time step in another view
    jepa     1-D V-JEPA: predict the teacher's latent of masked time blocks from the visible context (latent space, no reconstruction)
    infonce  dense contrastive: same time step in two views = positive (+ neighbours within `tpos` samples = temporal persistence prior)
Views = noise, gain, small time-warp, dropouts. Nothing here knows what a saccade is.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import detectors as D


# ----------------------------------------------------------------------------- features
def features(pos, fs):
    """(n, T, 5) float32 rotation-invariant features from position windows"""
    v, valid = D.velocity(pos, fs)
    sg = D.robust_sigma(v).mean(2, keepdims=True)                         # (n,1,1)
    sp = np.hypot(v[..., 0], v[..., 1])[..., None]
    sig = np.maximum(sg, 1.0)                                              # deg/s floor so that very clean data do not explode
    c0 = np.arcsinh(sp / sig)
    acc = np.gradient(sp[..., 0], axis=1) * fs
    c1 = np.arcsinh(acc / (sig[..., 0] * fs / 10.0))[..., None]
    ang = np.arctan2(v[..., 1], v[..., 0]); dang = np.diff(ang, axis=1, prepend=ang[:, :1]); dang = (dang + np.pi) % (2 * np.pi) - np.pi
    gate = (sp[..., 0] / (sp[..., 0] + 3 * sig[..., 0]))
    c2, c3 = (np.cos(dang) * gate)[..., None], (np.sin(dang) * gate)[..., None]
    f = np.concatenate([c0, c1, c2, c3, valid[..., None].astype(float)], 2)
    f[~valid] = 0.0; f[..., 4] = valid
    return f.astype(np.float32)


def augment_pos(pos, rng, fs, noise=(0.0, 0.5), gain=(0.8, 1.25), warp=0.1):
    """a random view of position windows: rotation / flip (does nothing to the features but keeps the contract), gain, small time-warp,
    extra gaussian noise (fraction of the window's own noise level)."""
    n, T, _ = pos.shape
    out = pos.copy()
    th = rng.uniform(0, 2 * np.pi, n); R = np.stack([np.cos(th), -np.sin(th), np.sin(th), np.cos(th)], 1).reshape(n, 2, 2)
    out = np.einsum("nij,ntj->nti", R, out) * rng.uniform(*gain, size=(n, 1, 1))
    t = np.arange(T)
    for k in range(n):                                                     # time warp (speed changes slightly, shape stays)
        r = 1.0 + rng.uniform(-warp, warp); tt = np.clip(t * r + rng.uniform(-1, 1), 0, T - 1)
        ok = np.isfinite(out[k]).all(1)
        if ok.sum() > 4:
            for a in range(2): out[k, :, a] = np.interp(tt, t[ok], out[k, ok, a])
            nan = ~np.isfinite(pos[k]).all(1); out[k, nan[np.clip(np.round(tt).astype(int), 0, T - 1)]] = np.nan
    sd = np.sqrt(np.maximum(np.median(np.diff(np.nan_to_num(pos), axis=1) ** 2, axis=(1, 2)), 1e-12))[:, None, None]
    out = out + rng.normal(size=out.shape) * sd * rng.uniform(*noise, size=(n, 1, 1))
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- networks
class DenseEncoder(nn.Module):
    def __init__(self, cin=5, dim=64, depth=6, k=5):
        super().__init__()
        self.inp = nn.Conv1d(cin, dim, 1)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            d = 2 ** (i % 4)
            self.blocks.append(nn.Sequential(nn.Conv1d(dim, dim, k, padding=d * (k - 1) // 2, dilation=d), nn.GELU(), nn.BatchNorm1d(dim),
                                             nn.Conv1d(dim, dim, 1)))
        self.dim = dim
        self.receptive = 1 + sum(2 * (2 ** (i % 4)) * (k - 1) // 2 for i in range(depth))

    def forward(self, x):                                                  # x (B, T, C) -> (B, D, T)
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks: h = h + b(h)
        return h


class Head(nn.Module):
    def __init__(self, dim, out):
        super().__init__(); self.net = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.GELU(), nn.Conv1d(dim, out, 1))
    def forward(self, h): return self.net(h)


def _ema(teacher, student, m):
    with torch.no_grad():
        for pt, ps in zip(teacher.parameters(), student.parameters()): pt.mul_(m).add_(ps.detach(), alpha=1 - m)
        for bt, bs in zip(teacher.buffers(), student.buffers()): bt.copy_(bs)


class SSL:
    """train(win) learns an encoder from unlabeled windows; embed(win) returns (n, T, D) embeddings."""

    def __init__(self, method="dino", dim=64, depth=6, protos=32, device=None, seed=0):
        torch.manual_seed(seed); self.rng = np.random.RandomState(seed)
        self.method, self.dim, self.K = method, dim, protos
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = DenseEncoder(5, dim, depth).to(self.dev)
        self.teacher = copy.deepcopy(self.enc).to(self.dev).requires_grad_(False)
        if method == "dino":
            self.head = Head(dim, protos).to(self.dev); self.thead = copy.deepcopy(self.head).requires_grad_(False)
            self.center = torch.zeros(1, protos, 1, device=self.dev)
        elif method == "jepa":
            self.pred = nn.Sequential(nn.Conv1d(dim + 1, dim, 5, padding=2), nn.GELU(), nn.Conv1d(dim, dim, 5, padding=4, dilation=2), nn.GELU(), nn.Conv1d(dim, dim, 1)).to(self.dev)
            self.head = self.pred
        else:
            self.head = Head(dim, dim).to(self.dev)
        self.hist = []

    def params(self):
        return list(self.enc.parameters()) + list(self.head.parameters())

    # one training step on a batch of windows (numpy, NaN allowed)
    def step(self, pos, fs, opt, it, total):
        T = pos.shape[1]
        a = torch.as_tensor(features(augment_pos(pos, self.rng, fs), fs), device=self.dev)
        b = torch.as_tensor(features(augment_pos(pos, self.rng, fs), fs), device=self.dev)
        m = 0.99 + 0.009 * it / total
        if self.method == "dino":
            hs = self.head(self.enc(a));
            with torch.no_grad(): ht = self.thead(self.teacher(b))
            p_t = F.softmax((ht - self.center) / 0.04, 1)
            loss = -(p_t * F.log_softmax(hs / 0.1, 1)).sum(1).mean()
            loss = loss - (p_t * F.log_softmax(self.head(self.enc(b)) / 0.1, 1)).sum(1).mean() * 0
            with torch.no_grad(): self.center = 0.9 * self.center + 0.1 * ht.mean((0, 2), keepdim=True)
        elif self.method == "jepa":
            B = a.shape[0]; mask = torch.zeros(B, 1, T, device=self.dev)
            for i in range(B):
                for _ in range(3):
                    L = int(self.rng.randint(8, 24)); s = int(self.rng.randint(0, T - L)); mask[i, 0, s:s + L] = 1
            with torch.no_grad(): tgt = F.layer_norm(self.teacher(b).transpose(1, 2), (self.dim,)).transpose(1, 2)
            h = self.enc(a * (1 - mask.transpose(1, 2)))
            pr = self.pred(torch.cat([h, mask], 1))
            loss = (F.smooth_l1_loss(pr, tgt, reduction="none") * mask).sum() / (mask.sum() * self.dim + 1e-6)
        else:
            za = F.normalize(self.head(self.enc(a)), dim=1); zb = F.normalize(self.head(self.enc(b)), dim=1)
            B, D_, T_ = za.shape; n = min(T_, 64); idx = torch.as_tensor(self.rng.choice(T_, n, replace=False), device=self.dev)
            za, zb = za[:, :, idx], zb[:, :, idx]                                   # (B, D, n)
            lg = torch.einsum("bdi,bdj->bij", za, zb) / 0.1                          # same window: pairs of time steps
            tpos = getattr(self, "tpos", 0)
            dist = (idx[:, None] - idx[None, :]).abs()
            pos_m = (dist <= tpos).float()[None].expand(B, -1, -1)
            logp = F.log_softmax(lg, 2)
            loss = -((logp * pos_m).sum(2) / pos_m.sum(2)).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.params(), 1.0); opt.step()
        if self.method in ("dino", "jepa"):
            _ema(self.teacher, self.enc, m)
            if self.method == "dino": _ema(self.thead, self.head, m)
        return float(loss)

    def train(self, win, steps=600, batch=64, lr=2e-3, T=128, tpos=0, verbose=True):
        self.tpos = tpos
        opt = torch.optim.AdamW(self.params(), lr=lr, weight_decay=0.04)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        self.enc.train(); n = len(win)
        for it in range(steps):
            idx = self.rng.randint(0, n, batch); s = self.rng.randint(0, win.pos.shape[1] - T + 1)
            loss = self.step(win.pos[idx, s:s + T], win.fs, opt, it, steps); sched.step()
            self.hist.append(loss)
            if verbose and (it + 1) % max(steps // 5, 1) == 0: print(f"  [{self.method}] step {it + 1}/{steps} loss {np.mean(self.hist[-50:]):.4f}")
        return self

    @torch.no_grad()
    def embed(self, win, bs=128):
        self.enc.eval(); out = []
        for i in range(0, len(win), bs):
            f = torch.as_tensor(features(win.pos[i:i + bs], win.fs), device=self.dev)
            out.append(self.enc(f).transpose(1, 2).cpu().numpy())
        return np.concatenate(out)


# ----------------------------------------------------------------------------- embedding -> label (no human labels)
def kmeans_labels(emb, win, k=6, seed=0, score_fn=None, sub=60000):
    """cluster the embeddings of all valid time steps, then decide which clusters are 'saccade'. The decision uses no label:
    clusters are ordered by their mean speed, and the cut (clusters above it = saccade) is the one with the best intrinsic score
    (`score_fn(labels) -> float`, e.g. the persistence + main-sequence score of intrinsic.py); default: largest speed gap."""
    from sklearn.cluster import MiniBatchKMeans
    n, T, d = emb.shape
    sp, valid = D.speed_of(win.pos, win.fs)
    X = emb.reshape(-1, d); ok = valid.reshape(-1)
    rng = np.random.RandomState(seed); idx = np.where(ok)[0]; idx = rng.choice(idx, min(sub, len(idx)), replace=False)
    km = MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=4096).fit(X[idx])
    cl = km.predict(X).reshape(n, T)
    mean_sp = np.array([np.log10(sp[(cl == c) & valid] + 1).mean() if ((cl == c) & valid).any() else -1 for c in range(k)])
    order = np.argsort(mean_sp)
    best, best_cut = -1e9, None
    for cut in range(1, k):
        sacc_clusters = order[cut:]
        lab = np.isin(cl, sacc_clusters) & valid
        sc = score_fn(lab) if score_fn else float(mean_sp[order[cut]] - mean_sp[order[cut - 1]])
        if sc > best: best, best_cut = sc, cut
    lab = np.isin(cl, order[best_cut:]) & valid
    return lab, dict(clusters=cl, mean_log_speed=mean_sp, cut=best_cut, order=order)


# ----------------------------------------------------------------------------- physics-regularised self-training (noisy student)
class Student(nn.Module):
    """small dense classifier trained on pseudo-labels, with augmented inputs (the labels are a property of the signal, not of the view)"""
    def __init__(self, dim=48, depth=5, init_encoder=None, freeze=False):
        super().__init__()
        self.enc = init_encoder if init_encoder is not None else DenseEncoder(5, dim, depth)
        self.out = nn.Conv1d(self.enc.dim, 1, 1)
        self.freeze = freeze
    def forward(self, x):
        if self.freeze:
            with torch.no_grad(): h = self.enc(x)
        else: h = self.enc(x)
        return self.out(h)[:, 0]


def self_train(win, label, weight, fs, init_encoder=None, freeze=False, steps=500, batch=64, T=128, lr=2e-3, seed=0, device=None, rounds=1, verbose=True):
    """label/weight: pseudo-labels (n,T) bool and (n,T) weights (0 = ignored). round > 1: the student's confident predictions become the next labels
    (noisy student) -- confident = probability < 0.1 or > 0.9, everything else keeps weight 0."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    n = len(win)
    lab, w = label.astype(np.float32), weight.astype(np.float32)
    net = None
    for r in range(rounds):
        net = Student(init_encoder=copy.deepcopy(init_encoder) if init_encoder is not None else None, freeze=freeze).to(dev)
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        net.train()
        for it in range(steps):
            idx = rng.randint(0, n, batch); s = rng.randint(0, win.pos.shape[1] - T + 1)
            pos = win.pos[idx, s:s + T]
            x = torch.as_tensor(features(augment_pos(pos, rng, fs, gain=(1.0, 1.0), warp=0.0, noise=(0.0, 0.3)), fs), device=dev)
            y = torch.as_tensor(lab[idx, s:s + T], device=dev); ww = torch.as_tensor(w[idx, s:s + T], device=dev)
            loss = (F.binary_cross_entropy_with_logits(net(x), y, reduction="none") * ww).sum() / (ww.sum() + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose: print(f"  [self-train round {r + 1}/{rounds}] last loss {float(loss):.4f}")
        if r < rounds - 1:
            p = predict_proba(net, win, fs, dev)
            conf = (p < 0.1) | (p > 0.9)
            lab = (p > 0.5).astype(np.float32); w = (conf * (np.isfinite(win.pos).all(2))).astype(np.float32)
    return net


@torch.no_grad()
def predict_proba(net, win, fs=None, device=None, bs=128):
    dev = device or next(net.parameters()).device; net.eval(); out = []
    for i in range(0, len(win), bs):
        f = torch.as_tensor(features(win.pos[i:i + bs], win.fs), device=dev)
        out.append(torch.sigmoid(net(f)).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- detectors with a fixed decision rule (needed for invariance tests)
class ClusterDetector:
    """embedding -> k-means -> saccade clusters. fit() decides the clusters without labels (best intrinsic score, see kmeans_labels);
    predict() applies the frozen model to any windows, so it can be tested for invariances."""

    def __init__(self, ssl, k=6, seed=0, post=(6.0, 6.0)):
        self.ssl, self.k, self.seed, self.post = ssl, k, seed, post

    def fit(self, win, score_fn=None, sub=60000):
        from sklearn.cluster import MiniBatchKMeans
        emb = self.ssl.embed(win); n, T, d = emb.shape
        sp, valid = D.speed_of(win.pos, win.fs)
        X = emb.reshape(-1, d); idx = np.where(valid.reshape(-1))[0]; idx = np.random.RandomState(self.seed).choice(idx, min(sub, len(idx)), replace=False)
        self.km = MiniBatchKMeans(self.k, random_state=self.seed, n_init=3, batch_size=4096).fit(X[idx])
        cl = self.km.predict(X).reshape(n, T)
        ms = np.array([np.log10(sp[(cl == c) & valid] + 1).mean() if ((cl == c) & valid).any() else -1 for c in range(self.k)])
        self.order = np.argsort(ms); best = (-1e9, 1)
        for cut in range(1, self.k):
            lab = np.isin(cl, self.order[cut:]) & valid
            lab = D.cleanup(lab, win.fs, *self.post)
            sc = score_fn(lab) if score_fn else float(ms[self.order[cut]] - ms[self.order[cut - 1]])
            if sc > best[0]: best = (sc, cut)
        self.cut = best[1]; self.mean_log_speed = ms; self.cluster_scores = best[0]
        return self

    def predict(self, win):
        emb = self.ssl.embed(win); n, T, d = emb.shape
        cl = self.km.predict(emb.reshape(-1, d)).reshape(n, T)
        valid = np.isfinite(win.pos).all(2)
        lab = np.isin(cl, self.order[self.cut:]) & valid & ~D.invalid_zone(valid, win.fs)
        return D.cleanup(lab, win.fs, *self.post)


class StudentDetector:
    def __init__(self, net, thr=0.5, post=(6.0, 6.0)):
        self.net, self.thr, self.post = net, thr, post

    def predict(self, win):
        p = predict_proba(self.net, win); valid = np.isfinite(win.pos).all(2)
        lab = (p > self.thr) & valid & ~D.invalid_zone(valid, win.fs)
        return D.cleanup(lab, win.fs, *self.post)
