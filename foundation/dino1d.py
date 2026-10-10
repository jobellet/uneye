#!/usr/bin/env python3
"""DINO for eye traces (Caron et al. 2021): self-distillation with no labels, a transformer on patches of the (unit-free, velocity-only) trace,
and the self-attention map of the [CLS] token as a candidate saccade segmentation that EMERGES, as the object masks emerge in DINO's ViT.

Training uses archive/ (EMTeC, GazeBase, GazeCom, Lund2013) AND the unlabeled train splits of datasets 1-4 and Andersson (owner's allowance); the benchmarks' test splits are used for evaluation only, and only
to score the attention map (human labels never enter training, the crops or the monitoring of the loss).
- Input per sample: the 6 channels of free_saccade/cebra_seed.py::feats (speed, acceleration, direction change, validity, detrended speed): all
  velocities normalised by the robust noise of the window, rotation invariant: no absolute position.
- Patches of P = 4 samples (4 ms) -> tokens; [CLS] token; sinusoidal positions inside the crop; pre-LN transformer, depth 6, dim 192, 6 heads.
- DINO: student sees 2 global crops (336 samples) and 6 local crops (84 samples) of an augmented 480-sample window (rotation, gain, noise,
  simulated lower sampling rate, added pursuit: foundation/train_lodo.py::augment, a different draw per global view); the EMA teacher sees the 2 global crops;
  cross-entropy between the centred + sharpened teacher distribution (temperature 0.04) and the student's (0.1) over K = 512 prototypes.
Evaluation (foundation/dino1d.py --eval): last-layer attention of [CLS] to the patch tokens (teacher), per head and mean, averaged over sliding
windows, repeated to samples; scored by the AUC against the human saccade labels (threshold-free), by event F1 / kappa at the best threshold
(chosen WITH labels: an optimistic bound) and after the universal HMM (fitted on the log attention of archive/ windows, no label).
Run from the repository root:  python foundation/dino1d.py --train [--steps 4000]   |   python foundation/dino1d.py --eval
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from foundation.train_lodo import augment, DEV
from free_saccade import data as FS
from free_saccade import detectors as D
from free_saccade.cebra_seed import feats as feats6


def feats(pos, fs):
    """the 6 rotation-invariant channels of cebra_seed.feats + the two velocity components (vx, vy) / robust noise, arcsinh-compressed:
    NOT invariant to rotation (like U'n'Eye's input); the random rotations of the augmentation make the network learn it. They let it integrate the
    displacement along a direction, i.e. see the position jump between before and after a saccade (bidirectional attention, as for an image)."""
    p = np.asarray(pos, np.float64); f6 = feats6(p, fs)
    v, valid = D.velocity(p, fs); sg = np.maximum(D.robust_sigma(v).mean(2, keepdims=True), 1e-9)
    vv = np.arcsinh(v / sg); vv[~valid] = 0.0
    return np.concatenate([f6, vv.astype(np.float32)], 2)

def feats2(pos, fs):
    """only the two velocity components (vx, vy) / robust noise, arcsinh-compressed (the vv channels of feats): the minimal input"""
    p = np.asarray(pos, np.float64); v, valid = D.velocity(p, fs); sg = np.maximum(D.robust_sigma(v).mean(2, keepdims=True), 1e-9)
    vv = np.arcsinh(v / sg); vv[~valid] = 0.0; return vv.astype(np.float32)


P, D_MODEL, DEPTH, HEADS, K_PROTO, N_IN = 4, 192, 6, 6, 512, 8
BASE, G_LEN, L_LEN, N_LOCAL = 480, 336, 84, 6


def sincos(n, d):
    pos = np.arange(n)[:, None]; i = np.arange(d // 2)[None]; ang = pos / (10000 ** (2 * i / d))
    return torch.as_tensor(np.concatenate([np.sin(ang), np.cos(ang)], 1), dtype=torch.float32)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h; self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, want_attn=False):
        B, N, D = x.shape; q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if want_attn:
            a = ((q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)).softmax(-1); o = a @ v
        else:
            a = None; o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(o.transpose(1, 2).reshape(B, N, D)); return x + self.mlp(self.n2(x)), a


class ViT1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv1d(N_IN, D_MODEL, P, stride=P); self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL)); nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(D_MODEL, HEADS) for _ in range(DEPTH)]); self.norm = nn.LayerNorm(D_MODEL)
        self.register_buffer("pos", sincos(256, D_MODEL), persistent=False)
    def forward(self, x, want_attn=False):                     # x (B, T, 6) -> cls (B, D), attention of the last layer (B, H, 1 + N, 1 + N)
        t = self.patch(x.transpose(1, 2)).transpose(1, 2); n = t.shape[1]
        h = torch.cat([self.cls.expand(len(t), -1, -1), t + self.pos[:n]], 1); a = None
        for i, b in enumerate(self.blocks): h, a = b(h, want_attn and i == len(self.blocks) - 1)
        return self.norm(h)[:, 0], a


class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(D_MODEL, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 128))
        self.W = nn.Parameter(torch.randn(K_PROTO, 128) * 0.1)
    def forward(self, x): return F.normalize(self.mlp(x), dim=-1) @ F.normalize(self.W, dim=-1).T


class DINO(nn.Module):
    def __init__(self): super().__init__(); self.vit = ViT1D(); self.head = Head()
    def forward(self, x): return self.head(self.vit(x)[0])


def archive_windows(n_per, seed=0):
    src = FS.H5Source(os.path.join(ROOT, "archive", "datasets_by_subject", "datasets_by_subject"), os.path.join(ROOT, "archive", "subject_h5_metadata.csv"),
                      "dataset", 1.0, FS.ARCHIVE_FS, limit=None, clip=FS.ARCHIVE_CLIP)
    W = src.sample(n_per, BASE, np.random.RandomState(seed), fs=1000.0, groups=("EMTeC", "GazeBase", "GazeCom", "Lund2013"))
    return W.pos, W.group


def training_pool(n_arc=1500, n_bench=1500, seed=0):
    """unlabeled windows: archive/ (4 sources) AND the train splits of datasets 1-4 and Andersson (their labels are never read)"""
    from foundation import data as FD
    rng = np.random.RandomState(seed); P, G = [], []
    pos, grp = archive_windows(n_arc, seed); P += list(pos); G += list(grp)
    for k in FD.ALL:
        S = FD.load(k, "train"); got = 0; n, T = S.pos.shape[:2]
        while got < n_bench:
            i, s0 = rng.randint(n), rng.randint(0, T - BASE + 1); w = S.pos[i, s0:s0 + BASE]
            if np.isnan(w).any(1).mean() > 0.2: continue
            P.append(w); G.append(k); got += 1
    return np.stack(P).astype(np.float32), np.array(G)


def resample_noise(p, rng):
    """independent noise for every view: smooth the trajectory (Gaussian, 2-4 samples), replace the residual (the noise) by fresh Gaussian noise of the
    same robust level. The two views of a window then share the saccades and the drift but NOT the noise realisation, which DINO would otherwise use
    to recognise the window (the CLS attention then goes to the quiet parts and avoids the saccades: observed in the first run)."""
    from scipy.ndimage import gaussian_filter1d
    p = np.asarray(p, np.float64); nan = ~np.isfinite(p); out = np.empty_like(p)
    for i in range(len(p)):
        q, _ = D.fill(p[i:i + 1]); q = q[0]; sm = gaussian_filter1d(q, rng.uniform(2, 4), axis=0, mode="nearest"); r = q - sm
        sd = 1.4826 * np.median(np.abs(r - np.median(r, 0)), 0) * rng.uniform(0.8, 1.2)
        out[i] = sm + rng.normal(size=q.shape) * sd
    out[nan] = np.nan
    return out


def train(steps, batch=48, seed=0, monitor=None, ckpt=None):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    pos, grp = training_pool()
    print(f"{len(pos)} unlabeled windows of {BASE} samples at 1 kHz:", {g: int((grp == g).sum()) for g in np.unique(grp)}, f"| device {DEV}", flush=True)
    student = DINO().to(DEV); teacher = DINO().to(DEV); teacher.load_state_dict(student.state_dict()); [p.requires_grad_(False) for p in teacher.parameters()]
    LR = 5e-4 * batch / 256                                       # DINO's rule (1e-4 for a batch of 48); 5e-4 collapsed to the uniform output
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.04)
    center = torch.zeros(1, K_PROTO, device=DEV); hist = []; t0 = time.time()
    for it in range(steps):
        lr = LR * min(1.0, (it + 1) / 300) * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.99 + 0.01)
        for g in opt.param_groups: g["lr"] = lr; g["weight_decay"] = 0.04 + 0.36 * it / steps
        mom = 1 - (1 - 0.996) * (math.cos(math.pi * it / steps) + 1) / 2
        p = pos[rng.randint(0, len(pos), batch)]
        views = [torch.as_tensor(feats(resample_noise(augment(p, rng, True), rng), 1000.0), device=DEV) for _ in range(2)]        # two augmented versions of each window
        gl = [v[:, s:s + G_LEN] for v in views for s in [rng.randint(0, BASE - G_LEN + 1)]]
        lc = [views[j % 2][:, s:s + L_LEN] for j in range(N_LOCAL) for s in [rng.randint(0, BASE - L_LEN + 1)]]
        with torch.no_grad(): t_out = [teacher(x) for x in gl]
        s_g = [student(x) for x in gl]; s_l = [student(x) for x in lc]; s_all = s_g + s_l
        t_p = [F.softmax((t - center) / 0.04, -1) for t in t_out]
        loss, n = 0.0, 0
        for i, tp in enumerate(t_p):
            for j, so in enumerate(s_all):
                if j == i: continue
                loss = loss - (tp * F.log_softmax(so / 0.1, -1)).sum(-1).mean(); n += 1
        loss = loss / n
        opt.zero_grad(); loss.backward()
        if it < 300: student.head.W.grad = None                  # DINO freezes the last layer during the first epoch
        torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0); opt.step()
        with torch.no_grad():
            for ps, pt in zip(student.parameters(), teacher.parameters()): pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
            center = 0.9 * center + 0.1 * torch.cat(t_out).mean(0, keepdim=True)
        hist.append(float(loss.detach()))
        if (it + 1) % 250 == 0:
            ent = float(-(t_p[0] * (t_p[0] + 1e-9).log()).sum(-1).mean()); msg = f"step {it + 1}/{steps} loss {np.mean(hist[-250:]):.3f} teacher entropy {ent:.2f} (max {math.log(K_PROTO):.2f}) ({time.time() - t0:.0f} s)"
            if monitor and (it + 1) % 500 == 0: msg += " | " + monitor(teacher)
            print(msg, flush=True)
            if ckpt and (it + 1) % 500 == 0: torch.save({"state": teacher.state_dict(), "steps": it + 1}, ckpt)
    return teacher


# ----------------------------------------------------------------------------- attention maps
@torch.no_grad()
def attention_maps(model, pos, bs=64):
    """(n, T, 2) positions at 1 kHz -> (n, T, HEADS) attention of [CLS] to the patches, averaged over sliding windows (stride 84 samples, central
    crop of 336 samples of 480-sample windows, as in training), each window's map normalised to mean 1; NaN where no window covers."""
    model.eval(); n, T, _ = pos.shape; pad = (BASE - G_LEN) // 2
    padded = np.concatenate([np.full((n, pad, 2), np.nan, np.float32), pos.astype(np.float32), np.full((n, pad + BASE, 2), np.nan, np.float32)], 1)
    starts = list(range(0, T + pad, L_LEN)); acc = np.zeros((n, T + 2 * BASE, HEADS)); cnt = np.zeros((n, T + 2 * BASE, 1))
    for s in starts:
        win = padded[:, s:s + BASE]
        for i in range(0, n, bs):
            f = torch.as_tensor(feats(win[i:i + bs], 1000.0)[:, pad:pad + G_LEN], device=DEV)
            _, a = model.vit(f, want_attn=True); a = a[:, :, 0, 1:]; a = (a / a.sum(-1, keepdim=True) * a.shape[-1]).cpu().numpy()      # (B, H, tokens)
            m = np.repeat(a, P, axis=2).transpose(0, 2, 1)                                                                            # (B, 336, H)
            acc[i:i + bs, s + pad:s + pad + G_LEN] += m; cnt[i:i + bs, s + pad:s + pad + G_LEN] += 1
    out = acc[:, pad:pad + T] / np.maximum(cnt[:, pad:pad + T], 1); out[cnt[:, pad:pad + T, 0] == 0] = np.nan
    return out


def auc_on(model, S, n=80):
    from sklearn.metrics import roc_auc_score
    from foundation import data as FD
    sel = np.random.RandomState(1).permutation(len(S))[:n]; A = attention_maps(model, S.pos[sel]); lab = S.lab[sel]
    m = (lab >= 0) & np.isfinite(A[..., 0]) & np.isfinite(S.pos[sel]).all(2)
    y = (lab == 1)[m]; per_head = [roc_auc_score(y, A[..., h][m]) for h in range(HEADS)]
    return roc_auc_score(y, A.mean(2)[m]), per_head


def make_monitor():
    from foundation import data as FD
    sets = {k: FD.load(k, "test") for k in ("d1", "d2")}
    def mon(teacher):
        r = [auc_on(teacher, S, 40) for S in sets.values()]; teacher.train()
        return "AUC of the CLS attention (mean / best head / worst head) " + ", ".join(f"{k}: {a:.3f}/{max(h):.3f}/{min(h):.3f}" for k, (a, h) in zip(sets, r))
    return mon


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", action="store_true"); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(ROOT, "foundation", "runs", "dino1d.pt")); a = ap.parse_args()
    if a.train:
        teacher = train(a.steps, monitor=make_monitor(), ckpt=a.out); torch.save({"state": teacher.state_dict(), "steps": a.steps}, a.out); print("wrote", a.out)
