#!/usr/bin/env python3
"""Networks trained FOR the causal landing estimate (where will the ongoing saccade end?), compared with the gaze engine's linear tables.
Owner's question (2026-10-13): a generic masked auto-encoder barely helps (foundation/landing_ssl_andersson.py); does a model trained exactly for this task do better,
and which architecture?

Task: at a sample t inside a saccade (human onset `on`, tau = t - on in 0..59 ms), only samples <= t are visible; predict the 2-D vector landing - p(t),
landing = position at the human offset. Rows are the same as foundation/landing_error_nonlinear.py (amplitude >= 0.5 deg, d > 0.02 deg, finite data around).
Train: Andersson 'train' split (RA-only trials), 10 % of its trials held out for early stopping. Test: Andersson 'test' split (two-coder trials, RA labels), read once.
Optional --more adds the set-A saccades of datasets 1 and 2 (1 kHz) to the training data.
Inputs of the sequence models (last H samples): vx, vy (deg/ms), valid, displacement since the onset (deg / 10), in-saccade flag, tau / 50. Augmentation: random rotation + mirror.
Models: tcn (causal dilated convolutions), gru, transformer (last token), mlp (last 64 samples), unet_pre (the masked-auto-encoder U-Net of unet_ssl.py, fine-tuned:
the 100 samples after t are hidden and its predicted velocity there is integrated; + an in-saccade channel), unet_scratch (same, random init).
Weights: foundation/runs/landing_<model>[_more].pt (best validation state). Baselines on the same rows: 'stays here', linear tables refit on the same training rows (the C++ form).
  python foundation/landing_train.py [--models tcn,gru,transformer,mlp,unet_pre,unet_scratch] [--more] [--steps 6000]
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
from foundation import data as FD, unet_ssl as U
from foundation.dino1d import feats2
from foundation.train_lodo import DEV
H, SPAN, UVIS = 256, 100, 225          # sequence-model history; U-Net: 225 visible + 100 hidden = 325 samples (multiple of 25)
PAD = max(H, UVIS) + 14
CTX = KIN = WEIGHT = TGT = False
HIST = False        # subject history: sp.HIST (n, 51) = last 8 saccades of the same recording (dx, dy, amplitude, peak speed, duration, valid) + 3 summaries (mean amplitude, mean peak speed / amplitude, count)
CAND = 0            # number of candidate-target slots (GazeBase RAN with distractors); sp.CAND (n, CAND, 2) = candidate positions relative to the start, true target first; b['K'] valid slots per row          # set by the command line: --ctx (no rotation augmentation; absolute positions + direction harmonics), --kin (Savitzky-Golay v, a, j along the chord), --weight (tau-weighted loss + auxiliary remaining-duration head)
NIN = 7
import scipy.signal as _sg
_SG = [np.asarray(_sg.savgol_coeffs(9, 3, deriv=k, use='dot', pos=8), np.float64) for k in (1, 2, 3)]

def rows_of(S, trials=None):
    pos = S.pos.astype(np.float64); lab = S.lab; R = []
    for i in (range(len(pos)) if trials is None else trials):
        X, Y = pos[i, :, 0], pos[i, :, 1]; dd = np.diff(np.r_[0, (lab[i] == 1).astype(int), 0])
        for on, off in zip(np.where(dd == 1)[0], np.where(dd == -1)[0] - 1):
            if on < 6 or off - on < 6 or off + 5 >= len(X) or not (np.isfinite(X[on-6:off+6]).all() and np.isfinite(Y[on-6:off+6]).all()): continue
            p0 = np.array([X[on-1], Y[on-1]])
            if np.linalg.norm(np.array([X[off], Y[off]]) - p0) < 0.5: continue
            for t in range(on, min(off, on + 59) + 1):
                if np.linalg.norm(np.array([X[t], Y[t]]) - p0) > 0.02: R.append((i, on, off, t))
    return np.array(R, np.int64)

class Split:
    def __init__(self, S, trials=None):
        self.R = rows_of(S, trials); n, T = S.pos.shape[:2]
        self.P = np.full((n, T + PAD, 2), np.nan); self.P[:, PAD:] = S.pos      # front padding so that every window exists
    def batch(self, idx, rng=None):
        i, on, off, t = self.R[idx].T; tp, onp, offp = t + PAD, on + PAD, off + PAD
        land = self.P[i, offp]; cur = self.P[i, tp]; p0 = self.P[i, onp - 1]; y = land - cur
        aug = rng is not None and not CTX; rot = rng.uniform(0, 2 * np.pi, len(idx)) if aug else np.zeros(len(idx)); mir = rng.choice([-1.0, 1.0], len(idx)) if aug else np.ones(len(idx))
        K = rng.choice([1, 2, 4, 8], len(idx)) if (rng is not None and CAND) else np.full(len(idx), getattr(self, 'KFIX', 1))
        return dict(i=i, tp=tp, onp=onp, y=y, p0=p0, cur=cur, rot=rot, mir=mir, tau=(t - on).astype(float), dur=((off - t) / 50.0).astype(float), K=K)

def rotate(v, b):                       # v (B, ..., 2)
    c, s = np.cos(b["rot"]), np.sin(b["rot"]); sh = (-1,) + (1,) * (v.ndim - 2)
    x = v[..., 0] * b["mir"].reshape(sh); y = v[..., 1]
    return np.stack([c.reshape(sh) * x - s.reshape(sh) * y, s.reshape(sh) * x + c.reshape(sh) * y], -1)

def seq_input(sp, b, L=H):
    k = np.arange(-L + 1, 1); idx = b["tp"][:, None] + k[None]                 # (B, L)
    P = sp.P[b["i"][:, None], idx]                                            # (B, L, 2)
    Pm = sp.P[b["i"][:, None], idx - 1]; v = P - Pm; valid = np.isfinite(v).all(2)
    v = np.where(valid[..., None], v, 0.0)
    ins = (idx >= b["onp"][:, None]).astype(float); disp = np.where(ins[..., None] > 0, P - b["p0"][:, None], 0.0); disp = np.nan_to_num(disp)
    tau = np.where(ins > 0, (idx - b["onp"][:, None]) / 50.0, 0.0)
    v, disp = rotate(v, b), rotate(disp, b)
    parts = [v, valid[..., None], disp / 10.0, ins[..., None], tau[..., None]]
    if CTX or KIN:
        chord = b["cur"] - b["p0"]; dn = np.maximum(np.linalg.norm(chord, axis=1), 1e-6); u = chord / dn[:, None]; th = np.arctan2(chord[:, 1], chord[:, 0])
    if CTX:
        ab = np.nan_to_num(P) / 10.0; Lw = len(k)
        har = np.stack([np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)], 1)[:, None, :].repeat(Lw, 1)
        parts += [ab, np.nan_to_num(b["p0"])[:, None, :].repeat(Lw, 1) / 10.0, har]
    if KIN:
        ext = np.arange(-L + 1 - 8, 1); Pe = np.nan_to_num(sp.P[b["i"][:, None], b["tp"][:, None] + ext[None]])      # (B, L + 8, 2)
        from numpy.lib.stride_tricks import sliding_window_view as swv
        W = swv(Pe, 9, axis=1)                                                  # (B, L, 2, 9)
        proj = lambda c: np.einsum("blcw,w,bc->bl", W, c, u)
        parts += [proj(_SG[0])[..., None], proj(_SG[1])[..., None] * 5.0, proj(_SG[2])[..., None] * 25.0]
    if TGT:                                                                     # the known target of the saccade, relative to the start position (GazeBase RAN), rotated like the rest
        tv = rotate(sp.TV[b["i"]], b); Lw = len(k)
        parts += [(tv / 10.0)[:, None, :].repeat(Lw, 1), (np.linalg.norm(tv, axis=1) / 10.0)[:, None, None].repeat(Lw, 1)]
    if HIST:                                                                     # what the same subject did before: its previous saccade vectors are rotated like the current one
        hv = sp.HIST[b["i"]].copy(); Hs = hv[:, :48].reshape(len(hv), 8, 6); Hs[..., 0:2] = rotate(Hs[..., 0:2], b); hv[:, :48] = Hs.reshape(len(hv), 48); parts += [hv[:, None, :].repeat(len(k), 1)]
    if CAND:                                                                     # candidate targets, each a (cx, cy, |c|, cos to the current chord, valid); the K valid ones sorted by that cosine
        C = rotate(sp.CAND[b["i"]][:, :CAND], b); Lw = len(k); chord = b["cur"] - b["p0"]; ch = rotate(chord[:, None, :], b)[:, 0]; chn = np.maximum(np.linalg.norm(ch, axis=1), 1e-6)
        valid = np.arange(CAND)[None, :] < b["K"][:, None]; nc = np.maximum(np.linalg.norm(C, axis=2), 1e-6); cs = (C * ch[:, None, :]).sum(2) / (nc * chn[:, None])
        order = np.argsort(np.where(valid, -cs, 9.0), axis=1); take = lambda a: np.take_along_axis(a, order, 1)
        cf = np.stack([take(C[..., 0]) / 10.0, take(C[..., 1]) / 10.0, take(nc) / 10.0, take(cs), take(valid.astype(float))], 2) * take(valid.astype(float))[..., None]   # (B, CAND, 5)
        parts += [cf.reshape(len(cf), 1, -1).repeat(Lw, 1)]
    x = np.concatenate(parts, 2)
    return torch.as_tensor(x, dtype=torch.float32, device=DEV)

def unet_input(sp, b):
    k = np.arange(-UVIS + 1, SPAN + 1); idx = b["tp"][:, None] + k[None]; idx = np.minimum(idx, b["tp"][:, None])   # the hidden future = the last visible position (no leak, zero velocity)
    P = rotate(sp.P[b["i"][:, None], idx], b); P = np.where(np.isfinite(P), P, np.nan)
    f = feats2(P, 1000.0)                                                     # (B, 325, 2) arcsinh(v / noise)
    from free_saccade import detectors as D
    vv, _ = D.velocity(np.nan_to_num(P, nan=0.0), 1000.0); sg = np.maximum(D.robust_sigma(vv).mean(2, keepdims=True), 1e-9)   # (B,1,1) noise scale to undo the normalisation
    m = np.zeros(f.shape[:2], np.float32); m[:, UVIS:] = 1.0
    full = b["tp"][:, None] + np.arange(-UVIS + 1, SPAN + 1)[None]; ins = (full >= b["onp"][:, None]).astype(np.float32)
    x = np.concatenate([f * (1 - m[..., None]), m[..., None], ins[..., None]], 2)
    return torch.as_tensor(x, dtype=torch.float32, device=DEV), torch.as_tensor(sg[:, 0, :], dtype=torch.float32, device=DEV)

class TCN(nn.Module):
    def __init__(self, c=64, nin=None):
        nin = nin or NIN
        super().__init__(); self.inp = nn.Conv1d(nin, c, 1); self.ds = [1, 2, 4, 8, 16, 32, 64]
        self.convs = nn.ModuleList(nn.Conv1d(c, c, 3, dilation=d) for d in self.ds); self.norms = nn.ModuleList(nn.GroupNorm(8, c) for _ in self.ds); self.out = nn.Linear(c, 3)
    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for d, cv, nm in zip(self.ds, self.convs, self.norms): h = h + F.gelu(nm(cv(F.pad(h, (2 * d, 0)))))
        return self.out(h[:, :, -1])
class GRU(nn.Module):
    def __init__(self, c=96, nin=None, L=128):
        nin = nin or NIN; super().__init__(); self.L = L; self.g = nn.GRU(nin, c, 2, batch_first=True); self.out = nn.Linear(c, 3)
    def forward(self, x): o, _ = self.g(x[:, -self.L:]); return self.out(o[:, -1])
class Transf(nn.Module):
    def __init__(self, d=64, nin=None, L=128):
        nin = nin or NIN; super().__init__(); self.L = L; self.emb = nn.Linear(nin, d); self.pos = nn.Parameter(torch.zeros(1, L, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True), 3); self.out = nn.Linear(d, 3)
    def forward(self, x): h = self.enc(self.emb(x[:, -self.L:]) + self.pos); return self.out(h[:, -1])
class MLP(nn.Module):
    def __init__(self, nin=None, L=64):
        nin = nin or NIN; super().__init__(); self.L = L; self.f = nn.Sequential(nn.Flatten(), nn.Linear(nin * L, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 3))
    def forward(self, x): return self.f(x[:, -self.L:])
class UNetLand(nn.Module):                                                   # WUNet with a 4th input channel; landing = integral of the predicted velocity over the hidden span
    def __init__(self, pre):
        super().__init__(); self.u = U.WUNet(nin=4, nout=2)
        if pre:
            sd = torch.load(os.path.join(U.RUNS, "unet_pre.pt"), map_location="cpu", weights_only=False); w = sd["c0.0.weight"]
            sd["c0.0.weight"] = torch.cat([w, torch.zeros_like(w[:, :1])], 1); self.u.load_state_dict(sd)
    def forward(self, x, sg):
        out = self.u(x).transpose(1, 2)[:, UVIS:]                              # (B, SPAN, 2) arcsinh(v / noise)
        return (torch.sinh(out.clamp(-12, 12)) * sg[:, None, :]).sum(1) / 1000.0   # deg

def make(name):
    return {"tcn": TCN, "gru": GRU, "transformer": Transf, "mlp": MLP}[name]().to(DEV) if name in ("tcn", "gru", "transformer", "mlp") else UNetLand(name == "unet_pre").to(DEV)

def pred(*a): pass
def predict(net, name, sp, idx, rng=None):
    b = sp.batch(idx, rng); y = rotate(b["y"], b)
    if name.startswith("unet"): x, sg = unet_input(sp, b); out = net(x, sg)
    else: out = net(seq_input(sp, b))
    pred.aux = (torch.as_tensor(b["dur"], dtype=torch.float32, device=DEV), torch.as_tensor(b["tau"], dtype=torch.float32, device=DEV))
    return out, torch.as_tensor(y, dtype=torch.float32, device=DEV)

def train(name, tr, va, steps, seed=0):
    torch.manual_seed(seed); rng = np.random.RandomState(seed); net = make(name); bs = 128 if name.startswith("unet") else 256
    lr = 3e-4 if name == "unet_pre" else 1e-3; opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    best, bad, state, t0 = 1e9, 0, None, time.time(); vidx = np.random.RandomState(1).choice(len(va.R), min(3000, len(va.R)), replace=False)
    for it in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = lr * min(1.0, it / 200)
        net.train(); out, y = predict(net, name, tr, rng.randint(0, len(tr.R), bs), rng)
        if WEIGHT and out.shape[1] == 3:
            dur, tau = pred.aux; w = torch.where(tau < 3, 0.4 * torch.ones_like(tau), torch.where(tau <= 20, torch.ones_like(tau), 0.2 * torch.exp(-(tau - 20) / 15)))
            loss = ((out[:, :2] - y).norm(dim=1) * w).sum() / w.sum() + 0.3 * ((out[:, 2] - dur).abs() * w).sum() / w.sum()
        else: loss = (out[:, :2] - y).norm(dim=1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if it % 250 == 0:
            vl = evaluate(net, name, va, vidx).mean()
            if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else: bad += 1
            print(f"[{name}] step {it} train {float(loss):.3f} val {vl:.3f} best {best:.3f} {time.time() - t0:.0f} s", flush=True)
            if bad >= 6: break
    net.load_state_dict(state); return net, time.time() - t0

def evaluate(net, name, sp, idx=None):
    net.eval(); idx = np.arange(len(sp.R)) if idx is None else idx; errs = []
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            out, y = predict(net, name, sp, idx[s:s + 512]); errs.append((out[:, :2] - y).norm(dim=1).cpu().numpy())
    return np.concatenate(errs)

def linear_tables(tr, te):
    def feat(sp):
        i, on, off, t = sp.R.T; P = sp.P; tp, onp = t + PAD, on + PAD
        p0 = P[i, onp - 1]; cur = P[i, tp]; dsp = cur - p0; d = np.linalg.norm(dsp, axis=1); u = dsp / d[:, None]
        v = np.mean([np.linalg.norm(P[i, tp - k] - P[i, tp - k - 1], axis=1) for k in range(3)], 0)
        vmax = np.array([max(np.mean([np.linalg.norm(P[ii, s - k] - P[ii, s - k - 1]) for k in range(3)]) for s in range(oo, tt + 1)) for ii, oo, tt in zip(i, onp, tp)])
        return t - on, np.c_[d, v, vmax], u, P[i, off + PAD] - cur
    ta, Fa, ua, ya = feat(tr); tb, Fb, ub, yb = feat(te); ra = (ya * ua).sum(1); W = np.zeros((30, 3))
    for b in range(30):
        m = (ta // 2 == b) if b < 29 else (ta // 2 >= 29); W[b] = np.linalg.solve(Fa[m].T @ Fa[m] + 1e-3 * np.eye(3), Fa[m].T @ ra[m])
    r = np.maximum((W[np.minimum(tb // 2, 29)] * Fb).sum(1), 0)
    return np.linalg.norm(r[:, None] * ub - yb, axis=1), np.linalg.norm(yb, axis=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="tcn,gru,transformer,mlp,unet_pre,unet_scratch"); ap.add_argument("--more", action="store_true"); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--ctx", action="store_true"); ap.add_argument("--kin", action="store_true"); ap.add_argument("--weight", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); tag = ("_more" if a.more else "") + a.tag
    CTX, KIN, WEIGHT = a.ctx, a.kin, a.weight; NIN = 7 + (8 if CTX else 0) + (3 if KIN else 0)
    A = FD.load("andersson", "train"); perm = np.random.RandomState(0).permutation(len(A.pos)); nv = len(A.pos) // 10
    tr, va, te = Split(A, perm[nv:]), Split(A, perm[:nv]), Split(FD.load("andersson", "test"))
    if a.more:
        for k in ("d1", "d2"):
            S = Split(FD.load(k, "train")); off = len(tr.P); tr.P = np.concatenate([tr.P, S.P[:, :tr.P.shape[1]] if S.P.shape[1] >= tr.P.shape[1] else np.pad(S.P, ((0, 0), (0, tr.P.shape[1] - S.P.shape[1]), (0, 0)), constant_values=np.nan)]); R = S.R.copy(); R[:, 0] += off; tr.R = np.concatenate([tr.R, R])
    print("rows train / val / test:", len(tr.R), len(va.R), len(te.R), flush=True)
    tau = te.R[:, 3] - te.R[:, 1]; amp = np.linalg.norm(te.P[te.R[:, 0], te.R[:, 2] + PAD] - te.P[te.R[:, 0], te.R[:, 1] + PAD - 1], axis=1)
    lin, stay = linear_tables(tr, te); res = {"stays here": stay, "linear tables": lin}; info = {}
    for name in a.models.split(","):
        net, sec = train(name, tr, va, a.steps); res[name] = evaluate(net, name, te)
        torch.save({"state": net.state_dict(), "name": name, "more": a.more, "H": H, "SPAN": SPAN, "UVIS": UVIS}, os.path.join(U.RUNS, f"landing_{name}{tag}.pt")); info[name] = {"train_s": round(sec), "params": sum(p.numel() for p in net.parameters())}
        print(f"== {name}: test mean error {res[name].mean():.3f} deg ({sec:.0f} s, {info[name]['params']} params)", flush=True)
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]
    table = {k: {f"{lo}-{hi}": float(v[(tau >= lo) & (tau < hi)].mean()) for lo, hi in bins} | {"all": float(v.mean())} for k, v in res.items()}
    print("\nmean landing error (deg) by time since onset (ms):"); print(f"{'':18s}" + "".join(f"{lo:>2d}-{hi:<4d}" for lo, hi in bins) + "  all")
    for k, row in table.items(): print(f"{k:18s}" + "".join(f"{row[f'{lo}-{hi}']:7.2f}" for lo, hi in bins) + f"  {row['all']:.2f}")
    json.dump({"table": table, "info": info, "rows": {"train": len(tr.R), "test": len(te.R)}}, open(os.path.join(U.RUNS, f"landing_train{tag}.json"), "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    AMP = [0, 4, 7, 10, 15, 100]; names = ["<4°", "4-7°", "7-10°", "10-15°", ">15°"]; T4 = np.arange(0, 60, 4)
    fig, ax = plt.subplots(1, 5, figsize=(21, 4), sharey=True)
    for j in range(5):
        for k, v in res.items():
            ax[j].plot(T4 + 2, [v[(amp >= AMP[j]) & (amp < AMP[j + 1]) & (tau >= t) & (tau < t + 4)].mean() for t in T4], label=k, lw=2 if k in ("linear tables", "stays here") else 1.2, ls="--" if k in ("linear tables", "stays here") else "-")
        ax[j].set_title("amplitude " + names[j]); ax[j].set_xlabel("time since onset (ms)"); ax[j].grid(alpha=.3)
    ax[0].set_ylabel("mean landing error (deg)"); ax[0].legend(fontsize=7); fig.suptitle("Causal landing estimate, Andersson test: networks trained for it vs the engine's linear tables" + (" (+ d1, d2 training data)" if a.more else ""))
    fig.tight_layout(); fig.savefig(os.path.join(U.RUNS, f"landing_train{tag}.png"), dpi=120)
