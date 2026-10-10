#!/usr/bin/env python3
"""I-JEPA / V-JEPA for 1-D eye traces (Assran et al. 2023; Bardes et al. 2024), bidirectional, with validation, checkpoints and early stopping.

Objective (as in the papers, nothing invented): a context encoder sees only the CONTEXT tokens of a window; a narrow predictor, given the positions of
the masked blocks (learnable mask tokens + positional embeddings), predicts the representations of the TARGET blocks produced by an EMA target encoder that
sees the whole window (layer-normalised outputs, stop-gradient); loss = smooth L1 in representation space. 4 target blocks per window (one in each quarter),
context = every other token (before AND after the targets: the position jump around a saccade is available). No decoder, no pixel / sample reconstruction,
no contrastive pairs. Collapse is prevented by the EMA target + the predictor (not by negatives).
Input: dino1d.feats (8 channels: velocity-only, unit-free, vx / vy with rotation augmentation). Pool: archive/ (4 sources) + the UNLABELED train splits of
datasets 1-4 and Andersson (dino1d.training_pool).
Every `--eval-every` steps: leave-one-dataset-out linear probe (foundation/ssl_probe.py) on the labeled TRAIN splits (labels used only to choose checkpoints /
hyper-parameters; the test splits are never touched); the best checkpoint is saved; training stops after `--patience` evaluations without improvement.
Run from the repository root:  python foundation/ssl_vit.py --name jepa_a [--lr 3e-4] [--block 12] [--patch 4] [--ema 0.996] [--steps 6000] [--patience 3]
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from foundation import dino1d as DN, ssl_probe as SP
from foundation.train_lodo import augment, DEV
from foundation.dino1d import Block, sincos, feats


class TokViT(nn.Module):
    def __init__(self, patch=4, d=192, depth=6, heads=6, nin=8):
        super().__init__(); self.patch_n, self.d = patch, d
        self.patch = nn.Conv1d(nin, d, patch, stride=patch); self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)]); self.norm = nn.LayerNorm(d)
        self.register_buffer("pos", sincos(512, d), persistent=False)
    def tokens(self, x):                                      # (B, T, nin) -> (B, T // patch, d), positions added
        t = self.patch(x.transpose(1, 2)).transpose(1, 2); return t + self.pos[:t.shape[1]]
    def encode(self, t):
        for b in self.blocks: t, _ = b(t)
        return self.norm(t)


class Predictor(nn.Module):
    def __init__(self, d=192, dp=96, depth=4, heads=4, out_dim=None):
        super().__init__(); self.inp = nn.Linear(d, dp); self.mask = nn.Parameter(torch.zeros(1, 1, dp)); nn.init.trunc_normal_(self.mask, std=0.02)
        self.blocks = nn.ModuleList([Block(dp, heads) for _ in range(depth)]); self.norm = nn.LayerNorm(dp); self.out = nn.Linear(dp, out_dim or d)
        self.register_buffer("pos", sincos(512, dp), persistent=False)
    def forward(self, ctx, ci, ti):
        h = torch.cat([self.inp(ctx) + self.pos[ci], self.mask.expand(len(ctx), ti.shape[1], -1) + self.pos[ti]], 1)
        for b in self.blocks: h, _ = b(h)
        return self.out(self.norm(h[:, ctx.shape[1]:]))


def gather(t, idx): return torch.gather(t, 1, idx[..., None].expand(-1, -1, t.shape[-1]))


def make_masks(rng, B, N, Lb, nseg=4):
    seg = N // nseg; tgt = np.zeros((B, nseg * Lb), np.int64); ctx = np.zeros((B, N - nseg * Lb), np.int64)
    for i in range(B):
        m = np.zeros(N, bool); tl = []
        for s in range(nseg):
            st = s * seg + rng.randint(0, seg - Lb + 1); tl.extend(range(st, st + Lb)); m[st:st + Lb] = True
        tgt[i] = tl; ctx[i] = np.nonzero(~m)[0]
    return torch.as_tensor(ctx, device=DEV), torch.as_tensor(tgt, device=DEV)


def hubert_targets(pool, patch, k=64, n_tok=200000):
    """HuBERT iteration 1: k-means (k = 64) of the input tokens (patch x 8 channels, standardised), the 'MFCC k-means' teacher of the paper"""
    from sklearn.cluster import MiniBatchKMeans
    path = os.path.join(ROOT, "foundation", "runs", f"hubert_km_p{patch}.npz")
    if os.path.exists(path): z = np.load(path); return z["c"], z["mu"], z["sd"]
    rng = np.random.RandomState(0); idx = rng.randint(0, len(pool), 3000)
    toks = feats(augment(pool[idx], rng, True), 1000.0)[:, :DN.BASE // patch * patch].reshape(-1, patch * DN.N_IN)[:n_tok]
    mu, sd = toks.mean(0), toks.std(0) + 1e-6
    c = MiniBatchKMeans(k, batch_size=4096, n_init=3, random_state=0).fit((toks - mu) / sd).cluster_centers_
    np.savez(path, c=c, mu=mu, sd=sd); return c, mu, sd


def kmeans_labels(tokens, centers, mu, sd):
    z = (tokens.reshape(-1, tokens.shape[-1]) - mu) / sd
    return ((z ** 2).sum(1)[:, None] - 2 * z @ centers.T + (centers ** 2).sum(1)[None]).argmin(1).reshape(tokens.shape[:2])


def pool_cached():
    path = os.path.join(ROOT, "foundation", "runs", "pool.npz")
    if os.path.exists(path): z = np.load(path); return z["pos"]
    pos, grp = DN.training_pool(); np.savez(path, pos=pos, grp=grp); return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="jepa"); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--block", type=int, default=12, help="target block length in tokens")
    ap.add_argument("--patch", type=int, default=4); ap.add_argument("--ema", type=float, default=0.996); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--objective", default="jepa", choices=["jepa", "mae", "hubert"]); ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--depth", type=int, default=6); ap.add_argument("--wd", type=float, default=0.04); a = ap.parse_args()
    rng = np.random.RandomState(0); torch.manual_seed(0)
    pool = pool_cached(); N = DN.BASE // a.patch
    stu = TokViT(a.patch, depth=a.depth).to(DEV); tea = TokViT(a.patch, depth=a.depth).to(DEV); tea.load_state_dict(stu.state_dict()); [p.requires_grad_(False) for p in tea.parameters()]
    obj = a.objective; nin = DN.N_IN
    extra_params = []
    if obj == "jepa": pred = Predictor().to(DEV)
    elif obj == "mae": pred = Predictor(depth=2, out_dim=a.patch * nin).to(DEV)               # light decoder, reconstructs the masked patches (He et al. 2022)
    else:
        pred = nn.Linear(stu.d, 64).to(DEV); mask_emb = nn.Parameter(torch.zeros(stu.d, device=DEV)); extra_params = [mask_emb]; centers, km_mu, km_sd = hubert_targets(pool, a.patch)
    enc_eval = tea if obj == "jepa" else stu                                                      # JEPA: EMA target encoder (as in the paper); MAE / HuBERT: the encoder itself
    opt = torch.optim.AdamW(list(stu.parameters()) + list(pred.parameters()) + extra_params, lr=a.lr, weight_decay=a.wd)
    base_auc = SP.baseline_input_features()[1]; base_hmm = SP.baseline_hmm()
    print(f"[{a.name}] {len(pool)} windows, {N} tokens of {a.patch} samples, target blocks of {a.block} tokens x4 | input channels, probe AUC {base_auc:.3f}; probe + HMM: event F1 {base_hmm[1]:.3f} kappa {base_hmm[2]:.3f} (validation, set A) | device {DEV}", flush=True)
    best, bad, hist, log, t0 = -1.0, 0, [], [], time.time()
    token_fn = lambda f: enc_eval.encode(enc_eval.tokens(f))
    def validate():
        """features of the val trials computed once, then (1) leave-one-dataset-out probe AUC, (2) probe + HMM: event F1 / kappa (the early-stopping criterion)"""
        enc_eval.eval(); memo = {}
        def fn(pos):
            if id(pos) not in memo: memo[id(pos)] = SP.sliding_tokens(token_fn, pos, a.patch)
            return memo[id(pos)]
        res, auc = SP.loo_probe(fn); resh, f1m, km = SP.loo_probe_hmm(fn); return res, auc, resh, f1m, km
    res0, auc0, resh0, f10, k0 = validate()                                                                       # CONTROL: the same network before any training
    print(f"[{a.name}] UNTRAINED encoder (random init): probe AUC {auc0:.3f}; probe + HMM event F1 {f10:.3f} kappa {k0:.3f} ({' '.join(f'{k} {v[0]:.2f}/{v[1]:.2f}' for k, v in resh0.items())}) -> what training must improve on", flush=True)
    best = f10; torch.save({"state": enc_eval.state_dict(), "args": vars(a), "step": 0, "probe_auc": auc0, "event_f1": f10}, os.path.join(ROOT, "foundation", "runs", f"ssl_{a.name}.pt"))
    it = -1
    for it in range(a.steps):
        lr = a.lr * min(1.0, (it + 1) / 300) * (0.5 * (1 + math.cos(math.pi * it / a.steps)) * 0.98 + 0.02)
        for g in opt.param_groups: g["lr"] = lr
        mom = 1 - (1 - a.ema) * (math.cos(math.pi * it / a.steps) + 1) / 2
        x = torch.as_tensor(feats(augment(pool[rng.randint(0, len(pool), a.batch)], rng, True), 1000.0), device=DEV)
        stu.train(); pred.train(); tt = torch.zeros(1, 1, 1, device=DEV)
        if obj in ("jepa", "mae"):
            ci, ti = make_masks(rng, a.batch, N, a.block)
            z = stu.encode(gather(stu.tokens(x), ci)); p = pred(z, ci, ti)
            if obj == "jepa":
                with torch.no_grad(): tt = gather(F.layer_norm(tea.encode(tea.tokens(x)), (stu.d,)), ti)
                loss = F.smooth_l1_loss(p, tt)                                                       # I-JEPA / V-JEPA: smooth L1 in representation space
            else:
                patches = x.reshape(a.batch, N, a.patch * nin); loss = F.mse_loss(p, gather(patches, ti))   # MAE: MSE on the masked patches
        else:                                                                                        # HuBERT: predict the k-means class of the masked tokens
            starts = rng.rand(a.batch, N) < 0.10; m = np.zeros((a.batch, N), bool)
            for off in range(6): m[:, off:] |= starts[:, :N - off]
            m[:, 0] |= ~m.any(1)
            lab = torch.as_tensor(kmeans_labels(x.detach().cpu().numpy().reshape(a.batch, N, a.patch * nin), centers, km_mu, km_sd), device=DEV); mt = torch.as_tensor(m, device=DEV)
            tok0 = stu.patch(x.transpose(1, 2)).transpose(1, 2); tok0 = torch.where(mt[..., None], mask_emb.expand_as(tok0), tok0) + stu.pos[:N]
            loss = F.cross_entropy(pred(stu.encode(tok0))[mt], lab[mt])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(stu.parameters()) + list(pred.parameters()) + extra_params, 3.0); opt.step()
        if obj == "jepa":
            with torch.no_grad():
                for ps, pt in zip(stu.parameters(), tea.parameters()): pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
        hist.append(float(loss.detach()))
        if (it + 1) % a.eval_every == 0:
            res, m, resh, f1m, km = validate()
            sd = float(tt.std(1).mean()) if obj == "jepa" else 0.0                         # spread of the target features: ~0 = collapse (JEPA only)
            log.append(dict(step=it + 1, loss=float(np.mean(hist[-a.eval_every:])), probe_auc=m, event_f1=f1m, kappa=km, per_dataset={k: list(v) for k, v in resh.items()}, target_std=sd))
            m, f1m, km, sd, best = float(m), float(f1m), float(km), float(sd), float(best)
            improved = f1m > best + 1e-3
            if improved:
                best, bad = f1m, 0; torch.save({"state": enc_eval.state_dict(), "args": vars(a), "step": it + 1, "probe_auc": m, "event_f1": f1m, "kappa": km}, os.path.join(ROOT, "foundation", "runs", f"ssl_{a.name}.pt"))
            else: bad += 1
            print(f"[{a.name}] step {it + 1}/{a.steps} loss {log[-1]['loss']:.4f} target std {sd:.3f} | probe AUC {m:.3f} | probe + HMM: event F1 {f1m:.3f} kappa {km:.3f} (best F1 {best:.3f}{' *saved*' if improved else ''}, "
                  f"patience {bad}/{a.patience}) | " + " ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in resh.items()) + f" | {time.time() - t0:.0f} s", flush=True)
            if bad >= a.patience: print(f"[{a.name}] early stop", flush=True); break
            if (time.time() - t0) / 60 > a.max_minutes: print(f"[{a.name}] time cap", flush=True); break
    # ---- final evaluation of the best checkpoint on the common test subsets (foundation/night_eval.py)
    from foundation import night_eval as NE
    ck = torch.load(os.path.join(ROOT, "foundation", "runs", f"ssl_{a.name}.pt"), weights_only=False); enc_eval.load_state_dict(ck["state"]); enc_eval.eval(); t1 = time.time()
    res = NE.eval_features(lambda pos: SP.sliding_tokens(token_fn, pos, a.patch))
    NE.save(a.name, "self-supervised representation + linear probe + HMM", "labels of the other 4 datasets (linear probe only); the encoder itself saw no label",
            {"jepa": "Assran et al. 2023 (I-JEPA), Bardes et al. 2024 (V-JEPA)", "mae": "He et al. 2022 (MAE), Nie et al. 2023 (PatchTST)", "hubert": "Hsu et al. 2021 (HuBERT)"}[obj],
            {k: v for k, v in vars(a).items() if k not in ("name",)}, res, (time.time() - t0) / 60,
            dict(best_step=ck.get("step"), best_val_event_f1=best, untrained_val_event_f1=f10, input_channels_val_event_f1=base_hmm[1], steps_run=it + 1 if a.steps else 0))
    json.dump(dict(args=vars(a), baseline_probe_auc=base_auc, untrained_probe_auc=auc0, untrained_event_f1=f10, untrained_kappa=k0, baseline_event_f1=base_hmm[1], baseline_kappa=base_hmm[2], best_event_f1=best, log=log), open(os.path.join(ROOT, "foundation", "runs", f"ssl_{a.name}.json"), "w"), indent=1)
    print(f"[{a.name}] RESULT best event F1 {best:.3f} (kappa at that point: see log) | untrained encoder {f10:.3f} | input channels {base_hmm[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
