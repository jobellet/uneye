#!/usr/bin/env python3
"""TS2Vec (Yue et al., AAAI 2022) for eye traces: universal time-series representation by hierarchical contrastive learning over augmented context views.

As in the paper: an input projection, RANDOM TIMESTAMP MASKING (binary mask on the projected input), a stack of residual blocks of dilated 1-D convolutions
(a symmetric, bidirectional network), two overlapping random crops of the same series as the two views, and the hierarchical loss = instance-wise
contrastive (negatives: the same timestamp of the other series) + temporal contrastive (negatives: the other timestamps of the same series), repeated
after max-pooling along time until one timestamp is left. One representation per timestamp (no patches), no temperature, raw dot products, as in
the reference implementation. Input: dino1d.feats (8 velocity channels). Pool: dino1d.training_pool via ssl_vit.pool_cached. Validation / early stopping /
final evaluation exactly as foundation/ssl_vit.py (probe + HMM on the labeled TRAIN splits; test subsets only at the end).
Run:  python foundation/ts2vec.py --name ts2vec_a [--depth 7] [--dim 128] [--lr 1e-3] [--steps 5000]
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from foundation import dino1d as DN, ssl_probe as SP
from foundation.train_lodo import augment, DEV
from foundation.dino1d import feats
from foundation.ssl_vit import pool_cached


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dil, final=False):
        super().__init__(); self.c1 = nn.Conv1d(cin, cout, 3, padding=dil, dilation=dil); self.c2 = nn.Conv1d(cout, cout, 3, padding=dil, dilation=dil)
        self.proj = nn.Conv1d(cin, cout, 1) if (cin != cout or final) else None
    def forward(self, x):
        r = x if self.proj is None else self.proj(x); return self.c2(F.gelu(self.c1(F.gelu(x)))) + r


class TSEncoder(nn.Module):
    def __init__(self, nin=8, hid=64, dim=128, depth=7, mask_p=0.5):
        super().__init__(); self.fc = nn.Linear(nin, hid); self.mask_p = mask_p
        self.blocks = nn.Sequential(*[ConvBlock(hid if i else hid, dim if i == depth - 1 else hid, 2 ** i, final=(i == depth - 1)) for i in range(depth)])
    def forward(self, x, mask=None):                           # x (B, T, nin) -> (B, T, dim)
        h = self.fc(x)
        if (mask is None and self.training) or mask == "binomial":
            m = torch.rand(h.shape[0], h.shape[1], device=h.device) < self.mask_p; h = h.masked_fill(~m[..., None], 0.0)   # random timestamp masking
        return self.blocks(h.transpose(1, 2)).transpose(1, 2)


def instance_contrastive_loss(z1, z2):
    B = z1.size(0)
    if B == 1: return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], 0).transpose(0, 1)                # (T, 2B, C)
    sim = z @ z.transpose(1, 2); logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1); i = torch.arange(B, device=z1.device)
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2


def temporal_contrastive_loss(z1, z2):
    T = z1.size(1)
    if T == 1: return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], 1)                                # (B, 2T, C)
    sim = z @ z.transpose(1, 2); logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1); t = torch.arange(T, device=z1.device)
    return (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2


def hierarchical_contrastive_loss(z1, z2, alpha=0.5):
    loss, d = z1.new_tensor(0.0), 0
    while z1.size(1) > 1:
        if alpha != 0: loss = loss + alpha * instance_contrastive_loss(z1, z2)
        if d >= 0 and (1 - alpha) != 0: loss = loss + (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1; z1 = F.max_pool1d(z1.transpose(1, 2), 2).transpose(1, 2); z2 = F.max_pool1d(z2.transpose(1, 2), 2).transpose(1, 2)
    if z1.size(1) == 1 and alpha != 0: loss = loss + alpha * instance_contrastive_loss(z1, z2); d += 1
    return loss / max(d, 1)


def novelty_score(feature_fn, w=8):
    """label-free boundary / novelty score (Foote 2000; Kreuk et al. 2020 for phoneme boundaries): distance between the mean embedding of the w samples after
    and the w samples before each timestamp; high on and around a saccade. No probe, no label: it is decoded by the HMM directly."""
    def fn(pos):
        z = feature_fn(pos); ok = np.isfinite(z[..., 0]); z = z / (np.linalg.norm(np.nan_to_num(z), axis=-1, keepdims=True) + 1e-9); n, T, C = z.shape
        cs = np.concatenate([np.zeros((n, 1, C)), np.cumsum(np.nan_to_num(z), 1)], 1); idx = np.arange(T); lo, hi = np.clip(idx - w, 0, T), np.clip(idx + w, 0, T)
        left = (cs[:, idx] - cs[:, lo]) / np.maximum(idx - lo, 1)[None, :, None]; right = (cs[:, hi] - cs[:, idx]) / np.maximum(hi - idx, 1)[None, :, None]
        d = np.linalg.norm(right - left, axis=-1); s_ = np.log10(1 + d / (np.median(d[ok]) + 1e-9)); s_[~ok] = np.nan; return s_
    return fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="ts2vec"); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--depth", type=int, default=7); ap.add_argument("--dim", type=int, default=128); ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--patience", type=int, default=3); ap.add_argument("--max-minutes", type=float, default=1e9); a = ap.parse_args()
    rng = np.random.RandomState(0); torch.manual_seed(0); pool = pool_cached(); T = DN.BASE
    net = TSEncoder(hid=a.hid, dim=a.dim, depth=a.depth).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    base_hmm = SP.baseline_hmm(); print(f"[{a.name}] TS2Vec {sum(p.numel() for p in net.parameters())/1e6:.2f} M parameters | input channels: probe + HMM event F1 {base_hmm[1]:.3f} kappa {base_hmm[2]:.3f} (validation)", flush=True)
    token_fn = lambda f: net(f, mask="none")
    def validate():
        net.eval(); memo = {}
        def fn(pos):
            if id(pos) not in memo: memo[id(pos)] = SP.sliding_tokens(token_fn, pos, 1)
            return memo[id(pos)]
        res, auc = SP.loo_probe(fn); resh, f1m, km = SP.loo_probe_hmm(fn); return res, auc, resh, float(f1m), float(km)
    res0, auc0, resh0, f10, k0 = validate(); print(f"[{a.name}] UNTRAINED: probe AUC {auc0:.3f}; probe + HMM event F1 {f10:.3f} kappa {k0:.3f}", flush=True)
    best, bad, hist, log, t0, it = f10, 0, [], [], time.time(), -1
    torch.save({"state": net.state_dict(), "args": vars(a), "step": 0, "event_f1": f10}, os.path.join(ROOT, "foundation", "runs", f"ssl_{a.name}.pt"))
    for it in range(a.steps):
        lr = a.lr * min(1.0, (it + 1) / 200) * (0.5 * (1 + math.cos(math.pi * it / a.steps)) * 0.98 + 0.02)
        for g in opt.param_groups: g["lr"] = lr
        x = torch.as_tensor(feats(augment(pool[rng.randint(0, len(pool), a.batch)], rng, True), 1000.0), device=DEV)      # (B, 480, 8)
        # two overlapping random crops (TS2Vec): [a1, b1) and [a2, b2) with a1 < a2 <= b1 < b2; the overlap has the same timestamps in both views
        crop = rng.randint(2 ** (a.depth - 1) // 2 + 1 if False else 64, T + 1); a_ = rng.randint(0, T - crop + 1); b_ = a_ + crop
        ov = rng.randint(32, crop - 31) if crop > 64 else crop // 2; c1 = rng.randint(a_, b_ - ov + 1); c2 = c1 + ov
        s1 = rng.randint(max(a_, 0), c1 + 1); e2 = rng.randint(c2, b_ + 1)
        net.train(); o1 = net(x[:, s1:c2]); o2 = net(x[:, c1:e2])
        loss = hierarchical_contrastive_loss(o1[:, c1 - s1:], o2[:, :c2 - c1])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step(); hist.append(float(loss.detach()))
        if (it + 1) % a.eval_every == 0:
            res, m, resh, f1m, km = validate(); log.append(dict(step=it + 1, loss=float(np.mean(hist[-a.eval_every:])), probe_auc=float(m), event_f1=f1m, kappa=km))
            improved = f1m > best + 1e-3
            if improved: best, bad = f1m, 0; torch.save({"state": net.state_dict(), "args": vars(a), "step": it + 1, "event_f1": f1m, "kappa": km}, os.path.join(ROOT, "foundation", "runs", f"ssl_{a.name}.pt"))
            else: bad += 1
            print(f"[{a.name}] step {it + 1}/{a.steps} loss {log[-1]['loss']:.4f} | probe AUC {m:.3f} | probe + HMM: event F1 {f1m:.3f} kappa {km:.3f} (best F1 {best:.3f}{' *saved*' if improved else ''}, patience {bad}/{a.patience}) | "
                  + " ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in resh.items()) + f" | {time.time() - t0:.0f} s", flush=True)
            if bad >= a.patience: print(f"[{a.name}] early stop", flush=True); break
            if (time.time() - t0) / 60 > a.max_minutes: print(f"[{a.name}] time cap", flush=True); break
    from foundation import night_eval as NE
    ck = torch.load(os.path.join(ROOT, "foundation", "runs", f"ssl_{a.name}.pt"), weights_only=False); net.load_state_dict(ck["state"]); net.eval()
    res = NE.eval_features(lambda pos: SP.sliding_tokens(token_fn, pos, 1))
    NE.save(a.name, "self-supervised representation + linear probe + HMM", "labels of the other 4 datasets (linear probe only); the encoder itself saw no label", "Yue et al. 2022 (TS2Vec)",
            vars(a), res, (time.time() - t0) / 60, dict(best_step=ck.get("step"), best_val_event_f1=best, untrained_val_event_f1=f10, input_channels_val_event_f1=base_hmm[1], steps_run=it + 1))
    res_n = NE.eval_direct(novelty_score(lambda pos: SP.sliding_tokens(token_fn, pos, 1)))
    NE.save(a.name + "_novelty", "label-free: embedding novelty curve + HMM (no probe)", "none (the encoder saw no label; no probe either)", "Foote 2000; Kreuk et al. 2020; Yue et al. 2022", dict(vars(a), window=8, note="only the HMM column is meaningful"), res_n, (time.time() - t0) / 60)
    json.dump(dict(args=vars(a), log=log), open(os.path.join(ROOT, "foundation", "runs", f"ssl_{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
