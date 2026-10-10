#!/usr/bin/env python3
"""Does anything in the DINO-for-eye-traces model (foundation/dino1d.py) highlight the saccades without a label?
Per benchmark (test split) and per candidate map, the AUC against the human saccade labels (threshold-free; 0.5 = nothing, <0.5 = anti-correlated):
  CLS attention, mean of the 6 heads and best / worst head (teacher, last layer)
  PC1 of the patch tokens (last layer, direction fitted on the unlabeled trials of the dataset; sign ambiguous -> max(AUC, 1 - AUC))
  speed baseline: the detrended-speed input channel itself (what a model must beat to show anything beyond the trivial feature)
  the same CLS attention and PC1 for the UNTRAINED network (does training change anything?)
Run from the repository root:  python foundation/dino_eval.py [--ckpt foundation/runs/dino1d.pt]
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from foundation import data as FD, dino1d as DN

P, BASE, G_LEN, L_LEN, DEV = DN.P, DN.BASE, DN.G_LEN, DN.L_LEN, DN.DEV


@torch.no_grad()
def sliding(model, pos, what, bs=64):
    """(n, T, 2) -> (n, T, C) map averaged over sliding windows (same geometry as dino1d.attention_maps); what = 'attn' (C = heads) or 'tokens' (C = dim)"""
    model.eval(); n, T, _ = pos.shape; pad = (BASE - G_LEN) // 2
    padded = np.concatenate([np.full((n, pad, 2), np.nan, np.float32), pos.astype(np.float32), np.full((n, pad + BASE, 2), np.nan, np.float32)], 1)
    C = DN.HEADS if what == "attn" else DN.D_MODEL
    acc = np.zeros((n, T + 2 * BASE, C), np.float32); cnt = np.zeros((n, T + 2 * BASE, 1), np.float32)
    for s in range(0, T + pad, L_LEN):
        win = padded[:, s:s + BASE]
        for i in range(0, n, bs):
            f = torch.as_tensor(DN.feats(win[i:i + bs], 1000.0)[:, pad:pad + G_LEN], device=DEV)
            vit = model.vit
            t = vit.patch(f.transpose(1, 2)).transpose(1, 2); k = t.shape[1]
            h = torch.cat([vit.cls.expand(len(t), -1, -1), t + vit.pos[:k]], 1); a = None
            for j, b in enumerate(vit.blocks): h, a = b(h, want_attn=(j == len(vit.blocks) - 1))
            if what == "attn":
                m = a[:, :, 0, 1:]; m = (m / m.sum(-1, keepdim=True) * m.shape[-1]).cpu().numpy()                     # (B, H, tokens)
            else:
                m = vit.norm(h)[:, 1:].transpose(1, 2).cpu().numpy()                                                    # (B, dim, tokens)
            m = np.repeat(m, P, axis=2).transpose(0, 2, 1)
            acc[i:i + bs, s + pad:s + pad + G_LEN] += m; cnt[i:i + bs, s + pad:s + pad + G_LEN] += 1
    out = acc[:, pad:pad + T] / np.maximum(cnt[:, pad:pad + T], 1); out[cnt[:, pad:pad + T, 0] == 0] = np.nan
    return out


def auc(score, y, mask):
    return roc_auc_score(y[mask], score[mask])


def report(model, S, n_trials, label):
    sel = np.random.RandomState(1).permutation(len(S))[:n_trials]; pos, lab = S.pos[sel], S.lab[sel]
    y = lab == 1; ok = (lab >= 0) & np.isfinite(pos).all(2)
    A = sliding(model, pos, "attn"); ok &= np.isfinite(A[..., 0])
    ah = [auc(A[..., h], y, ok) for h in range(DN.HEADS)]
    T = sliding(model, pos, "tokens"); Tv = T[ok]
    pc = PCA(1).fit(Tv[np.random.RandomState(0).choice(len(Tv), min(40000, len(Tv)), replace=False)]); pc1 = np.zeros(ok.shape, np.float32); pc1[ok] = pc.transform(Tv)[:, 0]
    a_pc = auc(pc1, y, ok)
    return dict(attn_mean=auc(A.mean(2), y, ok), attn_best_head=max(ah), attn_worst_head=min(ah), pc1_tokens=max(a_pc, 1 - a_pc), pc1_sign=("+" if a_pc >= 0.5 else "-"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=os.path.join(ROOT, "foundation", "runs", "dino1d.pt")); ap.add_argument("--trials", type=int, default=60); a = ap.parse_args()
    trained = DN.DINO().to(DEV); trained.load_state_dict(torch.load(a.ckpt, weights_only=False)["state"])
    torch.manual_seed(0); untrained = DN.DINO().to(DEV); rows = []
    for nm in FD.ALL:
        S = FD.load(nm, "test"); r_t = report(trained, S, a.trials if nm != "andersson" else 20, "trained"); r_u = report(untrained, S, a.trials if nm != "andersson" else 20, "untrained")
        sel = np.random.RandomState(1).permutation(len(S))[:a.trials if nm != "andersson" else 20]
        f = DN.feats(S.pos[sel], 1000.0)[..., 5]; ok = (S.lab[sel] >= 0) & np.isfinite(S.pos[sel]).all(2); sp = auc(f, S.lab[sel] == 1, ok)
        row = dict(dataset=nm, speed_channel_AUC=sp, trained=r_t, untrained=r_u); rows.append(row)
        print(f"[{nm:9s}] AUC | speed channel {sp:.3f} | trained: CLS attention mean {r_t['attn_mean']:.3f}, best head {r_t['attn_best_head']:.3f}, worst head {r_t['attn_worst_head']:.3f}, "
              f"PC1 of patch tokens {r_t['pc1_tokens']:.3f} ({r_t['pc1_sign']}) | untrained: attention mean {r_u['attn_mean']:.3f}, PC1 {r_u['pc1_tokens']:.3f}", flush=True)
    out = os.path.join(ROOT, "foundation", "runs", "dino_eval.json"); json.dump(rows, open(out, "w"), indent=1, default=float); print("wrote", out)


if __name__ == "__main__":
    main()
