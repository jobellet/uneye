#!/usr/bin/env python3
"""Label-free saccade readouts from a self-supervised ViT WITHOUT using the [CLS] attention.

The literature that made DINO's attention useful for segmentation actually moved past the [CLS] attention (it is the
noisiest of the maps; LOST, Simeoni et al. 2021; TokenCut, Wang et al. 2022; MaskDistill, Van Gansbeke et al. 2022;
CutLER, Wang et al. 2023): they use the PATCH-TO-PATCH similarity of the last block's keys, spectral partitioning of the
token graph, and pseudo-mask distillation.  This script ports those readouts to the eye-trace ViT of
foundation/dino1d.py and scores every candidate map by the AUC against the human saccade labels (threshold-free):

  lost      - LOST: the token with the smallest degree of the patch-similarity graph is a foreground seed; the map is
              the mean cosine similarity of every token to the seed.  Saccades are the rarest event, the analogue of the
              salient object.
  tokencut  - TokenCut: bipartition of the same graph by ONE eigenvector of the normalised Laplacian (NCut); the map is
              the second-smallest eigenvector (Fiedler value), the continuous relaxation of the partition.
  kmeans    - k-means (k = 2) on the L2-normalised patch tokens of the window (MaskDistill's per-image clustering).
  memory    - PatchCore-style: the distance of each token to its nearest neighbour among a MEMORY of tokens of
              "ordinary" signal, sampled on the UNLABELED train split of the tested dataset (no label read).  Saccades
              are rare -> far from the memory; fixations are the bulk -> close.  The analogous readout on the raw input
              channels (arcsinh velocity) is the speed baseline that must be beaten.
  baselines - the detrended-speed input channel itself, and the same readouts on an UNTRAINED ViT (controls).

Geometry identical to dino1d.attention_maps / dino_eval.sliding: sliding 480-sample windows, central 336-sample crop,
stride 84, maps averaged per sample.  Keys are the key vectors of the LAST block (pre-projection, as in LOST).
Run from the repository root:  python foundation/lost1d.py [--ckpt foundation/runs/dino1d.pt] [--trials 60]
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from foundation import data as FD
from foundation import dino1d as DN

P, BASE, G_LEN, L_LEN, DEV = DN.P, DN.BASE, DN.G_LEN, DN.L_LEN, DN.DEV
MEM_N, MEM_PER, SEED = 16384, 600, 0


@torch.no_grad()
def keys_tokens(model, pos, bs=64):
    """(n, T, 2) positions at 1 kHz -> per-sample keys (n, T, D_MODEL) (heads concatenated), per-sample tokens (n, T, D_MODEL)
    of the LAST block, averaged over sliding windows (geometry of dino1d.attention_maps).  keys: the key vectors
    actually used by the last block's attention, i.e. what LOST / TokenCut consume in 2-D."""
    model.eval()
    n, T, _ = pos.shape
    pad = (BASE - G_LEN) // 2
    padded = np.concatenate([np.full((n, pad, 2), np.nan, np.float32), pos.astype(np.float32),
                             np.full((n, pad + BASE, 2), np.nan, np.float32)], 1)
    DK = DN.D_MODEL // DN.HEADS
    ak = np.zeros((n, T + 2 * BASE, DN.D_MODEL), np.float32)         # keys of ALL heads concatenated, as LOST / TokenCut do (an audit found a mean over the heads: different projection subspaces must not be added)
    at = np.zeros((n, T + 2 * BASE, DN.D_MODEL), np.float32)
    cnt = np.zeros((n, T + 2 * BASE, 1), np.float32)
    accm = {k: np.zeros((n, T + 2 * BASE), np.float32) for k in ("lost", "tokencut", "kmeans")}
    cntm = np.zeros((n, T + 2 * BASE), np.float32)
    for s in range(0, T + pad, L_LEN):
        win = padded[:, s:s + BASE]
        if win.shape[1] < pad + G_LEN:
            continue                                                                    # truncated trailing window: no full crop
        for i in range(0, n, bs):
            f = torch.as_tensor(DN.feats(win[i:i + bs, pad:pad + G_LEN], 1000.0), device=DEV)     # features of the NaN-free crop only (the NaN padding polluted the noise scale)
            vit = model.vit
            t = vit.patch(f.transpose(1, 2)).transpose(1, 2)
            k_tok = t.shape[1]
            h = torch.cat([vit.cls.expand(len(t), -1, -1), t + vit.pos[:k_tok]], 1)
            K = None
            for j, b in enumerate(vit.blocks):
                if j == len(vit.blocks) - 1:
                    x = b.n1(h)
                    qkv = b.qkv(x)
                    K = qkv.reshape(len(x), k_tok + 1, 3, DN.HEADS, DK)[:, 1:, 1]       # (B, tokens, H, DK) patch keys, [CLS] dropped
                    h, _ = b(h)
                else:
                    h, _ = b(h)
            last_keys = F.normalize(K.reshape(len(K), k_tok, DN.HEADS * DK), dim=-1)          # (B, tokens, D_MODEL) heads concatenated, L2-normalised
            last_toks = vit.norm(h)[:, 1:]
            m_k = np.repeat(last_keys.cpu().numpy(), P, axis=1)                         # (B, samples, D_MODEL)
            m_t = np.repeat(last_toks.cpu().numpy(), P, axis=1)                         # (B, samples, D_MODEL)
            ak[i:i + bs, s + pad:s + pad + G_LEN] += m_k
            at[i:i + bs, s + pad:s + pad + G_LEN] += m_t
            cnt[i:i + bs, s + pad:s + pad + G_LEN] += 1
            gm = window_graph_maps(last_keys)                                            # per-window LOST / TokenCut maps
            km = window_kmeans_map(last_toks)                                            # per-window MaskDistill map
            for nm2, mp in (("lost", gm[0]), ("tokencut", gm[1]), ("kmeans", km)):
                accm[nm2][i:i + bs, s + pad:s + pad + G_LEN] += np.repeat(mp.cpu().numpy(), P, axis=1)
            cntm[i:i + bs, s + pad:s + pad + G_LEN] += 1
    sl = slice(pad, pad + T)
    ok = cnt[:, pad:pad + T, 0] > 0
    K_out = np.full((n, T, DN.D_MODEL), np.nan, np.float32); K_out[ok] = (ak[:, sl] / np.maximum(cnt[:, sl], 1))[ok]
    T_out = np.full((n, T, DN.D_MODEL), np.nan, np.float32); T_out[ok] = (at[:, sl] / np.maximum(cnt[:, sl], 1))[ok]
    maps = {}
    okm = cntm[:, sl] > 0
    for name in ("lost", "tokencut", "kmeans"):
        maps[name] = np.full((n, T), np.nan, np.float32)
        maps[name][okm] = (accm[name][:, sl] / np.maximum(cntm[:, sl], 1))[okm]
    return K_out, T_out, maps


def window_graph_maps(keys):
    """keys (B, tokens, DK), L2-normalised -> (lost, tokencut) per-token maps (B, tokens).
    lost: similarity of every token to the token of smallest degree (LOST's foreground seed).
    tokencut: the second-smallest eigenvector of the normalised Laplacian of the token graph (Fiedler vector of NCut)."""
    B, W, _ = keys.shape
    A = torch.bmm(keys, keys.transpose(1, 2))                                           # (B, W, W) cosine similarity
    A = (A + 1.0) / 2.0
    deg = A.sum(-1)
    seed = deg.argmin(dim=1)                                                            # LOST: most isolated token = foreground
    lost_map = A[torch.arange(B, device=keys.device), seed]
    d = deg.clamp_min(1e-6)
    An = A / torch.sqrt(d[:, :, None] * d[:, None, :])
    L = torch.eye(W, device=keys.device).expand(B, W, W) - An                          # normalised Laplacian I - D^-1/2 A D^-1/2 (an audit found D - A_n here, which sorts tokens by degree)
    eye = torch.eye(W, device=keys.device).expand(B, W, W)
    _, vecs = torch.linalg.eigh(L + eye * 1e-4)
    fied = vecs[:, :, 1]                                                                # second-smallest eigenvector
    big = fied.abs().argmax(dim=1)                                                      # TokenCut's own sign rule: the token with the largest |value| is in the positive part (the eigensolver's sign is arbitrary and, without this, overlapping windows cancel each other)
    fied = torch.where(fied.gather(1, big[:, None]) < 0, -fied, fied)
    fied = fied - fied.mean(1, keepdim=True)
    fied = fied / fied.std(1, keepdim=True).clamp_min(1e-6)
    return lost_map, fied


def window_kmeans_map(tokens, iters=10, seed=0):
    """tokens (B, tokens, D) -> (B, tokens): k-means with k = 2 on the L2-normalised tokens (MaskDistill's per-image
    clustering); map = distance to the midpoint of the two centroids, positive for the minority cluster (saccades)."""
    X = F.normalize(tokens, dim=-1)
    B, W, D = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    i0 = torch.randint(0, W, (B,), generator=g); i1 = (i0 + torch.randint(1, W, (B,), generator=g)) % W        # two DIFFERENT initial tokens (an audit found draws with replacement could give two identical centroids)
    idx = torch.stack([i0, i1], 1).to(X.device)
    c = X[torch.arange(B, device=X.device)[:, None], idx]                               # (B, 2, D)
    for _ in range(iters):
        dist = torch.cdist(X, c)                                                        # (B, W, 2)
        lab = dist.argmin(-1)
        newc = c.clone()
        for j in range(2):
            sel = (lab == j)[..., None].float()
            cnt_j = sel.sum(1)
            newc[:, j] = torch.where(cnt_j > 0, (X * sel).sum(1) / cnt_j.clamp_min(1.0), c[:, j])
        c = newc
    lab = torch.cdist(X, c).argmin(-1)
    sizes = torch.stack([(lab == j).sum(1) for j in range(2)], 1)
    minority = sizes.argmin(1)                                                          # candidate saccade cluster
    centroid = (c[torch.arange(B, device=X.device), minority] + c[torch.arange(B, device=X.device), 1 - minority]) / 2.0
    mp = (X - centroid.unsqueeze(1)).norm(dim=-1)
    return torch.where(lab == minority.unsqueeze(1), mp, -mp)


def build_memory(model, S_train, mem_n=MEM_N):
    """PatchCore-style memory: `mem_n` token vectors sampled from the UNLABELED train split of the tested dataset
    (labels of that split are never read).  Saccades are rare -> tokens far from this memory are candidates."""
    rng = np.random.RandomState(SEED)
    n = len(S_train)
    sel = rng.permutation(n)[: min(MEM_PER, n)]
    _, tokens, _ = keys_tokens(model, S_train.pos[sel])
    ok = np.isfinite(tokens).all(2)
    T_ok = tokens[ok]
    if len(T_ok) > mem_n:
        T_ok = T_ok[rng.permutation(len(T_ok))[:mem_n]]
    return F.normalize(torch.as_tensor(T_ok, device=DEV), dim=-1)


def memory_maps(tokens, memory, bs=512):
    """distance of every token to its nearest neighbour in the memory -> (n, T) float32."""
    n, T, _ = tokens.shape
    out = np.full((n, T), np.nan, np.float32)
    dev = torch.device(DEV)
    M = memory.to(dev)
    for s in range(0, n, bs):
        X = torch.as_tensor(tokens[s:s + bs], device=dev).reshape(-1, tokens.shape[2])
        ok = torch.isfinite(X).all(1)
        if ok.sum() == 0:
            continue
        Xo = F.normalize(X[ok], dim=-1)
        best = torch.full((len(Xo),), -torch.inf, device=dev)
        for m0 in range(0, len(M), 4096):
            sim = Xo @ M[m0:m0 + 4096].T                                                 # cosine similarity, chunked over the memory
            best = torch.maximum(best, sim.max(1).values)                              # nearest neighbour = LARGEST cosine similarity over all chunks (an audit found minimum here)
        nn = torch.full((len(X),), np.nan, device=dev)
        nn[ok] = 2.0 - 2.0 * best
        out[s:s + bs] = nn.reshape(-1, T).cpu().numpy()
    return out


def auc(score, y, ok):
    m = ok & np.isfinite(score)
    return float(roc_auc_score(y[m], score[m]))


def report(model, S, S_train, trials):
    rng = np.random.RandomState(1)
    sel = rng.permutation(len(S))[:trials]
    pos, lab = S.pos[sel], S.lab[sel]
    y = lab == 1
    ok = (lab >= 0) & np.isfinite(pos).all(2)
    keys, tokens, maps = keys_tokens(model, pos)
    ok &= np.isfinite(keys[..., 0]) & np.isfinite(tokens[..., 0])
    res = {}
    for name in ("lost", "tokencut", "kmeans"):
        res[name] = auc(maps[name], y, ok)
    mem = build_memory(model, S_train)
    res["memory"] = auc(memory_maps(tokens, mem), y, ok)
    f = DN.feats(pos, 1000.0)[..., 5]                                                   # detrended-speed channel, the baseline
    res["speed_channel"] = auc(f, y, ok)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "foundation", "runs", "dino1d.pt"))
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(ROOT, "foundation", "runs", "lost1d.json"))
    a = ap.parse_args()
    trained = None
    if os.path.exists(a.ckpt):
        trained = DN.DINO().to(DEV)
        trained.load_state_dict(torch.load(a.ckpt, weights_only=False)["state"])
    else:
        print("checkpoint not found:", a.ckpt, "-> untrained control only")
    torch.manual_seed(0)
    untrained = DN.DINO().to(DEV)
    rows = []
    for nm in FD.ALL:
        S = FD.load(nm, "test")
        n_tr = a.trials if nm != "andersson" else 20
        S_train = FD.load(nm, "train")
        r_u = report(untrained, S, S_train, n_tr)
        r_t = report(trained, S, S_train, n_tr) if trained is not None else None
        rows.append(dict(dataset=nm, trained=r_t, untrained=r_u))
        if r_t is not None:
            print(f"[{nm:9s}] AUC (trained / untrained) | " + " | ".join(f"{k}: {r_t[k]:.3f} / {r_u[k]:.3f}" for k in
                  ("lost", "tokencut", "kmeans", "memory", "speed_channel")), flush=True)
        else:
            print(f"[{nm:9s}] AUC (untrained) | " + " | ".join(f"{k}: {r_u[k]:.3f}" for k in
                  ("lost", "tokencut", "kmeans", "memory", "speed_channel")), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
