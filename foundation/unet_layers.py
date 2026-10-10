#!/usr/bin/env python3
"""Which layers of the pre-trained U-Net (foundation/unet_ssl.py, 2 channels, masked-velocity pre-training) change when it is fine-tuned to detect saccades, and does freezing the layers that
hardly change help?
 stats   fine-tune ALL layers from the pre-trained weights on many different training sets (datasets d1..d4, andersson x N = 20, 50 labeled trials x 3 draws) and record per layer the relative
         weight change ||W_ft - W_pre|| / ||W_pre|| (conv / transposed-conv weights) and the mean absolute change of the batch-norm scale / shift -> docs/figs_layers/layer_change.png, night/unet_layers_stats.json
 freeze  dataset 1, N = 10 / 20 / 50, 10 draws (same trials and test set as unet_ssl.py): fine-tune (a) all layers, (b) only the layers that changed MOST (chosen on d2, d3, d4, andersson ONLY, never d1),
         (c) only the layers that changed LEAST (control), (d) the head only (linear probe); the new output head is always trained. Frozen blocks stay in eval mode (BN statistics frozen).
python foundation/unet_layers.py stats | freeze
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn.functional as F
from foundation import data as FD, compare as CP
from foundation.unet_ssl import WUNet, pad25, make_input, score, RUNS, W
from foundation.train_lodo import augment, DEV
BLOCKS = ["c0", "c1", "c2", "c3", "up1", "c4", "up2", "c5", "c6"]; NIGHT = os.path.join(RUNS, "night"); FIG = os.path.join(ROOT, "docs", "figs_layers"); os.makedirs(FIG, exist_ok=True)


def pre_state(): return {k: v for k, v in torch.load(os.path.join(RUNS, "unet_pre.pt"), weights_only=False).items() if not k.startswith("head")}


def ft(sel_pos, sel_lab, seed, trainable=None, steps=600, lr=3e-4, bs=32, prep=None, warm=0):
    """fine-tune from the pre-trained weights; trainable = set of block names (None = all) + the new head. Early stopping on its own validation split (as unet_ssl.finetune_one)."""
    rng = np.random.RandomState(seed); torch.manual_seed(seed); net = WUNet(nout=1).to(DEV); net.load_state_dict(pre_state(), strict=False)
    frozen = [] if trainable is None else [getattr(net, b) for b in BLOCKS if b not in trainable]
    for m in frozen:
        for p in m.parameters(): p.requires_grad = False
    if prep is not None: frozen = prep(net)                      # parameter-efficient variants (foundation/unet_peft.py): sets requires_grad itself, returns the modules kept in eval mode
    body = [p for n_, p in net.named_parameters() if not n_.startswith("head") and p.requires_grad]
    body_ids = {id(p) for p in body}
    if warm: [p.requires_grad_(False) for p in body]              # LP-FT: first the new head alone
    n = len(sel_pos); nv = max(n // 5, 2); perm = rng.permutation(n); vi, ti = perm[:nv], perm[nv:]
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad or id(p) in body_ids], lr=lr, weight_decay=1e-2); best, bad, state = 1e9, 0, None
    vp, vl = sel_pos[vi], sel_lab[vi] == 1; vX = torch.as_tensor(make_input(vp), device=DEV); vY = torch.as_tensor(vl.astype(np.float32), device=DEV); vV = torch.as_tensor(np.isfinite(vp).all(2), device=DEV).float()
    for it in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 30) * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.9 + 0.1)
        P, Lb = [], []
        for _ in range(bs):
            i = ti[rng.randint(len(ti))]; s0 = rng.randint(0, sel_pos.shape[1] - W + 1); P.append(sel_pos[i, s0:s0 + W]); Lb.append(sel_lab[i, s0:s0 + W] == 1)
        a = augment(np.stack(P), rng, True); x = torch.as_tensor(make_input(a), device=DEV); y = torch.as_tensor(np.stack(Lb).astype(np.float32), device=DEV); w = torch.as_tensor(np.isfinite(a).all(2), device=DEV).float()
        if warm and it == warm: [p.requires_grad_(True) for p in body]
        net.train()
        for m in (frozen if not (warm and it < warm) else [getattr(net, b) for b in BLOCKS]): m.eval()
        loss = (F.binary_cross_entropy_with_logits(net(x)[:, 0], y, pos_weight=torch.tensor(1.5, device=DEV), reduction="none") * w).sum() / w.sum(); opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 25 == 0:
            net.eval()
            with torch.no_grad(): xx, T0 = pad25(vX); lo = net(xx)[:, 0, :T0]; vloss = float((F.binary_cross_entropy_with_logits(lo, vY, reduction="none") * vV).sum() / vV.sum())
            if vloss < best - 1e-4: best, bad = vloss, 0; state = {k: v.clone() for k, v in net.state_dict().items()}
            else: bad += 1
            if bad >= 8: break
    net.load_state_dict(state); return net


def layer_change(net, ref):
    out = {}
    for b in BLOCKS:
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items() if k.startswith(b + ".")}
        w = [k for k in sd if k.endswith("0.weight") or (k.endswith(".weight") and sd[k].ndim == 3)]; wk = [k for k in sd if sd[k].ndim == 3][0]
        dw = float(torch.norm(sd[wk] - ref[wk].cpu()) / torch.norm(ref[wk].cpu()))
        bn = [k for k in sd if sd[k].ndim == 1 and (k.endswith("weight") or k.endswith("bias")) and not k.endswith("0.bias")]
        db = float(np.mean([(sd[k] - ref[k].cpu()).abs().mean() for k in bn if k in ref]))
        out[b] = dict(rel_w=dw, bn_abs=db, n_params=int(sum(v.numel() for k, v in sd.items() if v.dtype.is_floating_point and "running" not in k)))
    return out


def stats():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    OUT = os.path.join(NIGHT, "unet_layers_stats.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}; ref = pre_state()
    for k in FD.ALL:
        S = FD.load(k, "test") if k == "d1" else FD.load(k, "train")
        for N in (20, 50):
            for r in range(3):
                key = f"{k}_{N}_{r}"
                if key in res: continue
                sel = np.random.RandomState(1000 + 100 * N + r).permutation(len(S))[:N]; t0 = time.time(); net = ft(S.pos[sel], S.lab[sel], r)
                res[key] = layer_change(net, ref); json.dump(res, open(OUT, "w"), indent=1); print(f"[stats] {key} ({time.time() - t0:.0f} s) " + " ".join(f"{b} {res[key][b]['rel_w']:.2f}" for b in BLOCKS), flush=True)
    names = list(FD.ALL); M = np.array([[np.mean([res[f"{k}_{N}_{r}"][b]["rel_w"] for N in (20, 50) for r in range(3)]) for b in BLOCKS] for k in names])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4)); x = np.arange(len(BLOCKS))
    for i, k in enumerate(names): ax[0].plot(x, M[i], marker="o", label=k, lw=1)
    ax[0].errorbar(x, M.mean(0), M.std(0), color="k", lw=2.5, capsize=4, label="mean ± std over datasets"); ax[0].set_xticks(x); ax[0].set_xticklabels(BLOCKS); ax[0].set_ylabel("relative weight change ||ΔW|| / ||W||"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    npar = [res[f"d1_20_0"][b]["n_params"] for b in BLOCKS]; ax[1].bar(x, npar, color="#999"); ax[1].set_xticks(x); ax[1].set_xticklabels(BLOCKS); ax[1].set_ylabel("parameters in the block"); ax[1].set_yscale("log")
    fig.suptitle("Which blocks of the pre-trained U-Net change when fine-tuned for saccade detection (30 fine-tunings: 5 datasets x N = 20, 50 x 3 draws)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "layer_change.png"), dpi=130)
    print("mean rel. change over datasets:", {b: round(float(M.mean(0)[i]), 3) for i, b in enumerate(BLOCKS)})
    from scipy.stats import spearmanr; print("rank correlation between datasets:", np.round(np.array([[spearmanr(M[i], M[j])[0] for j in range(len(names))] for i in range(len(names))]), 2).tolist())


def freeze(reps, kk):
    OUT = os.path.join(NIGHT, "unet_layers_freeze.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    st = json.load(open(os.path.join(NIGHT, "unet_layers_stats.json"))); M = {b: np.mean([v[b]["rel_w"] for k, v in st.items() if not k.startswith("d1_")]) for b in BLOCKS}
    order = sorted(BLOCKS, key=lambda b: -M[b]); conds = {"all": None, f"top{kk}": set(order[:kk]), f"bottom{kk}": set(order[-kk:]), "head_only": set()}; print("order (changes most -> least, d2/d3/d4/andersson only):", order, {k: sorted(v) if v is not None else None for k, v in conds.items()}, flush=True)
    tr, te = FD.load("d1", "test"), FD.load("d1", "train"); tp, tl = te.pos[:300], te.lab[:300]; S = type("S", (), dict(lab=tl))()
    for N in (10, 20, 50):
        for r in range(reps):
            sel = np.random.RandomState(100 * N + r).permutation(len(tr))[:N]
            for name, tset in conds.items():
                key = f"{name}_{N}_{r}"
                if key in res: continue
                t0 = time.time(); net = ft(tr.pos[sel], tr.lab[sel], r, trainable=tset); s = score(net, tp); e = CP.event_f1((s > 0) & np.isfinite(tp).all(2), S)
                res[key] = [e["ev_f1"], e["kappa"]]; json.dump(res, open(OUT, "w"), indent=1); print(f"[freeze] {key}: F1 {e['ev_f1']:.3f} kappa {e['kappa']:.3f} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["stats", "freeze"]); ap.add_argument("--reps", type=int, default=10); ap.add_argument("--k", type=int, default=3); a = ap.parse_args()
    stats() if a.cmd == "stats" else freeze(a.reps, a.k)
