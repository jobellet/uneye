#!/usr/bin/env python3
"""Supervised bidirectional TCN trained on the labels of datasets 1, 2, 3 (train splits) with the augmentations of the article (random rotations) + gain, noise,
simulated lower sampling rates, added pursuit; tested zero-shot on dataset 4 and Andersson (never seen) and in-domain on d1-d3 (test subsets). The supervised
counterpart of selftrain.py and the non-causal supervised "foundation" ceiling asked for: same input (8 velocity channels), same scorer.
Training heuristics evaluated from ONE training run (each is a separate row of the report): plain model; + test-time augmentation (4 rotations x mirror, the
logits are averaged); + EMA of the weights (decay 0.999) + test-time augmentation. Early stopping on the validation F1 of 10 % held-out TRAIN trials of d1-d3 only
(no label of d4 / Andersson is read before the final test).
Run:  python foundation/sup_bitcn.py --name sup_bitcn [--steps 6000] [--max-minutes 30]
"""
import argparse, copy, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn.functional as F
from foundation import bitcn as BT, night_eval as NE, data as FD, compare as CP
from foundation.train_lodo import augment, DEV
from foundation.dino1d import feats

BASE = 480


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="sup_bitcn"); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-minutes", type=float, default=30); ap.add_argument("--pos-weight", type=float, default=3.0); ap.add_argument("--inputs", type=int, default=8, choices=[2, 8]); a = ap.parse_args()
    if a.inputs == 2:
        from foundation.dino1d import feats2; BT.FEATS = feats2
    rng = np.random.RandomState(0); torch.manual_seed(0)
    train, val = {}, {}
    for k in ("d1", "d2", "d3"):
        S = FD.load(k, "train"); perm = np.random.RandomState(2).permutation(len(S)); nv = max(len(S) // 10, 20)
        val[k] = FD.Set(k, S.pos[perm[:nv]], S.lab[perm[:nv]], S.coarse); train[k] = (S.pos[perm[nv:]], S.lab[perm[nv:]])
    net = BT.BiTCN(nin=a.inputs).to(DEV); ema = copy.deepcopy(net).eval(); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
    def val_f1(m):
        out = []
        for k, S in val.items():
            s = BT.score_fn_factory(m)(S.pos); out.append(CP.event_f1((s > 0) & np.isfinite(S.pos).all(2), S))
        return float(np.mean([o["ev_f1"] for o in out])), float(np.mean([o["kappa"] for o in out]))
    path = os.path.join(ROOT, "foundation", "runs", f"{a.name}.pt"); best, bad, hist, log, t0 = -1.0, 0, [], [], time.time(); best_n, best_e = -1.0, -1.0; sd_n = sd_e = None   # the weights and the EMA are checkpointed INDEPENDENTLY (an audit found the EMA was reloaded from the step where the plain weights peaked)
    for it in range(a.steps):
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (it + 1) / 200) * (0.5 * (1 + math.cos(math.pi * it / a.steps)) * 0.98 + 0.02)
        P, L = [], []
        for _ in range(64):
            k = ("d1", "d2", "d3")[rng.randint(3)]; pos, lab = train[k]; i = rng.randint(len(pos)); s0 = rng.randint(0, pos.shape[1] - BASE + 1)
            P.append(pos[i, s0:s0 + BASE]); L.append(lab[i, s0:s0 + BASE] == 1)
        aug = augment(np.stack(P), rng, True); x = torch.as_tensor(BT.FEATS(aug, 1000.0), device=DEV); y = torch.as_tensor(np.stack(L).astype(np.float32), device=DEV)
        w = torch.as_tensor(np.isfinite(aug).all(2), device=DEV).float(); net.train()
        loss = (F.binary_cross_entropy_with_logits(net(x), y, pos_weight=torch.tensor(a.pos_weight, device=DEV), reduction="none") * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step(); hist.append(float(loss.detach()))
        d = 0.0 if it < 500 else 0.999
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(d).add_(pn.detach(), alpha=1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if (it + 1) % 500 == 0:
            f1, kp = val_f1(net); f1e, kpe = val_f1(ema); log.append(dict(step=it + 1, loss=float(np.mean(hist[-500:])), val_f1=f1, val_kappa=kp, val_f1_ema=f1e, val_kappa_ema=kpe))
            if f1 > best_n + 1e-3: best_n = f1; sd_n = {k: v.clone() for k, v in net.state_dict().items()}
            if f1e > best_e + 1e-3: best_e = f1e; sd_e = {k: v.clone() for k, v in ema.state_dict().items()}
            cur = max(best_n, best_e); improved = cur > best + 1e-3
            if improved: best, bad = cur, 0; torch.save({"net": sd_n, "ema": sd_e if sd_e is not None else ema.state_dict()}, path)
            else: bad += 1
            print(f"[{a.name}] step {it + 1}/{a.steps} loss {log[-1]['loss']:.4f} | validation (d1-d3 held-out trials): F1 {f1:.3f} kappa {kp:.3f} | EMA F1 {f1e:.3f} kappa {kpe:.3f} (best {best:.3f}{' *saved*' if improved else ''}, patience {bad}/3) | {time.time() - t0:.0f} s", flush=True)
            if bad >= 3: print(f"[{a.name}] early stop", flush=True); break
            if (time.time() - t0) / 60 > a.max_minutes: print(f"[{a.name}] time cap", flush=True); break
    if not os.path.exists(path): torch.save({"net": net.state_dict(), "ema": ema.state_dict()}, path)
    ck = torch.load(path, weights_only=False); net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"]); minutes = (time.time() - t0) / 60
    lab_txt = "labels of d1+d2+d3 train splits (d4 and Andersson never seen)"
    for cid, m, tta, desc in ((a.name, net, False, "bidirectional TCN, supervised"), (a.name + "_tta", net, True, "+ test-time augmentation (4 rotations x mirror)"), (a.name + "_ema_tta", ema, True, "+ EMA weights + test-time augmentation")):
        res = NE.eval_direct(BT.score_fn_factory(m, tta=tta)); NE.save(cid, "supervised (non-causal)", lab_txt, "Bellet et al. 2019 (augmentation); Elmadjian et al. 2021 (TCN); Zemblys et al. 2019 (gazeNet)", dict(vars(a), variant=desc), res, minutes, dict(log=log))


if __name__ == "__main__":
    main()
