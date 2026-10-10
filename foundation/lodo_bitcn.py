#!/usr/bin/env python3
"""Zero-shot generalization: leave-one-dataset-out training of the 2-channel (vx, vy) bidirectional TCN. For each held-out dataset X in d1,d2,d3,d4,andersson:
train on the labeled TRAIN splits (set A) of the OTHER four datasets only (no trial, no label of X), early stopping on 10 % held-out trials of those same four datasets,
test on the common test subset of X (foundation/night_eval.data(), threshold at 0, test-time augmentation). Compared with U'n'Eye trained IN the dataset (or on d1+d2+d3):
rows ref_uneye / ref_uneye_andersson_own of foundation/runs/night/. Writes foundation/runs/night/lodo_bitcn.json.   python foundation/lodo_bitcn.py [--steps 5000] [--only d1,d2]
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn.functional as F
from foundation import bitcn as BT, data as FD, compare as CP, night_eval as NE
from foundation.dino1d import feats2
from foundation.train_lodo import augment, DEV
BT.FEATS = feats2; BASE = 480; OUT = os.path.join(NE.NIGHT, "lodo_bitcn.json")


def run(held, steps, max_minutes, lr=1e-3, seed=0):
    rng = np.random.RandomState(seed); torch.manual_seed(seed); train, val = {}, {}
    for k in FD.ALL:
        if k == held: continue
        S = FD.load(k, "train"); perm = np.random.RandomState(2).permutation(len(S)); nv = max(len(S) // 10, 20)
        val[k] = FD.Set(k, S.pos[perm[:nv]], S.lab[perm[:nv]], S.coarse); train[k] = (S.pos[perm[nv:]], S.lab[perm[nv:]])
    names = list(train); net = BT.BiTCN(nin=2).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    def val_f1():
        r = [CP.event_f1((BT.score_fn_factory(net)(S.pos[:60]) > 0) & np.isfinite(S.pos[:60]).all(2), type("S", (), dict(lab=S.lab[:60]))()) for S in val.values()]
        return float(np.mean([o["ev_f1"] for o in r])), float(np.mean([o["kappa"] for o in r]))
    path = os.path.join(ROOT, "foundation", "runs", f"lodo_bitcn_{held}.pt"); best, bad, t0, hist = -1.0, 0, time.time(), []
    for it in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 200) * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.98 + 0.02)
        P, L = [], []
        for _ in range(64):
            pos, lab = train[names[rng.randint(len(names))]]; i = rng.randint(len(pos)); s0 = rng.randint(0, pos.shape[1] - BASE + 1); P.append(pos[i, s0:s0 + BASE]); L.append(lab[i, s0:s0 + BASE])
        aug = augment(np.stack(P), rng, True); lab = np.stack(L); x = torch.as_tensor(feats2(aug, 1000.0), device=DEV); y = torch.as_tensor((lab == 1).astype(np.float32), device=DEV)
        w = torch.as_tensor(np.isfinite(aug).all(2) & (lab >= 0), device=DEV).float(); net.train()
        loss = (F.binary_cross_entropy_with_logits(net(x), y, pos_weight=torch.tensor(1.5, device=DEV), reduction="none") * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step(); hist.append(float(loss.detach()))
        if (it + 1) % 500 == 0:
            f1, kp = val_f1(); improved = f1 > best + 1e-3
            if improved: best, bad = f1, 0; torch.save(net.state_dict(), path)
            else: bad += 1
            print(f"[lodo {held}] step {it + 1}/{steps} loss {np.mean(hist[-500:]):.4f} | val (other datasets) F1 {f1:.3f} kappa {kp:.3f} (best {best:.3f}, patience {bad}/3) | {time.time() - t0:.0f} s", flush=True)
            if bad >= 3 or (time.time() - t0) / 60 > max_minutes: break
    net.load_state_dict(torch.load(path, weights_only=False)); S = NE.data()[held][1]
    s = BT.score_fn_factory(net, tta=True)(S.pos); r = CP.event_f1((s > 0) & np.isfinite(S.pos).all(2), S); return dict(f1=r["ev_f1"], kappa=r["kappa"], val_f1=best, minutes=(time.time() - t0) / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--max-minutes", type=float, default=12); ap.add_argument("--only", default=""); a = ap.parse_args()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for k in FD.ALL:
        if (a.only and k not in a.only.split(",")) or k in res: continue
        res[k] = run(k, a.steps, a.max_minutes); json.dump(res, open(OUT, "w"), indent=1); print(f"[lodo {k}] TEST (never seen): F1 {res[k]['f1']:.3f} kappa {res[k]['kappa']:.3f}", flush=True)
