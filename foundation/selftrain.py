#!/usr/bin/env python3
"""Noisy-student self-training (Xie et al., CVPR 2020) of a bidirectional TCN from the label-free universal HMM (fitted on archive/ only).

Round 1: pseudo-labels of every unlabeled window of the pool (archive/ + the unlabeled train splits of the five benchmarks, dino1d.training_pool) come from the
universal HMM; the student sees AUGMENTED positions (rotation, gain, noise, simulated lower sampling rate, added pursuit) while the labels come from the clean
trace (noise only on the student, as in the paper). Round 2: the round-1 student labels the pool again (hard labels) and a new student is trained on them.
No human label is used for training; the labeled TRAIN splits only choose the checkpoint (early stopping on the validation event F1 after HMM decoding).
Final evaluation on the common test subsets with foundation/night_eval.py (direct scores).
Run:  python foundation/selftrain.py --name selftrain [--rounds 2] [--steps 4000] [--max-minutes 25]
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn.functional as F
from foundation import bitcn as BT, dino1d as DN, night_eval as NE, universal_hmm as UH, compare as CP
from foundation.train_lodo import augment, DEV
from foundation.hmm_score import ScoreHMM, Scores
from foundation.dino1d import feats


def pseudo_labels_hmm(pool, grp):
    path = os.path.join(ROOT, "foundation", "runs", "pseudo_hmm.npy")
    if os.path.exists(path): return np.load(path)
    uni = {}
    for fs in (1000.0, 500.0):
        f, v = UH.unit_free_score(UH.archive_windows(fs), "detrended", fs); uni[fs] = ScoreHMM(fs).fit(Scores(f, v, fs))
    out = np.zeros(pool.shape[:2], bool)
    for g in np.unique(grp):
        idx = np.nonzero(grp == g)[0]; fs = 500.0 if g in ("d3", "andersson") else 1000.0; r = int(1000 / fs)
        f, v = UH.unit_free_score(pool[idx, ::r], "detrended", fs); lab = uni[fs].predict(Scores(f, v, fs))
        out[idx] = np.repeat(lab, r, axis=1)[:, :pool.shape[1]]
    np.save(path, out); return out


def train_student(pool, labels, name, steps, lr, max_minutes, seed=0, init=None, criterion="hmm"):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    net = BT.BiTCN().to(DEV)
    if init is not None: net.load_state_dict(init)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2); t0 = time.time(); best, bad, hist, log = -1.0, 0, [], []
    path = os.path.join(ROOT, "foundation", "runs", f"{name}.pt")
    for it in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1.0, (it + 1) / 200) * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.98 + 0.02)
        idx = rng.randint(0, len(pool), 64); aug = augment(pool[idx], rng, True)
        x = torch.as_tensor(feats(aug, 1000.0), device=DEV); y = torch.as_tensor(labels[idx].astype(np.float32), device=DEV); w = torch.as_tensor(np.isfinite(aug).all(2), device=DEV).float()
        net.train(); logit = net(x)
        loss = (F.binary_cross_entropy_with_logits(logit, y * 0.95 + 0.025, pos_weight=torch.tensor(2.0, device=DEV), reduction="none") * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0); opt.step(); hist.append(float(loss.detach()))
        if (it + 1) % 500 == 0:
            res, f1, kp, f1t = BT.val_direct(BT.score_fn_factory(net)); log.append(dict(step=it + 1, loss=float(np.mean(hist[-500:])), val_f1_hmm=f1, val_kappa_hmm=kp, val_f1_thr=f1t))
            cur = f1t if criterion == "thr" else f1; f1 = cur; improved = cur > best + 1e-3
            if improved: best, bad = f1, 0; torch.save(net.state_dict(), path)
            else: bad += 1
            print(f"[{name}] step {it + 1}/{steps} loss {log[-1]['loss']:.4f} | validation: HMM F1 {f1:.3f} kappa {kp:.3f}, threshold F1 {f1t:.3f} (best {best:.3f}{' *saved*' if improved else ''}, patience {bad}/3) | "
                  + " ".join(f"{k} {v['hmm'][0]:.2f}/{v['hmm'][1]:.2f}" for k, v in res.items()) + f" | {time.time() - t0:.0f} s", flush=True)
            if bad >= 3: print(f"[{name}] early stop", flush=True); break
            if (time.time() - t0) / 60 > max_minutes: print(f"[{name}] time cap", flush=True); break
    if not os.path.exists(path): torch.save(net.state_dict(), path)
    net.load_state_dict(torch.load(path, weights_only=False)); return net, best, log, (time.time() - t0) / 60


@torch.no_grad()
def relabel(net, pool, bs=128):
    net.eval(); out = []
    for i in range(0, len(pool), bs): out.append(net(torch.as_tensor(feats(pool[i:i + bs], 1000.0), device=DEV)).cpu().numpy() > 0)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="selftrain"); ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--max-minutes", type=float, default=25)
    ap.add_argument("--criterion", default="hmm", choices=["hmm", "thr"], help="early stopping on the validation F1 of the HMM decoding or of the threshold readout"); a = ap.parse_args()
    from foundation.ssl_vit import pool_cached
    z = np.load(os.path.join(ROOT, "foundation", "runs", "pool.npz")) if os.path.exists(os.path.join(ROOT, "foundation", "runs", "pool.npz")) else None
    pool = pool_cached(); grp = np.load(os.path.join(ROOT, "foundation", "runs", "pool.npz"))["grp"]
    labels = pseudo_labels_hmm(pool, grp); print(f"[{a.name}] pool {pool.shape}, HMM pseudo-labels: {100 * labels.mean():.1f} % saccade samples", flush=True); init = None
    for r in range(1, a.rounds + 1):
        name = f"{a.name}_r{r}"
        net, best, log, minutes = train_student(pool, labels, name, a.steps, a.lr, a.max_minutes, seed=r, criterion=a.criterion)
        res = NE.eval_direct(BT.score_fn_factory(net)); res_t = NE.eval_direct(BT.score_fn_factory(net, tta=True))
        NE.save(name, "label-free self-training (student of the universal HMM)", "none for training (labeled train splits only select the checkpoint)", "Xie et al. 2020 (Noisy Student)",
                dict(round=r, steps=a.steps, lr=a.lr, criterion=a.criterion, pseudo_label_source="universal HMM" if r == 1 else "student of the previous round"), res, minutes, dict(best_val_event_f1=best, log=log))
        NE.save(name + "_tta", "label-free self-training + test-time augmentation", "none for training", "Xie et al. 2020; test-time augmentation over 4 rotations x mirror", dict(round=r), res_t, minutes)
        labels = relabel(net, pool); print(f"[{name}] relabeled the pool: {100 * labels.mean():.1f} % saccade samples", flush=True)


if __name__ == "__main__":
    main()
