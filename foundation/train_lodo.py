#!/usr/bin/env python3
"""Leave-one-dataset-out evaluation of the foundation model (foundation/model.py).

For every target dataset T in (d1, d2, d3, d4, andersson), fixed before any result:
  lodo        trained on the 4 OTHER datasets (train splits), nuisance augmentation ON   -> zero-shot on T's test split
  lodo-noaug  same, augmentation OFF                                                     -> what the augmentation brings
  in-domain   trained on all 5 train splits including T's, augmentation ON              -> supervised ceiling
Augmentation (positions, per window): rotation, gain x0.7-1.4, white noise 0-0.005 deg, sampling rate 1000/500/250 Hz simulated by
decimating and re-interpolating, and (binary datasets only) added sinusoidal pursuit up to 15 deg/s.
Each step draws the same number of windows from every training dataset. Loss: 5-class cross-entropy (class weights 1/sqrt(freq))
on the multi-class dataset; on binary datasets p(saccade) vs 1 - p(saccade) (saccade weight 2).
Scores on T's test split (1 kHz, samples with a label only): saccade-vs-rest Cohen's kappa and event F1 (online/metrics.py);
for andersson also the 5-class kappa and the kappa between its two human coders (RA vs MN) on the same samples.
Run from the repository root:  python foundation/train_lodo.py [--quick] [--targets d1,d2]
"""
import argparse, os, sys, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
import metrics as M
from foundation import data as FD, model as FM

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
T_WIN = 700                                   # dataset 4 trials are 750 samples; receptive field 259


def augment(p, rng, coarse, on=True):
    p = p.astype(np.float64).copy()
    if not on: return p
    n, T, _ = p.shape
    th = rng.uniform(0, 2 * np.pi, n); c, s = np.cos(th), np.sin(th)
    p = np.stack([c[:, None] * p[..., 0] - s[:, None] * p[..., 1], s[:, None] * p[..., 0] + c[:, None] * p[..., 1]], 2)
    p *= rng.uniform(0.7, 1.4, (n, 1, 1))
    t = np.arange(T)
    for i in range(n):
        r = rng.choice([1, 1, 2, 4])
        if r > 1:                                                          # lower sampling rate, re-interpolated to 1 kHz
            for a in range(2):
                ok = np.isfinite(p[i, ::r, a])
                if ok.sum() > 2: p[i, :, a] = np.where(np.isfinite(p[i, :, a]), np.interp(t, t[::r][ok], p[i, ::r, a][ok]), np.nan)
        if coarse and rng.rand() < 0.3:                                    # added smooth pursuit (labels stay valid: non-saccade)
            f = rng.uniform(0.2, 1.0); A = rng.uniform(2, 15) / (2 * np.pi * f); ph = rng.uniform(0, 2 * np.pi); ang = rng.uniform(0, 2 * np.pi)
            m = A * np.sin(2 * np.pi * f * t / 1000.0 + ph)
            p[i, :, 0] += m * np.cos(ang); p[i, :, 1] += m * np.sin(ang)
    return p + rng.normal(0, 1, p.shape) * rng.uniform(0, 0.005, (n, 1, 1))


def batch(sets, rng, per_set, aug):
    X, Y, C = [], [], []
    for S in sets:
        n, T = S.lab.shape
        idx = rng.randint(0, n, per_set); s0 = rng.randint(0, T - T_WIN - FM.LOOKAHEAD + 1, per_set)
        p = np.stack([S.pos[i, s:s + T_WIN + FM.LOOKAHEAD] for i, s in zip(idx, s0)])
        l = np.stack([S.lab[i, s:s + T_WIN] for i, s in zip(idx, s0)])          # label of the sample LOOKAHEAD before the output
        X.append(FM.inputs(augment(p, rng, S.coarse, aug))); Y.append(l); C.append(np.full(per_set, S.coarse))
    return torch.as_tensor(np.concatenate(X), device=DEV), torch.as_tensor(np.concatenate(Y).astype(np.int64), device=DEV), np.concatenate(C)


def loss_fn(logits, y, coarse, cw):
    logits = logits[:, :, FM.LOOKAHEAD:]                                   # output t+L describes sample t
    logp = F.log_softmax(logits, 1)
    total, parts = 0.0, 0
    cm = torch.as_tensor(coarse, device=DEV)
    if (~cm).any():                                                        # multi-class
        lp, yy = logp[~cm], y[~cm]
        total = total + F.nll_loss(lp, yy.clamp(min=0) * (yy >= 0) + (yy < 0) * -100, weight=cw, ignore_index=-100); parts += 1
    if cm.any():                                                           # binary: saccade vs everything else
        lp, yy = logp[cm], y[cm]
        ls = lp[:, 1]; lr = torch.logsumexp(torch.cat([lp[:, :1], lp[:, 2:]], 1), 1)
        m = yy >= 0; w = torch.where(yy == 1, 2.0, 1.0)
        total = total + -((torch.where(yy == 1, ls, lr) * w)[m]).sum() / w[m].sum(); parts += 1
    return total / parts


@torch.no_grad()
def predict(net, S, bs=16):
    net.eval(); P = []
    for i in range(0, len(S), bs):
        p = S.pos[i:i + bs]
        p = np.concatenate([p, np.full((len(p), FM.LOOKAHEAD, 2), np.nan, np.float32)], 1)
        P.append(F.softmax(net(torch.as_tensor(FM.inputs(p), device=DEV)), 1)[:, :, FM.LOOKAHEAD:].cpu().numpy())
    return np.concatenate(P)                                               # (n, K, T)


def scores(P, S, other_coder=None):
    lab = S.lab; mask = lab >= 0
    ps = np.where(mask, P[:, 1], 0.0); ys = np.where(mask, lab == 1, False)
    m = M.evaluate_probs(ps.astype(np.float32), ys.astype(np.float32), FD.FS, thr=0.5, min_event=3, with_ap=False)
    pred = P.argmax(1)
    out = dict(kappa_sacc=cohen_kappa_score(ys[mask], (pred == 1)[mask]), ev_f1=m["ev_f1"], ev_recall=m["ev_recall"], ev_precision=m["ev_precision"])
    if not S.coarse:
        out["kappa_5class"] = cohen_kappa_score(lab[mask], pred[mask])
        for k, name in enumerate(FD.CLASSES):
            tp = ((pred == k) & (lab == k) & mask).sum(); fp = ((pred == k) & (lab != k) & mask).sum(); fn = ((pred != k) & (lab == k) & mask).sum()
            out[f"f1_{name}"] = 2 * tp / max(2 * tp + fp + fn, 1)
        if other_coder is not None:
            mm = mask & (other_coder >= 0)
            out["human_RA_vs_MN_kappa_5class"] = cohen_kappa_score(lab[mm], other_coder[mm])
            out["human_RA_vs_MN_kappa_sacc"] = cohen_kappa_score(lab[mm] == 1, other_coder[mm] == 1)
    return out


def train(sets, steps, aug, seed=0):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    net = FM.Foundation().to(DEV)
    multi = [S for S in sets if not S.coarse]
    if multi:
        y = np.concatenate([S.lab[S.lab >= 0] for S in multi]); f = np.bincount(y, minlength=FD.K) / len(y)
        cw = torch.as_tensor(1 / np.sqrt(np.maximum(f, 1e-3)), dtype=torch.float32, device=DEV); cw = cw / cw.mean()
    else:
        cw = None
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    hist = []
    for it in range(steps):
        net.train()
        x, y, c = batch(sets, rng, max(32 // len(sets), 4), aug)
        loss = loss_fn(net(x), y, c, cw)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        hist.append(float(loss.detach()))
    return net, hist


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--targets", default=",".join(FD.ALL))
    ap.add_argument("--steps", type=int, default=3000); a = ap.parse_args()
    steps = 200 if a.quick else a.steps
    t0 = time.time()
    train_sets = {k: FD.load(k, "train") for k in FD.ALL}; test_sets = {k: FD.load(k, "test") for k in FD.ALL}
    mn = FD.load_andersson("test", coder="MN").lab
    if a.quick:
        train_sets = {k: FD.Set(k, S.pos[:100], S.lab[:100], S.coarse) for k, S in train_sets.items()}
    print(f"data loaded in {time.time() - t0:.0f} s; device {DEV}; " + ", ".join(f"{k}: train {train_sets[k].pos.shape} test {test_sets[k].pos.shape}" for k in FD.ALL), flush=True)
    rows = []
    os.makedirs(os.path.join(ROOT, "foundation", "runs"), exist_ok=True)
    for tgt in a.targets.split(","):
        for cond, use, aug in (("lodo", [k for k in FD.ALL if k != tgt], True), ("lodo-noaug", [k for k in FD.ALL if k != tgt], False), ("in-domain", list(FD.ALL), True)):
            t1 = time.time()
            net, hist = train([train_sets[k] for k in use], steps, aug)
            S = test_sets[tgt]
            r = dict(target=tgt, condition=cond, trained_on="+".join(use), steps=steps, loss_end=float(np.mean(hist[-100:])),
                     **scores(predict(net, S), S, mn if tgt == "andersson" else None))
            rows.append(r)
            print(f"[{tgt:9s}] {cond:10s} kappa(sacc) {r['kappa_sacc']:.3f}  event F1 {r['ev_f1']:.3f}" +
                  (f"  kappa(5 classes) {r['kappa_5class']:.3f}  [humans RA vs MN: {r.get('human_RA_vs_MN_kappa_5class', float('nan')):.3f}]" if 'kappa_5class' in r else "") +
                  f"  ({time.time() - t1:.0f} s)", flush=True)
            if cond == "lodo" and not a.quick:
                torch.save({"state": net.state_dict(), "trained_on": use}, os.path.join(ROOT, "foundation", "runs", f"lodo_{tgt}.pt"))
    out = os.path.join(ROOT, "foundation", "runs", "lodo" + ("_quick" if a.quick else "") + ".json")
    json.dump(rows, open(out, "w"), indent=1); print("wrote", out)


if __name__ == "__main__":
    main()
