#!/usr/bin/env python3
"""Multi-session CEBRA across the 4 datasets of Bellet et al. 2019: align datasets with human labels, then adapt to a held-out
dataset with 0 or 1 human-labeled saccade (zero-shot / one-shot), as in CEBRA's multi-session training (model.fit([s1, s2], [c1, c2]))
and adaptation to a new session (model.fit(s, c, adapt=True)) (Schneider, Lee & Mathis, Nature 2023).

Protocol, fixed before any result (leave one dataset out, for each of the 4 datasets as target):
  1. Sources (3 datasets): one encoder per dataset, trained jointly on set A with ALL human labels. Positives of a labeled time
     step: a time step with the same label from ANOTHER source (p = 0.75, this is what aligns the datasets) or from the same one;
     unlabeled time steps: a time neighbour (CEBRA-Time). Negatives: random time steps of all sources.
     Alignment check: each source's set B is labeled by kNN to the OTHER sources' labeled embeddings (cross-dataset decoding).
  2. Target: a new encoder (initialised from the first source encoder), sources frozen, trained on target set A with
       zero-shot   label-free seeds only (Engbert-Kliegl on detrended velocity, free_saccade/cebra_seed.py)
       one-shot    ONE human-labeled saccade (+ the fixation 5-25 samples before and after it: the human clicked a saccade)
       one+seeds   both
     (one-shot: 3 different saccades drawn at random, seeds 0-2).
  3. Target set B labeled by kNN (k = 25, cosine) to a class-balanced bank of the sources' human-labeled samples (+ the one-shot
     samples). Score vs human labels of target set B: Cohen's kappa, event F1.
All at 500 Hz. Run from the repository root:  python free_saccade/cebra_align.py [--quick]
"""
import copy, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "online"))
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import metrics as M
from free_saccade import data as FD, detectors as D, cebra_seed as C
from free_saccade.ssl_models import DenseEncoder, augment_pos

QUICK = "--quick" in sys.argv
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NAMES = ("u1", "u2", "u3", "u4")
B_WIN, N_REF, P_CROSS, K_NN = 32, 64, 0.75, 25
STEPS_SRC, STEPS_TGT, DRAWS = (150, 100, 1) if QUICK else (1000, 400, 3)   # compute budget (about 1.5 h on the M1), not tuned


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.enc = DenseEncoder(6, 64, C.DEPTH); self.proj = torch.nn.Conv1d(64, C.DIM, 1)
    def forward(self, f): return F.normalize(self.proj(self.enc(f)), dim=1)


def views(enc, win, lab, rng):
    """two augmented views of B_WIN windows (half of them with a saccade label if there is one) -> za, zb (B, D, T), labels (B, T)"""
    n, T = lab.shape
    has = np.where((lab == 1).any(1))[0]
    idx = np.r_[rng.choice(has, B_WIN // 2), rng.randint(0, n, B_WIN - B_WIN // 2)] if len(has) else rng.randint(0, n, B_WIN)
    s0 = rng.randint(0, T - C.T_WIN + 1)
    p = win.pos[idx, s0:s0 + C.T_WIN]
    za = enc(torch.as_tensor(C.feats(augment_pos(p, rng, win.fs), win.fs), device=DEV))
    zb = enc(torch.as_tensor(C.feats(augment_pos(p, rng, win.fs), win.fs), device=DEV))
    return za, zb, lab[idx, s0:s0 + C.T_WIN]


def train(encs, wins, labs, trainable, steps, rng):
    params = [p for k in trainable for p in encs[k].parameters()]
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=0.04)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    for k in encs: encs[k].train(k in trainable)
    hist = []
    for it in range(steps):
        V = {k: views(encs[k], wins[k], labs[k], rng) for k in wins}
        pools = {k: {g: np.argwhere(V[k][2] == g) for g in (1, 0)} for k in V}           # (window, t) of each label in the batch
        R, Q, NEG = [], [], []
        for k, (za, zb, lab) in V.items():
            for g in (1, 0, -1):
                w, t = np.where(lab[:, C.TPOS:C.T_WIN - C.TPOS] == g); t = t + C.TPOS
                if not len(w): continue
                j = rng.randint(0, len(w), N_REF); w, t = w[j], t[j]
                R.append(za[w, :, t])
                if g == -1:                                                         # unlabeled: time neighbour (CEBRA-Time)
                    Q.append(zb[w, :, t + rng.randint(-C.TPOS, C.TPOS + 1, N_REF)]); continue
                q = []
                others = [o for o in V if o != k and len(pools[o][g])]
                for _ in range(N_REF):
                    src = others[rng.randint(len(others))] if (others and rng.rand() < P_CROSS) else k
                    pw, pt = pools[src][g][rng.randint(len(pools[src][g]))]
                    q.append(V[src][1][pw, :, pt])
                Q.append(torch.stack(q))
            NEG.append(zb[rng.randint(0, zb.shape[0], N_REF), :, rng.randint(0, C.T_WIN, N_REF)])
        r, q, neg = torch.cat(R), torch.cat(Q), torch.cat(NEG)
        pos_d = (r * q).sum(1) / C.TAU; neg_d = r @ neg.T / C.TAU
        c = neg_d.max(1).values.detach()
        loss = -(pos_d - c).mean() + torch.logsumexp(neg_d - c[:, None], 1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        hist.append(float(loss.detach()))
    return hist


@torch.no_grad()
def embed(enc, win, bs=128):
    enc.eval(); out = []
    for i in range(0, len(win), bs):
        out.append(enc(torch.as_tensor(C.feats(win.pos[i:i + bs], win.fs), device=DEV)).transpose(1, 2).cpu())
    return torch.cat(out)


def bank_of(z, lab, rng, per_class=2000):
    zs, cs = [], []
    for g in (1, 0):
        w, t = np.where(lab == g)
        if len(w): k = rng.choice(len(w), min(per_class, len(w)), replace=False); zs.append(z[w[k], t[k]]); cs.append(np.full(len(k), g))
    return torch.cat(zs), np.concatenate(cs)


@torch.no_grad()
def knn_label(z, bank, cls, win):
    bank = bank.to(DEV); cls = torch.as_tensor(cls, device=DEV, dtype=torch.float32)
    n, T, Dm = z.shape; zz = z.reshape(-1, Dm); p = torch.empty(zz.shape[0])
    for i in range(0, zz.shape[0], 20000):
        top = (zz[i:i + 20000].to(DEV) @ bank.T).topk(min(K_NN, bank.shape[0]), 1).indices
        p[i:i + 20000] = cls[top].mean(1).cpu()
    lab = p.reshape(n, T).numpy() > 0.5
    v, valid = D.velocity(win.pos, win.fs)
    return D.cleanup(lab & valid & ~D.invalid_zone(valid, win.fs), win.fs, min_ms=6.0, merge_ms=0)


def score(lab, w):
    m = M.evaluate_probs(lab.astype(np.float32), w.labels.astype(np.float32), w.fs, thr=0.5, min_event=2, with_ap=False)
    return dict(kappa=m["kappa"], ev_f1=m["ev_f1"])


def one_shot(lab_true, rng):
    """one human saccade of target set A, + fixation 5-25 samples before / after it; everything else unlabeled"""
    ev = []
    for w in range(lab_true.shape[0]):
        for s, e in D.runs_1d(lab_true[w]):
            if e - s + 1 >= 5 and s >= 25 and e < lab_true.shape[1] - 25 and not lab_true[w, s - 25:s].any() and not lab_true[w, e + 1:e + 26].any():
                ev.append((w, s, e))
    w, s, e = ev[rng.randint(len(ev))]
    out = np.full(lab_true.shape, -1, np.int8)
    out[w, s:e + 1] = 1; out[w, s - 25:s - 5] = 0; out[w, e + 6:e + 26] = 0
    return out


def main():
    A = {k: FD.load_labeled("data", names=(k,), which="A", n=200 if QUICK else 1000)[k] for k in NAMES}
    Bs = {k: FD.load_labeled("data", names=(k,), which="B", n=60 if QUICK else 1000)[k] for k in NAMES}
    rows = []
    for tgt in NAMES:
        t0 = time.time(); src = [k for k in NAMES if k != tgt]; rng = np.random.RandomState(0); torch.manual_seed(0)
        encs = {k: Encoder().to(DEV) for k in src}
        hist = train(encs, {k: A[k] for k in src}, {k: A[k].labels.astype(np.int8) for k in src}, src, STEPS_SRC, rng)
        zA = {k: embed(encs[k], A[k]) for k in src}
        # alignment check: each source decoded from the OTHER sources only
        for k in src:
            bk = [bank_of(zA[o], A[o].labels.astype(np.int8), rng) for o in src if o != k]
            lab = knn_label(embed(encs[k], Bs[k]), torch.cat([b[0] for b in bk]), np.concatenate([b[1] for b in bk]), Bs[k])
            r = score(lab, Bs[k]); rows.append(dict(target=tgt, condition=f"alignment: {k} from other sources", draw=0, **r))
            print(f"[target {tgt}] alignment {k} decoded from {[o for o in src if o != k]}: kappa {r['kappa']:.3f} event F1 {r['ev_f1']:.3f}", flush=True)
        sb = [bank_of(zA[o], A[o].labels.astype(np.int8), rng) for o in src]
        src_bank, src_cls = torch.cat([b[0] for b in sb]), np.concatenate([b[1] for b in sb])
        seedsA = C.seeds(A[tgt].pos, A[tgt].fs)
        conds = [("zero-shot (seeds, 0 human label)", None)] + [(c, d) for d in range(DRAWS) for c in ("one-shot (1 human saccade)", "one-shot + seeds")]
        for cond, d in conds:
            lab_t = seedsA.copy() if d is None else one_shot(A[tgt].labels, np.random.RandomState(100 + d))
            if cond == "one-shot + seeds":
                os_ = one_shot(A[tgt].labels, np.random.RandomState(100 + d)); lab_t = seedsA.copy(); lab_t[os_ >= 0] = os_[os_ >= 0]
            e = copy.deepcopy(encs[src[0]]); allenc = dict(encs); allenc[tgt] = e
            train(allenc, {**{k: A[k] for k in src}, tgt: A[tgt]}, {**{k: A[k].labels.astype(np.int8) for k in src}, tgt: lab_t}, [tgt], STEPS_TGT, np.random.RandomState(1 + (d or 0)))
            bank, cls = src_bank, src_cls
            if d is not None:                                                   # the one-shot samples join the bank
                os_ = one_shot(A[tgt].labels, np.random.RandomState(100 + d)); zt = embed(e, A[tgt]); w, t = np.where(os_ >= 0)
                bank = torch.cat([bank, zt[w, t]]); cls = np.concatenate([cls, os_[w, t]])
            lab = knn_label(embed(e, Bs[tgt]), bank, cls, Bs[tgt])
            r = score(lab, Bs[tgt]); rows.append(dict(target=tgt, condition=cond, draw=d if d is not None else 0, **r))
            print(f"[target {tgt}] {cond:34s} draw {d}: kappa {r['kappa']:.3f} event F1 {r['ev_f1']:.3f}", flush=True)
        print(f"[target {tgt}] done in {time.time() - t0:.0f} s (source loss {np.mean(hist[:50]):.2f} -> {np.mean(hist[-50:]):.2f})", flush=True)
    df = pd.DataFrame(rows); os.makedirs("free_results", exist_ok=True)
    out = os.path.join("free_results", "cebra_align" + ("_quick" if QUICK else "") + ".csv"); df.to_csv(out, index=False)
    t = df[~df.condition.str.startswith("alignment")].groupby(["condition", "target"])[["kappa", "ev_f1"]].agg(["mean", "std"]).round(3)
    print(t.to_string()); print("wrote", out)


if __name__ == "__main__":
    main()
