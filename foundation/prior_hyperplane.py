#!/usr/bin/env python3
"""Self-supervised embedding + ONE hyperplane per dataset chosen only by physiological priors (no label, no detector seed).

For a dataset: embed its unlabeled recordings (set A) with foundation/ssl_encoder.py, then search the direction w (unit vector in
the 32-d embedding) and the threshold b such that  label = (z . w > b)  maximises a label-free plausibility score S of the
resulting events. Blinks / lost signal are excluded by rule (invalid zone), PSOs are ignored.
S is fixed before any result (weights chosen by hand, never tuned on labels):
  S = mean over events of  -0.5 ((ln duration - ln 25 ms) / 0.7)^2                       saccades last about 5-50 ms
    + mean over inter-event intervals < 200 ms of  -0.5 ((ln interval - ln 200 ms) / 0.7)^2   intervals 100-300 ms, almost never < 50 ms
    - 0.5 ((ln rate - ln 2 per s) / 1.0)^2                                               event rate (rules out "nothing" and "everything")
    + 2 * Spearman(ln amplitude, ln peak speed) + 1 * Spearman(ln amplitude, ln duration)   ballistic main sequence
(events touching the trial border are ignored; fewer than 8 events -> S = -10).
Search: start from the direction that regresses the detrended-speed input channel, the first principal axes (both signs) and random
directions; hill-climbing with decreasing noise on w; the threshold is the best of 9 quantiles of z . w.
THE DECISIVE TEST: over many (w, b) the human-label event F1 is computed on the SAME recordings and compared with S (Spearman):
if S ranks hyperplanes like the human F1, the prior is a usable criterion. Human labels are used for nothing else in the search.
References on the same test split: oracle hyperplane (logistic regression on the human labels of set A, an upper bound for a linear
readout of this embedding) and the median of random hyperplanes.
Run from the repository root:  python foundation/prior_hyperplane.py [--encoder foundation/runs/ssl_encoder.pt] [--targets d1,d2,d3,d4,andersson]
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from foundation import data as FD, ssl_encoder as SE, compare as CP
from free_saccade import detectors as D, cebra_seed as C

FS = 1000.0
QUANTILES = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)
LN_DUR, SD_DUR, LN_INT, SD_INT, LN_RATE, SD_RATE = np.log(25.0), 0.7, np.log(200.0), 0.7, np.log(2.0), 1.0


class Rec:
    """what the score needs about the unlabeled recordings: filled positions, detrended speed, 'bad' mask, embeddings"""
    def __init__(self, S, net):
        pos = S.pos.astype(np.float64)
        self.pos, valid = D.fill(pos)
        vd, _ = C.detrended_velocity(pos, FS)
        self.sp = np.hypot(vd[..., 0], vd[..., 1])
        self.bad = ~valid | D.invalid_zone(valid, FS)
        self.valid_share = valid.mean()
        self.Z = SE.embed(net, S.pos); self.n, self.T = self.Z.shape[:2]
        self.Zf = self.Z.reshape(-1, self.Z.shape[2])
        self.seconds = (~self.bad).sum() / FS
        self.S = S


def events(lab):
    """(win, start, end_exclusive) arrays of all runs of True in lab (n, T)"""
    pad = np.pad(lab.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(pad, axis=1)
    ws, ss = np.nonzero(d == 1); we, se = np.nonzero(d == -1)          # same order: row-major, one end per start
    return ws, ss, se


def prior_score(lab, R):
    w, s, e = events(lab)
    ok = (s > 0) & (e < R.T)                                          # drop events touching the trial border
    w, s, e = w[ok], s[ok], e[ok]
    if len(w) < 8: return -10.0
    dur = (e - s).astype(np.float64)                                  # ms at 1 kHz
    s_dur = np.mean(-0.5 * ((np.log(np.maximum(dur, 1)) - LN_DUR) / SD_DUR) ** 2)
    same = w[1:] == w[:-1]; gap = (s[1:] - e[:-1])[same].astype(np.float64)
    gap = gap[gap < 200]
    s_int = np.mean(-0.5 * ((np.log(np.maximum(gap, 1)) - LN_INT) / SD_INT) ** 2) if len(gap) else 0.0
    rate = len(w) / max(R.seconds, 1e-9)
    s_rate = -0.5 * ((np.log(max(rate, 1e-6)) - LN_RATE) / SD_RATE) ** 2
    amp = np.hypot(*(R.pos[w, np.minimum(e, R.T - 1)] - R.pos[w, s - 1]).T)
    flat = np.concatenate([R.sp.ravel(), [0.0]]); a = w * R.T + s; b = w * R.T + e
    idx = np.empty(2 * len(a), np.int64); idx[0::2], idx[1::2] = a, b
    peak = np.maximum.reduceat(flat, idx)[0::2]
    good = (amp > 0) & (peak > 0)
    if good.sum() < 8: return -10.0
    r1 = spearmanr(np.log(amp[good]), np.log(peak[good]))[0]; r2 = spearmanr(np.log(amp[good]), np.log(dur[good]))[0]
    return float(s_dur + s_int + s_rate + 2 * np.nan_to_num(r1) + np.nan_to_num(r2))


def labels_of(R, w, q=None, b=None):
    s = (R.Zf @ w).reshape(R.n, R.T)
    if b is None: b = np.quantile(s[~R.bad], q)
    return (s > b) & ~R.bad, b


def best_threshold(R, w):
    best = (-1e9, None, None)
    for q in QUANTILES:
        lab, b = labels_of(R, w, q=q); sc = prior_score(lab, R)
        if sc > best[0]: best = (sc, b, q)
    return best


def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def search(R, rng, n_random=4, iters=120):
    ch = C.feats(R.S.pos, FS)[..., 5].reshape(-1)                      # detrended-speed input channel
    w_reg = unit(np.linalg.lstsq(R.Zf[::7], ch[::7], rcond=None)[0])
    U, sv, Vt = np.linalg.svd(R.Zf[::11] - R.Zf[::11].mean(0), full_matrices=False)
    inits = [w_reg, -w_reg] + [s * Vt[i] for i in range(3) for s in (1, -1)] + [unit(rng.randn(R.Zf.shape[1])) for _ in range(n_random)]
    visited, best = [], (-1e9, None, None, None)
    for w0 in inits:
        w = unit(w0); sc, b, q = best_threshold(R, w); visited.append((sc, w.copy(), b))
        for it in range(iters):
            sig = 0.4 * (1 - it / iters) + 0.03
            w2 = unit(w + sig * rng.randn(len(w))); sc2, b2, q2 = best_threshold(R, w2); visited.append((sc2, w2.copy(), b2))
            if sc2 > sc: w, sc, b, q = w2, sc2, b2, q2
        if sc > best[0]: best = (sc, w.copy(), b, q)
    return best, visited


def f1_of(R, w, b):
    lab, _ = labels_of(R, w, b=b)
    return CP.event_f1(lab, R.S)["ev_f1"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", default=None); ap.add_argument("--targets", default=",".join(FD.ALL))
    ap.add_argument("--trials", type=int, default=300); ap.add_argument("--iters", type=int, default=120); a = ap.parse_args()
    net = SE.load_encoder(a.encoder); rows = []
    for nm in a.targets.split(","):
        t0 = time.time(); rng = np.random.RandomState(0)
        A, B = FD.load(nm, "train"), FD.load(nm, "test")
        sel = np.random.RandomState(1).permutation(len(A))[:a.trials if nm != "andersson" else 80]
        A = FD.Set(nm, A.pos[sel], A.lab[sel], A.coarse); RA, RB = Rec(A, net), Rec(B, net)
        (sc, w, b, q), visited = search(RA, rng, iters=a.iters)
        # decisive test: prior score vs human event F1 over many hyperplanes of the search (+ random ones), on set A
        pick = [visited[i] for i in np.random.RandomState(2).choice(len(visited), min(200, len(visited)), replace=False)]
        scores = [p[0] for p in pick]; f1s = [f1_of(RA, p[1], p[2]) for p in pick]
        rho = spearmanr(scores, f1s)[0]
        rnd = []                                                      # random hyperplanes, all thresholds: what a direction chosen without any criterion gives
        for qq in QUANTILES:
            for _ in range(3):
                wr = unit(rng.randn(RA.Zf.shape[1])); lab_r, _ = labels_of(RA, wr, q=qq); rnd.append(CP.event_f1(lab_r, RA.S)["ev_f1"])
        # references on the test split
        f1_found = f1_of(RB, w, b)
        mask = (A.lab >= 0) & ~RA.bad
        lr = LogisticRegression(max_iter=300, class_weight="balanced").fit(RA.Zf[mask.ravel()][::3], (A.lab == 1)[mask][::3])
        lab_or = (RB.Zf @ lr.coef_[0] + lr.intercept_[0] > 0).reshape(RB.n, RB.T) & ~RB.bad
        f1_oracle = CP.event_f1(lab_or, B)["ev_f1"]
        r = dict(dataset=nm, prior_score=sc, f1_found_on_B=f1_found, f1_oracle_hyperplane_on_B=f1_oracle, f1_random_hyperplane_median_on_A=float(np.median(rnd)),
                 spearman_prior_vs_f1=float(rho), best_state_f1_on_A=f1_of(RA, w, b), n_states=len(visited), seconds=time.time() - t0)
        rows.append(r)
        print(f"[{nm:9s}] prior-chosen hyperplane: event F1 {f1_found:.3f} on B | oracle (labels) {f1_oracle:.3f} | random hyperplanes (median) {r['f1_random_hyperplane_median_on_A']:.3f} | "
              f"Spearman(prior score, human F1) over {len(pick)} hyperplanes = {rho:+.2f}  ({r['seconds']:.0f} s)", flush=True)
    out = os.path.join(ROOT, "foundation", "runs", "prior_hyperplane.json"); json.dump(rows, open(out, "w"), indent=1, default=float); print("wrote", out)


if __name__ == "__main__":
    main()
