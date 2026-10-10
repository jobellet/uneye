#!/usr/bin/env python3
"""Is the zero-shot gap an ANNOTATION-CONVENTION mismatch rather than a representation problem? (suggestions of the Antigravity `agy` review, see docs/BENCHMARK_SUMMARY.md section 7)
Stage 1 = the leave-one-dataset-out BiTCN of foundation/lodo_bitcn.py (trained WITHOUT the dataset it is applied to; checkpoints foundation/runs/lodo_bitcn_<d>.pt).
  scores   cache the logits of every LODO net on the train subset (calibration pool) and the test subset (common subsets of night_eval.data()) -> runs/lodo_scores.npz
  diag     (a) event F1 against the overlap tolerance (IoU 0.1 ... 0.7), (b) onset / offset error of matched events, (c) event F1 and kappa after removing predicted events
           below an amplitude floor (0 ... 1.5 deg), (d) human-human ceiling on Andersson (coders RA vs MN)  -> night/conventions_diag.json, docs/figs_diag/
  calib    stage 2 = a 3-parameter calibrator fitted by grid search on N labeled trials of the TARGET dataset: amplitude floor, onset / offset shift (pre, post samples), merge gap;
           N = 0, 3, 5, 10, 20, 50; dataset 1 uses the protocol of table 3 of the summary (calibrate on draws of set B, test on set A[:300]); the other datasets calibrate on their train subset
           and test on the common test subset  -> night/conventions_calib.json
  hybrid   scaled-posterior decoding (Bourlard & Morgan 1994): p(x|state) ~ posterior / prior fed to a minimum-duration chain decoded by Viterbi, no label used for the decoder
           (the prior is the saccade fraction of the TRAINING datasets), against the threshold at 0  -> night/conventions_hybrid.json
Labels are never used to select anything except the calibrator parameters (on the N calibration trials only); the test subsets are never used for selection.
python foundation/conventions.py scores | diag | calib | hybrid
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from free_saccade import detectors as D
from foundation import bitcn as BT, data as FD, compare as CP, night_eval as NE
from foundation.dino1d import feats2
from foundation.train_lodo import DEV
BT.FEATS = feats2; RUNS = os.path.join(ROOT, "foundation", "runs"); NIGHT = NE.NIGHT; FIG = os.path.join(ROOT, "docs", "figs_diag"); os.makedirs(FIG, exist_ok=True)
CACHE = os.path.join(RUNS, "lodo_scores.npz"); DS = list(FD.ALL)


def _net(k):
    net = BT.BiTCN(nin=2).to(DEV); net.load_state_dict(torch.load(os.path.join(RUNS, f"lodo_bitcn_{k}.pt"), map_location=DEV, weights_only=False)); return net.eval()


def make_scores():
    out = {}; d = NE.data()
    for k in DS:
        fn = BT.score_fn_factory(_net(k), tta=True); A, B = d[k]; out[f"{k}_train"] = fn(A.pos).astype(np.float32); out[f"{k}_test"] = fn(B.pos).astype(np.float32); print("scores", k, flush=True)
    fn = BT.score_fn_factory(_net("d1"), tta=True); setB, setA = FD.load("d1", "test"), FD.load("d1", "train")                      # dataset 1 protocol of table 3
    out["d1_setB_all"] = fn(setB.pos).astype(np.float32); out["d1_setA_300"] = fn(setA.pos[:300]).astype(np.float32); np.savez(CACHE, **out)


def scores():
    if not os.path.exists(CACHE): make_scores()
    return dict(np.load(CACHE))


# ------------------------------------------------------------------ events
def pos_filled(pos):
    p, _ = D.fill(np.asarray(pos, np.float64)[None]); return p[0]


def events_of(mask):
    return D.runs_1d(mask)


def amplitude(p, s, e):
    """largest displacement from the onset position inside the run, in the units of the positions (degrees)"""
    seg = p[s:e + 1] - p[s]; return float(np.sqrt((seg ** 2).sum(1)).max())


def apply_calibration(mask, p, amin, pre, post, gap):
    """mask (T,) bool -> mask after: merge runs separated by <= gap samples, drop merged runs of amplitude < amin, then move onset by -pre and offset by +post samples"""
    T = len(mask); ev = events_of(mask); merged = []
    for s, e in ev:
        if merged and s - merged[-1][1] - 1 <= gap: merged[-1][1] = e
        else: merged.append([s, e])
    out = np.zeros(T, bool)
    for s, e in merged:
        if amplitude(p, s, e) < amin: continue
        s2, e2 = max(s - pre, 0), min(e + post, T - 1)
        if e2 >= s2: out[s2:e2 + 1] = True
    return out


def valid_mask(pos): return np.isfinite(pos).all(2)


def pred_masks(logits, pos, thr=0.0): return (logits > thr) & valid_mask(pos)


def ev_score(pred, pos, lab):
    pred = pred & valid_mask(pos)                                              # nothing is predicted on invalid samples (a shift or a merge must not extend into a blink)
    S = type("S", (), dict(lab=lab))(); r = CP.event_f1(pred, S); return r["ev_f1"], r["kappa"]


# ------------------------------------------------------------------ (a)-(d) diagnostics
def iou_match(pr, tr, tau, min_len=3):
    pr = [r for r in pr if r[1] - r[0] + 1 >= min_len]; used = set(); tp = 0; offs = []
    for (ts, te) in tr:
        best, bj = 0.0, -1
        for j, (ps, pe) in enumerate(pr):
            if j in used: continue
            inter = min(te, pe) - max(ts, ps) + 1
            if inter <= 0: continue
            iou = inter / float((te - ts + 1) + (pe - ps + 1) - inter)
            if iou > best: best, bj = iou, j
        if bj >= 0 and best >= tau: used.add(bj); tp += 1; offs.append((pr[bj][0] - ts, pr[bj][1] - te))
    return tp, len(pr) - tp, len(tr) - tp, offs


def diag():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sc = scores(); d = NE.data(); res = {"iou": {}, "offsets": {}, "amp": {}, "ceiling": {}}; taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]; amps = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    for k in DS:
        B = d[k][1]; pred = pred_masks(sc[f"{k}_test"], B.pos) & (B.lab >= 0); truth = B.lab == 1; res["iou"][k] = []
        PR = [events_of(pred[i]) for i in range(len(pred))]; TR = [events_of(truth[i]) for i in range(len(truth))]
        for tau in taus:
            tp = fp = fn = 0
            for pr, tr in zip(PR, TR): a, b, c, o = iou_match(pr, tr, tau); tp += a; fp += b; fn += c
            res["iou"][k].append(2 * tp / max(2 * tp + fp + fn, 1))
        on, off = [], []
        for pr, tr in zip(PR, TR): _, _, _, o = iou_match(pr, tr, 0.1); on += [x[0] for x in o]; off += [x[1] for x in o]
        res["offsets"][k] = dict(n=len(on), onset_mean=float(np.mean(on)), onset_std=float(np.std(on)), onset_median=float(np.median(on)), offset_mean=float(np.mean(off)), offset_std=float(np.std(off)), offset_median=float(np.median(off)))
        res["amp"][k] = []
        for a in amps:
            m = np.zeros_like(pred)
            for i in range(len(pred)): m[i] = apply_calibration(pred[i], pos_filled(B.pos[i]), a, 0, 0, 0)
            res["amp"][k].append(ev_score(m, B.pos, B.lab)); print("amp", k, a, res["amp"][k][-1], flush=True)
    for a_, b_ in (("RA", "MN"), ("MN", "RA")):
        X, Y = FD.load_andersson("test", coder=a_), FD.load_andersson("test", coder=b_)
        if X.pos.shape == Y.pos.shape:
            Xs, Ys = NE.subset(X, 60, 5), NE.subset(Y, 60, 5)                                                          # the same 60 trials as the benchmark rows
            S = type("S", (), dict(lab=Xs.lab))(); r = CP.event_f1(Ys.lab == 1, S); res["ceiling"][f"{b_} scored against {a_}"] = [r["ev_f1"], r["kappa"], int(len(Xs.lab))]
        else: res["ceiling"][f"{b_} scored against {a_}"] = f"trial arrays differ {X.pos.shape} vs {Y.pos.shape}"
    json.dump(res, open(os.path.join(NIGHT, "conventions_diag.json"), "w"), indent=1, default=float)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k in DS: ax[0].plot(taus, res["iou"][k], marker="o", label=k); ax[1].plot(amps, [x[0] for x in res["amp"][k]], marker="o", label=k); ax[2].plot(amps, [x[1] for x in res["amp"][k]], marker="o", label=k)
    ax[0].set_xlabel("minimum IoU to count a match"); ax[0].set_ylabel("event F1"); ax[1].set_xlabel("amplitude floor on predicted events (deg)"); ax[1].set_ylabel("event F1"); ax[2].set_xlabel("amplitude floor (deg)"); ax[2].set_ylabel("kappa")
    for a in ax: a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle("Zero-shot (leave-one-dataset-out) BiTCN: overlap tolerance and amplitude floor"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "conventions.png"), dpi=130); plt.close(fig)
    print(json.dumps(res["offsets"], indent=0)); print(res["ceiling"])


# ------------------------------------------------------------------ calibrator (stage 2)
# the grid was first too narrow (post >= -4): the diagnostics showed the predicted offset 19 ms later than the human one on Andersson, and the optimum sat on the grid edge
GRID = [(a, pre, post, g) for a in (0.0, 0.2, 0.4, 0.6, 1.0) for pre in (-8, -4, 0, 4, 8) for post in (-24, -16, -8, -4, 0, 4, 8) for g in (0, 8, 20)]


def fit_calibrator(logits, pos, lab, thr=0.0):
    pm = pred_masks(logits, pos, thr); P = [pos_filled(pos[i]) for i in range(len(pos))]; best, bp = -9, (0.0, 0, 0, 0)
    for prm in GRID:
        m = np.stack([apply_calibration(pm[i], P[i], *prm) for i in range(len(pm))]); f, k = ev_score(m, pos, lab); obj = 0.5 * (f + (0.0 if np.isnan(k) else k))      # kappa is undefined when a small set has no saccade
        if obj > best: best, bp = obj, prm
    return bp


def calib():
    OUT = os.path.join(NIGHT, "conventions_calib.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}; sc = scores(); d = NE.data(); Ns = (0, 3, 5, 10, 20, 50)
    def run(tag, calL, calP, calY, teL, teP, teY, draws, ns):
        Pte = [pos_filled(teP[i]) for i in range(len(teP))]; pm_te = pred_masks(teL, teP)
        for N in ns:
            for r in range(draws):
                key = f"{tag}_{N}_{r}"
                if key in res: continue
                if N == 0: prm = (0.0, 0, 0, 0)
                else: sel = np.random.RandomState(100 * N + r).permutation(len(calP))[:N]; prm = fit_calibrator(calL[sel], calP[sel], calY[sel])
                m = np.stack([apply_calibration(pm_te[i], Pte[i], *prm) for i in range(len(pm_te))]); f, k = ev_score(m, teP, teY)
                res[key] = [f, k, list(prm)]; json.dump(res, open(OUT, "w"), indent=1, default=float); print(f"[calib] {key}: F1 {f:.3f} kappa {k:.3f} params {prm}", flush=True)
                if N == 0: break
    setB, setA = FD.load("d1", "test"), FD.load("d1", "train")
    run("d1", sc["d1_setB_all"], setB.pos, setB.lab, sc["d1_setA_300"], setA.pos[:300], setA.lab[:300], 10, Ns)
    for k in ("d2", "d3", "d4", "andersson"):
        A, B = d[k]; run(k, sc[f"{k}_train"], A.pos, A.lab, sc[f"{k}_test"], B.pos, B.lab, 5, (0, 3, 10, 30 if k != "andersson" else 20))


# ------------------------------------------------------------------ hybrid decoding
def viterbi_chain(loglik_fix, loglik_sac, d_min=4, p_on=1 / 150.0, p_off=1 / 20.0):
    """states: 0 = fixation, 1..d_min = saccade chain (the last one loops); returns the saccade mask. loglik_* (T,)"""
    T = len(loglik_fix); S = d_min + 1; NEG = -1e18; trans = np.full((S, S), NEG)
    trans[0, 0] = np.log(1 - p_on); trans[0, 1] = np.log(p_on)
    for i in range(1, d_min): trans[i, i + 1] = 0.0
    trans[d_min, d_min] = np.log(1 - p_off); trans[d_min, 0] = np.log(p_off)
    em = np.stack([loglik_fix] + [loglik_sac] * d_min, 1); dp = np.full((T, S), NEG); bp = np.zeros((T, S), int); dp[0, 0] = em[0, 0]
    for t in range(1, T):
        cand = dp[t - 1][:, None] + trans; bp[t] = cand.argmax(0); dp[t] = cand.max(0) + em[t]
    s = int(dp[-1].argmax()); path = np.zeros(T, int)
    for t in range(T - 1, -1, -1): path[t] = s; s = bp[t, s]
    return path > 0


def hybrid():
    OUT = os.path.join(NIGHT, "conventions_hybrid.json"); sc = scores(); d = NE.data(); res = {}
    for k in DS:
        others = [o for o in DS if o != k]; fr = float(np.mean([np.mean(d[o][0].lab[d[o][0].lab >= 0] == 1) for o in others]))      # saccade fraction of the TRAINING datasets
        B = d[k][1]; L = sc[f"{k}_test"]; v = valid_mask(B.pos); res[k] = {"prior": fr, "threshold": ev_score((L > 0) & v, B.pos, B.lab)}
        for temp in (1.0, 3.0):
            z = np.clip(L / temp, -12, 12); lp = -np.logaddexp(0, -z); lq = -np.logaddexp(0, z)                                  # log p, log (1 - p)
            ll_fix = np.where(v, lq - np.log(1 - fr), 0.0); ll_sac = np.where(v, lp - np.log(fr), 0.0)               # invalid samples carry no evidence (their logit is a placeholder 0)
            m = np.zeros(L.shape, bool)
            for i in range(len(L)): m[i] = viterbi_chain(ll_fix[i], ll_sac[i])
            res[k][f"hybrid_T{temp:g}"] = ev_score(m & v, B.pos, B.lab)
        print("hybrid", k, res[k], flush=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["scores", "diag", "calib", "hybrid"]); a = ap.parse_args()
    {"scores": lambda: (make_scores(), None)[1], "diag": diag, "calib": calib, "hybrid": hybrid}[a.cmd]()
