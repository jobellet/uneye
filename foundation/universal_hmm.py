#!/usr/bin/env python3
"""Strict protocol: a detector fitted ONLY on archive/ (unlabeled recordings of other datasets), applied unchanged to the benchmarks of U'n'Eye
(datasets 1-4, Andersson), which are used for testing only (no fit, no label, not even their unlabeled train split).

1. Universal HMM (one per sampling rate: 1000 and 500 Hz, the native rate of each benchmark): the minimum-duration HMM of free_saccade/detectors.py on a unit-free
   score, log10(1 + detrended speed / robust noise of the window) (detrended = velocity minus its 100 ms running median, chosen beforehand for all),
   Viterbi-trained on 2000 windows of archive/ (EMTeC, GazeBase, GazeCom, Lund2013 at 1 kHz). Compared with the same HMM fitted on the
   unlabeled train split of each benchmark (what the previous experiments did).
2. How much of Cohen's kappa is the expert's convention for the boundaries? Detected events are widened / narrowed by `pre` samples at the
   onset and `post` samples at the offset. (pre, post) is chosen on N human-labeled TRIALS of the target's train split (N = 1, 3, 10, 30),
   by maximising kappa on them, then applied to the test split; also the best (pre, post) chosen on the test split itself (a ceiling for
   a pure boundary shift). Event F1 barely depends on it: this is exactly the owner's argument for "same F1, different kappa".
Scorer: foundation/compare.py (event F1 + kappa, saccade vs rest, 1 kHz). Run from the repository root:  python foundation/universal_hmm.py
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from foundation import data as FD, compare as CP
from foundation.hmm_score import ScoreHMM, Scores, subsample
from free_saccade import data as FS, detectors as D, cebra_seed as C

FS_HZ = 1000.0


def unit_free_score(pos, kind="detrended", fs=FS_HZ):
    p = np.asarray(pos, np.float64)
    v, valid = (C.detrended_velocity(p, fs) if kind == "detrended" else D.velocity(p, fs))
    sg = np.maximum(D.robust_sigma(v).mean(2), 1e-9)                       # (n, 1): robust noise of each window, in the units of the data
    f = np.log10(1.0 + np.hypot(v[..., 0], v[..., 1]) / sg)
    return f.astype(np.float32), np.isfinite(p).all(2)


def archive_windows(fs, n_per=500, seed=0):
    T = int(round(0.32 * fs))                                              # 320 ms
    src = FS.H5Source(os.path.join(ROOT, "archive", "datasets_by_subject", "datasets_by_subject"), os.path.join(ROOT, "archive", "subject_h5_metadata.csv"),
                      "dataset", 1.0, FS.ARCHIVE_FS, limit=None, clip=FS.ARCHIVE_CLIP)
    return src.sample(n_per, T, np.random.RandomState(seed), fs=fs, groups=("EMTeC", "GazeBase", "GazeCom", "Lund2013")).pos


def widen(lab, pre, post):
    """every run [s, e] -> [s - pre, e + post] (negative values shrink it; a run that vanishes is dropped)"""
    out = np.zeros_like(lab)
    for k in range(lab.shape[0]):
        for s, e in D.runs_1d(lab[k]):
            s2, e2 = max(s - pre, 0), min(e + post, lab.shape[1] - 1)
            if e2 >= s2: out[k, s2:e2 + 1] = True
    return out


GRID = [(a, b) for a in range(-6, 13, 2) for b in range(-6, 13, 2)]


def best_shift(pred, S):
    best = (-9, 0, 0)
    for a, b in GRID:
        k = CP.event_f1(widen(pred, a, b), S)["kappa"]
        if k > best[0]: best = (k, a, b)
    return best


def main():
    rows = []; uni = {}
    for fs in (1000.0, 500.0):                                             # one universal HMM per sampling rate, both fitted on archive/ only
        f, v = unit_free_score(archive_windows(fs), "detrended", fs); h = ScoreHMM(fs).fit(Scores(f, v, fs)); uni[fs] = h
        print(f"universal HMM at {fs:.0f} Hz (archive/ only, {len(f)} windows): fixation mean {h.mu[0]:.2f} sd {h.sd[0]:.2f} | saccade mean {h.mu[1]:.2f} sd {h.sd[1]:.2f} | "
              f"p_on {h.p_on:.4f} p_stay {h.p_stay:.2f} | minimum duration {h.d} samples", flush=True)
    def predict(S, h, fs):
        f, v = unit_free_score(CP.native_pos(S), "detrended", fs); return CP.to_1khz(h.predict(Scores(f, v, fs)), S)
    for nm in FD.ALL:
        fs = CP.NATIVE[nm]; A, B = subsample(FD.load(nm, "train"), 300 if nm != "andersson" else 80), FD.load(nm, "test")
        res = {}
        predB = predict(B, uni[fs], fs); res["universal HMM (archive/ only), no label"] = CP.event_f1(predB, B)
        fA, vA = unit_free_score(CP.native_pos(A), "detrended", fs); hT = ScoreHMM(fs).fit(Scores(fA, vA, fs))
        res["same HMM fitted on the unlabeled train split of the target"] = CP.event_f1(predict(B, hT, fs), B)
        predA = predict(A, uni[fs], fs)
        for N in (3, 10, 30):
            shifts = []
            for seed in (7, 8, 9):                                         # 3 different draws of the N labeled trials
                sel = np.random.RandomState(seed).permutation(len(A))[:N]
                subA = FD.Set(nm, A.pos[sel], A.lab[sel], A.coarse); _, a, b = best_shift(predA[sel], subA)
                r = CP.event_f1(widen(predB, a, b), B); r.update(pre=a, post=b); shifts.append(r)
            m = {k: float(np.mean([r[k] for r in shifts])) for k in ("ev_f1", "kappa")}; m["kappa_min"] = min(r["kappa"] for r in shifts); m["shifts"] = [(r["pre"], r["post"]) for r in shifts]
            res[f"universal HMM + boundary shift from {N} labeled trials (mean of 3 draws)"] = m
        _, a, b = best_shift(predB, B); r = CP.event_f1(widen(predB, a, b), B); r.update(pre=a, post=b); res["universal HMM + best boundary shift (chosen on the test split: ceiling)"] = r
        for k_, r in res.items():
            rows.append(dict(dataset=nm, method=k_, **r))
            extra = f"  (kappa min over draws {r['kappa_min']:.3f}; shifts {r['shifts']})" if "kappa_min" in r else (f"  (pre {r['pre']}, post {r['post']})" if "pre" in r else "")
            print(f"[{nm:9s}] {k_:80s} F1 {r['ev_f1']:.3f}  kappa {r['kappa']:.3f}{extra}", flush=True)
    out = os.path.join(ROOT, "foundation", "runs", "universal_hmm.json"); json.dump(rows, open(out, "w"), indent=1, default=float); print("wrote", out)


if __name__ == "__main__":
    main()
