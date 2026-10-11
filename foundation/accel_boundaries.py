#!/usr/bin/env python3
"""HMM + acceleration-zero boundaries (owner's idea). Stage 1: the label-free universal minimum-duration HMM (fitted on archive/ only, foundation/universal_hmm.py) finds the events.
Stage 2 (no label): for every event the trace is SMOOTHED BEFORE differentiating (Savitzky-Golay, order 3, window `win_ms`, derivatives taken analytically from the smoothed polynomial), the tangential acceleration
a_t = d|v|/dt = (vx ax + vy ay) / |v| is computed, and the boundaries are the zeros of a_t around the peak speed:
  onset  = going back from the peak speed inside the event, the first zero of a_t (speed stops falling: the speed minimum before the peak), searched down to `margin_ms` before the HMM onset;
  offset 'first' (PSO NOT part of the saccade, Andersson's convention): going forward from the peak, the first zero of a_t (the first inflection: the speed stops falling), up to `margin_ms` after the HMM offset;
  offset 'last' (PSO part of the saccade, the convention of datasets 1-3): the last sample in [peak, HMM offset + margin] whose speed is above a noise floor (3 robust sigma of the velocity), then the next zero of a_t
         (so that the several zero crossings of the acceleration during post-saccadic oscillations are all included, while noise zero crossings after the oscillation are not).
Variants scored on the common test subsets (night_eval.data()) with compare.event_f1 and compared with the HMM alone: onset only, 'first', 'last', and the a-priori mix (Andersson 'first', the others 'last').
Also reports the onset / offset error (ms) of matched events before and after.   python foundation/accel_boundaries.py [--win-ms 15]  -> night/accel_boundaries.json
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from scipy.signal import savgol_filter
from free_saccade import detectors as D
from foundation import compare as CP, night_eval as NE, universal_hmm as UH, conventions as CV
from foundation.hmm_score import ScoreHMM, Scores


def kinematics(p, fs, win_ms, valid=None):
    """p (T, 2) filled positions -> speed (T,), tangential acceleration d|v|/dt (T,), robust sigma of the velocity (scalar, from the VALID samples only)"""
    w = max(int(round(win_ms * fs / 1000.0)) | 1, 5); w = max(min(w, (len(p) // 2) * 2 - 1), 5); dt = 1.0 / fs
    v = np.stack([savgol_filter(p[:, a], w, 3, deriv=1, delta=dt, mode="interp") for a in range(2)], 1); ac = np.stack([savgol_filter(p[:, a], w, 3, deriv=2, delta=dt, mode="interp") for a in range(2)], 1)
    sp = np.hypot(v[:, 0], v[:, 1]); at = (v * ac).sum(1) / np.maximum(sp, 1e-9); vv = v if valid is None or valid.sum() < 20 else v[valid]; sg = float(np.mean(D.robust_sigma(vv[None])))
    return sp, at, sg


def refine(pred, pos, fs, mode, win_ms=15.0, margin_ms=25.0, offset_rule="last"):
    """pred (n, T) bool at the native rate -> refined bool mask. mode: 'onset' (offset kept), 'both'. offset_rule: 'first' | 'last' (or a per-trial callable is not needed: the caller passes one rule per dataset)"""
    n, T = pred.shape; out = np.zeros_like(pred); m = max(int(round(margin_ms * fs / 1000.0)), 2)
    for i in range(n):
        ev = D.runs_1d(pred[i])
        if not ev: continue
        p, valid = D.fill(pos[i:i + 1]); sp, at, sg = kinematics(p[0], fs, win_ms, valid[0]); floor = 3.0 * sg
        for q, (s, e) in enumerate(ev):
            lo = max(s - m, ev[q - 1][1] + 1 if q > 0 else 0); hi = min(e + m, ev[q + 1][0] - 1 if q + 1 < len(ev) else T - 1)      # the search windows of two neighbouring events never overlap
            pk = s + int(np.argmax(sp[s:e + 1]))
            j = pk - 1
            while j >= lo and at[j] > 0: j -= 1                                           # still accelerating: keep going back to the speed minimum
            on = min(j + 1, pk)
            if mode == "onset": off = e
            elif offset_rule == "first":
                j = pk + 1
                while j <= hi and at[j] < 0: j += 1                                       # still decelerating: the first zero of a_t is the end of the main saccade
                off = j - 1 if j > pk + 1 else e                                           # no deceleration phase found: keep the HMM offset
            else:
                idx = np.nonzero(sp[pk:hi + 1] >= floor)[0]
                if len(idx) == 0: off = e                                                  # nothing above the noise floor after the peak: keep the HMM offset
                else:
                    j = pk + int(idx[-1])
                    while j <= hi and at[j] < 0: j += 1
                    off = min(j, hi)
            if off >= on: out[i, on:off + 1] = True
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--win-ms", type=float, default=15.0); a = ap.parse_args(); res = {}; uni = {}
    from foundation import hmm_score as HS
    for fs in (1000.0, 500.0):
        f, v = UH.unit_free_score(UH.archive_windows(fs), "detrended", fs); uni[fs] = ScoreHMM(fs).fit(Scores(f, v, fs))
    for k, (A, B) in NE.data().items():
        fs = CP.NATIVE[k]; pos = CP.native_pos(B).astype(np.float64); f, v = UH.unit_free_score(CP.native_pos(B), "detrended", fs); hmm = uni[fs].predict(Scores(f, v, fs)); res[k] = {}
        rule = "first" if k == "andersson" else "last"
        variants = {"hmm": hmm, "onset only": refine(hmm, pos, fs, "onset", a.win_ms), "offset first (no PSO)": refine(hmm, pos, fs, "both", a.win_ms, offset_rule="first"),
                    "offset last (PSO included)": refine(hmm, pos, fs, "both", a.win_ms, offset_rule="last")}
        variants["a-priori mix"] = variants["offset first (no PSO)"] if rule == "first" else variants["offset last (PSO included)"]
        truth = (B.lab == 1)
        for name, pr in variants.items():
            m1 = CP.to_1khz(pr, B); r = CP.event_f1(m1, B); on, off = [], []
            for i in range(len(m1)):
                _, _, _, o = CV.iou_match(D.runs_1d(m1[i] & (B.lab[i] >= 0)), D.runs_1d(truth[i]), 0.1); on += [x[0] for x in o]; off += [x[1] for x in o]
            stat = lambda x: [float(np.mean(x)), float(np.std(x))] if len(x) else [float('nan'), float('nan')]
            res[k][name] = dict(f1=float(r["ev_f1"]), kappa=float(r["kappa"]), onset_err_ms=stat(on), offset_err_ms=stat(off))
            print(f"{k:10s} {name:28s} F1 {r['ev_f1']:.3f} kappa {r['kappa']:.3f} | onset err {stat(on)[0]:+.1f}+-{stat(on)[1]:.1f} ms | offset err {stat(off)[0]:+.1f}+-{stat(off)[1]:.1f} ms", flush=True)
    json.dump(res, open(os.path.join(NE.NIGHT, f"accel_boundaries_{int(a.win_ms)}ms.json"), "w"), indent=1)
    # the rule chosen per dataset on N labeled trials of the TRAIN subset (labels only used for that choice), applied to the test subset
    sel = {}
    for k, (A, B) in NE.data().items():
        fs = CP.NATIVE[k]; posA = CP.native_pos(A).astype(np.float64); f, v = UH.unit_free_score(CP.native_pos(A), "detrended", fs); hA = uni[fs].predict(Scores(f, v, fs)); posB = CP.native_pos(B).astype(np.float64); f, v = UH.unit_free_score(CP.native_pos(B), "detrended", fs); hB = uni[fs].predict(Scores(f, v, fs))
        rules = {"hmm": lambda h, p_: h, "onset only": lambda h, p_: refine(h, p_, fs, "onset", a.win_ms), "offset first": lambda h, p_: refine(h, p_, fs, "both", a.win_ms, offset_rule="first"), "offset last": lambda h, p_: refine(h, p_, fs, "both", a.win_ms, offset_rule="last")}
        outA = {n: CP.to_1khz(fn(hA, posA), A) for n, fn in rules.items()}; outB = {n: CP.event_f1(CP.to_1khz(fn(hB, posB), B), B) for n, fn in rules.items()}; sel[k] = {}
        for N in (3, 10):
            vals = []
            for r in range(5):
                idx = np.random.RandomState(100 * N + r).permutation(len(A))[:N]; Asub = type("S", (), dict(lab=A.lab[idx]))()
                sc = {n: (lambda q: 0.5 * (q["ev_f1"] + (0.0 if np.isnan(q["kappa"]) else q["kappa"])))(CP.event_f1(outA[n][idx], Asub)) for n in rules}; best = max(sc, key=sc.get)
                vals.append((outB[best]["ev_f1"], outB[best]["kappa"], best))
            sel[k][N] = dict(f1=float(np.mean([x[0] for x in vals])), kappa=float(np.mean([x[1] for x in vals])), choices=[x[2] for x in vals])
            print(f"{k:10s} rule chosen on N={N} labeled trials: F1 {sel[k][N]['f1']:.3f} kappa {sel[k][N]['kappa']:.3f} choices {sel[k][N]['choices']}", flush=True)
    json.dump(sel, open(os.path.join(NE.NIGHT, f"accel_boundaries_select_{int(a.win_ms)}ms.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
