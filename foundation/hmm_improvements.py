#!/usr/bin/env python3
"""Three improvements of the universal HMM (foundation/universal_hmm.py, fitted on archive/ only), tested on the benchmarks (test only):
 1. boundary rule per event: onset / offset where the speed crosses a FRACTION of the event's peak speed (f_on, f_off chosen on N labeled
    trials of the target's train split by maximising kappa), instead of a constant shift of 2 integers;
 2. explicit duration law for the saccade state (hidden semi-Markov model): the duration pmf is either estimated WITHOUT labels from the events
    the universal HMM finds in archive/ ("archive") or set from physiology (lognormal, median 25 ms, sd 0.7 in log: "prior");
 3. diagnosis of the Andersson benchmark: where do the false positives of the saccade-vs-rest comparison live (which human class)?
Scorer: foundation/compare.py (event F1 + kappa, 1 kHz). Run from the repository root:  python foundation/hmm_improvements.py
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from foundation import data as FD, compare as CP, universal_hmm as UH
from foundation.hmm_score import ScoreHMM, Scores, subsample
from free_saccade import detectors as D

FRACS = (0.03, 0.06, 0.10, 0.15, 0.20, 0.30)


# ----------------------------------------------------------------------------- 1. boundary rule per event
def refine(pred, speed, f_on, f_off, ext=20):
    out = np.zeros_like(pred)
    for k in range(pred.shape[0]):
        for s, e in D.runs_1d(pred[k]):
            lo, hi = max(s - ext, 0), min(e + ext, pred.shape[1] - 1)
            seg = speed[k, lo:hi + 1]; pk = lo + int(np.argmax(speed[k, s:e + 1])) + (s - lo)
            vp = speed[k, s:e + 1].max()
            below_on = np.nonzero(seg[:pk - lo + 1] < f_on * vp)[0]; below_off = np.nonzero(seg[pk - lo:] < f_off * vp)[0]
            # walking out from the peak: the onset is the first sample after the last one below the fraction before the peak, the offset the last sample before the first one below it after the peak
            # (an audit found the first version took the first / last sample ABOVE the fraction anywhere in the extended window, which noise stretches)
            a = lo + (below_on[-1] + 1 if len(below_on) else 0); b = pk + (below_off[0] - 1 if len(below_off) else len(seg) - 1 - (pk - lo))
            if b >= a: out[k, a:b + 1] = True
    return out


def best_fracs(pred, speed, S):
    best = (-9, 0.1, 0.1)
    for fo in FRACS:
        for ff in FRACS:
            k = CP.event_f1(refine(pred, speed, fo, ff), S)["kappa"]
            if k > best[0]: best = (k, fo, ff)
    return best


# ----------------------------------------------------------------------------- 2. explicit duration law (semi-Markov decoding)
def lognormal_pmf(dmin, dmax, median, sd):
    d = np.arange(dmin, dmax + 1, dtype=np.float64)
    p = np.exp(-0.5 * ((np.log(d) - np.log(median)) / sd) ** 2) / d
    return p / p.sum()


def hsmm_decode(f, valid, h, pmf, dmin, fs):
    """f (n, T) score, h = fitted ScoreHMM (gives the 2 Gaussians and p_on). Saccade = ONE segment whose duration follows pmf (d = dmin ... dmax);
    fixation = geometric, stays with probability 1 - p_on per sample."""
    f = torch.as_tensor(f, dtype=torch.float64); v = torch.as_tensor(valid); n, T = f.shape; dmax = dmin + len(pmf) - 1
    lg = lambda m, s: -0.5 * ((f - m) / s) ** 2 - np.log(s)
    ef, es = lg(h.mu[0], h.sd[0]), lg(h.mu[1], h.sd[1]); ef[~v] = 0.0; es[~v] = 0.0
    C = torch.cat([torch.zeros(n, 1, dtype=torch.float64), torch.cumsum(es, 1)], 1)
    lpmf = torch.as_tensor(np.log(pmf + 1e-12), dtype=torch.float64); lon, lstay = np.log(h.p_on), np.log(1 - h.p_on)
    NEG = -1e18
    F = torch.full((n, T + 1), NEG, dtype=torch.float64); S = torch.full((n, T + 1), NEG, dtype=torch.float64); F[:, 0] = 0.0
    fromS = torch.zeros(n, T + 1, dtype=torch.bool); bd = torch.zeros(n, T + 1, dtype=torch.int32)
    ds = torch.arange(dmin, dmax + 1)
    for t in range(1, T + 1):
        a, b = F[:, t - 1] + lstay, S[:, t - 1]
        fromS[:, t] = b > a; F[:, t] = ef[:, t - 1] + torch.maximum(a, b)
        if t >= dmin:
            dd = ds[ds <= t]; idx = t - dd
            cand = F[:, idx] + lon + lpmf[:len(dd)][None] + (C[:, t:t + 1] - C[:, idx])
            S[:, t], arg = cand.max(1); bd[:, t] = dd[arg].to(torch.int32)
    lab = np.zeros((n, T), bool); Fn, Sn, fS, bdn = F.numpy(), S.numpy(), fromS.numpy(), bd.numpy()
    for k in range(n):
        t = T; inS = Sn[k, T] > Fn[k, T]
        while t > 0:
            if inS:
                d = int(bdn[k, t]); lab[k, t - d:t] = True; t -= d; inS = False
            else:
                inS = bool(fS[k, t]); t -= 1
    return lab


def decode_hsmm(S, h, pmf, dmin, fs):
    f, v = UH.unit_free_score(CP.native_pos(S), "detrended", fs)
    lab = hsmm_decode(f, v, h, pmf, dmin, fs)
    lab &= ~D.invalid_zone(v, fs)
    return CP.to_1khz(D.cleanup(lab, fs, min_ms=1000.0 * dmin / fs, merge_ms=0), S)


def main():
    rows = []; uni = {}; pmfs = {}
    for fs in (1000.0, 500.0):
        P = UH.archive_windows(fs); f, v = UH.unit_free_score(P, "detrended", fs); h = ScoreHMM(fs).fit(Scores(f, v, fs)); uni[fs] = h
        lab = h.predict(Scores(f, v, fs)); dur = np.array([e - s + 1 for k in range(len(lab)) for s, e in D.runs_1d(lab[k])], float)
        dmin = h.d; dmax = int(round(0.15 * fs)); edges = np.arange(dmin, dmax + 2) - 0.5
        hist = np.histogram(np.clip(dur, dmin, dmax), edges)[0].astype(float) + 0.5
        k = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2); hist = np.convolve(hist, k / k.sum(), "same")
        pmfs[fs] = dict(archive=hist / hist.sum(), prior=lognormal_pmf(dmin, dmax, 0.025 * fs, 0.7), dmin=dmin)
        print(f"{fs:.0f} Hz: the universal HMM finds {len(dur)} events in archive/, median duration {np.median(dur) * 1000 / fs:.0f} ms (5-95 %: {np.percentile(dur, 5) * 1000 / fs:.0f}-{np.percentile(dur, 95) * 1000 / fs:.0f} ms)", flush=True)
    for nm in FD.ALL:
        fs = CP.NATIVE[nm]; h = uni[fs]; A, B = subsample(FD.load(nm, "train"), 300 if nm != "andersson" else 80), FD.load(nm, "test")
        res = {}
        fB, vB = UH.unit_free_score(CP.native_pos(B), "detrended", fs); predB = CP.to_1khz(h.predict(Scores(fB, vB, fs)), B)
        res["universal HMM, no label"] = CP.event_f1(predB, B)
        # 1. fractions of the peak speed chosen on N labeled trials
        fA, vA = UH.unit_free_score(CP.native_pos(A), "detrended", fs); predA = CP.to_1khz(h.predict(Scores(fA, vA, fs)), A)
        spB = np.hypot(*np.moveaxis(D.velocity(B.pos.astype(np.float64), 1000.0)[0], 2, 0)); spA = np.hypot(*np.moveaxis(D.velocity(A.pos.astype(np.float64), 1000.0)[0], 2, 0))
        for N in (10,):
            ks = []
            for seed in (7, 8, 9):
                sel = np.random.RandomState(seed).permutation(len(A))[:N]
                _, fo, ff = best_fracs(predA[sel], spA[sel], FD.Set(nm, A.pos[sel], A.lab[sel], A.coarse))
                r = CP.event_f1(refine(predB, spB, fo, ff), B); r.update(f_on=fo, f_off=ff); ks.append(r)
            m = {q: float(np.mean([r[q] for r in ks])) for q in ("ev_f1", "kappa")}; m["fractions"] = [(r["f_on"], r["f_off"]) for r in ks]
            res[f"+ boundary at a fraction of the peak speed, from {N} labeled trials (mean of 3)"] = m
        # 2. duration law
        for name in ("archive", "prior"):
            res[f"semi-Markov, saccade duration law '{name}', no label"] = CP.event_f1(decode_hsmm(B, h, pmfs[fs][name], pmfs[fs]["dmin"], fs), B)
        for k_, r in res.items():
            rows.append(dict(dataset=nm, method=k_, **r)); print(f"[{nm:9s}] {k_:85s} F1 {r['ev_f1']:.3f}  kappa {r['kappa']:.3f}" + (f"  fractions {r['fractions']}" if "fractions" in r else ""), flush=True)
    # 3. Andersson: where are the false positives of "saccade vs rest"?
    B = FD.load("andersson", "test"); fs = 500.0
    pred = CP.to_1khz(uni[fs].predict(Scores(*UH.unit_free_score(CP.native_pos(B), "detrended", fs), fs)), B)
    names = {-1: "no label (padding)", 0: "fixation", 1: "SACCADE", 2: "PSO", 3: "pursuit", 4: "blink"}
    print("Andersson: composition of the samples the universal HMM calls saccade, and recall by class:", flush=True)
    tot = pred.sum()
    for c, n_ in names.items():
        m = B.lab == c
        print(f"   true class {n_:22s}: {100 * (pred & m).sum() / max(tot, 1):5.1f} % of predicted-saccade samples | {100 * (pred & m).sum() / max(m.sum(), 1):5.1f} % of this class is called saccade | class share {100 * m.mean():.1f} %", flush=True)
    for drop, label in (((2,), "PSO ignored"), ((4,), "blinks ignored"), ((2, 4), "PSO and blinks ignored"), ((-1, 2, 4), "padding, PSO and blinks ignored")):
        keep = ~np.isin(B.lab, drop); S2 = FD.Set("andersson", B.pos, np.where(keep, B.lab, -1), False)
        r = CP.event_f1(pred, S2); print(f"   scoring with {label:34s}: F1 {r['ev_f1']:.3f} kappa {r['kappa']:.3f}", flush=True); rows.append(dict(dataset="andersson", method="diagnosis: " + label, **r))
    out = os.path.join(ROOT, "foundation", "runs", "hmm_improvements.json"); json.dump(rows, open(out, "w"), indent=1, default=float); print("wrote", out)


if __name__ == "__main__":
    main()
