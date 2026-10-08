"""Label-free quality scores for a saccade labeling, based on properties every saccade has.

Given position windows and a boolean labeling (n, T), `score()` returns numbers in [0, 1] (1 = best) that need NO human label:

  persistence    no flicker: saccade labels are not cut into pieces by gaps < 20 ms, and no isolated sample/run shorter than 4 ms
  main_sequence  among the labeled events, peak speed grows with amplitude and duration grows with amplitude (rank correlations, pooled per source)
  coverage       clearly moving samples (> 8 sigma of the window's own noise and > 10 deg/s) are labeled (a recall proxy) (a labeling that detects nothing cannot get a good main sequence)
  precision      labeled events really contain movement above the noise (peak > 4 sigma and > 15 deg/s)
  stereotypy     the speed profiles, time- and amplitude-normalised, look alike (single-peaked, bell shaped)
  straightness   path length ~ chord length (saccades are nearly straight)
  ILS            the mean of the first five: the Intrinsic Label Score

plus `invariance()` (does the detector give the same answer after rotating / time-reversing / adding noise to the input?) and the injection
benchmark `inject()` / `bench_injection()` that adds synthetic saccades with KNOWN onset / offset to real quiet recordings.
"""
import numpy as np
from scipy.stats import spearmanr
from scipy.ndimage import binary_dilation
from . import detectors as D


def _spear(a, b):
    if len(a) < 8 or np.ptp(a) == 0 or np.ptp(b) == 0: return np.nan
    return float(spearmanr(a, b)[0])


def persistence(lab, fs, gap_ms=20.0, short_ms=4.0):
    """1 - (share of labeled samples that sit next to a short fixation gap (< gap_ms) between two saccade labels, or in a run shorter than short_ms)."""
    g, sh = int(round(gap_ms * fs / 1000.0)), max(int(round(short_ms * fs / 1000.0)), 1)
    tot = bad = 0
    for k in range(lab.shape[0]):
        r = D.runs_1d(lab[k]); tot += sum(e - s + 1 for s, e in r)
        for (s, e) in r:
            if e - s + 1 < sh: bad += e - s + 1
        for (s1, e1), (s2, e2) in zip(r[:-1], r[1:]):
            if s2 - e1 - 1 < g: bad += min(e1 - s1 + 1, e2 - s2 + 1)
    return float(np.clip(1 - bad / max(tot, 1), 0, 1)) if tot > 0 else np.nan


def flips_per_s(lab, fs):
    return float(np.abs(np.diff(lab.astype(int), axis=1)).sum() / (lab.size / fs))


def score(pos, lab, fs, groups=None, min_events=15):
    """dict of scores for one labeling. `groups` (n,) pooled separately per source, then averaged."""
    groups = np.zeros(len(pos), int) if groups is None else groups
    res = {k: [] for k in ("persistence", "main_sequence", "coverage", "precision", "stereotypy", "straightness", "events_per_min")}
    sp, valid = D.speed_of(pos, fs); sg = D.robust_sigma(D.velocity(pos, fs)[0]).mean(2)
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        L, P = lab[i], pos[i]
        res["persistence"].append(persistence(L, fs))
        fast = (sp[i] > 8.0 * sg[i]) & (sp[i] > 10.0) & valid[i]          # clearly moving: far above the window's own noise
        Ld = binary_dilation(L, structure=np.ones((1, 5), bool))
        res["coverage"].append(float((Ld & fast).sum() / max(fast.sum(), 1)) if fast.sum() > 20 else np.nan)
        ev = D.events_of(P, L, fs)
        res["events_per_min"].append(float(sum(e["on"] >= 0 for e in ev) / (L.size / fs / 60)))
        if len(ev) < min_events:
            for k in ("main_sequence", "precision", "stereotypy", "straightness"): res[k].append(np.nan)
            continue
        amp = np.array([e["amp"] for e in ev]); vp = np.array([e["vpeak"] for e in ev]); du = np.array([e["dur"] for e in ev])
        # main sequence: amplitude ~ peak speed, amplitude ~ duration; mapped from rank correlation to [0, 1]
        r1, r2 = _spear(amp, vp), _spear(amp, du)
        res["main_sequence"].append(float(np.clip(np.nanmean([r1, r2]), 0, 1)))
        sig = np.array([sg[i[e["win"]]].mean() for e in ev])
        res["precision"].append(float(np.mean((vp > 4 * sig) & (vp > 15.0))))
        # stereotypy: time-normalised, peak-normalised speed profile vs the median profile
        prof = []
        for e in ev:
            p = e["profile"]
            if len(p) >= 4: prof.append(np.interp(np.linspace(0, 1, 12), np.linspace(0, 1, len(p)), p / max(p.max(), 1e-9)))
        if len(prof) >= 8:
            prof = np.stack(prof); med = np.median(prof, 0)
            cc = [np.corrcoef(p, med)[0, 1] for p in prof if p.std() > 0]
            res["stereotypy"].append(float(np.clip(np.nanmean(cc), 0, 1)))
        else: res["stereotypy"].append(np.nan)
        res["straightness"].append(float(np.mean([e["amp"] / max(e["path"], 1e-9) for e in ev])))
    out = {k: (float(np.nanmean(v)) if len(v) and not np.all(np.isnan(v)) else np.nan) for k, v in res.items()}
    parts = [out[k] for k in ("persistence", "main_sequence", "coverage", "precision", "stereotypy")]
    # parts that cannot be measured (too few events, no fast movement in the data) are skipped; a labeling without any saccade scores 0
    ok = [p for p in parts if not np.isnan(p)]
    out["ILS"] = float(np.mean(ok)) if (ok and lab.any()) else 0.0
    out["flips_per_s"] = flips_per_s(lab, fs)
    out["label_fraction"] = float(lab.mean())
    return out


def main_sequence_table(pos, lab, fs):
    """arrays amp, vpeak, dur of all labeled events (for plotting)"""
    ev = D.events_of(pos, lab, fs)
    return (np.array([e["amp"] for e in ev]), np.array([e["vpeak"] for e in ev]), np.array([e["dur"] for e in ev]))


# ----------------------------------------------------------------------------- invariances (a detector must not care about these)
def _agree(a, b):
    tp, fp, fn, tn = (a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()
    n = tp + fp + fn + tn; po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return float((po - pe) / max(1 - pe, 1e-12)) if pe < 1 else 1.0


def invariance(detect, win, seed=0):
    """detect(Windows) -> bool labels. Cohen's kappa between the labeling of the original windows and of transformed windows, mapped back.
    rotation: rotate and mirror the gaze   |   reverse: play the recording backwards   |   noise: add noise of 30 % of the window's own noise
    |  shift: drop the first 1 sample (time origin)."""
    from .data import Windows
    rng = np.random.RandomState(seed); base = detect(win); out = {}
    th = rng.uniform(0, 2 * np.pi, len(win)); R = np.stack([np.cos(th), -np.sin(th), np.sin(th), np.cos(th)], 1).reshape(-1, 2, 2)
    pos_r = np.einsum("nij,ntj->nti", R, win.pos)
    out["rotation"] = _agree(base, detect(Windows(pos_r.astype(np.float32), win.fs, win.group)))
    out["reverse"] = _agree(base, detect(Windows(win.pos[:, ::-1].copy(), win.fs, win.group))[:, ::-1])
    sd = np.sqrt(np.maximum(np.median(np.diff(np.nan_to_num(win.pos), axis=1) ** 2, axis=(1, 2)), 1e-12))[:, None, None]
    out["noise"] = _agree(base, detect(Windows((win.pos + rng.normal(size=win.pos.shape) * sd * 0.3).astype(np.float32), win.fs, win.group)))
    sh = detect(Windows(win.pos[:, 1:].copy(), win.fs, win.group)); out["shift"] = _agree(base[:, 1:], sh)
    out["invariance"] = float(np.mean(list(out.values())))
    return out


# ----------------------------------------------------------------------------- injection benchmark (known ground truth on real noise)
def min_jerk(amp, dur_ms, fs):
    n = max(int(round(dur_ms * fs / 1000.0)), 3); tau = np.linspace(0, 1, n + 1)[1:]
    return amp * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)


def inject(win, n_per_window=2, seed=0, amp_range=(0.15, 8.0), margin_ms=120, quiet_sigma=8.0):
    """Adds synthetic saccades (main-sequence duration 2.2 ms/deg + 21 ms, minimum-jerk profile, random direction, log-uniform amplitude) to the
    quietest real windows. Returns (Windows with positions, labels (n,T) truth where the speed of the injected movement > 10 % of its peak, amplitudes list)."""
    from .data import Windows
    rng = np.random.RandomState(seed)
    sp, valid = D.speed_of(win.pos, win.fs); sg = D.robust_sigma(D.velocity(win.pos, win.fs)[0]).mean(2)
    quiet = np.where((np.nanmax(np.where(valid, sp, 0), 1) < quiet_sigma * sg[:, 0] + 20) & valid.all(1))[0]
    if len(quiet) < 5: quiet = np.where(valid.all(1))[0]
    pos = win.pos[quiet].astype(np.float64).copy(); lab = np.zeros(pos.shape[:2], bool); T = pos.shape[1]; amps = []
    m = int(margin_ms * win.fs / 1000.0)
    for k in range(len(pos)):
        t = m
        for _ in range(n_per_window):
            A = float(np.exp(rng.uniform(np.log(amp_range[0]), np.log(amp_range[1])))); dur = (2.2 * A + 21.0) * rng.uniform(0.8, 1.25)
            prof = min_jerk(A, dur, win.fs); L = len(prof)
            s = t + int(rng.randint(0, max(m // 2, 1)))
            if s + L + m > T: break
            ang = rng.uniform(0, 2 * np.pi); d = np.array([np.cos(ang), np.sin(ang)])
            add = np.zeros((T, 2)); add[s:s + L] = prof[:, None] * d; add[s + L:] = A * d
            pos[k] += add
            v = np.abs(np.diff(np.r_[0, prof])) * win.fs; lab[k, s:s + L][v > 0.1 * v.max()] = True
            amps.append(A); t = s + L + m
    return Windows(pos.astype(np.float32), win.fs, win.group[quiet]), lab, np.array(amps)


def bench_injection(detect, win, seed=0, **kw):
    """kappa / event recall / false alarms per minute on real quiet windows with injected saccades (truth known)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "online"))
    import metrics as M
    w2, truth, amps = inject(win, seed=seed, **kw)
    pred = detect(w2)
    m = M.evaluate_probs(pred.astype(np.float32), truth.astype(np.float32), w2.fs, thr=0.5, min_event=2, with_ap=False)
    return dict(inj_kappa=m["kappa"], inj_ev_recall=m["ev_recall"], inj_ev_precision=m["ev_precision"], inj_onset_err_ms=m["onset_err_ms"],
                inj_false_alarms_min=m["false_alarms_per_min"])
