"""Classic, label-free saccade detectors + helpers (kinematics, label clean-up, event extraction).

All functions take position windows `pos` of shape (n, T, 2) in degrees with NaN = invalid, and the sampling rate `fs`, and return boolean
saccade labels (n, T). They are the baselines and the teachers of the self-training experiments:
  ek            Engbert & Kliegl 2003 (velocity threshold in robust standard deviations of the window)
  ivt_adaptive  Nystrom & Holmqvist 2010 (iterated threshold on the speed distribution of a whole recording / source)
  HMM           2-state hidden Markov model on log speed with a minimum-duration chain (Viterbi training, no labels)
  consensus     samples where the detectors agree (confident), everything else is ignored
"""
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion


# ----------------------------------------------------------------------------- kinematics
def fill(pos):
    """linear interpolation over NaN (only to compute velocities; invalid samples stay masked). Returns filled pos, valid mask (n, T)"""
    pos = np.asarray(pos, np.float64)
    valid = np.isfinite(pos).all(2)
    out = pos.copy()
    x = np.arange(pos.shape[1])
    for k in np.where(~valid.all(1))[0]:
        g = valid[k]
        for a in range(2):
            out[k, :, a] = np.interp(x, x[g], pos[k, g, a]) if g.any() else 0.0
    return out, valid


def velocity(pos, fs):
    """5-point moving-difference velocity (Engbert & Kliegl), deg/s, (n, T, 2); first / last 2 samples are 0"""
    p, valid = fill(pos)
    v = np.zeros_like(p)
    v[:, 2:-2] = (p[:, 4:] + p[:, 3:-1] - p[:, 1:-3] - p[:, :-4]) * fs / 6.0
    return v, valid


def speed_of(pos, fs):
    v, valid = velocity(pos, fs)
    return np.hypot(v[..., 0], v[..., 1]), valid


def robust_sigma(v):
    """Engbert-Kliegl median-based standard deviation per window and component: sqrt(median(v^2) - median(v)^2)"""
    return np.sqrt(np.maximum(np.median(v ** 2, 1) - np.median(v, 1) ** 2, 1e-12))[:, None, :]


def invalid_zone(valid, fs, margin_ms=20):
    """samples near invalid data (blink margins): nothing is labeled there"""
    m = int(round(margin_ms * fs / 1000.0))
    return binary_dilation(~valid, structure=np.ones((1, 2 * m + 1), bool)) if m > 0 else ~valid


# ----------------------------------------------------------------------------- label utilities
def runs_1d(b):
    d = np.diff(np.r_[0, np.asarray(b).astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))   # inclusive ends


def cleanup(lab, fs, min_ms=6.0, merge_ms=6.0):
    """close gaps shorter than merge_ms between two saccade labels, then drop runs shorter than min_ms"""
    lab = np.asarray(lab, bool).copy()
    g, m = max(int(round(merge_ms * fs / 1000.0)), 0), max(int(round(min_ms * fs / 1000.0)), 1)
    for k in range(lab.shape[0]):
        if lab[k].any() and g > 0:
            r = runs_1d(~lab[k])
            for s, e in r:
                if s > 0 and e < lab.shape[1] - 1 and e - s + 1 <= g: lab[k, s:e + 1] = True
        for s, e in runs_1d(lab[k]):
            if e - s + 1 < m: lab[k, s:e + 1] = False
    return lab


# ----------------------------------------------------------------------------- Engbert & Kliegl
def ek(pos, fs, lam=6.0, min_ms=6.0):
    v, valid = velocity(pos, fs)
    sg = robust_sigma(v)
    lab = ((v / (lam * sg)) ** 2).sum(2) > 1
    lab &= ~invalid_zone(valid, fs)
    return cleanup(lab, fs, min_ms=min_ms, merge_ms=0)


# ----------------------------------------------------------------------------- adaptive velocity threshold (Nystrom & Holmqvist)
def ivt_adaptive(win, k_peak=6.0, k_edge=3.0, min_ms=6.0, pooled=True):
    """peak threshold PT = mean + k_peak * sd of the speeds below PT, iterated to convergence (per source if pooled, else per window);
    a saccade = run of speeds above PT, extended backward / forward while speed > mean + k_edge * sd of the noise."""
    sp, valid = speed_of(win.pos, win.fs)
    sp = np.where(valid, sp, np.nan)
    lab = np.zeros(sp.shape, bool)
    groups = np.unique(win.group) if pooled else None
    sets = [np.where(win.group == g)[0] for g in groups] if pooled else [[i] for i in range(len(win))]
    for idx in sets:
        s = sp[idx]; s = s[np.isfinite(s)]
        if s.size < 20: continue
        pt = 200.0
        for _ in range(100):
            b = s[s < pt]
            if b.size < 10: break
            new = b.mean() + k_peak * b.std()
            if abs(new - pt) < 1.0: pt = new; break
            pt = new
        b = s[s < pt]; edge = b.mean() + k_edge * b.std()
        for i in idx:
            x = np.nan_to_num(sp[i], nan=0.0)
            for a, e in runs_1d(x > pt):
                a2, e2 = a, e
                while a2 > 0 and x[a2 - 1] > edge: a2 -= 1
                while e2 < len(x) - 1 and x[e2 + 1] > edge: e2 += 1
                lab[i, a2:e2 + 1] = True
    lab &= ~invalid_zone(valid, win.fs)
    return cleanup(lab, win.fs, min_ms=min_ms, merge_ms=0)


# ----------------------------------------------------------------------------- HMM with minimum-duration chain
class HMM:
    """Fixation / saccade HMM on log10(speed). The saccade state is a chain of `d` states (so that every saccade lasts at least d samples).
    Trained without labels by Viterbi training (initialised from the speed distribution: lowest 90 % fixation, top 2 % saccade)."""

    def __init__(self, fs, min_ms=6.0):
        self.d = max(int(round(min_ms * fs / 1000.0)), 1); self.fs = fs

    @staticmethod
    def feat(win):
        sp, valid = speed_of(win.pos, win.fs)
        return np.log10(sp + 1.0).astype(np.float32), valid

    def _logA(self):
        K = self.d + 1
        A = torch.full((K, K), -1e9)
        A[0, 0] = np.log(1 - self.p_on); A[0, 1] = np.log(self.p_on)
        for i in range(1, self.d): A[i, i + 1] = 0.0
        A[self.d, self.d] = np.log(self.p_stay); A[self.d, 0] = np.log(1 - self.p_stay)
        return A

    def _emit(self, f, valid):
        f = torch.as_tensor(f); n, T = f.shape; K = self.d + 1
        ll = torch.empty(n, T, K)
        ll[..., 0] = -0.5 * ((f - self.mu[0]) / self.sd[0]) ** 2 - np.log(self.sd[0])
        ll[..., 1:] = (-0.5 * ((f - self.mu[1]) / self.sd[1]) ** 2 - np.log(self.sd[1]))[..., None]
        ll[~torch.as_tensor(valid)] = 0.0                                     # invalid sample: uninformative
        return ll

    def viterbi(self, f, valid):
        ll = self._emit(f, valid); A = self._logA(); n, T, K = ll.shape
        delta = torch.full((n, K), -1e9); delta[:, 0] = ll[:, 0, 0]
        back = torch.zeros(n, T, K, dtype=torch.int8)
        for t in range(1, T):
            sc = delta[:, :, None] + A[None]
            best, arg = sc.max(1)
            delta = best + ll[:, t]; back[:, t] = arg.to(torch.int8)
        st = torch.zeros(n, T, dtype=torch.long); st[:, -1] = delta.argmax(1)
        for t in range(T - 1, 0, -1):
            st[:, t - 1] = back[torch.arange(n), t, st[:, t]].long()
        return (st > 0).numpy()

    def fit(self, win, iters=6):
        f, valid = self.feat(win); fv = f[valid]
        lo, hi = np.percentile(fv, 90), np.percentile(fv, 98)
        self.mu = [float(fv[fv < lo].mean()), float(fv[fv > hi].mean())]
        self.sd = [float(fv[fv < lo].std() + 1e-3), float(fv[fv > hi].std() + 1e-3)]
        self.p_on, self.p_stay = 0.01, 0.8
        for _ in range(iters):
            lab = self.viterbi(f, valid)
            m0, m1 = valid & ~lab, valid & lab
            if m1.sum() < 10 or m0.sum() < 10: break
            self.mu = [float(f[m0].mean()), float(f[m1].mean())]; self.sd = [float(f[m0].std() + 1e-3), float(f[m1].std() + 1e-3)]
            n_runs = sum(len(runs_1d(l)) for l in lab)
            self.p_on = float(np.clip(n_runs / max(m0.sum(), 1), 1e-4, 0.2))
            self.p_stay = float(np.clip(1 - n_runs / max(m1.sum() - (self.d - 1) * n_runs, n_runs), 0.3, 0.99))     # transitions out of the last state = samples in it = run length - (d - 1), one of which leaves
        return self

    def predict(self, win):
        f, valid = self.feat(win)
        lab = self.viterbi(f, valid) & ~invalid_zone(valid, win.fs)
        return cleanup(lab, win.fs, min_ms=1000.0 * self.d / win.fs, merge_ms=0)


def hmm_detect(win, fit_on=None):
    return HMM(win.fs).fit(fit_on or win).predict(win)


# ----------------------------------------------------------------------------- consensus pseudo-labels
def consensus(win, votes=None):
    """votes: list of boolean label arrays. Returns (label, weight): saccade where all agree, fixation where none says saccade and the speed is
    small, weight 0 (ignored) everywhere else, near invalid data and next to saccades (the edges are the uncertain part)."""
    votes = votes or [ek(win.pos, win.fs), ivt_adaptive(win), hmm_detect(win)]
    S = np.stack(votes).astype(int).sum(0)
    sp, valid = speed_of(win.pos, win.fs); sg = robust_sigma(velocity(win.pos, win.fs)[0]).mean(2)
    lab = S == len(votes)
    quiet = (S == 0) & (sp < 3 * sg) & ~binary_dilation(S > 0, structure=np.ones((1, 7), bool))
    w = (lab | quiet) & valid & ~invalid_zone(valid, win.fs)
    return lab, w.astype(np.float32)


# ----------------------------------------------------------------------------- events
def events_of(pos, lab, fs, min_ms=0.0):
    """list of dicts (window, onset, offset, dur_ms, amp, vpeak, path, profile) for runs that do not touch the window border or invalid data."""
    p, valid = fill(pos); sp, _ = speed_of(pos, fs)
    out = []
    for k in range(lab.shape[0]):
        for s, e in runs_1d(lab[k]):
            if s < 2 or e > lab.shape[1] - 3 or not valid[k, s - 2:e + 3].all(): continue
            dur = (e - s + 1) * 1000.0 / fs
            if dur < min_ms: continue
            a, b = p[k, s - 1], p[k, e + 1]
            seg = p[k, s - 1:e + 2]
            out.append(dict(win=k, on=s, off=e, dur=dur, amp=float(np.hypot(*(b - a))), vpeak=float(sp[k, s:e + 1].max()),
                            path=float(np.hypot(*np.diff(seg, axis=0).T).sum()), profile=sp[k, s:e + 1].astype(np.float32)))
    return out
