"""Free (noisy) labels from a classic velocity-threshold detector (Engbert & Kliegl 2003), vectorised.
Used only to PRETRAIN a network on unlabeled recordings; the few human labels are then used to fine-tune."""
import numpy as np
from scipy.ndimage import binary_dilation


def _clean(a, limit=1e3):
    """interpolate over non-finite / absurd samples (blinks); returns the cleaned array and the mask of bad samples"""
    bad = ~np.isfinite(a) | (np.abs(np.nan_to_num(a, nan=0.0, posinf=1e30, neginf=-1e30)) > limit)
    out = np.where(bad, 0.0, a)
    x = np.arange(a.shape[1])
    for k in np.where(bad.any(1))[0]:
        good = ~bad[k]
        out[k] = np.interp(x, x[good], a[k, good]) if good.any() else 0.0
    return out, bad


def engbert_kliegl(X, Y, fs, lam=6.0, min_samples=3):
    X, bx = _clean(np.asarray(X, float)); Y, by = _clean(np.asarray(Y, float))
    bad = binary_dilation(bx | by, structure=np.ones((1, 11), bool))   # no labels around blinks / dropouts

    def vel(a):
        v = np.zeros_like(a)
        v[:, 2:-2] = (a[:, 4:] + a[:, 3:-1] - a[:, 1:-3] - a[:, :-4]) * fs / 6.0
        return v
    vx, vy = vel(X), vel(Y)

    def sd(v):  # median-based standard deviation per trial
        return np.sqrt(np.maximum(np.median(v ** 2, 1) - np.median(v, 1) ** 2, 1e-12))[:, None]
    lab = ((vx / (lam * sd(vx))) ** 2 + (vy / (lam * sd(vy))) ** 2) > 1
    lab &= ~bad
    out = np.zeros_like(lab)
    for k in range(lab.shape[0]):
        d = np.diff(np.r_[0, lab[k].astype(int), 0])
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s >= min_samples:
                out[k, max(s - 1, 0):e + 1] = True
    return (out & ~bad).astype(np.float32)
