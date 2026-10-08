"""Free (noisy) labels from a classic velocity-threshold detector (Engbert & Kliegl 2003), vectorised.
Used only to PRETRAIN a network on unlabeled recordings; the few human labels are then used to fine-tune."""
import numpy as np


def engbert_kliegl(X, Y, fs, lam=6.0, min_samples=3):
    X, Y = np.nan_to_num(X), np.nan_to_num(Y)
    def vel(a):
        v = np.zeros_like(a)
        v[:, 2:-2] = (a[:, 4:] + a[:, 3:-1] - a[:, 1:-3] - a[:, :-4]) * fs / 6.0
        return v
    vx, vy = vel(X), vel(Y)
    def sd(v):  # median-based standard deviation per trial
        return np.sqrt(np.maximum(np.median(v ** 2, 1) - np.median(v, 1) ** 2, 1e-12))[:, None]
    ex, ey = lam * sd(vx), lam * sd(vy)
    lab = ((vx / ex) ** 2 + (vy / ey) ** 2) > 1
    out = np.zeros_like(lab)
    for k in range(lab.shape[0]):
        d = np.diff(np.r_[0, lab[k].astype(int), 0])
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s >= min_samples:
                out[k, max(s - 1, 0):e + 1] = True
    return out.astype(np.float32)
