"""Unit check: the detectors must give the same labels whether the gaze is in degrees, pixels or z-scores.

The labeled repository sets (degrees) are multiplied by a factor (37 ~ pixels per degree, 0.01 ~ z-scored), then labeled with and without
`normalize_units`. Prints, per detector and factor, the agreement with the labels obtained on the original degrees (Cohen's kappa) and
the kappa against the human labels (pooled over u1-u4; u2 is pursuit data where EK labels about half of the samples, so the pooled
value is lower than the per-set values of RESULTS.md). Run from the repository root: python free_saccade/check_units.py
Result 2026-10-09 (M1): with normalize_units, kappa vs degrees = 1.000 for EK, I-VT and HMM at x37 and x0.01; without it, I-VT drops to
0.74 and HMM to 0.43 at x0.01.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import cohen_kappa_score
from free_saccade import data as FD, detectors as D
from free_saccade.data import Windows

FACTORS = (1.0, 37.0, 0.01)


def detectors(train):
    hmm = D.HMM(train.fs).fit(train)
    return {"EK lam=6": lambda w: D.ek(w.pos, w.fs, lam=6.0), "I-VT adaptive": D.ivt_adaptive, "HMM": hmm.predict}


def main():
    lab = FD.load_labeled("data", n=60)
    pos = np.concatenate([w.pos[:, :375] for w in lab.values()])          # shortest set: 375 samples at 500 Hz
    hum = np.concatenate([w.labels[:, :375] for w in lab.values()])
    grp = np.concatenate([[k] * len(w) for k, w in lab.items()])
    base = Windows(pos, FD.FS, grp)
    ref = {}
    print(f"{'detector':16s} {'factor':>7s} {'normalized':>10s} | kappa vs degrees  kappa vs human")
    for norm in (False, True):
        for f in FACTORS:
            w = Windows((pos * f).astype(np.float32), FD.FS, grp)
            if norm: w, _ = FD.normalize_units(w)
            dets = detectors(w)                                                 # HMM refitted on the rescaled data, like a new recording
            for name, det in dets.items():
                L = det(w)
                key = (name, norm)
                if f == 1.0: ref[key] = L
                ok = np.isfinite(w.pos).all(2)
                k_ref = cohen_kappa_score(ref[key][ok], L[ok]) if L[ok].any() or ref[key][ok].any() else 1.0
                k_hum = cohen_kappa_score(hum[ok], L[ok])
                print(f"{name:16s} {f:7.2f} {str(norm):>10s} | {k_ref:16.3f}  {k_hum:14.3f}")


if __name__ == "__main__":
    main()
