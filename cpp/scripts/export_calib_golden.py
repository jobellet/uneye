#!/usr/bin/env python3
import os
import sys
import numpy as np

# fix path resolution: __file__ is cpp/scripts/export_calib_golden.py
# we want ROOT to be /app
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "online"))

from foundation.conventions import apply_calibration, ev_score, GRID

def main():
    np.random.seed(42)
    out_dir = os.path.join(ROOT, "cpp", "app", "calib", "tests")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "golden_calib.txt")
    print(f"Writing golden file to {out_file}")

    with open(out_file, "w") as fp:
        for i in range(10):
            T = 1000
            pos = np.cumsum(np.random.randn(T, 2) * 0.1, axis=0)

            det_mask = np.zeros(T, bool)
            exp_mask = np.zeros(T, bool)

            for _ in range(5):
                s = np.random.randint(0, T - 20)
                e = s + np.random.randint(5, 15)
                det_mask[s:e+1] = True

            for _ in range(5):
                s = np.random.randint(0, T - 20)
                e = s + np.random.randint(5, 15)
                exp_mask[s:e+1] = True

            a = float(np.random.choice([0.0, 0.2, 0.4]))
            pre = int(np.random.choice([-4, 0, 4]))
            post = int(np.random.choice([-8, 0, 8]))
            gap = int(np.random.choice([0, 8, 20]))

            calib_mask = apply_calibration(det_mask, pos, a, pre, post, gap)

            pred_batch = calib_mask[None, :]
            pos_batch = pos[None, :]
            lab_batch = exp_mask.astype(np.int8)[None, :]
            f1, kappa = ev_score(pred_batch, pos_batch, lab_batch)

            best = -9
            bp = (0.0, 0, 0, 0)
            for prm in GRID:
                m = apply_calibration(det_mask, pos, *prm)
                f, k = ev_score(m[None, :], pos_batch, lab_batch)
                obj = 0.5 * (f + (0.0 if np.isnan(k) else k))
                if obj > best:
                    best = obj
                    bp = prm

            fp.write(f"TEST {i}\n")
            fp.write(f"POS\n")
            for t in range(T):
                fp.write(f"{pos[t,0]:.6f} {pos[t,1]:.6f}\n")

            def runs(mask):
                d = np.diff(np.r_[0, mask.astype(int), 0])
                return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

            det_ev = runs(det_mask)
            fp.write(f"DET {len(det_ev)}\n")
            for s, e in det_ev: fp.write(f"{s} {e}\n")

            exp_ev = runs(exp_mask)
            fp.write(f"EXP {len(exp_ev)}\n")
            for s, e in exp_ev: fp.write(f"{s} {e}\n")

            fp.write(f"APPLY_IN {a} {pre} {post} {gap}\n")

            calib_ev = runs(calib_mask)
            fp.write(f"APPLY_OUT {len(calib_ev)}\n")
            for s, e in calib_ev: fp.write(f"{s} {e}\n")

            fp.write(f"SCORE {f1:.6f} {0.0 if np.isnan(kappa) else kappa:.6f}\n")
            fp.write(f"FIT {bp[0]} {bp[1]} {bp[2]} {bp[3]} {best:.6f}\n")

if __name__ == "__main__":
    main()
