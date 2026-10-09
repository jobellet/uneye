#!/usr/bin/env python3
"""Calibration of the guard's out-of-distribution noise limit (GuardConfig::ood_sigma_hi_deg_s) WITHOUT looking at the test set.

Rule, fixed before any result was seen (docs comment thread, 2026-10-09):
  1. Run cpp/build/fault_injection on set A (training data of datasets 1+2, 120 trials each) for every candidate limit.
  2. Admissible = guard recall on CLEAN set A at most 2 points below the recall with the current limit (23.1 deg/s).
  3. Pick the admissible limit with the fewest dangerous outputs summed over all perturbations; tie -> the higher limit.
  4. Score set B ONCE with the chosen limit and compare with the current limit.
Caveat: the networks were trained on set A, so they behave a little better there than on new data.

Run from the repository root:  python cpp/scripts/calibrate_ood.py   (about 15-25 min on the M1, 4 runs in parallel)
Writes docs/slides/data/calibration_ood_setA.csv and docs/slides/data/fault_injection_ood<limit>.csv (set B, chosen limit).
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "docs", "slides"))
from make_figures import pooled  # noqa: E402

CANDIDATES = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 23.1, 28.0]
CURRENT, BUDGET = 23.1, 0.02
DATA = os.path.join(ROOT, "docs", "slides", "data")
TMP = os.path.join(ROOT, "cpp", "build", "calib")


def run(limit, dataset_set, out):
    cmd = [os.path.join(ROOT, "cpp", "build", "fault_injection"), "--trials", "120", "--set", dataset_set, "--ood", str(limit), "--out", out]
    subprocess.run(cmd, cwd=os.path.join(ROOT, "cpp"), check=True, stdout=subprocess.DEVNULL)
    return limit, out


def main():
    os.makedirs(TMP, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        done = list(ex.map(lambda l: run(l, "A", os.path.join(TMP, f"setA_{l:g}.csv")), CANDIDATES))
    rows = []
    for limit, path in done:
        g = pooled(pd.read_csv(path)); g = g[g.system == "guard"]
        rows.append(dict(limit=limit, recall_clean=float(g[g.perturbation == "none"].recall.iloc[0]),
                         dangerous_total=int(g.dangerous.sum()),
                         dangerous_noise=int(g[g.perturbation == "noise"].dangerous.sum()),
                         dangerous_burst=int(g[g.perturbation == "burst"].dangerous.sum()),
                         degraded_clean=float(g[g.perturbation == "none"].degraded.iloc[0])))
    t = pd.DataFrame(rows).sort_values("limit")
    ref = float(t[t.limit == CURRENT].recall_clean.iloc[0])
    t["admissible"] = t.recall_clean >= ref - BUDGET
    t.to_csv(os.path.join(DATA, "calibration_ood_setA.csv"), index=False)
    print("set A (calibration):"); print(t.to_string(index=False))
    adm = t[t.admissible]
    best = adm.sort_values(["dangerous_total", "limit"], ascending=[True, False]).iloc[0]
    chosen = float(best.limit)
    print(f"\nchosen limit: {chosen:g} deg/s (current {CURRENT:g}); recall budget {BUDGET:.2f} below {ref:.3f}")
    outB = os.path.join(DATA, f"fault_injection_ood{chosen:g}.csv")
    run(chosen, "B", outB)
    old = pooled(pd.read_csv(os.path.join(DATA, "fault_injection.csv"))); new = pooled(pd.read_csv(outB))
    o, n = old[old.system == "guard"].reset_index(drop=True), new[new.system == "guard"].reset_index(drop=True)
    cmp = pd.DataFrame(dict(perturbation=o.perturbation, level=o.level, dangerous_current=o.dangerous, dangerous_chosen=n.dangerous,
                            recall_current=o.recall.round(3), recall_chosen=n.recall.round(3)))
    print(f"\nset B (test, scored once): limit {CURRENT:g} vs {chosen:g}"); print(cmp.to_string(index=False))
    print(f"total dangerous: {int(o.dangerous.sum())} -> {int(n.dangerous.sum())}; cases with 0: {(o.dangerous == 0).sum()} -> {(n.dangerous == 0).sum()} of {len(o)}")


if __name__ == "__main__":
    main()
