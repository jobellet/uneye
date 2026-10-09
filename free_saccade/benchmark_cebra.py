#!/usr/bin/env python3
"""Benchmark: seeded CEBRA (free_saccade/cebra_seed.py) vs label-free baselines on the 4 labeled datasets of Bellet et al. 2019.

Protocol (no human label used for anything but the final score):
  * all recordings resampled to 500 Hz (free_saccade.data.FS); set A = unlabeled training recordings, set B = test.
  * per dataset: seeds from the signal (Engbert-Kliegl strict / quiet), encoder trained on set A, labels on set B by kNN to the
    seeds of set B itself (transductive, still label-free).
  * methods: Engbert-Kliegl lambda=6 (the classic), the seeds alone, a 2-state HMM fitted on set B, CEBRA-Time (+ kNN to seeds),
    CEBRA-Hybrid (+ kNN to seeds). 2 training seeds for the CEBRA variants (mean shown, both kept in the CSV).
  * metrics vs human labels of set B: Cohen's kappa (samples), event F1 (online/metrics.py, as in the rest of the repository).
Reference (NOT the same protocol: cross-validation on set A at native rate, Table 2 of the paper): supervised U'n'Eye kappa
0.89 / 0.92 / 0.82 on datasets 1 / 2 / 3, Engbert-Mergenthaler 0.66 / - / 0.58.

Run from the repository root:  python free_saccade/benchmark_cebra.py [--quick]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "online"))
import numpy as np, pandas as pd
import metrics as M
from free_saccade import data as FD, detectors as D, cebra_seed as C

QUICK = "--quick" in sys.argv
NAMES = {"u1": "dataset 1 (human, microsaccades)", "u2": "dataset 2 (monkey, pursuit)", "u3": "dataset 3 (monkey, 500 Hz)", "u4": "dataset 4 (human)"}


def score(lab, w):
    m = M.evaluate_probs(lab.astype(np.float32), w.labels.astype(np.float32), w.fs, thr=0.5, min_event=2, with_ap=False)
    return dict(kappa=m["kappa"], f1=m["f1"], ev_f1=m["ev_f1"], ev_recall=m["ev_recall"], ev_precision=m["ev_precision"])


def main():
    rows = []
    steps = 150 if QUICK else C.STEPS
    for nm, title in NAMES.items():
        t0 = time.time()
        A = FD.load_labeled("data", names=(nm,), which="A", n=200 if QUICK else 1000)[nm]
        B = FD.load_labeled("data", names=(nm,), which="B", n=60 if QUICK else 1000)[nm]
        sA, sB = C.seeds(A.pos, A.fs), C.seeds(B.pos, B.fs)
        # how good are the label-free seeds themselves? (only reported, never used)
        prec = float(B.labels[sB == 1].mean()) if (sB == 1).any() else float("nan")
        fprec = float(1 - B.labels[sB == 0].mean()) if (sB == 0).any() else float("nan")
        cov = float((sB[B.labels] == 1).mean())
        print(f"\n== {title}: set A {A.pos.shape}, set B {B.pos.shape} at {B.fs:.0f} Hz | seeds on B: saccade {100 * (sB == 1).mean():.2f} % of samples "
              f"(precision {prec:.3f}, cover {100 * cov:.1f} % of human saccade samples), fixation {100 * (sB == 0).mean():.1f} % (precision {fprec:.4f})", flush=True)
        res = {"EK lambda=6": D.ek(B.pos, B.fs, lam=6.0), "seeds only (EK strict)": D.cleanup(sB == 1, B.fs, min_ms=6.0, merge_ms=0)}
        hmm = D.HMM(B.fs).fit(B); res["HMM (unsupervised)"] = hmm.predict(B)
        for name, lab in res.items():
            r = dict(dataset=nm, method=name, run=0, **score(lab, B)); rows.append(r)
            print(f"  {name:28s} kappa {r['kappa']:.3f}  event F1 {r['ev_f1']:.3f}", flush=True)
        for hybrid, name in ((False, "CEBRA-Time + kNN seeds"), (True, "CEBRA-Hybrid + kNN seeds")):
            ks = []
            for run in range(1 if QUICK else 2):
                m = C.SeededCEBRA(seed=run, hybrid=hybrid).train(A, sA, steps=steps)
                p, lab = m.label(B, sB)
                r = dict(dataset=nm, method=name, run=run, **score(lab, B)); rows.append(r); ks.append((r["kappa"], r["ev_f1"]))
            k = np.mean(ks, 0)
            print(f"  {name:28s} kappa {k[0]:.3f}  event F1 {k[1]:.3f}   (mean of {len(ks)} runs; loss {np.mean(m.hist[:50]):.3f} -> {np.mean(m.hist[-50:]):.3f})", flush=True)
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join("free_results", "benchmark_cebra" + ("_quick" if QUICK else "") + ".csv")
    os.makedirs("free_results", exist_ok=True); df.to_csv(out, index=False)
    print("\nmean over runs, kappa / event F1:")
    t = df.groupby(["method", "dataset"])[["kappa", "ev_f1"]].mean().unstack("dataset").round(3)
    print(t.to_string()); print("wrote", out)


if __name__ == "__main__":
    main()
