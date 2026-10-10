#!/usr/bin/env python3
"""U'n'Eye (uneye.DNN, the 2019 code, CPU) retrained on exactly the trials used by paper_figs.py, same test trials, same scorer (compare.event_f1) -> docs/figs_paper/results_uneye.json
A: dataset 1, N labeled trials of set B (30 of them, or N // 5 if fewer than 150, are the early-stopping validation samples of U'n'Eye itself), test on 300 trials of set A.
B: dataset 4, ALL 330 trials of one subject of set A (30 for validation) -> test on each subject of set B; 'all' = 33 trials of each subject.
python foundation/uneye_curves.py [--only A|B] [--reps 3]
"""
import argparse, json, os, shutil, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from foundation import data as FD, compare as CP
import uneye
OUT = os.path.join(ROOT, "docs", "figs_paper"); RES = os.path.join(OUT, "results_uneye.json"); TMP = tempfile.mkdtemp(prefix="uneye_w_")


def fit_test(pos, lab, tests, seed, tag, dist=10):
    nv = 30 if len(pos) >= 150 else max(len(pos) // 5, 2)
    m = uneye.DNN(weights_name=os.path.join(TMP, tag), sampfreq=1000, val_samples=nv, min_sacc_dur=6, min_sacc_dist=dist, max_iter=500)
    m.train(pos[..., 0].astype(float), pos[..., 1].astype(float), lab.astype(float), seed=seed)
    out = []
    for tp, tl in tests:
        pred, _ = m.predict(tp[..., 0].astype(float), tp[..., 1].astype(float))
        pred = (np.asarray(pred) == 1) & np.isfinite(tp).all(2); r = CP.event_f1(pred, type("S", (), dict(lab=tl))()); out.append((r["ev_f1"], r["kappa"]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="AB"); ap.add_argument("--reps", type=int, default=3); a = ap.parse_args()
    res = json.load(open(RES)) if os.path.exists(RES) else {"A": {}, "B": {}}; save = lambda: json.dump(res, open(RES, "w"), indent=1)
    if "A" in a.only:
        tr, te = FD.load("d1", "test"), FD.load("d1", "train"); tp, tl = te.pos[:300], te.lab[:300]
        for N in (10, 20, 50, 100, 200, 300, 1000):
            for r in range(a.reps):
                key = f"{N}_{r}"
                if key in res["A"]: continue
                t0 = time.time(); sel = np.random.RandomState(100 * N + r).permutation(len(tr))[:N]
                res["A"][key] = fit_test(tr.pos[sel], tr.lab[sel], [(tp, tl)], r, "a", dist=10)[0]; save()
                print(f"[uneye A] N={N} draw {r}: F1 {res['A'][key][0]:.3f} kappa {res['A'][key][1]:.3f} ({time.time() - t0:.0f} s)", flush=True)
    if "B" in a.only:
        trA, teB = FD.load("d4", "train"), FD.load("d4", "test"); n = lambda f, k: np.loadtxt(os.path.join(ROOT, "data", "dataset4", f), delimiter=",")[:k]
        subA, subB = n("Subject_nb_setA.csv", len(trA)), n("Subject_nb_setB.csv", len(teB)); subjects = sorted(np.unique(subA)); rng = np.random.RandomState(0)
        tests = [(teB.pos[subB == s], teB.lab[subB == s]) for s in subjects]
        for tr_s in subjects + ["all"]:
            if str(tr_s) in res["B"]: continue
            idx = np.concatenate([rng.permutation(np.where(subA == s)[0])[:33] for s in subjects]) if tr_s == "all" else np.where(subA == tr_s)[0]
            t0 = time.time(); res["B"][str(tr_s)] = fit_test(trA.pos[idx], trA.lab[idx], tests, 0, "b", dist=6); save()
            print(f"[uneye B] train {tr_s}: mean test F1 {np.mean([v[0] for v in res['B'][str(tr_s)]]):.3f} ({time.time() - t0:.0f} s)", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
