"""Label efficiency of the representations: with N human-labeled trials of the TARGET dataset (its train split), a logistic regression on frozen features, scored on the
target's common test subset (threshold at 0, and the minimum-duration HMM fitted without labels). N = 1, 5, 20, 100 (3 draws of the trials for the threshold, 1 for the HMM).
Compares the checkpoints of the night (ssl_<name>.pt), the untrained controls and the plain input channels. The standard way to measure what a self-supervised
representation brings (linear probing with few labels). Writes foundation/runs/night/label_eff.json.   python foundation/label_eff.py [--names a,b,c]
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from sklearn.linear_model import LogisticRegression
from foundation import dino1d as DN, night_eval as NE, compare as CP, eval_ckpt as EC
from foundation.hmm_score import ScoreHMM, Scores

TARGETS, NS = ("d1", "d2", "d3", "andersson"), (1, 5, 20, 100)


def curve(feature_fn):
    D_ = NE.data(); out = {}; FA = {k: feature_fn(A.pos) for k, (A, B) in D_.items() if k in TARGETS}; FB = {k: feature_fn(B.pos) for k, (A, B) in D_.items() if k in TARGETS}
    for k in TARGETS:
        A, B = D_[k]; out[k] = {}
        okA = (A.lab >= 0) & np.isfinite(A.pos).all(2) & np.isfinite(FA[k][..., 0]); okB = np.isfinite(B.pos).all(2) & np.isfinite(FB[k][..., 0])
        for N in NS:
            thr, hmm = [], None
            for draw in range(3):
                for attempt in range(20):
                    sel = np.random.RandomState(100 * N + 10 * draw + attempt).permutation(len(A))[:N]; y = (A.lab[sel] == 1)[okA[sel]]
                    if y.any() and (~y).any(): break
                X = FA[k][sel][okA[sel]]; mu, sd = X.mean(0), X.std(0) + 1e-6
                lr = LogisticRegression(max_iter=300, class_weight="balanced").fit((X - mu) / sd, y)
                sB = np.where(okB, (np.nan_to_num(FB[k]) - mu) / sd @ lr.coef_[0] + lr.intercept_[0], 0.0).astype(np.float32)
                thr.append(CP.event_f1((sB > 0) & okB, B))
                if draw == 0:
                    sA = np.where(np.isfinite(FA[k][..., 0]) & np.isfinite(A.pos).all(2), (np.nan_to_num(FA[k]) - mu) / sd @ lr.coef_[0] + lr.intercept_[0], 0.0).astype(np.float32)
                    vA = np.isfinite(A.pos).all(2) & np.isfinite(FA[k][..., 0]); hmm = CP.event_f1(ScoreHMM(1000.0).fit(Scores(sA, vA)).predict(Scores(sB, okB)), B)
            out[k][N] = dict(thr_f1=float(np.mean([t["ev_f1"] for t in thr])), thr_kappa=float(np.mean([t["kappa"] for t in thr])), hmm_f1=hmm["ev_f1"], hmm_kappa=hmm["kappa"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--names", default=""); a = ap.parse_args()
    names = [n for n in a.names.split(",") if n] or sorted(f[4:-3] for f in os.listdir(os.path.join(ROOT, "foundation", "runs")) if f.startswith("ssl_") and f.endswith(".pt") and "encoder" not in f and "quick" not in f)
    path = os.path.join(NE.NIGHT, "label_eff.json"); res = json.load(open(path)) if os.path.exists(path) else {}
    for n in ["input_channels_8"] + names:
        if n in res: continue
        t0 = time.time()
        try: res[n] = curve(lambda pos: DN.feats(pos, 1000.0)) if n == "input_channels_8" else curve(EC.load_feature_fn(n)[0])
        except Exception as e: print(f"[label_eff] {n} FAILED: {type(e).__name__}: {e}", flush=True); continue
        json.dump(res, open(path, "w"), indent=1); print(f"[label_eff] {n}: " + " ".join(f"{k} N=20 HMM F1 {res[n][k][20]['hmm_f1']:.2f}/{res[n][k][20]['hmm_kappa']:.2f}" for k in TARGETS) + f" ({(time.time() - t0) / 60:.1f} min)", flush=True)
