#!/usr/bin/env python3
"""Do saccades and fixations separate in a PURELY self-supervised embedding (no detector, no seed, no label)?

CEBRA-Time (positives = time neighbours |dt| <= 2 samples, references drawn uniformly: no Engbert-Kliegl hint at all) is trained on
set A of one dataset and set B is embedded. Then, without labels: k-means (k=2), Gaussian mixture (2 components) and HDBSCAN
(the density-based successor of DBSCAN). Human labels are used ONLY afterwards, to score:
  * best cluster-to-class assignment: Cohen's kappa and adjusted Rand index;
  * linear separability (logistic regression, 5-fold, AUC) and silhouette of the human classes - a ceiling, not a method;
the same numbers for the 6 hand-made input features (what the network sees) show whether the self-supervised training adds anything.
Run from the repository root:  python free_saccade/separability.py [--dataset u3] [--steps 1500]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from free_saccade import data as FD, detectors as D, cebra_seed as C


def best_kappa(cl, y):
    """each cluster -> the class that is the majority of its samples (noise label -1 = fixation); kappa of that labeling"""
    pred = np.zeros_like(y)
    for c in np.unique(cl):
        if c < 0: continue
        m = cl == c
        pred[m] = y[m].mean() > 0.5
    return cohen_kappa_score(y, pred), pred


def evaluate(X, y, name, rng, out):
    sub = rng.choice(len(y), min(20000, len(y)), replace=False); Xs, ys = X[sub], y[sub]
    res = dict(space=name, saccade_share=float(ys.mean()))
    res["auc_linear (ceiling)"] = float(cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"), Xs, ys, cv=5, scoring="roc_auc").mean())
    res["silhouette_human"] = float(silhouette_score(Xs[:5000], ys[:5000], metric="cosine" if name != "input features" else "euclidean"))
    cls = {"kmeans k=2": KMeans(2, n_init=10, random_state=0).fit_predict(Xs),
           "GMM 2": GaussianMixture(2, random_state=0).fit(Xs).predict(Xs),
           "HDBSCAN": HDBSCAN(min_cluster_size=50).fit_predict(Xs)}
    for k, cl in cls.items():
        kap, _ = best_kappa(cl, ys)
        res[f"{k}: kappa"] = float(kap); res[f"{k}: ARI"] = float(adjusted_rand_score(ys, cl))
        if k == "HDBSCAN":
            n_cl = len(set(cl) - {-1}); purest = max((ys[cl == c].mean() for c in set(cl) - {-1}), default=float("nan"))
            res["HDBSCAN: clusters"] = n_cl; res["HDBSCAN: noise share"] = float((cl < 0).mean()); res["HDBSCAN: purest cluster saccade share"] = float(purest)
    out.append(res)
    return Xs, ys, cls


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="u3"); ap.add_argument("--steps", type=int, default=C.STEPS); a = ap.parse_args()
    A = FD.load_labeled("data", names=(a.dataset,), which="A", n=1000)[a.dataset]
    B = FD.load_labeled("data", names=(a.dataset,), which="B", n=1000)[a.dataset]
    no_hint = np.full(A.pos.shape[:2], -1, np.int8)                       # no seed, no label: references drawn uniformly
    m = C.SeededCEBRA(seed=0, hybrid=False).train(A, no_hint, steps=a.steps)
    z = m.embed(B).numpy()
    f = C.feats(B.pos, B.fs)
    v, valid = D.velocity(B.pos, B.fs); ok = valid & ~D.invalid_zone(valid, B.fs)
    y = B.labels[ok].astype(int)
    rng = np.random.RandomState(0); rows = []
    Zs, ys, cz = evaluate(z[ok], y, "CEBRA-Time embedding (no hint)", rng, rows)
    evaluate(f[ok][:, :4], y, "input features", np.random.RandomState(0), rows)        # the 4 kinematic features (valid flag / detrended excluded)
    evaluate(f[ok], y, "input features + detrended speed", np.random.RandomState(0), rows)
    df = pd.DataFrame(rows).set_index("space").T
    pd.set_option("display.width", 200); print(f"dataset {a.dataset}, set B, {ok.sum()} valid samples, saccade share {y.mean():.3f}"); print(df.round(3).to_string())
    os.makedirs("free_results", exist_ok=True); df.to_csv(f"free_results/separability_{a.dataset}.csv")
    P = PCA(2).fit_transform(Zs)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    for axi, (col, title) in zip(ax, ((ys, "human label (scoring only)"), (cz["kmeans k=2"], "k-means k=2 (no label)"), (cz["HDBSCAN"], "HDBSCAN (no label, -1 = noise)"))):
        o = np.argsort(col); axi.scatter(P[o, 0], P[o, 1], c=col[o], s=2, cmap="coolwarm" if col is ys else "tab10", alpha=0.6); axi.set_title(title)
    fig.suptitle(f"{a.dataset}: self-supervised CEBRA-Time embedding of set B (PCA of {C.DIM} dims)"); fig.tight_layout()
    fig.savefig(f"free_results/separability_{a.dataset}.png", dpi=110); print("wrote", f"free_results/separability_{a.dataset}.png")


if __name__ == "__main__":
    main()
