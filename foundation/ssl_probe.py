"""Validation of a self-supervised encoder during training: leave-one-dataset-out linear probe on the TRAIN splits (set A, human labels used ONLY here,
to choose checkpoints / hyper-parameters; the test splits (set B) are never touched).

For each of the 5 labeled datasets, 60 trials (20 for Andersson) of its train split are embedded (sliding windows, per-sample feature = the token
features of the window crops, averaged where crops overlap); a logistic regression fitted on the OTHER four datasets (balanced classes) is scored by the
AUC (saccade vs rest, sample by sample) on the fifth. The mean over the 5 folds is the early-stopping criterion. The same probe on the plain input
features (8 channels) is the level a representation must beat to add anything.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from foundation import data as FD, dino1d as DN

DEV, BASE, G_LEN, L_LEN = DN.DEV, DN.BASE, DN.G_LEN, DN.L_LEN
_CACHE = {}


def val_sets():
    if "sets" not in _CACHE:
        out = {}
        for k in FD.ALL:
            S = FD.load(k, "train"); sel = np.random.RandomState(3).permutation(len(S))[:60 if k != "andersson" else 20]
            out[k] = (S.pos[sel], S.lab[sel])
        _CACHE["sets"] = out
    return _CACHE["sets"]


@torch.no_grad()
def sliding_tokens(token_fn, pos, patch, bs=64):
    """token_fn(feats (B, G_LEN, 8) float tensor) -> (B, G_LEN // patch, C); returns (n, T, C) per-sample features (NaN where no window covers)"""
    n, T, _ = pos.shape; pad = (BASE - G_LEN) // 2
    padded = np.concatenate([np.full((n, pad, 2), np.nan, np.float32), pos.astype(np.float32), np.full((n, pad + BASE, 2), np.nan, np.float32)], 1)
    acc = cnt = None
    for s in range(0, T + pad, L_LEN):
        win = padded[:, s:s + BASE]
        for i in range(0, n, bs):
            f = torch.as_tensor(DN.feats(win[i:i + bs], 1000.0)[:, pad:pad + G_LEN], device=DEV)
            m = np.repeat(token_fn(f).float().cpu().numpy(), patch, axis=1)
            if acc is None: C = m.shape[2]; acc = np.zeros((n, T + 2 * BASE, C), np.float32); cnt = np.zeros((n, T + 2 * BASE, 1), np.float32)
            acc[i:i + bs, s + pad:s + pad + G_LEN] += m; cnt[i:i + bs, s + pad:s + pad + G_LEN] += 1
    out = acc[:, pad:pad + T] / np.maximum(cnt[:, pad:pad + T], 1); out[cnt[:, pad:pad + T, 0] == 0] = np.nan
    return out


def loo_probe(feature_fn, n_fit=20000, seed=0):
    """feature_fn(pos) -> (n, T, C); returns {dataset: AUC}, mean"""
    rng = np.random.RandomState(seed); X, Y = {}, {}
    for k, (pos, lab) in val_sets().items():
        F_ = feature_fn(pos); ok = (lab >= 0) & np.isfinite(pos).all(2) & np.isfinite(F_[..., 0])
        idx = np.nonzero(ok.ravel())[0]; idx = rng.choice(idx, min(n_fit, len(idx)), replace=False)
        X[k] = F_.reshape(-1, F_.shape[-1])[idx]; Y[k] = (lab == 1).ravel()[idx]
    res = {}
    for k in X:
        tr = [j for j in X if j != k]
        mu = np.concatenate([X[j] for j in tr]).mean(0); sd = np.concatenate([X[j] for j in tr]).std(0) + 1e-6
        lr = LogisticRegression(max_iter=200, class_weight="balanced").fit((np.concatenate([X[j] for j in tr]) - mu) / sd, np.concatenate([Y[j] for j in tr]))
        res[k] = roc_auc_score(Y[k], lr.decision_function((X[k] - mu) / sd))
    return res, float(np.mean(list(res.values())))


def baseline_input_features():
    if "base" not in _CACHE: _CACHE["base"] = loo_probe(lambda pos: DN.feats(pos, 1000.0))
    return _CACHE["base"]


def loo_probe_hmm(feature_fn, n_fit=20000, seed=0):
    """END-TO-END validation, same chain as foundation/hmm_score.py: linear score fitted on the labels of the OTHER four datasets (train splits), decoded by the
    minimum-duration HMM (fitted without labels on the held-out dataset's own unlabeled val trials), event F1 and kappa vs the human labels of the held-out
    val trials. Returns ({dataset: (F1, kappa)}, mean F1, mean kappa)."""
    from foundation import compare as CP
    from foundation.hmm_score import ScoreHMM, Scores
    rng = np.random.RandomState(seed); Fe, V = {}, {}
    for k, (pos, lab) in val_sets().items(): Fe[k] = feature_fn(pos)
    res = {}
    for k, (pos, lab) in val_sets().items():
        Xtr, Ytr = [], []
        for j, (pj, lj) in val_sets().items():
            if j == k: continue
            ok = (lj >= 0) & np.isfinite(pj).all(2) & np.isfinite(Fe[j][..., 0]); idx = np.nonzero(ok.ravel())[0]; idx = rng.choice(idx, min(n_fit, len(idx)), replace=False)
            Xtr.append(Fe[j].reshape(-1, Fe[j].shape[-1])[idx]); Ytr.append((lj == 1).ravel()[idx])
        Xtr, Ytr = np.concatenate(Xtr), np.concatenate(Ytr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        lr = LogisticRegression(max_iter=300, class_weight="balanced").fit((Xtr - mu) / sd, Ytr)
        valid = np.isfinite(pos).all(2) & np.isfinite(Fe[k][..., 0]); score = np.zeros(valid.shape, np.float32)
        score[valid] = lr.decision_function((Fe[k][valid] - mu) / sd)
        pred = ScoreHMM(1000.0).fit(Scores(score, valid)).predict(Scores(score, valid))
        r = CP.event_f1(pred, FD.Set(k, pos, lab, False)); res[k] = (r["ev_f1"], r["kappa"])
    return res, float(np.mean([v[0] for v in res.values()])), float(np.mean([v[1] for v in res.values()]))


def baseline_hmm():
    if "base_hmm" not in _CACHE: _CACHE["base_hmm"] = loo_probe_hmm(lambda pos: DN.feats(pos, 1000.0))
    return _CACHE["base_hmm"]


if __name__ == "__main__":
    r, m = baseline_input_features(); print("leave-one-dataset-out probe on the 8 input channels, AUC:", {k: round(v, 3) for k, v in r.items()}, "mean", round(m, 3))
    r, f, k = baseline_hmm(); print("same + HMM decoding, event F1 / kappa on the val trials:", {a: (round(b[0], 3), round(b[1], 3)) for a, b in r.items()}, "mean F1", round(f, 3), "mean kappa", round(k, 3))
