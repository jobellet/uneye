#!/usr/bin/env python3
"""The minimum-duration HMM of free_saccade/detectors.py applied to different 1-D scores (one value per sample, 1 kHz).

The HMM has two Gaussian emissions (fixation / saccade) on the score, a chain of d = 6 saccade states (a saccade lasts at least 6 samples)
and transition probabilities; it is fitted without labels on the unlabeled train split of the target dataset (Viterbi training) and then
decodes the test split. Scores compared (event F1 on the test split, same scorer as foundation/compare.py):
  speed          log10(1 + speed)                        (what free_saccade's HMM uses)
  detrended      log10(1 + speed of the velocity minus its 100 ms running median)
  emb-unsup      z . w_reg: direction of the self-supervised embedding that regresses the detrended-speed input (NO label at all)
  emb-sup LOO    logistic-regression logit on the embedding, trained with the human labels of the OTHER four datasets only
  feat-sup LOO   the same on the 6 plain input features (control: does the embedding add anything?)
The supervised scores are shown thresholded at 0.5 AND decoded by the HMM (what the HMM adds).
Run from the repository root:  python foundation/hmm_score.py [--encoder foundation/runs/ssl_encoder.pt]
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from sklearn.linear_model import LogisticRegression
from foundation import data as FD, ssl_encoder as SE, compare as CP
from free_saccade import detectors as D, cebra_seed as C

FS = 1000.0


class Scores:
    """what HMM.fit / HMM.predict need from a 'window' object: the score, the valid mask and fs"""
    def __init__(self, score, valid, fs=FS): self.score, self.valid, self.fs = score.astype(np.float32), valid, fs


class ScoreHMM(D.HMM):
    def feat(self, win): return win.score, win.valid


def hmm_decode(sA, vA, sB, vB):
    h = ScoreHMM(FS).fit(Scores(sA, vA))
    return h.predict(Scores(sB, vB))


def subsample(S, n, seed=1):
    sel = np.random.RandomState(seed).permutation(len(S))[:n]
    return FD.Set(S.name, S.pos[sel], S.lab[sel], S.coarse)


def prepare(S, net):
    pos = S.pos
    valid = np.isfinite(pos).all(2)
    sp, _ = D.speed_of(pos.astype(np.float64), FS); vd, _ = C.detrended_velocity(pos.astype(np.float64), FS)
    return dict(S=S, valid=valid, speed=np.log10(sp + 1.0), detr=np.log10(np.hypot(vd[..., 0], vd[..., 1]) + 1.0),
                Z=SE.embed(net, S.pos), F=C.feats(pos, FS))


def lr_score(Xtr, ytr, X):
    lr = LogisticRegression(max_iter=300, class_weight="balanced").fit(Xtr, ytr)
    return X @ lr.coef_[0] + lr.intercept_[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", default=None); ap.add_argument("--targets", default=",".join(FD.ALL)); a = ap.parse_args()
    net = SE.load_encoder(a.encoder); targets = a.targets.split(",")
    data = {}
    for nm in FD.ALL:
        A, B = FD.load(nm, "train"), FD.load(nm, "test")
        data[nm] = (prepare(subsample(A, 300 if nm != "andersson" else 80), net), prepare(B, net))
    rows = []
    for tgt in targets:
        PA, PB = data[tgt]; others = [k for k in FD.ALL if k != tgt]; res = {}
        # supervised scores: labels of the OTHER datasets only
        def stack(key):
            X, y = [], []
            for k in others:
                P = data[k][0]; m = (P["S"].lab >= 0) & P["valid"]
                idx = np.nonzero(m.ravel())[0]; idx = np.random.RandomState(0).choice(idx, min(40000, len(idx)), replace=False)
                X.append(P[key].reshape(-1, P[key].shape[-1])[idx]); y.append((P["S"].lab == 1).ravel()[idx])
            return np.concatenate(X), np.concatenate(y)
        sc = {}
        for key, name in (("Z", "emb-sup LOO"), ("F", "feat-sup LOO")):
            Xtr, ytr = stack(key)
            lr = LogisticRegression(max_iter=300, class_weight="balanced").fit(Xtr, ytr)
            sc[name] = tuple((P[key].reshape(-1, P[key].shape[-1]) @ lr.coef_[0] + lr.intercept_[0]).reshape(P["S"].lab.shape) for P in (PA, PB))
        # unsupervised embedding direction
        ZA = PA["Z"].reshape(-1, PA["Z"].shape[-1]); tA = C.feats(PA["S"].pos, FS)[..., 5].reshape(-1)
        w = np.linalg.lstsq(ZA[::7], tA[::7], rcond=None)[0]
        sc["emb-unsup"] = tuple((P["Z"].reshape(-1, P["Z"].shape[-1]) @ w).reshape(P["S"].lab.shape) for P in (PA, PB))
        sc["speed"] = (PA["speed"], PB["speed"]); sc["detrended"] = (PA["detr"], PB["detr"])
        for name, (sA, sB) in sc.items():
            pred = hmm_decode(sA, PA["valid"], sB, PB["valid"])
            res[f"{name} + HMM"] = CP.event_f1(pred, PB["S"])["ev_f1"]
            if "sup" in name and "unsup" not in name:
                res[f"{name}, threshold 0.5 (no HMM)"] = CP.event_f1((sB > 0) & PB["valid"], PB["S"])["ev_f1"]
        rows.append(dict(dataset=tgt, **res))
        print(f"[{tgt:9s}] " + " | ".join(f"{k}: {v:.3f}" for k, v in res.items()), flush=True)
    out = os.path.join(ROOT, "foundation", "runs", "hmm_score.json"); json.dump(rows, open(out, "w"), indent=1, default=float); print("wrote", out)


if __name__ == "__main__":
    main()
