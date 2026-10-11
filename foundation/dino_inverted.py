#!/usr/bin/env python3
"""The CLS attention of the trained DINO-for-eye-traces is ANTI-correlated with saccades (AUC 0.05-0.26, foundation/dino_eval.py): its sign is arbitrary for an unsupervised map, so the inverted map (1 - attention)
has AUC 0.74-0.95. Is that a real saccade signal or just the speed it was fed?  For each dataset (same 60 test trials, 20 for Andersson, as dino_eval.py):
  AUC of the inverted attention (sign fixed ONCE for all datasets, not chosen on the test labels), AUC of the speed channel, Spearman correlation between the two, AUC of their rank average, and the event F1 / kappa
  after the unlabeled minimum-duration HMM decoding (fitted on the unlabeled train trials of the same dataset) of: the speed channel, the inverted attention, and their average of z-scores.
python foundation/dino_inverted.py -> runs/night/dino_inverted.json
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
from foundation import dino1d as DN, dino_eval as DE, data as FD, compare as CP, night_eval as NE
from foundation.hmm_score import ScoreHMM, Scores
from foundation.train_lodo import DEV


def maps(model, S, n_trials):
    sel = np.random.RandomState(1).permutation(len(S))[:n_trials]; pos, lab = S.pos[sel], S.lab[sel]
    A = DE.sliding(model, pos, "attn").mean(2); sp = DN.feats(pos, 1000.0)[..., 5]; ok = np.isfinite(A) & np.isfinite(sp) & np.isfinite(pos).all(2)
    return pos, lab, -A, sp, ok


def zs(x, ok, ref=None):
    r = x[ok] if ref is None else ref; m, s = float(np.mean(r)), float(np.std(r)) + 1e-9; return np.where(ok, (x - m) / s, 0.0).astype(np.float32), (m, s)


def main():
    model = DN.DINO().to(DEV); model.load_state_dict(torch.load(os.path.join(ROOT, "foundation", "runs", "dino1d.pt"), weights_only=False)["state"]); model.eval(); rows = {}
    for nm in FD.ALL:
        nt = 20 if nm == "andersson" else 60; St, Ss = FD.load(nm, "test"), FD.load(nm, "train")
        pos, lab, inv, sp, ok = maps(model, St, nt); y = lab == 1; ok &= lab >= 0
        auc = lambda s: float(roc_auc_score(y[ok], s[ok])); rho = float(spearmanr(inv[ok], sp[ok])[0])
        rk = rankdata(inv[ok]) + rankdata(sp[ok]); comb_auc = float(roc_auc_score(y[ok], rk))
        posA, labA, invA, spA, okA = maps(model, Ss, nt)                                                     # unlabeled train trials (labels not read): to fit the HMM
        res = dict(auc_inverted_attention=auc(inv), auc_speed=auc(sp), spearman=rho, auc_rank_average=comb_auc)
        zinv, (mi, si) = zs(inv, ok); zsp, (ms, ss) = zs(sp, ok)
        for tag, sA, sB in (("speed", zs(spA, okA, spA[okA])[0], zs(sp, ok, spA[okA])[0]), ("inverted_attention", zs(invA, okA, invA[okA])[0], zs(inv, ok, invA[okA])[0]),
                            ("average", None, None)):
            if tag == "average":
                sA = 0.5 * (zs(spA, okA, spA[okA])[0] + zs(invA, okA, invA[okA])[0]); sB = 0.5 * (zs(sp, ok, spA[okA])[0] + zs(inv, ok, invA[okA])[0])
            validA = okA & np.isfinite(posA).all(2); validB = ok & np.isfinite(pos).all(2)
            pred = ScoreHMM(1000.0).fit(Scores(sA, validA)).predict(Scores(sB, validB)); S_ = type("S", (), dict(lab=lab))(); r = CP.event_f1(pred, S_); res[f"hmm_{tag}"] = [float(r["ev_f1"]), float(r["kappa"])]
        rows[nm] = res; print(nm, {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in res.items()}, flush=True)
    json.dump(rows, open(os.path.join(NE.NIGHT, "dino_inverted.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
