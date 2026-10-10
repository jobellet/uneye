"""Common evaluation of every candidate of the overnight comparison (docs/OVERNIGHT_REPORT.md): same data, same scorer, same references.

Test data: a fixed subset of the TEST split (set B) of each benchmark (300 trials, 60 for Andersson; the same trials for every candidate), scored at
1 kHz by foundation/compare.py (event F1 and Cohen's kappa, saccade vs rest). Train data for fitting probes / unlabeled HMMs: 150 trials of the train split
(40 for Andersson). The test split is never used to choose anything.

Two kinds of candidates:
  features   a representation (n, T, C) per sample -> linear score fitted on the human labels of the OTHER four datasets (train splits) [leave-one-dataset-out]
             -> (a) threshold, (b) minimum-duration HMM fitted without labels on the target's own train trials -> F1 / kappa on the target's test subset.
  direct     a model that outputs a score (n, T) (logit of "saccade") -> (a) threshold, (b) the same HMM decoding.
Both write a JSON in foundation/runs/night/ with the same schema.
"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from sklearn.linear_model import LogisticRegression
from foundation import data as FD, compare as CP
from foundation.hmm_score import ScoreHMM, Scores

NIGHT = os.path.join(ROOT, "foundation", "runs", "night"); os.makedirs(NIGHT, exist_ok=True)
_C = {}


def subset(S, n, seed):
    sel = np.random.RandomState(seed).permutation(len(S))[:n]
    return FD.Set(S.name, S.pos[sel], S.lab[sel], S.coarse)


def data():
    """{dataset: (train subset, test subset)}"""
    if "d" not in _C:
        _C["d"] = {k: (subset(FD.load(k, "train"), 150 if k != "andersson" else 40, 11), subset(FD.load(k, "test"), 300 if k != "andersson" else 60, 5)) for k in FD.ALL}
    return _C["d"]


def _decode(scoreA, validA, scoreB, validB, S_B):
    pred = ScoreHMM(1000.0).fit(Scores(scoreA, validA)).predict(Scores(scoreB, validB))
    return CP.event_f1(pred, S_B)


def _both(scoreA, validA, scoreB, validB, S_B):
    thr = CP.event_f1((scoreB > 0) & validB, S_B); hmm = _decode(scoreA, validA, scoreB, validB, S_B)
    return dict(threshold=dict(f1=thr["ev_f1"], kappa=thr["kappa"]), hmm=dict(f1=hmm["ev_f1"], kappa=hmm["kappa"]))


def eval_features(feature_fn, n_fit=20000, seed=0):
    """leave-one-dataset-out linear probe on `feature_fn(pos) -> (n, T, C)` (NaN allowed where undefined)"""
    D_ = data(); rng = np.random.RandomState(seed); FA, FB = {}, {}
    for k, (A, B) in D_.items(): FA[k], FB[k] = feature_fn(A.pos), feature_fn(B.pos)
    res = {}
    for k, (A, B) in D_.items():
        X, Y = [], []
        for j, (Aj, _) in D_.items():
            if j == k: continue
            ok = (Aj.lab >= 0) & np.isfinite(Aj.pos).all(2) & np.isfinite(FA[j][..., 0]); idx = np.nonzero(ok.ravel())[0]; idx = rng.choice(idx, min(n_fit, len(idx)), replace=False)
            X.append(FA[j].reshape(-1, FA[j].shape[-1])[idx]); Y.append((Aj.lab == 1).ravel()[idx])
        X, Y = np.concatenate(X), np.concatenate(Y); mu, sd = X.mean(0), X.std(0) + 1e-6
        lr = LogisticRegression(max_iter=300, class_weight="balanced").fit((X - mu) / sd, Y)
        sc = lambda F_, P_: (np.where(np.isfinite(F_[..., 0]) & np.isfinite(P_).all(2), (np.nan_to_num(F_) - mu) / sd @ lr.coef_[0] + lr.intercept_[0], 0.0).astype(np.float32),
                             np.isfinite(F_[..., 0]) & np.isfinite(P_).all(2))
        (sA, vA), (sB, vB) = sc(FA[k], A.pos), sc(FB[k], B.pos)
        res[k] = _both(sA, vA, sB, vB, B)
    return res


def eval_direct(score_fn):
    """score_fn(pos) -> (n, T) logit of saccade; no label of any dataset is used by this evaluation (the model may have been trained with some)"""
    res = {}
    for k, (A, B) in data().items():
        sA, sB = score_fn(A.pos), score_fn(B.pos)
        vA, vB = np.isfinite(A.pos).all(2) & np.isfinite(sA), np.isfinite(B.pos).all(2) & np.isfinite(sB)
        res[k] = _both(np.nan_to_num(sA).astype(np.float32), vA, np.nan_to_num(sB).astype(np.float32), vB, B)
    return res


def summarize(res):
    out = {}
    for mode in ("threshold", "hmm"):
        out[mode] = dict(f1=float(np.mean([v[mode]["f1"] for v in res.values()])), kappa=float(np.mean([v[mode]["kappa"] for v in res.values()])))
    return out


def save(cid, family, labels_used, paper, hyper, res, minutes, extra=None):
    d = dict(id=cid, family=family, labels_used=labels_used, paper=paper, hyper=hyper, test=res, mean=summarize(res), minutes=round(minutes, 1), finished=time.strftime("%Y-%m-%d %H:%M"))
    d.update(extra or {}); json.dump(d, open(os.path.join(NIGHT, f"{cid}.json"), "w"), indent=1, default=float)
    m = d["mean"]; print(f"[{cid}] mean over 5 datasets: HMM F1 {m['hmm']['f1']:.3f} kappa {m['hmm']['kappa']:.3f} | threshold F1 {m['threshold']['f1']:.3f} kappa {m['threshold']['kappa']:.3f} "
                         f"| " + " ".join(f"{k} {v['hmm']['f1']:.2f}/{v['hmm']['kappa']:.2f}" for k, v in res.items()) + f" ({minutes:.1f} min)", flush=True)
    return d


def references():
    """reference rows on the SAME test subsets: U'n'Eye (supervised on d1+d2+d3 set A; Andersson also with its own weights) and the universal HMM (archive/ only)"""
    from foundation import universal_hmm as UH
    t0 = time.time(); res = {}; res_a = {}
    for k, (A, B) in data().items():
        r = CP.event_f1(CP.pred_unet(B, "weights_1+2+3"), B); res[k] = dict(threshold=dict(f1=r["ev_f1"], kappa=r["kappa"]), hmm=dict(f1=r["ev_f1"], kappa=r["kappa"]))
    save("ref_uneye", "reference", "all labels of d1+d2+d3 train splits (in-domain for d1-d3, unseen for d4 and Andersson)", "Bellet et al. 2019", {"weights": "weights_1+2+3"}, res, (time.time() - t0) / 60)
    r = CP.event_f1(CP.pred_unet(data()["andersson"][1], "weights_Andersson", classes=5), data()["andersson"][1])
    json.dump(dict(f1=r["ev_f1"], kappa=r["kappa"]), open(os.path.join(NIGHT, "ref_uneye_andersson_own.json"), "w"))
    t0 = time.time(); uni = {}
    for fs in (1000.0, 500.0):
        f, v = UH.unit_free_score(UH.archive_windows(fs), "detrended", fs); uni[fs] = ScoreHMM(fs).fit(Scores(f, v, fs))
    res = {}
    for k, (A, B) in data().items():
        fs = CP.NATIVE[k]; f, v = UH.unit_free_score(CP.native_pos(B), "detrended", fs); pred = CP.to_1khz(uni[fs].predict(Scores(f, v, fs)), B)
        r = CP.event_f1(pred, B); res[k] = dict(threshold=dict(f1=r["ev_f1"], kappa=r["kappa"]), hmm=dict(f1=r["ev_f1"], kappa=r["kappa"]))
    save("ref_universal_hmm", "reference", "none (fitted on archive/ only)", "Rabiner 1989 (HMM); this work", {"states": "2 Gaussians, saccade = chain of 6 states", "score": "log10(1 + detrended speed / noise)"}, res, (time.time() - t0) / 60)


if __name__ == "__main__":
    from foundation import dino1d as DN
    if "--six" in sys.argv:
        from free_saccade.cebra_seed import feats as feats6
        t0 = time.time(); res = eval_features(lambda pos: feats6(np.asarray(pos, np.float64), 1000.0))
        save("base_input_6ch", "baseline", "labels of the other 4 datasets (linear probe)", "-", {"input": "6 rotation-invariant velocity channels"}, res, (time.time() - t0) / 60)
    else:
        references()
        t0 = time.time(); res = eval_features(lambda pos: DN.feats(pos, 1000.0))
        save("base_input_channels", "baseline", "labels of the other 4 datasets (linear probe)", "-", {"input": "8 velocity channels"}, res, (time.time() - t0) / 60)
