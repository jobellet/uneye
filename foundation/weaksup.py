#!/usr/bin/env python3
"""Weak supervision (data programming, Ratner et al. NeurIPS 2016) instead of a single pseudo-labeler (suggestion of the Antigravity review, rank 2).
Six label-free labeling functions vote on every sample of the unlabeled pool (archive/ + unlabeled train splits of the benchmarks, foundation/ssl_vit.pool_cached): Engbert-Kliegl, Engbert-Kliegl on the
detrended velocity, the universal minimum-duration HMM, a position-jump criterion (net displacement over +-8 samples against its robust noise), a high-precision speed rule (> 8 sigma) and a
high-recall detrended-speed rule (> 3 sigma). A Dawid-Skene generative label model (binary, labeling functions conditionally independent given the true state, fitted by EM WITHOUT any label) estimates the
accuracy of each function and gives a posterior p(saccade); a bidirectional TCN (the one of foundation/selftrain.py, 8 channels) is trained on the soft labels. Control: the same student on the plain majority vote.
No human label is used for training; as in selftrain.py the labeled TRAIN splits only pick the checkpoint (early stopping on the validation event F1). Test = common test subsets (night_eval).
python foundation/weaksup.py [--steps 4000] [--max-minutes 22]  ->  night/weaksup_ds*.json, weaksup_mv*.json, weaksup_lf.json
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np
from scipy.ndimage import binary_dilation
from free_saccade import detectors as D, cebra_seed as C
from foundation import bitcn as BT, night_eval as NE, selftrain as ST
from foundation.ssl_vit import pool_cached
FS = 1000.0


def labeling_functions(pool, grp):
    """-> votes (n, T, 6) bool, valid (n, T) bool"""
    n, T, _ = pool.shape; valid = np.isfinite(pool).all(2)
    v, _ = D.velocity(pool, FS); sp = np.hypot(v[..., 0], v[..., 1]); sg_sp = np.maximum(1.4826 * np.nanmedian(np.where(valid, sp, np.nan), 1)[:, None], 1e-9)       # robust speed noise per window
    lf = np.zeros((n, T, 6), bool)
    lf[..., 0] = D.ek(pool, FS, lam=6.0)
    vd, vvalid = C.detrended_velocity(pool, FS); sd = D.robust_sigma(vd); e = ((vd / sd) ** 2).sum(2); lf[..., 1] = D.cleanup((e > 36.0) & vvalid & ~D.invalid_zone(vvalid, FS), FS, min_ms=6.0, merge_ms=0)
    lf[..., 2] = ST.pseudo_labels_hmm(pool, grp)
    p, _ = D.fill(pool); w = 8; d = np.zeros((n, T)); disp = np.hypot(*(p[:, 2 * w:] - p[:, :T - 2 * w]).transpose(2, 0, 1)); d[:, w:T - w] = disp
    sdisp = np.maximum(1.4826 * np.median(disp, 1)[:, None], 1e-9)                                             # noise of the displacement over the window where it is defined (not the w zeros at each end)
    lf[..., 3] = binary_dilation(d > 6.0 * sdisp, structure=np.ones((1, 2 * w + 1), bool))
    lf[..., 4] = sp > 8.0 * sg_sp
    lf[..., 5] = D.cleanup((np.hypot(*vd.transpose(2, 0, 1)) > 3.0 * np.maximum(1.4826 * np.median(np.hypot(*vd.transpose(2, 0, 1)), 1)[:, None], 1e-9)) & vvalid, FS, min_ms=4.0, merge_ms=0)
    return lf & valid[..., None], valid


def dawid_skene(votes, iters=60, seed=0, n_max=2_000_000):
    """votes (M, J) bool. EM for pi = P(y=1), alpha_j = P(v_j=1 | y=1), beta_j = P(v_j=0 | y=0). Returns a function votes -> posterior P(y=1 | votes) and the parameters."""
    rng = np.random.RandomState(seed); V = votes[rng.permutation(len(votes))[:n_max]].astype(np.float64); pi, al, be = 0.08, np.full(V.shape[1], 0.7), np.full(V.shape[1], 0.95)
    def post(V_, pi, al, be):
        lo = np.log(pi / (1 - pi)) + (V_ * np.log(al / (1 - be)) + (1 - V_) * np.log((1 - al) / be)).sum(1); return np.clip(1 / (1 + np.exp(-np.clip(lo, -30, 30))), 0.01, 0.99)           # the functions are correlated, so the independent-votes posterior is over-confident: bounded
    for it in range(iters):
        q = post(V, pi, al, be); pi = float(np.clip(q.mean(), 1e-3, 0.5)); al = np.clip((q[:, None] * V).sum(0) / max(q.sum(), 1e-9), 1e-3, 1 - 1e-3); be = np.clip(((1 - q)[:, None] * (1 - V)).sum(0) / max((1 - q).sum(), 1e-9), 1e-3, 1 - 1e-3)
        bad = al + be <= 1.0; al[bad] = 1.0 - be[bad] + 1e-3                                                  # identifiability: every function must be better than chance (no label switching)
    return (lambda V_: post(V_.astype(np.float64), pi, al, be)), dict(pi=pi, alpha=al.tolist(), beta=be.tolist())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--max-minutes", type=float, default=22); a = ap.parse_args()
    pool = pool_cached(); grp = np.load(os.path.join(ROOT, "foundation", "runs", "pool.npz"))["grp"]; t0 = time.time(); lf, valid = labeling_functions(pool, grp); print(f"[weaksup] labeling functions {lf.shape} in {time.time() - t0:.0f} s, fraction of saccade votes per function:", np.round(lf[valid].mean(0), 3).tolist(), flush=True)
    names = ["EK", "EK detrended", "universal HMM", "position jump", "speed > 8 sigma", "detrended speed > 3 sigma"]
    model, par = dawid_skene(lf[valid]); post = np.zeros(valid.shape, np.float32); post[valid] = model(lf[valid]).astype(np.float32); mv = np.zeros(valid.shape, np.float32); mv[valid] = (lf[valid].sum(1) >= 3).astype(np.float32)
    # the human labels of the benchmark train windows are NOT used to fit anything: only to report how good each function and the label model are (sanity check)
    json.dump(dict(names=names, **par, saccade_fraction_posterior=float(post[valid].mean()), saccade_fraction_majority=float(mv[valid].mean())), open(os.path.join(NE.NIGHT, "weaksup_lf.json"), "w"), indent=1)
    print("[weaksup] Dawid-Skene:", {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in par.items()}, "| posterior saccade fraction %.3f, majority vote %.3f" % (post[valid].mean(), mv[valid].mean()), flush=True)
    for tag, labels, desc in (("weaksup_ds", post, "Dawid-Skene label model posterior (soft labels)"), ("weaksup_mv", mv, "majority vote of >= 3 of 6 labeling functions (control)")):
        net, best, log, minutes = ST.train_student(pool, labels, tag, a.steps, 2e-3, a.max_minutes, seed=1, criterion="thr")
        for suffix, tta in (("", False), ("_tta", True)):
            res = NE.eval_direct(BT.score_fn_factory(net, tta=tta))
            NE.save(tag + suffix, "label-free weak supervision (student of a label model)", "none for training (labeled train splits only select the checkpoint)", "Ratner et al. 2016 (data programming); Dawid & Skene 1979; Xie et al. 2020",
                    dict(labels=desc, steps=a.steps, tta=tta), res, minutes, dict(best_val_event_f1=best, log=log))


if __name__ == "__main__":
    main()
