#!/usr/bin/env python3
"""Simulation of the planned EyeLive workflow, to decide what to build (docs/EYELIVE_ROADMAP.md): ~50 s of a session are pre-labeled WITHOUT labels (universal HMM + acceleration-zero boundaries, the a-priori rule:
'first inflection' for Andersson, 'last zero' for the others), the expert corrects them, a U'n'Eye (the original network, uneye.DNN, CPU) is trained on them and tested on held-out trials of the same dataset.
The expert is SIMULATED with the human labels of the benchmark (never used by the pseudo-labeler):
  C1 pseudo-labels as they come (HMM + rule)           C2 + the expert deletes the FALSE POSITIVES (predicted events with no human saccade)
  C3 + the expert adds the MISSED saccades (human saccades with no predicted event), false positives kept     C4 = C2 + C3 (both corrections, boundaries still those of the rule for the events kept)
  C5 human labels (= supervised training on the same 50 s, the reference: what U'n'Eye gives with a fully annotated session)
Pretrained start (datasets unseen by the shipped weights d1+d2+d3, i.e. d4 and Andersson): P1 / P4 / P5 = C1 / C4 / C5 labels but the network starts from training/weights_1+2+3 (transfer learning) instead of random weights.
N trials = ceil(50 s / trial duration) of the TRAIN subset (night_eval.data()), 3 draws, test on the common test subset, same scorer (compare.event_f1). Datasets at 500 Hz are upsampled to 1 kHz as everywhere in foundation/.
python foundation/workflow_sim.py [--draws 3]  ->  night/workflow_sim.json
"""
import argparse, json, math, os, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from free_saccade import detectors as D
from foundation import compare as CP, night_eval as NE, universal_hmm as UH, accel_boundaries as AB, data as FD
from foundation.hmm_score import ScoreHMM, Scores
import uneye, uneye.classifier as UC, uneye.functions as UF
TMP = tempfile.mkdtemp(prefix="wf_"); OUT = os.path.join(NE.NIGHT, "workflow_sim.json")


def pseudo(Ssel, k, uni):
    fs = CP.NATIVE[k]; pos = CP.native_pos(Ssel).astype(np.float64); f, v = UH.unit_free_score(CP.native_pos(Ssel), "detrended", fs); hmm = uni[fs].predict(Scores(f, v, fs))
    m = AB.refine(hmm, pos, fs, "both", 15.0, offset_rule="first" if k == "andersson" else "last"); return CP.to_1khz(m, Ssel)


def overlaps(ev, other):  # events of `ev` (list of (s, e)) that touch the boolean mask `other`
    return [any(other[s:e + 1]) for s, e in ev]


def corrections(pseudo_m, human):
    """pseudo_m, human: (n, T) bool -> dict of label arrays for C1..C5"""
    out = {k: np.zeros_like(pseudo_m) for k in ("C1", "C2", "C3", "C4", "C5")}
    for i in range(len(pseudo_m)):
        pe, he = D.runs_1d(pseudo_m[i]), D.runs_1d(human[i]); hit_p = overlaps(pe, human[i]); hit_h = overlaps(he, pseudo_m[i])
        out["C1"][i] = pseudo_m[i]; out["C5"][i] = human[i]
        c2 = np.zeros(pseudo_m.shape[1], bool); [c2.__setitem__(slice(s, e + 1), True) for (s, e), h in zip(pe, hit_p) if h]; out["C2"][i] = c2
        c3 = pseudo_m[i].copy(); [c3.__setitem__(slice(s, e + 1), True) for (s, e), h in zip(he, hit_h) if not h]; out["C3"][i] = c3
        c4 = c2.copy(); [c4.__setitem__(slice(s, e + 1), True) for (s, e), h in zip(he, hit_h) if not h]; out["C4"][i] = c4
    return out


class UNetPre(UF.UNet):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.load_state_dict(torch.load(os.path.join(ROOT, "training", "weights_1+2+3"), map_location="cpu", weights_only=False))


def fit_predict(pos, labels, test_pos, seed, pretrained, tag):
    nv = max(min(10, len(pos) // 5), 2); wp = os.path.join(TMP, tag)
    if os.path.exists(wp): os.remove(wp)
    orig = (UC.UNet, UC.weights_init)
    if pretrained: UC.UNet = UNetPre; UC.weights_init = lambda m: None
    try:
        m = uneye.DNN(weights_name=wp, sampfreq=1000, val_samples=nv, min_sacc_dur=6, min_sacc_dist=10, max_iter=500, lr=1e-4 if pretrained else 1e-3)
        m.train(pos[..., 0].astype(float), pos[..., 1].astype(float), labels.astype(float), seed=seed)
    finally: UC.UNet, UC.weights_init = orig
    if not os.path.exists(wp): raise RuntimeError("no weights saved")
    pred, _ = m.predict(test_pos[..., 0].astype(float), test_pos[..., 1].astype(float)); return np.asarray(pred) == 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=3); ap.add_argument("--only", default=""); a = ap.parse_args(); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    uni = {}
    for fs in (1000.0, 500.0): f, v = UH.unit_free_score(UH.archive_windows(fs), "detrended", fs); uni[fs] = ScoreHMM(fs).fit(Scores(f, v, fs))
    for k, (A, B) in NE.data().items():
        if a.only and k not in a.only.split(","): continue
        N = min(int(math.ceil(50.0 / (A.pos.shape[1] / 1000.0))), len(A)); S_te = type("S", (), dict(lab=B.lab))()
        conds = ["C1", "C2", "C3", "C4", "C5"] + (["P1", "P4", "P5"] if k in ("d4", "andersson") else [])
        for r in range(a.draws):
            sel = np.random.RandomState(100 * 50 + r).permutation(len(A))[:N]; Ssel = FD.Set(A.name, A.pos[sel], A.lab[sel], A.coarse)
            pm = pseudo(Ssel, k, uni); human = (A.lab[sel] == 1); L = corrections(pm, human)
            for c in conds:
                key = f"{k}_{c}_{r}"
                if key in res and res[key] is not None: continue
                lab = L[c] if c[0] == "C" else L[{"P1": "C1", "P4": "C4", "P5": "C5"}[c]]
                if lab.sum() == 0 or (~lab).sum() == 0: res[key] = None; continue
                t0 = time.time()
                try: pred = fit_predict(Ssel.pos, lab, B.pos, r, c[0] == "P", f"{k}_{c}_{r}"); e = CP.event_f1(pred & np.isfinite(B.pos).all(2), S_te); res[key] = [float(e["ev_f1"]), float(e["kappa"]), N, float((lab.sum() / lab.size))]
                except Exception as ex: res[key] = None; print("FAILED", key, ex)
                json.dump(res, open(OUT, "w"), indent=1); print(f"[workflow] {key}: " + (f"F1 {res[key][0]:.3f} kappa {res[key][1]:.3f} (N={N}, {time.time() - t0:.0f} s)" if res[key] else "no result"), flush=True)


if __name__ == "__main__":
    main()
