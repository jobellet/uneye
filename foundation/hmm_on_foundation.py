#!/usr/bin/env python3
"""HMM decoding (minimum-duration chain) of the logit of the supervised foundation network (foundation/model.py, trained WITHOUT the target
dataset: foundation/runs/lodo_<target>.pt, causal with 20 ms lookahead). Event F1 on the test split, same scorer as compare.py.
Run from the repository root:  python foundation/hmm_on_foundation.py [d1,d2]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from foundation import data as FD, model as FM, compare as CP
from foundation.train_lodo import predict, DEV
from foundation.hmm_score import hmm_decode, subsample

for tgt in (sys.argv[1] if len(sys.argv) > 1 else "d1,d2").split(","):
    ck = torch.load(os.path.join(ROOT, "foundation", "runs", f"lodo_{tgt}.pt"), weights_only=False)
    net = FM.Foundation(); net.load_state_dict(ck["state"]); net.to(DEV)
    A, B = subsample(FD.load(tgt, "train"), 300), FD.load(tgt, "test")
    def logit(S):
        p = predict(net, S)[:, 1]; return np.log(p + 1e-6) - np.log(1 - p + 1e-6), p
    (lA, _), (lB, pB) = logit(A), logit(B)
    f_net = CP.event_f1(pB > 0.5, B)["ev_f1"]
    f_hmm = CP.event_f1(hmm_decode(lA, np.isfinite(A.pos).all(2), lB, np.isfinite(B.pos).all(2)), B)["ev_f1"]
    print(f"[{tgt}] foundation net (never saw {tgt}): threshold 0.5 event F1 {f_net:.3f} | + HMM decoding {f_hmm:.3f}", flush=True)
