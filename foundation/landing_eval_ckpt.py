#!/usr/bin/env python3
"""Reload the saved checkpoints of foundation/landing_train.py and score them on the Andersson test split (mean landing error, deg). python foundation/landing_eval_ckpt.py"""
import os, sys, torch, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
from foundation import data as FD, unet_ssl as U
import foundation.landing_train as L
te = L.Split(FD.load("andersson", "test")); print("test rows", len(te.R))
for tag in ("", "_more"):
    for name in ("tcn", "gru", "transformer", "mlp", "unet_pre", "unet_scratch"):
        ck = torch.load(os.path.join(U.RUNS, f"landing_{name}{tag}.pt"), map_location=L.DEV, weights_only=False)
        net = L.make(name); net.load_state_dict(ck["state"]); e = L.evaluate(net, name, te)
        print(f"{name:13s}{tag or '       '}  {e.mean():.3f} deg   median {np.median(e):.3f}")
