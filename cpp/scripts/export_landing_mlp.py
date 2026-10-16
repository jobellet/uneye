#!/usr/bin/env python3
"""Export the landing MLP trained by foundation/landing_gazebase.py (--ctx, model mlp) to cpp/models/landing_mlp_<name>.bin (cpp/app/engine/include/eyelive/landing_net.hpp) and write a golden test file
(cpp/app/engine/tests/data/landing_mlp_<name>_golden.bin: 24 windows of the GazeBase test split and the outputs of the PyTorch network).   python cpp/scripts/export_landing_mlp.py <checkpoint.pt> <name>"""
import os, struct, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import foundation.landing_train as L
import foundation.landing_gazebase as G
from foundation import unet_ssl as U
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); sd = ck["state"]; NAME = sys.argv[2]            # e.g. landing_pooled_mlp_drone_only.pt drone
L.CTX = True; L.NIN = 15; net = L.make("mlp").cpu(); net.load_state_dict(sd); net.eval(); Lw = net.L; w1, b1, w2, b2, w3, b3 = [sd[k].cpu().numpy().astype(np.float32) for k in ("f.1.weight", "f.1.bias", "f.3.weight", "f.3.bias", "f.5.weight", "f.5.bias")]
out = os.path.join(ROOT, "cpp", "models", f"landing_mlp_{NAME}.bin")
with open(out, "wb") as f:
    f.write(b"EYLNET01"); f.write(struct.pack("<6I", 1, Lw, 15, w1.shape[0], w2.shape[0], 3))
    for a in (w1, b1, w2, b2, w3, b3): f.write(np.ascontiguousarray(a).tobytes())
print("wrote", out, os.path.getsize(out) // 1024, "KB; L =", Lw, "hidden", w1.shape[0], w2.shape[0])
# golden cases from the test split of GazeBase RAN (same split as landing_gazebase.py)
z = np.load(os.path.join(ROOT, "data", "GazeBase", "ran_saccades.npz")); subj = z["subj"] % 1000; us = np.unique(subj); rng = np.random.RandomState(0); rng.shuffle(us)
nt, nv = int(0.8 * len(us)), int(0.1 * len(us)); test_users = us[nt + nv:]; idx = np.sort(np.random.RandomState(3).choice(np.where(np.isin(subj, test_users))[0], 400, replace=False)); sp = G.build(z, idx)
pick = np.random.RandomState(5).choice(len(sp.R), 24, replace=False); dev = L.DEV; L.DEV = "cpu"
with open(os.path.join(ROOT, "cpp", "app", "engine", "tests", "data", f"landing_mlp_{NAME}_golden.bin"), "wb") as f:
    f.write(struct.pack("<2I", len(pick), Lw))
    for r in pick:
        b = sp.batch(np.array([r])); x = L.seq_input(sp, b, L=Lw)
        with torch.no_grad(): o = net(x).numpy()[0]
        i, on, off, t = sp.R[r]; tp = t + L.PAD; P = sp.P[i, tp - Lw:tp + 1].astype(np.float32); f.write(struct.pack("<i", int(on + L.PAD - (tp - Lw)))); f.write(P.tobytes()); f.write(o.astype(np.float32).tobytes())
print("golden cases", len(pick))
