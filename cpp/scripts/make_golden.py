#!/usr/bin/env python3
"""Reference outputs from the ORIGINAL PyTorch pipeline, used by tests/test_parity.cpp.
Writes tests/golden_<model>.txt : window dxy (W lines 'dx dy') followed by C lines of W probabilities.
"""
import os, sys, json
import numpy as np, torch
root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, root)
from uneye.functions import UNet

for name, wfile, data in [("combined", "weights_1+2+3", "dataset1/dataset1_1000hz_%s_setA.csv"),
                          ("dataset1", "weights_dataset1", "dataset1/dataset1_1000hz_%s_setA.csv"),
                          ("andersson", "weights_Andersson", "dataset1/dataset1_1000hz_%s_setA.csv")]:
    meta = json.load(open(os.path.join(root, "cpp/models", name + ".json")))
    W = meta["window"]
    sd = torch.load(os.path.join(root, "training", wfile), map_location="cpu", weights_only=False)
    net = UNet(sd["c7.weight"].shape[0], 5, 5); net.load_state_dict(sd); net.eval()
    X = np.loadtxt(os.path.join(root, "data", data % "X"), delimiter=",")[3]
    Y = np.loadtxt(os.path.join(root, "data", data % "Y"), delimiter=",")[3]
    dx = np.r_[0, np.diff(X)]; dy = np.r_[0, np.diff(Y)]
    best = int(np.argmax(np.abs(dx[:800]) + np.abs(dy[:800])))  # window around largest velocity
    s = max(0, min(best - 80, len(X) - W))
    v = np.stack([dx[s:s + W], dy[s:s + W]], 1).astype(np.float32)
    with torch.no_grad():
        p = net(torch.from_numpy(v)[None, None], ["out"])[0][0].numpy()
    with open(os.path.join(root, "cpp/tests", "golden_%s.txt" % name), "w") as f:
        f.write("%d %d\n" % (W, p.shape[0]))
        for r in v: f.write("%.9g %.9g\n" % tuple(r))
        for c in p: f.write(" ".join("%.9g" % q for q in c) + "\n")
    print(name, "max saccade prob in window", 1 - p[0].min())
