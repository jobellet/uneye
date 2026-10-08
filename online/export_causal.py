#!/usr/bin/env python3
"""Export the trained causal net for C++: <out>.bin (streaming native engine, magic UNCZ), <out>.onnx
(window version [1,2,W] -> [1,C,W], take the last column) and a golden test file for cpp/tests."""
import argparse, os, struct, sys
import numpy as np, torch
from causal_net import CausalTCN
from common import load, velocity

ap = argparse.ArgumentParser()
ap.add_argument("weights"); ap.add_argument("out")
ap.add_argument("--window", type=int, default=200)
ap.add_argument("--golden", default="../cpp/tests/golden_causal.txt")
a = ap.parse_args()
ck = torch.load(a.weights, weights_only=False)
net = CausalTCN(2, ck["channels"], dilations=ck.get("dilations", (1, 2, 4, 8, 16))); net.load_state_dict(ck["state"]); net.eval()
sd = net.state_dict()

with open(a.out + ".bin", "wb") as f:
    f.write(b"UNCZ")
    f.write(struct.pack("<8i", 1, net.classes, net.channels, net.ks, net.stem_ks, len(net.dilations), net.in_ch, 0))
    f.write(struct.pack("<%di" % len(net.dilations), *net.dilations))
    def w(t): f.write(t.detach().cpu().numpy().astype("<f4").tobytes())
    def bn(prefix):
        for k in ("weight", "bias", "running_mean", "running_var"): w(sd[prefix + k])
    w(sd["stem.conv.weight"]); w(sd["stem.conv.bias"]); bn("stem_bn.")
    for i in range(len(net.dilations)):
        w(sd[f"blocks.{i}.conv.conv.weight"]); w(sd[f"blocks.{i}.conv.conv.bias"]); bn(f"blocks.{i}.bn.")
    w(sd["head.weight"][:, :, 0]); w(sd["head.bias"])

dummy = torch.zeros(1, 2, a.window)
torch.onnx.export(net, dummy, a.out + ".onnx", input_names=["dxy"], output_names=["prob"], opset_version=13,
                  dynamo=False, dynamic_axes={"dxy": {0: "batch", 2: "time"}, "prob": {0: "batch", 2: "time"}})
import onnxruntime as ort
X, Y, L, fs = load("1", "B")
V = velocity(X[:1], Y[:1])[:, :, :400]
ref = net(torch.from_numpy(V)).detach().numpy()
got = ort.InferenceSession(a.out + ".onnx", providers=["CPUExecutionProvider"]).run(None, {"dxy": V})[0]
print("receptive field", net.receptive_field, "samples; max |onnx - torch| =", np.abs(ref - got).max())
with open(a.golden, "w") as g:
    g.write("%d %d\n" % (V.shape[2], net.classes))
    for t in range(V.shape[2]): g.write("%.9g %.9g\n" % (V[0, 0, t], V[0, 1, t]))
    for c in range(net.classes): g.write(" ".join("%.9g" % q for q in ref[0, c]) + "\n")
