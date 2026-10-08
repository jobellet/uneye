#!/usr/bin/env python3
"""Export a trained causal TCN for the C++ pipeline: <out>.bin (streaming native engine, magic UNCZ), <out>.onnx
(window version [1,2,W] -> [1,C,W], take the last column), a golden test file for cpp/tests and <out>.json (label delay).

Works with the model of the first study (checkpoint {"state": ..., "channels": ...}) and with archs.SaccadeNet('tcn') models of
the notebook (`export_saccadenet(model, out, lookahead_samples)`; the light TCN and GRU are not supported by the C++ engine).
"""
import argparse, json, struct
import numpy as np, torch
from causal_net import CausalTCN
from common import load, velocity


def to_causal_tcn(model):
    """archs.SaccadeNet with a (non-lite) TCN backbone -> causal_net.CausalTCN with the same weights"""
    bb = model.backbone
    assert hasattr(bb, "blocks") and not isinstance(bb.blocks[0].conv, torch.nn.Sequential), "only the full TCN is supported by the C++ engine"
    dil = tuple(b.conv.conv.dilation[0] for b in bb.blocks)
    net = CausalTCN(model.head.out_channels, bb.channels, ks=bb.blocks[0].conv.conv.kernel_size[0], dilations=dil, stem_ks=bb.stem.conv.kernel_size[0])
    sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in model.state_dict().items()}
    net.load_state_dict(sd)
    return net.cpu().eval()


def export(net, out, window=200, golden=None, lookahead_samples=0):
    sd = net.state_dict()
    with open(out + ".bin", "wb") as f:
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
    json.dump({"label_delay_samples": int(lookahead_samples), "receptive_field": net.receptive_field,
               "note": "run uneye_rt with --label-delay <label_delay_samples>"}, open(out + ".json", "w"))
    torch.onnx.export(net, torch.zeros(1, 2, window), out + ".onnx", input_names=["dxy"], output_names=["prob"], opset_version=13,
                      dynamo=False, dynamic_axes={"dxy": {0: "batch", 2: "time"}, "prob": {0: "batch", 2: "time"}})
    X, Y, L, fs = load("1", "B")
    V = velocity(X[:1], Y[:1])[:, :, :400]
    ref = net(torch.from_numpy(V)).detach().numpy()
    try:  # the check is optional: the .bin / .onnx files are already written
        import onnxruntime as ort
        got = ort.InferenceSession(out + ".onnx", providers=["CPUExecutionProvider"]).run(None, {"dxy": V})[0]
        print("receptive field", net.receptive_field, "samples; max |onnx - torch| =", np.abs(ref - got).max())
    except ImportError:
        print("receptive field", net.receptive_field, "samples; (pip install onnxruntime to also check the ONNX file)")
    if golden:
        with open(golden, "w") as g:
            g.write("%d %d\n" % (V.shape[2], net.classes))
            for t in range(V.shape[2]): g.write("%.9g %.9g\n" % (V[0, 0, t], V[0, 1, t]))
            for c in range(net.classes): g.write(" ".join("%.9g" % q for q in ref[0, c]) + "\n")


def export_saccadenet(model, out, lookahead_samples=0, window=200, golden=None):
    return export(to_causal_tcn(model), out, window, golden, lookahead_samples)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("out")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--lookahead-samples", type=int, default=0)
    ap.add_argument("--golden", default="../cpp/tests/golden_causal.txt")
    a = ap.parse_args()
    ck = torch.load(a.weights, weights_only=False)
    net = CausalTCN(2, ck["channels"], dilations=ck.get("dilations", (1, 2, 4, 8, 16))); net.load_state_dict(ck["state"]); net.eval()
    export(net, a.out, a.window, a.golden, a.lookahead_samples)
