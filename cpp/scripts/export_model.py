#!/usr/bin/env python3
"""Export a trained U'n'Eye PyTorch weight file for the C++ real-time pipeline.

Writes, next to each other:
  <out>.onnx   fixed-window ONNX model, input  [1,1,W,2] (dX,dY), output [1,C,W] (softmax)
  <out>.bin    raw float32 weights for the dependency-free native C++ engine
  <out>.json   metadata (window, classes, ks, mp)

It also checks ONNX Runtime output against PyTorch.

Usage: export_model.py ../../training/weights_Andersson models/andersson --window 200
"""
import argparse, json, os, struct, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# functions.py imports skimage/sklearn/scipy at module level; they are only
# needed for training, so import the UNet class directly if they are missing.
from uneye.functions import UNet  # noqa: E402

# Order of tensors in the .bin file (must match cpp/src/native_engine.cpp)
BIN_ORDER = [
    "c0.0", "c0.2", "c1.0", "c1.2", "c2.0", "c2.2", "c3.0", "c3.2",
    "up1.0", "up1.2", "c4.0", "c4.2", "up2.0", "up2.2", "c5.0", "c5.2",
    "c6.0", "c6.2", "c7",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=200, help="samples; multiple of mp**2")
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--mp", type=int, default=5)
    a = ap.parse_args()
    assert a.window % (a.mp ** 2) == 0, "window must be a multiple of mp**2"

    sd = torch.load(a.weights, map_location="cpu", weights_only=False)
    classes = sd["c7.weight"].shape[0]
    net = UNet(classes, a.ks, a.mp)
    net.load_state_dict(sd)
    net.eval()

    class Wrap(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            return self.n(x, ["out"])[0]

    m = Wrap(net).eval()
    dummy = torch.randn(1, 1, a.window, 2) * 0.1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.onnx.export(m, dummy, a.out + ".onnx", input_names=["dxy"],
                      output_names=["prob"], opset_version=13, dynamo=False,
                      dynamic_axes={"dxy": {0: "batch"}, "prob": {0: "batch"}})

    # raw weights
    with open(a.out + ".bin", "wb") as f:
        f.write(b"UNEY")
        f.write(struct.pack("<iiii", 1, classes, a.ks, a.mp))
        for key in BIN_ORDER:
            if key == "c7":
                tensors = [sd["c7.weight"], sd["c7.bias"]]
            elif key.endswith(".0"):
                tensors = [sd[key + ".weight"], sd[key + ".bias"]]
            else:  # batchnorm
                tensors = [sd[key + ".weight"], sd[key + ".bias"],
                           sd[key + ".running_mean"], sd[key + ".running_var"]]
            for t in tensors:
                f.write(t.detach().cpu().numpy().astype("<f4").tobytes())
    json.dump({"window": a.window, "classes": classes, "ks": a.ks, "mp": a.mp,
               "source": os.path.basename(a.weights)}, open(a.out + ".json", "w"))

    # verify
    import onnxruntime as ort
    s = ort.InferenceSession(a.out + ".onnx", providers=["CPUExecutionProvider"])
    x = np.random.randn(4, 1, a.window, 2).astype(np.float32) * 0.2
    x[:, :, 80:90, :] += 1.0
    ref = m(torch.from_numpy(x)).detach().numpy()
    got = s.run(None, {"dxy": x})[0]
    print("classes", classes, "max |onnx - torch| =", np.abs(ref - got).max())


if __name__ == "__main__":
    main()
