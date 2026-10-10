"""Export a trained BiTCN (foundation/runs/<name>.pt) to ONNX (dynamic length, input = dino1d.feats 8 channels) and check it against PyTorch.
Python-side features (dino1d.feats) must be reproduced in C++ before the ONNX file can be used there.   python foundation/export_bitcn.py --name sup_bitcn_b
"""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from foundation import bitcn as BT, dino1d as DN, ssl_probe as SP

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="sup_bitcn_b"); a = ap.parse_args()
    net = BT.BiTCN(); net.load_state_dict(torch.load(os.path.join(ROOT, "foundation", "runs", f"{a.name}.pt"), map_location="cpu", weights_only=False)["net"]); net.eval()
    out = os.path.join(ROOT, "cpp", "models", f"bitcn_{a.name}.onnx")
    torch.onnx.export(net, torch.zeros(1, 2000, 8), out, input_names=["feats"], output_names=["logit"], dynamic_axes={"feats": {0: "b", 1: "t"}, "logit": {0: "b", 1: "t"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(out); pos, lab = SP.val_sets()["d1"]; x = DN.feats(pos[:4, :3000], 1000.0).astype(np.float32)
    with torch.no_grad(): ref = net(torch.as_tensor(x)).numpy()
    got = sess.run(None, {"feats": x})[0]; print(out, os.path.getsize(out) // 1024, "kB, max |onnx - torch| =", float(np.abs(ref - got).max()))
