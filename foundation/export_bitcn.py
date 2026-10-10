"""Export a trained BiTCN (foundation/runs/<name>.pt, "net" weights) for C++: ONNX (dynamic length), a flat float32 .bin for the native engine
(cpp/src/bitcn.cpp, batch-norm folded to scale / shift) and a golden file (raw positions, 2-channel features, logits) for cpp/tests/test_bitcn.cpp.
python foundation/export_bitcn.py --name sup_bitcn_v2 --inputs 2
.bin layout (little endian): int32 magic 0x42544e32, nin, ch, nblocks, k_stem; then stem w[ch][nin][k] b[ch] scale[ch] shift[ch]; per block int32 dilation,
conv w[ch][ch][3] b[ch] scale[ch] shift[ch]; head w[ch] b.  Block: x + scale * relu(conv(x)) + shift; stem: scale * relu(conv) + shift.
"""
import argparse, os, struct, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from foundation import bitcn as BT, dino1d as DN, ssl_probe as SP


def bn_fold(bn):
    s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach(); return s.numpy(), (bn.bias - bn.running_mean * s).detach().numpy()


def write_bin(net, nin, path):
    f = lambda a: np.ascontiguousarray(a, np.float32).tobytes()
    with open(path, "wb") as o:
        o.write(struct.pack("<5i", 0x42544e32, nin, net.stem.out_channels, len(net.blocks), net.stem.kernel_size[0]))
        s, t = bn_fold(net.stem_bn); o.write(f(net.stem.weight.detach().numpy()) + f(net.stem.bias.detach().numpy()) + f(s) + f(t))
        for b in net.blocks:
            s, t = bn_fold(b.bn); o.write(struct.pack("<i", b.conv.dilation[0]) + f(b.conv.weight.detach().numpy()) + f(b.conv.bias.detach().numpy()) + f(s) + f(t))
        o.write(f(net.head.weight.detach().numpy()) + f(net.head.bias.detach().numpy()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="sup_bitcn_v2"); ap.add_argument("--inputs", type=int, default=2); a = ap.parse_args()
    ff = DN.feats2 if a.inputs == 2 else DN.feats
    net = BT.BiTCN(nin=a.inputs); net.load_state_dict(torch.load(os.path.join(ROOT, "foundation", "runs", f"{a.name}.pt"), map_location="cpu", weights_only=False)["net"]); net.eval()
    out = os.path.join(ROOT, "cpp", "models", f"bitcn_{a.name}.onnx")
    torch.onnx.export(net, torch.zeros(1, 2000, a.inputs), out, input_names=["feats"], output_names=["logit"], dynamic_axes={"feats": {0: "b", 1: "t"}, "logit": {0: "b", 1: "t"}}, opset_version=17, dynamo=False)
    write_bin(net, a.inputs, os.path.join(ROOT, "cpp", "models", f"bitcn_{a.name}.bin"))
    import onnxruntime as ort
    pos, lab = SP.val_sets()["d1"]; pos = pos[:4, :3000].copy(); x = ff(pos, 1000.0).astype(np.float32)
    with torch.no_grad(): ref = net(torch.as_tensor(x)).numpy()
    got = ort.InferenceSession(out).run(None, {"feats": x})[0]; print(out, os.path.getsize(out) // 1024, "kB, max |onnx - torch| =", float(np.abs(ref - got).max()))
    g = pos[0, :1500].copy(); g[400:430] = np.nan; g[900:905] = np.nan           # a blink and a short dropout test the interpolation
    xg = ff(g[None], 1000.0)[0]
    with torch.no_grad(): lg = net(torch.as_tensor(xg[None])).numpy()[0]
    with open(os.path.join(ROOT, "cpp", "tests", f"golden_bitcn_{a.inputs}ch.txt"), "w") as o:
        o.write(f"{len(g)} {a.inputs}\n"); o.write(" ".join("nan" if not np.isfinite(v) else f"{v:.9g}" for v in g.reshape(-1)) + "\n")
        o.write(" ".join(f"{v:.9g}" for v in xg.reshape(-1)) + "\n"); o.write(" ".join(f"{v:.9g}" for v in lg) + "\n")
    print("golden written; logits range", lg.min(), lg.max())
