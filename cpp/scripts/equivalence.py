#!/usr/bin/env python3
"""Step 2 of docs/ROADMAP.md: Python (PyTorch) vs C++ numerical equivalence of the causal TCN, at the OUTPUT (p(saccade) per sample).

The weights are read back from cpp/models/<model>.bin (the file the C++ engine uses), so both sides run exactly the same numbers.
References: the same network in PyTorch float32 and float64 (`.double()`), whole sequence at once (convolutions), while C++ runs it
as a streaming filter, one step per sample (cpp/build/causal_dump). Inputs (fixed, no randomness except a seeded generator):
  real     velocities of set B, datasets 1 and 2 (100 trials each, 1000 samples)
  stress   zeros, constant large velocity (5 deg/sample), tiny values (1e-30, denormal range after scaling), white noise
Metrics per pair (C++ vs Py32, C++ vs Py64, Py32 vs Py64): max / p99.9 absolute error, max relative error (where |ref| > 1e-6),
ULP distance in float32, share of samples with a different label at 0.5.
NOT covered: layer-by-layer comparison (the C++ engine has no trace hook yet), the U-Net window network (see test_parity).

Run from the repository root:  python cpp/scripts/equivalence.py [--model causal]
Writes docs/slides/data/equivalence_<model>.npz (errors for Figure 1) and prints the summary.
"""
import argparse, os, struct, subprocess, sys
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "online"))
from causal_net import CausalTCN  # noqa: E402


def load_bin(path):
    """inverse of online/export_causal.py::export"""
    b = open(path, "rb").read()
    assert b[:4] == b"UNCZ", "not a causal TCN file"
    ver, classes, channels, ks, stem_ks, nd, in_ch, _ = struct.unpack_from("<8i", b, 4)
    off = 4 + 32
    dil = struct.unpack_from("<%di" % nd, b, off); off += 4 * nd
    net = CausalTCN(classes, channels, ks=ks, dilations=dil, stem_ks=stem_ks, in_ch=in_ch)
    sd = net.state_dict()
    def take(key, shape=None):
        nonlocal off
        t = sd[key]; n = t.numel()
        a = np.frombuffer(b, "<f4", n, off).copy(); off += 4 * n
        sd[key] = torch.from_numpy(a.reshape(shape or tuple(t.shape)))
    def bn(prefix):
        for k in ("weight", "bias", "running_mean", "running_var"): take(prefix + k)
    take("stem.conv.weight"); take("stem.conv.bias"); bn("stem_bn.")
    for i in range(nd):
        take(f"blocks.{i}.conv.conv.weight"); take(f"blocks.{i}.conv.conv.bias"); bn(f"blocks.{i}.bn.")
    take("head.weight", (classes, channels)); sd["head.weight"] = sd["head.weight"][:, :, None]; take("head.bias")
    assert off == len(b), f"{len(b) - off} bytes left"
    net.load_state_dict(sd); return net.eval()


def inputs():
    old = os.getcwd(); os.chdir(os.path.join(ROOT, "online"))          # data paths in common.py are relative to online/
    from common import load, velocity
    seqs, kinds = [], []
    for s in ("1", "2"):
        X, Y, L, fs = load(s, "B")
        V = velocity(X[:100], Y[:100]).astype(np.float32)               # (100, 2, T)
        seqs.append(V); kinds += ["real"] * len(V)
    os.chdir(old)
    T = seqs[0].shape[2]
    rng = np.random.default_rng(2)
    stress = [np.zeros((2, T)), np.full((2, T), 5.0), np.full((2, T), 1e-30), rng.normal(0, 0.05, (2, T)), rng.normal(0, 1.0, (2, T))]
    seqs.append(np.stack(stress).astype(np.float32)); kinds += ["zeros", "large", "tiny", "noise", "loud noise"]
    return np.nan_to_num(np.concatenate(seqs), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), np.array(kinds)


def ulp32(a, b):
    """distance in float32 units in the last place (a, b float32 arrays)"""
    ia = a.astype(np.float32).view(np.int32).astype(np.int64); ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-2**31) - ia, ia); ib = np.where(ib < 0, np.int64(-2**31) - ib, ib)
    return np.abs(ia - ib)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="causal"); a = ap.parse_args()
    V, kinds = inputs()
    n, _, T = V.shape
    inp, outp = os.path.join(ROOT, "cpp", "build", "eq_in.bin"), os.path.join(ROOT, "cpp", "build", "eq_out.bin")
    with open(inp, "wb") as f:
        f.write(struct.pack("<2i", n, T)); f.write(np.ascontiguousarray(V.transpose(0, 2, 1)).astype("<f4").tobytes())
    model = os.path.join(ROOT, "cpp", "models", a.model + ".bin")
    subprocess.run([os.path.join(ROOT, "cpp", "build", "causal_dump"), model, inp, outp], check=True)
    cpp = np.fromfile(outp, "<f4")[2:].reshape(n, T)
    net = load_bin(model)
    with torch.no_grad():
        p32 = net(torch.from_numpy(V))[:, 1].numpy()
        p64 = net.double()(torch.from_numpy(V).double())[:, 1].numpy()
    pairs = {"C++ vs Py32": (cpp, p32), "C++ vs Py64": (cpp, p64), "Py32 vs Py64": (p32, p64)}
    print(f"model {a.model}: {n} sequences x {T} samples ({(kinds == 'real').sum()} real, {(kinds != 'real').sum()} stress)")
    out = {}
    for name, (x, r) in pairs.items():
        e = np.abs(x.astype(np.float64) - r.astype(np.float64)); m = np.abs(r) > 1e-6
        rel = e[m] / np.abs(r[m]); u = ulp32(x, r)
        lab = ((x >= 0.5) != (r >= 0.5)).mean()
        print(f"  {name:13s} abs max {e.max():.2e}  p99.9 {np.percentile(e, 99.9):.2e} | rel max {rel.max():.2e} | ULP max {u.max()} median {np.median(u):.0f} | labels differ {100 * lab:.4f} %")
        for k in np.unique(kinds):
            if k != "real": print(f"      {k:11s} abs max {e[kinds == k].max():.2e}")
        out[name.replace(" ", "_").replace("+", "p")] = e.ravel().astype(np.float32)
    # golden file for the ctest test_equivalence: 10 real + the 5 stress sequences, inputs (float32) and the float64 reference
    sel = np.r_[np.arange(10), np.where(kinds != "real")[0]]
    with open(os.path.join(ROOT, "cpp", "tests", f"golden_equivalence_{a.model}.bin"), "wb") as f:
        f.write(struct.pack("<2i", len(sel), T)); f.write(np.ascontiguousarray(V[sel].transpose(0, 2, 1)).astype("<f4").tobytes())
        f.write(p64[sel].astype("<f8").tobytes())
    dst = os.path.join(ROOT, "docs", "slides", "data", f"equivalence_{a.model}.npz")
    np.savez_compressed(dst, **out); print("wrote", dst)


if __name__ == "__main__":
    main()
