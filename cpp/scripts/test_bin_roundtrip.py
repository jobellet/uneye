#!/usr/bin/env python3
"""Test roundtrip conversion between PyTorch weights and .bin format.

Tests that:
1. Exporting PyTorch weights to .bin (via export_model logic)
2. Converting .bin back to PyTorch state dict (via bin_to_state_dict)
3. Results in identical tensor weights, biases, and running stats.
4. The converted state dict can be loaded by uneye.functions.UNet without warning.
"""
import os
import sys
import tempfile
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from uneye.functions import UNet
from bin_to_state_dict import load_bin_to_state_dict

def test_roundtrip(weight_path):
    print(f"Testing roundtrip on {weight_path}...")
    orig_sd = torch.load(weight_path, map_location="cpu", weights_only=False)
    classes = orig_sd["c7.weight"].shape[0]
    net_orig = UNet(classes, 5, 5)
    net_orig.load_state_dict(orig_sd)
    net_orig.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_bin = os.path.join(tmpdir, "model.bin")
        # Run export logic
        BIN_ORDER = [
            "c0.0", "c0.2", "c1.0", "c1.2", "c2.0", "c2.2", "c3.0", "c3.2",
            "up1.0", "up1.2", "c4.0", "c4.2", "up2.0", "up2.2", "c5.0", "c5.2",
            "c6.0", "c6.2", "c7",
        ]
        with open(tmp_bin, "wb") as f:
            f.write(b"UNEY")
            import struct
            f.write(struct.pack("<iiii", 1, classes, 5, 5))
            for key in BIN_ORDER:
                if key == "c7":
                    tensors = [orig_sd["c7.weight"], orig_sd["c7.bias"]]
                elif key.endswith(".0"):
                    tensors = [orig_sd[key + ".weight"], orig_sd[key + ".bias"]]
                else:
                    tensors = [orig_sd[key + ".weight"], orig_sd[key + ".bias"],
                               orig_sd[key + ".running_mean"], orig_sd[key + ".running_var"]]
                for t in tensors:
                    f.write(t.detach().cpu().numpy().astype("<f4").tobytes())

        # Load back via bin_to_state_dict
        loaded_sd, meta = load_bin_to_state_dict(tmp_bin)
        assert meta["classes"] == classes
        assert meta["ks"] == 5
        assert meta["mp"] == 5

        # Check all tensors
        net_loaded = UNet(classes, 5, 5)
        net_loaded.load_state_dict(loaded_sd)
        net_loaded.eval()

        # Compare outputs on dummy input
        x = torch.randn(2, 1, 200, 2)
        out_orig = net_orig(x, ["out"])[0].detach().numpy()
        out_loaded = net_loaded(x, ["out"])[0].detach().numpy()
        max_diff = np.max(np.abs(out_orig - out_loaded))
        print(f"Max difference in model output: {max_diff:.2e}")
        assert max_diff < 1e-6, f"Mismatch too large: {max_diff}"

        # Compare each tensor directly
        for k in orig_sd.keys():
            if k.endswith("num_batches_tracked"):
                continue
            t_orig = orig_sd[k].detach().cpu().numpy()
            t_loaded = loaded_sd[k].numpy()
            t_diff = np.max(np.abs(t_orig - t_loaded))
            assert t_diff == 0.0, f"Tensor {k} differs by {t_diff}"

        print(f"PASS: {weight_path} roundtrip is exact (diff = 0.0)")

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    test_roundtrip(os.path.join(root, "training", "weights_1+2+3"))
    test_roundtrip(os.path.join(root, "training", "weights_Andersson"))
    print("\nAll roundtrip tests PASSED successfully!")

if __name__ == "__main__":
    main()
