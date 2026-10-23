#!/usr/bin/env python3
"""Convert a trained U'n'Eye C++ .bin weight file back to a PyTorch state_dict.

Inverse of export_model.py.
Usage:
    python cpp/scripts/bin_to_state_dict.py cpp/models/combined.bin output_weights.pt
"""
import argparse
import json
import os
import struct
import sys
import torch
import numpy as np

BIN_ORDER = [
    "c0.0", "c0.2", "c1.0", "c1.2", "c2.0", "c2.2", "c3.0", "c3.2",
    "up1.0", "up1.2", "c4.0", "c4.2", "up2.0", "up2.2", "c5.0", "c5.2",
    "c6.0", "c6.2", "c7",
]

def load_bin_to_state_dict(bin_path):
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"UNEY":
            raise ValueError(f"Not a U'n'Eye model file (invalid magic: {magic})")
        version, classes, ks, mp = struct.unpack("<iiii", f.read(16))
        
        shapes = {
            "c0.0": [("weight", (10, 1, ks, 2)), ("bias", (10,))],
            "c0.2": [("weight", (10,)), ("bias", (10,)), ("running_mean", (10,)), ("running_var", (10,))],
            "c1.0": [("weight", (20, 10, ks)), ("bias", (20,))],
            "c1.2": [("weight", (20,)), ("bias", (20,)), ("running_mean", (20,)), ("running_var", (20,))],
            "c2.0": [("weight", (20, 20, ks)), ("bias", (20,))],
            "c2.2": [("weight", (20,)), ("bias", (20,)), ("running_mean", (20,)), ("running_var", (20,))],
            "c3.0": [("weight", (20, 20, ks)), ("bias", (20,))],
            "c3.2": [("weight", (20,)), ("bias", (20,)), ("running_mean", (20,)), ("running_var", (20,))],
            "up1.0": [("weight", (20, 20, mp)), ("bias", (20,))],
            "up1.2": [("weight", (20,)), ("bias", (20,)), ("running_mean", (20,)), ("running_var", (20,))],
            "c4.0": [("weight", (20, 40, ks)), ("bias", (20,))],
            "c4.2": [("weight", (20,)), ("bias", (20,)), ("running_mean", (20,)), ("running_var", (20,))],
            "up2.0": [("weight", (20, 20, mp)), ("bias", (20,))],
            "up2.2": [("weight", (20,)), ("bias", (20,)), ("running_mean", (20,)), ("running_var", (20,))],
            "c5.0": [("weight", (20, 40, ks)), ("bias", (20,))],
            "c5.2": [("weight", (20,)), ("bias", (20,)), ("running_mean", (20,)), ("running_var", (20,))],
            "c6.0": [("weight", (10, 20, ks)), ("bias", (10,))],
            "c6.2": [("weight", (10,)), ("bias", (10,)), ("running_mean", (10,)), ("running_var", (10,))],
            "c7": [("weight", (classes, 10, 1)), ("bias", (classes,))],
        }

        sd = {}
        for key in BIN_ORDER:
            for param_name, shape in shapes[key]:
                full_name = f"{key}.{param_name}" if key != "c7" else f"c7.{param_name}"
                numel = 1
                for s in shape:
                    numel *= s
                raw = f.read(numel * 4)
                if len(raw) < numel * 4:
                    raise ValueError(f"Unexpected end of file while reading {full_name}")
                arr = np.frombuffer(raw, dtype="<f4").reshape(shape)
                sd[full_name] = torch.from_numpy(arr.copy())
            if key.endswith(".2"):
                sd[f"{key}.num_batches_tracked"] = torch.tensor(0, dtype=torch.int64)

        extra = f.read()
        if len(extra) > 0:
            print(f"Warning: {len(extra)} trailing bytes in {bin_path}", file=sys.stderr)

        meta = {"version": version, "classes": classes, "ks": ks, "mp": mp}

        # Check for sidecar JSON
        json_path = os.path.splitext(bin_path)[0] + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    sidecar = json.load(jf)
                    meta["sidecar"] = sidecar
                    print(f"Loaded sidecar metadata from {json_path}:")
                    print(f"  Rate: {sidecar.get('rate_hz', 'unknown')} Hz")
                    print(f"  Deg/unit: {sidecar.get('deg_per_unit', 1.0)}")
                    print(f"  Speed limit: {sidecar.get('max_speed_deg_s', 1300.0)} deg/s")
                    if "preprocessing" in sidecar:
                        print(f"  Preprocessing: {sidecar['preprocessing']}")
            except Exception as e:
                print(f"Notice: Could not parse sidecar {json_path}: {e}", file=sys.stderr)

        return sd, meta

def main():
    parser = argparse.ArgumentParser(description="Convert .bin U'n'Eye model to PyTorch state dict")
    parser.add_argument("bin_path", help="Path to input .bin file")
    parser.add_argument("out_path", help="Path to output PyTorch weights (.pt)")
    args = parser.parse_args()

    sd, meta = load_bin_to_state_dict(args.bin_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    torch.save(sd, args.out_path)
    
    # Save accompanying metadata sidecar
    out_json = os.path.splitext(args.out_path)[0] + "_config.json"
    sidecar_data = meta.get("sidecar", {
        "classes": meta["classes"],
        "ks": meta["ks"],
        "mp": meta["mp"],
        "max_speed_deg_s": 1300.0
    })
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(sidecar_data, jf, indent=2)

    print(f"Converted {args.bin_path} -> {args.out_path} (classes: {meta['classes']}, ks: {meta['ks']}, mp: {meta['mp']})")
    print(f"Saved configuration to {out_json}")

if __name__ == "__main__":
    main()
