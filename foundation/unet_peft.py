#!/usr/bin/env python3
"""Parameter-efficient adaptation of the pre-trained U-Net to saccade detection with few labels (dataset 1, N = 10 / 20 / 50, 10 draws, same trials and test set as unet_ssl.py):
  lora     LoRA (Hu et al. 2022): every conv / transposed-conv weight W becomes W + B A (rank 4, B = 0 at start), W frozen, only A, B (+ the new head) trained; BN frozen
  bitfit   BitFit (Ben Zaken et al. 2022): only the conv biases (+ head) trained
  bn_only  only the batch-norm scale and shift (+ head) trained (normalisation tuning)
  lp_ft    linear probing then fine-tuning (Kumar et al. 2022): the new head alone for 100 steps, then all layers
Reference rows from unet_ssl.py / unet_layers.py: all layers (full fine-tuning), U'n'Eye.   python foundation/unet_peft.py [--reps 10]
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn as nn
from torch.nn.utils import parametrize
from foundation import data as FD, compare as CP
from foundation.unet_ssl import score, RUNS
from foundation.unet_layers import ft, BLOCKS, NIGHT


class LoRA(nn.Module):
    def __init__(self, shape, device, r=4, alpha=8.0):
        super().__init__(); n_out, rest = shape[0], int(np.prod(shape[1:])); self.shape = shape; self.A = nn.Parameter(torch.randn(r, rest, device=device) / np.sqrt(rest)); self.B = nn.Parameter(torch.zeros(n_out, r, device=device)); self.s = alpha / r
    def forward(self, W): return W + (self.B @ self.A).reshape(self.shape) * self.s


def prep_lora(net):
    for p in net.parameters(): p.requires_grad = False
    for b in BLOCKS:
        for m in getattr(net, b).modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)): parametrize.register_parametrization(m, "weight", LoRA(tuple(m.weight.shape), m.weight.device))
    for n, p in net.named_parameters():
        if n.startswith("head") or n.endswith(".A") or n.endswith(".B"): p.requires_grad = True
    return [getattr(net, b) for b in BLOCKS]


def prep_bias(net, which):
    for p in net.parameters(): p.requires_grad = False
    for n, p in net.named_parameters():
        if n.startswith("head"): p.requires_grad = True
        elif which == "bitfit" and n.endswith(".0.bias") and p.ndim == 1: p.requires_grad = True                  # conv bias = first layer of each block
        elif which == "bn_only" and (n.endswith(".2.weight") or n.endswith(".2.bias")): p.requires_grad = True      # BatchNorm1d scale / shift = third layer of each block
    return [getattr(net, b) for b in BLOCKS]


def main(reps):
    OUT = os.path.join(NIGHT, "unet_peft.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    conds = {"lora": dict(prep=prep_lora), "bitfit": dict(prep=lambda n: prep_bias(n, "bitfit")), "bn_only": dict(prep=lambda n: prep_bias(n, "bn_only")), "lp_ft": dict(warm=100)}
    tr, te = FD.load("d1", "test"), FD.load("d1", "train"); tp, tl = te.pos[:300], te.lab[:300]; S = type("S", (), dict(lab=tl))()
    for N in (10, 20, 50):
        for r in range(reps):
            sel = np.random.RandomState(100 * N + r).permutation(len(tr))[:N]
            for name, kw in conds.items():
                key = f"{name}_{N}_{r}"
                if key in res: continue
                t0 = time.time(); net = ft(tr.pos[sel], tr.lab[sel], r, **kw); s = score(net, tp); e = CP.event_f1((s > 0) & np.isfinite(tp).all(2), S)
                ntrain = sum(p.numel() for p in net.parameters() if p.requires_grad); res[key] = [e["ev_f1"], e["kappa"], ntrain]; json.dump(res, open(OUT, "w"), indent=1)
                print(f"[peft] {key}: F1 {e['ev_f1']:.3f} kappa {e['kappa']:.3f} trainable {ntrain} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=10); main(ap.parse_args().reps)
