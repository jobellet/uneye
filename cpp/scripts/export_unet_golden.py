#!/usr/bin/env python3
"""Golden reference for the C++ trainer (task T09a): runs the repository's PyTorch UNet (uneye/functions.py) on CPU in training mode on a small deterministic input, and writes weights, input, targets, loss (MCLoss + 0.001 * L2 as in uneye/classifier.py),
all parameter gradients and the batch-norm running statistics after the forward pass to cpp/app/train/tests/golden_unet_grads.txt (double precision, .bin tensor order).
Usage: python cpp/scripts/export_unet_golden.py   (needs torch: pip install torch --index-url https://download.pytorch.org/whl/cpu)"""
import os, sys
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from uneye.functions import UNet, MCLoss, weights_init

KS, MP, K, B, T = 5, 5, 3, 2, 50
ORDER = ["c0.0", "c0.2", "c1.0", "c1.2", "c2.0", "c2.2", "c3.0", "c3.2", "up1.0", "up1.2", "c4.0", "c4.2", "up2.0", "up2.2", "c5.0", "c5.2", "c6.0", "c6.2", "c7"]
torch.manual_seed(3); np.random.seed(3)
net = UNet(K, KS, MP).double(); net.apply(weights_init)
for n, p in net.named_parameters():                                  # non-trivial biases / batch-norm parameters so that every gradient is exercised
    if n.endswith("bias") or ".2." in n: p.data.normal_(0, 0.1)
    if ".2.weight" in n: p.data += 1.0
x = torch.randn(B, 1, T, 2).double() * 0.3
x[:, :, 20:26, :] += 1.5
lab = np.zeros((B, T), int); lab[:, 20:26] = 1; lab[0, 40:44] = 2
tgt = torch.zeros(B, K, T).double()
for b in range(B):
    for t in range(T): tgt[b, lab[b, t], t] = 1
tgt[1, :, 5:8] = 0                                                    # ignored samples (all-zero target)
import copy
sd0 = copy.deepcopy(net.state_dict())
net.train()
out = net(x, ["out"])[0]
loss = MCLoss()(out, tgt)
l2 = sum(torch.sum(p ** 2) for p in net.parameters()); loss = loss + 0.001 * l2
loss.backward()
sd = net.state_dict()

params = dict(net.named_parameters())
def state_of(sdict, key):
    if key == "c7": return [sdict["c7.weight"], sdict["c7.bias"]]
    if key.endswith(".0"): return [sdict[key + ".weight"], sdict[key + ".bias"]]
    return [sdict[key + ".weight"], sdict[key + ".bias"], sdict[key + ".running_mean"], sdict[key + ".running_var"]]
def grads_of(key):
    if key == "c7": return [params["c7.weight"].grad, params["c7.bias"].grad]
    return [params[key + ".weight"].grad, params[key + ".bias"].grad]       # torch Conv2d weight [10][1][ks][2] is already the file layout [o][k][c]

path = os.path.join(ROOT, "cpp", "app", "train", "tests", "golden_unet_grads.txt")
with open(path, "w") as f:
    def w(name, a): a = np.asarray(a.detach().numpy() if torch.is_tensor(a) else a, dtype=np.float64).reshape(-1); f.write(f"{name} {a.size} " + " ".join(f"{v:.17g}" for v in a) + "\n")
    f.write(f"dims {B} {T} {K} {KS} {MP}\n")
    w("x", x[:, 0].permute(0, 2, 1)); w("target", tgt); w("loss", [loss.item()]); w("out", out)       # x as [B][2][T]
    for i, k in enumerate(ORDER):
        for j, t in enumerate(state_of(sd0, k)): w(f"state_{i}_{j}", t)
        for j, t in enumerate(state_of(sd, k)[2:] if k not in ("c7",) and not k.endswith(".0") else []): w(f"after_{i}_{j}", t)
        for j, t in enumerate(grads_of(k)): w(f"grad_{i}_{j}", t)
print("wrote", path, "loss", loss.item())
