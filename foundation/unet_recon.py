#!/usr/bin/env python3
"""Visual progress of the masked-auto-encoder pre-training: for fixed windows and a fixed mask, the U-Net of each snapshot (foundation/runs/unet_snaps/step_*.pt) fills
the masked spans of the velocity; the predicted velocity (denormalised) is integrated from the same starting position as the trace and drawn over the real trace.
Outside the masks the real velocity is used (the network only has to explain what was hidden).  -> docs/figs_unet/recon_position.png, recon_velocity.png
python foundation/unet_recon.py
"""
import glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from free_saccade import detectors as D
from foundation.dino1d import feats2
from foundation import unet_ssl as U
from foundation.train_lodo import DEV
OUT = os.path.join(ROOT, "docs", "figs_unet"); os.makedirs(OUT, exist_ok=True)

z = np.load(U.POOL); pos, grp = z["pos"], z["grp"]; rng = np.random.RandomState(3)
pick = [rng.choice(np.where(grp == g)[0]) for g in ("EMTeC", "d1", "Lund2013")]; W = pos[pick].astype(np.float64)        # one window of three different sources
f = feats2(W, 1000.0); v, _ = D.velocity(W, 1000.0); sg = np.maximum(D.robust_sigma(v).mean(2, keepdims=True), 1e-9)
m = np.zeros(f.shape[:2], bool); mr = np.random.RandomState(11)
for i in range(len(m)):
    while m[i].mean() < 0.3: L = mr.randint(8, 41); s = mr.randint(0, U.W - L); m[i, s:s + L] = True
x = torch.as_tensor(np.concatenate([f * (1 - m[..., None]), m[..., None]], 2).astype(np.float32), device=DEV)
KEEP = (0, 250, 500, 1000, 2000, 3000, 6250, 10750); snaps = [p for p in sorted(glob.glob(os.path.join(U.RUNS, "unet_snaps", "step_*.pt"))) if int(os.path.basename(p)[5:-3]) in KEEP]; steps = [int(os.path.basename(p)[5:-3]) for p in snaps]; t = np.arange(U.W)
fp, fv = [plt.subplots(len(snaps), 3, figsize=(15, 1.9 * len(snaps)), squeeze=False) for _ in range(2)]
for r, (p, st) in enumerate(zip(snaps, steps)):
    net = U.WUNet(nout=2).to(DEV); net.load_state_dict(torch.load(p, weights_only=False)); net.eval()
    with torch.no_grad(): out = net(x).transpose(1, 2).cpu().numpy()                                           # (3, T, 2) predicted arcsinh velocity
    fill = np.where(m[..., None], out, f)                                                                      # predicted inside the masks, real outside
    vel = np.sinh(fill) * sg; rec = W[:, :1] + np.cumsum(vel, 1) / 1000.0                                      # integrate from the same starting position
    for c in range(3):
        a, b = fp[1][r, c], fv[1][r, c]
        a.plot(t, W[c, :, 0], color="0.6", lw=1.4); a.plot(t, rec[c, :, 0], color="#d9534f", lw=.9); a.plot(t, W[c, :, 1], color="0.6", lw=1.4); a.plot(t, rec[c, :, 1], color="#337ab7", lw=.9)
        b.plot(t, f[c, :, 0], color="0.6", lw=1.2); b.plot(t, np.where(m[c], out[c, :, 0], np.nan), color="#d9534f", lw=1.0); b.plot(t, f[c, :, 1] - 8, color="0.6", lw=1.2); b.plot(t, np.where(m[c], out[c, :, 1] - 8, np.nan), color="#337ab7", lw=1.0)
        for s_ in (a, b): s_.fill_between(t, 0, 1, where=m[c], color="#f0ad4e", alpha=.18, transform=s_.get_xaxis_transform()); s_.tick_params(labelsize=6)
        if r == 0: a.set_title(("EMTeC", "dataset 1", "Lund2013")[c], fontsize=9); b.set_title(("EMTeC", "dataset 1", "Lund2013")[c], fontsize=9)
        if c == 0: a.set_ylabel(f"step {st}", fontsize=8); b.set_ylabel(f"step {st}", fontsize=8)
fp[0].suptitle("Pre-training progress: gray = real position (x, y), color = position integrated from the velocity predicted inside the masked spans (shaded)", fontsize=9); fp[0].tight_layout(); fp[0].savefig(os.path.join(OUT, "recon_position.png"), dpi=110)
fv[0].suptitle("Velocity (arcsinh, normalised): gray = real, color = predicted inside the masks; y shifted by -8", fontsize=9); fv[0].tight_layout(); fv[0].savefig(os.path.join(OUT, "recon_velocity.png"), dpi=110); print(steps)
