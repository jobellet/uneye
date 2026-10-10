"""Bidirectional (non-causal) TCN that outputs a saccade logit per sample, plus the shared training / validation helpers of selftrain.py and sup_bitcn.py.

Input: dino1d.feats (8 velocity channels, no absolute position). Symmetric dilated convolutions (past AND future context), residual blocks conv -> ReLU -> BatchNorm
(the block of online/causal_net.py made symmetric), receptive field 515 samples. Whole trials are processed in one pass (fully convolutional).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from foundation import dino1d as DN, compare as CP, ssl_probe as SP, night_eval as NE
from foundation.train_lodo import DEV
from foundation.hmm_score import ScoreHMM, Scores


FEATS = DN.feats          # feature function of the whole module; sup_bitcn.py --inputs 2 switches it to dino1d.feats2


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__(); self.conv = nn.Conv1d(c, c, 3, padding=d, dilation=d); self.bn = nn.BatchNorm1d(c)
    def forward(self, x): return x + self.bn(F.relu(self.conv(x)))


class BiTCN(nn.Module):
    def __init__(self, nin=8, ch=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128)):
        super().__init__(); self.stem = nn.Conv1d(nin, ch, 5, padding=2); self.stem_bn = nn.BatchNorm1d(ch)
        self.blocks = nn.Sequential(*[Block(ch, d) for d in dilations]); self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):                                       # (B, T, nin) -> (B, T) logit of "saccade"
        h = self.stem_bn(F.relu(self.stem(x.transpose(1, 2)))); return self.head(self.blocks(h))[:, 0]


def score_fn_factory(model, tta=False, bs=8):
    """pos (n, T, 2) at 1 kHz -> (n, T) logits; tta = average over the 4 rotations x mirror (the vx / vy channels are not rotation invariant)"""
    @torch.no_grad()
    def fn(pos):
        model.eval(); out = []
        tr = [(th, mir) for th in ((0, 90, 180, 270) if tta else (0,)) for mir in ((False, True) if tta else (False,))]
        for i in range(0, len(pos), bs):
            p = pos[i:i + bs].astype(np.float64); acc = 0.0
            for th, mir in tr:
                c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)); q = np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]], 2)
                if mir: q = np.stack([q[..., 0], -q[..., 1]], 2)
                acc = acc + model(torch.as_tensor(FEATS(q, 1000.0), device=DEV)).float().cpu().numpy()
            out.append(acc / len(tr))
        o = np.concatenate(out); o[~np.isfinite(pos).all(2)] = 0.0; return o
    return fn


def val_direct(score_fn):
    """validation on the labeled TRAIN splits (SP.val_sets: 60 trials per dataset): mean event F1 / kappa with the HMM decoding, and with the 0 threshold.
    Labels are used only to choose checkpoints."""
    res = {}
    for k, (pos, lab) in SP.val_sets().items():
        s = score_fn(pos); v = np.isfinite(pos).all(2); S = type("S", (), dict(lab=lab))()
        thr = CP.event_f1((s > 0) & v, S); hmm = CP.event_f1(ScoreHMM(1000.0).fit(Scores(s, v)).predict(Scores(s, v)), S)
        res[k] = dict(thr=(thr["ev_f1"], thr["kappa"]), hmm=(hmm["ev_f1"], hmm["kappa"]))
    return res, float(np.mean([v["hmm"][0] for v in res.values()])), float(np.mean([v["hmm"][1] for v in res.values()])), float(np.mean([v["thr"][0] for v in res.values()]))
