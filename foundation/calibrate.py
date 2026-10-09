#!/usr/bin/env python3
"""How many labeled trials of a NEW dataset does the foundation model need to equal the supervised U'n'Eye?

Target: dataset 1 (the foundation model foundation/runs/lodo_<target>.pt never saw it). Reference: the original U'n'Eye
(training/weights_1+2+3, trained on set A of datasets 1, 2, 3 = supervised in-domain for dataset 1), offline on set B.
Calibration with N labeled trials of the target's set A (N = 0, 1, 5, 20, 100, 1000; first N of a fixed shuffle):
  head   only the embedding layer and the class prototypes are trained (the part an online C++ head can update)
  full   the whole network is fine-tuned
300 steps, each batch half target windows (the N labeled trials) and half windows of the 4 source datasets (so that a few target
trials do not make the model forget), BatchNorm statistics frozen. Same scorer for every model: set B of the target at 1 kHz,
saccade-vs-rest Cohen's kappa and event F1 (online/metrics.py, min_event=3).
Run from the repository root:  python foundation/calibrate.py [--target d1]
"""
import argparse, copy, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from sklearn.metrics import cohen_kappa_score
import metrics as M
from common import velocity
from uneye.functions import UNet
from foundation import data as FD, model as FM
from foundation.train_lodo import batch, loss_fn, predict, scores, DEV, T_WIN


def unet_reference(S, weights):
    """original U'n'Eye offline on the target's test trials (1 kHz datasets only), same scorer"""
    unet = UNet(2, 5, 5); unet.load_state_dict(torch.load(weights, weights_only=False)); unet.eval()
    X, Y = S.pos[..., 0].astype(np.float64), S.pos[..., 1].astype(np.float64)
    V = velocity(X, Y).astype(np.float32)
    T = V.shape[2] // 25 * 25; P = np.zeros((V.shape[0], V.shape[2]), np.float32)
    with torch.no_grad():
        for i in range(0, V.shape[0], 50):
            P[i:i + 50, :T] = unet(torch.from_numpy(V[i:i + 50, :, :T]).transpose(1, 2).unsqueeze(1), ["out"])[0][:, 1].numpy()
    K = np.zeros((V.shape[0], FD.K, V.shape[2]), np.float32); K[:, 1] = P; K[:, 0] = 1 - P
    return scores(K, S)


def calibrate(net0, target_trials, sources, mode, steps=300, seed=0):
    net = copy.deepcopy(net0).to(DEV); rng = np.random.RandomState(seed); torch.manual_seed(seed)
    for p in net.parameters(): p.requires_grad = mode == "full"
    if mode == "head":
        for p in list(net.emb.parameters()) + [net.proto]: p.requires_grad = True
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-3)
    for it in range(steps):
        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm1d): m.eval()                # frozen statistics
        xt, yt, ct = batch([target_trials], rng, 16, True)
        xs, ys, cs = batch(sources, rng, 4, True)
        loss = loss_fn(net(torch.cat([xt, xs])), torch.cat([yt, ys]), np.concatenate([ct, cs]), None)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    return net


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", default="d1"); a = ap.parse_args()
    tgt = a.target
    ck = torch.load(os.path.join(ROOT, "foundation", "runs", f"lodo_{tgt}.pt"), weights_only=False)
    net0 = FM.Foundation(); net0.load_state_dict(ck["state"]); net0.to(DEV)
    trainA, testB = FD.load(tgt, "train"), FD.load(tgt, "test")
    sources = [FD.load(k, "train") for k in ck["trained_on"]]
    rows = []
    t0 = time.time()
    ref = unet_reference(testB, os.path.join(ROOT, "training", "weights_1+2+3"))
    rows.append(dict(model="U'n'Eye (weights_1+2+3, supervised in-domain)", N=None, **ref))
    print(f"reference U'n'Eye: kappa {ref['kappa_sacc']:.3f}  event F1 {ref['ev_f1']:.3f}  (recall {ref['ev_recall']:.3f}, precision {ref['ev_precision']:.3f})", flush=True)
    r0 = scores(predict(net0, testB), testB); rows.append(dict(model="foundation, zero-shot", N=0, **r0))
    print(f"foundation zero-shot (never saw {tgt}): kappa {r0['kappa_sacc']:.3f}  event F1 {r0['ev_f1']:.3f}  (recall {r0['ev_recall']:.3f}, precision {r0['ev_precision']:.3f})", flush=True)
    order = np.random.RandomState(0).permutation(len(trainA))
    for N in (1, 5, 20, 100, 1000):
        sub = FD.Set(tgt, trainA.pos[order[:N]], trainA.lab[order[:N]], trainA.coarse)
        for mode in ("head", "full"):
            t1 = time.time()
            r = scores(predict(calibrate(net0, sub, sources, mode), testB), testB)
            rows.append(dict(model=f"foundation + calibration ({mode})", N=N, **r))
            print(f"N={N:5d} {mode:4s}: kappa {r['kappa_sacc']:.3f}  event F1 {r['ev_f1']:.3f}  (recall {r['ev_recall']:.3f}, precision {r['ev_precision']:.3f})  ({time.time() - t1:.0f} s)", flush=True)
    out = os.path.join(ROOT, "foundation", "runs", f"calibrate_{tgt}.json")
    json.dump(rows, open(out, "w"), indent=1, default=float); print("wrote", out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
