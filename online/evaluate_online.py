#!/usr/bin/env python3
"""Compare the causal net with the original U'n'Eye on the LAST SAMPLE of a 200 ms time bin.

Online protocol (identical for both networks): at every time t the network sees the 200 ms bin that ends at t
and the label of t (the newest sample, no lookahead) is the output. U-Net: one forward pass per window.
Causal net: output at t of a causal pass over the recording (identical to the window result as the
receptive field, 67 samples, is shorter than the bin; verified below).
Also reported: original U-Net used offline (whole trial, uses the future) as an upper reference.

Test data: set B of datasets 1,2,3 (300 random trials each, seed 10, like analysis scripts/evaluation_networks.ipynb).
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uneye.functions import UNet
from causal_net import CausalTCN
from common import load, velocity

ap = argparse.ArgumentParser()
ap.add_argument("--causal", default="weights_causal")
ap.add_argument("--unet", default="../training/weights_1+2+3")
ap.add_argument("--bin-ms", type=float, default=200)
ap.add_argument("--n-test", type=int, default=300)
ap.add_argument("--min-event", type=int, default=3, help="samples; shorter prediction runs are ignored for event metrics")
ap.add_argument("--out", default="results_online.json")
a = ap.parse_args()
torch.set_num_threads(4)

ck = torch.load(a.causal, weights_only=False)
cnet = CausalTCN(2, ck["channels"]); cnet.load_state_dict(ck["state"]); cnet.eval()
sd = torch.load(a.unet, weights_only=False)
unet = UNet(2, 5, 5); unet.load_state_dict(sd); unet.eval()


def causal_probs(V):  # (n,2,T) -> (n,T) P(saccade), output at t uses only <= t
    with torch.no_grad():
        return cnet(torch.from_numpy(V))[:, 1].numpy()


def unet_offline(V):
    T = V.shape[2] // 25 * 25
    out = np.zeros((V.shape[0], V.shape[2]), np.float32)
    with torch.no_grad():
        for i in range(0, V.shape[0], 50):
            x = torch.from_numpy(V[i:i + 50, :, :T]).transpose(1, 2).unsqueeze(1)
            out[i:i + 50, :T] = unet(x, ["out"])[0][:, 1].numpy()
        if T < V.shape[2]:  # tail: last 25 samples window
            x = torch.from_numpy(V[:, :, -25:]).transpose(1, 2).unsqueeze(1)
            out[:, T:] = unet(x, ["out"])[0][:, 1, -(V.shape[2] - T):].detach().numpy()
    return out


def unet_lastbin(V, W):
    """for every t: U-Net on the bin [t-W+1, t] (zeros before the recording start), output at the newest sample"""
    n, _, T = V.shape
    P = np.zeros((n, T), np.float32)
    pad = np.concatenate([np.zeros((n, 2, W - 1), np.float32), V], 2)
    with torch.no_grad():
        for i in range(n):
            win = torch.from_numpy(pad[i]).unfold(1, W, 1)            # (2, T, W)
            x = win.permute(1, 2, 0).unsqueeze(1)                    # (T,1,W,2)
            for j in range(0, T, 500):
                P[i, j:j + 500] = unet(x[j:j + 500].contiguous(), ["out"])[0][:, 1, -1].numpy()
    return P


def runs(b):
    d = np.diff(np.r_[0, b.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def metrics(P, L, fs, thr, min_event):
    pred, t = P > thr, L > 0
    tp, fp, fn, tn = (t & pred).sum(), (~t & pred).sum(), (t & ~pred).sum(), (~t & ~pred).sum()
    N = t.size
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    po = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / N ** 2
    kappa = (po - pe) / (1 - pe)
    ev_t = ev_hit = ev_p = ev_pok = 0
    delay, onset_err = [], []
    for k in range(t.shape[0]):
        te = runs(t[k]); pe_ = [r for r in runs(pred[k]) if r[1] - r[0] + 1 >= min_event]
        used = set(); ev_t += len(te); ev_p += len(pe_)
        for (s, e) in te:
            for j, (ps, pe2) in enumerate(pe_):
                if j not in used and ps <= e and pe2 >= s:
                    used.add(j); ev_hit += 1
                    onset_err.append((ps - s) * 1000 / fs)
                    delay.append((ps + min_event - 1 - s) * 1000 / fs)  # time the online system could raise the alarm
                    break
        ev_pok += len(used)
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else float("nan")
    return dict(F1=float(f1), kappa=float(kappa), precision=float(prec), recall=float(rec),
                ev_recall=ev_hit / max(ev_t, 1), ev_precision=ev_pok / max(ev_p, 1),
                alarm_delay_ms_median=q(delay, 50), alarm_delay_ms_p95=q(delay, 95), onset_err_ms_median=q(onset_err, 50),
                n_events=ev_t)


def tune_threshold(P, L):
    best = (0, 0.5)
    for thr in np.arange(0.1, 0.91, 0.05):
        pred, t = P > thr, L > 0
        tp, fp, fn = (t & pred).sum(), (~t & pred).sum(), (t & ~pred).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        best = max(best, (f1, thr))
    return float(best[1])


# threshold tuning data: validation trials of set A (same split as train_causal.py)
np.random.seed(1)
val = {}
for s in "123":
    X, Y, L, fs = load(s, "A"); p = np.random.permutation(X.shape[0])[:30]
    val[s] = (velocity(X[p], Y[p]), L[p], fs)

np.random.seed(10)
results = {}
allP = {"causal": [], "unet_lastbin": [], "unet_offline": []}
allL, allfs = [], []
for s in "123":
    X, Y, L, fs = load(s, "B")
    ind = np.random.permutation(X.shape[0])[:a.n_test]
    V, Lt = velocity(X[ind], Y[ind]), L[ind]
    W = int(round(a.bin_ms * fs / 1000)); W = (W + 24) // 25 * 25
    Pc, Pu, Po = causal_probs(V), unet_lastbin(V, W), unet_offline(V)
    # sanity: causal net on a 200 ms bin == causal net on the whole recording (last sample)
    chk = []
    for i in range(3):
        for t in (300, 500, 700):
            x = torch.from_numpy(V[i:i + 1, :, t - W + 1:t + 1])
            with torch.no_grad(): chk.append(abs(cnet(x)[0, 1, -1].item() - Pc[i, t]))
    print(f"set {s}: bin {W} samples; max |causal(bin) - causal(full)| = {max(chk):.2e}", flush=True)
    for name, P in (("causal", Pc), ("unet_lastbin", Pu), ("unet_offline", Po)):
        results[f"set{s}/{name}"] = metrics(P, Lt, fs, 0.5, a.min_event)
        allP[name].append(P)
    allL.append(Lt); allfs.append(fs)

thr = {"causal": tune_threshold(np.concatenate([causal_probs(v[0]) for v in val.values()]), np.concatenate([v[1] for v in val.values()]))}
print("validation-tuned threshold (causal):", thr["causal"])
# pooled over datasets (event metrics need equal length -> per-dataset then weighted by sample count below)
for name in allP:
    for s_i, s in enumerate("123"):
        pass
results["_thr_causal_tuned"] = thr["causal"]
for s_i, s in enumerate("123"):
    results[f"set{s}/causal_tuned"] = metrics(allP["causal"][s_i], allL[s_i], allfs[s_i], thr["causal"], a.min_event)
json.dump(results, open(a.out, "w"), indent=1)

keys = ["F1", "kappa", "precision", "recall", "ev_recall", "ev_precision", "alarm_delay_ms_median", "alarm_delay_ms_p95", "onset_err_ms_median"]
print("\n| set | model | " + " | ".join(keys) + " |\n|---|---|" + "---|" * len(keys))
for s in "123":
    for name in ("unet_offline", "unet_lastbin", "causal", "causal_tuned"):
        r = results[f"set{s}/{name}"]
        print(f"| {s} | {name} | " + " | ".join(f"{r[k]:.3f}" if k.startswith(("F1", "k", "p", "r", "ev")) else f"{r[k]:.1f}" for k in keys) + " |")
