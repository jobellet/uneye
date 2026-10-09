"""One scorer for every predictor: event F1 AND Cohen's kappa (sample by sample) on the test split of each labeled dataset (saccade vs rest), at 1 kHz.

Event F1 (online/metrics.py): a predicted run of >= 3 samples that overlaps a human saccade is a hit; precision / recall / F1 over
events. It does NOT depend on the exact onset / offset the expert chose (the owner's reason to use it instead of Cohen's kappa).
Every predictor runs at the NATIVE rate of the dataset (1000 Hz for d1, d2, d4; 500 Hz for d3 and andersson) and its labels are then
repeated to 1 kHz (nearest), so that all of them are scored on the same samples. Samples with no human label (andersson padding)
count as "not saccade" for both prediction and truth.
"""
import importlib.util, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
import metrics as M
from common import velocity as unet_velocity
from foundation import data as FD
from free_saccade import detectors as D, cebra_seed as C
from free_saccade.data import Windows

NATIVE = {"d1": 1000.0, "d2": 1000.0, "d3": 500.0, "d4": 1000.0, "andersson": 500.0}


def native_pos(S):
    r = int(round(FD.FS / NATIVE[S.name]))
    return S.pos[:, ::r].astype(np.float32)


def to_1khz(lab, S):
    """(n, T_native) -> (n, T) by repeating each label"""
    r = int(round(FD.FS / NATIVE[S.name]))
    out = np.repeat(lab, r, axis=1)
    T = S.lab.shape[1]
    return out[:, :T] if out.shape[1] >= T else np.pad(out, ((0, 0), (0, T - out.shape[1])))


def event_f1(pred, S):
    """pred (n, T) bool at 1 kHz -> dict(ev_f1, ev_recall, ev_precision)"""
    mask = S.lab >= 0
    P = (pred & mask).astype(np.float32); L = ((S.lab == 1) & mask).astype(np.float32)
    m = M.evaluate_probs(P, L, FD.FS, thr=0.5, min_event=3, with_ap=False)
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(L[mask] > 0, P[mask] > 0)                       # Cohen's kappa, sample by sample, saccade vs rest
    return dict(ev_f1=m["ev_f1"], kappa=kappa, ev_recall=m["ev_recall"], ev_precision=m["ev_precision"])


# ----------------------------------------------------------------------------- the predictors (all return (n, T) bool at 1 kHz)
def pred_ek(S, lam=6.0):
    pos = native_pos(S); return to_1khz(D.ek(pos, NATIVE[S.name], lam=lam), S)


def pred_ek_detrended(S, lam=6.0):
    """Engbert-Kliegl rule on the velocity minus its 100 ms running median (physics-based, pursuit-robust)"""
    pos = native_pos(S); fs = NATIVE[S.name]
    v, valid = C.detrended_velocity(pos, fs); sg = D.robust_sigma(v)
    e = ((v / sg) ** 2).sum(2); bad = ~valid | D.invalid_zone(valid, fs)
    return to_1khz(D.cleanup((e > lam ** 2) & ~bad, fs, min_ms=6.0, merge_ms=0), S)


def pred_hmm(S, S_fit):
    """2-state HMM with a minimum-duration chain, fitted without labels on the unlabeled train split of the same dataset"""
    fs = NATIVE[S.name]
    hmm = D.HMM(fs).fit(Windows(native_pos(S_fit), fs, np.array([S.name] * len(S_fit))))
    return to_1khz(hmm.predict(Windows(native_pos(S), fs, np.array([S.name] * len(S)))), S)


def _causal_from_bin(path):
    spec = importlib.util.spec_from_file_location("equivalence", os.path.join(ROOT, "cpp", "scripts", "equivalence.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.load_bin(path)


def pred_causal_tcn(S, bin_path, delay_ms=0.0):
    """the causal TCN of online/ (cpp/models/causal.bin = trained on set A of datasets 1, 2, 3, supervised), no lookahead
    (tcn_l10.bin: label of sample t-10 emitted at t)"""
    net = _causal_from_bin(bin_path).eval(); pos = native_pos(S); fs = NATIVE[S.name]
    V = unet_velocity(pos[..., 0].astype(np.float64), pos[..., 1].astype(np.float64))
    with torch.no_grad():
        P = np.concatenate([net(torch.from_numpy(V[i:i + 50]))[:, 1].numpy() for i in range(0, len(V), 50)])
    d = int(round(delay_ms * fs / 1000.0))
    if d: P = np.concatenate([P[:, d:], np.zeros((len(P), d), np.float32)], 1)       # re-align: the label of t-d is read at t
    return to_1khz(P > 0.5, S)


def pred_unet(S, weights, classes=2):
    """the original U'n'Eye (training/weights_*), offline, whole trial"""
    from uneye.functions import UNet
    net = UNet(classes, 5, 5); net.load_state_dict(torch.load(os.path.join(ROOT, "training", weights), weights_only=False)); net.eval()
    pos = native_pos(S); V = unet_velocity(pos[..., 0].astype(np.float64), pos[..., 1].astype(np.float64))
    T = V.shape[2] // 25 * 25; P = np.zeros((V.shape[0], V.shape[2]), np.float32)
    with torch.no_grad():
        for i in range(0, V.shape[0], 50):
            P[i:i + 50, :T] = net(torch.from_numpy(V[i:i + 50, :, :T]).transpose(1, 2).unsqueeze(1), ["out"])[0][:, 1].numpy()
        if T < V.shape[2]:
            P[:, T:] = net(torch.from_numpy(V[:, :, -25:]).transpose(1, 2).unsqueeze(1), ["out"])[0][:, 1, -(V.shape[2] - T):].numpy()
    return to_1khz(P > 0.5, S)


def run_baselines(names=FD.ALL):
    rows = []
    for nm in names:
        A, B = FD.load(nm, "train"), FD.load(nm, "test")
        preds = {"Engbert-Kliegl lambda=6 (physics)": lambda: pred_ek(B),
                 "EK on detrended velocity (physics)": lambda: pred_ek_detrended(B),
                 "HMM (unsupervised, 0 label)": lambda: pred_hmm(B, A),
                 "causal TCN, no lookahead (supervised d1+d2+d3)": lambda: pred_causal_tcn(B, os.path.join(ROOT, "cpp", "models", "causal.bin")),
                 "causal TCN, 10 ms lookahead (supervised d1+d2+d3)": lambda: pred_causal_tcn(B, os.path.join(ROOT, "cpp", "models", "tcn_l10.bin"), 10.0),
                 "U'n'Eye, offline (supervised d1+d2+d3)": lambda: pred_unet(B, "weights_1+2+3"),
                 }
        if nm == "andersson": preds["U'n'Eye, offline (supervised Andersson, 5 classes)"] = lambda: pred_unet(B, "weights_Andersson", classes=5)
        for k, f in preds.items():
            try:
                r = event_f1(f(), B); rows.append(dict(dataset=nm, predictor=k, **r))
                print(f"[{nm:9s}] {k:55s} event F1 {r['ev_f1']:.3f}  kappa {r['kappa']:.3f}  (recall {r['ev_recall']:.3f}, precision {r['ev_precision']:.3f})", flush=True)
            except Exception as e:
                print(f"[{nm:9s}] {k:55s} FAILED: {type(e).__name__}: {e}", flush=True)
    return rows


if __name__ == "__main__":
    import json
    rows = run_baselines()
    os.makedirs(os.path.join(ROOT, "foundation", "runs"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "foundation", "runs", "baselines_f1.json"), "w"), indent=1, default=float)
