#!/usr/bin/env python3
"""Score the output of cpp/build/gaze_replay against the human labels and the true future positions.
usage: gaze_score.py out.bin X.csv Y.csv Labels.csv [--name label]"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import metrics as M, forecast as Fc

STATE = {0: "unknown", 1: "fixation", 2: "saccade", 3: "blink", 4: "invalid"}
SOURCE = {0: "none", 1: "guard", 2: "heuristic", 3: "network", 4: "veto", 5: "override"}
FLAGS = {0: "NonFinite", 1: "OutOfRange", 2: "TooFast", 3: "Gap", 4: "TimeBackwards", 5: "Missing", 6: "BlinkMargin", 7: "NnInvalid", 8: "NnStuck", 9: "NnVetoed",
         10: "HeuristicOverride", 11: "Reset", 12: "UnsupportedRate", 13: "NnDegraded", 14: "NoForecast"}


def load(path):
    raw = np.fromfile(path, np.float32)
    n, c = raw[:2].view(np.int32); return raw[2:].reshape(n, c)


def trial_arrays(R, ntr, T):
    """per-trial arrays indexed by the real sample number t (dropped samples stay NaN)"""
    cols = {k: np.full((ntr, T), np.nan, np.float32) for k in ("state", "p", "source", "flags", "health", "st30", "a10x", "a10y", "s10x", "s10y", "a20x", "a20y", "s20x", "s20y", "onset", "ballistic", "pred_amp", "ev_cls")}
    names = {3: "state", 4: "p", 5: "source", 6: "flags", 7: "health", 8: "st30", 9: "a10x", 10: "a10y", 11: "s10x", 12: "s10y", 13: "a20x", 14: "a20y", 15: "s20x", 16: "s20y", 17: "onset", 18: "ballistic", 19: "pred_amp", 20: "ev_cls"}
    k, t = R[:, 0].astype(int), R[:, 2].astype(int)
    for c, nm in names.items(): cols[nm][k, t] = R[:, c]
    return cols


def report(path, xf, yf, lf, name=""):
    X = np.loadtxt(xf, delimiter=","); Y = np.loadtxt(yf, delimiter=","); L = np.loadtxt(lf, delimiter=",") > 0
    R = load(path); ntr = int(R[:, 0].max()) + 1; T = X.shape[1]
    t0 = int(sys.argv[sys.argv.index("--from-trial") + 1]) if "--from-trial" in sys.argv else 0
    X, Y, L = X[:ntr], Y[:ntr], L[:ntr]
    A = trial_arrays(R, ntr, T)
    if t0:
        X, Y, L = X[t0:], Y[t0:], L[t0:]; A = {k: v[t0:] for k, v in A.items()}; ntr -= t0
    ok = np.isfinite(A["state"])
    fast = np.where(ok, A["state"] == 2, False)
    # final label of bin b = the age-30 view read 30 samples later (row t describes bin t-30)
    final = np.zeros_like(fast)
    st30 = A["st30"]
    for k in range(ntr):
        for t in range(30, T):
            if np.isfinite(st30[k, t]) and st30[k, t] >= 0: final[k, t - 30] = st30[k, t] == 2
    valid_final = np.zeros_like(fast); valid_final[:, :T - 30] = True
    out = {}
    for nm, pred in (("age 0 (fast)", fast), ("age 30 (revised)", final)):
        m = valid_final if "30" in nm else np.ones_like(fast)
        # event metrics need rectangular arrays: restrict to the common length
        Tm = T - 30 if "30" in nm else T
        e = M.evaluate_probs(pred[:, :Tm].astype(float), L[:, :Tm], 1000, 0.5, with_ap=False)
        out[nm] = e
    print(f"\n=== {name or path}: {R.shape[0]} samples, {ntr} trials")
    print(f"{'label':20s} {'kappa':>6} {'mcc':>6} {'event F1':>9} {'ev.recall':>9} {'ev.prec.':>9} {'|onset err| ms':>14} {'alarm delay ms':>14} {'false al./min':>13}")
    for nm, e in out.items():
        print(f"{nm:20s} {e['kappa']:6.3f} {e['mcc']:6.3f} {e['ev_f1']:9.3f} {e['ev_recall']:9.3f} {e['ev_precision']:9.3f} {e['onset_err_ms']:14.1f} {e['alarm_delay_ms']:14.1f} {e['false_alarms_per_min']:13.1f}")
    src = A["source"][ok]; st = A["state"][ok]
    print("who decided the label of the bin (age 0):", {SOURCE[int(s)]: round(float(np.mean(src == s)), 4) for s in np.unique(src)})
    fl = A["flags"][ok].astype(np.int64)
    print("flags seen:", {FLAGS[b]: int(((fl >> b) & 1).sum()) for b in range(15) if ((fl >> b) & 1).any()})
    hs = A["health"][ok]; print("health: ok %.4f degraded %.4f failed %.4f" % tuple(np.mean(hs == v) for v in (0, 1, 2)))
    return A, X, Y, L


def forecast_score(A, X, Y, L):
    """position forecast at +10 / +20 ms against the true future position, vs 'stay'"""
    import pandas as pd
    rows = []
    for h, ax, ay, sx, sy in ((10, "a10x", "a10y", "s10x", "s10y"), (20, "a20x", "a20y", "s20x", "s20y")):
        tgt = Fc.displacement_targets(X, Y, 1000.0, (h,))
        k = h
        ph = Fc.situations(L.astype(float), k)
        okm = np.isfinite(tgt).all(1) & np.isfinite(A[ax]) & (A["state"] != 3) & (A["state"] != 4)
        px, py = X, Y                                          # current position
        ex = A[ax] - (X + tgt[:, 0]); ey = A[ay] - (Y + tgt[:, 1])
        err = np.hypot(ex, ey); stay = np.hypot(tgt[:, 0], tgt[:, 1])
        zx = np.abs(ex) / A[sx]; zy = np.abs(ey) / A[sy]
        for c, name in Fc.SITUATION.items():
            m = okm & (ph == c)
            if m.sum() == 0: continue
            rows.append(dict(horizon_ms=h, situation=name, n=int(m.sum()), rmse_stay=float(np.sqrt((stay[m] ** 2).mean())), rmse_engine=float(np.sqrt((err[m] ** 2).mean())),
                             median_err_stay=float(np.median(stay[m])), median_err_engine=float(np.median(err[m])),
                             within_80pct=float(np.mean(np.r_[zx[m], zy[m]] < 1.609)), within_95pct=float(np.mean(np.r_[zx[m], zy[m]] < 2.996))))
    return pd.DataFrame(rows)


def landing_score(A, X, Y, L):
    """amplitude predicted by the engine for the saccade in progress against the true amplitude, by time since the true onset; micro (<1 deg) decision"""
    import pandas as pd
    rows = []
    for k in range(X.shape[0]):
        d = np.diff(np.r_[0, L[k].astype(int), 0])
        for on, off in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
            if on < 1 or off - on < 3 or not (np.isfinite(X[k, on - 1:off + 1]).all() and np.isfinite(Y[k, on - 1:off + 1]).all()): continue
            amp = np.hypot(X[k, off] - X[k, on - 1], Y[k, off] - Y[k, on - 1])
            for t in range(on, min(off + 1, on + 60)):
                pa = A["pred_amp"][k, t]
                rows.append(dict(tau=t - on, amp=amp, pred=pa if (np.isfinite(pa) and A["ballistic"][k, t] == 1) else np.nan))
    D = pd.DataFrame(rows)
    out = []
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 60)):
        s = D[(D.tau >= lo) & (D.tau < hi)]
        have = s[np.isfinite(s.pred) & (s.pred > 0)]
        small = s.amp < 1.0
        correct = ((have.pred < 1.0) == (have.amp < 1.0)).mean() if len(have) else np.nan
        out.append(dict(ms_since_onset=f"{lo}-{hi}", n=len(s), engine_has_estimate=len(have) / max(len(s), 1), median_abs_amp_err=float(np.median(np.abs(have.pred - have.amp))) if len(have) else np.nan,
                        rmse_amp=float(np.sqrt(((have.pred - have.amp) ** 2).mean())) if len(have) else np.nan, micro_vs_saccade_accuracy=correct, majority_class=max(small.mean(), 1 - small.mean())))
    return pd.DataFrame(out)


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 220)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--from-trial" in sys.argv: args.remove(sys.argv[sys.argv.index("--from-trial") + 1])
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else ""
    if name: args.remove(name)
    A, X, Y, L = report(*args[:4], name=name)
    if "--forecast" in sys.argv:
        print("\nposition forecast (deg):"); print(forecast_score(A, X, Y, L).round(3).to_string(index=False))
        print("\namplitude of the saccade in progress (engine estimate vs truth):"); print(landing_score(A, X, Y, L).round(3).to_string(index=False))
