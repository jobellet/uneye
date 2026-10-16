#!/usr/bin/env python3
"""Networks for the landing estimate on the DRONE VIDEOS (EyeTrackUAV2, 1 kHz, 30 free-viewing participants, 9 DTB70 sequences): no rotation augmentation, absolute position + direction (the 'ctx' variant of
foundation/landing_train.py, which gave -15 % on GazeBase). Participants with even numbers train (10 % of the saccades validate), odd numbers test. Compared: the engine's tables; a GRU trained on this data
alone; the GRU trained on GazeBase RAN (foundation/runs/landing_gazebase_gru_ctx.pt) used as it is; the same fine-tuned on the drone training saccades (lr 3e-4, 1500 steps).
Saccades: speed (5-sample central difference) > 40 deg/s for >= 10 ms, peak >= 80 deg/s, amplitude >= 1 deg, no missing sample 30 ms before / 10 ms after.   -> foundation/runs/landing_uav_nets.json"""
import glob, json, os, re, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online")); sys.path.insert(0, os.path.join(ROOT, "foundation", "deepgaze_kit"))
import foundation.landing_train as L
from foundation import unet_ssl as U
from uav_common import SEQS, f2deg
PRE, T = 100, 250
def collect():
    P, OFF, SUBJ = [], [], []
    for seq in SEQS:
        for fn in sorted(glob.glob(os.path.join(ROOT, "data", "EyeTrackUAV2", "RAW_FV", seq, "*.csv"))):
            a = np.loadtxt(fn, delimiter=","); x, y = f2deg(a[:, 11], a[:, 12]); bad = ~np.isfinite(a[:, 11]) | (a[:, 11] <= 0) | (a[:, 12] <= 0); x[bad] = np.nan; y[bad] = np.nan
            p = np.c_[x, y]; sp = np.full(len(p), np.nan); sp[2:-2] = np.hypot(*((p[4:] + p[3:-1] - p[1:-3] - p[:-4]) / 6.0).T) * 1000; subj = int(re.search(r"subj_(\d+)", fn).group(1))
            fast = np.nan_to_num(sp) > 40; dd = np.diff(np.r_[0, fast.astype(int), 0])
            for on, off in zip(np.where(dd == 1)[0], np.where(dd == -1)[0] - 1):
                if off - on + 1 < 10 or off - on + 1 > T - PRE - 20 or on < PRE or off + 10 >= len(p) or np.isnan(p[on - 30:off + 10]).any() or np.nanmax(sp[on:off + 1]) < 80 or np.linalg.norm(p[off] - p[on - 1]) < 1.0: continue
                w = np.full((T, 2), np.nan, np.float32); seg = p[on - PRE:min(on - PRE + T, len(p))]; w[:len(seg)] = seg; w[off - on + PRE + 11:] = np.nan; P.append(w); OFF.append(off - on + PRE); SUBJ.append(subj)
    return np.array(P), np.array(OFF), np.array(SUBJ)
def build(P, off, sel):
    sp = L.Split.__new__(L.Split); sp.HIST = None; sp.P = np.full((len(sel), T + L.PAD, 2), np.nan); sp.P[:, L.PAD:] = P[sel]; R = []
    for k, i in enumerate(sel):
        p0 = P[i, PRE - 1]; ts = np.arange(PRE, min(off[i], PRE + 59) + 1); d = np.linalg.norm(P[i, ts] - p0, axis=1)
        for t in ts[d > 0.02]: R.append((k, PRE, off[i], t))
    sp.R = np.array(R, np.int64); return sp
if __name__ == "__main__":
    P, off, subj = collect(); print("saccades", len(P), "participants", len(set(subj)), flush=True)
    rng = np.random.RandomState(0); trn = np.where(subj % 2 == 0)[0]; tst = np.where(subj % 2 == 1)[0]; rng.shuffle(trn); nv = len(trn) // 10; va, tr = trn[:nv], trn[nv:]
    S = {"train": build(P, off, tr), "val": build(P, off, va), "test": build(P, off, tst)}; print("saccades train/val/test", len(tr), len(va), len(tst), "rows", {k: len(v.R) for k, v in S.items()}, flush=True)
    L.CTX = True; L.NIN = 7 + 8; te = S["test"]; tau = te.R[:, 3] - te.R[:, 1]; lin, stay = L.linear_tables(S["train"], te); res = {"stays here": stay, "tables (refit)": lin}
    ck = torch.load(os.path.join(U.RUNS, "landing_gazebase_gru_ctx.pt"), map_location=L.DEV, weights_only=False); net = L.make("gru"); net.load_state_dict(ck["state"]); res["GRU trained on GazeBase (as is)"] = L.evaluate(net, "gru", te)
    n1, _ = L.train("gru", S["train"], S["val"], 6000); res["GRU, drone data only"] = L.evaluate(n1, "gru", te)
    n2, _ = L.train("gru", S["train"], S["val"], 1500, init=ck["state"], lr=3e-4); res["GRU, GazeBase then drone data"] = L.evaluate(n2, "gru", te); torch.save({"state": n2.state_dict()}, os.path.join(U.RUNS, "landing_uav_gru_ft.pt"))
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60)]; table = {k: {f"{lo}-{hi}": float(v[(tau >= lo) & (tau < hi)].mean()) for lo, hi in bins} | {"all": float(v.mean())} for k, v in res.items()}
    print(f"{'':34s}" + "".join(f"{lo:>2d}-{hi:<4d}" for lo, hi in bins) + "  all")
    for k, r in table.items(): print(f"{k:34s}" + "".join(f"{r[f'{lo}-{hi}']:7.2f}" for lo, hi in bins) + f"  {r['all']:.2f}")
    json.dump({"table": table, "saccades": {"train": len(tr), "val": len(va), "test": len(tst)}}, open(os.path.join(ROOT, "foundation", "runs", "landing_uav_nets.json"), "w"), indent=1)
