#!/usr/bin/env python3
"""Analysis of a real EyeLink recording (pixels) with the detectors of EyeLive: sampling rate, label-free detection (universal HMM + acceleration-zero boundaries), AUTOMATIC UNIT-SCALE selection (the scale that makes
the degree-trained U'n'Eye agree best with the unit-free HMM, foundation/scale_select.py), comparison with the scale given by the screen geometry (if provided), saccade statistics in degrees, and agreement with EyeLink's own
online saccade events (ESACC lines of the ASC file).  Input: an ASC file (EDF converted with edf2asc) or a CSV (t_ms,x,y).
  python foundation/lab_analyze.py data/lab/session.asc [--width-cm 53 --distance-cm 70 --res-x 1920] [--eye LEFT|RIGHT] [--max-seconds 600]
  python foundation/lab_analyze.py --selftest      # builds a fake ASC from benchmark data (known geometry) and checks the pipeline end to end
Output: data/lab/report_<name>/report.md + PNG figures (data/lab is not in git).
"""
import argparse, json, math, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np

# ------------------------------------------------------------------ parsing
def parse_asc(path, eye=None):
    """-> dict(t_ms, x, y, rate, eye, esacc (list of dicts), eblink, msgs, display)"""
    t, X, Y, rate, mode_cols, eyes_hdr = [], [], [], None, None, []
    esacc, eblink, msgs, display = [], [], [], None
    num = re.compile(r"^-?\d+(?:\.\d+)?$")
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.strip(): continue
            c = line[0]
            if c.isdigit() or c == "-":                                       # a sample line: timestamp then tab / space separated fields
                p = line.split()
                if len(p) < 3 or not num.match(p[0]): continue
                def val(s): return float(s) if num.match(s) else np.nan             # "." = missing
                t.append(float(p[0])); X.append([val(p[1]), val(p[4]) if len(p) > 6 and len(eyes_hdr) == 2 else np.nan]); Y.append([val(p[2]), val(p[5]) if len(p) > 6 and len(eyes_hdr) == 2 else np.nan])
            elif line.startswith("SAMPLES"):
                m = re.search(r"RATE\s+([\d.]+)", line); rate = float(m.group(1)) if m else rate
                eyes_hdr = [w for w in line.split() if w in ("LEFT", "RIGHT")]
            elif line.startswith("ESACC"):
                p = line.split()
                try: esacc.append(dict(eye=p[1], t0=float(p[2]), t1=float(p[3]), dur=float(p[4]), sx=float(p[5]), sy=float(p[6]), ex=float(p[7]), ey=float(p[8]), amp=float(p[9]), pv=float(p[10])))
                except (ValueError, IndexError): pass
            elif line.startswith("EBLINK"):
                p = line.split()
                try: eblink.append(dict(eye=p[1], t0=float(p[2]), t1=float(p[3]), dur=float(p[4])))
                except (ValueError, IndexError): pass
            elif line.startswith("MSG"):
                msgs.append(line.strip())
                m = re.search(r"DISPLAY_COORDS\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)", line)
                if m: display = tuple(int(v) for v in m.groups())
    t = np.array(t); X = np.array(X); Y = np.array(Y)
    if len(eyes_hdr) == 2:                                                    # binocular: pick the eye with more valid samples unless told
        ev = [np.isfinite(X[:, k]).sum() for k in (0, 1)]; k = (0 if eye in (None, "LEFT") and ev[0] >= ev[1] else 1) if eye is None else (0 if eye == "LEFT" else 1); name = eyes_hdr[k]; x, y = X[:, k], Y[:, k]
    else: name = eyes_hdr[0] if eyes_hdr else "?"; x, y = X[:, 0], Y[:, 0]
    if rate is None and len(t) > 10: rate = 1000.0 / float(np.median(np.diff(t)))
    return dict(t_ms=t, x=x, y=y, rate=rate, eye=name, esacc=[e for e in esacc if e["eye"][0] == name[0]] if esacc else [], eblink=eblink, msgs=msgs, display=display)

def parse_csv(path):
    a = np.genfromtxt(path, delimiter=",", comments="#"); a = a[np.isfinite(a[:, 0])]
    return dict(t_ms=a[:, 0], x=a[:, 1], y=a[:, 2], rate=1000.0 / float(np.median(np.diff(a[:, 0]))), eye="?", esacc=[], eblink=[], msgs=[], display=None)

# ------------------------------------------------------------------ analysis
def to_trials(d, max_seconds=None, chunk_ms=2000):
    """continuous samples -> (n, T, 2) at 1 kHz (resampled if needed), chunks of 2 s, NaN where data are missing; also returns the start time of every chunk"""
    t, x, y = d["t_ms"], d["x"], d["y"]
    if max_seconds: m = t <= t[0] + 1000 * max_seconds; t, x, y = t[m], x[m], y[m]
    grid = np.arange(t[0], t[-1], 1.0); ok = np.isfinite(x) & np.isfinite(y)
    xi = np.interp(grid, t[ok], x[ok]) if ok.sum() > 2 else np.full(len(grid), np.nan); yi = np.interp(grid, t[ok], y[ok]) if ok.sum() > 2 else np.full(len(grid), np.nan)
    gap = np.interp(grid, t, (~ok).astype(float)) > 0.5                       # grid points that fall in a stretch of missing samples
    xi[gap] = np.nan; yi[gap] = np.nan
    n = len(grid) // chunk_ms; pos = np.stack([xi[:n * chunk_ms], yi[:n * chunk_ms]], 1).reshape(n, chunk_ms, 2).astype(np.float32)
    return pos, grid[:n * chunk_ms:chunk_ms]

def geometry_scale(res_x, width_cm, distance_cm):
    return 2 * math.degrees(math.atan2(width_cm / 2.0, distance_cm)) / res_x             # degrees per pixel (small linear approximation across the screen centre)

def analyse(d, out, args):
    from foundation import compare as CP, universal_hmm as UH, accel_boundaries as AB, data as FD
    from foundation.hmm_score import ScoreHMM, Scores
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True); R = []; rate = d["rate"]; t = d["t_ms"]; ok = np.isfinite(d["x"]) & np.isfinite(d["y"])
    R += [f"# EyeLive analysis of {os.path.basename(args.path)}", "", f"* samples {len(t)}, duration {(t[-1] - t[0]) / 1000:.1f} s, declared / measured rate {rate:.1f} Hz (median dt {np.median(np.diff(t)):.3f} ms), eye {d['eye']}",
          f"* valid samples {100 * ok.mean():.1f} %, positions in the file: x {np.nanmin(d['x']):.0f}..{np.nanmax(d['x']):.0f}, y {np.nanmin(d['y']):.0f}..{np.nanmax(d['y']):.0f}, display coordinates message: {d['display']}"]
    pos, t0s = to_trials(d, args.max_seconds); n = len(pos); S = FD.Set("d1", pos, np.zeros(pos.shape[:2], np.int8), True)
    # unit-free label-free detection (works whatever the unit)
    uni = {}; f, v = UH.unit_free_score(UH.archive_windows(1000.0), "detrended", 1000.0); uni[1000.0] = ScoreHMM(1000.0).fit(Scores(f, v, 1000.0))
    f, v = UH.unit_free_score(pos, "detrended", 1000.0); hmm = uni[1000.0].predict(Scores(f, v, 1000.0)); filled = pos.astype(np.float64)
    ref = AB.refine(hmm, np.where(np.isfinite(filled), filled, np.nan), 1000.0, "both", 15.0, offset_rule="last")
    # automatic scale: degree-trained U'n'Eye at every candidate scale against the HMM events
    S_ref = type("S", (), dict(lab=ref.astype(np.int8)))(); grid = [0.004 * 1.4 ** k for k in range(0, 24)]; curve = []
    for c in grid:
        pr = CP.pred_unet(FD.Set("d1", (pos * c).astype(np.float32), S.lab, True), "weights_1+2+3"); e = CP.event_f1(pr, S_ref); curve.append((c, 0.5 * (e["ev_f1"] + (0 if np.isnan(e["kappa"]) else e["kappa"]))))
    best = max(curve, key=lambda x: x[1]); c_hat = best[0]; srt = sorted(curve, key=lambda x: -x[1]); low = best[1] < 0.4 or (abs(srt[1][1] - srt[0][1]) < 0.02 and max(srt[0][0], srt[1][0]) / min(srt[0][0], srt[1][0]) > 2.1)
    R += ["", "## Unit scale", f"* automatic choice: **{c_hat:.4f} deg per pixel** ({1 / c_hat:.1f} pixels per degree), agreement {best[1]:.2f}" + ("  **LOW CONFIDENCE**" if low else ""), f"* agreement at the neighbouring candidates: " + ", ".join(f"{c:.4f}: {a:.2f}" for c, a in sorted(curve, key=lambda x: abs(math.log(x[0] / c_hat)))[1:5])]
    c_true = None
    if args.width_cm and args.distance_cm and (args.res_x or d["display"]):
        rx = args.res_x or (d["display"][2] - d["display"][0] + 1); c_true = geometry_scale(rx, args.width_cm, args.distance_cm); R += [f"* from the geometry ({rx} px, {args.width_cm} cm, {args.distance_cm} cm): **{c_true:.4f} deg per pixel** ({1 / c_true:.1f} px/deg); automatic / geometry = {c_hat / c_true:.2f}"]
    c_use = c_true or c_hat
    plt.figure(figsize=(6, 3.5)); plt.semilogx([c for c, a in curve], [a for c, a in curve], "o-"); plt.axvline(c_hat, color="r", label="automatic");
    if c_true: plt.axvline(c_true, color="g", ls="--", label="geometry")
    plt.xlabel("scale (degrees per pixel)"); plt.ylabel("agreement U'n'Eye / HMM"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out, "scale_curve.png"), dpi=120); plt.close()
    # events in degrees with the scale in use
    amps, durs, vpk, ons = [], [], [], []
    for i in range(n):
        p = filled[i] * c_use
        for s, e in D_runs(ref[i]):
            if s < 2 or e > 1997: continue
            seg = p[s - 1:e + 2]
            if not np.isfinite(seg).all(): continue
            amps.append(float(np.hypot(*(seg[-1] - seg[0])))); durs.append(e - s + 1); sp = np.hypot(*(np.gradient(p[s:e + 1], axis=0) * 1000.0).T); vpk.append(float(sp.max())); ons.append(t0s[i] + s)
    amps, durs, vpk = np.array(amps), np.array(durs), np.array(vpk)
    if len(amps):
        R += ["", f"## Saccades (label-free, scale {c_use:.4f} deg/px)", f"* {len(amps)} events in {n * 2} s analysed = {len(amps) / (n * 2):.2f} per second; amplitude median {np.median(amps):.2f} deg (10-90 %: {np.percentile(amps, 10):.2f}-{np.percentile(amps, 90):.2f}); microsaccades (< 1 deg) {100 * np.mean(amps < 1):.0f} %",
              f"* duration median {np.median(durs):.0f} ms; peak speed median {np.median(vpk):.0f} deg/s; peak speed / amplitude median {np.median(vpk / np.maximum(amps, 1e-6)):.0f} per second (typical 20-200)"]
        plt.figure(figsize=(9, 3.3)); plt.subplot(1, 2, 1); plt.loglog(amps, durs, ".", alpha=.5); plt.xlabel("amplitude (deg)"); plt.ylabel("duration (ms)"); plt.title("duration vs amplitude")
        plt.subplot(1, 2, 2); plt.loglog(amps, vpk, ".", alpha=.5); plt.xlabel("amplitude (deg)"); plt.ylabel("peak speed (deg/s)"); plt.title("main sequence"); plt.tight_layout(); plt.savefig(os.path.join(out, "main_sequence.png"), dpi=120); plt.close()
    # EyeLink's own saccade events
    if d["esacc"] and len(ons):
        el = [(e["t0"], e["t1"]) for e in d["esacc"]]; mine = [(o, o + du) for o, du in zip(ons, durs)]; tp = sum(any(a0 <= b1 and b0 <= a1 for (a0, a1) in el) for (b0, b1) in mine); hit_el = sum(any(a0 <= b1 and b0 <= a1 for (b0, b1) in mine) for (a0, a1) in el)
        R += ["", "## Agreement with EyeLink's online saccade events (no ground truth: EyeLink's parser has its own velocity / acceleration thresholds)", f"* EyeLink ESACC events: {len(el)}; ours: {len(mine)}; ours overlapping an EyeLink event: {tp} ({100 * tp / max(len(mine), 1):.0f} %); EyeLink events overlapping ours: {hit_el} ({100 * hit_el / max(len(el), 1):.0f} %)"]
        ea = np.array([e["amp"] for e in d["esacc"]]); R += [f"* EyeLink amplitude median {np.median(ea):.2f} deg (as computed by the tracker's own calibration: compare with ours)"]
    # a picture of a few seconds
    k = int(np.argmax([ref[i].sum() for i in range(n)])); plt.figure(figsize=(10, 3.5)); tt = np.arange(2000); p = filled[k] * c_use
    plt.plot(tt, p[:, 0], lw=.8, label="x"); plt.plot(tt, p[:, 1], lw=.8, label="y")
    for s, e in D_runs(ref[k]): plt.axvspan(s, e, color="orange", alpha=.3)
    plt.xlabel("ms"); plt.ylabel("deg (scale in use, offset arbitrary)"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out, "trace.png"), dpi=120); plt.close()
    open(os.path.join(out, "report.md"), "w").write("\n".join(R) + "\n"); print("\n".join(R)); return dict(c_hat=c_hat, c_true=c_true, low=low, agreement=best[1], n_events=len(amps))

def D_runs(m):
    from free_saccade import detectors as D
    return D.runs_1d(m)

# ------------------------------------------------------------------ self test: a fake ASC from benchmark data with a known geometry
def make_fake_asc(path, px_per_deg=34.0, res=(1920, 1080), n_trials=80):
    from foundation import night_eval as NE
    A, B = NE.data()["d1"]; pos = B.pos[:n_trials].astype(np.float64); lab = B.lab[:n_trials]; rng = np.random.RandomState(0); t = 1_000_000.0; lines = ["** CONVERTED FROM FAKE.EDF (EyeLive self test)", f"MSG\t{int(t)} DISPLAY_COORDS 0 0 {res[0] - 1} {res[1] - 1}", f"START\t{int(t)}\tLEFT\tSAMPLES\tEVENTS",
                                                                                                                                       "EVENTS\tGAZE\tLEFT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2", "SAMPLES\tGAZE\tLEFT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2"]
    for i in range(n_trials):
        p = pos[i] * px_per_deg + np.array([res[0] / 2, res[1] / 2]); inb = False; ev = []
        for s, e in __import__("free_saccade.detectors", fromlist=["x"]).runs_1d(lab[i] == 1): ev.append((s, e))
        for k in range(len(p)):
            x, y = p[k]
            if not (np.isfinite(x) and np.isfinite(y)): lines.append(f"{int(t) + k}\t   .\t   .\t    0.0\t.....")
            else: lines.append(f"{int(t) + k}\t{x:7.1f}\t{y:7.1f}\t{1000 + rng.randn():7.1f}\t.....")
        for s, e in ev:
            if e - s > 4: lines.append(f"ESACC\tL\t{int(t) + s}\t{int(t) + e}\t{e - s + 1}\t{p[s][0]:.1f}\t{p[s][1]:.1f}\t{p[e][0]:.1f}\t{p[e][1]:.1f}\t{np.hypot(*(p[e] - p[s])) / px_per_deg:.2f}\t{300.0:.0f}")
        t += len(p)
    open(path, "w").write("\n".join(lines) + "\n"); return px_per_deg

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path", nargs="?"); ap.add_argument("--width-cm", type=float); ap.add_argument("--distance-cm", type=float); ap.add_argument("--res-x", type=int); ap.add_argument("--eye"); ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--out"); a = ap.parse_args()
    if a.selftest:
        os.makedirs(os.path.join(ROOT, "data", "lab"), exist_ok=True); a.path = os.path.join(ROOT, "data", "lab", "fake_selftest.asc"); ppd = make_fake_asc(a.path, 34.0)
        a.width_cm, a.distance_cm = 53.0, 7.0; a.width_cm = 2 * a.distance_cm * math.tan(math.radians(1920 / ppd / 2)); a.res_x = 1920            # geometry consistent with 34 px/deg
    d = parse_asc(a.path, a.eye) if a.path.lower().endswith(".asc") else parse_csv(a.path); out = a.out or os.path.join(ROOT, "data", "lab", "report_" + os.path.splitext(os.path.basename(a.path))[0]); r = analyse(d, out, a)
    if a.selftest: print("SELFTEST", "OK" if r["c_true"] and 0.6 <= r["c_hat"] / r["c_true"] <= 1.6 else "FAIL", f"(automatic / geometry = {r['c_hat'] / r['c_true']:.2f})")

if __name__ == "__main__":
    main()
