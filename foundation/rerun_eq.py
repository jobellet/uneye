#!/usr/bin/env python3
"""Re-run of the self-supervised candidates with the axis-equalised input (UNEYE_AXIS_EQ=1: every position axis divided by the robust noise of ITS OWN velocity, so that the per-axis z-scoring of the archive no
longer distorts the input). Fast version (12 min cap per training) with a stop rule: baseline (plain input channels) and the untrained control first, then TS2Vec a and I-JEPA a; the MAE / HuBERT runs are launched only
if one of the first two beats the equalised plain-input baseline by >= 0.02 in mean event F1 or kappa (best of the two readouts).  Ids carry the suffix _c.   python foundation/rerun_eq.py"""
import json, os, subprocess, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); NIGHT = os.path.join(ROOT, "foundation", "runs", "night"); PY = os.path.join(ROOT, ".venv", "bin", "python"); ENV = dict(os.environ, UNEYE_AXIS_EQ="1")
V = ["foundation/ssl_vit.py"]; T2 = ["foundation/ts2vec.py"]
def run(name, args):
    t0 = time.time(); print(f"=== {time.strftime('%H:%M')} {name}", flush=True)
    with open(os.path.join(NIGHT, "logs", f"eq_{name}.log"), "w") as lf: r = subprocess.run([PY, "-u", *args], cwd=ROOT, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    print(f"    -> exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
def score(i):
    p = os.path.join(NIGHT, i + ".json")
    if not os.path.exists(p): return None
    t = json.load(open(p))["test"]; ds = ["d1", "d2", "d3", "d4", "andersson"]
    return max(((np.mean([t[k][r]["f1"] for k in ds]), np.mean([t[k][r]["kappa"] for k in ds])) for r in ("hmm", "threshold")), key=lambda x: x[0])
run("base", ["foundation/night_eval.py"]); run("ctrl", V + ["--name", "ctrl_untrained_c", "--objective", "jepa", "--steps", "0"])
run("ts2vec_a", T2 + ["--name", "ts2vec_a_c", "--steps", "6000", "--max-minutes", "12"]); run("jepa_a", V + ["--name", "jepa_a_c", "--objective", "jepa", "--steps", "6000", "--max-minutes", "12"])
b = score("base_input_channels_c"); cands = {i: score(i) for i in ("ts2vec_a_c", "jepa_a_c")}; print("baseline (F1, kappa):", b, "| candidates:", cands, flush=True)
go = b is not None and any(v is not None and (v[0] >= b[0] + 0.02 or v[1] >= b[1] + 0.02) for v in cands.values()); print("continue with MAE / HuBERT:", go, flush=True)
if go: run("mae_a", V + ["--name", "mae_a_c", "--objective", "mae", "--steps", "6000", "--max-minutes", "12"]); run("hubert_a", V + ["--name", "hubert_a_c", "--objective", "hubert", "--steps", "6000", "--max-minutes", "12"])
open(os.path.join(NIGHT, "EQ_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M"))
