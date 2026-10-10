#!/usr/bin/env python3
"""Re-runs everything that depended on the pre-training pool that contained the unlabeled positions of the few-label test set (dataset 1, set A) -- found by the independent audit
(docs/AUDIT.md). Old results are kept in foundation/runs/night/before_audit/. Steps run one after the other, logs in foundation/runs/night/logs/rerun_*.log.   python foundation/rerun_audit.py
"""
import json, os, shutil, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); NIGHT = os.path.join(ROOT, "foundation", "runs", "night"); BK = os.path.join(NIGHT, "before_audit"); os.makedirs(BK, exist_ok=True)
PY = os.path.join(ROOT, ".venv", "bin", "python")
for f in ("unet_ssl_finetune.json", "unet_ms_finetune.json", "unet_layers_stats.json", "unet_layers_freeze.json", "unet_peft.json"):
    p = os.path.join(NIGHT, f)
    if os.path.exists(p) and not os.path.exists(os.path.join(BK, f)): shutil.copy(p, os.path.join(BK, f))
for f in ("unet_pool.npz", "unet_pre.pt", "unet_ms_pre.pt"):
    p = os.path.join(ROOT, "foundation", "runs", f)
    if os.path.exists(p) and not os.path.exists(os.path.join(BK, f)): shutil.copy(p, os.path.join(BK, f))
for f in ("unet_ssl_finetune.json", "unet_ms_finetune.json"):                       # keep the from-scratch draws (they never used the pool), drop the pre-trained ones
    p = os.path.join(NIGHT, f); d = json.load(open(p)); json.dump({k: v for k, v in d.items() if k.startswith("scratch_")}, open(p, "w"), indent=1)
for f in ("unet_layers_stats.json", "unet_layers_freeze.json", "unet_peft.json"):
    p = os.path.join(NIGHT, f)
    if os.path.exists(p): os.remove(p)
pool = os.path.join(ROOT, "foundation", "runs", "unet_pool.npz")
if os.path.exists(pool): os.remove(pool)
STEPS = [("pool", ["foundation/unet_ssl.py", "pool"]), ("pretrain", ["foundation/unet_ssl.py", "pretrain", "--minutes", "25"]), ("finetune", ["foundation/unet_ssl.py", "finetune", "--reps", "10"]),
         ("ms_pretrain", ["foundation/unet_ms.py", "pretrain", "--minutes", "25"]), ("ms_finetune", ["foundation/unet_ms.py", "finetune", "--reps", "10"]),
         ("layers_stats", ["foundation/unet_layers.py", "stats"]), ("layers_freeze", ["foundation/unet_layers.py", "freeze", "--reps", "10", "--k", "3"]), ("peft", ["foundation/unet_peft.py", "--reps", "10"])]
for name, args in STEPS:
    t0 = time.time(); print(f"=== {time.strftime('%H:%M')} {name}", flush=True)
    with open(os.path.join(NIGHT, "logs", f"rerun_{name}.log"), "w") as lf: r = subprocess.run([PY, "-u", *args], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    print(f"    -> exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if r.returncode != 0: sys.exit(1)
open(os.path.join(NIGHT, "RERUN_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M"))
