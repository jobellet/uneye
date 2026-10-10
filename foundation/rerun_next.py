#!/usr/bin/env python3
"""Second queue (after foundation/rerun_audit.py has finished): the tests proposed by the Antigravity review, each script reviewed by agy before it was launched (docs/AUDIT.md), then the two
low-priority re-runs (EMA of the 8-channel supervised BiTCN, hard-negative CEBRA encoder). Logs: night/logs/next_*.log.   python foundation/rerun_next.py
"""
import os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); NIGHT = os.path.join(ROOT, "foundation", "runs", "night"); PY = os.path.join(ROOT, ".venv", "bin", "python")
while not os.path.exists(os.path.join(NIGHT, "RERUN_DONE")): time.sleep(30)
STEPS = [("weaksup", ["foundation/weaksup.py"]), ("phys_pretrain", ["foundation/unet_phys.py", "pretrain", "--minutes", "25"]), ("phys_finetune", ["foundation/unet_phys.py", "finetune", "--reps", "10"]),
         ("film", ["foundation/film_bitcn.py"]), ("sup_bitcn_b", ["foundation/sup_bitcn.py", "--name", "sup_bitcn_b", "--steps", "6000", "--lr", "1e-3", "--pos-weight", "1.5", "--max-minutes", "28"]),
         ("ssl_encoder_hard", ["foundation/ssl_encoder.py", "--hard", "--out", "foundation/runs/ssl_encoder_hard2.pt"])]
for name, args in STEPS:
    t0 = time.time(); print(f"=== {time.strftime('%H:%M')} {name}", flush=True)
    with open(os.path.join(NIGHT, "logs", f"next_{name}.log"), "w") as lf: r = subprocess.run([PY, "-u", *args], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    print(f"    -> exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
open(os.path.join(NIGHT, "NEXT_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M"))
