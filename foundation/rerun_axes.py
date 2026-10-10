#!/usr/bin/env python3
"""Pre-training and few-label fine-tuning of the wider U-Net with the per-component noise normalisation (feats2c), same pool and protocol as unet_ssl.py: tests whether the per-axis z-scoring of the archive
(x noise 2-4x smaller than y noise) hurt the plain masked-velocity pre-training.  Results: night/unet_ssl_finetune_c.json (pre_ and scratch_ draws).   python foundation/rerun_axes.py"""
import os, subprocess, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); NIGHT = os.path.join(ROOT, "foundation", "runs", "night"); PY = os.path.join(ROOT, ".venv", "bin", "python")
for name, args in (("axes_pretrain", ["foundation/unet_ssl.py", "pretrain", "--minutes", "25", "--feats", "2c", "--tag", "_c"]), ("axes_finetune", ["foundation/unet_ssl.py", "finetune", "--reps", "10", "--feats", "2c", "--tag", "_c"])):
    t0 = time.time(); print(f"=== {time.strftime('%H:%M')} {name}", flush=True)
    with open(os.path.join(NIGHT, "logs", f"{name}.log"), "w") as lf: r = subprocess.run([PY, "-u", *args], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    print(f"    -> exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if r.returncode: break
open(os.path.join(NIGHT, "AXES_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M"))
