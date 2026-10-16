#!/usr/bin/env python3
"""Frames to send to a GPU machine: every STEP-th frame (default 10) of the 9 DTB70 sequences -> data/DeepGazeKit/frames/<seq>_<frame>.jpg + manifest.json + frames.zip (about 40 MB).
   python foundation/deepgaze_kit/make_frame_subset.py [step]"""
import glob, json, os, shutil, sys, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from foundation.deepgaze_kit.uav_common import SEQS
step = int(sys.argv[1]) if len(sys.argv) > 1 else 10; out = os.path.join(ROOT, "data", "DeepGazeKit"); os.makedirs(os.path.join(out, "frames"), exist_ok=True); man = []
for seq in SEQS:
    files = sorted(glob.glob(os.path.join(ROOT, "data", "DTB70", seq, "img", "*.jpg")))
    for k in range(5, len(files), step):
        name = f"{seq}_{k:05d}.jpg"; shutil.copy(files[k], os.path.join(out, "frames", name)); man.append({"file": name, "seq": seq, "frame_index": k})
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=0)
with zipfile.ZipFile(os.path.join(out, "frames.zip"), "w", zipfile.ZIP_STORED) as z:
    for m in man: z.write(os.path.join(out, "frames", m["file"]), "frames/" + m["file"])
    z.write(os.path.join(out, "manifest.json"), "manifest.json")
print(len(man), "frames", round(os.path.getsize(os.path.join(out, "frames.zip")) / 1e6, 1), "MB ->", os.path.join(out, "frames.zip"))
