#!/usr/bin/env python3
"""RUN THIS ON A MACHINE WITH A 24 GB+ NVIDIA GPU (not on the Mac: 8B parameters in bf16 need ~16 GB of GPU memory and the repository uses vLLM, which needs CUDA).
It writes the free-viewing scanpath (8 fixations, 0-100 grid) that DeepGaze3.5-VL predicts for every frame of frames.zip.  NOT TESTED by the author (no GPU available): it follows
predict_scanpath.py of https://github.com/Susmit-A/DeepGaze3.5-VL (commit of 2026-09-18) line by line.

  git clone https://github.com/Susmit-A/DeepGaze3.5-VL && cd DeepGaze3.5-VL && git lfs pull        # the adapter weights are Git LFS files (134-byte pointers without git lfs)
  pip install -r requirements.txt                                                                      # vllm 0.11.2, torch 2.9.0, transformers 4.57.6 ... (Python 3.12)
  unzip frames.zip && python run_scanpaths_gpu.py --frames frames --manifest manifest.json --repo . --out predictions.json
Downloads OpenGVLab/InternVL3_5-8B-HF from HuggingFace on first use (about 16 GB). Send predictions.json back; evaluate with foundation/deepgaze_kit/evaluate_deepgaze.py.
"""
import argparse, json, os, sys
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--frames", default="frames"); ap.add_argument("--manifest", default="manifest.json"); ap.add_argument("--repo", default=".")
ap.add_argument("--out", default="predictions.json"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--base-model", default="OpenGVLab/InternVL3_5-8B-HF"); a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.repo))
from evaluate_vllm_unified import load_model_vllm, FewShotPromptBuilder, parse_scanpath_reduced
from predict_scanpath import build_freeview_prompt
from transformers import AutoProcessor
from vllm import SamplingParams
llm, _ = load_model_vllm(base_model=a.base_model, adapter_path=os.path.join(os.path.abspath(a.repo), "model", "combined_adapter"), num_images=1, gpu_memory_utilization=0.90, max_model_len=4096, max_num_seqs=32)
processor = AutoProcessor.from_pretrained(a.base_model, trust_remote_code=True); builder = FewShotPromptBuilder(processor); man = json.load(open(a.manifest)); prompt = build_freeview_prompt(a.n)
reqs = []
for m in man:
    img = Image.open(os.path.join(a.frames, m["file"])).convert("RGB"); p, mm = builder.build_prompt(test_image=img, test_prompt=prompt, shot_examples=[], partial_response=""); reqs.append({"prompt": p, "multi_modal_data": mm})
outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=max(64, 16 * a.n + 16), seed=42))
res = {m["file"]: {"seq": m["seq"], "frame_index": m["frame_index"], "fixations_0_100": parse_scanpath_reduced(o.outputs[0].text), "text": o.outputs[0].text} for m, o in zip(man, outs)}
json.dump(res, open(a.out, "w"), indent=0); print("wrote", a.out, len(res), "frames; mean fixations per frame", sum(len(v["fixations_0_100"]) for v in res.values()) / max(len(res), 1))
