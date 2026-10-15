#!/usr/bin/env python3
"""Reading: candidates on WORDS and a prior on the saccade vector (small steps to the right, large return sweeps to the left and down). GazeBase TEX, cache of
foundation/tex_saccades_extract.py. Participants are split by parity (train: even ids, test: odd ids). Prior P(dx, dy) = smoothed 2-D histogram (0.5 deg bins) of the landing vector
(landing - position before the onset) of the train saccades. Candidates = word boxes found in the page screenshot (letters merged by a horizontal dilation, connected components,
centres converted to degrees). Selection as in cpp/app/engine/include/eyelive/candidates.hpp with the prior as candidate weight; amplitude rule as shipped.
Compared at tau = 10 and 20 ms: tables alone; words, flat weights; words with the prior; the prior alone (most probable word, kinematics ignored); saliency peaks (N = 16).
"""
import os, re, sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label, center_of_mass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from foundation.landing_tex_saliency import px2deg, pqft, peaks, cand_deg, pick, rule, COEF, W, H
stim = os.path.join(ROOT, "data", "GazeBase", "stim", "TEXScreenshots")
z = np.load(os.path.join(ROOT, "data", "GazeBase", "tex_saccades.npz")); n = len(z["p0"]); subj = z["subj"]; train = subj % 2 == 0; test = ~train
print("saccades", n, "train", int(train.sum()), "test", int(test.sum()))
# prior
EX, EY, B = 25.0, 8.0, 0.5
dxy = z["land"] - z["p0"]; H2, _, _ = np.histogram2d(dxy[train, 0], dxy[train, 1], bins=[int(2 * EX / B), int(2 * EY / B)], range=[[-EX, EX], [-EY, EY]]); H2 = gaussian_filter(H2, 1.0) + 1e-3; H2 /= H2.sum()
def prior(v):
    i = np.clip(((v[:, 0] + EX) / B).astype(int), 0, H2.shape[0] - 1); j = np.clip(((v[:, 1] + EY) / B).astype(int), 0, H2.shape[1] - 1); return H2[i, j] * (H2.size)
print("prior: share of train saccades with dx > 0:", round(float((dxy[train, 0] > 0).mean()), 2), " median dx of forward / backward:", round(float(np.median(dxy[train, 0][dxy[train, 0] > 0])), 1), round(float(np.median(dxy[train, 0][dxy[train, 0] < 0])), 1))
# word candidates and saliency peaks per page
words, sal = {}, {}
for r in range(1, 10):
    for s in (1, 2):
        im = Image.open(os.path.join(stim, f"TEX_R{r}S{s}_bg.png")); g = np.asarray(im.convert("L")) > 60
        lab, k = label(binary_dilation(g, structure=np.ones((1, 9), bool))); c = np.array(center_of_mass(g, lab, range(1, k + 1))); area = np.bincount(lab.ravel())[1:]
        c = c[area > 40]; words[(r, s)] = np.stack(px2deg(c[:, 1], c[:, 0]), 1); sal[(r, s)] = cand_deg(*peaks(pqft(im), 16)[:2])
print("words per page (R1S1):", len(words[(1, 1)]), " saliency peaks:", len(sal[(1, 1)]))
res = {}
for tau in (10, 20):
    k = f"{tau}"; cur = z[f"cur{k}"]; v = z[f"v{k}"]; vmax = z[f"vmax{k}"]; p0 = z["p0"]; land = z["land"]; out = {m: [] for m in ("tables", "words flat", "words + prior", "prior only", "saliency", "cover words", "cover sal")}; wgrid = (0.0, 0.25, 0.5, 0.75, 1.0); wres = {w_: [] for w_ in wgrid}
    for i in np.where(test)[0]:
        d = np.linalg.norm(cur[i] - p0[i])
        if d < 0.02: continue
        u = (cur[i] - p0[i]) / d; rem = max(float(COEF[min(tau // 2, 29)] @ np.array([d, v[i], vmax[i]])), 0.0); tab = cur[i] + rem * u; page = (int(z["rnd"][i]), int(z["ses"][i]))
        out["tables"].append(np.linalg.norm(tab - land[i]))
        C = words[page]; pw = prior(C - p0[i]); j = pick(p0[i], cur[i], rem, C, np.ones(len(C))); out["words flat"].append(np.linalg.norm(rule(p0[i], C[j], tab, tau) - land[i]))
        j = pick(p0[i], cur[i], rem, C, pw); out["words + prior"].append(np.linalg.norm(rule(p0[i], C[j], tab, tau) - land[i]))
        gpos = p0[i] + 0.95 * (C[j] - p0[i])
        for w_ in wgrid: wres[w_].append(np.linalg.norm(tab + w_ * (gpos - tab) - land[i]))
        j = int(np.argmax(pw)); out["prior only"].append(np.linalg.norm(rule(p0[i], C[j], tab, tau) - land[i]))
        S = sal[page]; j = pick(p0[i], cur[i], rem, S, np.ones(len(S))); out["saliency"].append(np.linalg.norm(rule(p0[i], S[j], tab, tau) - land[i]))
        out["cover words"].append(float(np.linalg.norm(C - land[i], axis=1).min() < 1.0)); out["cover sal"].append(float(np.linalg.norm(S - land[i], axis=1).min() < 1.0))
    res[tau] = {m: np.mean(v_) for m, v_ in out.items()}; print(f"\ntau = {tau} ms ({len(out['tables'])} saccades of the test participants)")
    for m in ("tables", "words flat", "words + prior", "prior only", "saliency"): print(f"  {m:14s} {res[tau][m]:.2f} deg")
    print("  words + prior, weight toward the candidate w (landing = tables + w * (0.95 * candidate - tables)): " + "  ".join(f"w={w_}: {np.mean(v_):.2f}" for w_, v_ in wres.items()))
    print(f"  coverage < 1 deg: words {100 * res[tau]['cover words']:.0f} %, saliency peaks {100 * res[tau]['cover sal']:.0f} %")
