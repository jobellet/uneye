"""Figures of foundation/paper_figs.py results: docs/figs_paper/{n_labeled.png, subjects.png}"""
import json, os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figs_paper"); R = json.load(open(os.path.join(OUT, "results.json")))
A = {}
for k, v in R["A"].items(): A.setdefault(int(k.split("_")[0]), []).append(v)
if A:
    Ns = sorted(A); fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for j, (nm, c) in enumerate((("F1", "#d9534f"), ("Cohen's kappa", "#337ab7"))):
        for N in Ns: ax[j].scatter([N] * len(A[N]), [v[j] for v in A[N]], s=12, color=c, alpha=.4)
        m = [np.mean([v[j] for v in A[N]]) for N in Ns]; s = [np.std([v[j] for v in A[N]]) for N in Ns]
        ax[j].errorbar(Ns, m, s, color=c, marker="o", capsize=3); ax[j].set_xscale("log"); ax[j].set_xlabel("number of labeled trials (1 s each)"); ax[j].set_ylabel(nm); ax[j].set_ylim(0.4, 1); ax[j].grid(alpha=.3)
    fig.suptitle("Dataset 1: train on N trials of set B, test on set A (2-channel BiTCN)"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "n_labeled.png"), dpi=150)
if R["B"]:
    rows = [k for k in R["B"] if k != "all"]; rows = sorted(rows, key=float) + (["all"] if "all" in R["B"] else []); n = len(R["B"][rows[0]])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for j, nm in enumerate(("F1", "Cohen's kappa")):
        M = np.array([[v[j] for v in R["B"][r]] for r in rows]); im = ax[j].imshow(M, vmin=0.5, vmax=1, cmap="YlGnBu")
        for i in range(M.shape[0]):
            for k in range(M.shape[1]): ax[j].text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=7, color="k" if M[i, k] < .85 else "w")
        ax[j].set_xticks(range(n)); ax[j].set_xticklabels([str(i + 1) for i in range(n)]); ax[j].set_yticks(range(len(rows))); ax[j].set_yticklabels([r if r == "all" else str(int(float(r))) for r in rows])
        ax[j].set_xlabel("test subject (set B)"); ax[j].set_ylabel("training subject (33 trials, set A)"); ax[j].set_title(nm); plt.colorbar(im, ax=ax[j], fraction=.046)
        d = np.mean([M[i, i] for i in range(n)]); o = np.mean([M[i, k] for i in range(n) for k in range(n) if i != k]); ax[j].set_xlabel(f"test subject (set B)    same subject {d:.2f}, other subjects {o:.2f}")
    fig.suptitle("Dataset 4: generalization between subjects (2-channel BiTCN)"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "subjects.png"), dpi=150)
