#!/usr/bin/env python3
"""All slide figures, one function per figure, German labels, 16:9, PNG + SVG into docs/slides/figs/.

Run from the repository root:  python docs/slides/make_figures.py
Inputs (made by scripts that were run, see docs/ROADMAP.md):
  docs/slides/data/fault_injection.csv   cpp/build/fault_injection --trials 120 --out ... --trace ...   (Figures 3 and 4)
  docs/slides/data/trace_frozen.csv      cpp/build/fault_injection --only frozen --trace-kind frozen --trace ...  (dataset 1)
  docs/slides/data/equivalence_causal.npz   python cpp/scripts/equivalence.py  (Figure 1)
  docs/slides/data/latency_{window,nowindow}.bin   cpp/build/bench_latency --samples 1000000 [--no-window] --out ...  (Figure 2)
Figure 5 (architecture) is drawn from fixed text.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "figs")
# Okabe-Ito (colour-blind safe)
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", red="#D55E00", purple="#CC79A7", grey="#7f7f7f")
SYS = {"nn": ("Nur Netz", OI["red"]), "engine": ("Engine ohne Wächter", OI["orange"]), "guard": ("Engine + Wächter", OI["blue"])}
plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17, "legend.fontsize": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".png/.svg"))


def label_de(kind, level):
    lv = f"{level:g}".replace(".", ",")
    return {"none": "keine Störung", "noise": f"Rauschen σ={lv}°", "burst": f"Störimpulse {lv}°", "dropout": f"Signalausfälle p={lv}",
            "spikes": f"Spitzen p={lv}", "saturation": f"Sättigung ±{lv}°", "frozen": "eingefrorenes Signal", "drift": f"Drift {lv}°/s",
            "scale": f"Skalierung ×{lv}", "rate_wrong": "500 Hz als 1 kHz", "rate_correct": "500 Hz, richtige Zeit",
            "deadline": f"Zeitüberschreitung {float(level) * 100:g} %".replace(".", ","), "net_stuck": "Netz hängt (p=1)"}[kind]


def pooled(df):
    """both datasets together: false events per minute weighted by minutes, dangerous events summed, time in DEGRADED/SAFE weighted"""
    g = []
    for (k, lv, s), d in df.groupby(["perturbation", "level", "system"], sort=False):
        w = d["minutes"]
        g.append(dict(perturbation=k, level=lv, system=s, false_per_min=np.average(d["false_per_min"], weights=w), dangerous=int(d["dangerous"].sum()),
                      recall=np.average(d["recall"], weights=w), degraded=np.average(d["degraded"], weights=w), safe=np.average(d["safe"], weights=w)))
    return pd.DataFrame(g)


def fig1_equivalence():
    """|p(saccade) C++ - PyTorch| per sample, causal TCN, 205 sequences x 1000 samples (cpp/scripts/equivalence.py)"""
    d = np.load(os.path.join(DATA, "equivalence_causal.npz"))
    fig, ax = plt.subplots(figsize=(16, 9))
    bins = np.logspace(-10, -4, 80)
    sets = (("Cpp_vs_Py64", "C++ (float32) gegen Python float64", OI["blue"]), ("Py32_vs_Py64", "Python float32 gegen Python float64", OI["orange"]),
            ("Cpp_vs_Py32", "C++ gegen Python float32", OI["grey"]))
    for k, lab, c in sets:
        e = d[k]; e = np.maximum(e, 1e-10)
        ax.hist(e, bins=bins, histtype="step", lw=2.5, color=c, label=f"{lab}: Max {d[k].max():.1e}".replace(".", ",").replace("e-0", "e-"))
    ax.axvline(5e-5, color=OI["red"], lw=2.5); ax.text(5.6e-5, ax.get_ylim()[1] * 0.4 if ax.get_yscale() == "linear" else 1e4, "Toleranz im Test\n5·10⁻⁵", color=OI["red"], fontsize=15)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Absoluter Fehler der Sakkaden-Wahrscheinlichkeit"); ax.set_ylabel("Anzahl Proben")
    ax.legend(loc="center right", bbox_to_anchor=(0.97, 0.62), frameon=False, fontsize=14)
    ax.text(1.05e-10, 3e4, "≤ 10⁻¹⁰\n(inkl. exakt gleich)", fontsize=11, color=OI["grey"], va="bottom")
    ax.set_title("C++ rechnet so genau wie PyTorch selbst: Fehler ≤ 6·10⁻⁶, 0 von 205 000 Labels verschieden", loc="left", fontsize=19)
    fig.text(0.02, -0.06, "Kausales TCN, 200 echte Sequenzen (Set B, Datensätze 1+2) + 5 Stresssequenzen, je 1000 Proben. C++ läuft Probe für Probe (Streaming), PyTorch über die ganze Sequenz.\n"
             "Gleicher Build (clang 22.1.8 -O3, M1): bitgleich reproduzierbar; -O0 oder -ffp-contract=off ändern bis zu 3,3·10⁻⁶. Kein Vergleich Schicht für Schicht.",
             fontsize=12, color=OI["grey"])
    save(fig, "fig1_paritaet")


def fig2_latency():
    """per-sample time of push() + guard, 1e6 samples, with and without the window network (cpp/build/bench_latency)"""
    lat = {k: np.fromfile(os.path.join(DATA, f"latency_{k}.bin"), dtype=np.float32) / 1000.0 for k in ("nowindow", "window")}   # µs
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.25))
    bins = np.logspace(np.log10(30), np.log10(60000), 120)
    for k, lab, c in (("nowindow", "ohne Fensternetz", OI["sky"]), ("window", "mit Fensternetz (alle 10 Proben)", OI["blue"])):
        v = lat[k]; q = np.percentile(v, [50, 99, 99.9])
        a1.hist(v, bins=bins, color=c, alpha=0.75, label=f"{lab}: Median {q[0]:.0f} µs · p99 {q[1]:.0f} µs · p99,9 {q[2]:.0f} µs · Max {v.max() / 1000:.1f} ms".replace(".", ","))
    a1.axvline(1000, color=OI["red"], lw=2.5); a1.text(1080, 3e3, "Budget 1 ms\n(1 kHz)", color=OI["red"], fontsize=15, va="top")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel("Zeit pro Probe (µs)"); a1.set_ylabel("Anzahl Proben")
    a1.legend(loc="upper left", frameon=False, fontsize=13, bbox_to_anchor=(0.0, -0.12), ncol=1)
    a1.set_title("Verteilung über 1 000 000 Proben", loc="left", fontsize=17)
    v = lat["window"][:120]
    a2.plot(np.arange(len(v)), v, color=OI["blue"], marker="o", ms=4, lw=1)
    a2.axhline(1000, color=OI["red"], lw=2.5); a2.set_ylim(0, 1100)
    a2.set_xlabel("Probe Nr."); a2.set_ylabel("Zeit (µs)"); a2.set_title("Jede 10. Probe: Fensternetz läuft mit", loc="left", fontsize=17)
    over = int((lat["window"] > 1000).sum())
    fig.suptitle(f"Zeit pro Probe: Median 78 µs, p99,9 unter 0,33 ms – aber {over} von 1 Mio. Proben über 1 ms (Betriebssystem, kein WCET)",
                 x=0.02, ha="left", fontsize=19, y=1.0)
    fig.text(0.02, -0.17, "MacBook Air M1 (8 Kerne), macOS, clang 22.1.8 -O3 -march=native, keine CPU-Bindung. Zeit mit steady_clock außerhalb der Engine gemessen, "
             "nach 2000 Proben Aufwärmen.\nEin Laptop ist nicht das Zielsystem: das beobachtete Maximum ist KEINE garantierte Obergrenze (WCET).", fontsize=12, color=OI["grey"])
    save(fig, "fig2_zeitmessung")


def fig3_fault_injection():
    """one row per perturbation (strongest level of each kind, plus the clean data), network alone vs engine + guard"""
    df = pooled(pd.read_csv(os.path.join(DATA, "fault_injection.csv")))
    keep = [("none", 0), ("noise", 0.05), ("noise", 0.1), ("burst", 0.1), ("burst", 0.5), ("spikes", 0.01), ("dropout", 0.01), ("saturation", 2.0),
            ("frozen", 0.002), ("scale", 10.0), ("rate_wrong", 1), ("deadline", 0.01), ("net_stuck", 1)]
    rows = [label_de(k, lv) for k, lv in keep][::-1]
    get = lambda s, col: [float(df[(df.system == s) & (df.perturbation == k) & np.isclose(df.level, lv)][col].iloc[0]) for k, lv in keep][::-1]
    y = np.arange(len(keep)); h = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 9), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, col, title in ((a1, "false_per_min", "Falsche Ereignisse pro Minute"), (a2, "dangerous", "Gefährliche Ausgaben (Anzahl)")):
        for off, s in ((h / 2, "nn"), (-h / 2, "guard")):
            v = get(s, col)
            ax.barh(y + off, v, h, color=SYS[s][1], label=SYS[s][0])
            for yi, vi in zip(y + off, v):
                ax.text(vi * 1.08 + 0.3, yi, f"{vi:.0f}", va="center", fontsize=12, color=SYS[s][1], fontweight="bold" if s == "guard" else "normal")
        ax.set_xscale("symlog", linthresh=10); ax.set_title(title, loc="left", fontsize=17)
        ax.set_xlim(0, 2000); ax.tick_params(axis="y", length=0)
    a1.set_yticks(y); a1.set_yticklabels(rows)
    h_, l_ = a2.get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.55, 0.03), ncol=2, frameon=False)
    zeros = sum(v == 0 for v in get("guard", "dangerous"))
    fig.suptitle(f"Fehlerinjektion: mit Wächter 0 gefährliche Ausgaben in {zeros} von {len(keep)} Fällen – nicht bei schwachen\nStörimpulsen, ×10-Skalierung und 500-Hz-Daten mit 1-kHz-Zeitstempeln", x=0.02, ha="left", fontsize=19, y=1.02)
    fig.text(0.02, -0.07, "Testdaten: Set B, Datensätze 1+2, je 120 Versuche (1 kHz). Gefährlich = auf dem UNGESTÖRTEN Signal physiologisch unmöglich (keine Bewegung < 10°/s, > 40°, > 150 ms).\n"
             "MacBook Air M1, clang 22.1.8 -O3, feste Seeds. Alle 24 Fälle: docs/slides/data/fault_injection.csv. Rauschgrenze des Wächters (16°/s) nur auf Set A kalibriert.", fontsize=12, color=OI["grey"])
    save(fig, "fig3_fehlerinjektion")


def fig4_example_trace():
    """frozen signal (200 ms, tracker stuck): the network alone reports saccades, the guard goes SAFE and reports none"""
    t = pd.read_csv(os.path.join(DATA, "trace_frozen.csv"))
    frozen = (t.x_in.diff() == 0) & (t.y_in.diff() == 0) if "y_in" in t else (t.x_in.diff() == 0)
    run = frozen.astype(int).groupby((~frozen).cumsum()).cumsum()
    c = int(np.argmax(run.values >= 150))                     # inside the first long frozen period
    s = t.iloc[max(c - 800, 0): c + 1700].reset_index(drop=True)
    ms = np.arange(len(s))
    fig, ax = plt.subplots(4, 1, figsize=(16, 9), sharex=True, gridspec_kw=dict(height_ratios=[2.6, 1.3, 0.9, 1.3], hspace=0.18))
    ax[0].plot(ms, s.x_clean, color="black", lw=1.4, label="ungestörtes Signal")
    ax[0].plot(ms, s.x_in, color=OI["red"], lw=2.2, alpha=0.75, label="Eingang (200 ms eingefroren)")
    ax[0].set_ylabel("Position x (°)"); ax[0].legend(loc="upper left", frameon=False, ncol=2)
    ax[0].set_title("Beispiel eingefrorenes Signal: der Wächter geht auf SAFE und meldet nichts –\nkeine falsche Sakkade, aber auch die echte bei 780 ms fehlt (Preis der Sicherheit)", loc="left", fontsize=18)
    ax[0].text(0.5, 0.04, "Lücke = Grenze zwischen zwei Versuchen", transform=ax[0].transAxes, ha="center", fontsize=11, color=OI["grey"])
    ax[1].plot(ms, s.p_nn, color=OI["red"], lw=1.2); ax[1].axhline(0.5, color=OI["grey"], lw=0.8, ls="--")
    ax[1].set_ylabel("p(Sakkade)\nnur Netz"); ax[1].set_ylim(-0.05, 1.05)
    cols = {0: OI["green"], 1: OI["yellow"], 2: "#a50f15"}; names = {0: "NORMAL", 1: "DEGRADED", 2: "SAFE"}
    for st in (0, 1, 2):
        ax[2].fill_between(ms, 0, 1, where=s.state_guard.values == st, color=cols[st], step="mid", alpha=0.85, label=names[st], linewidth=0)
    ax[2].set_yticks([]); ax[2].set_ylabel("Wächter"); ax[2].legend(loc="upper right", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(1, 1.45))
    lanes = ((2.4, s.label_guard.values == 1, OI["blue"]), (1.2, s.label_nn.values == 1, OI["red"]), (0.0, s.human.values == 1, "black"))
    for y0, m, c_ in lanes:
        ax[3].fill_between(ms, y0, y0 + 1, where=m, color=c_, step="mid", alpha=0.8, linewidth=0)
    ax[3].set_yticks([0.5, 1.7, 2.9]); ax[3].set_yticklabels(["Mensch", "nur Netz", "mit Wächter"], fontsize=13)
    ax[3].set_ylabel("Sakkade"); ax[3].set_xlabel("Zeit (ms)")
    i = int(np.argmax(s.state_guard.values == 2)) if (s.state_guard.values == 2).any() else None
    if i is not None:
        ax[2].annotate(f"Grund: {s.reason.values[i]}", xy=(i, 0.5), xytext=(i + 80, 0.5), fontsize=13, va="center", color="white", fontweight="bold")
    save(fig, "fig4_beispiel_zeitreihe")


def fig5_architecture():
    fig, ax = plt.subplots(figsize=(16, 9)); ax.set_xlim(0, 16.2); ax.set_ylim(0, 9); ax.axis("off")
    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec="black", lw=1.2, alpha=0.9))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=11.5)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"))
    y = 5.2; h = 1.7
    box(0.1, y, 3.1, h, "Python-Forschung", "Training (PyTorch,\nM1-GPU), Auswertung", "#d9ecf7")
    box(3.6, y, 2.4, h, "Export", ".bin (float32)\n.onnx", "#eeeeee")
    box(6.4, y, 3.0, h, "C++-Engine", "kausale Netze + Physik\nkein Heap nach Start", "#fde9c4")
    box(9.8, y, 3.3, h, "Sicherheits-\nwächter", "deterministisch, 3 Zustände", "#cfe8d9")
    box(13.5, y, 2.4, h, "Ausgabe", "Labels + geprüfte\nEreignisse", "#eeeeee")
    for a, b in ((3.2, 3.6), (6.0, 6.4), (9.4, 9.8), (13.1, 13.5)):
        arrow(a, y + h / 2, b, y + h / 2)
    ax.text(8.0, 8.4, "Vom Python-Modell zu vorhersagbarem, getestetem und abgesichertem C++", ha="center", fontsize=21, fontweight="bold")
    tests = [(4.8, "Parität Py ↔ C++", "test_equivalence\nmax. Fehler 6·10⁻⁶"), (7.9, "Kein Heap", "test_zero_heap\n0 Alloz. / 100 000"),
             (11.45, "Wächter-Regeln", "test_safety_guard\n15 Abschnitte"), (14.6, "Fehlerinjektion", "fault_injection\n13 Störungsarten")]
    for x, t1, t2 in tests:
        ax.add_patch(FancyBboxPatch((x - 1.35, 1.2), 2.7, 1.9, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=OI["blue"], lw=1.6, ls="--"))
        ax.text(x, 2.55, t1, ha="center", fontsize=14, fontweight="bold", color=OI["blue"]); ax.text(x, 1.75, t2, ha="center", fontsize=12)
    for x_t, x_b in ((4.8, 4.8), (7.9, 7.9), (11.45, 11.45), (14.6, 14.6)):
        ax.annotate("", xy=(x_b, y), xytext=(x_t, 3.1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=OI["blue"], ls="--"))
    ax.text(0.3, 0.4, "Gestrichelt: automatische Tests (ctest) und Messwerkzeuge. Zeitmessung: bench_latency (Abb. 2).", fontsize=12, color=OI["grey"])
    save(fig, "fig5_architektur")


if __name__ == "__main__":
    fig1_equivalence()
    fig2_latency()
    fig3_fault_injection()
    fig4_example_trace()
    fig5_architecture()
