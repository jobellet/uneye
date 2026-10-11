import os
import json
import numpy as np

def runs(b):
    d = np.diff(np.r_[0, np.asarray(b).astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

def compute_agreement(pred_events, truth_events, n_samples):
    # Pred and Truth boolean arrays
    P = np.zeros(n_samples, dtype=bool)
    for s, e in pred_events:
        P[s:e+1] = True

    L = np.zeros(n_samples, dtype=bool)
    for s, e in truth_events:
        L[s:e+1] = True

    # sample metrics
    truth = L
    pred = P
    tp = (truth & pred).sum()
    fp = (~truth & pred).sum()
    fn = (truth & ~pred).sum()
    tn = (~truth & ~pred).sum()

    n = tp + fp + fn + tn
    po = (tp + tn) / max(n, 1)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n ** 2, 1)
    if n == 0:
        kappa = 0.0
    else:
        kappa = (po - pe) / max(1 - pe, 1e-12)

    # event counts
    te = runs(L)
    pe = [r for r in runs(P) if r[1] - r[0] + 1 >= 3]

    hits = 0
    used = set()
    for (s, e) in te:
        for j, (ps, pe2) in enumerate(pe):
            if j not in used and ps <= e and pe2 >= s:
                used.add(j)
                hits += 1
                break

    nt = len(te)
    npred = len(pe)
    nok = len(used)

    ev_recall = hits / max(nt, 1)
    ev_precision = nok / max(npred, 1)
    ev_f1 = 2 * ev_recall * ev_precision / max(ev_recall + ev_precision, 1e-12)

    return {
        "n_samples": n_samples,
        "pred": pred_events,
        "truth": truth_events,
        "tp": hits, # note: event tp
        "fp": npred - nok, # event fp
        "fn": nt - hits, # event fn
        "event_f1": ev_f1,
        "kappa": kappa
    }

def main():
    cases = []

    # 1. Empty sets
    cases.append(compute_agreement([], [], 100))
    cases.append(compute_agreement([(10, 20)], [], 100))
    cases.append(compute_agreement([], [(10, 20)], 100))

    # 2. Perfect match
    cases.append(compute_agreement([(10, 20)], [(10, 20)], 100))
    cases.append(compute_agreement([(10, 20), (30, 40)], [(10, 20), (30, 40)], 100))

    # 3. Partial overlap (hit)
    cases.append(compute_agreement([(15, 25)], [(10, 20)], 100))
    cases.append(compute_agreement([(5, 15)], [(10, 20)], 100))
    cases.append(compute_agreement([(10, 15)], [(10, 20)], 100)) # pred inside truth
    cases.append(compute_agreement([(5, 25)], [(10, 20)], 100)) # truth inside pred

    # 4. Too short prediction (doesn't count as hit for event metric)
    cases.append(compute_agreement([(10, 11)], [(10, 20)], 100)) # len 2
    cases.append(compute_agreement([(10, 10)], [(10, 20)], 100)) # len 1

    # 5. Multiple predictions matching one truth
    cases.append(compute_agreement([(10, 15), (17, 22)], [(9, 25)], 100))

    # 6. One prediction matching multiple truths
    cases.append(compute_agreement([(8, 25)], [(10, 15), (20, 22)], 100))

    # 7. Long Andersson-like traces
    np.random.seed(42)
    for _ in range(15):
        n_samples = 500000
        p_events = []
        t_events = []
        for i in range(200):
            s = np.random.randint(100, n_samples - 100)
            e = s + np.random.randint(5, 100)
            p_events.append((s, e))
        for i in range(200):
            s = np.random.randint(100, n_samples - 100)
            e = s + np.random.randint(5, 100)
            t_events.append((s, e))

        p_events = sorted(p_events)
        t_events = sorted(t_events)

        # fix overlaps
        for lst in [p_events, t_events]:
            for i in range(len(lst)-1, 0, -1):
                if lst[i][0] <= lst[i-1][1]:
                    lst.pop(i)

        cases.append(compute_agreement(p_events, t_events, n_samples))

    # Ensure at least 30 cases
    while len(cases) < 30:
        cases.append(compute_agreement([(1, 5)], [(10, 20)], 50))

    # Export
    out_dir = os.path.join(os.path.dirname(__file__), "..", "app", "annot", "tests")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "golden_agreement.txt"), "w") as f:
        json.dump(cases, f, indent=2)

if __name__ == "__main__":
    main()
