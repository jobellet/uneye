"""Metrics for online saccade labeling.

Sample level (every time bin is one decision): Cohen's kappa, MCC, F1, precision, recall, PR-AUC.
Event level (what the experimenter cares about): event recall / precision / F1 (overlap matching),
alarm delay (time from true onset to `min_event` consecutive saccade labels) and false alarms per minute.
"""
import numpy as np
from sklearn.metrics import average_precision_score


def runs(b):
    d = np.diff(np.r_[0, np.asarray(b).astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def confusion(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    return (truth & pred).sum(), (~truth & pred).sum(), (truth & ~pred).sum(), (~truth & ~pred).sum()


def sample_metrics(tp, fp, fn, tn):
    tp, fp, fn, tn = (float(x) for x in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    kappa = (po - pe) / max(1 - pe, 1e-12)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return dict(kappa=kappa, mcc=mcc, f1=f1, precision=prec, recall=rec)


def event_counts(P, L, thr, min_event=3):
    """per-recording event matching. Returns hits, n_true, n_pred, n_pred_ok, delays(samples), onset_err(samples)"""
    pred, truth = P > thr, L > 0
    hits = n_true = n_pred = n_ok = 0
    delay, onset_err = [], []
    for k in range(truth.shape[0]):
        te = runs(truth[k])
        pe = [r for r in runs(pred[k]) if r[1] - r[0] + 1 >= min_event]
        used = set(); n_true += len(te); n_pred += len(pe)
        for (s, e) in te:
            for j, (ps, pe2) in enumerate(pe):
                if j not in used and ps <= e and pe2 >= s:
                    used.add(j); hits += 1
                    onset_err.append(ps - s); delay.append(ps + min_event - 1 - s)
                    break
        n_ok += len(used)
    return hits, n_true, n_pred, n_ok, delay, onset_err


def evaluate_probs(P, L, fs, thr=0.5, min_event=3, with_ap=True):
    """P, L: (n_trials, T) saccade probability / labels (>0 = saccade) of ONE sampling rate."""
    m = sample_metrics(*confusion(P > thr, L > 0))
    if with_ap:
        m["pr_auc"] = float(average_precision_score((L > 0).ravel(), P.ravel()))
    hits, nt, npred, nok, delay, onerr = event_counts(P, L, thr, min_event)
    ms = 1000.0 / fs
    m["ev_recall"] = hits / max(nt, 1)
    m["ev_precision"] = nok / max(npred, 1)
    m["ev_f1"] = 2 * m["ev_recall"] * m["ev_precision"] / max(m["ev_recall"] + m["ev_precision"], 1e-12)
    m["alarm_delay_ms"] = float(np.median(delay)) * ms if delay else float("nan")
    m["false_alarms_per_min"] = (npred - nok) / (P.size / fs / 60)
    m["n_events"] = nt
    return m


def tune_threshold(P, L, metric="kappa", grid=np.arange(0.1, 0.91, 0.05)):
    """threshold maximizing `metric` (kappa by default) on validation data"""
    best = (-2, 0.5)
    for t in grid:
        v = sample_metrics(*confusion(P > t, L > 0))[metric]
        best = max(best, (v, float(t)))
    return best[1]


def pool(results):
    """average a list of per-dataset metric dicts, weighted by the number of true events"""
    keys = [k for k in results[0] if k != "n_events"]
    w = np.array([r["n_events"] for r in results], float); w /= w.sum()
    return {k: float(np.nansum([r[k] * wi for r, wi in zip(results, w)])) for k in keys}
