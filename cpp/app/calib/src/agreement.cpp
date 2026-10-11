#include "eyelive/annotation.hpp"
#include <algorithm>
#include <vector>

namespace eyelive {

// The task asks to implement scoring internally and not depend on T07.
// Event F1: a predicted run of >= 3 samples that overlaps a human saccade is a hit.
// precision = nok / npred, recall = hits / nt. f1 = 2 * rec * prec / (rec + prec)
// Cohen's kappa sample by sample, class-vs-rest.
AgreementStats agreement(const std::vector<Event>& detected, const std::vector<Event>& expert, int64_t n_samples, uint8_t cls) {
    AgreementStats stats;
    if (n_samples <= 0) return stats;

    std::vector<bool> exp_mask(n_samples, false);
    std::vector<std::pair<int64_t, int64_t>> exp_runs;
    for (const auto& ev : expert) {
        if (ev.cls == cls) {
            int64_t s = std::max<int64_t>(0, ev.onset);
            int64_t e = std::min<int64_t>(n_samples - 1, ev.offset);
            if (s <= e) {
                exp_runs.push_back({s, e});
                for (int64_t i = s; i <= e; ++i) exp_mask[i] = true;
            }
        }
    }

    std::vector<bool> det_mask(n_samples, false);
    std::vector<std::pair<int64_t, int64_t>> det_runs;
    for (const auto& ev : detected) {
        // filter for class not needed for detected? detected typically only gives one class in calibration.
        // Python metric checks if run >= 3.
        int64_t s = std::max<int64_t>(0, ev.onset);
        int64_t e = std::min<int64_t>(n_samples - 1, ev.offset);
        if (s <= e) {
            if (e - s + 1 >= 3) {
                det_runs.push_back({s, e});
            }
            // For kappa, does it include all runs or only >= 3?
            // Python metrics: P = (pred & mask); pe = [r for r in runs(pred) if r[1]-r[0]+1 >= 3]
            // Wait, Python's P (and kappa) uses `pred`, not filtered by length.
            // Oh actually, Python metrics uses `P > thr` which is unfiltered for kappa.
            for (int64_t i = s; i <= e; ++i) det_mask[i] = true;
        }
    }

    int hits = 0, nok = 0;
    std::vector<bool> used(det_runs.size(), false);
    for (const auto& tr : exp_runs) {
        for (size_t j = 0; j < det_runs.size(); ++j) {
            if (!used[j] && det_runs[j].first <= tr.second && det_runs[j].second >= tr.first) {
                used[j] = true;
                hits++;
                break;
            }
        }
    }
    for (bool u : used) if (u) nok++;

    int n_true = exp_runs.size();
    int n_pred = det_runs.size();

    double rec = n_true > 0 ? (double)hits / n_true : 0;
    double prec = n_pred > 0 ? (double)nok / n_pred : 0;
    stats.event_f1 = (rec + prec > 0) ? (2 * rec * prec / (rec + prec)) : 0;

    int64_t tp = 0, fp = 0, fn = 0, tn = 0;
    for (int64_t i = 0; i < n_samples; ++i) {
        if (exp_mask[i] && det_mask[i]) tp++;
        else if (!exp_mask[i] && det_mask[i]) fp++;
        else if (exp_mask[i] && !det_mask[i]) fn++;
        else tn++;
    }
    stats.tp = tp; stats.fp = fp; stats.fn = fn;

    double n = n_samples;
    double po = (tp + tn) / n;
    double pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n);
    stats.kappa = (po - pe) / std::max(1 - pe, 1e-12);

    return stats;
}

} // namespace eyelive
