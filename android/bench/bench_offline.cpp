// Phase 0 benchmark for the Android offline-labeling app (task plan, phase 0).
// NOT part of the shipped app: a host/device harness that measures the full
// OfflineSession pipeline on real data: load CSV (trials x time) -> cut into
// 1 s trials -> linear resample to 1 kHz when needed -> HMM pre-label
// (cpp/app/detect_hmm, params cpp/models/universal_hmm.txt) -> agreement of
// the pre-labels against the file's own labels (eyelive::agreement) ->
// render ExpertLabels -> train the eyelive U-Net (cpp/app/train) -> report
// seconds per epoch and validation scores.
//
// Usage: bench_offline <X.csv> <Y.csv> <L.csv> <hmm_params.txt> <train_seconds> [threads]
//   X.csv / Y.csv / L.csv: trials x time matrices, comma separated (one row
//   per trial), or a single column (cut into 1 s trials at the declared rate).
//   The sampling rate is taken from --rate (default 1000); anything that is
//   not 500 or 1000 Hz is resampled to 1 kHz with linear interpolation.
//
// Build (host): see android/build_android.sh, target bench_offline.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "eyelive/annotation.hpp"
#include "eyelive/detector.hpp"
#include "eyelive/hmm_detector.hpp"   // not in the frozen contracts: pulls the class so the static registrar survives linking
#include "eyelive/trace.hpp"
#include "eyelive/trainer.hpp"
#include "eyelive/types.hpp"

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Parse a CSV of doubles: rows = trials, columns = time. A single column is
// returned as one row.
std::vector<std::vector<double>> read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::vector<std::vector<double>> rows; std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> row; std::stringstream ss(line); std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        rows.push_back(std::move(row));
    }
    if (rows.size() == 1 && rows[0].size() > 1) { /* one row, many columns: a single trial */ }
    return rows;
}

// Linear resampling of one trial to `target_hz`.
std::vector<double> resample(const std::vector<double>& in, double from_hz, double to_hz) {
    if (std::fabs(from_hz - to_hz) < 1e-9) return in;
    const double ratio = from_hz / to_hz; const size_t n_out = size_t(std::lround(in.size() / ratio));
    std::vector<double> out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double pos = i * ratio; const size_t j = size_t(pos);
        const double frac = pos - j;
        const double a = in[std::min(j, in.size() - 1)], b = in[std::min(j + 1, in.size() - 1)];
        out[i] = a + frac * (b - a);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s X.csv Y.csv L.csv hmm_params.txt train_seconds [threads] [--rate hz]\n", argv[0]); return 2; }
    const std::string xf = argv[1], yf = argv[2], lf = argv[3], hmmf = argv[4];
    double train_seconds = std::atof(argv[5]);
    int threads = argc > 6 ? std::atoi(argv[6]) : 4;
    double rate = 1000.0;
    for (int i = 6; i + 1 < argc; ++i) if (!std::strcmp(argv[i], "--rate")) rate = std::atof(argv[i + 1]);

    const double t_start = now_s();
    auto X = read_csv(xf), Y = read_csv(yf), L = read_csv(lf);
    if (X.size() != Y.size() || X.size() != L.size()) { std::fprintf(stderr, "X/Y/L row counts differ\n"); return 2; }
    const size_t T_in = X[0].size();
    for (auto& r : X) if (r.size() != T_in) { std::fprintf(stderr, "ragged X matrix\n"); return 2; }

    // Cut into 1 s trials at the declared rate (the app's rule for a single
    // column / long matrix; dataset files already are trials x time).
    const size_t win = size_t(rate);   // 1 s
    std::vector<std::vector<double>> Xt, Yt, Lt;
    for (size_t k = 0; k < X.size(); ++k) {
        for (size_t s = 0; s + win <= X[k].size(); s += win) {
            Xt.emplace_back(X[k].begin() + s, X[k].begin() + s + win);
            Yt.emplace_back(Y[k].begin() + s, Y[k].begin() + s + win);
            Lt.emplace_back(L[k].begin() + s, L[k].begin() + s + win);
        }
    }
    const size_t n_trials = Xt.size();
    std::printf("loaded %zu rows x %zu samples @ %.0f Hz -> %zu trials of 1 s\n", X.size(), T_in, rate, n_trials);

    // Resample every trial to 1 kHz (the HMM has parameters only for 500 / 1000 Hz;
    // the app shows HmmDetector::error() instead when the rate is unsupported).
    std::vector<std::vector<double>> Xr, Yr, Lr;
    for (size_t k = 0; k < n_trials; ++k) {
        Xr.push_back(resample(Xt[k], rate, 1000.0));
        Yr.push_back(resample(Yt[k], rate, 1000.0));
        auto l = resample(Lt[k], rate, 1000.0);
        for (auto& v : l) v = v > 0.5 ? 1.0 : 0.0;   // threshold for labels; only used for the agreement report
        Lr.push_back(std::move(l));
    }
    const size_t T = Xr[0].size();

    // HMM pre-labeling on every trial (batch mode = the offline path of the app).
    // The registrar lives in an anonymous namespace of the module: instantiate
    // the class once so the linker keeps the object file (same trick as
    // engine.cpp::register_detectors).
    (void)hmmf;   // hmm_accel carries the universal parameters for 500 / 1000 Hz in code
    auto det = std::make_unique<eyelive::HmmDetector>();
    std::string herr;
    double t_hmm = 0; size_t n_events = 0, n_sacc = 0;
    std::vector<std::vector<eyelive::Event>> pre(n_trials);
    {
        const double t0 = now_s();
        eyelive::SessionInfo info; info.rate_hz = 1000.0; info.n_classes = 3;
        for (size_t k = 0; k < n_trials; ++k) {
            eyelive::VectorTrace tr(info);
            for (size_t i = 0; i < T; ++i) { eyelive::Sample s; s.t = double(i) / 1000.0; s.x = float(Xr[k][i]); s.y = float(Yr[k][i]); tr.push(s); }
            det->reset(info);
            det->detect_all(tr, pre[k]);
            n_events += pre[k].size();
            for (auto& e : pre[k]) if (e.cls == 1) ++n_sacc;
        }
        t_hmm = now_s() - t0;
    }
    std::printf("HMM pre-label: %zu events (%zu saccades) over %zu trials in %.2f s (%.1f ms per trial)\n", n_events, n_sacc, n_trials, t_hmm, 1000.0 * t_hmm / n_trials);

    // Agreement of the pre-labels against the file's labels (the report the app
    // shows next to the pre-labels; eyelive::agreement matches foundation/compare.py).
    {
        size_t tp = 0, fp = 0, fn = 0; std::vector<eyelive::Event> expert;
        double f1_sum = 0;
        for (size_t k = 0; k < n_trials; ++k) {
            expert.clear();
            for (size_t i = 0; i < T;) {
                if (Lr[k][i] > 0.5) { size_t j = i; while (j < T && Lr[k][j] > 0.5) ++j; eyelive::Event e; e.onset = int64_t(i); e.offset = int64_t(j - 1); e.cls = 1; expert.push_back(e); i = j; } else ++i;
            }
            auto a = eyelive::agreement(pre[k], expert, int64_t(T), 1);
            tp += a.tp; fp += a.fp; fn += a.fn; f1_sum += a.event_f1;
            if (k < 10 || k == n_trials - 1)
                std::printf("  trial %3zu: pre %3zu events, file-labels %3zu, event F1 %.3f kappa %.3f\n", k, pre[k].size(), expert.size(), a.event_f1, a.kappa);
        }
        std::printf("  mean event F1 vs file labels: %.3f (pooled tp %zu fp %zu fn %zu)\n", f1_sum / n_trials, tp, fp, fn);
    }

    // ExpertLabels per trial: load the pre-labels, mark reviewed, add the
    // events (in the app the expert confirms / corrects them first).
    std::vector<eyelive::ExpertLabels> labels(n_trials, eyelive::ExpertLabels(3));
    {
        const double t0 = now_s();
        for (size_t k = 0; k < n_trials; ++k) {
            labels[k].load_prelabels(pre[k], 0, int64_t(T - 1));
            labels[k].mark_reviewed(0, int64_t(T - 1), true, false);
            for (auto& e : pre[k]) labels[k].add(e.onset, e.offset, e.cls);
        }
        std::printf("ExpertLabels: %zu events over %zu trials, built in %.3f s\n", labels[0].events().size() * n_trials, n_trials, now_s() - t0);
    }

    // Training data: rendered reviewed ranges of the first trials, up to
    // `train_seconds`, positions in degrees (deg_per_unit applied by the caller).
    std::vector<eyelive::TrainTrace> traces;
    {
        double used = 0;
        for (size_t k = 0; k < n_trials && used < train_seconds; ++k) {
            std::vector<uint8_t> lab;
            labels[k].render(0, int64_t(T - 1), true, lab);
            const double take = std::min(1.0, train_seconds - used);
            const size_t n = size_t(take * 1000.0);
            eyelive::TrainTrace t; t.rate_hz = 1000.0;
            for (size_t i = 0; i < n; ++i) { t.x.push_back(float(Xr[k][i])); t.y.push_back(float(Yr[k][i])); t.labels.push_back(lab[i]); }
            traces.push_back(std::move(t)); used += take;
        }
        std::printf("train set: %zu trace(s), %.1f s of reviewed data\n", traces.size(), used);
    }

    // Train with the existing trainer, exactly the app's phase 3.
    eyelive::TrainConfig cfg; cfg.n_classes = 3; cfg.n_threads = threads; cfg.max_epochs = 8; cfg.deg_per_unit = 1.0;
    auto trainer = eyelive::make_trainer();
    double t_train = 0, s_per_epoch = 0; int epochs = 0; eyelive::TrainProgress last;
    {
        const double t0 = now_s(); double t_first = 0, t_last = 0;
        trainer->start(traces, cfg, [&](const eyelive::TrainProgress& p) {
            if (!p.finished) { ++epochs; if (epochs == 1) t_first = p.elapsed_s; t_last = p.elapsed_s; std::printf("  epoch %d train %.4f val %.4f valF1 %.3f kappa %.3f (%.2f s)\n", p.epoch, p.train_loss, p.val_loss, p.val_event_f1, p.val_kappa, p.elapsed_s); }
            last = p;
        });
        while (trainer->running()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        t_train = now_s() - t0;
        if (epochs > 1) s_per_epoch = (t_last - t_first) / (epochs - 1);
        else s_per_epoch = t_first;
    }
    std::printf("training: %d epochs in %.2f s -> %.3f s/epoch (%d threads), finished=%d failed=%d msg='%s'\n",
                epochs, t_train, s_per_epoch, threads, (int)last.finished, (int)last.failed, last.message.c_str());
    std::string serr;
    if (trainer->save_model("/tmp/bench_model.bin", &serr)) std::printf("saved /tmp/bench_model.bin\n");
    else std::fprintf(stderr, "save_model: %s\n", serr.c_str());
    std::printf("TOTAL pipeline: %.2f s\n", now_s() - t_start);
    return 0;
}
