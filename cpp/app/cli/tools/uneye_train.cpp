// uneye_train: command-line U'n'Eye training, cross-validation and prediction with the native C++ trainer (eyelive::make_trainer, the same recipe as
// the web / Android labelers: cpp/app/train), no Python and no PyTorch. One process = one training, so cross-validation folds run in parallel as
// separate processes (one CPU each: --threads 1), e.g. one SLURM array task per fold.
//
// Data: the U'n'Eye CSV convention, as exported by the labelers ("Export -> CSV"): X.csv and Y.csv = one trial per row (positions, trials may have
// different lengths, empty / nan = invalid sample), L.csv = one row per trial of labels (0 fixation, 1 saccade, 2 blink; -1, empty or nan = not reviewed,
// never trained on nor scored). Optional groups file (one value per trial, e.g. the session): the trials of a group are always in the same fold.
//
//   uneye_train train   --x X.csv --y Y.csv --labels L.csv --out model.bin [--init combined.bin] [--classes 2] [--rate 1000] [--savgol 0]
//                       [--max-epochs 500] [--patience 4] [--lr 0.001] [--rotations 4] [--no-rotation] [--val 0.2] [--threads 0] [--seed 1]
//   uneye_train cv      (same options) --folds 5 --fold 0 [--groups G.csv] [--cv-seed 0] --metrics fold0.json [--pred-out fold0_pred.csv]
//                       trains on the other folds, predicts the trials of fold --fold and scores them on their reviewed samples
//   uneye_train predict --model model.bin --x X.csv --y Y.csv --out pred.csv [--classes 2] [--rate 1000] [--savgol 0] [--probs]
//
// Scores (cv): event F1 of the saccade class (a predicted run of >= 3 samples overlapping a labelled saccade, one match per event), precision and
// recall, Cohen's kappa sample by sample (saccade vs rest), on the reviewed samples of the held-out trials only: the metric of the labelers
// (cpp/web/uneye_web.cpp) and of foundation/compare.py. Prediction: argmax of the softmax (2 classes: p(saccade) > 0.5), no post-processing.
// --savgol W: cubic Savitzky-Golay smoothing of the positions over W samples (odd; polynomial at the edges, invalid samples held at the last valid value
// and invalid again afterwards), the labelers' display filter, applied before training AND before prediction (0 = off). Use the same W everywhere.
// Exit code 0 on success, 1 on a failure (data, training), 2 for a bad command line.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "eyelive/trainer.hpp"
#include "eyelive/types.hpp"
#include "eyelive/train/model_io.hpp"
#include "eyelive/train/unet.hpp"

using namespace eyelive;

namespace {

using Rows = std::vector<std::vector<double>>;

bool read_csv(const std::string& path, Rows& rows, std::string* err) {
    std::ifstream f(path);
    if (!f) { *err = "cannot open " + path; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::vector<double> r;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            const auto a = cell.find_first_not_of(" \t"), b = cell.find_last_not_of(" \t");
            const std::string c = a == std::string::npos ? "" : cell.substr(a, b - a + 1);
            if (c.empty() || c == "nan" || c == "NaN" || c == "NAN") { r.push_back(std::nan("")); continue; }
            char* end = nullptr;
            const double v = std::strtod(c.c_str(), &end);
            r.push_back(end == c.c_str() ? std::nan("") : v);
        }
        rows.push_back(std::move(r));
    }
    return true;
}

bool read_lines(const std::string& path, std::vector<std::string>& out, std::string* err) {
    std::ifstream f(path);
    if (!f) { *err = "cannot open " + path; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(line);
    }
    return true;
}

// cubic Savitzky-Golay smoothing (polynomial fit at the edges, as scipy mode="interp" and app.js applySavitzkyGolay); invalid samples held, then NaN again
std::vector<double> savgol(const std::vector<double>& v, int w) {
    const int T = int(v.size());
    if (w < 5 || T < w) return v;
    if (w % 2 == 0) ++w;
    const int h = (w - 1) / 2;
    double VtV[4][4] = {}, M[4][8];
    std::vector<std::array<double, 4>> V(w);
    for (int i = 0; i < w; ++i) { const double x = i - h; V[i] = {1, x, x * x, x * x * x}; }
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) for (int i = 0; i < w; ++i) VtV[a][b] += V[i][a] * V[i][b];
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) { M[a][b] = VtV[a][b]; M[a][4 + b] = a == b; }
    for (int c = 0; c < 4; ++c) {
        int p = c; for (int r = c + 1; r < 4; ++r) if (std::fabs(M[r][c]) > std::fabs(M[p][c])) p = r;
        for (int k = 0; k < 8; ++k) std::swap(M[c][k], M[p][k]);
        const double d = M[c][c]; for (int k = 0; k < 8; ++k) M[c][k] /= d;
        for (int r = 0; r < 4; ++r) if (r != c) { const double fct = M[r][c]; for (int k = 0; k < 8; ++k) M[r][k] -= fct * M[c][k]; }
    }
    std::vector<double> P(4 * w);
    for (int a = 0; a < 4; ++a) for (int i = 0; i < w; ++i) { double acc = 0; for (int b = 0; b < 4; ++b) acc += M[a][4 + b] * V[i][b]; P[a * w + i] = acc; }
    std::vector<double> y(T), out(T);
    double last = 0;
    for (int i = 0; i < T; ++i) { if (std::isfinite(v[i])) last = v[i]; y[i] = last; }
    for (int i = 0; i < T; ++i) {
        const int s0 = std::min(std::max(i - h, 0), T - w);
        double c[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; ++a) for (int j = 0; j < w; ++j) c[a] += P[a * w + j] * y[s0 + j];
        const double u = i - (s0 + h);
        out[i] = std::isfinite(v[i]) ? c[0] + c[1] * u + c[2] * u * u + c[3] * u * u * u : std::nan("");
    }
    return out;
}

struct Trial { std::vector<double> x, y; std::vector<int> lab; std::string group; };

struct Args {
    std::string mode, x, y, labels, out, init, model, groups, metrics, pred_out;
    int classes = 2, savgol = 0, max_epochs = 500, patience = 4, rotations = 4, threads = 0, folds = 0, fold = -1;
    double rate = 1000, lr = 1e-3, val = 0.2;
    uint64_t seed = 1, cv_seed = 0;
    bool rotation = true, probs = false;
};

int usage(const std::string& msg) {
    std::cerr << "uneye_train: " << msg << "\n"
              << "usage: uneye_train train|cv|predict --x X.csv --y Y.csv [--labels L.csv] [--out model.bin|pred.csv] [--init model.bin] [--model model.bin]\n"
              << "       [--classes 2] [--rate 1000] [--savgol 0] [--max-epochs 500] [--patience 4] [--lr 0.001] [--rotations 4] [--no-rotation]\n"
              << "       [--val 0.2] [--threads 0] [--seed 1] [--folds K --fold i --metrics m.json [--groups G.csv] [--cv-seed 0] [--pred-out p.csv]] [--probs]\n"
              << "(see the header of cpp/app/cli/tools/uneye_train.cpp)\n";
    return 2;
}

bool parse(int argc, char** argv, Args& a, std::string* err) {
    if (argc < 2) { *err = "no mode"; return false; }
    a.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&](std::string& dst) { if (i + 1 >= argc) return false; dst = argv[++i]; return true; };
        std::string s;
        if (k == "--x") { if (!val(a.x)) return false; }
        else if (k == "--y") { if (!val(a.y)) return false; }
        else if (k == "--labels") { if (!val(a.labels)) return false; }
        else if (k == "--out") { if (!val(a.out)) return false; }
        else if (k == "--init") { if (!val(a.init)) return false; }
        else if (k == "--model") { if (!val(a.model)) return false; }
        else if (k == "--groups") { if (!val(a.groups)) return false; }
        else if (k == "--metrics") { if (!val(a.metrics)) return false; }
        else if (k == "--pred-out") { if (!val(a.pred_out)) return false; }
        else if (k == "--no-rotation") a.rotation = false;
        else if (k == "--probs") a.probs = true;
        else if (val(s)) {
            if (k == "--classes") a.classes = std::atoi(s.c_str());
            else if (k == "--rate") a.rate = std::atof(s.c_str());
            else if (k == "--savgol") a.savgol = std::atoi(s.c_str());
            else if (k == "--max-epochs") a.max_epochs = std::atoi(s.c_str());
            else if (k == "--patience") a.patience = std::atoi(s.c_str());
            else if (k == "--lr") a.lr = std::atof(s.c_str());
            else if (k == "--rotations") a.rotations = std::atoi(s.c_str());
            else if (k == "--val") a.val = std::atof(s.c_str());
            else if (k == "--threads") a.threads = std::atoi(s.c_str());
            else if (k == "--seed") a.seed = std::strtoull(s.c_str(), nullptr, 10);
            else if (k == "--cv-seed") a.cv_seed = std::strtoull(s.c_str(), nullptr, 10);
            else if (k == "--folds") a.folds = std::atoi(s.c_str());
            else if (k == "--fold") a.fold = std::atoi(s.c_str());
            else { *err = "unknown option " + k; return false; }
        } else { *err = "option " + k + " needs a value"; return false; }
    }
    return true;
}

bool load_trials(const Args& a, bool need_labels, std::vector<Trial>& out, std::string* err) {
    Rows X, Y, L;
    if (!read_csv(a.x, X, err) || !read_csv(a.y, Y, err)) return false;
    if (X.size() != Y.size()) { *err = "X and Y have different numbers of trials"; return false; }
    if (need_labels) {
        if (!read_csv(a.labels, L, err)) return false;
        if (L.size() != X.size()) { *err = "labels and X have different numbers of trials"; return false; }
    }
    std::vector<std::string> G;
    if (!a.groups.empty()) {
        if (!read_lines(a.groups, G, err)) return false;
        while (!G.empty() && G.back().empty()) G.pop_back();
        if (G.size() != X.size()) { *err = "groups and X have different numbers of trials"; return false; }
    }
    for (size_t k = 0; k < X.size(); ++k) {
        Trial t;
        const size_t n = std::min(X[k].size(), Y[k].size());
        t.x.assign(X[k].begin(), X[k].begin() + n);
        t.y.assign(Y[k].begin(), Y[k].begin() + n);
        for (size_t i = 0; i < n; ++i) if (!std::isfinite(t.x[i]) || !std::isfinite(t.y[i])) t.x[i] = t.y[i] = std::nan("");
        if (a.savgol > 0) { t.x = savgol(t.x, a.savgol); t.y = savgol(t.y, a.savgol); }
        t.lab.assign(n, -1);
        if (need_labels) for (size_t i = 0; i < std::min(n, L[k].size()); ++i) {
            const double v = L[k][i];
            t.lab[i] = (std::isfinite(v) && v >= 0 && v < a.classes) ? int(v) : -1;
        }
        t.group = G.empty() ? std::to_string(k) : G[k];
        out.push_back(std::move(t));
    }
    return true;
}

bool train(const Args& a, const std::vector<Trial>& trials, const std::vector<int>& use, const std::string& out, int* epochs, std::string* err) {
    std::vector<TrainTrace> traces;
    for (int k : use) {
        const Trial& t = trials[size_t(k)];
        bool any = false;
        TrainTrace tr;
        tr.rate_hz = a.rate;
        for (size_t i = 0; i < t.x.size(); ++i) {
            tr.x.push_back(float(t.x[i])); tr.y.push_back(float(t.y[i]));
            tr.labels.push_back(t.lab[i] < 0 ? kIgnoreLabel : uint8_t(t.lab[i]));
            any = any || t.lab[i] >= 0;
        }
        if (any) traces.push_back(std::move(tr));
    }
    if (traces.empty()) { *err = "no reviewed sample to train on"; return false; }
    TrainConfig cfg;
    cfg.n_classes = a.classes; cfg.max_seconds_of_data = 1e9; cfg.rotation_aug = a.rotation; cfg.rotations = a.rotations; cfg.val_fraction = a.val;
    cfg.max_epochs = a.max_epochs; cfg.patience = a.patience; cfg.learning_rate = a.lr; cfg.n_threads = a.threads; cfg.seed = a.seed; cfg.init_model = a.init;
    auto tr = make_trainer();
    TrainProgress last;
    tr->start(std::move(traces), cfg, [&](const TrainProgress& p) {
        last = p;
        std::fprintf(stderr, "epoch %d  train %.4f  val %.4f  val F1 %.3f  kappa %.3f  %.0f s%s%s\n", p.epoch, p.train_loss, p.val_loss, p.val_event_f1,
                     p.val_kappa, p.elapsed_s, p.message.empty() ? "" : "  ", p.message.c_str());
    });
    if (!tr->save_model(out, err)) return false;                   // waits for the worker
    if (last.failed) { *err = last.message; return false; }
    *epochs = last.epoch;
    return true;
}

// whole-trial prediction, as cpp/web/uneye_web.cpp::predict_probs_internal: first difference (NaN -> 0), speed limit 1300 deg/s, the trial cut into a
// multiple of 25 samples plus a last 25-sample window for the remainder
std::vector<float> predict_probs(train::UNet<float>& net, const Trial& t, double rate, double max_speed = 1300.0) {
    const int K = net.K;
    const size_t n = t.x.size();
    std::vector<float> dxy(2 * n, 0.f), logits(size_t(K) * n, 0.f), probs(size_t(K) * n, 0.f);
    const double lim = max_speed / rate;
    for (size_t i = 1; i < n; ++i) {
        double dx = t.x[i] - t.x[i - 1], dy = t.y[i] - t.y[i - 1];
        if (!std::isfinite(dx) || !std::isfinite(dy)) dx = dy = 0;
        const double nr = std::hypot(dx, dy);
        if (nr > lim) { dx *= lim / nr; dy *= lim / nr; }
        dxy[i] = float(dx); dxy[n + i] = float(dy);
    }
    auto run = [&](size_t from, int len, size_t keep_from) {          // forward on [from, from + len) and copy samples >= keep_from
        std::vector<float> in(2 * size_t(len), 0.f);
        for (int i = 0; i < len; ++i) if (from + size_t(i) < n) { in[size_t(i)] = dxy[from + size_t(i)]; in[size_t(len + i)] = dxy[n + from + size_t(i)]; }
        const auto& z = net.forward(in.data(), 1, len, false);
        for (int k = 0; k < K; ++k) for (int i = 0; i < len; ++i) {
            const size_t s = from + size_t(i);
            if (s >= keep_from && s < n) logits[size_t(k) * n + s] = z[size_t(k) * size_t(len) + size_t(i)];
        }
    };
    if (n < 25) run(0, 25, 0);
    else {
        const size_t t1 = n - n % 25;
        run(0, int(t1), 0);
        if (t1 < n) run(n - 25, 25, t1);
    }
    for (size_t s = 0; s < n; ++s) {
        float mx = logits[s];
        for (int k = 1; k < K; ++k) mx = std::max(mx, logits[size_t(k) * n + s]);
        double sum = 0;
        for (int k = 0; k < K; ++k) sum += probs[size_t(k) * n + s] = std::exp(logits[size_t(k) * n + s] - mx);
        for (int k = 0; k < K; ++k) probs[size_t(k) * n + s] = float(probs[size_t(k) * n + s] / sum);
    }
    return probs;
}

std::vector<int> classify(const std::vector<float>& probs, int K, size_t n) {
    std::vector<int> c(n, 0);
    for (size_t s = 0; s < n; ++s) {
        if (K == 2) { c[s] = probs[n + s] > 0.5f ? 1 : 0; continue; }
        int b = 0;
        for (int k = 1; k < K; ++k) if (probs[size_t(k) * n + s] > probs[size_t(b) * n + s]) b = k;
        c[s] = b;
    }
    return c;
}

std::vector<std::pair<size_t, size_t>> runs(const std::vector<int>& v, int cls, size_t minlen) {
    std::vector<std::pair<size_t, size_t>> r;
    for (size_t i = 0; i < v.size();) {
        if (v[i] != cls) { ++i; continue; }
        size_t j = i; while (j < v.size() && v[j] == cls) ++j;
        if (j - i >= minlen) r.push_back({i, j - 1});
        i = j;
    }
    return r;
}

bool load_model(const std::string& path, int classes, train::UNet<float>& net, std::string* err) {   // net constructed with `classes`
    (void)classes;
    net.init_random(1);
    return train::load_bin(path, net, err);
}

int cmd_train(const Args& a) {
    std::string err;
    if (a.x.empty() || a.y.empty() || a.labels.empty() || a.out.empty()) return usage("train needs --x --y --labels --out");
    std::vector<Trial> T;
    if (!load_trials(a, true, T, &err)) { std::cerr << "uneye_train: " << err << "\n"; return 1; }
    std::vector<int> all(T.size()); std::iota(all.begin(), all.end(), 0);
    int ep = 0;
    if (!train(a, T, all, a.out, &ep, &err)) { std::cerr << "uneye_train: " << err << "\n"; return 1; }
    std::cout << "saved " << a.out << " (" << ep << " epochs, " << T.size() << " trials)\n";
    return 0;
}

int cmd_cv(const Args& a) {
    std::string err;
    if (a.x.empty() || a.y.empty() || a.labels.empty() || a.folds < 2 || a.fold < 0 || a.fold >= a.folds || a.metrics.empty())
        return usage("cv needs --x --y --labels --folds K (>= 2) --fold i (0..K-1) --metrics m.json");
    std::vector<Trial> T;
    if (!load_trials(a, true, T, &err)) { std::cerr << "uneye_train: " << err << "\n"; return 1; }
    // groups -> folds: the sorted distinct groups are shuffled with --cv-seed and dealt round-robin (no group is split between folds)
    std::vector<std::string> g;
    for (const auto& t : T) g.push_back(t.group);
    std::sort(g.begin(), g.end()); g.erase(std::unique(g.begin(), g.end()), g.end());
    std::mt19937_64 rng(a.cv_seed);
    std::shuffle(g.begin(), g.end(), rng);
    std::map<std::string, int> fold_of;
    for (size_t i = 0; i < g.size(); ++i) fold_of[g[i]] = int(i % size_t(a.folds));
    std::vector<int> tr_idx, te_idx;
    for (size_t k = 0; k < T.size(); ++k) (fold_of[T[k].group] == a.fold ? te_idx : tr_idx).push_back(int(k));
    const std::string model = a.out.empty() ? a.metrics + ".bin" : a.out;
    int ep = 0;
    if (!train(a, T, tr_idx, model, &ep, &err)) { std::cerr << "uneye_train: " << err << "\n"; return 1; }
    train::UNet<float> net(a.classes, 5, 5, 1);
    if (!load_model(model, a.classes, net, &err)) { std::cerr << "uneye_train: " << err << "\n"; return 1; }
    long tp = 0, fp = 0, fn = 0, tn = 0, n_true = 0, n_pred = 0, hit_true = 0, hit_pred = 0;
    std::ofstream pred_out;
    if (!a.pred_out.empty()) pred_out.open(a.pred_out);
    for (int k : te_idx) {
        const Trial& t = T[size_t(k)];
        const size_t n = t.x.size();
        auto c = classify(predict_probs(net, t, a.rate), a.classes, n);
        if (pred_out) { pred_out << k; for (size_t s = 0; s < n; ++s) pred_out << "," << c[s]; pred_out << "\n"; }
        std::vector<int> P(n, 0), L(n, 0);                              // scored on the reviewed samples only
        for (size_t s = 0; s < n; ++s) {
            if (t.lab[s] < 0) continue;
            const bool p = c[s] == 1, l = t.lab[s] == 1;
            P[s] = p; L[s] = l;
            tp += p && l; fp += p && !l; fn += !p && l; tn += !p && !l;
        }
        const auto pr = runs(P, 1, 3), lr = runs(L, 1, 1);
        n_true += long(lr.size()); n_pred += long(pr.size());
        std::vector<bool> used(pr.size(), false);
        for (const auto& e : lr) for (size_t j = 0; j < pr.size(); ++j)
            if (!used[j] && pr[j].first <= e.second && pr[j].second >= e.first) { used[j] = true; ++hit_true; break; }
        hit_pred += long(std::count(used.begin(), used.end(), true));
    }
    const double nn = double(tp + fp + fn + tn);
    const double po = nn > 0 ? (tp + tn) / nn : 0, pe = nn > 0 ? (double(tp + fp) * (tp + fn) + double(fn + tn) * (fp + tn)) / (nn * nn) : 0;
    const double kappa = pe < 1 ? (po - pe) / (1 - pe) : 0;
    const double rec = n_true ? double(hit_true) / n_true : 0, prec = n_pred ? double(hit_pred) / n_pred : 0;
    const double f1 = rec + prec > 0 ? 2 * rec * prec / (rec + prec) : 0;
    std::ofstream m(a.metrics);
    m << "{\"fold\": " << a.fold << ", \"folds\": " << a.folds << ", \"n_train_trials\": " << tr_idx.size() << ", \"n_test_trials\": " << te_idx.size()
      << ", \"epochs\": " << ep << ", \"event_f1\": " << f1 << ", \"event_recall\": " << rec << ", \"event_precision\": " << prec << ", \"kappa\": " << kappa
      << ", \"n_true\": " << n_true << ", \"n_pred\": " << n_pred << ", \"true_matched\": " << hit_true << ", \"pred_matched\": " << hit_pred
      << ", \"tp\": " << tp << ", \"fp\": " << fp << ", \"fn\": " << fn << ", \"tn\": " << tn << ", \"model\": \"" << model << "\"}\n";
    std::printf("fold %d/%d: %zu train / %zu test trials, %d epochs; event F1 %.3f (recall %.3f, precision %.3f), kappa %.3f\n", a.fold, a.folds, tr_idx.size(),
                te_idx.size(), ep, f1, rec, prec, kappa);
    return 0;
}

int cmd_predict(const Args& a) {
    std::string err;
    if (a.x.empty() || a.y.empty() || a.model.empty() || a.out.empty()) return usage("predict needs --model --x --y --out");
    std::vector<Trial> T;
    if (!load_trials(a, false, T, &err)) { std::cerr << "uneye_train: " << err << "\n"; return 1; }
    train::UNet<float> net(a.classes, 5, 5, 1);
    if (!load_model(a.model, a.classes, net, &err)) { std::cerr << "uneye_train: " << err << "\n"; return 1; }
    std::ofstream o(a.out);
    if (!o) { std::cerr << "uneye_train: cannot write " << a.out << "\n"; return 1; }
    for (const auto& t : T) {
        const auto p = predict_probs(net, t, a.rate);
        const size_t n = t.x.size();
        if (a.probs) { for (size_t s = 0; s < n; ++s) o << (s ? "," : "") << p[n + s]; }          // p(saccade)
        else { const auto c = classify(p, a.classes, n); for (size_t s = 0; s < n; ++s) o << (s ? "," : "") << c[s]; }
        o << "\n";
    }
    std::cout << "wrote " << a.out << " (" << T.size() << " trials)\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    std::string err;
    if (!parse(argc, argv, a, &err)) return usage(err);
    if (a.mode == "train") return cmd_train(a);
    if (a.mode == "cv") return cmd_cv(a);
    if (a.mode == "predict") return cmd_predict(a);
    return usage("unknown mode " + a.mode);
}
