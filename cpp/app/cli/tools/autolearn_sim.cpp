// Simulation of the assisted-annotation workflow on a labelled dataset of the repository: the ground-truth labels play the expert, window by window. The real engine runs (recording, label-free labelling, retraining after a quiet delay,
// adoption of the trained model); after every adopted model its quality is measured on HELD-OUT trials against the ground truth, together with the time and the CPU its training took.
//   eyelive_autolearn_sim --data data/dataset1 --name dataset1_1000hz [--train-seconds 60] [--test-seconds 60] [--mode oracle|accept] [--block 5] [--delay 2] [--pause 2.5] [--epochs 150] [--threads 0] [--rate 1000]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include "eyelive/engine.hpp"
#include "eyelive/net_detectors.hpp"
using namespace eyelive;
namespace fs = std::filesystem;

static double cpu_s() {
#ifndef _WIN32
    rusage r; getrusage(RUSAGE_SELF, &r); return double(r.ru_utime.tv_sec) + r.ru_utime.tv_usec * 1e-6 + double(r.ru_stime.tv_sec) + r.ru_stime.tv_usec * 1e-6;
#else
    return 0;
#endif
}
static double wall_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<std::vector<double>> load_matrix(const std::string& path) {
    std::ifstream f(path); if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); } std::vector<std::vector<double>> m; std::string line;
    while (std::getline(f, line)) { std::vector<double> row; std::stringstream ss(line); std::string tok; while (std::getline(ss, tok, ',')) row.push_back(tok.empty() || tok == "nan" || tok == "NaN" ? std::nan("") : std::strtod(tok.c_str(), nullptr)); m.push_back(std::move(row)); }
    return m;
}
struct Rec { std::vector<float> x, y; std::vector<uint8_t> valid, gt; };
static Rec build(const std::string& dir, const std::string& name, const char* set, double seconds, double rate) {      // trials one after the other, positions continued so that there is no artificial jump
    const auto X = load_matrix(dir + "/" + name + "_X_set" + set + ".csv"), Y = load_matrix(dir + "/" + name + "_Y_set" + set + ".csv"), L = load_matrix(dir + "/" + name + "_Labels_set" + set + ".csv");
    Rec r; const size_t need = size_t(seconds * rate); double ox = 0, oy = 0;
    for (size_t t = 0; t < X.size() && r.x.size() < need; ++t) {
        size_t first = 0; while (first < X[t].size() && !(std::isfinite(X[t][first]) && std::isfinite(Y[t][first]))) ++first; if (first >= X[t].size()) continue; const double dx = ox - X[t][first], dy = oy - Y[t][first];
        for (size_t i = 0; i < X[t].size(); ++i) { const bool ok = std::isfinite(X[t][i]) && std::isfinite(Y[t][i]); r.x.push_back(ok ? float(X[t][i] + dx) : 0.f); r.y.push_back(ok ? float(Y[t][i] + dy) : 0.f); r.valid.push_back(ok ? 1 : 0); r.gt.push_back(L[t][i] > 0.5 ? 1 : 0); if (ok) { ox = X[t][i] + dx; oy = Y[t][i] + dy; } }
    }
    r.x.resize(need); r.y.resize(need); r.valid.resize(need); r.gt.resize(need); return r;
}
static std::vector<Event> runs(const std::vector<uint8_t>& gt, double rate) {
    std::vector<Event> v; const int64_t n = int64_t(gt.size()); for (int64_t i = 0; i < n;) { if (!gt[size_t(i)]) { ++i; continue; } int64_t e = i; while (e + 1 < n && gt[size_t(e + 1)]) ++e; Event ev; ev.onset = i; ev.offset = e; ev.t_onset = double(i) / rate; ev.t_offset = double(e) / rate; ev.cls = 1; v.push_back(ev); i = e + 1; } return v;
}
static AgreementStats score(IDetector& d, const Rec& test, const std::vector<Event>& gt_events, double rate) {
    SessionInfo si; si.rate_hz = rate; VectorTrace tr(si); for (size_t i = 0; i < test.x.size(); ++i) { Sample s; s.t = double(i) / rate; s.x = test.x[i]; s.y = test.y[i]; s.flags = test.valid[i] ? kFlagValid : kFlagBlink; tr.push(s); }
    d.reset(si); d.set_scale(1.0); std::vector<Event> ev; d.detect_all(tr, ev); std::vector<Event> sac; for (auto& e : ev) if (e.cls == 1) sac.push_back(e); return agreement(sac, gt_events, int64_t(test.x.size()), 1);
}

int main(int argc, char** argv) {
    std::string dir = "data/dataset1", name = "dataset1_1000hz", mode = "oracle"; double train_s = 60, test_s = 60, rate = 1000, delay = 2.0, pause = 2.5; int block = 5, epochs = 150, threads = 0; double min_f1 = 0.6;
    for (int i = 1; i < argc; ++i) { const std::string a = argv[i]; auto v = [&]() { return i + 1 < argc ? argv[++i] : (std::fprintf(stderr, "%s needs a value\n", a.c_str()), std::exit(2), argv[0]); };
        if (a == "--data") dir = v(); else if (a == "--name") name = v(); else if (a == "--mode") mode = v(); else if (a == "--train-seconds") train_s = std::atof(v()); else if (a == "--test-seconds") test_s = std::atof(v()); else if (a == "--rate") rate = std::atof(v());
        else if (a == "--block") block = std::atoi(v()); else if (a == "--delay") delay = std::atof(v()); else if (a == "--pause") pause = std::atof(v()); else if (a == "--epochs") epochs = std::atoi(v()); else if (a == "--threads") threads = std::atoi(v()); else if (a == "--min-f1") min_f1 = std::atof(v());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; } }
    if (models_dir().empty()) { std::error_code ec; fs::path d = fs::absolute(argv[0], ec).parent_path(); for (int up = 0; up < 7 && !d.empty(); ++up, d = d.parent_path()) for (const char* rel : {"models", "cpp/models"}) if (fs::exists(d / rel / "combined.bin", ec)) { set_models_dir((d / rel).string()); up = 99; break; } }
    const Rec train = build(dir, name, "A", train_s, rate), test = build(dir, name, "B", test_s, rate); const auto gt_train = runs(train.gt, rate), gt_test = runs(test.gt, rate);
    const fs::path tmp = fs::temp_directory_path() / ("eyelive_alsim_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())); fs::create_directories(tmp); const std::string csv = (tmp / "train.csv").string();
    { std::ofstream f(csv); f << "t,x,y,valid\n"; for (size_t i = 0; i < train.x.size(); ++i) f << double(i) / rate << ',' << train.x[i] << ',' << train.y[i] << ',' << int(train.valid[i]) << '\n'; }
    std::printf("training stream: %.0f s (%zu ground-truth saccades), held-out test: %.0f s (%zu saccades); expert = %s, %d s blocks, quiet delay %.1f s, pause %.1f s, %d CPU threads per training (0 = all)\n", train_s, gt_train.size(), test_s, gt_test.size(), mode.c_str(), block, delay, pause, threads);
    auto engine = make_engine(); SessionConfig sc; sc.source.kind = "file"; sc.source.path = csv; sc.source.speed = 0; sc.rate_hz = rate; sc.deg_per_unit = 1.0; sc.session_path = (tmp / "s.eyelive").string(); sc.duration_s = train_s + 10; std::string err;
    if (!engine->start(sc, &err)) { std::fprintf(stderr, "start: %s\n", err.c_str()); return 1; } while (engine->stats().samples < train.x.size()) std::this_thread::sleep_for(std::chrono::milliseconds(20)); std::this_thread::sleep_for(std::chrono::milliseconds(500)); engine->stop();
    { auto hmm = DetectorRegistry::instance().create("hmm_accel", nullptr); const AgreementStats a = score(*hmm, test, gt_test, rate); std::printf("teacher (label-free HMM) on the held-out test: F1 %.3f, kappa %.3f (tp %d, fp %d, fn %d)\n\n", a.event_f1, a.kappa, a.tp, a.fp, a.fn); }
    AutoLearnConfig ac; ac.enabled = true; ac.retrain_delay_s = delay; ac.chunk_s = 2.0; ac.max_epochs = epochs; ac.expected_epochs = 50; ac.adopt_min_f1 = min_f1; ac.n_threads = threads; ac.max_seconds_of_data = 50; engine->autolearn_set(ac);
    ExpertLabels& ex = engine->expert(); AutoLearnStatus st; auto poll = [&](bool edited) { st = engine->autolearn_poll(edited); return st; }; const int windows = int(train_s); int last_gen = 0; bool was_training = false; double train_cpu0 = 0, train_wall0 = 0; int trainings = 0;
    std::printf("%-8s %-8s %-6s %-14s %-9s %-9s %-10s %-9s %-s\n", "reviewed", "model", "epochs", "train time (s)", "CPU (s)", "CPU/wall", "test F1", "test kappa", "next-window F1 before review (labeler vs ground truth)");
    int tp_acc = 0, fp_acc = 0, fn_acc = 0; int tp_pre = 0, fp_pre = 0, fn_pre = 0;      // agreement of the labeler's labels with the ground truth on windows BEFORE the expert touches them (cumulative over the current model)
    const double t_start = wall_s();
    for (int k = 0; k < windows; ++k) {
        while (poll(false).labelled_s < double(k + 1) + 0.3 && wall_s() - t_start < 3600) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // labels of the window as they are now (labeler's output), compared with the ground truth BEFORE the expert corrects them
        const int64_t a = int64_t(k * rate), b = int64_t((k + 1) * rate) - 1; std::vector<Event> cur, truth; for (const auto& e : ex.events()) if (e.cls == 1 && e.onset >= a && e.onset <= b) cur.push_back(e); for (const auto& g : gt_train) if (g.onset >= a && g.onset <= b) truth.push_back(g);
        { const AgreementStats w = agreement(cur, truth, int64_t(rate) * 2 + b, 1); tp_pre += w.tp; fp_pre += w.fp; fn_pre += w.fn; }
        // the expert reviews the window: oracle = replaces the class-1 events of the window by the ground-truth ones; accept = confirms the labels as they are
        if (mode == "oracle") {
            for (size_t i = ex.events().size(); i-- > 0;) { const Event& e = ex.events()[i]; if (e.cls == 1 && e.offset >= a && e.onset <= b) ex.remove(i); } for (const auto& g : truth) ex.add(g.onset, g.offset, 1);
        }
        ex.mark_reviewed(std::max<int64_t>(0, a - 500), b + 500, true, false); poll(true);
        const bool end_block = ((k + 1) % block == 0) || k + 1 == windows; if (!end_block) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
        // pause after a block: the quiet delay elapses, a training starts; the loop keeps polling (a human would look at the plot meanwhile)
        const double until = wall_s() + pause; while (wall_s() < until || st.training) {
            poll(false); if (st.training && !was_training) { train_cpu0 = cpu_s(); train_wall0 = wall_s(); was_training = true; }
            if (!st.training && was_training) { was_training = false; ++trainings; const double wl = wall_s() - train_wall0, cp = cpu_s() - train_cpu0; const TrainProgress tp = engine->training_progress();
                if (st.generation > last_gen) { last_gen = st.generation; auto d = DetectorRegistry::instance().create(st.labeler, nullptr); AgreementStats a2 = d ? score(*d, test, gt_test, rate) : AgreementStats();
                    std::printf("%-8d #%-7d %-6d %-14.1f %-9.1f %-9.2f %-10.3f %-9.3f %.3f (%d events)\n", k + 1, st.generation, tp.epoch, wl, cp, wl > 0 ? cp / wl : 0, a2.event_f1, a2.kappa, (2.0 * tp_pre) / std::max(1, 2 * tp_pre + fp_pre + fn_pre), tp_pre + fn_pre); tp_pre = fp_pre = fn_pre = 0; }
                else std::printf("%-8d rejected %-6d %-14.1f %-9.1f %-9.2f (validation F1 %.2f < %.2f: not adopted, keeps learning)\n", k + 1, tp.epoch, wl, cp, wl > 0 ? cp / wl : 0, tp.val_event_f1, min_f1); std::fflush(stdout); }
            std::this_thread::sleep_for(std::chrono::milliseconds(30)); if (wall_s() - t_start > 7200) break; }
        (void)tp_acc; (void)fp_acc; (void)fn_acc;
    }
    // final: the last adopted model and what the expert ended up with
    poll(false); std::printf("\nfinished: %d trainings, %d adopted models, last labeler %s; %.0f s of windows reviewed\n", trainings, st.generation, st.labeler.c_str(), ex.reviewed_seconds(rate));
    std::error_code ec; fs::remove_all(tmp, ec); return 0;
}
