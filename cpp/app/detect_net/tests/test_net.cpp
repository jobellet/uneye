// T06: neural detectors. Label parity with the engines on the golden windows, class maps (5 classes -> default 3), scale applied to the network input, user models, unknown model path, streaming against batch.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <algorithm>
#include <cstdlib>
#include "eyelive/net_detectors.hpp"
#include "eyelive/trace.hpp"
#include "uneye/bitcn.hpp"
#include "uneye/causal_engine.hpp"
#include "uneye/engine.hpp"
using namespace eyelive;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct Run { int cls; int64_t on, off; };
static std::vector<Run> runs_of(const std::vector<int>& cls, int min_len) {
    std::vector<Run> r; const int64_t T = int64_t(cls.size());
    for (int64_t t = 0; t < T;) { if (cls[size_t(t)] == 0) { ++t; continue; } int64_t e = t; while (e + 1 < T && cls[size_t(e + 1)] == cls[size_t(t)]) ++e; if (e - t + 1 >= min_len) r.push_back({cls[size_t(t)], t, e}); t = e + 1; }
    return r;
}
static VectorTrace trace_from_positions(const std::vector<float>& x, const std::vector<float>& y, double fs) {
    SessionInfo s; s.rate_hz = fs; VectorTrace tr(s); for (size_t i = 0; i < x.size(); ++i) { Sample sm; sm.t = double(i) / fs; sm.x = x[i]; sm.y = y[i]; sm.flags = (std::isfinite(x[i]) && std::isfinite(y[i])) ? kFlagValid : kFlagBlink; tr.push(sm); } return tr;
}
static bool same_events(const std::vector<Event>& ev, const std::vector<Run>& want) { if (ev.size() != want.size()) return false; for (size_t i = 0; i < want.size(); ++i) if (ev[i].onset != want[i].on || ev[i].offset != want[i].off || ev[i].cls != want[i].cls) return false; return true; }

// golden window file: W C, then dX/dY (W x 2), then C x W probabilities
struct Golden { int W = 0, C = 0; std::vector<float> dxy, prob; };
static bool load_golden(const std::string& path, Golden& g) { std::ifstream f(path); if (!f) return false; f >> g.W >> g.C; g.dxy.resize(size_t(2) * g.W); g.prob.resize(size_t(g.C) * g.W); for (auto& v : g.dxy) f >> v; for (auto& v : g.prob) f >> v; return bool(f); }
// positions whose velocity (first sample 0) equals the golden dxy with the first value set to 0, as the detectors compute it
static void positions_from_velocity(const Golden& g, std::vector<float>& x, std::vector<float>& y) { x.assign(size_t(g.W), 0.f); y.assign(size_t(g.W), 0.f); for (int t = 1; t < g.W; ++t) { x[size_t(t)] = x[size_t(t - 1)] + g.dxy[size_t(2 * t)]; y[size_t(t)] = y[size_t(t - 1)] + g.dxy[size_t(2 * t + 1)]; } }
static std::vector<int> engine_classes(const std::string& model, const Golden& g, double threshold) {
    auto eng = uneye::make_native_engine(model, g.W); std::vector<float> in = g.dxy; in[0] = in[1] = 0.f; std::vector<float> prob(size_t(g.C) * g.W); eng->infer(in.data(), prob.data()); std::vector<int> cls(size_t(g.W), 0);
    for (int t = 0; t < g.W; ++t) { if (g.C == 2) cls[size_t(t)] = (1.f - prob[size_t(t)]) > threshold ? 1 : 0; else { int b = 0; for (int c = 1; c < g.C; ++c) if (prob[size_t(c) * g.W + t] > prob[size_t(b) * g.W + t]) b = c; cls[size_t(t)] = b; } }
    return cls;
}

// a double engine that records its input and answers "saccade" where |dx| > 0.1
struct Recorder { std::vector<float> last; int calls = 0; };
class DoubleEngine : public uneye::Engine {
 public:
    explicit DoubleEngine(std::shared_ptr<Recorder> r) : r_(std::move(r)) {}
    int window() const override { return 200; }
    int classes() const override { return 2; }
    void infer(const float* dxy, float* prob) override { r_->last.assign(dxy, dxy + 400); ++r_->calls; for (int t = 0; t < 200; ++t) { const bool s = std::fabs(dxy[2 * t]) > 0.1f; prob[t] = s ? 0.f : 1.f; prob[200 + t] = s ? 1.f : 0.f; } }
 private:
    std::shared_ptr<Recorder> r_;
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : "."; set_models_dir(root + "/models");
    // 1. label parity on the golden windows
    for (const char* name : {"combined", "andersson"}) {
        Golden g; CHECK(load_golden(root + "/tests/golden_" + name + ".txt", g), "golden %s", name); if (g.W == 0) continue;
        const std::string id = std::string(name) == "combined" ? "unet_1+2+3" : "unet_andersson5"; std::string err; auto det = DetectorRegistry::instance().create(id, &err); CHECK(det != nullptr, "%s: %s", id.c_str(), err.c_str()); if (!det) continue;
        std::vector<float> x, y; positions_from_velocity(g, x, y); auto tr = trace_from_positions(x, y, 1000.0);
        const std::vector<int> cls = engine_classes(root + "/models/" + (std::string(name) == "combined" ? "combined" : "andersson") + ".bin", g, 0.5);
        const int minlen = g.C == 2 ? 6 : 3;
        CHECK(det->set_option("class_map", "identity"), "class_map identity"); det->reset(tr.info()); std::vector<Event> ev; det->detect_all(tr, ev); const auto want = runs_of(cls, minlen);
        CHECK(same_events(ev, want), "%s: %zu events, engine reference %zu (identity map)", name, ev.size(), want.size());
        std::printf("%s: %zu events from detect_all, %zu from the engine's argmax / threshold (identity map)\n", name, ev.size(), want.size());
        if (g.C == 5) {
            std::vector<int> mapped(cls.size()); const int m[5] = {0, 1, 1, 0, 2}; for (size_t i = 0; i < cls.size(); ++i) mapped[i] = m[cls[i]]; auto want3 = runs_of(mapped, minlen);
            CHECK(det->set_option("class_map", "default3"), "class_map default3"); std::vector<Event> e3; det->detect_all(tr, e3); CHECK(same_events(e3, want3), "default3: %zu events vs reference %zu", e3.size(), want3.size());
            for (auto& e : e3) CHECK(e.cls <= 2, "default3 gave class %d", e.cls); CHECK(det->info().n_classes == 3, "info().n_classes %d", det->info().n_classes);
        } else for (auto& e : ev) CHECK(e.cls == 1, "2-class model gave class %d", e.cls);
    }
    // causal TCN: batch events against the engine stepped by hand (label delay 10)
    {
        std::string err; auto det = DetectorRegistry::instance().create("tcn_causal", &err); CHECK(det != nullptr, "tcn_causal: %s", err.c_str());
        if (det) { std::mt19937 rng(3); std::normal_distribution<float> nd(0, 0.01f); const int T = 1500; std::vector<float> x(T), y(T); float px = 0, py = 0; for (int t = 0; t < T; ++t) { float vx = nd(rng); if (t >= 500 && t < 520) vx += 0.25f; if (t >= 1100 && t < 1112) vx -= 0.3f; px += vx; py += nd(rng); x[size_t(t)] = px; y[size_t(t)] = py; }
            auto tr = trace_from_positions(x, y, 1000.0); det->reset(tr.info()); std::vector<Event> ev; det->detect_all(tr, ev);
            auto step = uneye::make_causal_engine(root + "/models/tcn_l10.bin"); std::vector<float> p(2); std::vector<int> cls(T, 0); for (int t = 0; t < T; ++t) { const float dx = t ? x[size_t(t)] - x[size_t(t - 1)] : 0.f, dy = t ? y[size_t(t)] - y[size_t(t - 1)] : 0.f; step->step(dx, dy, p.data()); if (t - 10 >= 0) cls[size_t(t - 10)] = (1.f - p[0]) > 0.5f ? 1 : 0; }
            for (int t = T - 10; t < T; ++t) cls[size_t(t)] = cls[size_t(t - 1)]; const auto want = runs_of(cls, 6);
            CHECK(same_events(ev, want), "tcn_causal: %zu events vs reference %zu", ev.size(), want.size()); std::printf("tcn_causal: %zu events (reference %zu); the two injected movements %s\n", ev.size(), want.size(), ev.size() >= 2 ? "found" : "NOT found");
            CHECK(ev.size() >= 2, "tcn_causal found %zu events for 2 injected saccades", ev.size()); CHECK(det->info().causal, "tcn_causal must be causal"); }
    }
    // BiTCN against uneye::detect on the golden trace (T x 2 positions with a blink and a dropout)
    {
        std::ifstream f(root + "/tests/golden_bitcn_2ch.txt"); int T, C; f >> T >> C; std::vector<float> pos(size_t(2) * T); std::string s; for (auto& v : pos) { f >> s; v = float(std::strtod(s.c_str(), nullptr)); }
        std::vector<float> x(T), y(T); for (int t = 0; t < T; ++t) { x[size_t(t)] = pos[size_t(2 * t)]; y[size_t(t)] = pos[size_t(2 * t + 1)]; }
        auto tr = trace_from_positions(x, y, 1000.0); std::string err; auto det = DetectorRegistry::instance().create("bitcn_v2", &err); CHECK(det != nullptr, "bitcn_v2: %s", err.c_str());
        if (det) { det->reset(tr.info()); std::vector<Event> ev; det->detect_all(tr, ev); uneye::BiTCN net(root + "/models/bitcn_sup_bitcn_v2.bin"); const auto m = uneye::detect(net, pos.data(), T, 1000.0); std::vector<int> cls(T); for (int t = 0; t < T; ++t) cls[size_t(t)] = m[size_t(t)] ? 1 : 0;
            const auto want = runs_of(cls, 6); CHECK(same_events(ev, want), "bitcn_v2: %zu events vs reference %zu", ev.size(), want.size()); std::printf("bitcn_v2: %zu events (reference %zu)\n", ev.size(), want.size()); }
    }
    // set_scale multiplies the positions seen by the network
    {
        auto rec = std::make_shared<Recorder>(); std::string err; auto det = make_detector_with_engine([rec] { return std::unique_ptr<uneye::Engine>(new DoubleEngine(rec)); }, "double", &err); CHECK(det != nullptr, "double: %s", err.c_str());
        if (det) { std::vector<float> x(300), y(300, 0.f); for (int t = 0; t < 300; ++t) x[size_t(t)] = float(t);                      // a constant velocity of 1 unit / sample
            auto tr = trace_from_positions(x, y, 1000.0); det->reset(tr.info()); det->set_scale(0.5); std::vector<Event> ev; det->detect_all(tr, ev);
            double mx = 0; for (int t = 1; t < 200; ++t) mx = std::max<double>(mx, rec->last[size_t(2 * t)]); double mn = 1e9; for (int t = 1; t < 200; ++t) mn = std::min<double>(mn, rec->last[size_t(2 * t)]);
            std::printf("set_scale(0.5): the network saw dX = %.3f .. %.3f for a velocity of 1 unit / sample\n", mn, mx); CHECK(std::fabs(mx - 0.5) < 1e-6 && std::fabs(mn - 0.5) < 1e-6, "network input %.3f / %.3f, expected 0.5", mn, mx);
        }
    }
    // user model written by the trainer: the 2-class synthetic model of the repository, with and without sidecar; unknown path; bad file
    {
        std::string err; auto d1 = make_user_model_detector(root + "/models/synthetic.bin", &err); CHECK(d1 != nullptr, "user model: %s", err.c_str()); if (d1) CHECK(d1->info().needs_degrees, "needs_degrees");
        err.clear(); auto d2 = make_user_model_detector("no_such_model.bin", &err); CHECK(!d2 && !err.empty(), "unknown model path must return nullptr with a message"); std::printf("unknown model -> '%s'\n", err.c_str());
        err.clear(); auto d3 = make_user_model_detector(root + "/tests/golden_combined.txt", &err); CHECK(!d3 && !err.empty(), "a text file is not a model"); 
        err.clear(); CHECK(DetectorRegistry::instance().create("unet_1+2+3", &err) != nullptr, "registry create");
    }
    // options
    {
        std::string err; auto det = DetectorRegistry::instance().create("unet_1+2+3", &err); if (det) { CHECK(det->set_option("class_map", "identity") && det->set_option("class_map", "default3") && !det->set_option("class_map", "x"), "class_map"); CHECK(!det->set_option("nope", "1") && det->set_option("threshold", "0.6") && !det->set_option("threshold", "2"), "options"); } }
    // streaming against batch for the window U'n'Eye on a synthetic trace with 6 saccades
    {
        std::mt19937 rng(8); std::normal_distribution<double> nd(0, 0.01); const int T = 6000; std::vector<float> x(T), y(T); double px = 0, py = 0; for (int t = 0; t < T; ++t) { double vx = 0.002 * nd(rng), vy = 0.002 * nd(rng); for (int k = 0; k < 6; ++k) { const int s0 = 500 + 900 * k; if (t >= s0 && t < s0 + 24) { const double u = double(t - s0 + 0.5) / 24.0; vx += 0.55 * std::sin(3.14159265 * u) * std::pow(-1.0, k); } } px += vx; py += vy; x[size_t(t)] = float(px + nd(rng)); y[size_t(t)] = float(py + nd(rng)); }
        auto tr = trace_from_positions(x, y, 1000.0); std::string err; auto det = DetectorRegistry::instance().create("unet_1+2+3", &err); CHECK(det != nullptr, "create");
        if (det) { det->reset(tr.info()); std::vector<Event> batch; det->detect_all(tr, batch); det->reset(tr.info()); std::vector<Event> st, prov; for (int64_t n = 1; n <= T; ++n) det->advance(tr, n, st, prov);
            int matched = 0; for (auto& b : batch) for (auto& s : st) if (std::abs(s.onset - b.onset) <= 5 && std::abs(s.offset - b.offset) <= 5) { ++matched; break; }
            for (auto& b : batch) std::printf("  batch [%lld,%lld]\n", (long long)b.onset, (long long)b.offset); for (auto& q : st) std::printf("  stream [%lld,%lld]\n", (long long)q.onset, (long long)q.offset);
            std::printf("unet_1+2+3 on 6 injected saccades: batch %zu events, streaming %zu, %d batch events matched within 5 samples\n", batch.size(), st.size(), matched); CHECK(!batch.empty(), "no event in batch"); CHECK(matched >= int(batch.size()) - 1, "streaming matched %d of %zu", matched, batch.size()); }
    }
    // 5-class model on the same kind of trace: default map gives only classes 0 / 1 / 2, the identity map the model's own ids (PSO / pursuit allowed)
    {
        std::mt19937 rng(9); std::normal_distribution<double> nd(0, 0.01); const int T = 4000; std::vector<float> x(T), y(T); double px = 0, py = 0;
        for (int t = 0; t < T; ++t) { double vx = 0.002 * nd(rng), vy = 0.002 * nd(rng); for (int k = 0; k < 4; ++k) { const int s0 = 500 + 900 * k; if (t >= s0 && t < s0 + 24) vx += 0.55 * std::sin(3.14159265 * double(t - s0 + 0.5) / 24.0) * (k % 2 ? -1 : 1); } px += vx; py += vy; x[size_t(t)] = float(px + nd(rng)); y[size_t(t)] = float(py + nd(rng)); }
        auto tr = trace_from_positions(x, y, 1000.0); std::string err; auto det = DetectorRegistry::instance().create("unet_andersson5", &err); CHECK(det != nullptr, "andersson: %s", err.c_str());
        if (det) { det->set_option("class_map", "identity"); det->reset(tr.info()); std::vector<Event> id; det->detect_all(tr, id); det->set_option("class_map", "default3"); det->reset(tr.info()); std::vector<Event> d3; det->detect_all(tr, d3);
            int in_range = 0; for (auto& e : d3) in_range += e.cls >= 1 && e.cls <= 2; std::printf("unet_andersson5 on 4 injected saccades: %zu events with the identity map (classes", id.size()); for (auto& e : id) std::printf(" %d", e.cls); std::printf("), %zu with default3 (classes", d3.size()); for (auto& e : d3) std::printf(" %d", e.cls); std::printf(")\n");
            CHECK(in_range == int(d3.size()), "default3 produced a class outside 1..2"); CHECK(d3.size() <= id.size(), "default3 cannot have more events than identity"); CHECK(d3.size() >= 3, "default3 found %zu of 4 injected saccades", d3.size()); }
    }
    std::printf(fails ? "FAILED\n" : "all net detector tests passed\n"); return fails ? 1 : 0;
}
