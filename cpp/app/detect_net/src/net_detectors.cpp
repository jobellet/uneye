#include "eyelive/net_detectors.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include "uneye/bitcn.hpp"
#include "uneye/causal_engine.hpp"
#include "uneye/detector.hpp"
#include "uneye/engine.hpp"

namespace eyelive {
namespace {

std::mutex g_mu; std::string g_dir;
bool file_exists(const std::string& p) { std::ifstream f(p, std::ios::binary); return bool(f); }
bool is_absolute(const std::string& p) { return !p.empty() && (p[0] == '/' || p[0] == '\\' || (p.size() > 1 && p[1] == ':')); }
std::string find_model(const std::string& name) {
    if (is_absolute(name)) return file_exists(name) ? name : std::string();
    std::vector<std::string> dirs; { std::lock_guard<std::mutex> l(g_mu); if (!g_dir.empty()) dirs.push_back(g_dir); }
    if (const char* e = std::getenv("EYELIVE_MODELS")) dirs.push_back(e);
    for (const char* d : {"models", "../models", "cpp/models", "../cpp/models", "../../cpp/models"}) dirs.push_back(d);
    if (file_exists(name)) return name;                                                      // relative to the working directory
    for (const auto& d : dirs) { const std::string p = d + "/" + name; if (file_exists(p)) return p; }
    return std::string();
}
std::string read_all(const std::string& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }
double json_num(const std::string& j, const std::string& key, double def) {          // flat {"key": number} files written by the exporters / the trainer
    const size_t k = j.find("\"" + key + "\""); if (k == std::string::npos) return def; const size_t c = j.find(':', k); if (c == std::string::npos) return def;
    char* end = nullptr; const double v = std::strtod(j.c_str() + c + 1, &end); return end == j.c_str() + c + 1 ? def : v;
}
bool parse_double(const std::string& s, double* out) { char* end = nullptr; const double v = std::strtod(s.c_str(), &end); if (end == s.c_str() || *end != '\0' || !std::isfinite(v)) return false; *out = v; return true; }

enum class Kind { Window, Causal, BiTcn };

class NetDetector final : public IDetector {
 public:
    NetDetector(std::string id, std::string name, Kind kind, std::string bin, int classes, int window, int label_delay, double model_scale)
        : id_(std::move(id)), name_(std::move(name)), kind_(kind), bin_(std::move(bin)), classes_(classes), window_(window), label_delay_(label_delay), scale_(model_scale) {}

    DetectorInfo info() const override {
        DetectorInfo i; i.id = id_; i.name = name_; i.n_classes = classes_ == 5 && map_default3_ ? 3 : classes_; i.needs_degrees = true; i.provisional_latency_ms = -1; const double ms = 1000.0 / fs_;
        if (kind_ == Kind::Window) { i.causal = false; i.decision_delay_ms = (lookahead() + hop() + 6) * ms; }
        else if (kind_ == Kind::Causal) { i.causal = true; i.decision_delay_ms = (label_delay_ + 6) * ms; }
        else { i.causal = false; i.decision_delay_ms = (rf_half() + bi_hop() + 6) * ms; }
        return i;
    }
    void set_scale(double s) override { if (s > 0) { scale_ = s; user_scale_ = true; } }
    bool set_option(const std::string& key, const std::string& value) override {
        double v;
        if (key == "class_map") { if (value == "default3") map_default3_ = true; else if (value == "identity") map_default3_ = false; else return false; return true; }
        if (key == "threshold") { if (!parse_double(value, &v) || v <= 0 || v >= 1) return false; threshold_ = v; return true; }
        if (key == "min_duration_ms") { if (!parse_double(value, &v) || v < 0 || v > 200) return false; min_ms_ = v; return true; }
        return false;
    }
    void reset(const SessionInfo& s) override {
        fs_ = s.rate_hz > 0 ? s.rate_hz : 1000.0; if (!user_scale_ && s.deg_per_unit > 0) scale_ = s.deg_per_unit;       // the engine normally calls set_scale() with the scale in use; the session value is the fallback
        load(); n_ = 0; last_pub_off_ = -1; last_decode_ = 0; pending_valid_ = false; stream_.reset(); events_.clear(); if (!error_.empty()) return;
        uneye::Config c; c.fs = fs_; c.hop = hop(); c.lookahead = lookahead(); c.input_scale = scale_; c.threshold = threshold_; c.min_sacc_dur_ms = min_ms(); c.min_sacc_dist_ms = 4; c.label_delay = label_delay_;
        if (kind_ == Kind::Window) { c.window_ms = 1000.0 * window_ / fs_; stream_ = std::make_unique<uneye::StreamingDetector>(open_engine(), c); }
        else if (kind_ == Kind::Causal) stream_ = std::make_unique<uneye::StreamingDetector>(open_step(), c);
        if (stream_) stream_->on_event = [this](const uneye::Event& e) { events_.push_back(e); };
    }
    void advance(const ITraceView& tr, int64_t to, std::vector<Event>& final_events, std::vector<Event>&) override {
        if (!loaded_) reset(tr.info()); if (!error_.empty()) return; to = std::min<int64_t>(to, tr.size());
        if (kind_ == Kind::BiTcn) { advance_bitcn(tr, to, final_events); return; }
        while (n_ < to) {
            const double x = tr.x(n_), y = tr.y(n_); const bool ok = tr.valid(n_) && std::isfinite(x) && std::isfinite(y); stream_->push(ok ? x : std::nan(""), ok ? y : std::nan("")); ++n_;
            for (const auto& e : events_) accept(tr, e.cls, e.onset, e.offset, final_events); events_.clear();
            flush_pending(tr, n_, final_events);
        }
    }
    void detect_all(const ITraceView& tr, std::vector<Event>& out) override {
        out.clear(); const SessionInfo s = tr.info(); if (!loaded_ || (s.rate_hz > 0 && std::fabs(s.rate_hz - fs_) > 1e-6)) reset(s); if (!error_.empty()) return;
        const int64_t T = tr.size(); const std::vector<float> pos = positions(tr, 0, T);
        for (const auto& r : runs(classes_of(pos, T))) build(tr, r.cls, r.on, r.off, out);
    }
    // per-sample classes of a position array (T x 2 interleaved, NaN = invalid) in the model's own numbering. Window models: windows of W samples every W/2, each sample read from the window whose centre is nearest
    // (T < W: one zero-padded window); causal TCN: one step per sample, the output at t describes sample t - label_delay; BiTCN: logit > 0. Input to the networks: velocities (position differences times the scale).
    std::vector<uint8_t> classes_of(const std::vector<float>& pos, int64_t T) {
        std::vector<uint8_t> cls(size_t(T), 0); if (T <= 0) return cls;
        if (kind_ == Kind::BiTcn) { const auto m = uneye::detect(*bitcn_, pos.data(), int(T), fs_); for (int64_t i = 0; i < T; ++i) cls[size_t(i)] = m[size_t(i)] ? 1 : 0; return cls; }
        const std::vector<float> dxy = velocities(pos, T);
        if (kind_ == Kind::Causal) {
            auto step = open_step(); std::vector<float> p(size_t(step->classes()));
            for (int64_t t = 0; t < T; ++t) { step->step(dxy[size_t(2 * t)], dxy[size_t(2 * t + 1)], p.data()); const int64_t at = t - label_delay_; if (at >= 0) cls[size_t(at)] = decide(p.data()); }
            for (int64_t t = std::max<int64_t>(T - label_delay_, 0); t < T; ++t) cls[size_t(t)] = t > 0 ? cls[size_t(t - 1)] : uint8_t(0);       // the last label_delay samples have no output yet: keep the last decision
            return cls;
        }
        auto eng = open_engine(); const int W = eng->window(), S = std::max(1, W / 2); std::vector<int64_t> starts;
        if (T <= W) starts.push_back(0); else { for (int64_t s0 = 0; s0 + W < T; s0 += S) starts.push_back(s0); starts.push_back(T - W); }
        std::vector<float> win(size_t(2) * W), prob(size_t(eng->classes()) * W), p(size_t(eng->classes())); size_t lo = 0;
        std::vector<size_t> owner(size_t(T), 0);
        for (int64_t i = 0; i < T; ++i) {                                                         // nearest centre among the windows that contain sample i
            while (lo + 1 < starts.size() && starts[lo] + W <= i) ++lo; size_t best = lo; double bd = 1e18;
            for (size_t k = lo; k < starts.size() && starts[k] <= i; ++k) { const double d = std::fabs(double(i) - (double(starts[k]) + 0.5 * W)); if (d < bd) { bd = d; best = k; } }
            owner[size_t(i)] = best;
        }
        for (size_t k = 0; k < starts.size(); ++k) {
            const int64_t s0 = starts[k]; for (int j = 0; j < W; ++j) { const int64_t i = s0 + j; win[size_t(2 * j)] = i < T ? dxy[size_t(2 * i)] : 0.f; win[size_t(2 * j + 1)] = i < T ? dxy[size_t(2 * i + 1)] : 0.f; }
            eng->infer(win.data(), prob.data());
            for (int j = 0; j < W; ++j) { const int64_t i = s0 + j; if (i >= T || owner[size_t(i)] != k) continue; for (int c = 0; c < eng->classes(); ++c) p[size_t(c)] = prob[size_t(c) * W + j]; cls[size_t(i)] = decide(p.data()); }
        }
        return cls;
    }
    bool load_now(std::string* err) { load(); if (err) *err = error_; return error_.empty(); }
    void inject_engine(EngineFactory f) { inject_ = std::move(f); }

 private:
    struct Run { int cls; int64_t on, off; };
    int lookahead() const { return std::max(1, std::min(int(std::nearbyint(0.090 * fs_)), (window_ > hop() + 2 ? window_ - hop() - 2 : 1))); }       // decisions are taken ~90 ms after the sample, close to the window centre (as in batch)
    int hop() const { return std::max(1, int(std::nearbyint(0.010 * fs_))); }
    int rf_half() const { return bitcn_ ? bitcn_->receptive_field() / 2 : 130; }
    int bi_hop() const { return std::max(1, int(std::nearbyint(0.050 * fs_))); }
    double min_ms() const { return min_ms_ >= 0 ? min_ms_ : (classes_ == 2 ? 6.0 : 3.0); }
    void load() {
        if (loaded_) return; error_.clear();
        if (inject_) { auto e = inject_(); classes_ = e->classes(); window_ = e->window(); loaded_ = true; return; }
        path_ = find_model(bin_); if (path_.empty()) { error_ = "model file not found: " + bin_; return; }
        try {
            if (kind_ == Kind::Window) { auto e = uneye::make_native_engine(path_, window_); classes_ = e->classes(); }
            else if (kind_ == Kind::Causal) { auto s = uneye::make_causal_engine(path_); classes_ = s->classes(); }
            else bitcn_ = std::make_unique<uneye::BiTCN>(path_);
        } catch (const std::exception& e) { error_ = e.what(); return; }
        loaded_ = true;
    }
    std::unique_ptr<uneye::Engine> open_engine() const { return inject_ ? inject_() : uneye::make_native_engine(path_, window_); }
    std::unique_ptr<uneye::StepEngine> open_step() const { return uneye::make_causal_engine(path_); }
    static std::vector<float> positions(const ITraceView& tr, int64_t c0, int64_t T) {
        std::vector<float> pos(size_t(2) * size_t(T), 0.f);
        for (int64_t i = 0; i < T; ++i) { const double x = tr.x(c0 + i), y = tr.y(c0 + i); const bool ok = tr.valid(c0 + i) && std::isfinite(x) && std::isfinite(y); pos[size_t(2 * i)] = ok ? float(x) : std::nanf(""); pos[size_t(2 * i + 1)] = ok ? float(y) : std::nanf(""); }
        return pos;
    }
    std::vector<float> velocities(const std::vector<float>& pos, int64_t T) const {           // as StreamingDetector::push and uneye/classifier.py: first sample 0, NaN -> 0
        std::vector<float> d(size_t(2) * size_t(T), 0.f); double px = 0, py = 0; bool have = false;
        for (int64_t i = 0; i < T; ++i) { const double x = double(pos[size_t(2 * i)]) * scale_, y = double(pos[size_t(2 * i + 1)]) * scale_; double dx = 0, dy = 0; if (have) { dx = x - px; dy = y - py; } if (std::isnan(dx) || std::isnan(dy)) dx = dy = 0; if (std::isinf(dx)) dx = 1.5; if (std::isinf(dy)) dy = 1.5;
            if (std::isfinite(x) && std::isfinite(y)) { px = x; py = y; have = true; } d[size_t(2 * i)] = float(dx); d[size_t(2 * i + 1)] = float(dy); }
        return d;
    }
    uint8_t decide(const float* p) const {                                                    // argmax; 2-class models: threshold on 1 - P(fixation)
        if (classes_ == 2) return (1.f - p[0]) > threshold_ ? 1 : 0; int best = 0; for (int c = 1; c < classes_; ++c) if (p[c] > p[best]) best = c; return uint8_t(best);
    }
    int map_class(int c) const { if (!map_default3_ || classes_ != 5) return c; static const int m[5] = {0, 1, 1, 0, 2}; return m[c]; }     // 5-class model: 0 fixation, 1 saccade, 2 PSO, 3 pursuit, 4 blink
    std::vector<Run> runs(const std::vector<uint8_t>& cls) const {                           // runs of the MAPPED class (> 0); runs shorter than the minimum duration are dropped
        std::vector<Run> r; const int64_t T = int64_t(cls.size()); const int64_t minl = std::max<int64_t>(1, int64_t(min_ms() * fs_ / 1000.0));
        for (int64_t t = 0; t < T;) { const int c = map_class(cls[size_t(t)]); if (c == 0) { ++t; continue; } int64_t e = t; while (e + 1 < T && map_class(cls[size_t(e + 1)]) == c) ++e; r.push_back({c, t, e}); t = e + 1; }
        std::vector<Run> m; for (const auto& q : r) { if (!m.empty() && m.back().cls == q.cls && q.on - m.back().off <= 4) m.back().off = q.off; else m.push_back(q); }       // flicker inside one movement: gaps of <= 3 samples are merged
        std::vector<Run> keep; for (const auto& q : m) if (q.off - q.on + 1 >= minl) keep.push_back(q);
        return keep;
    }
    void build(const ITraceView& tr, int cls, int64_t on, int64_t off, std::vector<Event>& out) const {
        Event e; e.onset = on; e.offset = off; e.t_onset = tr.t(on); e.t_offset = tr.t(off); e.cls = uint8_t(cls); e.origin = EventOrigin::Detector; e.detector_id = 2; e.model_version = 1;
        const double dx = (tr.x(off) - tr.x(on)) * scale_, dy = (tr.y(off) - tr.y(on)) * scale_; e.amplitude_deg = float(std::hypot(dx, dy)); e.direction_rad = float(std::atan2(dy, dx));
        double pk = 0; for (int64_t t = std::max<int64_t>(on, 1); t <= off && t + 1 < tr.size(); ++t) pk = std::max(pk, std::hypot(tr.x(t + 1) - tr.x(t - 1), tr.y(t + 1) - tr.y(t - 1)) * fs_ / 2.0 * scale_);
        e.peak_speed_deg_s = float(pk); out.push_back(e);
    }
    void accept(const ITraceView& tr, int cls, int64_t on, int64_t off, std::vector<Event>& out) {        // streaming events (the model's own class ids): map, merge neighbours of the same mapped class, publish a few samples later
        const int m = map_class(cls); if (m == 0 || on <= last_pub_off_) return;
        if (pending_valid_ && pending_cls_ == m && on - pending_off_ <= 3) { pending_off_ = std::max(pending_off_, off); return; }
        if (pending_valid_) { build(tr, pending_cls_, pending_on_, pending_off_, out); last_pub_off_ = pending_off_; }
        pending_valid_ = true; pending_cls_ = m; pending_on_ = on; pending_off_ = off;
    }
    void flush_pending(const ITraceView& tr, int64_t n, std::vector<Event>& out) { if (pending_valid_ && n - pending_off_ > 6) { build(tr, pending_cls_, pending_on_, pending_off_, out); last_pub_off_ = pending_off_; pending_valid_ = false; } }
    void advance_bitcn(const ITraceView& tr, int64_t to, std::vector<Event>& out) {
        if (to - last_decode_ < bi_hop()) return; last_decode_ = to; n_ = to;
        const int64_t c0 = std::max<int64_t>(0, to - int64_t(4.0 * fs_)), T = to - c0, margin = rf_half() + 5; const std::vector<float> pos = positions(tr, c0, T);
        for (const auto& r : runs(classes_of(pos, T))) { const int64_t on = c0 + r.on, off = c0 + r.off; if (on <= last_pub_off_) continue; if (off + margin >= to) break; build(tr, r.cls, on, off, out); last_pub_off_ = off; }
    }

    EngineFactory inject_; std::string id_, name_; Kind kind_; std::string bin_, path_; int classes_, window_, label_delay_; double scale_; bool user_scale_ = false;
    bool map_default3_ = true; double threshold_ = 0.5, min_ms_ = -1.0, fs_ = 1000.0; bool loaded_ = false; std::string error_;
    std::unique_ptr<uneye::BiTCN> bitcn_; std::unique_ptr<uneye::StreamingDetector> stream_; std::vector<uneye::Event> events_;
    int64_t n_ = 0, last_pub_off_ = -1, last_decode_ = 0, pending_on_ = 0, pending_off_ = 0; int pending_cls_ = 0; bool pending_valid_ = false;
};

std::unique_ptr<IDetector> make_registered(const char* id, const char* name, Kind kind, const char* bin, int classes, int window, int label_delay, std::string* error) {
    auto d = std::make_unique<NetDetector>(id, name, kind, bin, classes, window, label_delay, 1.0); std::string e; if (!d->load_now(&e)) { if (error) *error = e; return nullptr; } return d;
}
DetectorRegistrar r1("unet_1+2+3", [](std::string* e) { return make_registered("unet_1+2+3", "U'n'Eye (datasets 1+2+3)", Kind::Window, "combined.bin", 2, 200, 0, e); });
DetectorRegistrar r2("unet_andersson5", [](std::string* e) { return make_registered("unet_andersson5", "U'n'Eye (Andersson, 5 classes)", Kind::Window, "andersson.bin", 5, 200, 0, e); });
DetectorRegistrar r3("tcn_causal", [](std::string* e) {
    const std::string j = find_model("tcn_l10.json"); const int delay = j.empty() ? 10 : int(json_num(read_all(j), "label_delay_samples", 10));
    return make_registered("tcn_causal", "Causal TCN (label delay 10 ms)", Kind::Causal, "tcn_l10.bin", 2, 1, delay, e); });
DetectorRegistrar r4("bitcn_v2", [](std::string* e) { return make_registered("bitcn_v2", "BiTCN v2 (two channels)", Kind::BiTcn, "bitcn_sup_bitcn_v2.bin", 2, 1, 0, e); });
}  // namespace

std::unique_ptr<IDetector> make_detector_with_engine(EngineFactory factory, const std::string& id, std::string* error) {
    auto d = std::make_unique<NetDetector>(id, id, Kind::Window, std::string(), 2, 200, 0, 1.0); d->inject_engine(std::move(factory)); std::string e; if (!d->load_now(&e)) { if (error) *error = e; return nullptr; } return d;
}

void set_models_dir(const std::string& dir) { std::lock_guard<std::mutex> l(g_mu); g_dir = dir; }
std::string models_dir() { std::lock_guard<std::mutex> l(g_mu); return g_dir; }

std::unique_ptr<IDetector> make_user_model_detector(const std::string& path, std::string* error) {
    const std::string bin = find_model(path); if (bin.empty()) { if (error) *error = "model file not found: " + path; return nullptr; }
    std::ifstream f(bin, std::ios::binary); char magic[4] = {0, 0, 0, 0}; f.read(magic, 4); if (std::string(magic, 4) != "UNEY") { if (error) *error = "not a U'n'Eye model file (bad magic): " + path; return nullptr; }
    const std::string side = (bin.size() > 4 && bin.compare(bin.size() - 4, 4, ".bin") == 0 ? bin.substr(0, bin.size() - 4) : bin) + ".json"; const std::string js = file_exists(side) ? read_all(side) : std::string();
    const int classes = int(json_num(js, "classes", 0)); const double scale = json_num(js, "deg_per_unit", 1.0); const int window = int(json_num(js, "window", 200.0));
    auto d = std::make_unique<NetDetector>("user:" + path, "User model " + path, Kind::Window, bin, classes > 0 ? classes : 2, window > 0 ? window : 200, 0, scale);
    std::string e; if (!d->load_now(&e)) { if (error) *error = e; return nullptr; } return d;
}

}  // namespace eyelive
