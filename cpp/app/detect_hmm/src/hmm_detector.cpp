#include "eyelive/hmm_detector.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace eyelive {
namespace {
DetectorRegistrar reg("hmm_accel", [](std::string*) { return std::make_unique<HmmDetector>(); });

bool parse_double(const std::string& s, double* out) { char* end = nullptr; const double v = std::strtod(s.c_str(), &end); if (end == s.c_str() || *end != '\0' || !std::isfinite(v)) return false; *out = v; return true; }
double median_inplace(std::vector<double>& v) {                    // numpy.median, reorders v
    const size_t n = v.size(); if (!n) return 0.0; const size_t h = n / 2; std::nth_element(v.begin(), v.begin() + h, v.end()); const double hi = v[h];
    if (n % 2) return hi; return 0.5 * (*std::max_element(v.begin(), v.begin() + h) + hi);
}
void sorted_replace(std::vector<double>& s, double out, double in) {
    s.erase(std::lower_bound(s.begin(), s.end(), out)); s.insert(std::upper_bound(s.begin(), s.end(), in), in);
}
}  // namespace

DetectorInfo HmmDetector::info() const {
    DetectorInfo i; i.id = "hmm_accel"; i.name = "Label-free HMM + acceleration boundaries"; i.n_classes = 2; i.causal = false;
    i.decision_delay_ms = (h_ + 2 + margin_s_ + lag_s_ + hop_) * 1000.0 / fs_;    // centred median (h samples) + 5-point velocity (2) + refinement margin + Viterbi lag + hop; the test measures it
    i.provisional_latency_ms = -1; i.needs_degrees = false; return i;
}

bool HmmDetector::select_params(double rate_hz) {
    error_.clear();
    if (std::fabs(rate_hz - 1000.0) < 50.0) hmm_ = {rate_hz, 6, 0.416573763, 1.32039225, 0.201105669, 0.766239583, 0.00554218027, 0.949469511};      // cpp/models/universal_hmm.txt
    else if (std::fabs(rate_hz - 500.0) < 25.0) hmm_ = {rate_hz, 3, 0.406975567, 1.3010565, 0.20460619, 0.573560297, 0.00969478964, 0.928820044};
    else { error_ = "hmm_accel has parameters for 500 and 1000 Hz only (stream rate " + std::to_string(rate_hz) + " Hz); resample the stream or use another detector"; return false; }
    hmm_.fs = rate_hz; return true;
}

bool HmmDetector::set_option(const std::string& key, const std::string& value) {
    double v;
    if (key == "offset_rule") {
        if (value == "last") { refine_ = true; rf_.offset_first = false; rf_.offset_too = true; }
        else if (value == "first") { refine_ = true; rf_.offset_first = true; rf_.offset_too = true; }
        else if (value == "none") refine_ = false;
        else return false;
        return true;
    }
    if (key == "refine_win_ms") { if (!parse_double(value, &v) || v < 5.0 || v > 100.0) return false; refine_win_ms_ = rf_.win_ms = v; return true; }
    if (key == "margin_ms") { if (!parse_double(value, &v) || v < 0.0 || v > 200.0) return false; margin_ms_ = rf_.margin_ms = v; margin_s_ = int(std::nearbyint(v * fs_ / 1000.0)); return true; }
    return false;
}

void HmmDetector::reset(const SessionInfo& session) {
    session_ = session; fs_ = session.rate_hz > 0 ? session.rate_hz : 1000.0; scale_ = session.deg_per_unit > 0 ? session.deg_per_unit : 1.0; init_ = true;
    const bool ok = select_params(fs_); (void)ok;
    size_t R = 8192; while (R < size_t(8.0 * fs_) + 64) R <<= 1; mask_ = R - 1;
    for (auto* v : {&fx_, &fy_, &vx_, &vy_, &dvx_, &dvy_}) v->assign(R, 0.0); score_.assign(R, 0.f); valid_.assign(R, 1);
    h_ = (int(0.1 * fs_) | 1) / 2; hop_ = std::max(1, int(std::nearbyint(0.010 * fs_))); margin_s_ = int(std::nearbyint(rf_.margin_ms * fs_ / 1000.0)); lag_s_ = int(std::nearbyint(0.030 * fs_)); keep_s_ = int(0.5 * fs_);
    n_ = v_end_ = dv_end_ = score_end_ = anchor_ = 0; last_pub_off_ = -1; last_noise_t_ = 0; hop_count_ = 0; noise_ = 0.0; lx_ = ly_ = 0.0; have_valid_ = false;
    sorted_x_.clear(); sorted_y_.clear(); sorted_x_.reserve(2 * h_ + 2); sorted_y_.reserve(2 * h_ + 2); scratch_a_.reserve(size_t(4.0 * fs_) + 16); scratch_b_.reserve(size_t(4.0 * fs_) + 16);
}

double HmmDetector::sigma_of_last(const std::vector<double>& ring, int64_t t_end) const {      // robust sigma (dsp::robust_sigma_axis) of the last 4 s of the ring, t_end exclusive
    const int64_t t0 = std::max<int64_t>(0, t_end - int64_t(4.0 * fs_)); auto& a = const_cast<std::vector<double>&>(scratch_a_); auto& b = const_cast<std::vector<double>&>(scratch_b_);
    a.clear(); b.clear(); for (int64_t t = t0; t < t_end; ++t) { const double v = ring[at(t)]; a.push_back(v); b.push_back(v * v); }
    const double m1 = median_inplace(a), m2 = median_inplace(b); return std::sqrt(std::max(m2 - m1 * m1, 1e-12));
}

void HmmDetector::ingest(double x, double y, bool valid) {
    if (valid) { lx_ = x; ly_ = y; have_valid_ = true; } else { x = lx_; y = ly_; }                 // invalid samples hold the last valid position (uninformative to the HMM, removed by the invalid-zone rule)
    const size_t k = at(n_); fx_[k] = x; fy_[k] = y; valid_[k] = valid ? 1 : 0; ++n_;
    // velocity: v(t) for t <= n-3 (t = 0, 1 are 0 as in dsp::velocity5)
    while (v_end_ < n_ - 2 || (v_end_ < 2 && v_end_ < n_)) {
        const int64_t t = v_end_; double ax = 0, ay = 0;
        if (t >= 2) { ax = (fx_[at(t + 2)] + fx_[at(t + 1)] - fx_[at(t - 1)] - fx_[at(t - 2)]) * fs_ / 6.0; ay = (fy_[at(t + 2)] + fy_[at(t + 1)] - fy_[at(t - 1)] - fy_[at(t - 2)]) * fs_ / 6.0; }
        vx_[at(t)] = ax; vy_[at(t)] = ay; ++v_end_;
    }
    // centred running median of v over 2h+1 samples ("nearest" at the start) -> detrended velocity
    while (dv_end_ + h_ < v_end_) {
        const int64_t t = dv_end_;
        if (t == 0) {
            sorted_x_.clear(); sorted_y_.clear();
            for (int k2 = -h_; k2 <= h_; ++k2) { sorted_x_.push_back(vx_[at(std::max(k2, 0))]); sorted_y_.push_back(vy_[at(std::max(k2, 0))]); }
            std::sort(sorted_x_.begin(), sorted_x_.end()); std::sort(sorted_y_.begin(), sorted_y_.end());
        } else {
            const int64_t out = std::max<int64_t>(t - 1 - h_, 0), in = t + h_;
            sorted_replace(sorted_x_, vx_[at(out)], vx_[at(in)]); sorted_replace(sorted_y_, vy_[at(out)], vy_[at(in)]);
        }
        dvx_[at(t)] = vx_[at(t)] - sorted_x_[h_]; dvy_[at(t)] = vy_[at(t)] - sorted_y_[h_]; ++dv_end_;
        if (dv_end_ - last_noise_t_ >= int64_t(0.25 * fs_) || (noise_ == 0.0 && dv_end_ >= int64_t(0.25 * fs_))) { noise_ = std::max(0.5 * (sigma_of_last(dvx_, dv_end_) + sigma_of_last(dvy_, dv_end_)), 1e-9); last_noise_t_ = dv_end_; }
    }
    if (noise_ > 0.0) while (score_end_ < dv_end_) { const size_t q = at(score_end_); score_[q] = float(std::log10(1.0 + std::hypot(dvx_[q], dvy_[q]) / noise_)); ++score_end_; }
}

Event HmmDetector::make_event(const ITraceView& trace, int64_t on, int64_t off) const {
    Event e; e.onset = on; e.offset = off; e.t_onset = trace.t(on); e.t_offset = trace.t(off); e.cls = 1; e.origin = EventOrigin::Detector; e.detector_id = 1; e.model_version = 1; e.provisional = false;
    const double dx = fx_[at(off)] - fx_[at(on)], dy = fy_[at(off)] - fy_[at(on)]; e.amplitude_deg = float(std::hypot(dx, dy) * scale_); e.direction_rad = float(std::atan2(dy, dx));
    double pk = 0; for (int64_t t = on; t <= off && t < v_end_; ++t) pk = std::max(pk, std::hypot(vx_[at(t)], vy_[at(t)])); e.peak_speed_deg_s = float(pk * scale_);
    e.confidence = float(noise_ > 0 ? std::min(1.0, pk / (10.0 * noise_)) : 0.5); return e;
}

void HmmDetector::decode(const ITraceView& trace, std::vector<Event>& out) {
    const int64_t a = anchor_, e = score_end_; if (e - a < 2 * hop_) return;
    const size_t len = size_t(e - a); std::vector<float> f(len, 0.f); dsp::Mask valid(len, 1);
    for (size_t i = 0; i < len; ++i) { const size_t q = at(a + int64_t(i)); f[i] = score_[q]; valid[i] = valid_[q]; }
    dsp::Mask lab = dsp::viterbi(f, valid, hmm_); dsp::drop_invalid_zone_and_short(lab, valid, fs_, hmm_.d); const dsp::Events ev = dsp::runs_of(lab);
    const int64_t need = std::max<int64_t>(margin_s_, 0) + lag_s_ + 2; bool published = false; int64_t new_anchor = anchor_;
    bool candidate = false; for (const auto& r : ev) if (a + r.second + need < e && a + r.first > last_pub_off_) { candidate = true; break; }
    if (candidate) {
        // refinement on a context of up to 4 s so that its noise floor (3 x robust sigma of the SG velocity) is that of the whole trace, as in batch; done once per candidate event, not per sample
        const int64_t c0 = std::max<int64_t>(0, std::min<int64_t>(a, e - int64_t(7.0 * fs_))); const size_t clen = size_t(e - c0);
        std::vector<double> px(clen, 0.0), py(clen, 0.0); dsp::Mask cv(clen, 1);
        for (size_t i = 0; i < clen; ++i) { const size_t q = at(c0 + int64_t(i)); px[i] = fx_[q]; py[i] = fy_[q]; cv[i] = valid_[q]; }
        dsp::Events shifted = ev; for (auto& r : shifted) { r.first += int(a - c0); r.second += int(a - c0); }
        dsp::Events fin = shifted;
        if (refine_) { dsp::Refine r = rf_; r.win_ms = refine_win_ms_; r.margin_ms = margin_ms_; fin = dsp::runs_of(dsp::refine_events(shifted, px, py, cv, fs_, r)); }
        for (const auto& r : fin) {
            const int64_t on = c0 + r.first, off = c0 + r.second; if (on <= last_pub_off_) continue;
            if (off + need >= e) break;                                                         // not enough data after it yet: later events cannot be published before this one
            out.push_back(make_event(trace, on, off)); last_pub_off_ = off; new_anchor = std::max(new_anchor, off + 1); published = true;
        }
    }
    if (published) { anchor_ = new_anchor; return; }
    const int64_t cand = e - keep_s_;                                                           // quiet stretch: move the anchor forward so the window stays short (it must lie in fixation, away from any run)
    if (cand > a + int64_t(keep_s_ / 4)) {
        bool quiet = true; const int64_t pad = margin_s_ + lag_s_ + 5;
        for (const auto& r : ev) if (a + r.second + pad >= cand && a + r.first - pad <= cand) { quiet = false; break; }
        if (quiet) anchor_ = cand;
    }
}

void HmmDetector::advance(const ITraceView& trace, int64_t to, std::vector<Event>& final_events, std::vector<Event>&) {
    if (!init_) reset(trace.info()); if (!error_.empty()) return;
    to = std::min<int64_t>(to, trace.size());
    while (n_ < to) { ingest(trace.x(n_), trace.y(n_), trace.valid(n_) && std::isfinite(trace.x(n_)) && std::isfinite(trace.y(n_))); if (++hop_count_ >= hop_) { hop_count_ = 0; decode(trace, final_events); } }
}

void HmmDetector::detect_all(const ITraceView& trace, std::vector<Event>& events) {
    SessionInfo s = trace.info(); if (!(s.rate_hz > 0)) s.rate_hz = fs_; const double fs = s.rate_hz; if (!select_params(fs)) return;
    const int64_t T = trace.size(); std::vector<double> x(static_cast<size_t>(T), 0.0), y(static_cast<size_t>(T), 0.0); dsp::Mask valid(static_cast<size_t>(T), 1);
    for (int64_t i = 0; i < T; ++i) { x[size_t(i)] = trace.x(i); y[size_t(i)] = trace.y(i); valid[size_t(i)] = trace.valid(i) && std::isfinite(x[size_t(i)]) && std::isfinite(y[size_t(i)]); }
    dsp::Kin k; dsp::score_trace(x, y, valid, fs, k); const dsp::Events ev = dsp::runs_of(dsp::hmm_detect(k, hmm_)); dsp::Events fin = ev;
    if (refine_) { dsp::Refine r = rf_; r.win_ms = refine_win_ms_; r.margin_ms = margin_ms_; fin = dsp::runs_of(dsp::refine_events(ev, k.fx, k.fy, valid, fs, r)); }
    events.clear(); const double sc = s.deg_per_unit > 0 ? s.deg_per_unit : scale_;
    for (const auto& r : fin) {
        Event e; e.onset = r.first; e.offset = r.second; e.t_onset = trace.t(r.first); e.t_offset = trace.t(r.second); e.cls = 1; e.origin = EventOrigin::Detector; e.detector_id = 1; e.model_version = 1;
        const double dx = k.fx[size_t(r.second)] - k.fx[size_t(r.first)], dy = k.fy[size_t(r.second)] - k.fy[size_t(r.first)]; e.amplitude_deg = float(std::hypot(dx, dy) * sc); e.direction_rad = float(std::atan2(dy, dx));
        double pk = 0; for (int t = r.first; t <= r.second; ++t) pk = std::max(pk, std::hypot(k.vx[size_t(t)], k.vy[size_t(t)])); e.peak_speed_deg_s = float(pk * sc); e.confidence = float(std::min(1.0, pk / (10.0 * k.noise)));
        events.push_back(e);
    }
}

}  // namespace eyelive
