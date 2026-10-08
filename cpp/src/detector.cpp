#include "uneye/detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace uneye {

StreamingDetector::StreamingDetector(std::unique_ptr<StepEngine> engine, const Config& cfg)
    : step_(std::move(engine)), cfg_(cfg), W_(1), C_(step_->classes()) {
    cfg_.lookahead = 0;
    sprob_.assign(C_, 0.f);
    init_common();
}

StreamingDetector::StreamingDetector(std::unique_ptr<Engine> engine, const Config& cfg)
    : eng_(std::move(engine)), cfg_(cfg), W_(eng_->window()), C_(eng_->classes()) {
    init_common();
    ring_.assign(size_t(2) * W_, 0.f);
    win_.assign(size_t(2) * W_, 0.f);
    prob_.assign(size_t(C_) * W_, 0.f);
}

void StreamingDetector::init_common() {
    const double ms = 1000.0 / cfg_.fs;
    // same conversions as uneye/classifier.py::predict (2-class models only)
    min_dur_ = C_ == 2 ? int(cfg_.min_sacc_dur_ms / ms) : 1;
    min_dist_ = C_ == 2 ? int(cfg_.min_sacc_dist_ms * (cfg_.fs / 1000.0)) : 0;
    cfg_.hop = std::max(1, cfg_.hop);
    if (!step_ && cfg_.lookahead >= W_ - cfg_.hop) cfg_.lookahead = W_ - cfg_.hop - 1;
}

void StreamingDetector::step_causal() {
    const auto t0 = std::chrono::steady_clock::now();
    step_->step(float(last_dx_), float(last_dy_), sprob_.data());
    last_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    int cls = 0; float best = sprob_[0];
    for (int c = 1; c < C_; ++c) if (sprob_[c] > best) { best = sprob_[c]; cls = c; }
    provisional_ = 1.f - sprob_[0];
    if (C_ == 2) cls = provisional_ > cfg_.threshold ? 1 : 0;
    const int64_t idx = n_ - 1;
    if (on_fast) on_fast({idx, cls, provisional_});
    fast_run_ = cls ? fast_run_ + 1 : 0;
    if (fast_run_ == cfg_.fast_confirm && on_fast_onset) on_fast_onset(idx - cfg_.fast_confirm + 1);
    feed_label(idx, cls, provisional_);
    committed_ = n_;
}

void StreamingDetector::push(double x, double y) {
    x *= cfg_.input_scale; y *= cfg_.input_scale;
    // differentiated signal, as in training (first sample -> 0, NaN -> 0, Inf -> inf_correction)
    double dx = 0, dy = 0;
    if (have_prev_) { dx = x - px_; dy = y - py_; }
    if (std::isnan(dx) || std::isnan(dy)) { dx = dy = 0; }
    if (std::isinf(dx)) dx = cfg_.inf_correction;
    if (std::isinf(dy)) dy = cfg_.inf_correction;
    if (std::isfinite(x) && std::isfinite(y)) { px_ = x; py_ = y; have_prev_ = true; }
    if (step_) { last_dx_ = dx; last_dy_ = dy; ++n_; step_causal(); return; }
    const size_t slot = size_t(n_ % W_) * 2;
    ring_[slot] = float(dx); ring_[slot + 1] = float(dy);
    ++n_;
    if (n_ - last_infer_n_ >= cfg_.hop) run_network();
}

void StreamingDetector::run_network() {
    last_infer_n_ = n_;
    // linearise the last W samples (older than sample 0 -> zeros = no movement)
    const int64_t start = n_ - W_;
    for (int j = 0; j < W_; ++j) {
        const int64_t idx = start + j;
        if (idx < 0) { win_[2 * j] = win_[2 * j + 1] = 0.f; continue; }
        const size_t slot = size_t(idx % W_) * 2;
        win_[2 * j] = ring_[slot]; win_[2 * j + 1] = ring_[slot + 1];
    }
    const auto t0 = std::chrono::steady_clock::now();
    eng_->infer(win_.data(), prob_.data());
    last_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    provisional_ = 1.f - prob_[W_ - 1];  // P(not fixation) at newest sample
    if (on_fast || on_fast_onset) {
        int cls = 0; float best = prob_[W_ - 1];
        for (int c = 1; c < C_; ++c)
            if (prob_[size_t(c) * W_ + W_ - 1] > best) { best = prob_[size_t(c) * W_ + W_ - 1]; cls = c; }
        if (C_ == 2) cls = provisional_ > cfg_.threshold ? 1 : 0;
        if (on_fast) on_fast({n_ - 1, cls, provisional_});
        fast_run_ = cls ? fast_run_ + 1 : 0;
        if (fast_run_ == cfg_.fast_confirm && on_fast_onset) on_fast_onset(n_ - cfg_.fast_confirm);
    }
    commit(n_ - cfg_.lookahead, prob_, start);
}

void StreamingDetector::commit(int64_t upto, const std::vector<float>& prob, int64_t win_start) {
    for (int64_t i = std::max<int64_t>(committed_, 0); i < upto; ++i) {
        const int j = int(i - win_start);
        if (j < 0) continue;
        int cls = 0;
        float best = prob[j];
        for (int c = 1; c < C_; ++c)
            if (prob[size_t(c) * W_ + j] > best) { best = prob[size_t(c) * W_ + j]; cls = c; }
        const float ps = 1.f - prob[j];
        if (C_ == 2) cls = ps > cfg_.threshold ? 1 : 0;
        feed_label(i, cls, ps);
    }
    committed_ = std::max(committed_, upto);
}

// ---- event state machine on committed labels -------------------------------
void StreamingDetector::feed_label(int64_t idx, int cls, float p) {
    if (on_label) on_label({idx, cls, p});
    if (cls != run_cls_) {
        if (run_cls_ != 0) close_run(idx - 1);
        run_cls_ = cls;
        run_start_ = idx;
        run_announced_ = false;
    } else if (run_cls_ != 0 && !run_announced_ && idx - run_start_ + 1 >= min_dur_) {
        run_announced_ = true;
        if (on_onset) on_onset(run_start_);
    }
    flush_pending(idx, false);
}

void StreamingDetector::close_run(int64_t end_idx) {
    if (end_idx - run_start_ + 1 < min_dur_) return;  // too short: noise
    Event e{run_cls_, run_start_, end_idx, n_};
    if (have_pending_ && pending_.cls == e.cls && e.onset - pending_.offset - 1 < min_dist_) {
        pending_.offset = e.offset;  // merge close events
        pending_.confirmed = n_;
        return;
    }
    flush_pending(n_, true);
    pending_ = e; have_pending_ = true;
}

void StreamingDetector::flush_pending(int64_t now, bool force) {
    if (!have_pending_) return;
    if (force || now - pending_.offset - 1 >= min_dist_) {
        have_pending_ = false;
        if (on_event) on_event(pending_);
    }
}

void StreamingDetector::reset() {
    n_ = committed_ = last_infer_n_ = 0;
    px_ = py_ = 0; have_prev_ = false;
    std::fill(ring_.begin(), ring_.end(), 0.f);
    if (step_) step_->reset();
    provisional_ = 0; last_ns_ = 0; fast_run_ = 0;
    run_cls_ = 0; run_start_ = 0; run_announced_ = false; have_pending_ = false;
}

void StreamingDetector::finish() {
    if (step_) {
        if (run_cls_ != 0) close_run(n_ - 1);
        run_cls_ = 0;
        flush_pending(n_ + min_dist_ + 1, true);
        return;
    }
    // commit remaining samples with the newest window (edge effects accepted)
    if (n_ > last_infer_n_ || committed_ < n_) {
        if (n_ > last_infer_n_) {
            const int64_t keep = cfg_.lookahead;
            cfg_.lookahead = 0;
            run_network();
            cfg_.lookahead = int(keep);
        } else {
            commit(n_, prob_, n_ - W_);
        }
    }
    if (run_cls_ != 0) close_run(n_ - 1);
    run_cls_ = 0;
    flush_pending(n_ + min_dist_ + 1, true);
}

}  // namespace uneye
