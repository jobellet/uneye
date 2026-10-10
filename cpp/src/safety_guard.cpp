// Deterministic safety guard. See include/uneye/safety_guard.hpp for the states, the rules and where every limit comes from.
// Everything in step(): fixed-size arrays only, bounded loops, no clock / randomness / I/O.
#include "uneye/safety_guard.hpp"

#include <algorithm>
#include <cmath>

namespace uneye {
namespace safety {
namespace {

constexpr int kNoiseEvery = 8;                 // recompute the noise every 8 valid samples (as in the engine)

float median_of(float* a, int n) {             // partially sorts the scratch array; n >= 1
    float* mid = a + n / 2;
    std::nth_element(a, mid, a + n);
    return *mid;
}

GuardState level_of(Reason r) {                // what each reason asks for (the state machine takes the maximum over one sample)
    switch (r) {
        case Reason::Saturated: case Reason::Flat: case Reason::RateMismatch: case Reason::LongInvalid: case Reason::EngineFailed:
            return GuardState::Safe;
        case Reason::NetworkUntrusted: case Reason::LowConfidence: case Reason::OutOfDistribution: case Reason::NoMovement:
        case Reason::RepeatedViolations:
            return GuardState::Degraded;
        default:
            return GuardState::Normal;         // a single bad sample only makes that sample Invalid; a dropped event only drops the event
    }
}

}  // namespace

const char* reason_name(Reason r) noexcept {
    static const char* const names[] = {"None", "NonFinite", "OutOfRange", "Saturated", "Flat", "Spike", "TimeBackwards", "Gap",
                                        "RateMismatch", "LongInvalid", "EngineFailed", "NetworkUntrusted", "LowConfidence",
                                        "OutOfDistribution", "NoMovement", "DeadlineMiss", "EventTooFast", "EventTooLarge", "EventTooLong", "EventTooShort", "EventTooSlow",
                                        "EventRefractory", "EventAborted", "RepeatedViolations", "Hysteresis"};
    static_assert(sizeof(names) / sizeof(names[0]) == static_cast<size_t>(Reason::Count), "one name per reason");
    const int i = static_cast<int>(r);
    return i >= 0 && i < static_cast<int>(Reason::Count) ? names[i] : "?";
}

SafetyGuard::SafetyGuard(const GuardConfig& cfg) noexcept : cfg_(cfg) {
    // the sliding counters live in a ring of kRing samples: clamp the windows so that they always fit
    cfg_.deadline_window = std::min(std::max(cfg_.deadline_window, 1), kRing - 1);
    cfg_.violation_window = std::min(std::max(cfg_.violation_window, 1), kRing - 1);
    cfg_.recover_samples = std::max(cfg_.recover_samples, 1);
    cfg_.min_run = std::max(cfg_.min_run, 1);
    reset();
}

void SafetyGuard::reset() noexcept {
    state_ = GuardState::Normal; calm_run_ = 0; n_ = 0; st_ = Stats{};
    have_valid_ = have_t_ = false; last_x_ = last_y_ = last_t_ = dt_ema_ = 0; dt_count_ = 0;
    same_xy_run_ = sat_run_x_ = sat_run_y_ = invalid_run_ = 0;
    dxh_.fill(0); dyh_.fill(0); dpos_ = 0;
    nvx_.fill(0); nvy_.fill(0); scratch_.fill(0); ncount_ = npos_ = nsince_ = 0; sigx_ = sigy_ = 20.0; above_run_ = 0;
    unsure_ema_ = nomove_ema_ = 0; prev_raw_x_ = prev_raw_y_ = 0; have_raw_ = false;
    miss_bit_.fill(0); viol_bit_.fill(0); miss_count_ = viol_count_ = 0;
    in_event_ = suppress_ = false; ev_reason_ = Reason::None; ev_onset_ = ev_last_ = last_offset_ = -1; ev_gap_ = 0;
    ev_x0_ = ev_y0_ = ev_x1_ = ev_y1_ = ev_peak_ = 0;
}

void SafetyGuard::slide(std::array<uint8_t, kRing>& bits, int& count, int window, bool now) noexcept {
    // count of 'now' over the last `window` samples: drop the bit that leaves the window, add the new one
    if (n_ >= window) count -= bits[static_cast<size_t>((n_ - window) & (kRing - 1))];
    bits[static_cast<size_t>(n_ & (kRing - 1))] = now ? 1 : 0;
    count += now ? 1 : 0;
}

void SafetyGuard::update_noise() noexcept {    // Engbert & Kliegl: sd = sqrt(median(v^2) - median(v)^2) per axis, floor 1 deg/s (as the engine)
    for (int axis = 0; axis < 2; ++axis) {
        const std::array<float, kNoiseWin>& v = axis == 0 ? nvx_ : nvy_;
        for (int k = 0; k < ncount_; ++k) scratch_[static_cast<size_t>(k)] = v[static_cast<size_t>(k)] * v[static_cast<size_t>(k)];
        const double m2 = median_of(scratch_.data(), ncount_);
        for (int k = 0; k < ncount_; ++k) scratch_[static_cast<size_t>(k)] = v[static_cast<size_t>(k)];
        const double m1 = median_of(scratch_.data(), ncount_);
        (axis == 0 ? sigx_ : sigy_) = std::sqrt(std::max(m2 - m1 * m1, 1.0));
    }
}

Label SafetyGuard::fallback_label(bool valid, double speed, double vx, double vy) noexcept {
    // causal velocity threshold: outside the ellipse of lambda robust sd AND above an absolute floor, for min_run samples in a row.
    // Unlike the engine, earlier samples of the run are not relabeled (the guard never revises), so a detection starts min_run - 1 samples late.
    bool above = false;
    if (valid) {
        const double a = vx / (cfg_.lambda * sigx_), b = vy / (cfg_.lambda * sigy_);
        above = a * a + b * b > 1.0 && speed > cfg_.floor_deg_s;
    }
    above_run_ = above ? above_run_ + 1 : 0;
    return above_run_ >= cfg_.min_run ? Label::Saccade : Label::Fixation;
}

void SafetyGuard::end_event(int64_t offset, Reason forced, GuardOutput& out) noexcept {
    GuardEvent e;
    e.onset = ev_onset_; e.offset = offset;
    e.duration_ms = static_cast<float>(static_cast<double>(offset - ev_onset_ + 1) * 1000.0 / cfg_.fs_hz);
    e.amplitude_deg = static_cast<float>(std::hypot(ev_x1_ - ev_x0_, ev_y1_ - ev_y0_));
    e.peak_speed_deg_s = static_cast<float>(ev_peak_);
    // physiology, checked in a fixed order; the first violated limit is the reason
    Reason r = forced != Reason::None ? forced : ev_reason_;
    if (r == Reason::None && e.peak_speed_deg_s > cfg_.max_peak_deg_s) r = Reason::EventTooFast;
    if (r == Reason::None && e.amplitude_deg > cfg_.max_amplitude_deg) r = Reason::EventTooLarge;
    if (r == Reason::None && e.duration_ms > cfg_.max_duration_ms) r = Reason::EventTooLong;
    if (r == Reason::None && e.duration_ms < cfg_.min_duration_ms) r = Reason::EventTooShort;
    if (r == Reason::None && e.peak_speed_deg_s < cfg_.min_peak_deg_s) r = Reason::EventTooSlow;
    e.accepted = r == Reason::None;
    e.reason = r;
    out.event_ended = true; out.event = e;
    if (e.accepted) ++st_.events_emitted; else ++st_.events_dropped;
    in_event_ = false; ev_reason_ = Reason::None;
    last_offset_ = offset;
}

void SafetyGuard::step(const gaze::Sample& raw, const gaze::Output& eo, double compute_us, GuardOutput& out) noexcept {
    out = GuardOutput{};
    out.index = n_;
    GuardState target = GuardState::Normal;
    Reason main = Reason::None;
    GuardState main_level = GuardState::Normal;
    // record a reason and the state it asks for (default: level_of(r)); the most severe reason of the sample becomes `main`
    auto add = [&](Reason r, int level = -1) {
        out.reasons |= 1u << static_cast<unsigned>(r);
        ++st_.by_reason[static_cast<size_t>(r)];
        const GuardState l = level < 0 ? level_of(r) : static_cast<GuardState>(level);
        if (main == Reason::None || l > main_level) { main = r; main_level = l; }
        if (l > target) target = l;
    };

    // ---- (a) input plausibility
    bool valid = true, gap = false;
    const double x = raw.x_deg, y = raw.y_deg;
    if (!std::isfinite(x) || !std::isfinite(y)) { valid = false; add(Reason::NonFinite); }
    else if (std::fabs(x) > cfg_.max_abs_deg || std::fabs(y) > cfg_.max_abs_deg) { valid = false; add(Reason::OutOfRange); }
    const double nominal_us = 1e6 / cfg_.fs_hz;
    if (raw.t_us >= 0.0 && std::isfinite(raw.t_us)) {
        if (have_t_) {
            const double dt = raw.t_us - last_t_;
            if (dt <= 0.0) { valid = false; add(Reason::TimeBackwards); }
            else {
                if (dt > 1.5 * nominal_us) { gap = true; add(Reason::Gap); }
                // measured mean interval (all steps, also gaps, clamped so that one long pause does not dominate): a wrong rate shows here
                dt_ema_ += (std::min(dt, 10.0 * nominal_us) - dt_ema_) / (dt_count_ < 256 ? static_cast<double>(++dt_count_) : 256.0);
                last_t_ = raw.t_us;
            }
        } else { last_t_ = raw.t_us; have_t_ = true; }
    }
    if (dt_count_ >= 64 && std::fabs(dt_ema_ / nominal_us - 1.0) > cfg_.rate_tolerance) add(Reason::RateMismatch);
    if (std::isfinite(x) && std::isfinite(y)) {
        // frozen signal: both axes bit-identical to the previous raw sample. Saturation: one axis held beyond the training range.
        // (compared with the previous RAW sample, not the last valid one: the jump onto a rail is itself rejected as a spike)
        same_xy_run_ = (have_raw_ && x == prev_raw_x_ && y == prev_raw_y_) ? same_xy_run_ + 1 : 0;
        sat_run_x_ = (have_raw_ && x == prev_raw_x_ && std::fabs(x) >= cfg_.saturation_deg) ? sat_run_x_ + 1 : 0;
        sat_run_y_ = (have_raw_ && y == prev_raw_y_ && std::fabs(y) >= cfg_.saturation_deg) ? sat_run_y_ + 1 : 0;
        prev_raw_x_ = x; prev_raw_y_ = y; have_raw_ = true;
        if (same_xy_run_ >= cfg_.flat_samples) { valid = false; add(Reason::Flat); }
        else if (std::max(sat_run_x_, sat_run_y_) >= cfg_.saturation_samples) { valid = false; add(Reason::Saturated); }
    }
    if (valid && have_valid_) {
        if (!gap) {
            const double jump = std::hypot(x - last_x_, y - last_y_) / static_cast<double>(invalid_run_ + 1);
            if (jump * cfg_.fs_hz > cfg_.max_speed_deg_s) { valid = false; add(Reason::Spike); }
        }
    }

    // ---- velocity (3-sample mean of displacements; none across a gap or invalid data, as in the engine) and robust noise
    const double px = have_valid_ ? last_x_ : (valid ? x : 0.0), py = have_valid_ ? last_y_ : (valid ? y : 0.0);     // last valid position before this sample (start point of an event); the sample itself before any valid one
    double dx = 0, dy = 0;
    if (valid) {
        if (have_valid_ && invalid_run_ == 0 && !gap) { dx = x - last_x_; dy = y - last_y_; }
        last_x_ = x; last_y_ = y; have_valid_ = true; invalid_run_ = 0;
    } else {
        ++invalid_run_;
    }
    dxh_[static_cast<size_t>(dpos_)] = dx; dyh_[static_cast<size_t>(dpos_)] = dy; dpos_ = (dpos_ + 1) % 3;
    const double vx = (dxh_[0] + dxh_[1] + dxh_[2]) / 3.0 * cfg_.fs_hz, vy = (dyh_[0] + dyh_[1] + dyh_[2]) / 3.0 * cfg_.fs_hz;
    const double speed = std::hypot(vx, vy);
    if (valid) {
        nvx_[static_cast<size_t>(npos_)] = static_cast<float>(vx); nvy_[static_cast<size_t>(npos_)] = static_cast<float>(vy);
        npos_ = (npos_ + 1) % kNoiseWin; if (ncount_ < kNoiseWin) ++ncount_;
        if (++nsince_ >= kNoiseEvery && ncount_ >= 50) { nsince_ = 0; update_noise(); }
    }
    if (static_cast<double>(invalid_run_) * 1000.0 / cfg_.fs_hz >= cfg_.safe_invalid_ms) add(Reason::LongInvalid);

    // ---- engine and network
    if (eo.health == gaze::Health::Failed) add(Reason::EngineFailed);
    else if (eo.health == gaze::Health::Degraded) add(Reason::NetworkUntrusted);

    // ---- (c) confidence and out of distribution
    const bool unsure = valid && eo.source == gaze::Source::Network && std::fabs(eo.p_saccade - 0.5f) <= cfg_.low_conf_band;
    unsure_ema_ += ((unsure ? 1.0 : 0.0) - unsure_ema_) / 1024.0;
    if (unsure_ema_ > cfg_.low_conf_rate) add(Reason::LowConfidence);
    if (ncount_ >= kNoiseWin && std::max(sigx_, sigy_) > cfg_.ood_sigma_hi_deg_s) add(Reason::OutOfDistribution);
    const bool nomove = valid && eo.state == gaze::State::Saccade && speed < cfg_.nomove_deg_s;   // judged in every state (also DEGRADED)
    nomove_ema_ += ((nomove ? 1.0 : 0.0) - nomove_ema_) / 512.0;
    if (nomove_ema_ > cfg_.nomove_rate) add(Reason::NoMovement);

    // ---- (d) deadline: a late result is replaced by the safe value (Invalid) for this sample
    const bool late = compute_us > cfg_.budget_us;
    if (late) ++st_.deadline_misses;
    slide(miss_bit_, miss_count_, cfg_.deadline_window, late);
    // the state follows the number of misses in the window (bit DeadlineMiss is set while the window holds a miss)
    if (miss_count_ >= cfg_.deadline_safe) add(Reason::DeadlineMiss, static_cast<int>(GuardState::Safe));
    else if (miss_count_ >= cfg_.deadline_degraded) add(Reason::DeadlineMiss, static_cast<int>(GuardState::Degraded));

    // ---- repeated physiology violations (counted from the previous samples' dropped events)
    if (viol_count_ >= cfg_.violation_degraded) add(Reason::RepeatedViolations);

    // ---- state machine: up at once, down one level after recover_samples calmer samples
    bool went_up = false;
    if (target > state_) { state_ = target; calm_run_ = 0; went_up = true; }
    else if (target < state_) {
        if (++calm_run_ >= cfg_.recover_samples) { state_ = static_cast<GuardState>(static_cast<int>(state_) - 1); calm_run_ = 0; }
    } else calm_run_ = 0;
    if (state_ != GuardState::Normal && main == Reason::None) add(Reason::Hysteresis);

    // ---- label of this sample
    const Label fb = fallback_label(valid, speed, vx, vy);     // always run, so that its state is ready when DEGRADED starts
    Label label;
    if (!valid || late || state_ == GuardState::Safe) label = Label::Invalid;
    else if (state_ == GuardState::Degraded) label = fb;
    else label = eo.state == gaze::State::Saccade ? Label::Saccade : eo.state == gaze::State::Fixation ? Label::Fixation : Label::Invalid;

    // ---- (b) events from the guard's own label stream, each checked before it is emitted
    bool dropped_now = false;
    const double ms_per_sample = 1000.0 / cfg_.fs_hz;
    if (went_up && in_event_) {                  // the event was labeled by a source that is no longer trusted: never emit it
        end_event(ev_last_, Reason::EventAborted, out);
        ev_gap_ = 0;
    }
    if (label == Label::Saccade && suppress_) {
        // a run that was already dropped as too long goes on: every further max_duration counts as one more violation, so that a network
        // stuck at "saccade" pushes the guard to DEGRADED (3 violations in 2 s) instead of only losing events
        ev_gap_ = 0; ev_last_ = n_;
        if (static_cast<double>(n_ - ev_onset_ + 1) * ms_per_sample > cfg_.max_duration_ms) {
            ev_onset_ = n_; dropped_now = true; add(Reason::EventTooLong);
        }
    } else if (label == Label::Saccade) {
        if (!in_event_) {
            in_event_ = true; ev_onset_ = n_; ev_x0_ = px; ev_y0_ = py; ev_peak_ = 0;
            ev_reason_ = (last_offset_ >= 0 && static_cast<double>(n_ - last_offset_) * ms_per_sample < cfg_.refractory_ms)
                             ? Reason::EventRefractory : Reason::None;
        }
        ev_gap_ = 0; ev_last_ = n_;
        ev_x1_ = x; ev_y1_ = y; ev_peak_ = std::max(ev_peak_, speed);
        if (static_cast<double>(n_ - ev_onset_ + 1) * ms_per_sample > cfg_.max_duration_ms) {   // do not wait for the end of an endless run
            end_event(n_, Reason::EventTooLong, out); suppress_ = true; dropped_now = true; ev_onset_ = n_;
        }
    } else if ((in_event_ || suppress_) && label == Label::Fixation && ++ev_gap_ <= cfg_.merge_gap_samples) {
        // a fixation gap of at most merge_gap_samples inside a saccade (label flicker at the edges): the event goes on; whether it
        // really ended is known merge_gap_samples later
    } else {
        if (suppress_) last_offset_ = ev_last_;   // the dropped run really ended there (refractory time counts from it)
        suppress_ = false;
        if (in_event_) {
            // invalid data or the SAFE state interrupt the event: it is dropped (never completed by a guess)
            const Reason forced = label == Label::Invalid ? Reason::EventAborted : Reason::None;
            end_event(ev_last_, forced, out);
            // only impossible kinematics count towards DEGRADED (too fast, too large, too long, too slow). Aborts (blinks), 1-2 sample
            // blips (EventTooShort) and a second piece of a split saccade (EventRefractory: e.g. the post-saccadic oscillation labeled
            // apart, 4 times in 10 s of clean dataset 2) are dropped too, but they are normal and say nothing about a failure.
            const Reason r = out.event.reason;
            dropped_now = !out.event.accepted && r != Reason::EventAborted && r != Reason::EventTooShort && r != Reason::EventRefractory;
        }
        ev_gap_ = 0;
    }
    if (out.event_ended && !out.event.accepted) add(out.event.reason);
    slide(viol_bit_, viol_count_, cfg_.violation_window, dropped_now);

    out.state = state_; out.label = label; out.reason = main;
    ++st_.samples;
    if (state_ == GuardState::Normal) ++st_.normal; else if (state_ == GuardState::Degraded) ++st_.degraded; else ++st_.safe;
    ++n_;
}

}  // namespace safety
}  // namespace uneye
