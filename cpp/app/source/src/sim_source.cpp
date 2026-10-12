#include "eyelive/sim_source.hpp"
#include <algorithm>

namespace eyelive {

SimSource::SimSource(const SourceSpec& spec) : spec_(spec) {}

bool SimSource::open(std::string* error) {
    (void)error;
    uneye::SimConfig cfg;
    cfg.seed = spec_.seed;
    cfg.fs = spec_.rate_hz;
    trace_ = uneye::simulate(cfg);
    pos_ = 0;
    finished_ = false;
    return true;
}

void SimSource::close() {
    finished_ = true;
}

std::string SimSource::description() const {
    return "sim://";
}

size_t SimSource::read(std::vector<Sample>& out, size_t max_samples, int timeout_ms) {
    (void)timeout_ms;
    if (finished_ || pos_ >= trace_.x.size()) {
        finished_ = true;
        return 0;
    }

    size_t initial_size = out.size();
    size_t to_read = std::min(max_samples, trace_.x.size() - pos_);

    for (size_t i = 0; i < to_read; ++i) {
        Sample s;
        s.t = (pos_ + i) / spec_.rate_hz;
        s.x = trace_.x[pos_ + i];
        s.y = trace_.y[pos_ + i];
        if (spec_.pixels_per_degree > 0) {
            s.x *= spec_.pixels_per_degree;
            s.y *= spec_.pixels_per_degree;
        }
        s.flags = kFlagValid;
        out.push_back(s);
    }

    pos_ += to_read;
    if (pos_ >= trace_.x.size()) finished_ = true;

    return out.size() - initial_size;
}

std::vector<Event> SimSource::truth() const {
    std::vector<Event> events;
    for (const auto& ev : trace_.events) {
        Event e;
        e.onset = ev.on;
        e.offset = ev.off;
        e.t_onset = e.onset / spec_.rate_hz;
        e.t_offset = e.offset / spec_.rate_hz;
        e.amplitude_deg = ev.amp;
        e.cls = 1; // 1: saccade
        events.push_back(e);
    }
    return events;
}

} // namespace eyelive
