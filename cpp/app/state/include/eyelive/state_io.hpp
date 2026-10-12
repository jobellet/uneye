#pragma once

#include "eyelive/state_block.hpp"
#include "eyelive/types.hpp"
#include <atomic>
#include <cstring>
#include <cstdint>
#include <thread>

namespace eyelive {

class StateBlockWriter {
public:
    explicit StateBlockWriter(void* block);

    void init(double rate_hz, double deg_per_unit, int n_classes);
    void publish_sample(const Sample& sample, float speed_deg_s, uint8_t cls, bool in_event, uint64_t n_samples);
    void publish_event(const Event& event, uint64_t id);
    void publish_prediction(float x, float y, float sx, float sy, double t_valid);
    void set_health(uint32_t flags, uint32_t dropped, float lat_p50_ms, float lat_p95_ms);

private:
    StateBlock* block_;
    std::atomic<uint64_t>* seq_;
};

bool read_state_block(const void* block, StateBlock* out, int max_retries = 100);

}  // namespace eyelive
