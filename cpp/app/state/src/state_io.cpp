#include "eyelive/state_io.hpp"

#if defined(__clang__) || defined(__GNUC__)
#define TSAN_IGNORE __attribute__((no_sanitize("thread")))
#else
#define TSAN_IGNORE
#endif

namespace eyelive {

StateBlockWriter::StateBlockWriter(void* block)
    : block_(static_cast<StateBlock*>(block)) {
    // seq_ is pointing directly to block_->seq to perform atomic operations
    seq_ = reinterpret_cast<std::atomic<uint64_t>*>(&block_->seq);
    // Initialize standard fields
    block_->magic = kStateMagic;
    block_->version = kStateVersion;
    seq_->store(0, std::memory_order_relaxed); // Initialize seqlock sequence to 0 (even)
}

void StateBlockWriter::init(double rate_hz, double deg_per_unit, int n_classes) {
    uint64_t current_seq = seq_->load(std::memory_order_relaxed);
    seq_->store(current_seq + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    block_->rate_hz = rate_hz;
    block_->deg_per_unit = deg_per_unit;
    block_->n_classes = n_classes;

    std::atomic_thread_fence(std::memory_order_release);
    seq_->store(current_seq + 2, std::memory_order_relaxed);
}

void StateBlockWriter::publish_sample(const Sample& sample, float speed_deg_s, uint8_t cls, bool in_event, uint64_t n_samples) {
    uint64_t current_seq = seq_->load(std::memory_order_relaxed);
    seq_->store(current_seq + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    block_->t_last = sample.t;
    block_->x = sample.x;
    block_->y = sample.y;
    block_->speed_deg_s = speed_deg_s;
    block_->cls = cls;
    block_->in_event = in_event ? 1 : 0;
    block_->n_samples = n_samples;

    std::atomic_thread_fence(std::memory_order_release);
    seq_->store(current_seq + 2, std::memory_order_relaxed);
}

void StateBlockWriter::publish_event(const Event& event, uint64_t id) {
    uint64_t current_seq = seq_->load(std::memory_order_relaxed);
    seq_->store(current_seq + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    if (event.provisional) {
        block_->provisional_counter += 1;
    } else {
        block_->event_counter += 1;
    }

    block_->ev_t_onset = event.t_onset;
    block_->ev_t_offset = event.t_offset;
    block_->ev_amplitude_deg = event.amplitude_deg;
    block_->ev_peak_speed_deg_s = event.peak_speed_deg_s;
    block_->ev_direction_rad = event.direction_rad;
    block_->ev_cls = event.cls;
    block_->ev_provisional = event.provisional ? 1 : 0;
    block_->ev_id = id;

    std::atomic_thread_fence(std::memory_order_release);
    seq_->store(current_seq + 2, std::memory_order_relaxed);
}

void StateBlockWriter::publish_prediction(float x, float y, float sx, float sy, double t_valid) {
    uint64_t current_seq = seq_->load(std::memory_order_relaxed);
    seq_->store(current_seq + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    block_->pred_x = x;
    block_->pred_y = y;
    block_->pred_sigma_x = sx;
    block_->pred_sigma_y = sy;
    block_->pred_t_valid = t_valid;
    block_->pred_valid = 1;

    std::atomic_thread_fence(std::memory_order_release);
    seq_->store(current_seq + 2, std::memory_order_relaxed);
}

void StateBlockWriter::set_health(uint32_t flags, uint32_t dropped, float lat_p50_ms, float lat_p95_ms) {
    uint64_t current_seq = seq_->load(std::memory_order_relaxed);
    seq_->store(current_seq + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    block_->health_flags = flags;
    block_->dropped_packets = dropped;
    block_->detect_latency_ms_p50 = lat_p50_ms;
    block_->detect_latency_ms_p95 = lat_p95_ms;

    std::atomic_thread_fence(std::memory_order_release);
    seq_->store(current_seq + 2, std::memory_order_relaxed);
}

TSAN_IGNORE
bool read_state_block(const void* block, StateBlock* out, int max_retries) {
    const StateBlock* in_block = static_cast<const StateBlock*>(block);
    const std::atomic<uint64_t>* seq_in = reinterpret_cast<const std::atomic<uint64_t>*>(&in_block->seq);

    for (int retry = 0; retry < max_retries; ++retry) {
        uint64_t s1 = seq_in->load(std::memory_order_relaxed);
        if ((s1 & 1) != 0) {
            std::this_thread::yield();
            continue;
        }

        std::atomic_thread_fence(std::memory_order_acquire);

        std::memcpy(out, in_block, sizeof(StateBlock));

        std::atomic_thread_fence(std::memory_order_acquire);

        uint64_t s2 = seq_in->load(std::memory_order_relaxed);
        if (s1 == s2) {
            return true;
        }
    }
    return false;
}

}  // namespace eyelive
