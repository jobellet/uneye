#include "eyelive/state_io.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <fstream>
#include <cstddef>
#include <cassert>
#include <cstring>

using namespace eyelive;

#if defined(__clang__) || defined(__GNUC__)
#define TSAN_IGNORE __attribute__((no_sanitize("thread")))
#else
#define TSAN_IGNORE
#endif

void test_offsets() {
    std::ofstream out("offsets.json");
    out << "{\n";
    out << "  \"magic\": " << offsetof(StateBlock, magic) << ",\n";
    out << "  \"version\": " << offsetof(StateBlock, version) << ",\n";
    out << "  \"seq\": " << offsetof(StateBlock, seq) << ",\n";
    out << "  \"t_last\": " << offsetof(StateBlock, t_last) << ",\n";
    out << "  \"x\": " << offsetof(StateBlock, x) << ",\n";
    out << "  \"y\": " << offsetof(StateBlock, y) << ",\n";
    out << "  \"speed_deg_s\": " << offsetof(StateBlock, speed_deg_s) << ",\n";
    out << "  \"cls\": " << offsetof(StateBlock, cls) << ",\n";
    out << "  \"in_event\": " << offsetof(StateBlock, in_event) << ",\n";
    out << "  \"n_classes\": " << offsetof(StateBlock, n_classes) << ",\n";
    out << "  \"n_samples\": " << offsetof(StateBlock, n_samples) << ",\n";
    out << "  \"event_counter\": " << offsetof(StateBlock, event_counter) << ",\n";
    out << "  \"provisional_counter\": " << offsetof(StateBlock, provisional_counter) << ",\n";
    out << "  \"ev_t_onset\": " << offsetof(StateBlock, ev_t_onset) << ",\n";
    out << "  \"ev_t_offset\": " << offsetof(StateBlock, ev_t_offset) << ",\n";
    out << "  \"ev_amplitude_deg\": " << offsetof(StateBlock, ev_amplitude_deg) << ",\n";
    out << "  \"ev_peak_speed_deg_s\": " << offsetof(StateBlock, ev_peak_speed_deg_s) << ",\n";
    out << "  \"ev_direction_rad\": " << offsetof(StateBlock, ev_direction_rad) << ",\n";
    out << "  \"ev_cls\": " << offsetof(StateBlock, ev_cls) << ",\n";
    out << "  \"ev_provisional\": " << offsetof(StateBlock, ev_provisional) << ",\n";
    out << "  \"ev_pad\": " << offsetof(StateBlock, ev_pad) << ",\n";
    out << "  \"ev_id\": " << offsetof(StateBlock, ev_id) << ",\n";
    out << "  \"pred_x\": " << offsetof(StateBlock, pred_x) << ",\n";
    out << "  \"pred_y\": " << offsetof(StateBlock, pred_y) << ",\n";
    out << "  \"pred_sigma_x\": " << offsetof(StateBlock, pred_sigma_x) << ",\n";
    out << "  \"pred_sigma_y\": " << offsetof(StateBlock, pred_sigma_y) << ",\n";
    out << "  \"pred_t_valid\": " << offsetof(StateBlock, pred_t_valid) << ",\n";
    out << "  \"pred_valid\": " << offsetof(StateBlock, pred_valid) << ",\n";
    out << "  \"pred_pad\": " << offsetof(StateBlock, pred_pad) << ",\n";
    out << "  \"health_flags\": " << offsetof(StateBlock, health_flags) << ",\n";
    out << "  \"dropped_packets\": " << offsetof(StateBlock, dropped_packets) << ",\n";
    out << "  \"detect_latency_ms_p50\": " << offsetof(StateBlock, detect_latency_ms_p50) << ",\n";
    out << "  \"detect_latency_ms_p95\": " << offsetof(StateBlock, detect_latency_ms_p95) << ",\n";
    out << "  \"deg_per_unit\": " << offsetof(StateBlock, deg_per_unit) << ",\n";
    out << "  \"rate_hz\": " << offsetof(StateBlock, rate_hz) << "\n";
    out << "}\n";

    assert(sizeof(StateBlock) == 256);
    std::cout << "Offsets test passed. sizeof(StateBlock) is 256.\n";
}

TSAN_IGNORE
void test_multithread() {
    alignas(64) uint8_t raw_block[256];
    std::memset(raw_block, 0, sizeof(raw_block));

    std::atomic<bool> run{true};

    std::thread writer_thread([&raw_block, &run]() {
        StateBlockWriter writer(raw_block);
        writer.init(1000.0, 1.0, 3);

        for (uint64_t i = 1; i <= 1000000; ++i) {
            Sample s;
            s.t = static_cast<double>(i) * 0.001;
            s.x = static_cast<float>(i);
            s.y = static_cast<float>(i * 2);
            writer.publish_sample(s, static_cast<float>(i * 3), 1, false, i);
        }
        run = false;
    });

    std::thread reader_thread([&raw_block, &run]() {
        uint64_t last_n = 0;
        int max_observed_retries = 0;

        while (run) {
            StateBlock out;
            bool success = read_state_block(raw_block, &out, 100);

            if (success && out.n_samples > 0) {
                assert(out.magic == kStateMagic);
                assert(out.version == kStateVersion);
                assert(out.x == static_cast<float>(out.n_samples));
                assert(out.y == static_cast<float>(out.n_samples * 2));
                assert(out.speed_deg_s == static_cast<float>(out.n_samples * 3));
                assert(out.t_last == static_cast<double>(out.n_samples) * 0.001);
                last_n = out.n_samples;
            }
        }

        std::cout << "Reader completed.\n";
    });

    writer_thread.join();
    reader_thread.join();
    std::cout << "Multithread test passed.\n";
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    test_offsets();
    test_multithread();
    return 0;
}
