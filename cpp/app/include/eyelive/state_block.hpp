// The real-time state block published at offset 256 of the session file (CONTRACT). MATLAB (Psychtoolbox, memmapfile), Python (PsychoPy, numpy.memmap / mmap) and any other program read it
// to follow the eye in real time without a driver. Implemented by task T03 (writer, readers, documentation). Layout is FIXED: version 1, little endian, 256 bytes.
//
// Seqlock protocol. Writer: seq += 1 (odd = being written); memory barrier; write the fields; memory barrier; seq += 1 (even). Reader: s1 = seq; if odd retry; copy the block; memory barrier; s2 = seq;
// if s1 != s2 retry. A reader may retry at most a few times: the writer holds the block for microseconds.
#pragma once
#include <cstdint>

namespace eyelive {

#pragma pack(push, 1)
struct StateBlock {
    uint32_t magic;                // 0x42534C45 ("ELSB")
    uint32_t version;              // 1
    uint64_t seq;                  // seqlock counter
    double   t_last;               // time of the last sample (tracker clock, s)
    float    x, y;                 // last gaze position, degrees if deg_per_unit was set, else stream units
    float    speed_deg_s;          // current speed
    uint8_t  cls;                  // class of the last sample (0 = background)
    uint8_t  in_event;             // 1 while an event is in progress (provisional or confirmed)
    uint16_t n_classes;
    uint64_t n_samples;            // samples recorded so far
    uint64_t event_counter;        // increments by one each time a FINAL event is published (readers compare it with their last value)
    uint64_t provisional_counter;  // increments each time a PROVISIONAL event is published
    // last event (final if final_valid, else the last provisional)
    double   ev_t_onset, ev_t_offset;
    float    ev_amplitude_deg, ev_peak_speed_deg_s, ev_direction_rad;
    uint8_t  ev_cls;
    uint8_t  ev_provisional;
    uint16_t ev_pad;
    uint64_t ev_id;                // = event_counter or provisional_counter at publication
    // predicted landing point of the saccade in progress (task: position prediction)
    float    pred_x, pred_y, pred_sigma_x, pred_sigma_y;
    double   pred_t_valid;         // time at which the prediction was made
    uint8_t  pred_valid;
    uint8_t  pred_pad[3];
    // health
    uint32_t health_flags;         // bit0 no data for 100 ms, bit1 gaps, bit2 recorder full soon, bit3 detector degraded, bit4 scale estimated (not given)
    uint32_t dropped_packets;
    float    detect_latency_ms_p50, detect_latency_ms_p95;
    double   deg_per_unit;         // scale in use
    double   rate_hz;
    uint8_t  reserved[256 - 4 - 4 - 8 - 8 - 4 - 4 - 4 - 1 - 1 - 2 - 8 - 8 - 8 - 8 - 8 - 4 - 4 - 4 - 1 - 1 - 2 - 8 - 16 - 8 - 1 - 3 - 4 - 4 - 4 - 4 - 8 - 8];
};
#pragma pack(pop)
static_assert(sizeof(StateBlock) == 256, "StateBlock must be exactly 256 bytes");

constexpr uint32_t kStateMagic = 0x42534C45u;
constexpr uint32_t kStateVersion = 1;

}  // namespace eyelive
