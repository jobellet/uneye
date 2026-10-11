// EyeLive shared types (CONTRACT: frozen; change only through a contract PR by the integrator, see docs/tasks/README.md).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace eyelive {

constexpr int kMaxClasses = 5;                 // class 0 = background (fixation), 1..4 = user-defined (default: 1 saccade, 2 PSO, 3 pursuit, 4 blink)
constexpr uint8_t kIgnoreLabel = 255;          // per-sample label meaning "not used for training"

enum SampleFlags : uint32_t { kFlagValid = 1u, kFlagBlink = 2u };

struct Sample {                                // one gaze sample; x, y in the stream's own units (degrees or pixels, see SessionInfo::deg_per_unit)
    double t = 0;                              // seconds, tracker clock (or arrival clock if the tracker sends none)
    float x = 0, y = 0;
    uint32_t flags = kFlagValid;
};

enum class EventOrigin : uint8_t { Detector = 0, Expert = 1 };

struct Event {                                 // one labelled movement
    int64_t onset = 0, offset = 0;             // INCLUSIVE sample indices in the recording
    double t_onset = 0, t_offset = 0;          // seconds
    float amplitude_deg = 0, peak_speed_deg_s = 0, direction_rad = 0;
    uint8_t cls = 1;                           // 1..kMaxClasses-1 (0 is the background, never an event)
    bool provisional = false;                  // early event (threshold crossing); replaced later by the final one
    EventOrigin origin = EventOrigin::Detector;
    uint16_t detector_id = 0;                  // which detector / model produced it (provenance)
    uint32_t model_version = 0;
    float confidence = 1.0f;                   // 0..1, used to rank "uncertain first" in the review queue
};

struct SessionInfo {
    double rate_hz = 1000.0;                   // sampling rate (measured or declared)
    double deg_per_unit = 1.0;                 // 1.0 = the stream is in degrees; otherwise degrees per stream unit
    bool scale_estimated = false;              // true when deg_per_unit was selected automatically (calib.hpp: select_scale)
    int n_classes = 2;
    std::vector<std::string> class_names{"fixation", "saccade"};
};

}  // namespace eyelive
