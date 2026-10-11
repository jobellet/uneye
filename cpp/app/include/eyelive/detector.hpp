// Detector interface and registry (CONTRACT). A "detector" is any pre-labeler / online detector: the label-free HMM, the fast threshold detector, a pre-trained U'n'Eye, a TCN, a network trained by the user.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "eyelive/trace.hpp"
#include "eyelive/types.hpp"

namespace eyelive {

struct DetectorInfo {
    std::string id;                    // stable machine name, e.g. "hmm_accel", "fast_threshold", "unet_1+2+3", "unet_andersson5", "tcn_causal", "user:<file>"
    std::string name;                  // label shown in the GUI
    int n_classes = 2;                 // classes it can output (background included)
    bool causal = true;                // false: needs samples after the event (see decision_delay_ms)
    double decision_delay_ms = 0;      // typical delay between an event's offset and its FINAL publication
    double provisional_latency_ms = -1;// delay between the onset and the PROVISIONAL event, -1 = no provisional events
    bool needs_degrees = false;        // trained on degree data: the engine multiplies positions by SessionInfo::deg_per_unit (or the selected scale) before calling it
};

class IDetector {
 public:
    virtual ~IDetector() = default;
    virtual DetectorInfo info() const = 0;
    virtual void reset(const SessionInfo& session) = 0;                       // new recording
    virtual void set_scale(double deg_per_unit) {}                            // used by detectors with needs_degrees; default: ignore
    virtual bool set_option(const std::string& key, const std::string& value) { (void)key; (void)value; return false; }   // detector-specific settings, e.g. hmm_accel: ("offset_rule", "first"|"last"|"none"); false = unknown key or bad value
    // streaming: consume samples up to index `to` (exclusive) of the growing trace. FINAL events are appended once each, in increasing onset order, as soon as they are final;
    // PROVISIONAL events (if supported) are appended when first detected and are superseded by the final event covering the same movement.
    virtual void advance(const ITraceView& trace, int64_t to, std::vector<Event>& final_events, std::vector<Event>& provisional_events) = 0;
    // batch: label the whole available trace (used for pre-labeling a recorded block); must give the same events as streaming up to the boundary effects documented by the implementation
    virtual void detect_all(const ITraceView& trace, std::vector<Event>& events) = 0;
};

using DetectorFactory = std::function<std::unique_ptr<IDetector>(std::string* error)>;

class DetectorRegistry {                                                        // header-only singleton, filled by static registrars in each detector's .cpp
 public:
    static DetectorRegistry& instance() { static DetectorRegistry r; return r; }
    void add(const std::string& id, DetectorFactory f) { factories_[id] = std::move(f); }
    std::vector<std::string> ids() const { std::vector<std::string> v; for (auto& kv : factories_) v.push_back(kv.first); return v; }
    std::unique_ptr<IDetector> create(const std::string& id, std::string* error = nullptr) const {
        auto it = factories_.find(id); if (it == factories_.end()) { if (error) *error = "unknown detector " + id; return nullptr; } return it->second(error);
    }
 private:
    std::map<std::string, DetectorFactory> factories_;
};

struct DetectorRegistrar {                                                      // usage: static eyelive::DetectorRegistrar reg("hmm_accel", [](std::string*){ return std::make_unique<HmmDetector>(); });
    DetectorRegistrar(const std::string& id, DetectorFactory f) { DetectorRegistry::instance().add(id, std::move(f)); }
};

}  // namespace eyelive
