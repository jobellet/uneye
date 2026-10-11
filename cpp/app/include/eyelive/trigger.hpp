// Trigger rules and sinks (CONTRACT). Implemented by task T19. Besides the shared-memory state block, rules can fire UDP messages or write a log line, optionally after a delay.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "eyelive/types.hpp"

namespace eyelive {

enum class TriggerAnchor : uint8_t { OnsetProvisional, OnsetFinal, Offset };

struct TriggerRule {
    std::string name;
    uint8_t cls = 1; bool any_class = false;
    float amp_min_deg = 0, amp_max_deg = 1e9f;       // microsaccade rule: amp_max_deg = 1.0
    TriggerAnchor anchor = TriggerAnchor::OnsetProvisional;
    double delay_ms = 0;                              // fire this long after the anchor (event time mapped to the host clock)
    double refractory_ms = 0;                         // ignore events closer than this to the last firing of the same rule
    std::string sink;                                 // "udp:127.0.0.1:6000", "log:<path>", "stdout"
    std::string payload = "T";                        // sent as is; "{t}", "{amp}", "{cls}" are replaced
    bool enabled = true;
};

struct TriggerLogEntry { std::string rule; double requested_host_s, fired_host_s, anchor_event_t; bool late = false; };   // late: the anchor was published after the requested fire time

class ITriggerEngine {
 public:
    virtual ~ITriggerEngine() = default;
    virtual void set_rules(const std::vector<TriggerRule>& rules) = 0;
    // The engine reports every event (provisional and final) with the host clock time at which the corresponding sample ARRIVED, so that "delay after onset" is exact.
    virtual void on_event(const Event& e, double sample_arrival_host_s) = 0;
    virtual std::vector<TriggerLogEntry> log_snapshot() const = 0;          // last 1000 firings
    virtual void start() = 0;                                               // timer thread
    virtual void stop() = 0;
};
std::unique_ptr<ITriggerEngine> make_trigger_engine();                      // defined by T19

}  // namespace eyelive
