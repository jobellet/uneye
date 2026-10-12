// The host clock of the trigger engine: seconds on std::chrono::steady_clock (monotonic). The engine passes `sample_arrival_host_s` on this clock to ITriggerEngine::on_event and reads TriggerLogEntry times on it.
#pragma once
#include <chrono>
namespace eyelive {
inline double trigger_host_now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace eyelive
