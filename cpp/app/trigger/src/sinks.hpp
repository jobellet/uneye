// Trigger sinks (private): UDP datagram, append-only log file, stdout. send() never blocks and never throws; failures are counted.
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace eyelive {

class Sink {
 public:
    virtual ~Sink() = default;
    virtual bool open(std::string* error) = 0;                 // create the socket / open the file (called by start() or by set_rules() when running)
    virtual void send(const char* data, size_t n) = 0;
    std::atomic<uint64_t> sent{0}, errors{0};
};

// "udp:host:port", "log:<path>", "stdout"; nullptr + message for anything else
std::shared_ptr<Sink> make_sink(const std::string& spec, std::string* error);

}  // namespace eyelive
