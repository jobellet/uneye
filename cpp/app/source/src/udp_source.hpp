#pragma once
#include "eyelive/source.hpp"
#include <deque>
#include <string>
#include <vector>

namespace eyelive {

class UdpSource : public ISource {
 public:
    UdpSource(const SourceSpec& spec);
    ~UdpSource() override;

    bool open(std::string* error) override;
    size_t read(std::vector<Sample>& out, size_t max_samples, int timeout_ms) override;
    bool finished() const override { return false; }
    void close() override;
    std::string description() const override;
    uint64_t dropped_packets() const override { return dropped_; }

 private:
    SourceSpec spec_;
    uint64_t dropped_ = 0;
    std::deque<Sample> pending_;     // samples of a packet that did not fit in max_samples: delivered by the next read() (a text packet can hold thousands of lines)
    static constexpr size_t kMaxPending = 1u << 20;   // a sender that floods faster than the engine reads loses the oldest samples (counted in dropped_packets), memory stays bounded
#ifdef _WIN32
    unsigned long long socket_ = ~0ull; // INVALID_SOCKET
#else
    int socket_ = -1;
#endif
};

} // namespace eyelive
