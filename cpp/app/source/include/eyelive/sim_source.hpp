#pragma once
#include "eyelive/source.hpp"
#include "../../../include/uneye/simulator.hpp"
#include <string>
#include <vector>

namespace eyelive {

class SimSource : public ISource {
 public:
    SimSource(const SourceSpec& spec);
    ~SimSource() override = default;

    bool open(std::string* error) override;
    size_t read(std::vector<Sample>& out, size_t max_samples, int timeout_ms) override;
    bool finished() const override { return finished_; }
    void close() override;
    std::string description() const override;

    std::vector<Event> truth() const;

 private:
    SourceSpec spec_;
    bool finished_ = false;
    uneye::SimTrace trace_;
    size_t pos_ = 0;
};

} // namespace eyelive
