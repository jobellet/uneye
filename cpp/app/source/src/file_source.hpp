#pragma once
#include "eyelive/source.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

namespace eyelive {

class FileSource : public ISource {
 public:
    FileSource(const SourceSpec& spec);
    ~FileSource() override;

    bool open(std::string* error) override;
    size_t read(std::vector<Sample>& out, size_t max_samples, int timeout_ms) override;
    bool finished() const override { return finished_; }
    void close() override;
    std::string description() const override;

 private:
    SourceSpec spec_;
    std::ifstream file_;
    bool finished_ = false;
    double first_t_ = -1;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace eyelive
