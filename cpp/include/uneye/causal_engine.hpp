// Stateful, strictly causal network (see online/causal_net.py). One call per new sample,
// cost independent of the time-bin length. Output at t depends only on samples <= t.
#pragma once
#include <memory>
#include <string>

namespace uneye {

class StepEngine {
public:
    virtual ~StepEngine() = default;
    virtual int classes() const = 0;
    virtual int receptive_field() const = 0;
    virtual void reset() = 0;                                    // zero history (= start of a recording)
    virtual void step(float dx, float dy, float* prob) = 0;      // prob[classes], softmax
};

// Reads the .bin written by online/export_causal.py (magic "UNCZ").
std::unique_ptr<StepEngine> make_causal_engine(const std::string& bin_path);
bool is_causal_file(const std::string& path);

}  // namespace uneye
