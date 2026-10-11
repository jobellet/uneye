// The contract headers compile together and their small concrete parts behave (VectorTrace, DetectorRegistry, StateBlock size).
#include <cstdio>
#include "eyelive/annotation.hpp"
#include "eyelive/calib.hpp"
#include "eyelive/detector.hpp"
#include "eyelive/engine.hpp"
#include "eyelive/gui_tab.hpp"
#include "eyelive/recorder.hpp"
#include "eyelive/source.hpp"
#include "eyelive/state_block.hpp"
#include "eyelive/trace.hpp"
#include "eyelive/trainer.hpp"
#include "eyelive/trigger.hpp"
using namespace eyelive;
namespace { struct Null : IDetector { DetectorInfo info() const override { DetectorInfo i; i.id = "null"; return i; } void reset(const SessionInfo&) override {} void advance(const ITraceView&, int64_t, std::vector<Event>&, std::vector<Event>&) override {} void detect_all(const ITraceView&, std::vector<Event>&) override {} };
            DetectorRegistrar reg("null", [](std::string*) { return std::make_unique<Null>(); }); }
int main() {
    int bad = 0; VectorTrace tr; for (int i = 0; i < 5; ++i) { Sample s; s.t = i * 0.001; s.x = (float)i; tr.push(s); } bad += tr.size() != 5 || !tr.valid(3) || tr.x(4) != 4.0f;
    bad += DetectorRegistry::instance().create("null") == nullptr || DetectorRegistry::instance().create("nope") != nullptr;
    bad += sizeof(StateBlock) != 256 || kMaxClasses != 5 || kIgnoreLabel != 255;
    std::printf("contracts %s\n", bad ? "FAIL" : "OK"); return bad ? 1 : 0;
}
