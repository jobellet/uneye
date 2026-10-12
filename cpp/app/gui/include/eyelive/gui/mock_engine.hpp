// MockEngine for GUI development (T12)
#pragma once
#include "eyelive/engine.hpp"

namespace eyelive {
namespace gui {

class MockEngine : public IEngine {
public:
    MockEngine() : trace_() {
        // 1 second of fixation
        for (int i = 0; i < 1000; ++i) {
            trace_.push({i * 0.001, 0.0f, 0.0f, kFlagValid});
        }
        // A saccade of 0.05 seconds
        for (int i = 1000; i < 1050; ++i) {
            trace_.push({i * 0.001, (i - 1000) * 0.1f, 0.0f, kFlagValid});
        }
        // A fixation of 0.5 seconds
        for (int i = 1050; i < 1550; ++i) {
            trace_.push({i * 0.001, 5.0f, 0.0f, kFlagValid});
        }
        // A blink of 0.2 seconds
        for (int i = 1550; i < 1750; ++i) {
            trace_.push({i * 0.001, 5.0f, 0.0f, kFlagBlink});
        }
        // Back to fixation
        for (int i = 1750; i < 2000; ++i) {
            trace_.push({i * 0.001, 5.0f, 0.0f, kFlagValid});
        }

        events_.push_back({1000, 1049, 1.0, 1.049, 5.0f, 200.0f, 0.0f, 1, false, EventOrigin::Detector, 0, 0, 1.0f});
        events_.push_back({1550, 1749, 1.55, 1.749, 0.0f, 0.0f, 0.0f, 2, false, EventOrigin::Detector, 0, 0, 1.0f});
    }

    bool start(const SessionConfig& cfg, std::string* error) override {
        stats_.running = true;
        stats_.rate_hz_measured = 1000.0;
        stats_.rate_hz_standard = 1000.0;
        stats_.samples = trace_.size();
        stats_.capacity_s = 3600.0;
        return true;
    }

    void stop() override {
        stats_.running = false;
    }

    EngineStats stats() const override {
        return stats_;
    }

    const ITraceView& trace() const override {
        return trace_;
    }

    std::vector<Event> events() const override {
        return events_;
    }

    std::vector<DetectorInfo> available_detectors() const override {
        return { {"mock_hmm", "Mock HMM", 3, true, 0, -1, false} };
    }

    bool set_detector(const std::string& id, std::string* error) override {
        return true;
    }

    ExpertLabels& expert() override {
        if (!expert_) {
            expert_ = std::make_unique<ExpertLabels>();
        }
        return *expert_;
    }

    bool prelabel(const std::string& detector_id, int64_t from, int64_t to, std::string* error) override {
        return true;
    }

    AgreementStats agreement_with_expert(uint8_t cls) const override {
        return AgreementStats{100, 5, 5, 0.95, 0.90};
    }

    CalibParams fit_calibration_to_expert(uint8_t cls, std::string* error) override {
        return CalibParams{};
    }

    bool start_training(const TrainConfig& cfg, std::string* error) override {
        train_prog_.running = true;
        train_prog_.prog.epoch = 1;
        train_prog_.prog.train_loss = 0.5;
        return true;
    }

    TrainProgress training_progress() const override {
        return train_prog_.prog;
    }

    void cancel_training() override {
        train_prog_.running = false;
        train_prog_.prog.cancelled = true;
    }

    bool deploy_trained_model(const std::string& name, std::string* error) override {
        return true;
    }

    void set_trigger_rules(const std::vector<TriggerRule>& rules) override {
    }

    std::vector<TriggerLogEntry> trigger_log() const override {
        return { {"blink_rule", 1.0, 1.1, 1.0, false} };
    }

private:
    EngineStats stats_;
    VectorTrace trace_;
    std::vector<Event> events_;
    std::unique_ptr<ExpertLabels> expert_;
    struct { bool running = false; TrainProgress prog; } train_prog_;
};

} // namespace gui
} // namespace eyelive
