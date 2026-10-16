// Engine facade used by the GUI and the CLI (CONTRACT). The real implementation is task T18 (threads, queues, wiring of recorder, source, detectors, state block, triggers, trainer);
// the GUI tasks develop against MockEngine (gui/include/eyelive/gui/mock_engine.hpp, provided by T12) and never include a module implementation.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "eyelive/annotation.hpp"
#include "eyelive/calib.hpp"
#include "eyelive/detector.hpp"
#include "eyelive/source.hpp"
#include "eyelive/trace.hpp"
#include "eyelive/trainer.hpp"
#include "eyelive/trigger.hpp"

namespace eyelive {

struct SessionConfig {
    SourceSpec source;
    std::string session_path = "session.eyelive";
    double duration_s = 3600;           // planned duration: with the measured rate it sizes the preallocation
    double rate_hz = 0;                 // 0 = measure from the first 500 ms
    double deg_per_unit = 0;            // 0 = automatic (calib.hpp: select_scale once enough events exist); otherwise exact (e.g. 1/pixels_per_degree)
    std::string detector_id = "hmm_accel";
    int n_classes = 3;                  // fixation, saccade, blink (blink events come from the invalid-sample runs flagged by the input guard)
};

// Forecast of the eye position made at every sample by the gaze engine (cpp/include/uneye/gaze_engine.hpp: two causal networks + fixed landing tables, 1 kHz only). Positions are in STREAM units like the trace;
// the forecast of sample i describes the position of sample i + 10 ms and i + 20 ms. The scales are those of a Laplace distribution: an 80 % interval is +- 1.61 scale.
struct ForecastSample { float x10 = 0, y10 = 0, x20 = 0, y20 = 0, sx10 = 0, sy10 = 0, sx20 = 0, sy20 = 0; bool valid = false, ballistic = false;
                       float lx = 0, ly = 0; bool landing = false;          // estimated end point of the ongoing saccade (stream units): the tables' estimate, or the chosen candidate when candidates are set
                       int candidate = -1; float candidate_confidence = 0; };   // index in the candidate list given to candidates_set (-1: none chosen)
struct Candidate { float x = 0, y = 0, weight = 1; };                      // a possible target (stream units, same frame as the gaze); weight = prior (saliency, relevance), > 0

struct EngineStats {
    bool running = false; double rate_hz_measured = 0; double rate_hz_standard = 0; uint64_t samples = 0, dropped_packets = 0, gaps = 0;
    double recorded_s = 0, capacity_s = 0; uint64_t bytes_allocated = 0;
    double cpu_percent = 0; double detect_latency_ms_p50 = 0, detect_latency_ms_p95 = 0; double publish_jitter_ms_p95 = 0;
    uint64_t events_final = 0, events_provisional = 0;
    double deg_per_unit = 1; bool scale_estimated = false; double scale_agreement = 0; bool scale_low_confidence = false;
    std::string warning;                // human-readable, empty if none
};

// Assisted annotation (the owner's workflow): the label-free detector labels the data as it arrives; every correction of the expert marks its neighbourhood as reviewed and, after a quiet delay (default 5 s), a U'n'Eye is trained on
// "expert labels where reviewed, automatic labels elsewhere"; when it is ready it labels the windows that follow the reviewed ones (and the new data) instead of the label-free detector, and a new model is trained after the next corrections.
struct AutoLearnConfig {
    bool enabled = false;
    double retrain_delay_s = 5.0;         // quiet time after the last correction before a (new) model is trained
    double chunk_s = 2.0;                 // new data is labelled in blocks of this length (live recording)
    double adopt_min_f1 = 0.6;            // a trained model labels the following windows only if its validation event F1 is at least this (a model that predicts nothing must not replace the labels); a rejected one is kept and continues learning
    int n_threads = 0;                    // threads of a training (0 = automatic: one while a session records, all cores otherwise)
    int max_epochs = 400;                 // hard cap of a training (early stopping normally ends it long before)
    int expected_epochs = 50;             // only for the progress bar: training goes on while the validation loss still decreases and stops by early stopping
    bool pretrained_start = false;        // first model starts from the pretrained network of the repository (later ones start from the previous model)
    int n_classes = 3;
    double max_seconds_of_data = 50.0;    // most recent labelled data used by a training
    std::string first_labeler = "hmm_accel";
};
struct AutoGeneration {                   // one labeler: generation 0 = the label-free detector, k = the k-th trained model
    int id = 0; std::string labeler; double reviewed_s = 0; int n_events = 0; double f1 = 0, kappa = 0;   // agreement of ITS labels with your final labels, on the windows it labelled and you reviewed
    double val_f1 = 0, val_kappa = 0, train_s = 0; int epochs = 0;                                          // from its training (held-out windows)
};
struct AutoLearnStatus {
    bool enabled = false; std::string labeler = "hmm_accel"; int generation = 0; bool labeling = false; bool training = false;
    int epoch = 0, expected_epochs = 50; double progress = 0, train_loss = 0, val_loss = 0, val_f1 = 0; double retrain_in_s = -1; double labelled_s = 0;
    std::vector<AutoGeneration> history; std::string message;
};

// Stimulus feed: a script shares frames and / or candidate targets through a memory-mapped file (cpp/app/engine/include/eyelive/frame_mailbox.hpp, writers in cpp/app/tools).
// Explicit candidates written by the script have priority; otherwise, with saliency = true, the PQFT saliency peaks of the newest frame become the candidates.
struct StimulusFeedConfig { std::string path; bool saliency = false; int max_peaks = 16; double rel_threshold = 0.3; double use_until_ms = 1e9; };
struct StimulusFeedStatus { bool open = false; uint64_t frames = 0, candidate_updates = 0; int candidates = 0; bool explicit_candidates = false; double last_frame_age_s = -1; int width = 0, height = 0; std::string error; };

class IEngine {
 public:
    virtual ~IEngine() = default;
    // session
    virtual bool start(const SessionConfig& cfg, std::string* error) = 0;   // allocates the session file (fails loudly), starts the threads
    virtual void stop() = 0;
    virtual EngineStats stats() const = 0;
    virtual const ITraceView& trace() const = 0;                            // valid between start() and stop(); also after stop() until the next start()
    virtual void events_from(uint64_t first_final, std::vector<Event>& out) const = 0;      // the final events from index first_final on, then the provisional ones (cheap: a viewer keeps the finals it already has)
    virtual std::vector<Event> events() const = 0;                          // detector events (final, then still-provisional ones), snapshot copy
    // forecast of the future eye position (see ForecastSample); empty / invalid when the stream is not at 1 kHz or the models are missing
    virtual bool forecast_available() const = 0;
    virtual void forecast_enable(bool on) = 0;                 // the forecast costs CPU: it is computed only while someone asks for it (call every frame while it is displayed; it stops 3 s after the last call)
    virtual void candidates_set(const std::vector<Candidate>& c, double use_until_ms = 1e9) = 0;   // possible targets of the next saccades (empty = none): the landing estimate then picks the most plausible one; can be called at any time (e.g. at every fixation, with weights from a prior on the saccade vector). Later than use_until_ms after the onset the tables' estimate is kept (reading: ~ 12 ms)
    virtual std::vector<Candidate> candidates() const = 0;                 // the list currently used
    virtual bool stimulus_feed_open(const StimulusFeedConfig& cfg, std::string* error) = 0;   // polls the mailbox file every 5 ms in its own thread (independent of start / stop)
    virtual void stimulus_feed_close() = 0;
    virtual StimulusFeedStatus stimulus_feed_status() const = 0;
    virtual size_t forecasts(int64_t from, int64_t to, std::vector<ForecastSample>& out) const = 0;      // one entry for each sample of [from, to) (valid = false when not available)
    // detectors
    virtual std::vector<DetectorInfo> available_detectors() const = 0;
    virtual bool set_detector(const std::string& id, std::string* error) = 0;     // hot swap at a safe moment; the previous events are kept (provenance)
    // expert review
    virtual ExpertLabels& expert() = 0;
    virtual bool prelabel(const std::string& detector_id, int64_t from, int64_t to, std::string* error) = 0;   // runs the detector in batch on [from, to] and loads the result into expert()
    virtual AgreementStats agreement_with_expert(uint8_t cls) const = 0;
    virtual CalibParams fit_calibration_to_expert(uint8_t cls, std::string* error) = 0;     // mode 2
    // training
    virtual bool start_training(const TrainConfig& cfg, std::string* error) = 0;            // uses the reviewed material of expert(); the trainer runs in the background with capped cores
    virtual TrainProgress training_progress() const = 0;
    virtual void cancel_training() = 0;
    virtual bool deploy_trained_model(const std::string& name, std::string* error) = 0;     // saves the model and registers it as detector "user:<name>"
    // assisted annotation: call autolearn_poll from the thread that owns expert() (the GUI thread, once per frame, or a loop in a program without window); pass true when the expert changed something since the previous call
    virtual void autolearn_set(const AutoLearnConfig& cfg) = 0;
    virtual AutoLearnStatus autolearn_poll(bool expert_edited) = 0;
    virtual bool autolearn_train_now(std::string* error) = 0;
    // triggers
    virtual void set_trigger_rules(const std::vector<TriggerRule>& rules) = 0;
    virtual std::vector<TriggerLogEntry> trigger_log() const = 0;
};

std::unique_ptr<IEngine> make_engine();                                     // defined by T18

}  // namespace eyelive
