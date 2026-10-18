// eyelive_cli (task T20): headless front-end of the engine, for scripts, servers and the end-to-end tests. See cli/README.md.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <algorithm>
#include "eyelive/engine.hpp"
#include "eyelive/setup.hpp"
#include "eyelive/net_detectors.hpp"

using namespace eyelive;
static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

static void usage() {
    std::puts("usage: eyelive_cli (--udp <port> | --replay <file.csv> [--speed <x>] [--eye auto|left|right] [--ppd <pixels per degree>] | --sim [--rate <Hz>]) [options]\n"
              "  --bind <addr>            UDP bind address (default 127.0.0.1)\n"
              "  --duration <s>           stop after this many seconds of acquisition (replay and sim also stop at the end of the data)\n"
              "  --samples <n>            stop after n recorded samples\n"
              "  --detector <id>          hmm_accel (default), fast_threshold, unet_1+2+3, unet_andersson5, tcn_causal, bitcn_v2 or user:<name>\n"
              "  --session <file>         session file to create (default eyelive_cli_session.eyelive); it must not exist\n"
              "  --scale <deg_per_unit>|auto   unit scale (default 1 = degrees; auto = estimate it)\n"
              "  --rate-hz <Hz>           declared sampling rate (default: measured)\n"
              "  --triggers <file>        trigger rules, one per line: name=..;cls=1;any=0;anchor=provisional|final|offset;delay_ms=0;refractory_ms=0;amp_min=0;amp_max=1e9;sink=udp:127.0.0.1:6000;payload=SACC {t}\n"
              "  --prelabel-seconds <n>   after the session, pre-label the first n seconds with the detector in batch (0 = all)\n"
              "  --events-out <file>      write the pre-labelled (batch) events to this CSV: onset_s,offset_s,class,amplitude_deg,peak_speed_deg_s\n"
              "  --train                  train a U'n'Eye on the pre-labels (no review), deploy it as detector user:cli and report its agreement\n"
              "  --models-dir <dir>       directory of the pretrained models (default: found next to the executable)\n"
              "  --quiet                  do not print events while recording");
}

static bool parse_rule(const std::string& line, TriggerRule& r) { std::string e; if (parse_trigger_rule(line, r, &e)) return true; std::fprintf(stderr, "trigger rule: %s\n", e.c_str()); return false; }

int main(int argc, char** argv) {
    SessionConfig cfg; cfg.session_path = "eyelive_cli_session.eyelive"; cfg.source.kind = ""; cfg.source.speed = -1.0; cfg.duration_s = 0; cfg.deg_per_unit = 1.0; double duration = 0; uint64_t samples = 0; bool train = false, quiet = false; double prelabel_s = -1; std::string triggers, events_out;
    for (int i = 1; i < argc; ++i) { const std::string a = argv[i]; auto need = [&](const char* name) -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); std::exit(2); } return argv[++i]; };
        if (a == "--udp") { cfg.source.kind = "udp"; cfg.source.port = std::atoi(need("--udp")); } else if (a == "--bind") cfg.source.bind_address = need("--bind"); else if (a == "--replay") { cfg.source.kind = "file"; cfg.source.path = need("--replay"); } else if (a == "--speed") cfg.source.speed = std::atof(need("--speed")); else if (a == "--eye") cfg.source.eye = need("--eye"); else if (a == "--ppd") cfg.source.pixels_per_degree = std::atof(need("--ppd"));
        else if (a == "--sim") cfg.source.kind = "sim"; else if (a == "--rate") cfg.source.rate_hz = std::atof(need("--rate")); else if (a == "--rate-hz") cfg.rate_hz = std::atof(need("--rate-hz")); else if (a == "--duration") duration = std::atof(need("--duration")); else if (a == "--samples") samples = uint64_t(std::atoll(need("--samples")));
        else if (a == "--detector") cfg.detector_id = need("--detector"); else if (a == "--session") cfg.session_path = need("--session"); else if (a == "--scale") { const std::string v = need("--scale"); cfg.deg_per_unit = v == "auto" ? 0.0 : std::atof(v.c_str()); }
        else if (a == "--triggers") triggers = need("--triggers"); else if (a == "--prelabel-seconds") prelabel_s = std::atof(need("--prelabel-seconds")); else if (a == "--events-out") events_out = need("--events-out"); else if (a == "--train") train = true; else if (a == "--models-dir") set_models_dir(need("--models-dir")); else if (a == "--quiet") quiet = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; } else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; } }
    if (cfg.source.kind.empty()) { usage(); return 2; }
    if (models_dir().empty()) { namespace fs = std::filesystem; std::error_code ec; fs::path d = fs::absolute(argv[0], ec).parent_path(); for (int up = 0; up < 7 && !d.empty(); ++up, d = d.parent_path()) for (const char* rel : {"models", "cpp/models"}) if (fs::exists(d / rel / "combined.bin", ec)) { set_models_dir((d / rel).string()); up = 99; break; } }
    if (cfg.source.speed < 0) cfg.source.speed = cfg.source.kind == "sim" ? 0.0 : 1.0;                       // a simulation runs as fast as possible unless --speed is given; a replay at real time
    if (cfg.source.kind == "sim" && cfg.source.speed == 0 && duration > 0 && samples == 0) { samples = uint64_t(duration * cfg.source.rate_hz); duration = 0; }   // as fast as possible: --duration counts seconds of DATA
    cfg.duration_s = duration > 0 ? duration + 5 : (samples > 0 ? double(samples) / std::max(1.0, cfg.rate_hz > 0 ? cfg.rate_hz : 1000.0) + 5 : 3600.0);
    std::signal(SIGINT, on_signal); std::signal(SIGTERM, on_signal);
    auto engine = make_engine();
    if (!triggers.empty()) { std::ifstream f(triggers); if (!f) { std::fprintf(stderr, "cannot read %s\n", triggers.c_str()); return 2; } std::vector<TriggerRule> rules; std::string line; int n = 0; while (std::getline(f, line)) { ++n; if (line.empty() || line[0] == '#') continue; TriggerRule r; if (!parse_rule(line, r)) { std::fprintf(stderr, "%s:%d: bad trigger rule\n", triggers.c_str(), n); return 2; } rules.push_back(r); } engine->set_trigger_rules(rules); }
    std::string err; if (!engine->start(cfg, &err)) { std::fprintf(stderr, "start failed: %s\n", err.c_str()); return 1; }
    std::printf("recording to %s (detector %s)\n", cfg.session_path.c_str(), cfg.detector_id.c_str()); std::fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now(); size_t printed = 0; uint64_t last_samples = 0; auto last_change = t0;
    auto print_new = [&]() { if (quiet) return; const auto ev = engine->events(); for (; printed < ev.size(); ++printed) { const Event& e = ev[printed]; if (e.provisional) continue; std::printf("event %.4f %.4f class=%d amp=%.2f peak=%.0f\n", e.t_onset, e.t_offset, int(e.cls), double(e.amplitude_deg), double(e.peak_speed_deg_s)); } std::fflush(stdout); };
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); const auto st = engine->stats(); print_new(); const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (duration > 0 && el >= duration) break; if (samples > 0 && st.samples >= samples) break; if (!st.running) break;
        if (st.samples != last_samples) { last_samples = st.samples; last_change = std::chrono::steady_clock::now(); }
        if (cfg.source.kind != "udp" && st.samples > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - last_change).count() > 1.0) break;      // replay / sim: the data ended
        if (cfg.source.kind == "udp" && duration <= 0 && samples == 0 && st.samples > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - last_change).count() > 10.0) break;       // a stream that went silent for 10 s
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300)); engine->stop(); print_new(); const EngineStats st = engine->stats(); const int64_t n = engine->trace().size();
    std::printf("samples %llu | rate %.2f Hz (standard %.0f) | dropped packets %llu | gaps %llu | events %llu final | latency p50 %.3f ms p95 %.3f ms | scale %.5g deg/unit%s\n", (unsigned long long)st.samples, st.rate_hz_measured, st.rate_hz_standard, (unsigned long long)st.dropped_packets, (unsigned long long)st.gaps, (unsigned long long)st.events_final, st.detect_latency_ms_p50, st.detect_latency_ms_p95, st.deg_per_unit, st.scale_estimated ? " (estimated)" : "");
    if (!st.warning.empty()) std::printf("warning: %s\n", st.warning.c_str());
    if (n < 10) { std::fprintf(stderr, "no data recorded\n"); return 1; }
    if (prelabel_s >= 0 || !events_out.empty() || train) {
        const double rate = engine->trace().info().rate_hz > 0 ? engine->trace().info().rate_hz : 1000.0; const int64_t to = (prelabel_s > 0) ? std::min<int64_t>(n, int64_t(prelabel_s * rate)) : n;
        if (!engine->prelabel(cfg.detector_id, 0, to, &err)) { std::fprintf(stderr, "pre-label failed: %s\n", err.c_str()); return 1; }
        const auto& ev = engine->expert().events(); std::printf("pre-labelled %zu events in the first %.1f s\n", ev.size(), double(to) / rate);
        if (!events_out.empty()) { std::ofstream f(events_out); if (!f) { std::fprintf(stderr, "cannot write %s\n", events_out.c_str()); return 1; } f << "onset_s,offset_s,class,amplitude_deg,peak_speed_deg_s,onset_sample,offset_sample\n"; for (const auto& e : ev) f << e.t_onset << ',' << e.t_offset << ',' << int(e.cls) << ',' << e.amplitude_deg << ',' << e.peak_speed_deg_s << ',' << e.onset << ',' << e.offset << '\n'; }
        if (train) {
            engine->expert().mark_reviewed(0, to, true); TrainConfig tc; tc.n_classes = 3; tc.max_epochs = 200; if (!engine->start_training(tc, &err)) { std::fprintf(stderr, "training failed to start: %s\n", err.c_str()); return 1; }
            TrainProgress p; do { std::this_thread::sleep_for(std::chrono::milliseconds(200)); p = engine->training_progress(); if (g_stop) { engine->cancel_training(); break; } } while (!p.finished);
            std::printf("training: %d epochs, %.1f s, validation event F1 %.3f, kappa %.3f (%s)\n", p.epoch, p.elapsed_s, p.val_event_f1, p.val_kappa, p.message.c_str()); if (p.failed) return 1;
            if (!engine->deploy_trained_model("cli", &err)) { std::fprintf(stderr, "deploy failed: %s\n", err.c_str()); return 1; } std::printf("deployed as detector user:cli\n"); }
    }
    return 0;
}
