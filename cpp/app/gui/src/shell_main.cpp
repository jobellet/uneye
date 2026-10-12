// EyeLive window (task T12): GLFW + OpenGL 3.2 core + Dear ImGui, a tab bar built from tab_registry(), a status bar, render on demand, and an offscreen --shot mode.
//   eyelive [--mock] [--demo] [--tab <title>] [--size WxH] [--shot file.ppm [--frames N]] [--settings file] [--run-seconds S]
//   --script "5:m:400:300,6:d,7:u,9:k:Delete" (with --shot): simulated input at given frames: m:x:y mouse move (window points), d / u mouse down / up, k:Name key press (Delete, Backspace, A, Z, Escape, S, M, R, 1-4)
//   --mock: synthetic engine; --demo: real engine on a simulated session (60 s recorded as fast as possible, pre-labelled); --run-seconds: exit after S seconds (idle CPU measurement)
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <memory>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "eyelive/gui/mock_engine.hpp"
#include "eyelive/gui/settings.hpp"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"
#include "eyelive/net_detectors.hpp"

using namespace eyelive;
using Clock = std::chrono::steady_clock;

struct Args { bool mock = false, demo = false; std::string shot, tab, settings, script; int frames = 5, w = 1360, h = 820; double run_seconds = 0; };
static Args parse(int argc, char** argv) {
    Args a; for (int i = 1; i < argc; ++i) { const std::string s = argv[i]; auto next = [&](std::string& out) { if (i + 1 < argc) out = argv[++i]; };
        if (s == "--mock") a.mock = true; else if (s == "--demo") a.demo = true; else if (s == "--script") next(a.script); else if (s == "--run-seconds" && i + 1 < argc) a.run_seconds = std::atof(argv[++i]); else if (s == "--shot") next(a.shot); else if (s == "--tab") next(a.tab); else if (s == "--settings") next(a.settings);
        else if (s == "--frames" && i + 1 < argc) a.frames = std::max(1, std::atoi(argv[++i]));
        else if (s == "--size" && i + 1 < argc) { std::sscanf(argv[++i], "%dx%d", &a.w, &a.h); } }
    return a;
}
static double cpu_seconds() {
#ifndef _WIN32
    rusage r; getrusage(RUSAGE_SELF, &r); return double(r.ru_utime.tv_sec) + r.ru_utime.tv_usec * 1e-6 + double(r.ru_stime.tv_sec) + r.ru_stime.tv_usec * 1e-6;
#else
    return 0.0;
#endif
}
static bool write_ppm(const std::string& path, int w, int h) {
    std::vector<unsigned char> px(size_t(w) * h * 3); glPixelStorei(GL_PACK_ALIGNMENT, 1); glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    std::ofstream f(path, std::ios::binary); if (!f) return false; f << "P6\n" << w << " " << h << "\n255\n"; for (int y = h - 1; y >= 0; --y) f.write(reinterpret_cast<const char*>(&px[size_t(y) * w * 3]), std::streamsize(w) * 3); return bool(f);
}

struct ScriptStep { int frame; std::string kind; float x = 0, y = 0; std::string key; };
static std::vector<ScriptStep> parse_script(const std::string& text) {
    std::vector<ScriptStep> v; size_t pos = 0; while (pos < text.size()) { size_t e = text.find(',', pos); if (e == std::string::npos) e = text.size(); std::string tok = text.substr(pos, e - pos); pos = e + 1; ScriptStep st; size_t c = tok.find(':'); if (c == std::string::npos) continue; st.frame = std::atoi(tok.substr(0, c).c_str()); std::string rest = tok.substr(c + 1);
        if (rest[0] == 'm') { st.kind = "m"; std::sscanf(rest.c_str(), "m:%f:%f", &st.x, &st.y); } else if (rest[0] == 'd') st.kind = "d"; else if (rest[0] == 'u') st.kind = "u"; else if (rest.rfind("k:", 0) == 0) { st.kind = "k"; st.key = rest.substr(2); } else continue; v.push_back(st); }
    return v;
}
static ImGuiKey key_from_name(const std::string& k) {
    if (k == "Delete") return ImGuiKey_Delete; if (k == "Backspace") return ImGuiKey_Backspace; if (k == "Escape") return ImGuiKey_Escape; if (k.size() == 1 && k[0] >= 'A' && k[0] <= 'Z') return ImGuiKey(ImGuiKey_A + (k[0] - 'A')); if (k.size() == 1 && k[0] >= '0' && k[0] <= '9') return ImGuiKey(ImGuiKey_0 + (k[0] - '0')); return ImGuiKey_None;
}
static void inject(const std::vector<ScriptStep>& script, int frame, ImGuiIO& io) {
    for (const auto& st : script) { if (st.frame != frame) continue; if (st.kind == "m") io.AddMousePosEvent(st.x, st.y); else if (st.kind == "d") io.AddMouseButtonEvent(0, true); else if (st.kind == "u") io.AddMouseButtonEvent(0, false); else if (st.kind == "k") { const ImGuiKey k = key_from_name(st.key); io.AddKeyEvent(k, true); } }
    for (const auto& st : script) if (st.frame == frame - 1 && st.kind == "k") io.AddKeyEvent(key_from_name(st.key), false);       // release the key one frame later
}

static void draw_status(IEngine& engine, const ImVec2& display) {
    const float h = ImGui::GetFrameHeight() + 8; ImGui::SetNextWindowPos(ImVec2(0, display.y - h)); ImGui::SetNextWindowSize(ImVec2(display.x, h));
    ImGui::Begin("##status", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
    const EngineStats s = engine.stats(); ImGui::TextColored(s.running ? ImVec4(0.3f, 0.9f, 0.4f, 1) : ImVec4(0.7f, 0.7f, 0.7f, 1), s.running ? "REC" : "idle"); ImGui::SameLine();
    ImGui::Text("| rate %.1f Hz | %llu samples | %s / %s | mem %s | cpu %.1f %% | latency p50 %.1f ms", s.rate_hz_measured, (unsigned long long)s.samples, gui::format_time(s.recorded_s).c_str(), gui::format_time(s.capacity_s).c_str(), gui::format_bytes(double(s.bytes_allocated)).c_str(), s.cpu_percent, s.detect_latency_ms_p50);
    if (s.dropped_packets) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1), "| %llu dropped", (unsigned long long)s.dropped_packets); }
    if (!s.warning.empty()) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "| %s", s.warning.c_str()); }
    ImGui::End();
}

int main(int argc, char** argv) {
    const Args args = parse(argc, argv); if (!args.settings.empty()) gui::Settings::instance().set_path(args.settings);
    { namespace fs = std::filesystem; std::error_code ec; fs::path d = fs::absolute(argv[0], ec).parent_path(); for (int up = 0; up < 7 && !d.empty(); ++up, d = d.parent_path()) { for (const char* rel : {"models", "cpp/models"}) if (fs::exists(d / rel / "combined.bin", ec)) { set_models_dir((d / rel).string()); up = 99; break; } } }      // models next to the executable or in a repository checkout
    std::unique_ptr<IEngine> engine; if (args.mock) engine = std::make_unique<gui::MockEngine>(); else engine = make_engine();
    std::vector<std::unique_ptr<ITab>> tabs; for (auto& f : tab_registry()) tabs.push_back(f()); std::stable_sort(tabs.begin(), tabs.end(), [](const std::unique_ptr<ITab>& a, const std::unique_ptr<ITab>& b) { return a->order() < b->order(); });
    if (args.demo && !args.mock) {
        SessionConfig c; c.source.kind = "sim"; c.source.speed = 0; c.source.rate_hz = 1000; c.session_path = (std::filesystem::temp_directory_path() / ("eyelive_demo_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".eyelive")).string(); c.duration_s = 70; c.deg_per_unit = 1.0; c.detector_id = "hmm_accel"; std::string err;
        if (!engine->start(c, &err)) { std::fprintf(stderr, "demo: %s\n", err.c_str()); return 5; }
        for (int i = 0; i < 400 && engine->stats().samples < 60000; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(25)); std::this_thread::sleep_for(std::chrono::milliseconds(600)); engine->stop(); engine->prelabel("hmm_accel", 0, engine->trace().size(), &err);
    }
    if (!glfwInit()) { std::fprintf(stderr, "cannot initialise GLFW\n"); return 2; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2); glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); if (!args.shot.empty()) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(args.w, args.h, "EyeLive", nullptr, nullptr); if (!win) { std::fprintf(stderr, "cannot create a window (no display?)\n"); glfwTerminate(); return 3; }
    glfwMakeContextCurrent(win); glfwSwapInterval(args.shot.empty() ? 1 : 0);
    IMGUI_CHECKVERSION(); ImGui::CreateContext(); ImGuiIO& io = ImGui::GetIO(); io.IniFilename = nullptr; io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; ImGui::StyleColorsDark();
    float xs = 1, ys = 1; glfwGetWindowContentScale(win, &xs, &ys); int ww = 0, wh = 0, fbw = 0, fbh = 0; glfwGetWindowSize(win, &ww, &wh); glfwGetFramebufferSize(win, &fbw, &fbh); const bool retina = ww > 0 && fbw > ww;      // macOS: window units are points and ImGui rasterises at the framebuffer scale itself
    const float ui = retina ? 1.0f : std::max(1.0f, xs); ImGuiStyle& style = ImGui::GetStyle(); style.FontSizeBase = 15.0f; style.FontScaleDpi = ui; style.ScaleAllSizes(ui); style.WindowRounding = 0; style.FrameRounding = 3; style.TabRounding = 3; style.ItemSpacing = ImVec2(8, 6);
    ImGui_ImplGlfw_InitForOpenGL(win, true); ImGui_ImplOpenGL3_Init("#version 150");
    int select_tab = -1; for (size_t i = 0; i < tabs.size(); ++i) if (!args.tab.empty() && args.tab == tabs[i]->title()) select_tab = int(i);
    const std::vector<ScriptStep> script = parse_script(args.script); io.AddFocusEvent(true);
    int frames_drawn = 0, extra_frames = 3, active = 0; auto last_render = Clock::now(); const double cpu0 = cpu_seconds(); const auto wall0 = Clock::now(); const int max_frames = args.shot.empty() ? -1 : args.frames; int frame_counter = 0;
    while (!glfwWindowShouldClose(win)) {
        if (args.run_seconds > 0 && std::chrono::duration<double>(Clock::now() - wall0).count() >= args.run_seconds) break;
        gui::Settings::instance().tick();
        const bool wants_live = !tabs.empty() && active >= 0 && active < int(tabs.size()) && tabs[size_t(active)]->wants_redraw(); double since = std::chrono::duration<double>(Clock::now() - last_render).count();
        // pacing: live = one frame per 1/60 s counted from the previous one (the swap already waits for the display refresh: waiting a full 1/60 s again halved the frame rate); idle = sleep up to 1 s, woken by any input
        const double timeout = wants_live ? std::max(0.0, 1.0 / 60.0 - since) : 1.0; bool event = false;
        if (args.shot.empty()) { const auto t0 = Clock::now(); if (timeout > 0) glfwWaitEventsTimeout(timeout); else glfwPollEvents(); event = timeout > 0.006 && std::chrono::duration<double>(Clock::now() - t0).count() < timeout - 0.004; } else glfwPollEvents();
        since = std::chrono::duration<double>(Clock::now() - last_render).count();
        const bool render = !args.shot.empty() || event || extra_frames > 0 || wants_live || since >= 1.0; if (!render) continue;
        ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); inject(script, frame_counter, io); ImGui::NewFrame(); const ImVec2 disp = io.DisplaySize; const float status_h = ImGui::GetFrameHeight() + 8;
        ImGui::SetNextWindowPos(ImVec2(0, 0)); ImGui::SetNextWindowSize(ImVec2(disp.x, disp.y - status_h));
        ImGui::Begin("EyeLive", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
        if (tabs.empty()) ImGui::TextUnformatted("no tabs"); else if (ImGui::BeginTabBar("tabs")) {
            for (size_t i = 0; i < tabs.size(); ++i) { ImGuiTabItemFlags fl = 0; if (select_tab == int(i)) { fl |= ImGuiTabItemFlags_SetSelected; if (frames_drawn >= 1) select_tab = -1; }
                if (ImGui::BeginTabItem(tabs[i]->title(), nullptr, fl)) { active = int(i); if (ImGui::BeginChild("body", ImVec2(0, 0), ImGuiChildFlags_None)) tabs[i]->draw(*engine); ImGui::EndChild(); ImGui::EndTabItem(); } }
            ImGui::EndTabBar(); }
        ImGui::End(); draw_status(*engine, disp); ImGui::Render();
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh); glViewport(0, 0, fw, fh); glClearColor(0.09f, 0.09f, 0.10f, 1); glClear(GL_COLOR_BUFFER_BIT); ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        ++frames_drawn; ++frame_counter; last_render = Clock::now(); if (extra_frames > 0) --extra_frames; if (ImGui::IsAnyItemActive() || event) extra_frames = std::max(extra_frames, 2);        // one more frame after any interaction: widgets show their new state a frame late
        if (!args.shot.empty() && frame_counter >= max_frames) { if (!write_ppm(args.shot, fw, fh)) { std::fprintf(stderr, "cannot write %s\n", args.shot.c_str()); return 4; } break; }
        if (!args.shot.empty() && !args.script.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(5));       // scripted runs advance in wall-clock time (training, delays)
        glfwSwapBuffers(win);
    }
    const double wall = std::chrono::duration<double>(Clock::now() - wall0).count(), cpu = cpu_seconds() - cpu0; std::printf("frames %d, wall %.2f s, cpu %.2f s (%.1f %% of one core)\n", frames_drawn, wall, cpu, wall > 0 ? 100.0 * cpu / wall : 0.0);
    gui::Settings::instance().flush(); engine.reset(); ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown(); ImGui::DestroyContext(); glfwDestroyWindow(win); glfwTerminate(); return 0;
}
