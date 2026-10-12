// GUI Shell Main Entry Point
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/resource.h>
#include <sys/time.h>
#include <signal.h>

#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "eyelive/engine.hpp"
#include "eyelive/gui_tab.hpp"
#include "eyelive/gui/mock_engine.hpp"

#if __has_include("eyelive/engine_impl.hpp") || defined(HAS_REAL_ENGINE)
// Wait, engine_impl.hpp doesn't exist, we just rely on `eyelive::make_engine` being weakly linked if needed
#else
namespace eyelive {
    __attribute__((weak)) std::unique_ptr<IEngine> make_engine() {
        return nullptr;
    }
}
#endif

struct ShellConfig {
    bool mock = false;
    std::string shot_file;
    int shot_frames = 1;
    std::string tab;
};

ShellConfig parse_args(int argc, char** argv) {
    ShellConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mock") {
            cfg.mock = true;
        } else if (arg == "--shot" && i + 1 < argc) {
            cfg.shot_file = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            cfg.shot_frames = std::atoi(argv[++i]);
        } else if (arg == "--tab" && i + 1 < argc) {
            cfg.tab = argv[++i];
        }
    }
    return cfg;
}

int main(int argc, char** argv) {
    ShellConfig cfg = parse_args(argc, argv);

    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    if (!cfg.shot_file.empty()) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }

    GLFWwindow* window = glfwCreateWindow(1024, 768, "EyeLive", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 2;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    std::unique_ptr<eyelive::IEngine> engine;
    if (!cfg.mock) {
        engine = eyelive::make_engine();
    }
    if (!engine) {
        engine = std::make_unique<eyelive::gui::MockEngine>();
    }

    std::vector<std::unique_ptr<eyelive::ITab>> tabs;
    for (auto& factory : eyelive::tab_registry()) {
        tabs.push_back(factory());
    }

    std::sort(tabs.begin(), tabs.end(), [](const std::unique_ptr<eyelive::ITab>& a, const std::unique_ptr<eyelive::ITab>& b) {
        return a->order() < b->order();
    });

    eyelive::ITab* active_tab = nullptr;
    int shot_frames = cfg.shot_frames;
    bool is_shot = !cfg.shot_file.empty();

    static volatile sig_atomic_t g_shutdown = 0;
    struct sigaction sa;
    sa.sa_handler = [](int) { g_shutdown = 1; };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    while (!glfwWindowShouldClose(window) && !g_shutdown) {
        if (g_shutdown) break;
        if (is_shot) {
            if (shot_frames <= 0) break;
            shot_frames--;
            glfwPollEvents();
        } else {
            bool wants_redraw = (active_tab != nullptr) && active_tab->wants_redraw();
            if (wants_redraw) {
                glfwWaitEventsTimeout(1.0 / 30.0);
            } else {
                glfwWaitEventsTimeout(1.0);
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("EyeLive", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus);

        if (tabs.empty()) {
            ImGui::Text("no tabs");
        } else {
            if (ImGui::BeginTabBar("Tabs")) {
                for (auto& tab : tabs) {
                    bool is_open = true;
                    if (ImGui::BeginTabItem(tab->title(), &is_open)) {
                        active_tab = tab.get();
                        tab->draw(*engine);
                        ImGui::EndTabItem();
                    }
                }
                ImGui::EndTabBar();
            }
        }

        // Status bar
        eyelive::EngineStats stats = engine->stats();
        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - ImGui::GetFrameHeightWithSpacing());
        ImGui::Separator();
        ImGui::Text("Rate: %.1f Hz | Memory: %llu B | Mode: %s | Warning: %s",
                    stats.rate_hz_measured,
                    (unsigned long long)stats.bytes_allocated,
                    cfg.mock ? "Mock" : "Real",
                    stats.warning.empty() ? "None" : stats.warning.c_str());

        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (is_shot) {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        std::vector<uint8_t> pixels(width * height * 3);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        FILE* fp = std::fopen(cfg.shot_file.c_str(), "wb");
        if (fp) {
            std::fprintf(fp, "P6\n%d %d\n255\n", width, height);
            // OpenGL reads bottom-to-top, PPM expects top-to-bottom
            for (int y = height - 1; y >= 0; --y) {
                std::fwrite(pixels.data() + y * width * 3, 1, width * 3, fp);
            }
            std::fclose(fp);
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        double utime = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6;
        double stime = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;
        std::printf("CPU time: %.3f s user, %.3f s sys\n", utime, stime);
    }

    return 0;
}
