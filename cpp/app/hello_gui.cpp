// Toolchain check: one ImGui window through GLFW + OpenGL 3 (no application logic).
#include <cstdio>
#include <cstdlib>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
int main(int argc, char** argv) {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2); glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    GLFWwindow* w = glfwCreateWindow(900, 520, "EyeLive toolchain check", nullptr, nullptr); if (!w) { std::printf("no window\n"); glfwTerminate(); return 2; }
    glfwMakeContextCurrent(w); glfwSwapInterval(1);
    IMGUI_CHECKVERSION(); ImGui::CreateContext(); ImGui_ImplGlfw_InitForOpenGL(w, true); ImGui_ImplOpenGL3_Init("#version 150");
    int frames = argc > 1 ? std::atoi(argv[1]) : 120;
    for (int f = 0; f < frames && !glfwWindowShouldClose(w); ++f) {
        glfwPollEvents(); ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
        ImGui::Begin("EyeLive"); ImGui::Text("ImGui %s + GLFW + OpenGL3 via zig c++", IMGUI_VERSION); ImGui::Text("frame %d", f); ImGui::End();
        ImGui::Render(); int dw, dh; glfwGetFramebufferSize(w, &dw, &dh); glViewport(0, 0, dw, dh); glClearColor(0.1f, 0.1f, 0.12f, 1); glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); glfwSwapBuffers(w);
    }
    ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown(); ImGui::DestroyContext(); glfwDestroyWindow(w); glfwTerminate(); std::printf("ok\n"); return 0;
}
