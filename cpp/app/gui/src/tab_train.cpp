// Train tab (order 50): train a U'n'Eye on the corrected data (default 50 s, rotation augmentation on), watch the curves, deploy the result as a detector.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include "imgui.h"
#include "eyelive/gui/widgets.hpp"
#include "eyelive/gui_tab.hpp"
#include "eyelive/net_detectors.hpp"

namespace eyelive {
namespace {
class TrainTab final : public ITab {
 public:
    const char* title() const override { return "Train"; }
    int order() const override { return 50; }
    bool wants_redraw() const override { return active_; }
    void draw(IEngine& engine) override {
        const TrainProgress p = engine.training_progress(); active_ = last_started_ && !p.finished && !p.failed && !p.cancelled; update_history(p);
        const double rate = engine.trace().info().rate_hz > 0 ? engine.trace().info().rate_hz : 1000.0; const double reviewed = engine.expert().reviewed_seconds(rate);
        ImGui::SeparatorText("Data"); ImGui::Text("reviewed (corrected) data: %.1f s", reviewed); ImGui::SameLine(); ImGui::ProgressBar(float(std::min(1.0, reviewed / std::max(1.0, double(seconds_)))), ImVec2(160, 0));
        if (reviewed < 20) ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "Correct and mark at least 50 s in the Review tab for a good model (10 s is the minimum that trains at all).");
        ImGui::SetNextItemWidth(120); ImGui::InputFloat("seconds of data to use (most recent first)", &seconds_, 10, 50, "%.0f"); ImGui::SetNextItemWidth(120); ImGui::SliderInt("classes (2-5; default 3 = fixation, saccade, blink)", &classes_, 2, 5);
        ImGui::SeparatorText("Augmentation"); ImGui::Checkbox("rotations of the gaze (x, y) trace (4 copies, recommended)", &rotation_); ImGui::SameLine(); ImGui::Checkbox("noise", &noise_); ImGui::SameLine(); ImGui::Checkbox("gain 0.7-1.4", &gain_);
        ImGui::SeparatorText("Start from"); if (ImGui::RadioButton("random weights", &init_kind_, 0)) init_path_[0] = 0; ImGui::SameLine(); if (ImGui::RadioButton("pretrained on datasets 1+2+3 (2 classes)", &init_kind_, 1)) set_init("combined.bin"); ImGui::SameLine(); if (ImGui::RadioButton("pretrained, Andersson (5 classes)", &init_kind_, 2)) set_init("andersson.bin"); ImGui::SameLine(); ImGui::RadioButton("model file", &init_kind_, 3);
        if (init_kind_ != 0) { ImGui::SetNextItemWidth(560); ImGui::InputText("model path", init_path_, sizeof init_path_); }
        ImGui::SeparatorText("Optimiser"); ImGui::SetNextItemWidth(100); ImGui::InputInt("max epochs", &epochs_); ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::InputInt("patience", &patience_); ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::InputFloat("learning rate", &lr_, 0, 0, "%.4f"); ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::InputInt("CPU threads (0 = auto)", &threads_);
        ImGui::TextDisabled("While a session is recording the trainer uses one core at low priority; after the session it uses all cores (about 5 s per epoch per core for 50 s of data, 15 to 60 epochs).");
        ImGui::Spacing(); if (!active_) { if (ImGui::Button("Train", ImVec2(140, 0))) { TrainConfig c; c.n_classes = classes_; c.max_seconds_of_data = double(seconds_); c.rotation_aug = rotation_; c.noise_aug = noise_; c.gain_aug = gain_; c.max_epochs = epochs_; c.patience = patience_; c.learning_rate = double(lr_); c.n_threads = threads_; if (init_kind_ != 0) c.init_model = init_path_; std::string err; msg_.clear(); hist_train_.clear(); hist_val_.clear(); hist_f1_.clear(); last_epoch_ = 0;
                if (engine.start_training(c, &err)) last_started_ = true; else msg_ = err; } } else { if (ImGui::Button("Cancel", ImVec2(140, 0))) engine.cancel_training(); }
        if (!msg_.empty()) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "%s", msg_.c_str()); }
        ImGui::SeparatorText("Progress"); ImGui::Text("epoch %d | train loss %.4f | validation loss %.4f | validation event F1 %.3f, kappa %.3f | %.0f s %s", p.epoch, p.train_loss, p.val_loss, p.val_event_f1, p.val_kappa, p.elapsed_s, p.message.c_str());
        gui::draw_curves("curves", ImVec2(std::max(200.0f, ImGui::GetContentRegionAvail().x - 8), 200), {hist_train_, hist_val_, hist_f1_}, {IM_COL32(90, 180, 255, 255), IM_COL32(255, 170, 70, 255), IM_COL32(120, 220, 140, 255)}, {"train loss", "val loss", "val F1"});
        if (p.failed) ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "training failed: %s", p.message.c_str());
        if (p.finished && !p.failed && !p.cancelled) { ImGui::SeparatorText("Use the model"); ImGui::SetNextItemWidth(220); ImGui::InputText("name", name_, sizeof name_); ImGui::SameLine(); if (ImGui::Button("Deploy as detector")) { std::string err; deploy_msg_ = engine.deploy_trained_model(name_, &err) ? std::string("deployed as detector user:") + name_ + " (Detection tab)" : err; } if (!deploy_msg_.empty()) ImGui::Text("%s", deploy_msg_.c_str()); }
    }
 private:
    void set_init(const char* file) { const std::string d = models_dir(); std::snprintf(init_path_, sizeof init_path_, "%s", ((d.empty() ? std::string("models") : d) + "/" + file).c_str()); }
    void update_history(const TrainProgress& p) { if (p.epoch > last_epoch_) { hist_train_.push_back(float(p.train_loss)); hist_val_.push_back(float(p.val_loss)); hist_f1_.push_back(float(p.val_event_f1)); last_epoch_ = p.epoch; } }
    bool active_ = false, last_started_ = false, rotation_ = true, noise_ = false, gain_ = false; float seconds_ = 50, lr_ = 0.001f; int classes_ = 3, init_kind_ = 0, epochs_ = 500, patience_ = 4, threads_ = 0, last_epoch_ = 0; char init_path_[512] = "", name_[64] = "my_model";
    std::string msg_, deploy_msg_; std::vector<float> hist_train_, hist_val_, hist_f1_;
};
static TabRegistrar reg([] { return std::make_unique<TrainTab>(); });
}  // namespace
}  // namespace eyelive
