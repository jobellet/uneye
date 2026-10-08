#include <stdexcept>

#include "uneye/engine.hpp"

#ifdef UNEYE_WITH_ONNX
#include <onnxruntime_cxx_api.h>

namespace uneye {
namespace {
class OnnxEngine : public Engine {
public:
    OnnxEngine(const std::string& path, int window, int classes)
        : T_(window), C_(classes), env_(ORT_LOGGING_LEVEL_WARNING, "uneye"), mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(1);  // lowest latency for tiny model
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), so);
    }
    int window() const override { return T_; }
    int classes() const override { return C_; }
    void infer(const float* dxy, float* prob) override {
        const int64_t ish[4] = {1, 1, T_, 2};
        const int64_t osh[3] = {1, C_, T_};
        Ort::Value in = Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(dxy), size_t(T_) * 2, ish, 4);
        Ort::Value out = Ort::Value::CreateTensor<float>(mem_, prob, size_t(C_) * T_, osh, 3);
        const char* in_names[] = {"dxy"};
        const char* out_names[] = {"prob"};
        session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, &out, 1);
    }
private:
    int T_, C_;
    Ort::Env env_;
    Ort::MemoryInfo mem_;
    std::unique_ptr<Ort::Session> session_;
};
}  // namespace
std::unique_ptr<Engine> make_onnx_engine(const std::string& p, int w, int c) { return std::make_unique<OnnxEngine>(p, w, c); }
bool onnx_available() { return true; }
}  // namespace uneye
#else
namespace uneye {
std::unique_ptr<Engine> make_onnx_engine(const std::string&, int, int) {
    throw std::runtime_error("built without ONNX Runtime (configure with -DONNXRUNTIME_ROOT=...)");
}
bool onnx_available() { return false; }
}  // namespace uneye
#endif
