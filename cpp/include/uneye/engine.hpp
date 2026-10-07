// Inference back-ends for the U'n'Eye U-Net.
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace uneye {

// Runs the network on one fixed-length window.
//   dxy : window*2 floats, interleaved (dX0,dY0,dX1,dY1,...)  [eye velocity, deg/sample]
//   prob: classes*window floats, class-major (softmax output)
class Engine {
public:
    virtual ~Engine() = default;
    virtual int window() const = 0;
    virtual int classes() const = 0;
    virtual void infer(const float* dxy, float* prob) = 0;
};

// Dependency-free C++ re-implementation of the network (reads <model>.bin).
std::unique_ptr<Engine> make_native_engine(const std::string& bin_path, int window);

// ONNX Runtime back-end (reads <model>.onnx). Returns nullptr-throwing stub
// if the library was built without UNEYE_WITH_ONNX.
std::unique_ptr<Engine> make_onnx_engine(const std::string& onnx_path, int window, int classes);

bool onnx_available();

}  // namespace uneye
