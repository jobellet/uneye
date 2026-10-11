// Model files: the cpp/models/*.bin layout read by uneye::make_native_engine + a JSON sidecar.
#pragma once
#include <string>
#include "eyelive/train/unet.hpp"

namespace eyelive { namespace train {
bool save_bin(const std::string& path, UNet<float>& net, std::string* error);
// loads weights into net; when the class count of the file differs from net.K the output layer is left untouched (transfer to another class count)
bool load_bin(const std::string& path, UNet<float>& net, std::string* error);
}}
