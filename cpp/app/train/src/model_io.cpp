#include "eyelive/train/model_io.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>

namespace eyelive { namespace train {

bool save_bin(const std::string& path, UNet<float>& net, std::string* error) {
    std::ofstream f(path, std::ios::binary); if (!f) { if (error) *error = "cannot write " + path; return false; }
    const int32_t hdr[4] = {1, net.K, net.ks, net.mp}; f.write("UNEY", 4); f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
    for (const auto& t : net.state()) f.write(reinterpret_cast<const char*>(t.data()), std::streamsize(t.size() * sizeof(float)));
    if (!f) { if (error) *error = "write failed: " + path; return false; } return true;
}

bool load_bin(const std::string& path, UNet<float>& net, std::string* error) {
    std::ifstream f(path, std::ios::binary); if (!f) { if (error) *error = "cannot open " + path; return false; }
    char magic[4]; int32_t hdr[4]; f.read(magic, 4); f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    if (!f || std::memcmp(magic, "UNEY", 4) != 0) { if (error) *error = "not a U'n'Eye model file: " + path; return false; }
    const int classes = hdr[1];
    if (hdr[2] != net.ks || hdr[3] != net.mp) { if (error) *error = "kernel / pooling size of the model differ"; return false; }
    UNet<float> src(classes, net.ks, net.mp); auto layout = src.state(); const bool keep_head = classes != net.K;
    for (auto& t : layout) { f.read(reinterpret_cast<char*>(t.data()), std::streamsize(t.size() * sizeof(float))); if (!f) { if (error) *error = "model file too short: " + path; return false; } }
    std::vector<std::vector<float>> mine = layout;
    if (keep_head) { auto own = net.state(); mine[mine.size() - 2] = own[own.size() - 2]; mine[mine.size() - 1] = own[own.size() - 1]; }
    if (!net.load_state(mine)) { if (error) *error = "model layout mismatch"; return false; }
    return true;
}

}}  // namespace eyelive::train
