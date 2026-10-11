#pragma once
#include <cstdint>
#include <string>

namespace eyelive {
namespace recorder {

struct MmapResult {
    void* ptr = nullptr;
    uint64_t size = 0;
    void* handle = nullptr;
    void* map_handle = nullptr;
    bool success = false;
};

MmapResult map_file(const std::string& path, uint64_t size, bool create, bool reserve, bool lock, std::string* error);
void unmap_file(MmapResult& map, bool sync);

} // namespace recorder
} // namespace eyelive
