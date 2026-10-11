// Platform layer of the recorder (private): create / open / map / sync / unmap a file. POSIX (Linux, macOS) and Windows behind one small interface; errors are returned as text, never thrown.
#pragma once
#include <cstdint>
#include <string>

namespace eyelive { namespace os {

struct MappedFile {
    void* base = nullptr; uint64_t size = 0; bool writable = false;
    intptr_t handle = -1, mapping = 0;          // fd (POSIX) or HANDLE of the file / the file mapping (Windows)
};

// creates a NEW file of `bytes` (fails if it exists), optionally really allocates the disk blocks, maps it read-write, optionally locks the pages in RAM (best effort; *locked tells)
bool create_map(const std::string& path, uint64_t bytes, bool reserve_disk, bool lock_memory, MappedFile& m, bool* locked, std::string* error);
bool open_map_readonly(const std::string& path, MappedFile& m, std::string* error);
bool sync(MappedFile& m, std::string* error);        // msync / FlushViewOfFile + FlushFileBuffers
void unmap(MappedFile& m);                           // idempotent

}}  // namespace eyelive::os
