#include "mmap_win32.hpp"

#ifdef _WIN32
#include <windows.h>

namespace eyelive {
namespace recorder {

MmapResult map_file(const std::string& path, uint64_t size, bool create, bool reserve, bool lock, std::string* error) {
    MmapResult res;

    DWORD access = create ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    DWORD disposition = create ? CREATE_ALWAYS : OPEN_EXISTING;

    HANDLE hFile = CreateFileA(path.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        if (error) *error = "Failed to open file";
        return res;
    }
    res.handle = hFile;

    if (create) {
        LARGE_INTEGER li;
        li.QuadPart = size;
        if (!SetFilePointerEx(hFile, li, nullptr, FILE_BEGIN) || !SetEndOfFile(hFile)) {
            if (error) *error = "Failed to set file size";
            CloseHandle(hFile);
            return res;
        }
        if (reserve) {
            // In Windows SetEndOfFile essentially reserves the space, but we can't easily zero it fast without writing.
            // The mapping will zero it lazily.
        }
    } else {
        LARGE_INTEGER li;
        if (!GetFileSizeEx(hFile, &li)) {
            if (error) *error = "Failed to get file size";
            CloseHandle(hFile);
            return res;
        }
        res.size = li.QuadPart;
        size = res.size;
    }

    DWORD protect = create ? PAGE_READWRITE : PAGE_READONLY;
    HANDLE hMap = CreateFileMappingA(hFile, nullptr, protect, (DWORD)(size >> 32), (DWORD)size, nullptr);
    if (!hMap) {
        if (error) *error = "Failed to create file mapping";
        CloseHandle(hFile);
        return res;
    }
    res.map_handle = hMap;

    DWORD map_access = create ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ;
    res.ptr = MapViewOfFile(hMap, map_access, 0, 0, size);
    if (!res.ptr) {
        if (error) *error = "Failed to map view of file";
        CloseHandle(hMap);
        CloseHandle(hFile);
        return res;
    }

    if (create && lock) {
        VirtualLock(res.ptr, size); // Best effort
    }

    res.size = size;
    res.success = true;
    return res;
}

void unmap_file(MmapResult& map, bool sync) {
    if (map.ptr) {
        if (sync) {
            FlushViewOfFile(map.ptr, map.size);
        }
        UnmapViewOfFile(map.ptr);
        map.ptr = nullptr;
    }
    if (map.map_handle) {
        CloseHandle((HANDLE)map.map_handle);
        map.map_handle = nullptr;
    }
    if (map.handle) {
        CloseHandle((HANDLE)map.handle);
        map.handle = nullptr;
    }
    map.success = false;
}

} // namespace recorder
} // namespace eyelive
#endif
