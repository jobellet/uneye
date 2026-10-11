#include "os_mmap.hpp"
#include <cerrno>
#include <cstring>
#include <vector>

#ifdef _WIN32
// Windows branch: written against the Win32 documentation, NOT built or run in CI yet (see docs/tasks/status/T01.md).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
namespace eyelive { namespace os {
static std::string winerr(const char* what) { return std::string(what) + " failed (Win32 error " + std::to_string(GetLastError()) + ")"; }
static std::wstring widen(const std::string& s) { if (s.empty()) return std::wstring(); int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0); std::wstring w(size_t(n), L'\0'); MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n); if (!w.empty()) w.pop_back(); return w; }
bool create_map(const std::string& path, uint64_t bytes, bool reserve_disk, bool lock_memory, MappedFile& m, bool* locked, std::string* error) {
    if (locked) *locked = false;
    HANDLE f = CreateFileW(widen(path).c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);   // CREATE_NEW: fails if the file exists
    if (f == INVALID_HANDLE_VALUE) { if (error) *error = winerr("CreateFile"); return false; }
    LARGE_INTEGER sz; sz.QuadPart = LONGLONG(bytes);
    if (!SetFilePointerEx(f, sz, nullptr, FILE_BEGIN) || !SetEndOfFile(f)) { if (error) *error = winerr("SetEndOfFile (not enough space?)"); CloseHandle(f); DeleteFileW(widen(path).c_str()); return false; }
    if (reserve_disk) {                       // force the allocation of the blocks: write zeros over the whole file (SetEndOfFile only reserves the length on some file systems)
        LARGE_INTEGER z; z.QuadPart = 0; SetFilePointerEx(f, z, nullptr, FILE_BEGIN); std::vector<char> zeros(1 << 20, 0); uint64_t left = bytes;
        while (left) { DWORD n = DWORD(left < zeros.size() ? left : zeros.size()), w = 0; if (!WriteFile(f, zeros.data(), n, &w, nullptr) || w != n) { if (error) *error = winerr("WriteFile (disk full?)"); CloseHandle(f); DeleteFileW(widen(path).c_str()); return false; } left -= n; }
        SetFilePointerEx(f, z, nullptr, FILE_BEGIN);
    }
    HANDLE mp = CreateFileMappingW(f, nullptr, PAGE_READWRITE, DWORD(bytes >> 32), DWORD(bytes & 0xFFFFFFFFu), nullptr);
    if (!mp) { if (error) *error = winerr("CreateFileMapping"); CloseHandle(f); return false; }
    void* p = MapViewOfFile(mp, FILE_MAP_ALL_ACCESS, 0, 0, size_t(bytes));
    if (!p) { if (error) *error = winerr("MapViewOfFile"); CloseHandle(mp); CloseHandle(f); return false; }
    if (lock_memory) { SIZE_T mn, mx; if (GetProcessWorkingSetSize(GetCurrentProcess(), &mn, &mx)) SetProcessWorkingSetSize(GetCurrentProcess(), mn + size_t(bytes), mx + size_t(bytes)); if (locked) *locked = VirtualLock(p, size_t(bytes)) != 0; }
    m.base = p; m.size = bytes; m.writable = true; m.handle = reinterpret_cast<intptr_t>(f); m.mapping = reinterpret_cast<intptr_t>(mp); return true;
}
bool open_map_readonly(const std::string& path, MappedFile& m, std::string* error) {
    HANDLE f = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { if (error) *error = winerr("CreateFile"); return false; }
    LARGE_INTEGER sz; if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0) { if (error) *error = "empty or unreadable file"; CloseHandle(f); return false; }
    HANDLE mp = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr); if (!mp) { if (error) *error = winerr("CreateFileMapping"); CloseHandle(f); return false; }
    void* p = MapViewOfFile(mp, FILE_MAP_READ, 0, 0, 0); if (!p) { if (error) *error = winerr("MapViewOfFile"); CloseHandle(mp); CloseHandle(f); return false; }
    m.base = p; m.size = uint64_t(sz.QuadPart); m.writable = false; m.handle = reinterpret_cast<intptr_t>(f); m.mapping = reinterpret_cast<intptr_t>(mp); return true;
}
bool sync(MappedFile& m, std::string* error) {
    if (!m.base || !m.writable) return true;
    if (!FlushViewOfFile(m.base, 0) || !FlushFileBuffers(reinterpret_cast<HANDLE>(m.handle))) { if (error) *error = winerr("flush"); return false; } return true;
}
void unmap(MappedFile& m) {
    if (m.base) UnmapViewOfFile(m.base); if (m.mapping) CloseHandle(reinterpret_cast<HANDLE>(m.mapping)); if (m.handle != -1) CloseHandle(reinterpret_cast<HANDLE>(m.handle));
    m = MappedFile();
}
}}  // namespace
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace eyelive { namespace os {
static std::string sys(const char* what) { return std::string(what) + ": " + std::strerror(errno); }

bool create_map(const std::string& path, uint64_t bytes, bool reserve_disk, bool lock_memory, MappedFile& m, bool* locked, std::string* error) {
    if (locked) *locked = false;
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);        // O_EXCL: never overwrite an existing session
    if (fd < 0) { if (error) *error = sys(("cannot create " + path).c_str()); return false; }
    auto fail = [&](const std::string& msg) { if (error) *error = msg; ::close(fd); ::unlink(path.c_str()); return false; };
    if (reserve_disk) {
#if defined(__APPLE__)
        fstore_t st = {F_ALLOCATECONTIG, F_PEOFPOSMODE, 0, off_t(bytes), 0};
        int r = ::fcntl(fd, F_PREALLOCATE, &st);
        if (r == -1) { st.fst_flags = F_ALLOCATEALL; r = ::fcntl(fd, F_PREALLOCATE, &st); }      // not contiguous: any free blocks
        if (r == -1) return fail(sys("cannot reserve the disk space (F_PREALLOCATE)"));
        if (::ftruncate(fd, off_t(bytes)) != 0) return fail(sys("ftruncate"));
#else
        const int r = ::posix_fallocate(fd, 0, off_t(bytes));
        if (r != 0) return fail(std::string("cannot reserve the disk space (posix_fallocate): ") + std::strerror(r));
#endif
    } else if (::ftruncate(fd, off_t(bytes)) != 0) return fail(sys("ftruncate"));
    void* p = ::mmap(nullptr, size_t(bytes), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) return fail(sys("mmap"));
    if (lock_memory && locked) *locked = ::mlock(p, size_t(bytes)) == 0;                    // best effort (RLIMIT_MEMLOCK is small by default)
    m.base = p; m.size = bytes; m.writable = true; m.handle = fd; m.mapping = 0; return true;
}
bool open_map_readonly(const std::string& path, MappedFile& m, std::string* error) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { if (error) *error = sys(("cannot open " + path).c_str()); return false; }
    struct stat sb; if (::fstat(fd, &sb) != 0 || sb.st_size <= 0) { if (error) *error = "empty or unreadable file"; ::close(fd); return false; }
    void* p = ::mmap(nullptr, size_t(sb.st_size), PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { if (error) *error = sys("mmap"); ::close(fd); return false; }
    m.base = p; m.size = uint64_t(sb.st_size); m.writable = false; m.handle = fd; m.mapping = 0; return true;
}
bool sync(MappedFile& m, std::string* error) {
    if (!m.base || !m.writable) return true;
    if (::msync(m.base, size_t(m.size), MS_SYNC) != 0) { if (error) *error = sys("msync"); return false; }
    return true;
}
void unmap(MappedFile& m) {
    if (m.base) ::munmap(m.base, size_t(m.size));
    if (m.handle >= 0) ::close(int(m.handle));
    m = MappedFile();
}
}}  // namespace
#endif
