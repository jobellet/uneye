#include "mmap_posix.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

namespace eyelive {
namespace recorder {

MmapResult map_file(const std::string& path, uint64_t size, bool create, bool reserve, bool lock, std::string* error) {
    MmapResult res;
    int flags = create ? (O_RDWR | O_CREAT | O_TRUNC) : O_RDONLY;
    res.fd = ::open(path.c_str(), flags, 0666);
    if (res.fd < 0) {
        if (error) *error = "Failed to open file: " + std::string(std::strerror(errno));
        return res;
    }

    if (create) {
        if (reserve) {
#ifdef __APPLE__
            fstore_t store = {F_ALLOCATECONTIG, F_PEOFPOSMODE, 0, (off_t)size, 0};
            if (fcntl(res.fd, F_PREALLOCATE, &store) < 0) {
                store.fst_flags = F_ALLOCATEALL;
                fcntl(res.fd, F_PREALLOCATE, &store);
            }
            ftruncate(res.fd, size);
#else
            if (posix_fallocate(res.fd, 0, size) != 0) {
                if (error) *error = "Failed to reserve disk space";
                ::close(res.fd);
                res.fd = -1;
                return res;
            }
#endif
        } else {
            if (ftruncate(res.fd, size) < 0) {
                if (error) *error = "Failed to resize file";
                ::close(res.fd);
                res.fd = -1;
                return res;
            }
        }
    } else {
        struct stat st;
        if (fstat(res.fd, &st) < 0) {
            if (error) *error = "Failed to stat file";
            ::close(res.fd);
            res.fd = -1;
            return res;
        }
        res.size = st.st_size;
        size = res.size;
    }

    int prot = create ? (PROT_READ | PROT_WRITE) : PROT_READ;
    res.ptr = mmap(nullptr, size, prot, MAP_SHARED, res.fd, 0);
    if (res.ptr == MAP_FAILED) {
        if (error) *error = "Failed to mmap file: " + std::string(std::strerror(errno));
        ::close(res.fd);
        res.fd = -1;
        res.ptr = nullptr;
        return res;
    }

    if (create && lock) {
        mlock(res.ptr, size); // Best effort, ignore errors
    }

    res.size = size;
    res.success = true;
    return res;
}

void unmap_file(MmapResult& map, bool sync) {
    if (map.ptr && map.ptr != MAP_FAILED) {
        if (sync) {
            msync(map.ptr, map.size, MS_SYNC);
        }
        munmap(map.ptr, map.size);
        map.ptr = nullptr;
    }
    if (map.fd >= 0) {
        ::close(map.fd);
        map.fd = -1;
    }
    map.success = false;
}

} // namespace recorder
} // namespace eyelive
