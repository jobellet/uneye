// Fixed-capacity single-producer / single-consumer lock-free queue: the ONLY supported way to hand samples from a tracker (reader) thread
// to the thread that owns the GazeEngine. No allocation, no locks, wait-free. N must be a power of two.
#pragma once
#include <array>
#include <atomic>
#include <cstddef>

namespace uneye {

template <typename T, size_t N>
class SpscQueue {
    static_assert(N >= 2 && (N & (N - 1)) == 0, "capacity must be a power of two");
public:
    bool try_push(const T& v) noexcept {                    // producer thread only; false = full (the caller decides: drop or retry)
        const size_t t = tail_.load(std::memory_order_relaxed);
        if (t - head_.load(std::memory_order_acquire) == N) return false;
        buf_[t & (N - 1)] = v;
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }
    bool try_pop(T& v) noexcept {                           // consumer thread only; false = empty
        const size_t h = head_.load(std::memory_order_relaxed);
        if (h == tail_.load(std::memory_order_acquire)) return false;
        v = buf_[h & (N - 1)];
        head_.store(h + 1, std::memory_order_release);
        return true;
    }
private:
    std::array<T, N> buf_{};
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

}  // namespace uneye
