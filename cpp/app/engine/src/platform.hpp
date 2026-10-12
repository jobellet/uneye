// Small platform helpers of the engine (private): CPU time of the process, lower priority for the calling thread.
#pragma once
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
namespace eyelive { namespace plat {
inline double cpu_seconds() { FILETIME c, e, k, u; if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0; auto t = [](FILETIME f) { return double((uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime) * 1e-7; }; return t(k) + t(u); }
inline void lower_thread_priority() { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL); }
}}
#else
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <pthread.h>
#include <pthread/qos.h>
#endif
#ifdef __linux__
#include <sys/syscall.h>
#endif
namespace eyelive { namespace plat {
inline double cpu_seconds() { rusage r; getrusage(RUSAGE_SELF, &r); return double(r.ru_utime.tv_sec) + r.ru_utime.tv_usec * 1e-6 + double(r.ru_stime.tv_sec) + r.ru_stime.tv_usec * 1e-6; }
// the calling thread (and the threads it creates afterwards) runs at a lower priority: the background trainer must not disturb the acquisition
inline void lower_thread_priority() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#elif defined(__linux__)
    setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
#endif
}
}}
#endif
