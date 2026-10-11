#pragma once

#ifdef _WIN32
#include "mmap_win32.hpp"
#else
#include "mmap_posix.hpp"
#endif
