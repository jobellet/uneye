# Top-level project for the Android build of the offline-labeling app modules.
# Wraps the frozen cpp/ tree (never edited) and adds the benchmark harness.
# Host build:   cmake -S android -B build-host -DCMAKE_BUILD_TYPE=Release && cmake --build build-host
# Android build: sh android/build_android.sh /path/to/ndk
cmake_minimum_required(VERSION 3.14)
project(uneye_android CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Threads REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp ${CMAKE_BINARY_DIR}/cpp)
add_subdirectory(bench)
