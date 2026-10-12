cmake_minimum_required(VERSION 3.14)
project(uneye_rt CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(EYELIVE_PORTABLE "Release binaries that run on any CPU of the architecture (no -march=native): ON for distributed builds" OFF)
if(MSVC)
  add_compile_options(/utf-8 /bigobj /permissive- /D_USE_MATH_DEFINES /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /wd4244 /wd4267 /wd4305)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")     # static runtime: no Visual C++ redistributable to install
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
elseif(EYELIVE_PORTABLE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

option(ONNXRUNTIME_ROOT "Path to an unpacked onnxruntime release (enables the ONNX back-end)" "")

find_package(Threads REQUIRED)
option(UNEYE_SANITIZE "address | thread: build everything with that sanitizer (plus UBSan for address and libstdc++ assertions)" "")
if(UNEYE_SANITIZE STREQUAL "address")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -D_GLIBCXX_ASSERTIONS -g)
  add_link_options(-fsanitize=address,undefined)
elseif(UNEYE_SANITIZE STREQUAL "thread")
  add_compile_options(-fsanitize=thread -g)
  add_link_options(-fsanitize=thread)
endif()

add_library(uneye_rt STATIC src/native_engine.cpp src/causal_engine.cpp src/gaze_engine.cpp src/onnx_engine.cpp src/detector.cpp src/safety_guard.cpp src/bitcn.cpp)
target_include_directories(uneye_rt PUBLIC include)
if(NOT MSVC)
  set_source_files_properties(src/gaze_engine.cpp src/safety_guard.cpp PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Werror")
endif()   # the safety-relevant file must compile without any warning
if(ONNXRUNTIME_ROOT)
  target_compile_definitions(uneye_rt PUBLIC UNEYE_WITH_ONNX)
  target_include_directories(uneye_rt PUBLIC ${ONNXRUNTIME_ROOT}/include)
  find_library(ORT_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib REQUIRED NO_DEFAULT_PATH)
  target_link_libraries(uneye_rt PUBLIC ${ORT_LIB})
  set(CMAKE_BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib)
endif()

# Legacy command-line tools and tests of the real-time library (POSIX-only: malloc hooks, signals). Not built on Windows; the EyeLive application below is.
enable_testing()
if(NOT WIN32)
  add_executable(uneye_rt_cli tools/uneye_rt.cpp)
  target_link_libraries(uneye_rt_cli uneye_rt)
  set_target_properties(uneye_rt_cli PROPERTIES OUTPUT_NAME uneye_rt)

  add_executable(gaze_replay tools/gaze_replay.cpp)
  target_link_libraries(gaze_replay uneye_rt)
  target_compile_options(gaze_replay PRIVATE -Wall -Wextra)

  add_executable(test_parity tests/test_parity.cpp)
  target_link_libraries(test_parity uneye_rt)
  add_test(NAME parity COMMAND test_parity ${CMAKE_SOURCE_DIR})

  add_executable(test_zero_heap tests/test_zero_heap.cpp)
  target_link_libraries(test_zero_heap uneye_rt)
  target_compile_options(test_zero_heap PRIVATE -Wall -Wextra -Werror)
  if(UNEYE_SANITIZE)
    target_compile_definitions(test_zero_heap PRIVATE UNEYE_NO_MALLOC_HOOK)   # the sanitizer owns malloc; only operator new/delete are counted
  endif()
  add_test(NAME zero_heap COMMAND test_zero_heap ${CMAKE_SOURCE_DIR})

  add_executable(test_gaze tests/test_gaze_engine.cpp)
  target_link_libraries(test_gaze uneye_rt Threads::Threads)
  target_compile_options(test_gaze PRIVATE -Wall -Wextra -Werror)
  add_test(NAME gaze_engine COMMAND test_gaze ${CMAKE_SOURCE_DIR})

  add_executable(test_safety_guard tests/test_safety_guard.cpp)
  target_link_libraries(test_safety_guard uneye_rt)
  target_compile_options(test_safety_guard PRIVATE -Wall -Wextra -Werror)
  add_test(NAME safety_guard COMMAND test_safety_guard ${CMAKE_SOURCE_DIR})

  add_executable(fault_injection tools/fault_injection.cpp)     # Step 5: perturbations, network alone vs engine vs engine + guard
  target_link_libraries(fault_injection uneye_rt)
  target_compile_options(fault_injection PRIVATE -Wall -Wextra)

  add_executable(bench_latency tools/bench_latency.cpp)          # Step 3: per-sample latency of push() + guard (not a WCET)
  target_link_libraries(bench_latency uneye_rt)
  target_compile_options(bench_latency PRIVATE -Wall -Wextra)

  add_executable(causal_dump tools/causal_dump.cpp)              # Step 2: C++ causal TCN outputs for cpp/scripts/equivalence.py
  target_link_libraries(causal_dump uneye_rt)
  target_compile_options(causal_dump PRIVATE -Wall -Wextra)

  add_executable(test_equivalence tests/test_equivalence.cpp)    # Step 2: C++ vs PyTorch float64, documented tolerance
  target_link_libraries(test_equivalence uneye_rt)
  target_compile_options(test_equivalence PRIVATE -Wall -Wextra -Werror)
  add_test(NAME equivalence COMMAND test_equivalence ${CMAKE_SOURCE_DIR})

  add_executable(test_bitcn tests/test_bitcn.cpp)               # two-channel (vx, vy) bidirectional TCN vs PyTorch
  target_link_libraries(test_bitcn uneye_rt)
  target_compile_options(test_bitcn PRIVATE -Wall -Wextra -Werror)
  add_test(NAME bitcn COMMAND test_bitcn ${CMAKE_SOURCE_DIR})


endif()

# ---------------------------------------------------------------- EyeLive (cpp/app): see docs/EYELIVE_ROADMAP.md and docs/tasks/README.md
option(EYELIVE_BUILD_APP "Build the EyeLive modules and their tests (no GUI dependency)" ON)
if(EYELIVE_BUILD_APP)
  add_subdirectory(app)
endif()
