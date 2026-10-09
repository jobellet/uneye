cmake_minimum_required(VERSION 3.14)
project(uneye_rt CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

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

add_library(uneye_rt STATIC src/native_engine.cpp src/causal_engine.cpp src/gaze_engine.cpp src/onnx_engine.cpp src/detector.cpp src/safety_guard.cpp)
target_include_directories(uneye_rt PUBLIC include)
set_source_files_properties(src/gaze_engine.cpp src/safety_guard.cpp PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Werror")   # the safety-relevant file must compile without any warning
if(ONNXRUNTIME_ROOT)
  target_compile_definitions(uneye_rt PUBLIC UNEYE_WITH_ONNX)
  target_include_directories(uneye_rt PUBLIC ${ONNXRUNTIME_ROOT}/include)
  find_library(ORT_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib REQUIRED NO_DEFAULT_PATH)
  target_link_libraries(uneye_rt PUBLIC ${ORT_LIB})
  set(CMAKE_BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib)
endif()

add_executable(uneye_rt_cli tools/uneye_rt.cpp)
target_link_libraries(uneye_rt_cli uneye_rt)
set_target_properties(uneye_rt_cli PROPERTIES OUTPUT_NAME uneye_rt)

add_executable(gaze_replay tools/gaze_replay.cpp)
target_link_libraries(gaze_replay uneye_rt)
target_compile_options(gaze_replay PRIVATE -Wall -Wextra)

enable_testing()
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
