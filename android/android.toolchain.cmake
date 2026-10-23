# NDK toolchain file for the EyeLive Android modules (phase 0 of the offline
# labeling app). Use:
#   cmake -S cpp -B build-android -G Ninja \
#     -DCMAKE_TOOLCHAIN_FILE=android/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
#     -DEYELIVE_PORTABLE=ON -DEYELIVE_BUILD_GUI=OFF -DEYELIVE_BUILD_APP=ON
#   cmake --build build-android --target bench_offline
# GUI (Dear ImGui/GLFW) and the ONNX Runtime back-end stay OFF: the app uses the
# native engine only. EYELIVE_PORTABLE drops -march=native (an ARM build must
# not pass x86 flags).
set(ANDROID_ABI arm64-v8a CACHE STRING "")
set(ANDROID_PLATFORM android-24 CACHE STRING "")
set(ANDROID_STL c++_static CACHE STRING "")

if(DEFINED ENV{ANDROID_NDK_HOME} AND EXISTS "$ENV{ANDROID_NDK_HOME}")
  set(NDK_ROOT "$ENV{ANDROID_NDK_HOME}")
elseif(DEFINED ANDROID_NDK AND EXISTS "${ANDROID_NDK}")
  set(NDK_ROOT "${ANDROID_NDK}")
elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/build-android-ndk")
  get_filename_component(NDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/build-android-ndk" ABSOLUTE)
elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../build-android-ndk")
  get_filename_component(NDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../build-android-ndk" ABSOLUTE)
else()
  message(FATAL_ERROR "Set ANDROID_NDK_HOME to the unpacked NDK (or symlink android/build-android-ndk -> NDK).")
endif()

if(EXISTS "${NDK_ROOT}/build/cmake/android.toolchain.cmake")
  include("${NDK_ROOT}/build/cmake/android.toolchain.cmake")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
else()
  set(CMAKE_SYSTEM_NAME Android)
  set(CMAKE_SYSTEM_VERSION 24)
  set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
  set(CMAKE_ANDROID_STL_TYPE ${ANDROID_STL})

  file(GLOB PREBUILT_DIRS "${NDK_ROOT}/toolchains/llvm/prebuilt/*")
  if(PREBUILT_DIRS)
    list(GET PREBUILT_DIRS 0 PREBUILT_HOST_DIR)
    get_filename_component(HOST_TAG "${PREBUILT_HOST_DIR}" NAME)
  else()
    set(HOST_TAG "linux-x86_64")
  endif()

  set(TRIPLE aarch64-linux-android)
  set(CMAKE_C_COMPILER ${NDK_ROOT}/toolchains/llvm/prebuilt/${HOST_TAG}/bin/${TRIPLE}${CMAKE_SYSTEM_VERSION}-clang)
  set(CMAKE_CXX_COMPILER ${NDK_ROOT}/toolchains/llvm/prebuilt/${HOST_TAG}/bin/${TRIPLE}${CMAKE_SYSTEM_VERSION}-clang++)

  set(CMAKE_FIND_ROOT_PATH ${NDK_ROOT}/toolchains/llvm/prebuilt/${HOST_TAG}/sysroot)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

  # Portable release flags: no -march=native (set by cpp/CMakeLists.txt through
  # EYELIVE_PORTABLE), NEON is on by default for arm64-v8a.
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
endif()
