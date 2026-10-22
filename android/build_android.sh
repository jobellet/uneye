#!/bin/sh
# Phase 0 build of the EyeLive modules + benchmark.
#   sh android/build_android.sh              -> host build (first measurement, no phone)
#   sh android/build_android.sh /path/to/ndk -> arm64-v8a build for the phone
# The cpp/ tree and its CMakeLists are frozen contracts: this script never
# edits them; android/CMakeLists.txt is the top-level project that wraps them.
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
if [ $# -ge 1 ]; then
  NDK=$1
  if [ ! -d "$NDK/toolchains/llvm/prebuilt" ]; then echo "not an NDK: $NDK" >&2; exit 2; fi
  ln -sfn "$NDK" android/build-android-ndk
  GEN="Unix Makefiles"; command -v ninja >/dev/null 2>&1 && GEN="Ninja"
  cmake -S android -B build-android -G "$GEN" \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/android/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24
  cmake --build build-android --target bench_offline -j "$(nproc)"
  echo "built: build-android/bench/bench_offline"
  echo "push with: adb push build-android/bench/bench_offline /data/local/tmp/ && adb shell chmod +x /data/local/tmp/bench_offline"
else
  cmake -S android -B build-host -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
  cmake --build build-host --target bench_offline -j "$(nproc)"
  echo "built: build-host/bench/bench_offline"
fi
