#!/bin/sh
# Phase 0 build of the EyeLive modules + benchmark.
#   sh android/build_android.sh              -> host build (first measurement, no phone)
#   sh android/build_android.sh /path/to/ndk -> arm64-v8a build for the phone
# The cpp/ tree and its CMakeLists are frozen contracts: this script never
# edits them; android/CMakeLists.txt is the top-level project that wraps them.
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ $# -ge 1 ]; then
  NDK=$1
  if [ ! -d "$NDK/toolchains/llvm/prebuilt" ]; then echo "not an NDK: $NDK" >&2; exit 2; fi
  export ANDROID_NDK_HOME="$NDK"
  ln -sfn "$NDK" android/build-android-ndk
  GEN="Unix Makefiles"; command -v ninja >/dev/null 2>&1 && GEN="Ninja"
  NINJA_ARG=""
  if [ "$GEN" = "Ninja" ] && command -v ninja >/dev/null 2>&1; then
    NINJA_ARG="-DCMAKE_MAKE_PROGRAM=$(command -v ninja)"
  fi
  cmake -S android -B build-android -G "$GEN" $NINJA_ARG \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/android/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
    -DEYELIVE_PORTABLE=ON -DEYELIVE_BUILD_GUI=OFF
  cmake --build build-android --target bench_offline -j "$NPROC"
  echo "built: build-android/bench/bench_offline"
  echo "push with: adb push build-android/bench/bench_offline /data/local/tmp/ && adb shell chmod +x /data/local/tmp/bench_offline"
else
  cmake -S android -B build-host -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
  cmake --build build-host --target bench_offline -j "$NPROC"
  echo "built: build-host/bench/bench_offline"
fi
