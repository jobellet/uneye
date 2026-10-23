#!/usr/bin/env bash
# Build U'n'Eye C++ trainer and inference engine to WebAssembly with Emscripten.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "${EMSDK:-}" ]; then
    if [ -f "$HOME/emsdk/emsdk_env.sh" ]; then
        source "$HOME/emsdk/emsdk_env.sh"
    elif [ -f "/opt/emsdk/emsdk_env.sh" ]; then
        source "/opt/emsdk/emsdk_env.sh"
    fi
fi

if ! command -v em++ >/dev/null 2>&1; then
    echo "Error: em++ not found in PATH. Please install/activate emsdk." >&2
    exit 1
fi

mkdir -p "$ROOT_DIR/web"
cd "$ROOT_DIR"

echo "Building uneye_train.wasm and uneye_train.js (SIMD128, -O3)..."
em++ -O3 -msimd128 -std=c++17 --bind \
    -I cpp/app/include -I cpp/app/train/include -I cpp/include \
    cpp/web/uneye_web.cpp cpp/app/train/src/dataset.cpp cpp/app/train/src/model_io.cpp \
    -s MODULARIZE=1 -s EXPORT_NAME=createUnEyeModule \
    -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=536870912 \
    -s FILESYSTEM=0 -s ENVIRONMENT="web,worker,node" \
    -o web/uneye_train.js

echo "Build complete: web/uneye_train.js and web/uneye_train.wasm"
ls -lh web/uneye_train.js web/uneye_train.wasm
