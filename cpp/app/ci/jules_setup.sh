#!/bin/sh
# Environment setup for a cloud agent (Debian / Ubuntu VM). Safe to run twice.
set -e
sudo apt-get update -y >/dev/null
sudo apt-get install -y build-essential cmake ninja-build python3 python3-pip python3-numpy git curl >/dev/null
# only needed by the GUI tasks (T12-T17): OpenGL + X11 headers and a virtual display for the --shot mode
if [ "$1" = "gui" ]; then sudo apt-get install -y xorg-dev libgl1-mesa-dev libglu1-mesa-dev xvfb >/dev/null; sh "$(dirname "$0")/../../third_party/fetch.sh"; fi
echo "ready: build with   cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build -L <module> --output-on-failure"
