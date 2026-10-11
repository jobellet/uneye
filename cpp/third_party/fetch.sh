#!/bin/sh
# Downloads the two GUI dependencies of cpp/app (not stored in git): Dear ImGui (MIT) and GLFW (zlib). Run from anywhere:  sh cpp/third_party/fetch.sh
set -e
D="$(cd "$(dirname "$0")" && pwd)"; IMGUI=v1.92.9b; GLFW=3.4
[ -d "$D/imgui" ] || { curl -fL "https://github.com/ocornut/imgui/archive/refs/tags/$IMGUI.tar.gz" -o "$D/imgui.tar.gz" && mkdir -p "$D/imgui" && tar -xzf "$D/imgui.tar.gz" -C "$D/imgui" --strip-components=1 && rm "$D/imgui.tar.gz"; }
[ -d "$D/glfw" ]  || { curl -fL "https://github.com/glfw/glfw/archive/refs/tags/$GLFW.tar.gz" -o "$D/glfw.tar.gz" && mkdir -p "$D/glfw" && tar -xzf "$D/glfw.tar.gz" -C "$D/glfw" --strip-components=1 && rm "$D/glfw.tar.gz"; }
echo "imgui $IMGUI and glfw $GLFW in $D"
