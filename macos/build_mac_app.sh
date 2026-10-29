#!/usr/bin/env bash
# Builds the native macOS application bundle "U'n'Eye.app" and installs it to /Applications/U'n'Eye.app
# Usage:
#   ./macos/build_mac_app.sh              # builds build/macos/U'n'Eye.app and installs to /Applications/U'n'Eye.app
#   ./macos/build_mac_app.sh --no-install # only builds build/macos/U'n'Eye.app and build/macos/UnEye-macOS.zip
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/build/macos"
APP_NAME="U'n'Eye.app"
APP_BUNDLE="$OUT_DIR/$APP_NAME"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RES_DIR="$CONTENTS/Resources"
WEB_DEST="$RES_DIR/web"

INSTALL_TO_APPS=1
for arg in "$@"; do
    if [[ "$arg" == "--no-install" ]]; then
        INSTALL_TO_APPS=0
    fi
done

echo "==> Building $APP_NAME from $REPO_ROOT ..."
rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR" "$WEB_DEST/web" "$WEB_DEST/models"

# 1. Generate macOS AppIcon.icns from logo_circle.png
ICONSET_DIR="$OUT_DIR/AppIcon.iconset"
rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"
SRC_ICON="$REPO_ROOT/logo_circle.png"
if [[ -f "$SRC_ICON" ]]; then
    for sz in 16 32 128 256 512; do
        sips -z $sz $sz "$SRC_ICON" --out "$ICONSET_DIR/icon_${sz}x${sz}.png" >/dev/null 2>&1
        sz2=$((sz * 2))
        sips -z $sz2 $sz2 "$SRC_ICON" --out "$ICONSET_DIR/icon_${sz}x${sz}@2x.png" >/dev/null 2>&1
    done
    iconutil -c icns "$ICONSET_DIR" -o "$RES_DIR/AppIcon.icns"
    rm -rf "$ICONSET_DIR"
fi

# 2. Compile UnEyeApp.swift (Universal binary arm64 + x86_64 if supported by SDK, otherwise host arch)
echo "==> Compiling native macOS binary ..."
if swiftc -O "$REPO_ROOT/macos/UnEyeApp.swift" \
    -target arm64-apple-macos11.0 \
    -framework Cocoa -framework WebKit \
    -o "$OUT_DIR/UnEye_arm64" 2>/dev/null && \
   swiftc -O "$REPO_ROOT/macos/UnEyeApp.swift" \
    -target x86_64-apple-macos11.0 \
    -framework Cocoa -framework WebKit \
    -o "$OUT_DIR/UnEye_x86_64" 2>/dev/null; then
    lipo -create -output "$MACOS_DIR/UnEye" "$OUT_DIR/UnEye_arm64" "$OUT_DIR/UnEye_x86_64"
    rm -f "$OUT_DIR/UnEye_arm64" "$OUT_DIR/UnEye_x86_64"
else
    swiftc -O "$REPO_ROOT/macos/UnEyeApp.swift" \
        -framework Cocoa -framework WebKit \
        -o "$MACOS_DIR/UnEye"
fi
chmod +x "$MACOS_DIR/UnEye"

# 3. Copy web assets into Contents/Resources/web/ for immediate offline availability
echo "==> Bundling web assets ..."
for f in index.html manifest.json style.css app.js formats.js import.js import_ui.js hmm_prelabel.js nearmiss.js train_worker.js sw.js logo.jpeg logo_circle.png; do
    if [[ -f "$REPO_ROOT/$f" ]]; then
        cp "$REPO_ROOT/$f" "$WEB_DEST/$f"
    fi
done
for f in uneye_train.js uneye_train.wasm h5wasm.js; do
    if [[ -f "$REPO_ROOT/web/$f" ]]; then
        cp "$REPO_ROOT/web/$f" "$WEB_DEST/web/$f"
    fi
done
for f in combined.bin combined.json; do
    if [[ -f "$REPO_ROOT/models/$f" ]]; then
        cp "$REPO_ROOT/models/$f" "$WEB_DEST/models/$f"
    fi
done

# 4. Write Info.plist
LOCAL_REPO_PLIST=""
if [[ "$INSTALL_TO_APPS" -eq 1 ]]; then
    LOCAL_REPO_PLIST="    <key>UnEyeLocalRepoPath</key>
    <string>$REPO_ROOT</string>"
fi

cat > "$CONTENTS/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>U'n'Eye</string>
    <key>CFBundleExecutable</key>
    <string>UnEye</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.uneye.labeler</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>U'n'Eye</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.13</string>
    <key>CFBundleVersion</key>
    <string>13</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsLocalNetworking</key>
        <true/>
    </dict>
$LOCAL_REPO_PLIST
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>Eye Tracking Recording or Labels</string>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            <key>LSHandlerRank</key>
            <string>Alternate</string>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>h5</string>
                <string>hdf5</string>
                <string>hdf</string>
                <string>mat</string>
                <string>npz</string>
                <string>csv</string>
                <string>tsv</string>
                <string>txt</string>
                <string>asc</string>
                <string>arff</string>
                <string>json</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
EOF

# 5. Ad-hoc sign bundle
codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null 2>&1 || true

# 6. Create distributable zip archive
(cd "$OUT_DIR" && rm -f UnEye-macOS.zip && ditto -c -k --sequesterRsrc --keepParent "$APP_NAME" UnEye-macOS.zip)
echo "==> Built: $APP_BUNDLE (and $OUT_DIR/UnEye-macOS.zip)"

# 7. Install to /Applications/U'n'Eye.app (or ~/Applications if /Applications is not writable)
if [[ "$INSTALL_TO_APPS" -eq 1 ]]; then
    TARGET_DIR="/Applications"
    if [[ ! -w "$TARGET_DIR" ]]; then
        TARGET_DIR="$HOME/Applications"
        mkdir -p "$TARGET_DIR"
    fi
    rm -rf "$TARGET_DIR/$APP_NAME"
    cp -R "$APP_BUNDLE" "$TARGET_DIR/$APP_NAME"
    xattr -cr "$TARGET_DIR/$APP_NAME" 2>/dev/null || true

    # Also update ~/Library/Application Support/UnEye/web/ immediately
    USER_WEB_DIR="$HOME/Library/Application Support/UnEye/web"
    mkdir -p "$USER_WEB_DIR/web" "$USER_WEB_DIR/models"
    cp -R "$WEB_DEST/"* "$USER_WEB_DIR/"
    echo "==> Installed to: $TARGET_DIR/$APP_NAME"
fi
