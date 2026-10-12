#!/usr/bin/env python3
"""Builds the release archive of one platform: executables (stripped where the platform allows), models, INSTALL.txt, THIRD_PARTY.txt, SHA256SUMS.txt.
usage: package.py --build-dir build --os macos|linux|windows --version 0.1.0 --out dist
No packer or compressor is applied to the executables (they trigger antivirus alerts); only the zip container compresses."""
import argparse, hashlib, os, shutil, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def find(build, name, exe):
    for dirpath, _, files in os.walk(build):
        for f in files:
            if f == name + exe:
                return os.path.join(dirpath, f)
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build-dir", required=True); ap.add_argument("--os", required=True, choices=["macos", "linux", "windows"]); ap.add_argument("--version", default="dev"); ap.add_argument("--out", default="dist"); a = ap.parse_args()
    exe = ".exe" if a.os == "windows" else ""; name = f"eyelive-{a.version}-{a.os}"; stage = os.path.join(a.out, name); shutil.rmtree(stage, ignore_errors=True); os.makedirs(os.path.join(stage, "models"))
    found = []
    for tool in ("eyelive", "eyelive_cli"):
        src = find(a.build_dir, tool, exe)
        if not src:
            print(f"warning: {tool}{exe} not found in {a.build_dir}"); continue
        dst = os.path.join(stage, tool + exe); shutil.copy2(src, dst); before = os.path.getsize(dst)
        if a.os in ("macos", "linux"):
            subprocess.run(["strip"] + (["-x"] if a.os == "macos" else []) + [dst], check=False)
        if a.os == "macos":
            subprocess.run(["codesign", "--force", "--sign", "-", dst], check=False)         # ad-hoc signature: Apple Silicon refuses to run a binary without any (not a developer signature, see docs/eyelive/INSTALL_UNSIGNED.md)
        print(f"{tool}{exe}: {before / 1e6:.2f} MB, stripped {os.path.getsize(dst) / 1e6:.2f} MB"); found.append(dst)
    if not found:
        sys.exit("no executable found")
    rc = subprocess.run([sys.executable, os.path.join(HERE, "check_dependencies.py"), a.os] + found).returncode
    for f in sorted(os.listdir(os.path.join(ROOT, "cpp", "models"))):
        if f.endswith((".bin", ".json")):
            shutil.copy2(os.path.join(ROOT, "cpp", "models", f), os.path.join(stage, "models", f))
    shutil.copy2(os.path.join(ROOT, "docs", "eyelive", "INSTALL_UNSIGNED.md"), os.path.join(stage, "INSTALL.txt")); shutil.copy2(os.path.join(HERE, "THIRD_PARTY.txt"), os.path.join(stage, "THIRD_PARTY.txt"))
    shutil.copy2(os.path.join(ROOT, "cpp", "app", "cli", "README.md"), os.path.join(stage, "CLI.txt"))
    with open(os.path.join(stage, "SHA256SUMS.txt"), "w") as f:
        for dirpath, _, files in sorted(os.walk(stage)):
            for fn in sorted(files):
                if fn != "SHA256SUMS.txt":
                    p = os.path.join(dirpath, fn); f.write(f"{sha256(p)}  {os.path.relpath(p, stage).replace(os.sep, '/')}\n")
    zpath = os.path.join(a.out, name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, files in os.walk(stage):
            for fn in files:
                p = os.path.join(dirpath, fn); info = zipfile.ZipInfo.from_file(p, os.path.relpath(p, a.out))
                if fn in ("eyelive", "eyelive_cli"):
                    info.external_attr = 0o755 << 16
                with open(p, "rb") as fh:
                    z.writestr(info, fh.read(), zipfile.ZIP_DEFLATED)
    with open(zpath + ".sha256", "w") as f:
        f.write(f"{sha256(zpath)}  {os.path.basename(zpath)}\n")
    print(f"{zpath}: {os.path.getsize(zpath) / 1e6:.2f} MB, sha256 {sha256(zpath)}")
    sys.exit(rc)

if __name__ == "__main__":
    main()
