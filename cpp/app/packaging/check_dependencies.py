#!/usr/bin/env python3
"""Prints the direct dynamic dependencies of an executable and fails when one is not listed in ALLOWED_DEPENDENCIES.txt.
usage: check_dependencies.py <macos|linux|windows> <executable> [...]"""
import os, re, subprocess, sys

def allowed(os_name):
    out, cur = [], None
    for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALLOWED_DEPENDENCIES.txt")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"\[(\w+)\]$", line)
        if m:
            cur = m.group(1)
        elif cur == os_name:
            out.append(line.lower())
    return out

def deps(os_name, exe):
    if os_name == "macos":
        text = subprocess.run(["otool", "-L", exe], capture_output=True, text=True, check=True).stdout
        return [l.strip().split(" (")[0] for l in text.splitlines()[1:] if l.strip()]
    if os_name == "linux":
        text = subprocess.run(["readelf", "-d", exe], capture_output=True, text=True, check=True).stdout
        return re.findall(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", text)
    text = subprocess.run(["dumpbin", "/dependents", exe], capture_output=True, text=True, check=True).stdout
    return re.findall(r"^\s+(\S+\.dll)\s*$", text, re.M | re.I)

def main():
    os_name, exes = sys.argv[1], sys.argv[2:]
    ok_list = allowed(os_name); bad = 0
    for exe in exes:
        size = os.path.getsize(exe); print(f"{exe}: {size / 1e6:.2f} MB")
        for d in deps(os_name, exe):
            good = any(d.lower() == a or d.lower().startswith(a) for a in ok_list)
            print(f"   {'ok   ' if good else 'NOT ALLOWED'} {d}"); bad += not good
    if bad:
        print(f"{bad} dependency(ies) outside packaging/ALLOWED_DEPENDENCIES.txt"); sys.exit(1)

if __name__ == "__main__":
    main()
