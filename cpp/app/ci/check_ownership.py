#!/usr/bin/env python3
"""Fails when a pull request from a branch named jules/<TASK>-... changes files its task does not own (docs/tasks/OWNERSHIP.json).
Usage in CI:  python3 cpp/app/ci/check_ownership.py <branch-name> <base-ref>      (branches that do not start with 'jules/' are not checked: they belong to the integrator or to a human)
"""
import fnmatch, json, os, re, subprocess, sys

def glob_to_regex(g):
    out, i = "", 0
    while i < len(g):
        if g[i:i + 3] == "**/": out += "(?:.*/)?"; i += 3
        elif g[i:i + 2] == "**": out += ".*"; i += 2
        elif g[i] == "*": out += "[^/]*"; i += 1
        elif g[i] == "?": out += "[^/]"; i += 1
        else: out += re.escape(g[i]); i += 1
    return re.compile("^" + out + "$")

def main():
    branch, base = sys.argv[1], sys.argv[2]
    m = re.match(r"^jules/(T\d\d)(?:-|$)", branch)
    if not m:
        print(f"branch '{branch}' is not a task branch (jules/Txx-...): ownership not checked"); return 0
    task = m.group(1); own = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "tasks", "OWNERSHIP.json")))
    if task not in own: print(f"unknown task {task} in OWNERSHIP.json"); return 1
    pats = [glob_to_regex(g) for g in own[task]]
    changed = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"], capture_output=True, text=True, check=True).stdout.split()
    bad = [f for f in changed if not any(p.match(f) for p in pats)]
    print(f"task {task}: {len(changed)} changed file(s), allowed patterns: {own[task]}")
    if bad:
        print("FILES OUTSIDE THE OWNERSHIP OF", task); [print("  ", f) for f in bad]
        print("If the change is really needed, open an issue labelled 'contract' or 'ownership' instead of editing these files."); return 1
    print("ownership OK"); return 0

if __name__ == "__main__":
    sys.exit(main())
