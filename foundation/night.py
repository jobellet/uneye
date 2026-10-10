#!/usr/bin/env python3
"""Overnight driver: runs the candidates one after the other (one GPU job at a time), each with a time cap, continues after a failure, rebuilds docs/OVERNIGHT_REPORT.md
after every step, commits and pushes the report + results after every step. Stops launching new training after --deadline-hours (then label efficiency + final report).
Run (keeps the machine awake):  nohup caffeinate -i .venv/bin/python -u foundation/night.py --deadline-hours 8.5 > foundation/runs/night/driver.log 2>&1 &
"""
import argparse, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHT = os.path.join(ROOT, "foundation", "runs", "night"); LOGS = os.path.join(NIGHT, "logs"); os.makedirs(LOGS, exist_ok=True)
PY = os.path.join(ROOT, ".venv", "bin", "python")
V = ["-u", "foundation/ssl_vit.py"]; T2 = ["-u", "foundation/ts2vec.py"]
# (id, command arguments after python, cap in minutes, is_training)  -- ordered by priority
STEPS = [
    ("eval_jepa_a", ["-u", "foundation/eval_ckpt.py", "--name", "jepa_a"], 10, False),
    ("ctrl_untrained", V + ["--name", "ctrl_untrained", "--objective", "jepa", "--steps", "0"], 15, False),
    ("ctrl_untrained_ts2vec", T2 + ["--name", "ctrl_untrained_ts2vec", "--steps", "0"], 10, False),
    ("base_input_6ch", ["-u", "foundation/night_eval.py", "--six"], 8, False),
    ("selftrain", ["-u", "foundation/selftrain.py", "--name", "selftrain", "--rounds", "2", "--steps", "4000", "--max-minutes", "22"], 75, True),
    ("sup_bitcn", ["-u", "foundation/sup_bitcn.py", "--name", "sup_bitcn", "--steps", "6000", "--max-minutes", "28"], 50, True),
    ("ts2vec_a", T2 + ["--name", "ts2vec_a", "--steps", "6000", "--max-minutes", "35"], 65, True),
    ("mae_a", V + ["--name", "mae_a", "--objective", "mae", "--steps", "6000", "--max-minutes", "35"], 65, True),
    ("hubert_a", V + ["--name", "hubert_a", "--objective", "hubert", "--steps", "6000", "--max-minutes", "35"], 65, True),
    ("ts2vec_b", T2 + ["--name", "ts2vec_b", "--depth", "8", "--dim", "192", "--hid", "96", "--lr", "5e-4", "--steps", "6000", "--max-minutes", "35"], 65, True),
    ("jepa_b", V + ["--name", "jepa_b", "--lr", "1e-4", "--steps", "5000", "--max-minutes", "30"], 55, True),
    ("jepa_d", V + ["--name", "jepa_d", "--block", "6", "--ema", "0.99", "--steps", "5000", "--max-minutes", "30"], 55, True),
    ("jepa_e", V + ["--name", "jepa_e", "--patch", "8", "--block", "6", "--steps", "5000", "--max-minutes", "25"], 50, True),
    ("mae_b", V + ["--name", "mae_b", "--objective", "mae", "--lr", "1e-4", "--block", "6", "--steps", "5000", "--max-minutes", "25"], 50, True),
    ("label_eff", ["-u", "foundation/label_eff.py"], 60, False),
]


EXTRA = [   # second queue, started after the first has finished (--set extra): what the first results suggested
    ("selftrain_b", ["-u", "foundation/selftrain.py", "--name", "selftrain_b", "--rounds", "3", "--steps", "4000", "--lr", "1e-3", "--criterion", "thr", "--max-minutes", "22"], 90, True),
    ("sup_bitcn_b", ["-u", "foundation/sup_bitcn.py", "--name", "sup_bitcn_b", "--steps", "6000", "--lr", "1e-3", "--pos-weight", "1.5", "--max-minutes", "28"], 50, True),
]


def git_push(msg):
    env = dict(os.environ, GIT_EXEC_PATH="/Applications/GitHub Desktop.app/Contents/Resources/app/git/libexec/git-core", GCM_GUI_PROMPT="false")
    g = "/Applications/GitHub Desktop.app/Contents/Resources/app/git/bin/git"
    run = lambda *a, **k: subprocess.run([g, "-c", "user.name=Claude", "-c", "user.email=noreply@anthropic.com", *a], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, **k)
    try:
        run("add", "docs/OVERNIGHT_REPORT.md", "docs/figs_night", "foundation/runs/night/*.json")
        r = run("commit", "-q", "-m", msg + "\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>")
        subprocess.run(["script", "-q", "/dev/null", g, "-c", "credential.helper=", "-c", "credential.helper=manager", "-c", "credential.guiPrompt=false", "-c", "credential.credentialStore=keychain", "push", "origin", "master"],
                       cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    except Exception as e:
        print("   git step failed:", type(e).__name__, e, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--deadline-hours", type=float, default=8.5); ap.add_argument("--only", default=""); ap.add_argument("--set", default="main", choices=["main", "extra"])
    ap.add_argument("--wait-for-finished", action="store_true"); a = ap.parse_args()
    if a.wait_for_finished:
        while not os.path.exists(os.path.join(NIGHT, "FINISHED")): time.sleep(30)
        os.remove(os.path.join(NIGHT, "FINISHED"))
    t_start = time.time(); sp = os.path.join(NIGHT, "status.json" if a.set == "main" else "status_extra.json"); status = []
    # a training run started before the driver (jepa_a) may still be running: wait for it
    while subprocess.run(["pgrep", "-f", "ssl_vit.py --name jepa_a"], capture_output=True).stdout.strip(): time.sleep(20)
    for cid, args, cap, training in (STEPS if a.set == "main" else EXTRA):
        if a.only and cid not in a.only.split(","): continue
        done_file = os.path.join(NIGHT, f"{cid}.json")
        if (time.time() - t_start) / 3600 > a.deadline_hours and training: status.append(dict(id=cid, state="skipped (deadline)")); json.dump(status, open(sp, "w"), indent=1); continue
        if os.path.exists(done_file) and cid not in ("selftrain", "sup_bitcn", "label_eff"): status.append(dict(id=cid, state="done", minutes="(already done)")); continue
        t0 = time.time(); status.append(dict(id=cid, state="running", started=time.strftime("%H:%M"))); json.dump(status, open(sp, "w"), indent=1)
        print(f"=== {time.strftime('%H:%M')} {cid} (cap {cap} min)", flush=True)
        try:
            with open(os.path.join(LOGS, f"{cid}.log"), "w") as lf: r = subprocess.run([PY, *args], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=cap * 60)
            state = "done" if r.returncode == 0 else f"failed (exit {r.returncode})"
        except subprocess.TimeoutExpired: state = f"timed out after {cap} min"
        except Exception as e: state = f"error {type(e).__name__}"
        status[-1].update(state=state, minutes=round((time.time() - t0) / 60, 1)); json.dump(status, open(sp, "w"), indent=1); print(f"    -> {state} in {status[-1]['minutes']} min", flush=True)
        subprocess.run([PY, "foundation/night_report.py"], cwd=ROOT, capture_output=True); git_push(f"Overnight comparison: {cid} ({state})")
    subprocess.run([PY, "foundation/night_report.py"], cwd=ROOT, capture_output=True); git_push("Overnight comparison: final report")
    open(os.path.join(NIGHT, "FINISHED" if a.set == "main" else "FINISHED_EXTRA"), "w").write(time.strftime("%Y-%m-%d %H:%M")); print("=== finished", time.strftime("%H:%M"), flush=True)


if __name__ == "__main__":
    main()
