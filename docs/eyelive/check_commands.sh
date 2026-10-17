#!/bin/sh
# Runs the commands of MANUAL.md against the command-line programs. usage: sh docs/eyelive/check_commands.sh <path of eyelive_cli> [models dir] [path of eyelive_tool]
set -e
CLI="${1:-eyelive_cli}"; MODELS="${2:-}"; TOOL="${3:-}"; D="$(mktemp -d)"; trap 'rm -rf "$D"' EXIT; M=""; [ -n "$MODELS" ] && M="--models-dir $MODELS"
echo "== section 10, example 1"; "$CLI" --sim --duration 5 --session "$D/demo.eyelive" $M | grep -q "^event "
echo "== section 10, example 3 (shortened)"; "$CLI" --sim --duration 20 --prelabel-seconds 15 --train --session "$D/train.eyelive" $M | grep -q "deployed as detector user:cli"
if [ -n "$TOOL" ]; then
  echo "== section 10 (eyelive_tool): info, export, redetect, check, profiles"; "$CLI" --sim --duration 5 --session "$D/t.eyelive" $M >/dev/null
  "$TOOL" info "$D/t.eyelive" | grep -q "EyeLive session report"
  "$TOOL" export "$D/t.eyelive" --csv "$D/e.csv" --bids "$D/b" --mat "$D/x.mat" --report "$D/r.html" | grep -q "exported"; test -s "$D/e.csv"; test -s "$D/b_eyetrack.tsv"; test -s "$D/x.mat"; test -s "$D/r.html"
  mkdir -p "$D/sessions"; cp "$D/t.eyelive" "$D/sessions/"; "$TOOL" redetect "$D/sessions" --detector fast_threshold --out "$D/redo" | grep -q "0 failed"
  HOME="$D" "$TOOL" check --profile "Simulation (demo)" --session-dir "$D/chk" | grep -q "profile"; HOME="$D" "$TOOL" profiles list | grep -q "Simulation (demo)"
fi
echo "== a UDP session ends by itself when the stream is silent: skipped here (needs a sender)"
echo "== invalid use fails as documented"; if "$CLI" --bogus >/dev/null 2>&1; then echo "an unknown option must fail"; exit 1; fi
echo "all manual commands ran"
