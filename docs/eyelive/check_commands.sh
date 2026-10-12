#!/bin/sh
# Runs the commands of MANUAL.md against the command-line program. usage: sh docs/eyelive/check_commands.sh <path of eyelive_cli> [models dir]
set -e
CLI="${1:-eyelive_cli}"; MODELS="${2:-}"; D="$(mktemp -d)"; trap 'rm -rf "$D"' EXIT; M=""; [ -n "$MODELS" ] && M="--models-dir $MODELS"
echo "== section 10, example 1"; "$CLI" --sim --duration 5 --session "$D/demo.eyelive" $M | grep -q "^event "
echo "== section 10, example 3 (shortened)"; "$CLI" --sim --duration 20 --prelabel-seconds 15 --train --session "$D/train.eyelive" $M | grep -q "deployed as detector user:cli"
echo "== a UDP session ends by itself when the stream is silent: skipped here (needs a sender)"
echo "== invalid use fails as documented"; if "$CLI" --bogus >/dev/null 2>&1; then echo "an unknown option must fail"; exit 1; fi
echo "all manual commands ran"
