#!/usr/bin/env bash
#
# Clean spin-down for a wslc DeepSpec run. wslc keeps a per-user **session VM**
# (vmmemwslc-cli-<user>) alive after a container is killed -- the wslc equivalent
# of Docker Desktop's vmmem. It holds gigabytes of RAM and keeps volume vhdx files
# locked (so `volume remove` fails with ERROR_SHARING_VIOLATION, and `wsl --shutdown`
# does NOT touch it). This script does the FULL teardown so nothing lingers:
#   1. stop + remove the named container
#   2. prune dangling containers (half-created leftovers keep volumes "in use")
#   3. terminate the wslc session VM  -> releases RAM + unlocks volume vhdx
#   4. belt-and-suspenders: force-kill the vmmem process if it survives step 3
#
# Usage:  bash scripts/wslc_stop.sh [CONTAINER_NAME]
#   (defaults to $CONTAINER_NAME, then "deepspec-run")
set -uo pipefail

WSLC="${WSLC:-}"
if [[ -z "$WSLC" ]]; then
    if command -v wslc.exe >/dev/null 2>&1; then WSLC="wslc.exe"; else WSLC="/c/Program Files/WSL/wslc.exe"; fi
fi
NAME="${1:-${CONTAINER_NAME:-deepspec-run}}"

echo "[wslc-stop] killing + removing container '$NAME'..."
"$WSLC" kill "$NAME"          2>/dev/null | tr -d '\0' >/dev/null || true
"$WSLC" container rm "$NAME"  2>/dev/null | tr -d '\0' >/dev/null || true

echo "[wslc-stop] pruning dangling containers (release stale volume refs)..."
"$WSLC" container prune       2>/dev/null | tr -d '\0' | grep -i reclaim || true

echo "[wslc-stop] terminating session VM (frees RAM + unlocks volume vhdx)..."
"$WSLC" system session terminate 2>/dev/null | tr -d '\0' >/dev/null || true
sleep 2

# Force-kill the session VM process if `terminate` didn't take.
if powershell.exe -NoProfile -Command "if (Get-Process -Name 'vmmemwslc*' -ErrorAction SilentlyContinue) { Get-Process -Name 'vmmemwslc*' | Stop-Process -Force -ErrorAction SilentlyContinue; 'force-killed' }" 2>/dev/null | tr -d '\0' | grep -qi force-killed; then
    echo "[wslc-stop] force-killed a lingering vmmem process"
fi

echo "[wslc-stop] done -- container '$NAME' removed, session terminated, RAM released."
