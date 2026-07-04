#!/usr/bin/env bash
#
# Reusable launcher for the DeepSpec image on **WSL Containers (wslc)** -- the
# native Windows 11 Linux-container runtime (WSL 2.9.3+), no Docker Desktop.
# This is the wslc twin of scripts/docker_run.sh: identical model-agnostic
# design (the image bakes only deps, the repo is bind-mounted, pick the model
# via PIPELINE), just driven by `wslc run` instead of `docker run`.
#
# wslc's CLI is Docker-compatible for our needs: --gpus all, --shm-size, -v
# (named volumes AND bind mounts), -e/--env-file, -w, --name, -d, --rm, -i, -t.
# GPU passthrough is via the Container Device Interface and works on sm_120.
#
# Prereqs (one-time): build the image and create the volumes in the wslc store:
#   wslc build -t deepspec:latest .
#   wslc volume create deepspec-hf; wslc volume create deepspec-cache; wslc volume create deepspec-ckpt
#
# Launch from Git Bash on Windows. Examples:
#   bash scripts/wslc_run.sh                                    # default: qwen3_4b pipeline
#   bash scripts/wslc_run.sh -e SAMPLE_SIZE=200 -e REGEN=0      # smoke test
#   DETACH=1 CONTAINER_NAME=deepspec-full bash scripts/wslc_run.sh -e STAGE=train
#   CMD='bash' bash scripts/wslc_run.sh                         # interactive shell
#
# Overridable env: WSLC (path to wslc.exe), IMAGE, PIPELINE, SHM_SIZE, ENV_FILE, CMD.
set -euo pipefail

# wslc.exe ships with WSL 2.9.3+ under C:\Program Files\WSL. Prefer PATH, fall
# back to the install dir (a shell opened before the WSL update lacks it on PATH).
WSLC="${WSLC:-}"
if [[ -z "$WSLC" ]]; then
    if command -v wslc.exe >/dev/null 2>&1; then
        WSLC="wslc.exe"
    else
        WSLC="/c/Program Files/WSL/wslc.exe"
    fi
fi

IMAGE="${IMAGE:-deepspec:latest}"
PIPELINE="${PIPELINE:-scripts/qwen3_4b/run_pipeline.sh}"

# Big artifacts persist in wslc named volumes, NOT the Windows bind mount: HF
# downloads, target caches, and checkpoints (checkpoints use symlinks that don't
# survive a Windows bind mount). These are SEPARATE from the Docker volumes of the
# same name -- wslc has its own volume store, so a full data re-run is expected the
# first time you switch runtimes.
run_args=(--gpus all)
# DETACH=1 -> fire-and-forget: runs in the background, survives your shell closing,
# and keeps the container after exit so `wslc logs` still works. Follow with
# `wslc logs -f <CONTAINER_NAME>`; remove with `wslc kill`/`wslc container rm`.
# Default -> interactive (--rm -i -t), tied to this terminal.
if [[ "${DETACH:-0}" == "1" ]]; then
    run_args+=(-d --name "${CONTAINER_NAME:-deepspec-run}")
else
    run_args+=(--rm -i -t)
fi
run_args+=(
    --shm-size="${SHM_SIZE:-8G}"    # <8G -> DataLoader dies with a /dev/shm bus error (wslc wants uppercase G)
    # wslc bind mounts want a Windows forward-slash path; `pwd -W` yields F:/DeepSpec.
    -v "$(pwd -W):/workspace" -w /workspace
    -v deepspec-hf:/root/.cache/huggingface
    # CACHE_SRC: the ~180 GB target cache. Default is the C: SSD named volume (fast).
    # Override with a Windows path (e.g. D:/deepspec-cache) to keep the big cache OFF
    # the system drive -- the cache is symlink-free so a bind mount is safe (unlike
    # checkpoints below, which need a native volume for the step_latest symlink).
    -v "${CACHE_SRC:-deepspec-cache}:/root/.cache/deepspec"
    -v deepspec-ckpt:/root/checkpoints
)

# --env-file is optional (only if a .env with HF_TOKEN exists).
env_file="${ENV_FILE:-.env}"
[[ -f "$env_file" ]] && run_args+=(--env-file "$env_file")

# Trailing "$@" are extra wslc flags (e.g. -e SAMPLE_SIZE=200); they must come
# before the image name. The container command defaults to the chosen pipeline,
# but CMD='bash' (or any string) overrides it for an interactive/debug shell.
container_cmd=("bash" "$PIPELINE")
[[ -n "${CMD:-}" ]] && container_cmd=(${CMD})

MSYS_NO_PATHCONV=1 "$WSLC" run "${run_args[@]}" "$@" "$IMAGE" "${container_cmd[@]}"
