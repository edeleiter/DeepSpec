#!/usr/bin/env bash
#
# Reusable launcher for the DeepSpec CUDA image -- model-agnostic. The image
# bakes only deps (the repo is bind-mounted), so one image runs any model's
# pipeline; you pick the model by pointing PIPELINE at its runner and passing
# that runner's env knobs through as `-e KEY=value`.
#
# Launch from Git Bash on Windows. Examples:
#   bash scripts/docker_run.sh                                   # default: qwen3_4b pipeline
#   bash scripts/docker_run.sh -e SAMPLE_SIZE=200 -e REGEN=0     # qwen3_4b smoke test
#   PIPELINE=scripts/ornith/run_pipeline.sh bash scripts/docker_run.sh -e STAGE=cache
#   CMD='bash' bash scripts/docker_run.sh                        # interactive shell instead of a pipeline
#
# Overridable env: IMAGE, PIPELINE, SHM_SIZE, ENV_FILE, CMD.
set -euo pipefail

IMAGE="${IMAGE:-deepspec:latest}"
PIPELINE="${PIPELINE:-scripts/qwen3_4b/run_pipeline.sh}"

# Big artifacts persist in Docker named volumes, NOT the Windows bind mount: HF
# downloads, target caches, and checkpoints (checkpoints use symlinks that don't
# survive a Windows bind mount). Matches the volumes used for the Ornith runs.
run_args=(--gpus all)
# DETACH=1 -> fire-and-forget: runs in the background, survives your shell/session
# closing, and keeps the container after exit so `docker logs` still works. Follow
# with `docker logs -f <CONTAINER_NAME>`; remove with `docker rm <CONTAINER_NAME>`.
# Default -> interactive (--rm -it), tied to this terminal.
if [[ "${DETACH:-0}" == "1" ]]; then
    run_args+=(-d --name "${CONTAINER_NAME:-deepspec-run}")
else
    run_args+=(--rm -it)
fi
run_args+=(
    --shm-size="${SHM_SIZE:-8g}"    # <8g -> DataLoader dies with a /dev/shm bus error
    -v "$(pwd -W)":/workspace -w /workspace
    -v deepspec-hf:/root/.cache/huggingface
    -v deepspec-cache:/root/.cache/deepspec
    -v deepspec-ckpt:/root/checkpoints
)

# --env-file is optional (only if a .env with HF_TOKEN exists).
env_file="${ENV_FILE:-.env}"
[[ -f "$env_file" ]] && run_args+=(--env-file "$env_file")

# Trailing "$@" are extra docker flags (e.g. -e SAMPLE_SIZE=200); they must come
# before the image name. The container command defaults to the chosen pipeline,
# but CMD='bash' (or any string) overrides it for an interactive/debug shell.
container_cmd=("bash" "$PIPELINE")
[[ -n "${CMD:-}" ]] && container_cmd=(${CMD})

MSYS_NO_PATHCONV=1 docker run "${run_args[@]}" "$@" "$IMAGE" "${container_cmd[@]}"
