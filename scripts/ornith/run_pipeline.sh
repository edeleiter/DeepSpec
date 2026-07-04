#!/usr/bin/env bash
#
# End-to-end DSpark pipeline for the Ornith-1.0-9B target. This is the default
# CMD of the Docker image; it is read live from the bind-mounted repo so host
# edits apply without a rebuild. Override with `docker run ... bash` to debug.
#
# Stages: 0 preflight (gate) -> 1 data -> 2 target cache -> 3 train -> 4 eval.
# Everything is parameterized by env vars (all have defaults):
#
#   TARGET                  model id/path (default deepreinforce-ai/Ornith-1.0-9B)
#   CONFIG                  DSpark config (default config/dspark/dspark_ornith_9b.py)
#   SAMPLE_SIZE             rows for download_and_split (default 200, MVP-small)
#   TRAIN_JSONL             stage-1 train output (default cache/dataset/perfectblend_train.jsonl)
#   EVAL_JSONL              stage-1 eval output (default cache/dataset/perfectblend_eval.jsonl)
#   CACHE_DIR               target cache dir (default $HOME/.cache/deepspec/ornith_9b_target_cache)
#   CACHE_LOCAL_BATCH_SIZE  stage-2 memory knob (default 2)
#   DRAFT_CKPT              trained draft for eval (default $HOME/checkpoints/deepspec/dspark_ornith_9b/step_latest)
#   CUDA_VISIBLE_DEVICES    GPU to use (default 0 -- one value; spawn uses device_count())
#   MASTER_ADDR/MASTER_PORT single-node rendezvous (default 127.0.0.1/29500)
#   STAGE                   all | preflight | data | cache | train | eval (default all)
#   RUN_PREFLIGHT/DATA/CACHE/TRAIN/EVAL   per-stage on/off when STAGE=all (default 1)
#   FORCE                   1 to bypass idempotency skips (default 0)
set -euo pipefail

TARGET="${TARGET:-deepreinforce-ai/Ornith-1.0-9B}"
CONFIG="${CONFIG:-config/dspark/dspark_ornith_9b.py}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
TRAIN_JSONL="${TRAIN_JSONL:-cache/dataset/perfectblend_train.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-cache/dataset/perfectblend_eval.jsonl}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/deepspec/ornith_9b_target_cache}"
CACHE_LOCAL_BATCH_SIZE="${CACHE_LOCAL_BATCH_SIZE:-2}"
DRAFT_CKPT="${DRAFT_CKPT:-$HOME/checkpoints/deepspec/dspark_ornith_9b/step_latest}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# RANK/WORLD_SIZE here mean node_rank/node_count (single node = 0/1); the actual
# GPU workers come from torch.cuda.device_count() inside each script. init_dist()
# in cache-build and train reads all four of these.
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

STAGE="${STAGE:-all}"
FORCE="${FORCE:-0}"

banner() { echo; echo "=========== [$(date -u +%H:%M:%S)] $* ==========="; }
current_stage="startup"
trap 'echo "PIPELINE FAILED during stage: ${current_stage}" >&2' ERR

# should_run <name> <run_flag_default>: honor STAGE (single-stage selection) and
# the per-stage RUN_* toggles when STAGE=all.
should_run() {
    local name="$1" flag="$2"
    if [[ "$STAGE" != "all" ]]; then
        [[ "$STAGE" == "$name" ]]
        return
    fi
    [[ "$flag" == "1" ]]
}

banner "config"
echo "TARGET=$TARGET"
echo "CONFIG=$CONFIG"
echo "SAMPLE_SIZE=$SAMPLE_SIZE  TRAIN_JSONL=$TRAIN_JSONL"
echo "CACHE_DIR=$CACHE_DIR  CACHE_LOCAL_BATCH_SIZE=$CACHE_LOCAL_BATCH_SIZE"
echo "DRAFT_CKPT=$DRAFT_CKPT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  STAGE=$STAGE  FORCE=$FORCE"

# --- Stage 0: preflight gate -------------------------------------------------
if should_run preflight "${RUN_PREFLIGHT:-1}"; then
    current_stage="preflight"
    banner "stage 0: preflight (does Ornith load + expose the text backbone?)"
    python scripts/ornith/check_load.py --model "$TARGET"
fi

# --- Stage 1: data -----------------------------------------------------------
if should_run data "${RUN_DATA:-1}"; then
    current_stage="data"
    if [[ "$FORCE" != "1" && -f "$TRAIN_JSONL" ]]; then
        banner "stage 1: data -- SKIP (found $TRAIN_JSONL; FORCE=1 to regenerate)"
    else
        banner "stage 1: download + split (sample_size=$SAMPLE_SIZE)"
        # Write eval to a pipeline-owned path so we never collide with the repo's
        # eval_datasets/. download_and_split refuses to overwrite, so clear any
        # stale/partial outputs first.
        mkdir -p "$(dirname "$TRAIN_JSONL")" "$(dirname "$EVAL_JSONL")"
        rm -f "$TRAIN_JSONL" "$EVAL_JSONL"
        python scripts/data/download_and_split.py \
            --sample-size "$SAMPLE_SIZE" \
            --train-output-path "$TRAIN_JSONL" \
            --test-output-dir "$(dirname "$EVAL_JSONL")" \
            --test-output-name "$(basename "$EVAL_JSONL")"
    fi
fi

# --- Stage 2: target cache (idempotent on manifest.json) ---------------------
if should_run cache "${RUN_CACHE:-1}"; then
    current_stage="cache"
    if [[ "$FORCE" != "1" && -f "$CACHE_DIR/manifest.json" ]]; then
        banner "stage 2: target cache -- SKIP (found $CACHE_DIR/manifest.json; FORCE=1 to rebuild)"
    else
        banner "stage 2: build target cache -> $CACHE_DIR"
        # prepare_target_cache requires an empty/new output dir. We only reach
        # here when there is no valid manifest (partial/failed run) or FORCE=1,
        # so clearing the dir is safe -- a completed cache would have skipped above.
        rm -rf "$CACHE_DIR"
        python scripts/data/prepare_target_cache.py \
            --config "$CONFIG" \
            --train-data-path "$TRAIN_JSONL" \
            --output-dir "$CACHE_DIR" \
            --local-batch-size "$CACHE_LOCAL_BATCH_SIZE"
    fi
fi

# --- Stage 3: train (auto-resumes from step_latest) --------------------------
if should_run train "${RUN_TRAIN:-1}"; then
    current_stage="train"
    banner "stage 3: train draft (resumes from step_latest if present)"
    python train.py \
        --config "$CONFIG" \
        --opts "data.target_cache_path=$CACHE_DIR"
fi

# --- Stage 4: eval -----------------------------------------------------------
if should_run eval "${RUN_EVAL:-1}"; then
    current_stage="eval"
    banner "stage 4: acceptance eval"
    # Drive the parameterized eval.sh with the Ornith target + trained draft.
    target_name_or_path="$TARGET" \
    draft_name_or_path="$DRAFT_CKPT" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        bash scripts/eval/eval.sh
fi

current_stage="done"
banner "pipeline complete"
