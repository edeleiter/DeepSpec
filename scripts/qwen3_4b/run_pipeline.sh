#!/usr/bin/env bash
#
# End-to-end DSpark pipeline for the stock Qwen/Qwen3-4B target on a single GPU.
# A Qwen3-4B twin of scripts/ornith/run_pipeline.sh, with one addition: the
# SGLang answer-regeneration stage (the "full" DSpark recipe -- the draft learns
# from the target's own on-policy generations, which is what makes accept_len
# meaningful). The regen stage self-manages a single-worker SGLang server.
#
# Stages: 1 data -> 2 regen (SGLang) -> 3 cache -> 4 train -> 5 eval.
# Everything is parameterized by env vars (all have defaults):
#
#   TARGET                  model id/path (default Qwen/Qwen3-4B)
#   CONFIG                  DSpark config (default config/dspark/dspark_qwen3_4b_trial.py)
#   SAMPLE_SIZE             rows for download_and_split (default 4000; 200 = smoke test)
#   REGEN                   1 = regenerate answers with the target via SGLang (default 1);
#                           0 = skip regen and build the cache on raw split answers (fast smoke path)
#   TRAIN_JSONL             stage-1 train split       (default cache/dataset/qwen3_4b/perfectblend_train.jsonl)
#   EVAL_JSONL              stage-1 eval split        (default cache/dataset/qwen3_4b/perfectblend_eval.jsonl)
#   TRAIN_REGEN             stage-2 regenerated train (default cache/dataset/qwen3_4b/perfectblend_train_regen.jsonl)
#   CACHE_DIR               target cache dir          (default $HOME/.cache/deepspec/qwen3_4b_target_cache)
#   CACHE_LOCAL_BATCH_SIZE  stage-3 memory knob (default 2)
#   DRAFT_CKPT              trained draft for eval (default $HOME/checkpoints/deepspec/dspark_qwen3_4b_trial/step_latest)
#   SGLANG_PORT             regen server port (default 30000)
#   SGLANG_MEM_FRAC         --mem-fraction-static (default 0.8; Qwen3-4B ~8GB weights + KV in 16GB)
#   SGLANG_READY_TIMEOUT    seconds to wait for the server /health (default 600)
#   GEN_CONCURRENCY         in-flight requests (default 16)
#   GEN_MAX_TOKENS          generation cap per turn (default 4096)
#   GEN_TEMPERATURE/TOP_P/TOP_K/MIN_P  sampling (default 0.7/0.8/20/0)
#   CUDA_VISIBLE_DEVICES    GPU to use (default 0)
#   MASTER_ADDR/MASTER_PORT single-node rendezvous (default 127.0.0.1/29500)
#   STAGE                   all | data | regen | cache | train | eval (default all)
#   RUN_DATA/REGEN_STAGE/CACHE/TRAIN/EVAL   per-stage on/off when STAGE=all (default 1)
#   FORCE                   1 to bypass idempotency skips (default 0)
set -euo pipefail

TARGET="${TARGET:-Qwen/Qwen3-4B}"
CONFIG="${CONFIG:-config/dspark/dspark_qwen3_4b_trial.py}"
SAMPLE_SIZE="${SAMPLE_SIZE:-4000}"
REGEN="${REGEN:-1}"
TRAIN_JSONL="${TRAIN_JSONL:-cache/dataset/qwen3_4b/perfectblend_train.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-cache/dataset/qwen3_4b/perfectblend_eval.jsonl}"
TRAIN_REGEN="${TRAIN_REGEN:-cache/dataset/qwen3_4b/perfectblend_train_regen.jsonl}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/deepspec/qwen3_4b_target_cache}"
CACHE_LOCAL_BATCH_SIZE="${CACHE_LOCAL_BATCH_SIZE:-2}"
DRAFT_CKPT="${DRAFT_CKPT:-$HOME/checkpoints/deepspec/dspark_qwen3_4b_trial/step_latest}"

SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_MEM_FRAC="${SGLANG_MEM_FRAC:-0.8}"
SGLANG_READY_TIMEOUT="${SGLANG_READY_TIMEOUT:-600}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-16}"
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-4096}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.7}"
GEN_TOP_P="${GEN_TOP_P:-0.8}"
GEN_TOP_K="${GEN_TOP_K:-20}"
GEN_MIN_P="${GEN_MIN_P:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# RANK/WORLD_SIZE here mean node_rank/node_count (single node = 0/1); actual GPU
# workers come from torch.cuda.device_count() inside each script.
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

STAGE="${STAGE:-all}"
FORCE="${FORCE:-0}"

# When REGEN=0 the cache trains on the raw split answers instead of regenerated
# ones (fast smoke path, matching the Ornith MVP).
if [[ "$REGEN" == "1" ]]; then
    CACHE_INPUT="$TRAIN_REGEN"
else
    CACHE_INPUT="$TRAIN_JSONL"
fi

banner() { echo; echo "=========== [$(date -u +%H:%M:%S)] $* ==========="; }
current_stage="startup"
trap 'echo "PIPELINE FAILED during stage: ${current_stage}" >&2' ERR

should_run() {
    local name="$1" flag="$2"
    if [[ "$STAGE" != "all" ]]; then
        [[ "$STAGE" == "$name" ]]
        return
    fi
    [[ "$flag" == "1" ]]
}

banner "config"
echo "TARGET=$TARGET  CONFIG=$CONFIG"
echo "SAMPLE_SIZE=$SAMPLE_SIZE  REGEN=$REGEN"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "TRAIN_REGEN=$TRAIN_REGEN  (cache input: $CACHE_INPUT)"
echo "CACHE_DIR=$CACHE_DIR  CACHE_LOCAL_BATCH_SIZE=$CACHE_LOCAL_BATCH_SIZE"
echo "DRAFT_CKPT=$DRAFT_CKPT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  STAGE=$STAGE  FORCE=$FORCE"

# --- Stage 1: data -----------------------------------------------------------
if should_run data "${RUN_DATA:-1}"; then
    current_stage="data"
    if [[ "$FORCE" != "1" && -f "$TRAIN_JSONL" ]]; then
        banner "stage 1: data -- SKIP (found $TRAIN_JSONL; FORCE=1 to regenerate)"
    else
        banner "stage 1: download + split (sample_size=$SAMPLE_SIZE)"
        mkdir -p "$(dirname "$TRAIN_JSONL")" "$(dirname "$EVAL_JSONL")"
        rm -f "$TRAIN_JSONL" "$EVAL_JSONL"
        python scripts/data/download_and_split.py \
            --sample-size "$SAMPLE_SIZE" \
            --train-output-path "$TRAIN_JSONL" \
            --test-output-dir "$(dirname "$EVAL_JSONL")" \
            --test-output-name "$(basename "$EVAL_JSONL")"
    fi
fi

# --- Stage 2: regenerate answers with the target via SGLang ------------------
# Self-managed single-worker server: start in background, wait for /health, run
# the client, then always tear the server down (GPU is needed by later stages).
if [[ "$REGEN" == "1" ]] && should_run regen "${RUN_REGEN_STAGE:-1}"; then
    current_stage="regen"
    if [[ "$FORCE" != "1" && -f "$TRAIN_REGEN" ]]; then
        banner "stage 2: regen -- SKIP (found $TRAIN_REGEN; FORCE=1 to regenerate)"
    else
        banner "stage 2: regenerate answers with $TARGET (SGLang :$SGLANG_PORT)"
        mkdir -p "$(dirname "$TRAIN_REGEN")" logs
        sglang_log="logs/sglang_qwen3_4b_${SGLANG_PORT}.log"

        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" sglang serve \
            --model-path "$TARGET" \
            --host 127.0.0.1 \
            --port "$SGLANG_PORT" \
            --dtype bfloat16 \
            --mem-fraction-static "$SGLANG_MEM_FRAC" \
            > "$sglang_log" 2>&1 &
        sglang_pid=$!
        # Ensure the server dies with this stage even on error/interrupt.
        trap 'kill "$sglang_pid" >/dev/null 2>&1 || true' RETURN
        stop_sglang() { kill "$sglang_pid" >/dev/null 2>&1 || true; wait "$sglang_pid" 2>/dev/null || true; }

        echo "waiting up to ${SGLANG_READY_TIMEOUT}s for SGLang /health (pid=$sglang_pid, log=$sglang_log)"
        ready=0
        for ((i = 0; i < SGLANG_READY_TIMEOUT; i++)); do
            if ! kill -0 "$sglang_pid" 2>/dev/null; then
                echo "SGLang process exited early; tail of $sglang_log:" >&2
                tail -n 40 "$sglang_log" >&2 || true
                exit 1
            fi
            if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null 2>&1; then
                ready=1; break
            fi
            sleep 1
        done
        if [[ "$ready" != "1" ]]; then
            echo "SGLang did not become ready within ${SGLANG_READY_TIMEOUT}s; tail of $sglang_log:" >&2
            tail -n 40 "$sglang_log" >&2 || true
            stop_sglang
            exit 1
        fi
        echo "SGLang ready; regenerating -> $TRAIN_REGEN"

        python scripts/data/generate_train_data.py \
            --model "$TARGET" \
            --server-address "127.0.0.1:${SGLANG_PORT}" \
            --concurrency "$GEN_CONCURRENCY" \
            --temperature "$GEN_TEMPERATURE" \
            --top-p "$GEN_TOP_P" \
            --top-k "$GEN_TOP_K" \
            --min-p "$GEN_MIN_P" \
            --max-tokens "$GEN_MAX_TOKENS" \
            --disable-thinking \
            --resume \
            --input-file-path "$TRAIN_JSONL" \
            --output-file-path "$TRAIN_REGEN"

        stop_sglang
        echo "SGLang stopped; GPU released for the cache build."
    fi
fi

# --- Stage 3: target cache (idempotent on manifest.json) ---------------------
if should_run cache "${RUN_CACHE:-1}"; then
    current_stage="cache"
    if [[ "$FORCE" != "1" && -f "$CACHE_DIR/manifest.json" ]]; then
        banner "stage 3: target cache -- SKIP (found $CACHE_DIR/manifest.json; FORCE=1 to rebuild)"
    else
        banner "stage 3: build target cache from $CACHE_INPUT -> $CACHE_DIR"
        rm -rf "$CACHE_DIR"
        python scripts/data/prepare_target_cache.py \
            --config "$CONFIG" \
            --train-data-path "$CACHE_INPUT" \
            --output-dir "$CACHE_DIR" \
            --local-batch-size "$CACHE_LOCAL_BATCH_SIZE"
    fi
fi

# --- Stage 4: train (auto-resumes from step_latest) --------------------------
if should_run train "${RUN_TRAIN:-1}"; then
    current_stage="train"
    banner "stage 4: train draft (resumes from step_latest if present)"
    python train.py \
        --config "$CONFIG" \
        --opts "data.target_cache_path=$CACHE_DIR"
fi

# --- Stage 5: eval -----------------------------------------------------------
if should_run eval "${RUN_EVAL:-1}"; then
    current_stage="eval"
    banner "stage 5: acceptance eval"
    target_name_or_path="$TARGET" \
    draft_name_or_path="$DRAFT_CKPT" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        bash scripts/eval/eval.sh
fi

current_stage="done"
banner "pipeline complete"
