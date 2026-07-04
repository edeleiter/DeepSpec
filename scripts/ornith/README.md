# Training a DSpark draft for Ornith-1.0-9B (Qwen3.5)

Ornith-1.0-9B is a **Qwen3.5** model (hybrid linear/full attention, multimodal,
vocab 248320). This repo now has a `qwen3_5` DSpark adapter so it can be used as
a speculative-decoding **target**. The draft itself is a plain full-attention
transformer that consumes the target's cached hidden states — no Mamba/SSM porting.

**The GGUF Q4_K_M copy is not usable for training.** You need the bf16 HF
safetensors (~19GB). The GGUF is 4-bit lossy and lives in the llama.cpp runtime,
which can't be hooked for per-layer hidden states.

Run everything in a **Linux CUDA container** (Docker on the WSL2 backend). The
cache-build uses `sdpa` (portable), but draft training uses `flex_attention` +
`torch.compile` + triton, which are Linux/CUDA-first.

> **Shell:** the `docker run` line below is launched from **Git Bash** on
> Windows. Everything **inside** the container runs in the container's own Linux
> `bash` — those commands are plain Linux and need no special handling.

## What was added
- `deepspec/modeling/dspark/qwen3_5/` — DSpark draft adapter (config builder +
  model alias). The config builder extracts the flat text config, drops
  `vision_config`, and sanitizes partial/multiway RoPE to plain full RoPE.
- `Qwen35DSparkTrainer` in `deepspec/trainer/dspark_trainer.py` (+ export).
- `qwen3_5` branch in `scripts/data/prepare_target_cache.py` (`_get_target_backbone`).
- `config/dspark/dspark_ornith_9b.py` — MVP config (num_anchors=16, 3 target
  layers, compile off).
- `scripts/ornith/check_load.py` — Phase 0 smoke test / config+tree dump.
- `Dockerfile` + `.dockerignore` — image with the pinned deps baked in (no
  per-run reinstall).
- `scripts/ornith/run_pipeline.sh` — the image's default CMD; runs the whole
  pipeline (preflight → data → cache → train → eval), env-parameterized.

## Build the image (once)

Prereqs: Docker Desktop with the **WSL2 backend** and **GPU support** enabled
(Settings → Resources → WSL Integration, and an NVIDIA driver on Windows), and a
`.env` with your `HF_TOKEN` (`cp .env.example .env`, then edit).

Deps are baked into the image, so you build once and never `pip install` again.
The repo is bind-mounted at runtime, so **code edits do not require a rebuild** —
only a change to `requirements.txt` does.

```bash
cd /f/DeepSpec
MSYS_NO_PATHCONV=1 docker build -t deepspec-ornith:latest .
```

## Run the pipeline

`MSYS_NO_PATHCONV=1` stops Git Bash from rewriting container paths; `$(pwd -W)`
emits a Windows path Docker mounts cleanly. The three **named volumes** persist
across `--rm` runs: the ~19GB HF download, the target cache, and — importantly —
checkpoints (the trainer's `step_latest` symlink needs a real Linux fs, so
checkpoints must NOT live on the Windows bind mount).

```bash
MSYS_NO_PATHCONV=1 docker run --gpus all --rm -it \
  --shm-size=8g \
  --env-file .env \
  -v "$(pwd -W)":/workspace -w /workspace \
  -v deepspec-hf:/root/.cache/huggingface \
  -v deepspec-cache:/root/.cache/deepspec \
  -v deepspec-ckpt:/root/checkpoints \
  -e SAMPLE_SIZE=200 -e CUDA_VISIBLE_DEVICES=0 \
  deepspec-ornith:latest
```

`--shm-size=8g` is required: the training DataLoader shares large cached
hidden-state tensors through `/dev/shm`, and Docker's 64MB default makes workers
die with a bus error. Run a **single stage** with
`-e STAGE=<preflight|data|cache|train|eval>`, e.g.
`-e STAGE=cache`. Skip a stage in a full run with `-e RUN_PREFLIGHT=0` (once the
load gate is trusted, this saves a ~19GB reload). Rebuild the cache with
`-e FORCE=1`. See the header of `scripts/ornith/run_pipeline.sh` for the full
env-var surface (`TARGET`, `CONFIG`, `SAMPLE_SIZE`, `CACHE_DIR`, `DRAFT_CKPT`, …).

**Debug / interactive** — override the CMD with `bash`:

```bash
MSYS_NO_PATHCONV=1 docker run --gpus all --rm -it \
  --shm-size=8g \
  --env-file .env \
  -v "$(pwd -W)":/workspace -w /workspace \
  -v deepspec-hf:/root/.cache/huggingface \
  -v deepspec-cache:/root/.cache/deepspec \
  -v deepspec-ckpt:/root/checkpoints \
  deepspec-ornith:latest bash
```

Troubleshooting (Git Bash / Docker Desktop):
- `-w /workspace` becomes a `C:/Program Files/Git/...` path → you forgot
  `MSYS_NO_PATHCONV=1`.
- "invalid mode" / drive-not-shared → use `$(pwd -W)` (not `$PWD`) and share the
  `F:` drive with Docker Desktop.
- `--gpus all` errors → GPU support isn't enabled for the WSL2 backend / driver.
- `bash\r: No such file` on the entrypoint → CRLF line endings; `.gitattributes`
  forces LF, but re-checkout the repo if it was cloned before that was added.

The stages below (Phase 0/2/3/4) are what `run_pipeline.sh` automates — kept here
as reference for what each stage does and its gate, and for running them by hand
inside a `bash` container.

## Phase 0 — acquire weights + prove it loads (hardest gate)

```bash
hf download deepreinforce-ai/Ornith-1.0-9B
python scripts/ornith/check_load.py --model deepreinforce-ai/Ornith-1.0-9B
```

Read the output and confirm/adjust in `config/dspark/dspark_ornith_9b.py`:
- **BACKBONE PATH** — should be `language_model`; if not, fix the `qwen3_5`
  branch in `scripts/data/prepare_target_cache.py`.
- **full_attention layer indices** — confirm `target_layer_ids` are among them.
- **mask_token_id** — set to a valid, rarely-generated id from the suggestions.
- **RoPE fields** — confirm the config builder's assumptions (`partial_rotary_factor`,
  `rope_parameters`/`mrope_section`).

**GATE:** model loads on GPU and prints a 33-entry hidden-states tuple. If it
can't load even here → resolve the transformers/env issue before continuing.

## Phase 2 — tiny target cache (after a smoke train step; Phase 1 gate)

Build a small cache first (50–200 conversations, 2–3 layers). Point
`--train-data-path` at a small JSONL (see `scripts/data/README.md` for the data
pipeline) and `--output-dir` at scratch space:

```bash
export target_cache_dir="$HOME/.cache/deepspec/ornith_9b_target_cache"
python scripts/data/prepare_target_cache.py \
  --config config/dspark/dspark_ornith_9b.py \
  --train-data-path <path/to/small.jsonl> \
  --output-dir "$target_cache_dir" \
  --local-batch-size 2
```

**GATE:** cache builds, manifest validates, measured bytes/token ≈ prediction
(~48KB/token per layer). Watch storage before scaling the dataset.

## Phase 3 — MVP training

```bash
export CUDA_VISIBLE_DEVICES=0
export target_cache_dir="$HOME/.cache/deepspec/ornith_9b_target_cache"
python train.py \
  --config config/dspark/dspark_ornith_9b.py \
  --opts "data.target_cache_path=${target_cache_dir}"
```

**GATE:** loss decreases, no OOM. On OOM, lower `num_anchors`, `max_length`, or
the number of target layers.

## Phase 4 — acceptance eval

```bash
target_name_or_path=deepreinforce-ai/Ornith-1.0-9B \
draft_name_or_path="$HOME/checkpoints/deepspec/dspark_ornith_9b/step_latest" \
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval/eval.sh
```

**GATE:** mean accepted length > 1.0 on gsm8k. Eval does incremental decode with
`DynamicCache`; if Ornith's hybrid layers don't support incremental caching,
fix it or accept a train-only proof-of-concept and document the limitation.
