# DSpark Qwen3-4B — train, serve, and run natively in llama.cpp

Single-GPU project to train a **DSpark speculative-decoding draft** for the stock
**`Qwen/Qwen3-4B`** target, serve it to a coding agent, and add native DSpark
support to a llama.cpp fork. Developed on one **RTX 5070 Ti (Blackwell, `sm_120`,
16 GB)** under Docker/WSL2.

> This is the reference path. An earlier attempt targeted **Ornith-1.0-9B**
> (Qwen3.5, hybrid linear+full attention); its acceptance eval is architecturally
> blocked because DSpark's verifier rewinds a `DynamicCache` with `.crop()`, which
> linear-attention recurrent state can't support. Qwen3-4B is plain full attention,
> so every stage — including eval — works. See [`scripts/ornith/README.md`](scripts/ornith/README.md).

## The three tracks

| Track | What | Status |
| --- | --- | --- |
| **1 — Train** | Full DeepSpec pipeline → a DSpark draft + a real `accept_len`. | Runs on a 16 GB card (~82 min/560 steps) after the FSDP-single-GPU fix — see Gotchas. |
| **2a — Serve** | FastAPI OpenAI server so the **Pi** coding agent (or any OpenAI client) can drive the draft. | Ready (needs a checkpoint). |
| **2b — Native** | Fork `edeleiter/llama.cpp` to add DSpark as a first-class draft arch → GGUF / LM Studio. | Python converter done+verified; C++ remaining. See `DSPARK_PORT.md` in the fork (`F:\llama.cpp`). |

## Environment

Model-agnostic Docker image (cu128 torch for `sm_120`; the repo is bind-mounted,
so code edits need no rebuild). Build once:

```bash
MSYS_NO_PATHCONV=1 docker build -t deepspec:latest .
```

`.env` must hold your `HF_TOKEN` (`cp .env.example .env`). All runs go through the
model-agnostic launcher **`scripts/docker_run.sh`**, which sets `--gpus all`,
`--shm-size=8g` (required — else the DataLoader dies on a `/dev/shm` bus error),
and the persistent named volumes `deepspec-hf` / `deepspec-cache` / `deepspec-ckpt`
(checkpoints use symlinks that must not live on the Windows bind mount).

Launcher env: `PIPELINE` (which model's runner; default `scripts/qwen3_4b/run_pipeline.sh`),
`DETACH=1` (fire-and-forget background run, survives the shell closing),
`CONTAINER_NAME`, `IMAGE`, `SHM_SIZE`, `CMD='bash'` (interactive shell).

### Alternative runtime: WSL Containers (wslc, no Docker Desktop)

The same image/pipeline runs natively on **WSL Containers** (`wslc.exe`, WSL 2.9.3+) —
lighter than Docker Desktop (no always-on background service). Use the parallel
launcher **`scripts/wslc_run.sh`** (Docker-compatible flags; note wslc wants
`--shm-size 8G` uppercase). One-time setup:

```bash
WSLC="/c/Program Files/WSL/wslc.exe"   # or add C:\Program Files\WSL to PATH
"$WSLC" build -t deepspec:latest .
"$WSLC" volume create deepspec-hf; "$WSLC" volume create deepspec-ckpt
# full run, target cache on a roomy non-system drive (D:) to protect C:
CACHE_SRC="D:/deepspec-cache" DETACH=1 CONTAINER_NAME=deepspec-wslc \
  WSLC="$WSLC" bash scripts/wslc_run.sh -e SAMPLE_SIZE=4000 -e REGEN=0 \
  -e DSPARK_ADAM8BIT=1 -e EVAL_TASKS=gsm8k,humaneval -e EVAL_MAX_SAMPLES=50
```

Clean spin-down with **`scripts/wslc_stop.sh <name>`** — it MUST terminate the wslc
**session VM** (`vmmemwslc-cli-<user>`), which otherwise lingers after a container
is killed holding ~25 GB RAM and locking volume vhdx files (`wsl --shutdown` does
NOT touch it — this is the wslc equivalent of Docker's `vmmem`).

wslc gotchas:
- **Per-session isolation by elevation.** Containers/images/volumes live in a session
  keyed to the caller's integrity level. Run terminals **non-elevated** (an admin
  terminal — e.g. a Windows Terminal profile with `"elevate": true` — sees a separate
  empty `wslc-cli-admin-<user>` session). `wslc` must also be on PATH or the bare
  command is "not found" (looks like nothing exists).
- **`CACHE_SRC`** overrides the ~cache location. Default = C: SSD named volume (fast).
  A Windows path (`D:/deepspec-cache`) keeps the big cache off the system drive; the
  cache is symlink-free so a bind mount is safe. **Checkpoints stay on a named volume**
  (they use a `step_latest` symlink that a Windows bind mount can't hold).
- wslc stores images **and** volumes in the session's `storage.vhdx`; deleting that
  disk wipes both. To reclaim space, `volume remove` + `container prune`; a full
  reclaim = terminate session, delete `storage.vhdx` (re-download model + rebuild image).

## Track 1 — train

One command runs 5 stages: **data → regen (SGLang) → cache → train → eval**.

```bash
# Fast smoke test (proves the pipeline wires together; ~minutes):
bash scripts/docker_run.sh -e SAMPLE_SIZE=200 -e REGEN=0 \
     -e EVAL_TASKS=gsm8k -e EVAL_MAX_SAMPLES=20

# Full overnight run, detached (⚠️ ~180 GB cache at SAMPLE_SIZE=4000 — check disk):
DETACH=1 CONTAINER_NAME=deepspec-full bash scripts/docker_run.sh \
     -e SAMPLE_SIZE=4000 -e REGEN=1 \
     -e EVAL_TASKS=gsm8k,humaneval -e EVAL_MAX_SAMPLES=50
```

Pipeline env knobs (full list in the header of `scripts/qwen3_4b/run_pipeline.sh`):

| Env | Default | Meaning |
| --- | --- | --- |
| `SAMPLE_SIZE` | `4000` | prompts used. `200` = smoke. Cache ≈ `samples × ~1500 tok × ~30 KB`. |
| `REGEN` | `1` | regenerate answers with the target via SGLang (faithful recipe). `0` = raw split answers. |
| `STAGE` | `all` | run one stage: `data`/`regen`/`cache`/`train`/`eval`. |
| `RUN_EVAL` etc. | `1` | per-stage on/off toggles. |
| `FORCE` | `0` | `1` bypasses idempotency skips. |
| `EVAL_TASKS` / `EVAL_MAX_SAMPLES` | (all 9 / per-task caps) | narrow eval for fast iteration, e.g. `gsm8k,humaneval` / `50`. Read by `eval.py`. |

Config: **`config/dspark/dspark_qwen3_4b_trial.py`** — a faithful copy of the stock
`dspark_qwen3_4b.py` with only single-GPU knobs dialed (`num_anchors=256`,
`global_batch_size=64`, `torch_compile=False`; quality knobs left at stock).
Checkpoints → `~/checkpoints/deepspec/dspark_qwen3_4b_trial/step_*` (a final
`step_latest` is always written). **Deliverable: the `accept_len` column from eval.**

Monitor / collect a detached run:
```bash
docker logs -f deepspec-full                                  # follow; --tail 100 for the final table
docker run --rm -v deepspec-ckpt:/ckpt busybox ls -la /ckpt/deepspec/dspark_qwen3_4b_trial/
docker rm deepspec-full                                       # cleanup after grabbing logs
```

## Track 2a — serve to the Pi coding agent

The draft is target-coupled (needs the target's hidden states each step), so it
runs behind the DeepSpec Python stack. Start the OpenAI-compatible server:

```bash
TARGET=Qwen/Qwen3-4B \
DRAFT=$HOME/checkpoints/deepspec/dspark_qwen3_4b_trial/step_latest \
python scripts/serve/dspark_openai_server.py --host 0.0.0.0 --port 8000
```

Point **Pi** at it: copy `scripts/serve/pi_models.json` into `~/.pi/agent/models.json`,
then select `dspark-qwen3-4b`. Notes: the draft only changes **speed**, never output
(spec decoding is lossless → identical to plain Qwen3-4B); batch-size-1, single
request; `--default-temperature 0` gives deterministic greedy (used to validate
losslessness, including against the llama.cpp port). Full runbook:
[`scripts/qwen3_4b/README.md`](scripts/qwen3_4b/README.md).

## Track 2b — native llama.cpp

DSpark inference ≈ **DFlash + a Markov-head logit bias**, and the fork already ships
DFlash, so it's a scoped port. The **Python HF→GGUF converter is done and verified**;
the C++ (arch registration, `src/models/dspark.cpp`, speculative driver) remains and
is gated on a trained checkpoint + a compile loop. Details, file list, and validation
gates: **`DSPARK_PORT.md`** in the fork (`F:\llama.cpp`, or `edeleiter/llama.cpp`).

## Files added by this project

| Path | Purpose |
| --- | --- |
| `config/dspark/dspark_qwen3_4b_trial.py` | single-GPU trial config |
| `scripts/qwen3_4b/run_pipeline.sh` | 5-stage pipeline (with SGLang regen) |
| `scripts/qwen3_4b/README.md` | trial + serving runbook |
| `scripts/serve/dspark_openai_server.py` | OpenAI-compatible server (reuses the eval loop) |
| `scripts/serve/pi_models.json` | Pi provider config |
| `scripts/docker_run.sh` | model-agnostic Docker launcher (`PIPELINE`/`DETACH`/…) |
| `Dockerfile`, `.dockerignore` | cu128 (`sm_120`) image; repo bind-mounted |
| `eval.py` (mod) | `EVAL_TASKS` / `EVAL_MAX_SAMPLES` env knobs |
| (Ornith enablement) | `config/dspark/dspark_ornith_9b.py`, `scripts/ornith/`, `deepspec/modeling/dspark/qwen3_5/`, `Qwen35DSpark` trainer/evaluator |

## Gotchas

- **Fitting training on a 16 GB card (the load-bearing fix).** The single-GPU run
  first appeared to "hang" — it was actually **spilling VRAM to host RAM** (WSL2's
  driver silently overflows over PCIe → ~60 s/micro-batch at ~61 W). Root cause was
  **FSDP running on one GPU**: it flattens all params and re-cats a full-size gradient
  buffer each step (~5.6 GB of pure overhead, zero benefit at `world_size==1`).
  `base_trainer` now **skips the FSDP wrap when `world_size==1`**. Combined with the
  draft attention on **`sdpa`** (not eager `flex_attention`), a **loss softmax dedup**,
  and **8-bit Adam** (`DSPARK_ADAM8BIT=1`, bitsandbytes, verified on sm_120), the
  per-step peak drops 13.7 → ~11 GB and the full 560-step run finishes in **~82 min**,
  no spill. Keep `num_anchors=128` — raising it re-inflates the vocab-loss tensors
  and can push back over the 16 GB line. (Diagnosed with a `torch.cuda.memory`
  allocation snapshot; guessing knobs never moved the peak.)
- **`REGEN=1` needs SGLang, which is NOT in the image** (it's a heavy separate
  install that pins its own torch/flashinfer). The regen stage aborts with a clear
  message if `sglang` is missing. For now use **`REGEN=0`** (train on the raw split
  answers — off-policy, so a slightly lower `accept_len`, but fully runnable). Baking
  SGLang into the `sm_120`/cu128 image for the faithful on-policy run needs testing,
  not a blind `pip install`.
- **Eval runs the full 9-benchmark suite by default** (thousands of samples at
  spec-decode speed = hours) regardless of `SAMPLE_SIZE`. Use `EVAL_TASKS` /
  `EVAL_MAX_SAMPLES`, or `-e RUN_EVAL=0` and run eval separately.
- **Disk**: `SAMPLE_SIZE=4000` ≈ ~180 GB target cache. Lower it, or drop
  `target_layer_ids` to 3 layers, if short on space.
- **`target_layer_ids` must exclude the final target layer** (35 for Qwen3-4B) —
  `assert_no_final_target_layer` in `deepspec/eval/base_evaluator.py`. The stock
  `[1,9,17,25,33]` satisfies this.
- **cu128 wheels are mandatory** on `sm_120` (cu126 → "no kernel image available").
- The DSpark draft **cannot** run in stock LM Studio/llama.cpp/Unsloth-GGUF — only
  via the fork (Track 2b) or the Python server (Track 2a).
