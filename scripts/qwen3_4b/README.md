# DSpark Qwen3-4B — single-GPU trial + serving

End-to-end recipe for training a DSpark speculative-decoding draft against the
stock **`Qwen/Qwen3-4B`** target on one consumer GPU (developed for a 16 GB
RTX 5070 Ti / Blackwell, CUDA `sm_120`), then serving it to a coding agent.

Unlike the Ornith target (Qwen3.5, hybrid linear attention — eval is
architecturally blocked), Qwen3-4B is plain full attention, so **every stage,
including acceptance eval, runs**. This is the reference path.

## 0. Environment

Run inside the DeepSpec CUDA (devel) Docker image on WSL2: `cu128` torch
(`2.9.1`) + matching torchvision, `pip install -r requirements.txt`, and
`pip install "sglang[all]"` for the regen stage.

The image is model-agnostic (the repo is bind-mounted). Build once and launch
with the generic wrapper — it sets `--gpus all --shm-size=8g` and the persistent
named volumes for you:

```bash
docker build -t deepspec:latest .
# (or reuse an existing image: docker tag deepspec-ornith:latest deepspec:latest)

bash scripts/docker_run.sh -e SAMPLE_SIZE=200 -e REGEN=0    # smoke test (qwen3_4b is the default PIPELINE)
CMD='bash' bash scripts/docker_run.sh                        # interactive shell
```

To run a different model, point `PIPELINE` at its runner:
`PIPELINE=scripts/ornith/run_pipeline.sh bash scripts/docker_run.sh`.

## 1. Train (5 stages, one command)

```bash
bash scripts/qwen3_4b/run_pipeline.sh
```

Stages: **data → regen (SGLang) → cache → train → eval**. All knobs are env
vars (see the header of `run_pipeline.sh`). Key ones:

| Env | Default | Meaning |
| --- | --- | --- |
| `SAMPLE_SIZE` | `4000` | prompts used. `200` = fast smoke test. Cache ≈ `samples × ~1500 tok × ~30 KB`. |
| `REGEN` | `1` | regenerate answers with the target (the faithful recipe). `0` = train on raw split answers (fast). |
| `STAGE` | `all` | run one stage: `data`/`regen`/`cache`/`train`/`eval`. |
| `FORCE` | `0` | `1` bypasses idempotency skips. |
| `CACHE_LOCAL_BATCH_SIZE` | `2` | cache-build GPU-memory knob. |

⚠️ **Check free disk before the cache stage.** `SAMPLE_SIZE=4000` ≈ ~180 GB.
Lower `SAMPLE_SIZE`, or reduce `target_layer_ids` to 3 layers in the config, if
short on space.

Smoke-test first:

```bash
SAMPLE_SIZE=200 REGEN=0 bash scripts/qwen3_4b/run_pipeline.sh   # proves the pipeline wires together
```

Config: `config/dspark/dspark_qwen3_4b_trial.py` (faithful copy of the stock
`dspark_qwen3_4b.py`; only single-GPU knobs dialed down — `num_anchors=256`,
`global_batch_size=64`, `torch_compile=False`). Checkpoints land in
`~/checkpoints/deepspec/dspark_qwen3_4b_trial/step_*` (a final `step_latest` is
always written). The deliverable is the **`accept_len`** column printed by eval.

> Eval expects the 9 benchmark JSONLs under `eval_datasets/`. For a quick first
> pass, trim `TASKS` in `eval.py` to a subset (e.g. `gsm8k`, `humaneval`) or run
> `eval_datasets/convert_eval_datasets_to_jsonl.py` first.

## 2. Serve it to the Pi coding agent (Track 2a)

The trained draft is target-coupled — it can't run in LM Studio/llama.cpp yet
(that's Track 2b). To use it now, run the Python OpenAI-compatible server:

```bash
TARGET=Qwen/Qwen3-4B \
DRAFT=$HOME/checkpoints/deepspec/dspark_qwen3_4b_trial/step_latest \
python scripts/serve/dspark_openai_server.py --host 0.0.0.0 --port 8000
```

Smoke-check it:

```bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"dspark-qwen3-4b","messages":[{"role":"user","content":"Reverse a string in Python."}],"max_tokens":128}'
```

Point **Pi** at it: copy `scripts/serve/pi_models.json` into
`~/.pi/agent/models.json` (merge if you already have one), then run Pi and pick
`dspark-qwen3-4b`.

Notes:
- The draft only changes **speed**, never output — responses are token-identical
  to plain Qwen3-4B. Qwen3-4B is a small coding brain; this is a latency demo.
- Server is batch-size-1, single request. Fine for one Pi session.
- Set `--default-temperature 0` to get deterministic greedy output (used to
  validate losslessness against plain-target generation, and against the
  llama.cpp port in Track 2b).

## 3. Native llama.cpp / LM Studio (Track 2b)

Adding DSpark as a first-class draft arch to `edeleiter/llama.cpp` (by analogy
to the already-shipped DFlash) — see the plan file and the architect design doc.
That removes the Python dependency and unlocks the GGUF ecosystem. Begins once
Track 1 produces a checkpoint that passes the losslessness check in step 2.
