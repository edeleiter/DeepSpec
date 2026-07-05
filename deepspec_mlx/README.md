# DeepSpec-MLX — DSpark speculative decoding, native on Apple Silicon

A from-scratch reimplementation of DeepSpec's **DSpark** speculative-decoding draft
model — training, the Muon optimizer, and the acceptance-length eval — in Apple's
**MLX**. Runs entirely on the Mac GPU: **no NVIDIA, no Docker, no WSL, no PyTorch**.

The original PyTorch package (`../deepspec/`) is untouched and serves as the reference
"oracle" this port is validated against. This package is `deepspec_mlx/`.

> **Resuming / new here?** Read **`STATUS.md`** (project state + milestone ledger + what's next)
> and **`ARCHITECTURE.md`** (the design decisions) first. This README is the how-to-run.
> It also runs on a hybrid target — see **§5c (Ornith)** for the marquee result.

> **Canary target:** `Qwen/Qwen3-0.6B`. Everything below runs in ~1–2 minutes total on an
> M-series Mac. Scaling to 1.7B/4B/8B/14B is a config swap (see "Scaling up").

---

## 0. What you need

- An **Apple Silicon** Mac (M1–M5), macOS.
- **[uv](https://github.com/astral-sh/uv)** (you have it) — manages Python + packages.
- ~2 GB disk for the Qwen3-0.6B weights + a tiny cache. (128 GB RAM is plenty; the
  canary uses <5 GB.)
- Internet for the first run (downloads Qwen3-0.6B from Hugging Face).

Run everything from the repo root: `/Users/edele/Documents/ws/DeepSpec`.

---

## 1. One-time setup (uv)

This is a `uv` project (`deepspec_mlx/pyproject.toml` + `uv.lock`). One command creates the
environment (uv fetches Python 3.12 and all deps from the lockfile):

```bash
# from the repo root
uv sync --project deepspec_mlx
```

**Running commands.** Every `python …` command in this README runs under the uv env. Two ways:

```bash
# (a) activate once, then use `python` directly (what the commands below assume):
source deepspec_mlx/.venv/bin/activate

# (b) or, without activating, prefix any command with:
uv run --project deepspec_mlx python deepspec_mlx/scripts/eval_mlx.py ...
```

Scripts are working-directory-independent (imports + data paths resolve to the repo root), so
you can run from anywhere. To add/upgrade a dependency: edit `pyproject.toml` then `uv sync`.

Notes:
- `pyproject.toml` **pins `transformers==5.10.2`** on purpose — 5.13 breaks mlx-lm's tokenizer
  registration. Don't bump it without testing.
- Qwen3-0.6B is a **public** model, so no token is required. To avoid HF rate-limit warnings /
  get faster downloads, optionally `export HF_TOKEN=hf_...` (there's one in the repo's `.env`,
  but it is **not** auto-loaded — export it yourself if you want it).

---

## 2. The pipeline (soup to nuts)

Three stages: **generate a target cache → train the draft → measure acceptance length.**

### Stage 1 — Generate the target cache

Runs Qwen3-0.6B over a few prompts and captures the per-layer hidden states the draft
learns from. (First run downloads the model, ~1.2 GB.)

```bash
python deepspec_mlx/scripts/prepare_target_cache_mlx.py \
    --out ~/dspark_mlx/cache/qwen3_0_6b_canary \
    --num 8 --max-length 96
```

- Writes an ~7 MB cache (`shard-00000.bin`, `samples.idx`, `manifest.json`) in the exact
  on-disk format the torch reference uses.
- Knobs: `--num` (prompt count), `--max-length` (tokens/sample), `--layers`
  (`target_layer_ids`, default `1,6,13,20,26`), `--jsonl` (prompt source; defaults to
  `eval_datasets/gsm8k.jsonl`). Scale these up for a bigger cache (watch disk).
- **Rerun?** Delete the dir first: `rm -rf ~/dspark_mlx/cache/qwen3_0_6b_canary`.

### Stage 2 — Train (overfit the canary)

Trains the DSpark draft on that cache and watches it learn.

```bash
python deepspec_mlx/scripts/overfit_canary.py --steps 40
```

Expected output (the draft is learning the target's next-token distribution):

```
step   0 (init): ce=12.6  accept_rate(mean)=0.005  pos0=0.010
step  40:        ce= 1.0  accept_rate(mean)=0.53   pos0=0.58
RESULT: PASS — draft is learning (ce down, accept_rate up)
```

- Knobs: `--steps`, `--lr`, `--num-anchors`, `--num-draft-layers`, and
  **`--dtype {fp32,bf16}`** (see "Precision modes"). Default `fp32`.
- Uses the Muon + AdamW split optimizer with an fp32 master (bf16-safe).

### Stage 3 — Eval (the deliverable: acceptance length)

Trains a fresh draft, then measures **speculative-decoding acceptance length** — the
native-MLX equivalent of the paper's Table-1 metric — vs. a random-init baseline.

```bash
python deepspec_mlx/scripts/eval_mlx.py --steps 40 --max-new-tokens 32 --n-prompts 5
```

Expected output:

```
RANDOM-init draft: acceptance_length=1.000  verify_rate=0.125  draft_tokens/proposal=7.00
                   accept_rate@k = [0.00,0.00,0.00,0.00,0.00,0.00,0.00]
TRAINED draft    : acceptance_length=1.47   verify_rate=0.18   draft_tokens/proposal=7.00
                   accept_rate@k = [0.30,0.10,0.06,0.00,...]
RESULT: PASS — DSpark accept_len > 1 natively in MLX
```

`acceptance_length > 1` means the draft is accelerating decoding (1.0 = no speedup).
The canary number is a **floor** — it's gated by the tiny training set, not the port.
Metrics: `acceptance_length = accept_sum/proposal_count`, `verify_rate`,
`draft_tokens_per_proposal`, and per-position `accept_rate@k` (matches the oracle).

---

## 3. Run the tests

All torch-free; validate the port against numpy references and self-consistency.

```bash
for t in test_muon_parity test_cache_reader_parity test_dspark_forward \
         test_precision test_spec_decode test_cachefree_target; do
  echo "== $t =="; python deepspec_mlx/tests/$t.py; done
```

| Test | Checks |
|---|---|
| `test_muon_parity` | Newton-Schulz orthogonalizes; Muon step matches an fp32 numpy replica; MuonAdam split routing |
| `test_cache_reader_parity` | cache write→read round-trips bit-exact (incl. bf16) |
| `test_dspark_forward` | draft forward shapes, attention-bias masking, loss == numpy to 1e-9, gradients flow |
| `test_precision` | the real bf16-heads path in both compute modes + the dtype guardrail |
| `test_spec_decode` | spec-decode stop-token termination + metric invariants |
| `test_cachefree_target` | cache-free verify == the trim oracle on the canary (underpins the Ornith path) |

---

## 4. Precision modes (`--dtype`)

- **`fp32`** (default) — maximally robust/deterministic; best for the canary and debugging.
- **`bf16`** — forward/backward in bf16 with an **fp32 master** (matches the torch oracle,
  half the memory, faster; needed for scaling). Learning is identical to fp32 on the canary.

The model enforces **one** compute dtype (`assert_uniform_dtype`), so there's no accidental
mixed precision. Example: `python deepspec_mlx/scripts/overfit_canary.py --steps 40 --dtype bf16`.

---

## 5. The M1 de-risk spikes (optional / historical)

`deepspec_mlx/spikes/` holds the throwaway probes that de-risked the port (KV-cache
rewind, hidden capture, masked SDPA, draft cache). They still run and are a good way to
sanity-check mlx-lm on your machine:

```bash
python deepspec_mlx/spikes/m1_target_kv_trim.py      # KV-cache trim is bit-exact
python deepspec_mlx/spikes/m2_target_hidden_capture.py
python deepspec_mlx/spikes/m3_sdpa_dense_bias.py     # masked SDPA perf + the -1e9 sentinel
python deepspec_mlx/spikes/m4_draft_cache_trim.py
```

---

## 5b. Serve a trained draft (OpenAI-compatible)

Save a trained draft, then serve the DSpark-accelerated target over `/v1/chat/completions`.
Generic: the checkpoint self-describes its target + arch, so the same server handles a plain
Qwen3 draft (cached/trim runner) or an Ornith/qwen3_5 draft (cache-free runner) with no edits.

```bash
# 1) train + save a draft checkpoint (weights.safetensors + draft.json)
python deepspec_mlx/scripts/overfit_canary.py --steps 40 --save ~/dspark_mlx/checkpoints/qwen3_0_6b
#    (Ornith: eval_ornith.py --cache ... --save ~/dspark_mlx/checkpoints/ornith_9b)

# 2) launch the server
python deepspec_mlx/serve/server.py --draft ~/dspark_mlx/checkpoints/qwen3_0_6b --port 8000

# 3) drive it from any OpenAI-compatible client
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":64}'
# streaming:
curl -sN localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to five."}],"stream":true}'
```

Notes: batch-1, single in-flight request (a lock). Spec-decode is **lossless** — the draft only
changes speed, never output. This is a *server*: point an OpenAI-compatible **client** at it
(Open WebUI, Chatbox, curl, coding agents). It is **not** LM Studio-loadable — the DSpark draft
is a target-coupled arch, not a standalone GGUF/MLX model.

## 5c. Ornith (Qwen3.5, hybrid) — the marquee result

Ornith-1.0-9B (`model_type: qwen3_5`) is a **hybrid** model: ~3/4 of its layers are
linear-attention GatedDeltaNet whose recurrent state **cannot be rewound**. The PyTorch
reference **cannot run spec-decode eval on it** (its `DynamicCache.crop` can't trim linear
state — the torch repo shipped a train-only POC). Our **cache-free target verify**
(§ARCHITECTURE, validated on the canary) makes it work. `scripts/eval_ornith.py` is the driver.

```bash
# generate an Ornith target cache (hybrid capture), train + eval on a HELD-OUT prompt:
python deepspec_mlx/scripts/prepare_target_cache_mlx.py --arch qwen3_5 \
    --model deepreinforce-ai/Ornith-1.0-9B --out ~/dspark_mlx/cache/ornith_9b \
    --num 24 --max-length 96 --layers 7,15,23
python deepspec_mlx/scripts/eval_ornith.py --cache ~/dspark_mlx/cache/ornith_9b --steps 120
```

Two gates it checks: (1) the hybrid capture reproduces the stock logits **bit-exact**; (2) the
spec-decode loop runs via cache-free verify. With training it reports the honest **held-out
`acceptance_length` (~1.23 on a tiny 24-sample run)** — a real generalization number, modest only
because the training set is tiny. (A same-prompt run shows 6–8; that's a *memorized upper bound*,
not generalization.) First run downloads ~15 GB. See `STATUS.md` for the faithful-number next step.

## 6. Scaling up plain Qwen3 — M7 (DESIGNED, **not yet run**)

> ⚠️ This path is **untested aspiration** — only Ornith 9B (§5c) has actually been scaled to.
> The commands below should work (plain Qwen3 is full-attention → the fast trimmable cache), but
> they have not been executed. Treat as a starting point, not a validated recipe.

Swap the target — the draft dims are read from the target's config automatically:

```bash
python deepspec_mlx/scripts/prepare_target_cache_mlx.py --model Qwen/Qwen3-1.7B \
    --out ~/dspark_mlx/cache/qwen3_1_7b --num 200 --max-length 512 --layers 1,7,13,19,25
python deepspec_mlx/scripts/overfit_canary.py --model Qwen/Qwen3-1.7B \
    --cache ~/dspark_mlx/cache/qwen3_1_7b --dtype bf16 --steps 300
```

Watch for:
- **Disk** — the cache is ~30 KB/token; keep `--num`/`--max-length` sane (4B ≈ big).
- **`--dtype bf16`** at scale (memory + speed).
- **Untied targets (4B+)** are handled — the draft copies the target's real `lm_head`
  (tie-aware), not the embedding.
- `--layers` must exclude the target's final layer; `-1` = embedding output is supported.

---

## 7. Layout

```
deepspec_mlx/
  optim/          Muon (Newton-Schulz) + the MuonAdam split (MultiOptimizer) + cosine schedule
  data/           v2 target-cache format, reader, writer
  modeling/       qwen3_target_capture + qwen3_5_target_capture (instrumented targets), config,
                  dspark_common (attention bias / anchors / gathers), markov_head, dspark_qwen3
                  (draft, incl. eval backbone_block), loss
  trainer/        train_loop: value_and_grad + grad-accum + fp32 master (scheme C)
  eval/           spec_decode: the acceptance loop + TargetRunner (trim) + CacheFreeTargetRunner
  serve/          checkpoint (save/load) + server (FastAPI OpenAI-compatible, arch-aware)
  scripts/        prepare_target_cache_mlx, overfit_canary, eval_mlx, eval_ornith
  tests/          torch-free parity + self-consistency tests (6)
  spikes/         M1 de-risk probes (historical)
STATUS.md         project state + milestone ledger + how to resume  <- start here
ARCHITECTURE.md   the four load-bearing design decisions
```

## 8. Known deferred items

See `STATUS.md` for the full list + "what's next". In brief:
- **M7 plain-Qwen3 4B/8B/14B scale-up** — designed (§6) but **never run**; only Ornith 9B was scaled.
- **bf16 bit-parity vs the torch oracle** — needs fixtures generated from `../deepspec/`
  on a torch machine (out-of-band by design; unlocked by the bf16 mode).
- **Gated/RNN markov heads** — only `vanilla` is ported (the canary config's choice).
- **Incremental draft KV-cache** — eval uses a provably-equivalent cache-free draft
  (O(n²), fine for the canary; add the incremental cache if eval throughput matters).
