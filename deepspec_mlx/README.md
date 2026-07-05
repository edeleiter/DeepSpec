# DeepSpec-MLX — DSpark speculative decoding, native on Apple Silicon

A from-scratch reimplementation of DeepSpec's **DSpark** speculative-decoding draft
model — training, the Muon optimizer, and the acceptance-length eval — in Apple's
**MLX**. Runs entirely on the Mac GPU: **no NVIDIA, no Docker, no WSL, no PyTorch**.

The original PyTorch package (`../deepspec/`) is untouched and serves as the reference
"oracle" this port is validated against. This package is `deepspec_mlx/`.

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

## 1. One-time setup

```bash
# from the repo root
uv venv --python 3.12 deepspec_mlx/.venv          # uv fetches Python 3.12 for you
source deepspec_mlx/.venv/bin/activate            # activate (all commands below assume this)
uv pip install -r deepspec_mlx/requirements.txt   # mlx, mlx-lm, numpy, safetensors, hf-hub
```

Notes:
- `requirements.txt` **pins `transformers==5.10.2`** on purpose — 5.13 breaks mlx-lm's
  tokenizer registration. Don't bump it without testing.
- Qwen3-0.6B is a **public** model, so no token is required. To avoid HF rate-limit
  warnings / get faster downloads, optionally `export HF_TOKEN=hf_...` (there's one in
  the repo's `.env`, but it is **not** auto-loaded — export it yourself if you want it).
- If you'd rather not activate the venv, prefix each command with
  `deepspec_mlx/.venv/bin/python` instead of `python`.

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
         test_precision test_spec_decode; do
  echo "== $t =="; python deepspec_mlx/tests/$t.py; done
```

| Test | Checks |
|---|---|
| `test_muon_parity` | Newton-Schulz orthogonalizes; Muon step matches an fp32 numpy replica; MuonAdam split routing |
| `test_cache_reader_parity` | cache write→read round-trips bit-exact (incl. bf16) |
| `test_dspark_forward` | draft forward shapes, attention-bias masking, loss == numpy to 1e-9, gradients flow |
| `test_precision` | the real bf16-heads path in both compute modes + the dtype guardrail |
| `test_spec_decode` | spec-decode stop-token termination + metric invariants |

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

## 6. Scaling up (M7)

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
  modeling/       qwen3_target_capture (instrumented target), config, dspark_common
                  (attention bias / anchors / gathers), markov_head, dspark_qwen3 (draft), loss
  trainer/        train_loop: value_and_grad + grad-accum + fp32 master (scheme C)
  eval/           spec_decode: the acceptance-length loop (+ confidence early-exit)
  scripts/        prepare_target_cache_mlx, overfit_canary, eval_mlx
  tests/          torch-free parity + self-consistency tests
  spikes/         M1 de-risk probes (historical)
```

## 8. Known deferred items

- **bf16 bit-parity vs the torch oracle** — needs fixtures generated from `../deepspec/`
  on a torch machine (out-of-band by design; unlocked by the bf16 mode).
- **Gated/RNN markov heads** — only `vanilla` is ported (the canary config's choice).
- **Incremental draft KV-cache** — eval uses a provably-equivalent cache-free draft
  (O(n²), fine for the canary; add the incremental cache if eval throughput matters).
