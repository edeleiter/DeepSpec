# DeepSpec-MLX — Architecture & Key Decisions

Context for someone who didn't build this. Pairs with `STATUS.md` (state) and `README.md`
(how-to). The PyTorch `deepspec/` package is the reference oracle; this documents where the
MLX port deliberately differs and why.

## The shape of the thing

DSpark is **speculative decoding with a target-coupled draft**. The draft is *not* a standalone
LLM — it consumes the frozen target's hidden states each step and proposes a block of tokens; the
target verifies them in one forward; rejection sampling accepts the longest matching prefix.
Output is **lossless** (identical to plain target); the draft only adds speed (`acceptance_length`).

Pipeline: **target-cache generation** (run the target, capture per-layer hidden states) →
**train** the draft against the cache → **spec-decode eval** (`acceptance_length`) → **serve**.
Data path components:
- `data/` — the v2 on-disk target-cache format (byte-compatible with the torch reference).
- `modeling/` — `qwen3_target_capture` (instrument the target), `dspark_qwen3` (the draft:
  custom cross/block attention + markov + confidence heads), `loss`, `config`.
- `trainer/train_loop` — value_and_grad + gradient accumulation + the Muon/AdamW split optimizer.
- `eval/spec_decode` — the verify/accept loop and the two target runners.
- `serve/` — the OpenAI-compatible server.

## The four load-bearing decisions

### 1. Cache-free target verify (the headline enabler)
`eval/spec_decode.py` has two target runners with the same `forward/trim/offset` contract:
- **`TargetRunner`** — keeps an mlx-lm KV cache and *rewinds* it with `trim_prompt_cache` after
  each verify (mirrors the torch `DynamicCache.crop`). Fast; **full-attention targets only**.
- **`CacheFreeTargetRunner`** — keeps NO persistent cache; each verify recomputes the target over
  the full committed prefix + proposed block with `cache=None`. O(n²) but **correct for any
  attention type**, including linear/recurrent layers whose state can't be rewound.

**Why it exists:** Ornith/Qwen3.5 is hybrid — ~3/4 of its layers are linear-attention
GatedDeltaNet with a recurrent state. That state cannot be trimmed per-position, so the trim-based
verify (and the entire PyTorch reference eval) is architecturally impossible on it — the torch repo
shipped a *train-only* POC. Cache-free verify sidesteps rewinding entirely, which is what makes
Ornith spec-decode possible here. Both runners share one forward body (`_capture_target_forward`),
and cache-free was validated to produce identical acceptance to the trim path on full-attention
Qwen3 (`tests/test_cachefree_target.py`) — so it's a *trusted* tool, not a guess.

The **draft** side is cache-free too (`Qwen3DSparkModel.backbone_block`): a single draft block at
anchor=start attends to all context (< anchor) + its own block with no mask = plain full attention,
provably equal to the cached incremental draft.

### 2. Precision "scheme C" (fp32 master + selectable compute dtype)
`trainer/train_loop.py` + `modeling/dspark_qwen3.py`. The model has one `compute_dtype` (fp32
default; bf16 for scale/oracle-parity). The optimizer keeps an **fp32 master** copy of the trainable
params; the forward/backward run in `compute_dtype`; grads cast to fp32 for the update; master casts
back into the model. When `compute_dtype==fp32` the casts are no-ops (clean fp32). `assert_uniform_
dtype()` is a guardrail that makes an accidental mixed-precision model impossible — this fixed the
#1 review finding (an untested bf16-heads-in-fp32-body hybrid).

### 3. Tie-weights generality
`Qwen3DSparkModel.initialize_from_target(embed_weight, lm_head_weight)` takes the two frozen heads
**separately** (no shared-array aliasing) and casts both to the model dtype. This is correct for
*tied* targets (Qwen3-0.6B/1.7B: pass embed as the head) and *untied* ones (Ornith 9B, 4B+: pass
the real `lm_head`). The tie-aware helper is `modeling.target_embed_and_head` (Qwen3) /
`lm.lm_head.weight` (Ornith). Without this, an untied target's draft head would silently become the
embedding matrix.

### 4. Arch registry / runner + capture factory (genericity)
Two arch-aware switches, both tiny and the only per-arch code:
- **Capture module** per target arch: `modeling/qwen3_target_capture.py` (plain Qwen3, full
  attention) vs `modeling/qwen3_5_target_capture.py` (Ornith hybrid — replicates the fa/ssm mask
  split, validated bit-exact vs the stock forward). Same interface both sides.
- **Runner factory** `serve/server.py:build_target_runner(model, meta)`: `arch=="qwen3"` → cached
  `TargetRunner`; `arch=="qwen3_5"` → `CacheFreeTargetRunner(capture_fn=<qwen3_5 capture>)`.

The server is generic because a **draft checkpoint self-describes** its target + arch: `save_draft`
writes `weights.safetensors` + `draft.json` (the full config + `target_id`, `arch`, `compute_dtype`,
`model_id`); `load_draft` reconstructs the draft with no target needed; the server reads `arch` to
pick the runner. Adding a new target family = one new capture module + one branch in the factory.

## Notable non-obvious details

- **Attention mask fill is `-1e9`, not `finfo.min`** (`dspark_common.py`) — MLX's SDPA NaNs on
  fully-masked rows with `finfo.min`; the finite sentinel gives a uniform (discarded) row.
- **RoPE for the draft** uses explicit cos/sin tables over non-contiguous (anchor+offset) draft
  positions, with q using the last q_len slots and k all slots. Partial-rotary targets (Ornith,
  `partial_rotary_factor=0.25`) need **no draft change** — the draft never sees target positions,
  only target *hidden states* via `fc`.
- **The target cache stores raw pre-final-norm layer outputs**; `target_layer_ids` must exclude the
  target's final layer (`-1` = embedding output is supported). Ornith uses `[7,15,23]` (a subset of
  its full-attention layers `{3,7,…,31}`).
- **Determinism caveat:** MLX GPU reductions are slightly non-deterministic run-to-run, and cached
  vs cache-free verify differ by bf16 kernel noise — so acceptance *token sequences* aren't a
  reliable invariant, but *acceptance-length* is (see `test_cachefree_target.py`).
