# DeepSpec-MLX — Project Status & Resume Guide

**One-liner:** a native-Apple-**MLX** reimplementation of DeepSpec's **DSpark** speculative-
decoding stack (train a draft model → run lossless spec-decode → serve it), living in
`deepspec_mlx/`. The PyTorch `deepspec/` package is the untouched **reference oracle**. The
machine is Apple Silicon (M-series), pure-MLX, **no PyTorch installed**.

This file is the durable project state. Start here when resuming. See `ARCHITECTURE.md` for
the design decisions and `README.md` for how to run everything.

> Note: `~/.claude/plans/*.md` are **per-task** design records (the current one is just the
> server plan), not a project overview — this file is the overview. A plan file named
> `humble-doodling-blanket.md` in that dir belongs to a **different, unrelated** project; ignore it.

---

## Where we are (as of this writing)

**Everything below is done and green on the `mlx` branch, EXCEPT M7 (never run).** The full
train → spec-decode-eval → serve pipeline works for both a stock Qwen3 target and a hybrid
Ornith/Qwen3.5 target. Recent work (Ornith M8b, the server, the uv switch) may be uncommitted —
run `git status` on first resume.

The headline result: **DSpark speculative decoding runs on Ornith-1.0-9B (a hybrid
linear+full-attention model) — something the PyTorch reference cannot eval at all** (its
`DynamicCache.crop` can't rewind the linear-attention recurrent state; the torch repo shipped a
train-only POC). Our **cache-free target verify** sidesteps that.

## Milestone ledger

| Milestone | State | Result / notes |
|---|---|---|
| **M0** env | ✅ | uv project (`pyproject.toml` + `uv.lock`); Python 3.12; mlx 0.31.2 / mlx-lm 0.31.3; `transformers==5.10.2` pin |
| **M1** KV-trim spike | ✅ | 4 throwaway spikes (`spikes/`) — proved mlx-lm cache trim is bit-exact (the project's original risk gate) |
| **M2** Muon | ✅ | Newton-Schulz + MuonAdam split; 5 parity tests vs numpy |
| **M3** data path | ✅ | v2 target-cache format reader/writer/generator; round-trip bit-exact |
| **M4** draft + loss | ✅ | DSpark draft (custom attention, markov, confidence) + loss; loss matches numpy to 1e-9 |
| **M5** train | ✅ | training loop (value_and_grad + grad-accum + Muon/AdamW + fp32 master) |
| **M6** eval | ✅ | spec-decode acceptance eval; **canary accept_len ≈ 1.47** (Qwen3-0.6B) |
| **R1.1–R3** remediation | ✅ | post-review fixes: precision scheme C, tie-weights generality, stop-branch metric fix, oracle metric aggregation, confidence early-exit, `target_layer_ids==-1`, new tests |
| **Step A** cache-free verify | ✅ | `CacheFreeTargetRunner`; validated == trim path on the canary (`test_cachefree_target.py`). Unblocks linear-attention targets |
| **M7** scale plain Qwen3 (4B/8B/14B) | ❌ **NOT RUN** | README §6 documents the *designed* path only. Full-attention → uses the fast trimmable cache. No blockers known; just never executed |
| **M8a** Ornith eval loop | ✅ | loaded the qwen3_5 text backbone via mlx_lm; hybrid capture **bit-exact** vs stock; cache-free spec-decode runs (random draft ≈ 1.0) |
| **M8b** Ornith trained draft | ✅ | trained with Muon; **honest HELD-OUT accept_len ≈ 1.23** (a memorized-prompt run showed 6–8 — that's an upper bound, not generalization). Modest only because the training set is tiny (24 samples) |
| **Server** | ✅ | generic OpenAI-compatible MLX server (`serve/`): draft checkpoint save/load, arch-aware runner factory, FastAPI `/v1/chat/completions` with token streaming. Verified |

## Tests (6, all green) — `deepspec_mlx/tests/`

`test_muon_parity`, `test_cache_reader_parity`, `test_dspark_forward`, `test_precision`,
`test_spec_decode`, `test_cachefree_target`. Run each with `python deepspec_mlx/tests/<name>.py`.

## Artifacts on disk (not in git)

- Target caches: `~/dspark_mlx/cache/{qwen3_0_6b_canary, ornith_9b_canary, ornith_9b_v2}`
- Draft checkpoints: `~/dspark_mlx/checkpoints/qwen3_0_6b` (a saved canary draft for the server)
- HF model cache: Qwen3-0.6B and Ornith-1.0-9B (~15 GB) are downloaded

---

## How to resume

1. `git status` on the `mlx` branch; commit anything outstanding.
2. `uv sync --project deepspec_mlx` (recreates the env from the lockfile).
3. Sanity: run the 6 tests (above). Then `python deepspec_mlx/scripts/eval_mlx.py --steps 40` should
   print canary accept_len > 1.
4. Read `ARCHITECTURE.md` for the design; `README.md` for the full command set.

## What's next (pick up here) — roughly by value

1. **A faithful Ornith number** — bigger/more-diverse cache (hundreds+ of prompts, longer),
   train longer, eval held-out → a real generalization `acceptance_length` (the ~1.23 is
   training-set-limited, not a ceiling). Driver: `scripts/eval_ornith.py --cache … --steps …`.
2. **Serve a trained Ornith draft** through the generic server as the cross-arch proof
   (`eval_ornith.py --save …` then `serve/server.py --draft …`) — the arch switch is built but
   this end-to-end hasn't been exercised.
3. **M7 — plain Qwen3 4B/8B/14B scale-up** (never run). Config-swap the target; full attention
   uses the fast cached runner. Watch disk (cache ≈ 30 KB/token).

## Deferred (documented non-goals for now)

- **bf16-vs-torch bit-parity** of the loss — needs fixtures generated from `deepspec/` on a torch
  machine (out-of-band; we're torch-free here). Precision scheme C's bf16 mode makes it achievable.
- **Gated/RNN markov heads** — only `vanilla` is ported (the config default).
- **Incremental draft KV-cache** — eval uses a provably-equivalent **cache-free** draft (O(n²));
  fine for the canary/Ornith, add the incremental cache if eval throughput matters.
- **Checkpoint size** — fp32 canary draft checkpoint is ~1.8 GB (Ornith ~5 GB) because frozen
  embed/lm_head are saved. Save in bf16, or drop the frozen heads and re-copy from the target at
  load, to roughly halve it.
