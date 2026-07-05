"""Generic OpenAI-compatible server for a trained DSpark draft + its target (MLX).

Serves the DSpark-accelerated target over /v1/chat/completions so any OpenAI client
(Open WebUI, Chatbox, curl, coding agents) can drive it. Spec-decode is lossless — the
draft changes speed, never output. Generic: point --draft at any saved checkpoint; its
draft.json self-describes the target + arch (qwen3 -> cached/trim runner; qwen3_5/Ornith
-> cache-free runner). Batch-1, single in-flight request (a lock), like the torch server.

Run:
    python deepspec_mlx/serve/server.py --draft ~/dspark_mlx/checkpoints/qwen3_0_6b --port 8000
"""

from __future__ import annotations

import argparse
import json as _json
import os
import queue
import sys
import threading
import time
import uuid

sys.path.insert(0, __file__.rsplit("/deepspec_mlx/", 1)[0])
import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from deepspec_mlx.eval import TargetRunner, CacheFreeTargetRunner, generate
from deepspec_mlx.serve.checkpoint import load_draft


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False


def build_target_runner(target_model, meta):
    """Arch-aware runner factory — the only place a new target arch adds a branch."""
    tli = meta["target_layer_ids"]
    if meta["arch"] == "qwen3_5":
        from deepspec_mlx.modeling.qwen3_5_target_capture import capture_forward
        return CacheFreeTargetRunner(target_model, tli, capture_fn=capture_forward)
    return TargetRunner(target_model, tli)


class Engine:
    def __init__(self, cfg):
        from mlx_lm import load
        self.cfg = cfg
        self.draft, self.meta = load_draft(cfg.draft)
        self.model_id = cfg.model_id or self.meta.get("model_id") or "dspark-mlx"
        self.block_size = int(self.meta["block_size"])
        print(f"[dspark-mlx] draft loaded (arch={self.meta['arch']}, target={self.meta['target_id']}); "
              f"loading target ...", flush=True)
        self.target_model, self.tokenizer = load(self.meta["target_id"])
        eos = getattr(self.tokenizer, "eos_token_ids", None) or [self.tokenizer.eos_token_id]
        self.stop_ids = [int(x) for x in eos if x is not None]
        self.stop_set = set(self.stop_ids)
        self.lock = threading.Lock()

    def _prompt_ids(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return mx.array(list(ids), dtype=mx.int32)[None]

    def _decode(self, ids):
        return self.tokenizer.decode([t for t in ids if t not in self.stop_set])

    def complete(self, messages, max_new_tokens, temperature):
        prompt = self._prompt_ids(messages)
        with self.lock:
            runner = build_target_runner(self.target_model, self.meta)
            out = generate(runner, self.draft, prompt, max_new_tokens=max_new_tokens,
                           block_size=self.block_size, temperature=temperature,
                           stop_ids=self.stop_ids, confidence_threshold=self.cfg.confidence_threshold,
                           seed=self.cfg.seed)
        return {
            "text": self._decode(out.committed),
            "prompt_tokens": int(prompt.shape[1]),
            "completion_tokens": int(out.num_output),
            "finish_reason": "length" if int(out.num_output) >= max_new_tokens else "stop",
        }

    def stream(self, messages, max_new_tokens, temperature):
        """Return a Queue of text deltas (None sentinel at the end).

        The generation runs in a worker thread; MLX streams are thread-local, so the
        prompt array is built INSIDE that thread (not passed across threads).
        """
        prompt_ids = [int(t) for t in self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)]
        q: "queue.Queue" = queue.Queue()
        acc, prev = [], [""]

        def on_commit(new_ids):
            acc.extend(t for t in new_ids if t not in self.stop_set)
            text = self._decode(acc)                # decode-full-diff for correct detokenization
            delta = text[len(prev[0]):]
            prev[0] = text
            if delta:
                q.put(delta)

        def run():
            try:
                mx.set_default_device(mx.gpu)
                prompt = mx.array(prompt_ids, dtype=mx.int32)[None]
                with self.lock:
                    runner = build_target_runner(self.target_model, self.meta)
                    generate(runner, self.draft, prompt, max_new_tokens=max_new_tokens,
                             block_size=self.block_size, temperature=temperature,
                             stop_ids=self.stop_ids, confidence_threshold=self.cfg.confidence_threshold,
                             seed=self.cfg.seed, on_commit=on_commit)
            except Exception as e:  # surface instead of a silent empty stream
                q.put(f"[stream error: {e}]")
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return q


def build_app(engine):
    app = FastAPI(title="DSpark MLX server")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": engine.model_id, "object": "model", "owned_by": "deepspec-mlx"}]}

    def _chunk(cid, created, delta, finish=None):
        payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": engine.model_id,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {_json.dumps(payload)}\n\n"

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must be non-empty")
        max_new = int(req.max_tokens or engine.cfg.default_max_new_tokens)
        temp = engine.cfg.default_temperature if req.temperature is None else float(req.temperature)

        if not req.stream:
            r = engine.complete(req.messages, max_new, temp)
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": engine.model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]},
                             "finish_reason": r["finish_reason"]}],
                "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                          "total_tokens": r["prompt_tokens"] + r["completion_tokens"]},
            })

        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        q = engine.stream(req.messages, max_new, temp)

        def events():
            yield _chunk(cid, created, {"role": "assistant"})
            while True:
                delta = q.get()
                if delta is None:
                    break
                yield _chunk(cid, created, {"content": delta})
            yield _chunk(cid, created, {}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="draft checkpoint dir (save_draft output)")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID"))
    ap.add_argument("--default-max-new-tokens", type=int, default=512)
    ap.add_argument("--default-temperature", type=float, default=0.0)
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    import uvicorn
    engine = Engine(cfg)
    print(f"[dspark-mlx] ready: model_id={engine.model_id} on {cfg.host}:{cfg.port}", flush=True)
    uvicorn.run(build_app(engine), host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
