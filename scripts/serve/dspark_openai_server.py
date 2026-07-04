"""Minimal OpenAI-compatible server for a trained DSpark draft + its target.

This is Track 2a of the DSpark Qwen3-4B plan: it exposes the DSpark-accelerated
target model over `/v1/chat/completions` so any OpenAI-compatible client (the Pi
coding agent, aider, Cline, curl) can drive it. It also serves as the numeric
*reference oracle* for the llama.cpp port (Track 2b) -- at temperature 0 its
output must match plain-target greedy token-for-token, because speculative
decoding is lossless.

Design: rather than reimplement the speculative loop, this reuses the repo's
`Qwen3DSparkEvaluator` (which already wires `_init_context/_propose/_update`)
and calls its `generate_one_sample` primitive once per request. The evaluator's
`build_models` loads target + draft on one CUDA device in bf16/sdpa.

Constraints (documented, not bugs):
  * batch size 1, single in-flight request (the eval loop asserts bsz==1). A
    global lock serializes concurrent requests -- fine for one Pi session.
  * `stream=true` emits the full completion as a single SSE chunk. True
    per-token streaming would require refactoring the eval loop into a
    generator; deferred.
  * the draft changes speed, never output -- responses are token-identical to
    plain Qwen3-4B.

Run (inside the CUDA container):
    TARGET=Qwen/Qwen3-4B \
    DRAFT=$HOME/checkpoints/deepspec/dspark_qwen3_4b_trial/step_latest \
    python scripts/serve/dspark_openai_server.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import uuid

# init_dist() reads these; set defaults before importing deepspec so a plain
# `python server.py` (not torch.multiprocessing.spawn) initializes a 1-rank group.
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29555")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from deepspec.data.parser import encode_chat_messages  # noqa: E402
from deepspec.eval.base_evaluator import resolve_stop_token_ids  # noqa: E402
from deepspec.eval.dspark import (  # noqa: E402
    Gemma4DSparkEvaluator,
    Qwen3DSparkEvaluator,
    Qwen35DSparkEvaluator,
)

# Mirror eval.py's architecture -> evaluator mapping so the server works for any
# DSpark draft, not just Qwen3-4B.
EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Qwen35DSparkModel": Qwen35DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
}


def build_args():
    parser = argparse.ArgumentParser(description="OpenAI-compatible DSpark server")
    parser.add_argument("--target", default=os.environ.get("TARGET", "Qwen/Qwen3-4B"))
    parser.add_argument(
        "--draft",
        default=os.environ.get(
            "DRAFT",
            os.path.expanduser(
                "~/checkpoints/deepspec/dspark_qwen3_4b_trial/step_latest"
            ),
        ),
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", "dspark-qwen3-4b"),
        help="Model name this server advertises (must match models.json id in Pi).",
    )
    parser.add_argument(
        "--default-max-new-tokens",
        type=int,
        default=int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "2048")),
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=float(os.environ.get("DEFAULT_TEMPERATURE", "0.7")),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.0")),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "980406")))
    return parser.parse_args()


class Engine:
    """Loads the evaluator once and runs one blocking generation per call."""

    def __init__(self, cfg):
        self.cfg = cfg
        # The args namespace the evaluator/BaseEvaluator read (see eval.py).
        self.args = argparse.Namespace(
            target_name_or_path=cfg.target,
            draft_name_or_path=cfg.draft,
            max_new_tokens=cfg.default_max_new_tokens,
            temperature=cfg.default_temperature,
            confidence_threshold=cfg.confidence_threshold,
            tensorboard_dir=None,
            step=None,
            seed=cfg.seed,
            tasks=[],  # unused: we never call evaluate()
        )
        draft_config = AutoConfig.from_pretrained(cfg.draft)
        arch = draft_config.architectures[0]
        if arch not in EVALUATORS:
            raise ValueError(
                f"Unsupported draft architecture {arch!r}; expected one of "
                f"{sorted(EVALUATORS)}"
            )
        # Instantiating the evaluator calls init_dist(0) + build_models() -> loads
        # target + draft onto cuda:0. One-time, at startup.
        self.evaluator = EVALUATORS[arch](local_rank=0, args=self.args)
        self.stop_token_ids = resolve_stop_token_ids(
            self.evaluator.target_model,
            self.evaluator.tokenizer,
        )
        self.device = self.evaluator.device
        self.tokenizer = self.evaluator.tokenizer
        self.lock = threading.Lock()

    def generate(self, messages, max_new_tokens: int, temperature: float):
        input_ids = encode_chat_messages(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.device)
        with self.lock:
            # Per-request overrides; safe because the lock serializes calls.
            self.args.max_new_tokens = int(max_new_tokens)
            self.args.temperature = float(temperature)
            out = self.evaluator.generate_one_sample(
                input_ids=input_ids,
                stop_token_ids=self.stop_token_ids,
            )
        new_ids = out.output_ids[0, out.num_input_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        finish_reason = "stop" if int(out.num_output_tokens) < int(max_new_tokens) else "length"
        return {
            "text": text,
            "prompt_tokens": int(out.num_input_tokens),
            "completion_tokens": int(out.num_output_tokens),
            "finish_reason": finish_reason,
        }


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False


CFG = build_args()
ENGINE: Engine | None = None
app = FastAPI(title="DSpark OpenAI-compatible server")


@app.on_event("startup")
def _load():
    global ENGINE
    ENGINE = Engine(CFG)
    print(
        f"[dspark-server] ready: target={CFG.target} draft={CFG.draft} "
        f"model_id={CFG.model_id} device={ENGINE.device}",
        flush=True,
    )


@app.get("/health")
def health():
    return {"status": "ok" if ENGINE is not None else "loading"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": CFG.model_id, "object": "model", "owned_by": "deepspec"}
        ],
    }


def _completion_payload(result, model_id):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="model still loading")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")

    max_new_tokens = req.max_tokens or CFG.default_max_new_tokens
    temperature = CFG.default_temperature if req.temperature is None else req.temperature
    result = ENGINE.generate(req.messages, max_new_tokens, temperature)

    if not req.stream:
        return JSONResponse(_completion_payload(result, CFG.model_id))

    # Single-chunk SSE stream (see module docstring). Correct, if not incremental.
    def event_stream():
        created = int(time.time())
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"

        def chunk(delta, finish_reason=None):
            import json as _json

            payload = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": CFG.model_id,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
            return f"data: {_json.dumps(payload)}\n\n"

        yield chunk({"role": "assistant"})
        yield chunk({"content": result["text"]})
        yield chunk({}, finish_reason=result["finish_reason"])
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    print(
        f"[dspark-server] loading target={CFG.target} draft={CFG.draft} ...",
        flush=True,
    )
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="info")
