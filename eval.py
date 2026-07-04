from __future__ import annotations
import argparse
import json
import os
import torch
from transformers import AutoConfig
from deepspec.eval.dspark import (
    Gemma4DSparkEvaluator,
    Qwen3DSparkEvaluator,
    Qwen35DSparkEvaluator,
)
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.utils import CustomJSONEncoder

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Qwen35DSparkModel": Qwen35DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]

def resolve_tasks():
    """Resolve the benchmark suite, optionally narrowed via env for fast runs.

    EVAL_TASKS="gsm8k,humaneval" selects a subset (default: all 9 tasks).
    EVAL_MAX_SAMPLES caps the per-task sample count (default: each task's own cap).
    Unset -> the full suite, unchanged.
    """
    default_caps = {name: cap for name, cap in TASKS}
    names_env = os.environ.get("EVAL_TASKS")
    cap_env = os.environ.get("EVAL_MAX_SAMPLES")
    wanted = (
        [n.strip() for n in names_env.split(",") if n.strip()]
        if names_env
        else [name for name, _ in TASKS]
    )
    tasks = []
    for name in wanted:
        if name not in default_caps:
            raise ValueError(
                f"Unknown EVAL_TASKS entry {name!r}; valid: {sorted(default_caps)}"
            )
        tasks.append((name, int(cap_env) if cap_env else default_caps[name]))
    return tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    args = parser.parse_args()
    args.tasks = resolve_tasks()
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )
