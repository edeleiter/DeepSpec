"""Phase 0 smoke test for using Ornith-1.0-9B (Qwen3.5) as a DSpark target.

Run this FIRST, inside the container, before wiring anything into DeepSpec. It
answers the three questions the rest of the plan depends on:

  1. Does Ornith load at all in this transformers, on GPU, in bf16?
  2. Does one forward return a per-layer hidden-states tuple we can cache?
  3. What is the *actual* module tree + config, so the qwen3_5 DSpark config
     builder and cache backbone locator are grounded in real field names?

Qwen3.5 is a multimodal composite config: the text fields (num_hidden_layers,
vocab_size, head_dim, layer_types, RoPE, ...) live under ``config.text_config``,
NOT at the top level. This script resolves that automatically.

It is read-only. Usage:

    python scripts/ornith/check_load.py --model deepreinforce-ai/Ornith-1.0-9B
"""

import argparse

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


def _get(cfg, name, default="<absent>"):
    return getattr(cfg, name, default)


def _resolve_text_config(cfg):
    """Qwen3.5/Gemma4-style composites nest the text model under .text_config."""
    return getattr(cfg, "text_config", None) or cfg


def _resolve_int(cfg, tcfg, name, tokenizer=None):
    for src in (tcfg, cfg):
        val = getattr(src, name, None)
        if isinstance(val, int):
            return val
    if name == "vocab_size" and tokenizer is not None:
        val = getattr(tokenizer, "vocab_size", None)
        if isinstance(val, int):
            return val
        return len(tokenizer)
    raise ValueError(f"Could not resolve int field {name!r} from config/tokenizer.")


def dump_config(cfg, tcfg) -> None:
    print("\n===== top-level config =====")
    for name in ("model_type", "architectures", "tie_word_embeddings",
                 "pad_token_id", "eos_token_id"):
        print(f"  {name:24s} = {_get(cfg, name)!r}")
    has_vision = getattr(cfg, "vision_config", None) is not None
    print(f"  vision_config present   = {has_vision} (the draft config builder drops it)")
    print(f"  text config nested      = {tcfg is not cfg} "
          f"(source: {'config.text_config' if tcfg is not cfg else 'top level'})")

    print("\n===== TEXT config fields (the qwen3_5 config builder must handle these) =====")
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rms_norm_eps",
        "hidden_act",
        "attention_bias",
        "attention_dropout",
        "initializer_range",
        "sliding_window",
        # RoPE fields that MUST be sanitized for the draft's plain full RoPE:
        "partial_rotary_factor",
        "rope_theta",
        "rope_scaling",
        "rope_parameters",
        "mrope_section",
        "mrope_interleaved",
        "rope_sections",
    ):
        print(f"  {name:24s} = {_get(tcfg, name)!r}")

    layer_types = _get(tcfg, "layer_types", None)
    if isinstance(layer_types, (list, tuple)):
        full_attn = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        print(f"  layer_types (len {len(layer_types)}) = {list(layer_types)}")
        print(f"  --> full_attention layer indices = {full_attn}")
        print("  --> use a subset of these as target_layer_ids (they carry global context)")
    else:
        print(f"  layer_types             = {layer_types!r}")


def dump_backbone_tree(base_model) -> None:
    """Find where the text decoder `.layers` live so _get_target_backbone can reach it."""
    print("\n===== AutoModel module tree (grounds _get_target_backbone) =====")
    print(f"  top-level type: {type(base_model).__name__}")
    print(f"  top-level children: {[n for n, _ in base_model.named_children()]}")

    candidates = {
        "model": getattr(base_model, "model", None),
        "language_model": getattr(base_model, "language_model", None),
        "model.language_model": getattr(getattr(base_model, "model", None), "language_model", None),
    }
    for path, mod in candidates.items():
        if mod is None:
            continue
        layers = getattr(mod, "layers", None)
        has_layers = layers is not None
        n_layers = len(layers) if has_layers else 0
        has_embed = getattr(mod, "embed_tokens", None) is not None
        print(
            f"  candidate '{path}': type={type(mod).__name__} "
            f"has .layers={has_layers} (n={n_layers}) has .embed_tokens={has_embed}"
        )
        if has_layers:
            print(f"  --> BACKBONE PATH FOR _get_target_backbone: '{path}'")


def suggest_mask_token(tokenizer, cfg, tcfg) -> None:
    print("\n===== mask_token_id suggestions (for config.model.mask_token_id) =====")
    vocab_size = _resolve_int(cfg, tcfg, "vocab_size", tokenizer)
    print(f"  (resolved vocab_size = {vocab_size})")
    for label, tok_id in (
        ("tokenizer.mask_token_id", getattr(tokenizer, "mask_token_id", None)),
        ("tokenizer.pad_token_id", getattr(tokenizer, "pad_token_id", None)),
        ("tokenizer.eos_token_id", getattr(tokenizer, "eos_token_id", None)),
        ("config.pad_token_id", getattr(cfg, "pad_token_id", None)),
    ):
        ok = isinstance(tok_id, int) and 0 <= tok_id < vocab_size
        print(f"  {label:24s} = {tok_id!r} (valid: {ok})")
    print("  Pick a valid, rarely-generated token id (a dedicated <mask>/<pad>/reserved id).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepreinforce-ai/Ornith-1.0-9B")
    parser.add_argument("--seq-len", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}, torch = {torch.__version__}")

    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = _resolve_text_config(cfg)
    dump_config(cfg, tcfg)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    suggest_mask_token(tokenizer, cfg, tcfg)

    vocab_size = _resolve_int(cfg, tcfg, "vocab_size", tokenizer)
    n_layers = _resolve_int(cfg, tcfg, "num_hidden_layers")

    # --- GATE 1: does it load + forward on GPU and emit per-layer hidden states? ---
    print("\n===== Loading AutoModelForCausalLM (bf16) and running one forward =====")
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    input_ids = torch.randint(0, vocab_size, (1, args.seq_len), device=device)
    with torch.no_grad():
        out = lm(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    print(f"  hidden_states tuple length = {len(hs)} (expected {n_layers + 1})")
    print(f"  each hidden state shape     = {tuple(hs[0].shape)}")
    assert len(hs) == n_layers + 1, "Unexpected hidden-states count; check output_hidden_states."

    embed = lm.get_input_embeddings()
    head = lm.get_output_embeddings()
    print(f"  embed_tokens.weight shape   = {tuple(embed.weight.shape)}")
    print(f"  lm_head.weight shape        = {tuple(head.weight.shape) if head is not None else None}")

    del lm
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- GATE 2: does AutoModel (base) expose a hookable text backbone .layers? ---
    print("\n===== Loading AutoModel (base, as the cache builder does) =====")
    base = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    dump_backbone_tree(base)
    with torch.no_grad():
        base_out = base(input_ids=input_ids, output_hidden_states=False, use_cache=False)
    print(f"  base output has .last_hidden_state = {hasattr(base_out, 'last_hidden_state')}")
    if hasattr(base_out, "last_hidden_state"):
        print(f"  last_hidden_state shape = {tuple(base_out.last_hidden_state.shape)}")

    print("\nAll gates passed. Record the BACKBONE PATH, full_attention indices, "
          "mask_token_id, and RoPE fields above, then finalize the qwen3_5 adapter.")


if __name__ == "__main__":
    main()
