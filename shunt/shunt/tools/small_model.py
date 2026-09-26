"""Write a small Qwen3-MoE checkpoint with random weights.

For trying the serving stack and the harness on one machine
(``shunt/configs/cluster.single-node.yaml``). The checkpoint keeps the
architecture and the tokenizer of a Qwen3-MoE model (``--like``: a local
checkpoint directory or a model name on the Hugging Face Hub, from which only
the configuration and the tokenizer are read) but has few layers, a small
hidden size, and random weights, so its outputs carry no meaning.

Example::

    python -m shunt.tools.small_model --like Qwen/Qwen3-30B-A3B \\
        --out /models/small-qwen3-moe
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--like", required=True,
                    help="Qwen3-MoE checkpoint directory or Hub name (config and "
                         "tokenizer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--moe-intermediate", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(a.like)
    if cfg.model_type != "qwen3_moe":
        raise SystemExit(f"{a.like} is a {cfg.model_type} model, not qwen3_moe")
    cfg.num_hidden_layers = a.layers
    cfg.max_window_layers = a.layers
    cfg.hidden_size = a.hidden
    cfg.intermediate_size = 2 * a.hidden
    cfg.num_attention_heads = a.heads
    cfg.num_key_value_heads = a.kv_heads
    cfg.head_dim = a.head_dim
    cfg.num_experts = a.experts
    cfg.num_experts_per_tok = a.top_k
    cfg.moe_intermediate_size = a.moe_intermediate
    cfg.decoder_sparse_step = 1
    cfg.mlp_only_layers = []
    torch.manual_seed(a.seed)
    try:
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.like).save_pretrained(a.out)
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {a.out} ({n / 1e6:.0f}M parameters)")


if __name__ == "__main__":
    main()
