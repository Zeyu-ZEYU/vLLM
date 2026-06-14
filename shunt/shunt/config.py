"""Static configuration for Shunt: the served model and the evaluation testbed.

All values are taken from the paper (§2.2 model, §2.3 testbed). They are the
defaults used by every offline analysis, microbenchmark, and the online planner;
override them through the dataclasses if you deploy a different model or fabric.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Qwen3-235B-A22B (§2.2).

    A grouped-query-attention (GQA) block with ``num_q_heads`` query heads and
    ``num_kv_heads`` KV heads, followed by a fine-grained MoE FFN with
    ``num_experts`` routed experts, of which ``top_k`` are activated per token.
    """

    name: str = "Qwen3-235B-A22B"
    num_layers: int = 94
    hidden: int = 4096            # d_model
    num_q_heads: int = 64
    num_kv_heads: int = 4         # GQA: 16 query heads share one KV head
    head_dim: int = 128
    num_experts: int = 128
    top_k: int = 8                # experts activated per token
    moe_intermediate: int = 1536  # per-expert FFN intermediate width
    dtype_bytes: int = 2          # BF16

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim      # 8192

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim     # 512

    @property
    def q_per_kv(self) -> int:
        return self.num_q_heads // self.num_kv_heads  # 16 (GQA group size)

    @property
    def kv_bytes_per_token_per_layer(self) -> int:
        """K and V, both BF16, for the KV heads of one layer."""
        return self.num_kv_heads * self.head_dim * 2 * self.dtype_bytes  # 2048 B


@dataclass(frozen=True)
class TestbedConfig:
    """Four-node production cluster (§2.3).

    Two prefill nodes share a single DP=16, EP=16, TP=1 deployment; each of the
    two decode nodes runs DP=8, EP=8, TP=1. Every GPU is pinned to its own
    200 Gb/s backend RNIC port; each node has one 200 Gb/s frontend RNIC shared
    by its eight workers.
    """

    bw_port_GBps: float = 25.0    # 200 Gb/s NDR backend port, per GPU
    bw_fe_GBps: float = 25.0      # 200 Gb/s HDR frontend RNIC, one per node
    workers_per_node: int = 8     # DP workers (= backend ports) on a prefill node
    ep_group_workers: int = 16    # DP/EP width of the prefill deployment
    kv_chunk_tokens: int = 256    # token span at which LMCache moves KV (§3.1)

    @property
    def num_prefill_nodes(self) -> int:
        return self.ep_group_workers // self.workers_per_node


# Effective BF16 compute rate used to turn the analytical FLOP counts of the
# compute model into seconds when no measured profile is supplied. Imbalance
# ratios (max/mean) are scale-invariant in this constant; only absolute
# predicted times depend on it, and those should come from an offline profile
# (see ComputeModel.from_profile and shunt.profiling).
DEFAULT_EFFECTIVE_FLOPS: float = 1.0e14  # 100 TFLOP/s, nominal for H20-3e

MODEL = ModelConfig()
TESTBED = TestbedConfig()
