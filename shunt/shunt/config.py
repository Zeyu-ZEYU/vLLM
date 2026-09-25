"""Model and deployment descriptions shared by the planner, the engine runtime,
the proxy, the profiler, and the analysis scripts.

Both are plain dataclasses that round-trip through JSON, so one file describes a
deployment to every component. ``ModelSpec.from_hf_config`` reads the shape of
an MoE model from its Hugging Face ``config.json``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class ModelSpec:
    """Shape of an EP-served MoE model with grouped-query attention.

    The defaults describe Qwen3-235B-A22B.
    """

    name: str = "Qwen3-235B-A22B"
    num_layers: int = 94
    hidden: int = 4096
    num_q_heads: int = 64
    num_kv_heads: int = 4
    head_dim: int = 128
    num_experts: int = 128
    top_k: int = 8
    moe_intermediate: int = 1536
    dtype_bytes: int = 2

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def q_per_kv(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V bytes of one token in one layer."""
        return 2 * self.kv_dim * self.dtype_bytes

    @classmethod
    def from_hf_config(cls, path: str) -> "ModelSpec":
        """Read the model shape from a Hugging Face ``config.json`` (file or dir)."""
        if not path.endswith(".json"):
            path = path.rstrip("/") + "/config.json"
        with open(path) as f:
            c = json.load(f)
        c = c.get("text_config", c)
        heads = int(c["num_attention_heads"])
        return cls(
            name=str(c.get("_name_or_path", c.get("model_type", "model"))),
            num_layers=int(c["num_hidden_layers"]),
            hidden=int(c["hidden_size"]),
            num_q_heads=heads,
            num_kv_heads=int(c.get("num_key_value_heads", heads)),
            head_dim=int(c.get("head_dim", int(c["hidden_size"]) // heads)),
            num_experts=int(c.get("num_experts", c.get("num_local_experts",
                                                       c.get("n_routed_experts", 1)))),
            top_k=int(c.get("num_experts_per_tok", 1)),
            moe_intermediate=int(c.get("moe_intermediate_size",
                                       c.get("intermediate_size", 0))),
            dtype_bytes=2,
        )


@dataclass
class DeploySpec:
    """The prefill deployment and its fabric, as the planner sees it.

    ``ep_group_workers`` DP workers form one EP group across the prefill nodes,
    ``workers_per_node`` per node, each pinned to one backend port of
    ``bw_port`` bytes/s. Each node has one frontend RNIC; ``bw_fe`` is the
    frontend bandwidth available to KV per direction. ``pcie_distance`` is a
    ``workers_per_node`` x ``workers_per_node`` matrix that orders the ports a
    worker borrows from (nearest first); ``None`` means ``|i - j|``.
    """

    ep_group_workers: int = 16
    workers_per_node: int = 8
    bw_port: float = 25.0e9
    bw_fe: float = 25.0e9
    kv_chunk_tokens: int = 256
    pcie_distance: list[list[float]] | None = None

    @property
    def num_nodes(self) -> int:
        return self.ep_group_workers // self.workers_per_node

    def node_of(self, worker: int) -> int:
        return worker // self.workers_per_node

    def node_workers(self, node: int) -> range:
        lo = node * self.workers_per_node
        return range(lo, lo + self.workers_per_node)


@dataclass
class PlanOptions:
    """Which parts of the per-iteration plan run (the evaluation's arms).

    ``eap`` enables elastic attention with compute-straggler threshold
    ``theta``. ``kvlb_budget`` caps each port's KV at its window budget and
    offloads the overflow; with it off, all KV stays on its own port.
    ``kvlb_borrow`` and ``kvlb_frontend`` enable the two overflow paths.
    ``dbo`` selects the budget that accounts for the other micro-batch's A2A.
    """

    eap: bool = True
    theta: float = 1.5
    kvlb_budget: bool = True
    kvlb_borrow: bool = True
    kvlb_frontend: bool = True
    dbo: bool = False


def _from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class ShuntConfig:
    """Everything the planner needs, loadable from one JSON file."""

    model: ModelSpec = field(default_factory=ModelSpec)
    deploy: DeploySpec = field(default_factory=DeploySpec)
    options: PlanOptions = field(default_factory=PlanOptions)
    compute_profile: str | None = None

    @classmethod
    def from_json(cls, path: str) -> "ShuntConfig":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "ShuntConfig":
        model = d.get("model", {})
        if isinstance(model, str):
            model_spec = ModelSpec.from_hf_config(model)
        else:
            model_spec = _from_dict(ModelSpec, model)
        return cls(
            model=model_spec,
            deploy=_from_dict(DeploySpec, d.get("deploy", {})),
            options=_from_dict(PlanOptions, d.get("options", {})),
            compute_profile=d.get("compute_profile"),
        )

    def to_dict(self) -> dict:
        return {
            "model": asdict(self.model),
            "deploy": asdict(self.deploy),
            "options": asdict(self.options),
            "compute_profile": self.compute_profile,
        }
