"""Elastic attention: a straggler lends query heads to on-node helpers.

The group plan lists head moves ``(donor, recipient)``, one per query head
(Algorithm S1). A donor keeps its first ``keep`` query heads and gives each
recipient a contiguous range of the rest. For every layer of the iteration:

1. Before attention, the donor sends each recipient the layer input and the
   prefix KV of the KV heads that recipient's query heads use (point-to-point
   over the node's NVLink, overlapping the ranks' own attention).
2. The donor computes its kept heads; each recipient computes the donated
   heads (QKV projection, attention over prefix plus new KV, and its slice of
   the output projection).
3. After attention, the recipients send their output-projection partial sums
   back and the donor adds them to its own; for KV heads the donor gave away
   entirely, one recipient also returns their new KV, which the donor writes
   into its paged cache before LMCache saves the layer.

The sum over heads of the output projection is exact, so the result equals
single-GPU attention up to floating-point rounding. Grouped-query attention is
handled by giving each recipient the KV heads of the query heads it holds.

Requirements: unquantized BF16/FP16 attention weights, a BF16/FP16 KV cache,
the FlashAttention backend, TP=1, and ``--enforce-eager`` on the prefill
instance (the per-iteration Python logic is not traceable by torch.compile).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import state as _state


@dataclass
class Assignment:
    """This rank's part of the plan for one iteration.

    ``out``: ``(recipient, a, b)``, this rank's heads ``[a, b)`` computed by
    the recipient. ``inn``: ``(donor, a, b)``, the donor's heads ``[a, b)``
    computed here. ``returns_to[donor]`` / ``returned_by[recipient]``: KV
    heads whose new KV the recipient sends back to the donor.
    """

    keep: int
    out: list[tuple[int, int, int]] = field(default_factory=list)
    inn: list[tuple[int, int, int]] = field(default_factory=list)
    returned_by: dict[int, list[int]] = field(default_factory=dict)
    returns_to: dict[int, list[int]] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(self.out or self.inn)


def head_ranges(moves: list[tuple[int, int]], H: int) -> dict[int, list[tuple[int, int, int]]]:
    """Per donor: ``(recipient, a, b)`` ranges over its shed heads ``[keep, H)``.

    Recipients are ordered by rank, so every rank derives the same ranges.
    """
    counts: dict[int, dict[int, int]] = {}
    for s, d in moves:
        counts.setdefault(s, {})
        counts[s][d] = counts[s].get(d, 0) + 1
    out: dict[int, list[tuple[int, int, int]]] = {}
    for s, per in counts.items():
        cursor = H - sum(per.values())
        ranges = []
        for d in sorted(per):
            ranges.append((d, cursor, cursor + per[d]))
            cursor += per[d]
        out[s] = ranges
    return out


def derive_assignment(moves: list[tuple[int, int]], rank: int, H: int,
                      q_per_kv: int, num_kv: int) -> Assignment | None:
    """This rank's :class:`Assignment` from the plan's moves (or ``None``)."""
    if not moves:
        return None
    ranges = head_ranges(moves, H)
    mine = ranges.get(rank, [])
    keep = mine[0][1] if mine else H
    asg = Assignment(keep=keep, out=list(mine))
    for s, rs in sorted(ranges.items()):
        for d, a, b in rs:
            if d == rank:
                asg.inn.append((s, a, b))
    # new KV of fully donated KV heads comes back from the recipient holding
    # the first query head of that KV head
    for s, rs in ranges.items():
        keep_s = rs[0][1]
        for g in range(math.ceil(keep_s / q_per_kv), num_kv):
            h = g * q_per_kv
            owner = next(d for d, a, b in rs if a <= h < b)
            if s == rank:
                asg.returned_by.setdefault(owner, []).append(g)
            if owner == rank:
                asg.returns_to.setdefault(s, []).append(g)
    return asg if asg.active else None


@dataclass
class _DonorMeta:
    T: int
    P: int
    slots: torch.Tensor | None     # prefix slot ids in the paged cache
    header: torch.Tensor


@dataclass
class _RecvMeta:
    T: int
    P: int
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    positions: torch.Tensor
    idx_new: torch.Tensor
    idx_prefix: torch.Tensor | None


class ElasticAttention:
    """Per-process elastic-attention runtime of one prefill DP rank."""

    def __init__(self, rt: _state.StepRuntime, max_num_seqs: int):
        m = rt.model
        self.rt = rt
        self.H, self.Hkv, self.D = m.num_q_heads, m.num_kv_heads, m.head_dim
        self.qpk = m.q_per_kv
        self.max_seqs = max_num_seqs
        self.rank = rt.rank
        self.groups: dict[int, object] = {}
        self.asg: Assignment | None = None
        self.seq = -1
        self._meta: dict = {}
        self._lock = threading.Lock()
        rt.add_listener(self._on_plan)

    # --- setup ------------------------------------------------------------------

    def init_groups(self, node_size: int, num_ubatches: int) -> None:
        """Create one NCCL group per node (per micro-batch under DBO).

        Every rank of the world calls this in the same order.
        """
        world = dist.get_world_size()
        G = self.rt.deploy.ep_group_workers
        if world != G:
            raise RuntimeError(f"elastic attention needs TP=1: world={world}, G={G}")
        nodes = [list(range(n, min(n + node_size, G))) for n in range(0, G, node_size)]
        for u in range(num_ubatches):
            for ranks in nodes:
                g = dist.new_group(ranks=ranks, backend="nccl")
                if self.rank in ranks:
                    self.groups[u] = g
        # First collective on each group involves all members, so later
        # point-to-point batches may involve only the peers.
        for u, g in self.groups.items():
            t = torch.ones(1, device="cuda")
            dist.all_reduce(t, group=g)
        torch.cuda.synchronize()

    def _on_plan(self, plan, seq: int) -> None:
        asg = None
        if plan is not None and plan.moves:
            asg = derive_assignment(plan.moves, self.rank, self.H, self.qpk, self.Hkv)
        with self._lock:
            self.asg = asg
            self.seq = seq
            self._meta = {}

    # --- per-iteration metadata ---------------------------------------------------

    def _ubatch(self) -> int:
        try:
            from vllm.v1.worker.ubatching import dbo_current_ubatch_id
            return dbo_current_ubatch_id()
        except Exception:
            return 0

    def _exchange_headers(self, ub: int, md, kv_cache, group) -> tuple:
        """Once per iteration and micro-batch: donors tell recipients the shape
        of their batch; both sides precompute index tensors."""
        key = (self.seq, ub)
        got = self._meta.get(key)
        if got is not None:
            return got
        asg = self.asg
        dev = torch.device("cuda", torch.cuda.current_device())
        L = 3 + 2 * self.max_seqs
        donor = None
        ops = []
        if asg.out:
            qsl = md.query_start_loc
            R = int(md.seq_lens.shape[0])
            T = int(md.num_actual_tokens)
            q_lens = (qsl[1:R + 1] - qsl[:R]).to(torch.int32)
            ctx = (md.seq_lens[:R] - q_lens).to(torch.int32)
            P = int(ctx.sum().item())
            hdr = torch.zeros(L, dtype=torch.int32, device=dev)
            hdr[0], hdr[1], hdr[2] = T, R, P
            hdr[3:3 + R] = q_lens
            hdr[3 + self.max_seqs:3 + self.max_seqs + R] = ctx
            slots = None
            if P > 0:
                bs = kv_cache.shape[2]
                ctx64 = ctx.to(torch.int64)
                r_idx = torch.repeat_interleave(torch.arange(R, device=dev), ctx64)
                starts = torch.cumsum(ctx64, 0) - ctx64
                pos = torch.arange(P, device=dev) - torch.repeat_interleave(starts, ctx64)
                blk = md.block_table[r_idx, pos // bs].to(torch.int64)
                slots = blk * bs + pos % bs
            donor = _DonorMeta(T=T, P=P, slots=slots, header=hdr)
            for d, _, _ in asg.out:
                ops.append(dist.P2POp(dist.isend, hdr, d, group=group))
        recv_hdrs = {}
        for s, _, _ in asg.inn:
            h = torch.empty(L, dtype=torch.int32, device=dev)
            recv_hdrs[s] = h
            ops.append(dist.P2POp(dist.irecv, h, s, group=group))
        if ops:
            for w in dist.batch_isend_irecv(ops):
                w.wait()
        recv = {}
        for s, h in recv_hdrs.items():
            v = h.tolist()
            T, R, P = v[0], v[1], v[2]
            q_lens = torch.tensor(v[3:3 + R], dtype=torch.int64, device=dev)
            ctx = torch.tensor(v[3 + self.max_seqs:3 + self.max_seqs + R],
                               dtype=torch.int64, device=dev)
            seg = ctx + q_lens
            cu_q = torch.zeros(R + 1, dtype=torch.int32, device=dev)
            cu_q[1:] = torch.cumsum(q_lens, 0)
            cu_k = torch.zeros(R + 1, dtype=torch.int32, device=dev)
            cu_k[1:] = torch.cumsum(seg, 0)
            r_new = torch.repeat_interleave(torch.arange(R, device=dev), q_lens)
            off_new = torch.arange(T, device=dev) - cu_q[:-1].to(torch.int64)[r_new]
            positions = ctx[r_new] + off_new
            idx_new = cu_k[:-1].to(torch.int64)[r_new] + ctx[r_new] + off_new
            idx_prefix = None
            if P > 0:
                r_pre = torch.repeat_interleave(torch.arange(R, device=dev), ctx)
                starts = torch.cumsum(ctx, 0) - ctx
                off_pre = torch.arange(P, device=dev) - starts[r_pre]
                idx_prefix = cu_k[:-1].to(torch.int64)[r_pre] + off_pre
            recv[s] = _RecvMeta(T=T, P=P, cu_q=cu_q, cu_k=cu_k,
                                max_q=int(max(v[3:3 + R], default=0)),
                                max_k=int(seg.max().item()) if R else 0,
                                positions=positions, idx_new=idx_new,
                                idx_prefix=idx_prefix)
        self._meta[key] = (donor, recv)
        return donor, recv

    # --- the attention forward ------------------------------------------------------

    def forward(self, mod, positions: torch.Tensor, hidden_states: torch.Tensor, orig):
        asg = self.asg
        if asg is None:
            return orig(positions, hidden_states)
        from vllm.model_executor.layers.attention.attention import get_attention_context

        ub = self._ubatch()
        group = self.groups[ub]
        layer_name = mod.attn.layer_name
        is_donor = bool(asg.out)
        md = attn_layer = kv_cache = slot_mapping = None
        if is_donor:
            md, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
        donor, recv = self._exchange_headers(ub, md, kv_cache, group)
        W = _Weights(mod, self)

        # 1. donors wait for this layer's inbound prefix KV, then send
        if is_donor:
            _connector_wait(layer_name, md)
        ops = []
        if is_donor:
            x = hidden_states[:donor.T].contiguous()
            for d, a, b in asg.out:
                ops.append(dist.P2POp(dist.isend, x, d, group=group))
                if donor.P > 0:
                    g0, g1 = self._kv_range(a, b)
                    kp, vp = _gather_prefix(kv_cache, donor.slots, g0, g1)
                    ops.append(dist.P2POp(dist.isend, kp, d, group=group))
                    ops.append(dist.P2POp(dist.isend, vp, d, group=group))
        bufs = {}
        for s, a, b in asg.inn:
            m = recv[s]
            g0, g1 = self._kv_range(a, b)
            xs = hidden_states.new_empty(m.T, hidden_states.shape[-1])
            ops.append(dist.P2POp(dist.irecv, xs, s, group=group))
            kp = vp = None
            if m.P > 0:
                kp = hidden_states.new_empty(m.P, g1 - g0, self.D)
                vp = hidden_states.new_empty(m.P, g1 - g0, self.D)
                ops.append(dist.P2POp(dist.irecv, kp, s, group=group))
                ops.append(dist.P2POp(dist.irecv, vp, s, group=group))
            bufs[s] = (xs, kp, vp)
        works = dist.batch_isend_irecv(ops) if ops else []

        # 2. own attention (kept heads on a donor; everything otherwise)
        if is_donor:
            out = self._kept_attention(mod, W, positions, hidden_states, md,
                                       kv_cache, slot_mapping, attn_layer, donor.T)
        else:
            out = orig(positions, hidden_states)
        for w in works:
            w.wait()

        # 3. donated heads, then return partial outputs (and new KV)
        post = []
        for s, a, b in asg.inn:
            xs, kp, vp = bufs[s]
            y, k_new, v_new = self._donated_heads(mod, W, recv[s], xs, kp, vp, a, b)
            post.append(dist.P2POp(dist.isend, y, s, group=group))
            g0, _ = self._kv_range(a, b)
            for g in asg.returns_to.get(s, []):
                post.append(dist.P2POp(dist.isend, k_new[:, g - g0].contiguous(), s,
                                       group=group))
                post.append(dist.P2POp(dist.isend, v_new[:, g - g0].contiguous(), s,
                                       group=group))
        partials = []
        returned = []
        if is_donor:
            T = donor.T
            for d, _, _ in asg.out:
                y = hidden_states.new_empty(T, hidden_states.shape[-1])
                post.append(dist.P2POp(dist.irecv, y, d, group=group))
                partials.append(y)
                for g in asg.returned_by.get(d, []):
                    k = hidden_states.new_empty(T, self.D)
                    v = hidden_states.new_empty(T, self.D)
                    post.append(dist.P2POp(dist.irecv, k, d, group=group))
                    post.append(dist.P2POp(dist.irecv, v, d, group=group))
                    returned.append((g, k, v))
        works = dist.batch_isend_irecv(post) if post else []
        for w in works:
            w.wait()

        if is_donor:
            T = donor.T
            for y in partials:
                out[:T] += y
            key_cache, value_cache = kv_cache.unbind(0)
            for g, k, v in returned:
                _write_kv(attn_layer, k.view(T, 1, self.D), v.view(T, 1, self.D),
                          key_cache[:, :, g:g + 1], value_cache[:, :, g:g + 1],
                          slot_mapping[:T])
            _connector_save(layer_name, kv_cache, md)
        return out

    # --- pieces -------------------------------------------------------------------

    def _kv_range(self, a: int, b: int) -> tuple[int, int]:
        return a // self.qpk, (b - 1) // self.qpk + 1

    def _kept_attention(self, mod, W, positions, hidden_states, md, kv_cache,
                        slot_mapping, attn_layer, T: int) -> torch.Tensor:
        K = self.asg.keep
        out = hidden_states.new_zeros(hidden_states.shape[0], W.hidden)
        if K == 0:
            return out
        D, qpk = self.D, self.qpk
        Gk = math.ceil(K / qpk)
        x = hidden_states[:T]
        q, k, v = W.project(x, 0, K, 0, Gk)
        q, k = W.norm_rope(positions[:T], q, k, K, Gk)
        key_cache, value_cache = kv_cache.unbind(0)
        _write_kv(attn_layer, k.view(T, Gk, D), v.view(T, Gk, D),
                  key_cache[:, :, :Gk], value_cache[:, :, :Gk], slot_mapping[:T])
        q3 = q.view(T, K, D)
        o = torch.empty_like(q3)
        for g in range(Gk):
            h0, h1 = g * qpk, min((g + 1) * qpk, K)
            _fa(attn_layer, q3[:, h0:h1], key_cache[:, :, g:g + 1],
                value_cache[:, :, g:g + 1], o[:, h0:h1], mod.scaling,
                cu_q=md.query_start_loc, max_q=md.max_query_len,
                seqused_k=md.seq_lens, max_k=md.max_seq_len,
                block_table=md.block_table)
        out[:T] = W.o_proj(o.view(T, K * D), 0, K)
        return out

    def _donated_heads(self, mod, W, m: _RecvMeta, xs, kp, vp, a: int, b: int):
        D, qpk = self.D, self.qpk
        g0, g1 = self._kv_range(a, b)
        nh, gc, T = b - a, g1 - g0, m.T
        q, k, v = W.project(xs, a, b, g0, g1)
        q, k = W.norm_rope(m.positions, q, k, nh, gc)
        k3, v3 = k.view(T, gc, D), v.view(T, gc, D)
        kf = xs.new_empty(m.P + T, gc, D)
        vf = xs.new_empty(m.P + T, gc, D)
        kf.index_copy_(0, m.idx_new, k3)
        vf.index_copy_(0, m.idx_new, v3)
        if m.P > 0:
            kf.index_copy_(0, m.idx_prefix, kp)
            vf.index_copy_(0, m.idx_prefix, vp)
        q3 = q.view(T, nh, D)
        o = torch.empty_like(q3)
        attn_layer = mod.attn
        for g in range(g0, g1):
            lo, hi = max(a, g * qpk), min(b, (g + 1) * qpk)
            _fa(attn_layer, q3[:, lo - a:hi - a], kf[:, g - g0:g - g0 + 1],
                vf[:, g - g0:g - g0 + 1], o[:, lo - a:hi - a], mod.scaling,
                cu_q=m.cu_q, max_q=m.max_q, cu_k=m.cu_k, max_k=m.max_k)
        y = W.o_proj(o.view(T, nh * D), a, b)
        return y, k3, v3


class _Weights:
    """Row and column slices of one layer's attention weights."""

    def __init__(self, mod, ea: ElasticAttention):
        self.mod = mod
        self.D = ea.D
        self.q_size = ea.H * ea.D
        self.kv_size = ea.Hkv * ea.D
        self.w_qkv = mod.qkv_proj.weight
        self.b_qkv = getattr(mod.qkv_proj, "bias", None)
        self.w_o = mod.o_proj.weight
        self.hidden = self.w_o.shape[0]

    def _rows(self, x, r0: int, r1: int):
        b = self.b_qkv[r0:r1] if self.b_qkv is not None else None
        return F.linear(x, self.w_qkv[r0:r1], b)

    def project(self, x, a: int, b: int, g0: int, g1: int):
        D, qs, kvs = self.D, self.q_size, self.kv_size
        q = self._rows(x, a * D, b * D)
        k = self._rows(x, qs + g0 * D, qs + g1 * D)
        v = self._rows(x, qs + kvs + g0 * D, qs + kvs + g1 * D)
        return q, k, v

    def norm_rope(self, positions, q, k, nh: int, gk: int):
        mod, D, T = self.mod, self.D, q.shape[0]
        q = mod.q_norm(q.view(T, nh, D)).reshape(T, nh * D)
        k = mod.k_norm(k.view(T, gk, D)).reshape(T, gk * D)
        # q and k separately: the rotary kernel expects a whole number of query
        # heads per KV head, which a head subset need not have.
        q, _ = mod.rotary_emb(positions, q, None)
        k, _ = mod.rotary_emb(positions, k, None)
        return q, k

    def o_proj(self, o, a: int, b: int):
        return F.linear(o, self.w_o[:, a * self.D:b * self.D])


def _gather_prefix(kv_cache, slots, g0: int, g1: int):
    key_cache, value_cache = kv_cache.unbind(0)
    nb, bs, hk, d = key_cache.shape
    kf = key_cache.view(nb * bs, hk, d)[:, g0:g1]
    vf = value_cache.view(nb * bs, hk, d)[:, g0:g1]
    return kf.index_select(0, slots), vf.index_select(0, slots)


def _write_kv(attn_layer, k, v, key_cache, value_cache, slot_mapping) -> None:
    from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

    reshape_and_cache_flash(k, v, key_cache, value_cache, slot_mapping,
                            attn_layer.kv_cache_dtype, attn_layer._k_scale,
                            attn_layer._v_scale)


def _fa(attn_layer, q, k, v, out, scale, cu_q, max_q, max_k, seqused_k=None,
        cu_k=None, block_table=None) -> None:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    flash_attn_varlen_func(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, max_seqlen_q=max_q,
        seqused_k=seqused_k, cu_seqlens_k=cu_k, max_seqlen_k=max_k,
        softmax_scale=scale, causal=True, block_table=block_table,
        fa_version=attn_layer.impl.vllm_flash_attn_version)


def _connector_wait(layer_name: str, md) -> None:
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group, has_kv_transfer_group, is_v1_kv_transfer_group)

    if md is None or not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return
    conn = get_kv_transfer_group()
    if conn.has_connector_metadata():
        conn.wait_for_layer_load(layer_name)


def _connector_save(layer_name: str, kv_cache, md) -> None:
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group, has_kv_transfer_group, is_v1_kv_transfer_group)

    if md is None or not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return
    conn = get_kv_transfer_group()
    if conn.has_connector_metadata():
        conn.save_kv_layer(layer_name, kv_cache, md)


_EAP: ElasticAttention | None = None


def get() -> ElasticAttention | None:
    return _EAP


def init(max_num_seqs: int, node_size: int, num_ubatches: int) -> ElasticAttention:
    """Create the process's elastic-attention runtime and its node groups."""
    global _EAP
    rt = _state.get()
    assert rt is not None
    _EAP = ElasticAttention(rt, max_num_seqs)
    _EAP.init_groups(node_size, num_ubatches)
    return _EAP
