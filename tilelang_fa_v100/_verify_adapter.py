"""Exact grouped split-KV verifier adapter for SM70."""

import math

import torch

from ._kernels_paged_verify import VERIFY_Q_BLOCK, get_paged_verify_kernels

_FP16_LUTS: dict[str, torch.Tensor] = {}


def _fp16_dead_lut(device: torch.device) -> torch.Tensor:
    key = str(device)
    value = _FP16_LUTS.get(key)
    if value is None:
        value = torch.zeros(256, dtype=torch.float16, device=device)
        _FP16_LUTS[key] = value
    return value


def verify_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool,
) -> torch.Tensor:
    """Run the grouped verifier for one fixed-shape CUDA graph geometry."""
    if q.dtype != torch.float16:
        raise ValueError("The grouped verifier requires FP16 Q.")
    if k_cache.dtype != torch.float16 or v_cache.dtype != torch.float16:
        raise ValueError("The grouped verifier currently requires FP16 K/V.")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("Expected Q [tokens, heads, dim] and equal 4D K/V caches.")
    batch, max_blocks = block_table.shape
    num_tokens, heads, dim = q.shape
    heads_kv = k_cache.shape[2]
    if num_tokens > batch * VERIFY_Q_BLOCK:
        raise ValueError("The grouped verifier supports at most 16 query rows per batch item.")
    if k_cache.shape[1] != 16:
        raise ValueError("The grouped verifier requires page size 16.")
    if heads % heads_kv:
        raise ValueError("The query-head count must be divisible by the KV-head count.")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    partial, combine, _ = get_paged_verify_kernels(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        block_size=k_cache.shape[1],
        num_pages=k_cache.shape[0],
        max_blocks=max_blocks,
        causal=causal,
        fp8_kv=False,
    )
    partial_o, partial_lse = partial(
        q,
        k_cache,
        v_cache,
        _fp16_dead_lut(q.device),
        block_table.contiguous().to(device=q.device, dtype=torch.int32),
        seq_lens.contiguous().to(device=q.device, dtype=torch.int32),
        query_start_loc.contiguous().to(device=q.device, dtype=torch.int32),
        prefix_kv_lens.contiguous().to(device=q.device, dtype=torch.int32),
        softmax_scale,
    )
    result = combine(
        partial_o,
        partial_lse,
        seq_lens.contiguous().to(device=q.device, dtype=torch.int32),
        query_start_loc.contiguous().to(device=q.device, dtype=torch.int32),
    )[:num_tokens]
    if out is None:
        return result
    out.copy_(result)
    return out
