"""Exact-shape tests for the grouped SM70 verifier."""

import importlib.util
import math

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _reference(q, k_cache, v_cache, block_table, context: int, prefix: int, causal: bool):
    torch = importlib.import_module("torch")
    pages = block_table[0, : math.ceil(context / 16)].to(torch.int64)
    k = k_cache.index_select(0, pages).flatten(0, 1)[:context]
    v = v_cache.index_select(0, pages).flatten(0, 1)[:context]
    output = torch.empty_like(q)
    key_positions = torch.arange(context, device=q.device)
    for head in range(q.shape[1]):
        kv_head = head // (q.shape[1] // k.shape[1])
        scores = torch.matmul(q[:, head].float(), k[:, kv_head].float().T)
        scores *= q.shape[-1] ** -0.5
        if causal:
            query_positions = prefix + torch.arange(q.shape[0], device=q.device)
            scores.masked_fill_(key_positions[None, :] > query_positions[:, None], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1).to(torch.float16)
        output[:, head] = torch.matmul(
            probabilities.float(), v[:, kv_head].float()
        ).to(torch.float16)
    return output


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("identity_pages", [False, True])
def test_grouped_paged_verify_exact_shape(causal: bool, identity_pages: bool) -> None:
    if not TORCH_AVAILABLE:
        pytest.skip("This test requires PyTorch.")
    torch = importlib.import_module("torch")
    tilelang_verify_forward = importlib.import_module(
        "tilelang_fa_v100"
    ).tilelang_verify_forward

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("This test requires an NVIDIA V100 (SM70).")

    torch.manual_seed(20260831)
    context = 1008
    prefix = 1000
    page_count = math.ceil(context / 16)
    total_pages = page_count + 7
    q = torch.randn(8, 8, 128, device="cuda", dtype=torch.float16) * 0.1
    k_cache = torch.randn(
        total_pages, 16, 2, 128, device="cuda", dtype=torch.float16
    ) * 0.1
    v_cache = torch.randn_like(k_cache) * 0.1
    if identity_pages:
        pages = torch.arange(page_count, device="cuda", dtype=torch.int32)
    else:
        pages = torch.randperm(total_pages, device="cuda", dtype=torch.int32)[:page_count]
    block_table = torch.zeros(1, page_count, device="cuda", dtype=torch.int32)
    block_table[0] = pages
    seq_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
    query_start_loc = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    prefix_kv_lens = torch.tensor([prefix], device="cuda", dtype=torch.int32)

    first = tilelang_verify_forward(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        query_start_loc,
        prefix_kv_lens,
        causal=causal,
    )
    second = tilelang_verify_forward(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        query_start_loc,
        prefix_kv_lens,
        causal=causal,
    )
    expected = _reference(
        q, k_cache, v_cache, block_table, context, prefix, causal
    )

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, expected, rtol=2e-3, atol=2e-3)
