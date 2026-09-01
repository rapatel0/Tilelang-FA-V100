#!/usr/bin/env python3
# ruff: noqa: I001
"""Run deterministic verifier cases and save exact output hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import tilelang
import torch

from tilelang_fa_v100 import tilelang_verify_forward


def _sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _reference(q, k_cache, v_cache, block_table, *, context, prefix, causal):
    pages = block_table[0, : math.ceil(context / 16)].to(torch.int64)
    k = k_cache.index_select(0, pages).flatten(0, 1)[:context]
    v = v_cache.index_select(0, pages).flatten(0, 1)[:context]
    output = torch.empty_like(q)
    positions = torch.arange(context, device=q.device)
    for head in range(q.shape[1]):
        kv_head = head // (q.shape[1] // k.shape[1])
        scores = torch.matmul(q[:, head].float(), k[:, kv_head].float().T)
        scores *= q.shape[-1] ** -0.5
        if causal:
            query_positions = prefix + torch.arange(q.shape[0], device=q.device)
            scores.masked_fill_(positions[None, :] > query_positions[:, None], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1).to(torch.float16)
        output[:, head] = torch.matmul(
            probabilities.float(), v[:, kv_head].float()
        ).to(torch.float16)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("This runner requires an NVIDIA V100 (SM70).")

    args.output.mkdir(parents=True, exist_ok=True)
    context = 1008
    prefix = 1000
    page_count = math.ceil(context / 16)
    total_pages = page_count + 7
    results = []
    for causal in (False, True):
        for identity_pages in (False, True):
            torch.manual_seed(20260831)
            q = torch.randn(8, 8, 128, device="cuda", dtype=torch.float16) * 0.1
            k_cache = torch.randn(
                total_pages, 16, 2, 128, device="cuda", dtype=torch.float16
            ) * 0.1
            v_cache = torch.randn_like(k_cache) * 0.1
            if identity_pages:
                pages = torch.arange(page_count, device="cuda", dtype=torch.int32)
            else:
                pages = torch.randperm(
                    total_pages, device="cuda", dtype=torch.int32
                )[:page_count]
            block_table = pages.view(1, page_count).contiguous()
            seq_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
            query_start_loc = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
            prefix_kv_lens = torch.tensor([prefix], device="cuda", dtype=torch.int32)
            output = tilelang_verify_forward(
                q,
                k_cache,
                v_cache,
                block_table,
                seq_lens,
                query_start_loc,
                prefix_kv_lens,
                causal=causal,
            )
            repeat = tilelang_verify_forward(
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
                q,
                k_cache,
                v_cache,
                block_table,
                context=context,
                prefix=prefix,
                causal=causal,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(output, repeat, rtol=0, atol=0)
            torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
            causal_label = "1" if causal else "0"
            identity_label = "1" if identity_pages else "0"
            name = f"causal-{causal_label}-identity-{identity_label}"
            output_path = args.output / f"{name}.bin"
            output.detach().contiguous().cpu().numpy().tofile(output_path)
            delta = output.float() - expected.float()
            results.append(
                {
                    "case": name,
                    "output_sha256": _sha256(output),
                    "output_bytes": output_path.stat().st_size,
                    "reference_max_abs": delta.abs().max().item(),
                    "reference_rms": delta.square().mean().sqrt().item(),
                }
            )
    report = {
        "tilelang_version": getattr(tilelang, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "cases": results,
    }
    path = args.output / "results.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
