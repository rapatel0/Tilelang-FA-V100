#!/usr/bin/env python3
# ruff: noqa: I001
"""Export reproducible grouped-verifier CUDA artifacts and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import tilelang

from tilelang_fa_v100._kernels_paged_verify import get_paged_verify_kernels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def _export(kernel, stem: Path) -> list[Path]:
    outputs = [
        stem.with_suffix(".cu"),
        stem.with_name(stem.name + "_host.cc"),
        stem.with_suffix(".ptx"),
        stem.with_suffix(".sass"),
        stem.with_suffix(".so"),
    ]
    kernel.export_sources(
        kernel_path=str(outputs[0]),
        host_path=str(outputs[1]),
    )
    kernel.export_ptx(str(outputs[2]))
    kernel.export_sass(str(outputs[3]))
    kernel.export_library(str(outputs[4]))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-pages", type=int, default=1024)
    parser.add_argument("--max-blocks", type=int, default=1024)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    partial, combine, max_splits = get_paged_verify_kernels(
        batch=1,
        heads=8,
        heads_kv=2,
        dim=128,
        block_size=16,
        num_pages=args.num_pages,
        max_blocks=args.max_blocks,
        causal=args.causal,
        fp8_kv=False,
        min_tokens_per_split=128,
    )
    files = _export(partial, args.output / "partial")
    files.extend(_export(combine, args.output / "combine"))

    repository = Path(__file__).resolve().parents[1]
    source = repository / "tilelang_fa_v100" / "_kernels_paged_verify.py"
    manifest = {
        "schema": 1,
        "source_repository": _command("git", "-C", str(repository), "remote", "get-url", "origin"),
        "source_commit": _command("git", "-C", str(repository), "rev-parse", "HEAD"),
        "source_sha256": _sha256(source),
        "tilelang_version": getattr(tilelang, "__version__", "unknown"),
        "cuda_toolchain": _command("nvcc", "--version"),
        "shape": {
            "batch": 1,
            "query_tokens": 8,
            "query_heads": 8,
            "kv_heads": 2,
            "head_dim": 128,
            "page_size": 16,
            "num_pages": args.num_pages,
            "max_blocks": args.max_blocks,
            "causal": args.causal,
            "max_splits": max_splits,
            "tokens_per_split": 128,
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
