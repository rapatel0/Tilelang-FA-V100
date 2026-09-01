#!/usr/bin/env python3
# ruff: noqa: I001
"""Export reproducible grouped-verifier CUDA artifacts and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def _source_repository(repository: Path) -> str:
    branch = _command("git", "-C", str(repository), "rev-parse", "--abbrev-ref", "HEAD")
    remote = _command(
        "git", "-C", str(repository), "config", "--get", f"branch.{branch}.remote"
    )
    if remote.startswith("unavailable:") or not remote:
        remote = "origin"
    return _command("git", "-C", str(repository), "remote", "get-url", remote)


def _export(kernel, stem: Path) -> list[Path]:
    outputs = [
        stem.with_suffix(".cu"),
        stem.with_name(stem.name + "_host.cc"),
        stem.with_suffix(".ptx"),
        stem.with_suffix(".sass"),
    ]
    kernel.export_sources(
        kernel_path=str(outputs[0]),
        host_path=str(outputs[1]),
    )
    kernel.export_ptx(str(outputs[2]))
    kernel.export_sass(str(outputs[3]))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--causal", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num-pages", type=int, default=1024)
    parser.add_argument("--max-blocks", type=int, default=1024)
    parser.add_argument(
        "--source-repository", default=os.environ.get("TILELANG_FA_SOURCE_REPOSITORY")
    )
    parser.add_argument(
        "--source-commit", default=os.environ.get("TILELANG_FA_SOURCE_COMMIT")
    )
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
    source_repository = args.source_repository or _source_repository(repository)
    source_commit = args.source_commit or _command(
        "git", "-C", str(repository), "rev-parse", "HEAD"
    )
    if source_repository.startswith("unavailable:"):
        raise RuntimeError("source repository is unavailable; pass --source-repository")
    if source_commit.startswith("unavailable:"):
        raise RuntimeError("source commit is unavailable; pass --source-commit")
    manifest = {
        "schema": 1,
        "source_repository": source_repository,
        "source_commit": source_commit,
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
