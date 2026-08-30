"""Deterministic input discovery and hashing."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_output_root(requested: Path, resume: bool) -> Path:
    if not requested.exists() or resume:
        return requested
    index = 2
    while requested.with_name(f"{requested.name}_v{index}").exists():
        index += 1
    return requested.with_name(f"{requested.name}_v{index}")
