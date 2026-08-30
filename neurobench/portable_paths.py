"""Resolve local resources while recording only portable path identifiers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def configured_root(variable: str, fallback: Path) -> Path:
    """Resolve an optional environment-owned root without embedding a workstation path."""
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else fallback.expanduser().resolve()


def data_root(repository: Path) -> Path:
    """Return the checkout or external frozen-data authority selected at runtime."""
    return configured_root("NEUROBENCH_DATA_ROOT", repository)


def media_root(repository: Path) -> Path:
    """Return the root containing project-adjacent review-media bundles."""
    return configured_root("NEUROBENCH_MEDIA_ROOT", repository.parent)


def portable_path(
    path: Path,
    *,
    repository: Path,
    data: Path | None = None,
    media: Path | None = None,
) -> str:
    """Label a path by authority instead of serializing its absolute location.

    The returned identifiers are provenance labels, not filesystem paths. Callers
    continue to hash and read the resolved ``Path`` supplied to this function.
    """
    resolved = path.expanduser().resolve()
    roots = (("repo", repository), ("data", data), ("media", media))
    for scheme, root in roots:
        if root is None:
            continue
        try:
            relative = resolved.relative_to(root.expanduser().resolve())
        except ValueError:
            continue
        return f"{scheme}://{relative.as_posix()}"

    parts = resolved.parts
    for marker in ("Outputs", "Inputs"):
        if marker in parts:
            index = parts.index(marker)
            return "data://" + "/".join(parts[index:])
    if resolved.name == "manifest.json" and resolved.parent.name.startswith("v7_"):
        return f"media://{resolved.parent.name}/{resolved.name}"
    return f"external://{resolved.name}"


def portableize_paths(
    value: Any,
    *,
    repository: Path,
    data: Path | None = None,
    media: Path | None = None,
) -> Any:
    """Recursively replace absolute path strings (including mapping keys)."""
    if isinstance(value, dict):
        return {
            portableize_paths(key, repository=repository, data=data, media=media):
            portableize_paths(item, repository=repository, data=data, media=media)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [portableize_paths(item, repository=repository, data=data, media=media) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return portable_path(Path(value), repository=repository, data=data, media=media)
    return value
