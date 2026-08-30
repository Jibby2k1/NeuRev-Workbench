"""Configuration loading and path resolution."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProgramConfig:
    source: Path
    repository_root: Path
    raw: dict[str, Any]

    @property
    def data_root(self) -> Path:
        value = self.raw["paths"].get("data_root")
        return Path(value).expanduser().resolve() if value else self.repository_root

    def data_path(self, key: str) -> Path:
        return (self.data_root / self.raw["paths"][key]).resolve()

    @property
    def output_root(self) -> Path:
        return (self.repository_root / self.raw["paths"]["output_root"]).resolve()


def load_config(path: Path) -> ProgramConfig:
    source = path.resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not raw.get("stages"):
        raise ValueError("unsupported or incomplete program config")
    repository_root = source.parent.parent if source.parent.name == "examples" else Path.cwd().resolve()
    return ProgramConfig(source=source, repository_root=repository_root, raw=raw)
