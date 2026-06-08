"""Template artifact model helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from neurobench.manifests import load_json, write_json
from neurobench.validation.schemas import validate_dict


@dataclass
class TemplateSpec:
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemplateSpec":
        return cls(dict(payload))

    @classmethod
    def load_json(cls, path: str | Path) -> "TemplateSpec":
        return cls.from_dict(load_json(path))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def validate(self) -> None:
        validate_dict(self.payload, "template_spec")

    def write_json(self, path: str | Path) -> None:
        self.validate()
        write_json(path, self.payload)
