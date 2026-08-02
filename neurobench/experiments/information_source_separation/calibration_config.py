"""Strict manifest for disjoint identifiability calibration."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .config import InformationSeparationConfig
from .identifiability import ALL_IDENTIFIABILITY_CASES


@dataclass(frozen=True)
class CalibrationConfig:
    schema_version: int
    experiment_id: str
    scientific_config_path: Path
    scientific_config: InformationSeparationConfig
    output_dir: Path
    calibration: dict[str, Any]
    evaluation: dict[str, Any]
    methods: tuple[dict[str, Any], ...]
    confidence: dict[str, Any]
    gate: dict[str, Any]
    resources: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationConfig":
        manifest = Path(path).resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        required = {"schema_version", "experiment_id", "scientific_config", "output_dir", "calibration", "evaluation", "methods", "confidence", "gate", "resources"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("calibration manifest fields differ")
        scientific_path = (manifest.parent / str(raw["scientific_config"])).resolve()
        result = cls(
            schema_version=int(raw["schema_version"]), experiment_id=str(raw["experiment_id"]),
            scientific_config_path=scientific_path,
            scientific_config=InformationSeparationConfig.load(scientific_path),
            output_dir=(manifest.parent / str(raw["output_dir"])).resolve(),
            calibration=dict(raw["calibration"]), evaluation=dict(raw["evaluation"]),
            methods=tuple(dict(item) for item in raw["methods"]),
            confidence=dict(raw["confidence"]), gate=dict(raw["gate"]), resources=dict(raw["resources"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1 or not self.experiment_id:
            raise ValueError("invalid calibration schema/id")
        split_fields = {"case_ids", "seeds", "snr_levels"}
        for split in (self.calibration, self.evaluation):
            if set(split) != split_fields or not split["case_ids"] or not split["seeds"] or not split["snr_levels"]:
                raise ValueError("invalid calibration split fields")
            if not set(split["case_ids"]).issubset(ALL_IDENTIFIABILITY_CASES):
                raise ValueError("unknown identifiability case")
            if any(float(value) <= 0 for value in split["snr_levels"]):
                raise ValueError("SNR must be positive")
        if set(self.calibration["case_ids"]) & set(self.evaluation["case_ids"]):
            raise ValueError("calibration/evaluation cases must be disjoint")
        if set(map(int, self.calibration["seeds"])) & set(map(int, self.evaluation["seeds"])):
            raise ValueError("calibration/evaluation seeds must be disjoint")
        expected_methods = {"pca_reference", "multilag_sobi", "kernel_hsic_pairwise_rotation", "knn_mi_pairwise_rotation"}
        if {str(item.get("method_id")) for item in self.methods} != expected_methods or any(set(item) != {"method_id", "parameters"} for item in self.methods):
            raise ValueError("frozen calibration method panel differs")
        if set(self.confidence) != {"perturbations", "perturbation_scale", "logistic_c_grid"}:
            raise ValueError("confidence fields differ")
        if int(self.confidence["perturbations"]) not in (1, 2, 3) or float(self.confidence["perturbation_scale"]) <= 0:
            raise ValueError("invalid perturbation contract")
        if set(self.gate) != {"maximum_false_resolution_count", "minimum_identifiable_resolution_rate", "minimum_converged_fraction"}:
            raise ValueError("gate fields differ")
        if int(self.gate["maximum_false_resolution_count"]) != 0:
            raise ValueError("false-resolution gate must remain zero")
        if set(self.resources) != {"device", "cpu_threads", "minimum_free_gpu_mib", "minimum_free_disk_mib", "maximum_output_mib"}:
            raise ValueError("resource fields differ")

    def split_count(self, split: str) -> int:
        values = self.calibration if split == "calibration" else self.evaluation
        return len(values["case_ids"]) * len(values["seeds"]) * len(values["snr_levels"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "experiment_id": self.experiment_id,
            "scientific_config_path": str(self.scientific_config_path),
            "scientific_config": self.scientific_config.to_dict(), "output_dir": str(self.output_dir),
            "calibration": self.calibration, "evaluation": self.evaluation,
            "methods": list(self.methods), "confidence": self.confidence,
            "gate": self.gate, "resources": self.resources,
        }
