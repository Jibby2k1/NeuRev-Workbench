"""Strict manifest loader for the Gamma-LS support sufficiency extension."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from neurobench.portable_paths import data_root

from .config import GammaLSDifferenceConfig
from .support_grid import (
    MODE_FRACTIONS,
    SHAPES,
    STAGE_A_RADIUS_GUARD_PAIRS,
    STAGE_B_RADIUS_GUARD_PAIRS,
)


class GammaSupportConfigError(ValueError):
    """Raised when the support-sufficiency manifest is ambiguous or changed."""


def _exact(mapping: Mapping[str, Any], expected: set[str], scope: str) -> None:
    actual = set(mapping)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise GammaSupportConfigError(
            f"invalid {scope} fields: missing={sorted(missing)} unknown={sorted(unknown)}"
        )


def _resolve(value: str, *, repository: Path, authority: Path) -> Path:
    if value.startswith("repo://"):
        return (repository / value.removeprefix("repo://")).resolve()
    if value.startswith("data://"):
        return (authority / value.removeprefix("data://")).resolve()
    raise GammaSupportConfigError("paths must use repo:// or data:// identifiers")


@dataclass(frozen=True)
class GammaSupportConfig:
    manifest_path: Path
    repository: Path
    authority: Path
    payload: dict[str, Any]
    base_config_path: Path
    base_screen_path: Path
    base_config: GammaLSDifferenceConfig

    @classmethod
    def load(cls, path: str | Path) -> "GammaSupportConfig":
        manifest = Path(path).expanduser().resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise GammaSupportConfigError("manifest must be an object")
        repository = Path(__file__).resolve().parents[3]
        authority = data_root(repository)
        _exact(
            raw,
            {
                "schema_version",
                "experiment_id",
                "sources",
                "design",
                "stopping_rule",
                "latency",
                "resources",
                "scientific_audit",
            },
            "top-level",
        )
        _exact(raw["sources"], {"base_config", "base_gpu_screen"}, "sources")
        base_config_path = _resolve(
            str(raw["sources"]["base_config"]), repository=repository, authority=authority
        )
        base_screen_path = _resolve(
            str(raw["sources"]["base_gpu_screen"]), repository=repository, authority=authority
        )
        result = cls(
            manifest_path=manifest,
            repository=repository,
            authority=authority,
            payload=raw,
            base_config_path=base_config_path,
            base_screen_path=base_screen_path,
            base_config=GammaLSDifferenceConfig.load(base_config_path),
        )
        result.validate()
        return result

    def validate(self) -> None:
        raw = self.payload
        if raw["schema_version"] != 1 or raw["experiment_id"] != "spon_ca_burst_gamma_ls_support_sufficiency_v1":
            raise GammaSupportConfigError("schema version or experiment id changed")
        design = raw["design"]
        _exact(
            design,
            {
                "stage_a_radius_guard_pairs",
                "conditional_stage_b_radius_guard_pairs",
                "shape_n",
                "mode_fraction_of_half_width",
                "primary_candidate_max_half_width_px",
                "screen_representations",
                "controls",
            },
            "design",
        )
        stage_a = tuple(tuple(map(int, row)) for row in design["stage_a_radius_guard_pairs"])
        stage_b = tuple(tuple(map(int, row)) for row in design["conditional_stage_b_radius_guard_pairs"])
        if stage_a != STAGE_A_RADIUS_GUARD_PAIRS or stage_b != STAGE_B_RADIUS_GUARD_PAIRS:
            raise GammaSupportConfigError("radial support/guard pairs changed")
        if tuple(map(float, design["shape_n"])) != SHAPES:
            raise GammaSupportConfigError("Gamma shapes changed")
        if tuple(map(float, design["mode_fraction_of_half_width"])) != MODE_FRACTIONS:
            raise GammaSupportConfigError("Gamma mode fractions changed")
        if int(design["primary_candidate_max_half_width_px"]) != 23:
            raise GammaSupportConfigError("primary candidate size cap changed")
        if tuple(design["screen_representations"]) != (
            "raw",
            "difference_signed",
            "difference_energy_normalized",
        ):
            raise GammaSupportConfigError("fixed screen representation set changed")
        if design["controls"] != [
            "legacy_exact_n9_mode35_w23_eps64",
            "signed_square_annulus_ls_h11_g3",
            "maintained_positive_box_cfar_h11_g3",
        ]:
            raise GammaSupportConfigError("named control set changed")

        stopping = raw["stopping_rule"]
        _exact(
            stopping,
            {
                "training_near_optimal_absolute_contrast",
                "training_near_optimal_relative_fraction",
                "minimum_representation_contrast_floor",
                "stage_b_trigger",
                "stage_b_max_half_width_px",
                "protected_b58_total_match_tolerance",
                "protected_per_burst_match_tolerance",
                "protected_budget_curve_auc_tolerance",
                "final_claim_requires",
            },
            "stopping_rule",
        )
        if (
            float(stopping["training_near_optimal_absolute_contrast"]) != 0.005
            or float(stopping["training_near_optimal_relative_fraction"]) != 0.05
            or float(stopping["minimum_representation_contrast_floor"]) != -0.005
            or stopping["stage_b_trigger"]
            != "h31_best_or_within_training_near_optimal_tolerance_in_any_fold"
            or int(stopping["stage_b_max_half_width_px"]) != 47
            or int(stopping["protected_b58_total_match_tolerance"]) != 1
            or int(stopping["protected_per_burst_match_tolerance"]) != 1
            or float(stopping["protected_budget_curve_auc_tolerance"]) != 0.02
            or stopping["final_claim_requires"]
            != ["training_window_contrast", "protected_recall_sensitivity", "repeated_latency"]
        ):
            raise GammaSupportConfigError("predeclared stopping rule changed")

        latency = raw["latency"]
        _exact(
            latency,
            {
                "warmup_iterations",
                "timed_iterations",
                "single_frame_deadline_ms",
                "relative_p50_cap_vs_h11",
                "absolute_p50_increase_cap_ms",
            },
            "latency",
        )
        if (
            int(latency["warmup_iterations"]) != 50
            or int(latency["timed_iterations"]) != 200
            or float(latency["single_frame_deadline_ms"]) != 1.0
            or float(latency["relative_p50_cap_vs_h11"]) != 1.25
            or float(latency["absolute_p50_increase_cap_ms"]) != 0.10
        ):
            raise GammaSupportConfigError("latency contract changed")
        resources = raw["resources"]
        _exact(resources, {"device", "chunk_frames", "max_peak_vram_gib", "minimum_free_disk_gib"}, "resources")
        if (
            resources["device"] != "cuda"
            or int(resources["chunk_frames"]) != 64
            or float(resources["max_peak_vram_gib"]) != 8.0
            or float(resources["minimum_free_disk_gib"]) != 20.0
        ):
            raise GammaSupportConfigError("resource contract changed")
        if raw["scientific_audit"] != {"enabled": True}:
            raise GammaSupportConfigError("scientific audit must remain enabled")

    def portable_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload, sort_keys=True))


__all__ = ["GammaSupportConfig", "GammaSupportConfigError"]
