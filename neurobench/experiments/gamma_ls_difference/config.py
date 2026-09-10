"""Strict manifest loader for the Gamma-LS difference/ICA ablation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from neurobench.portable_paths import data_root


class GammaLSDifferenceConfigError(ValueError):
    """Raised when the frozen experiment manifest is ambiguous or invalid."""


_TOP_LEVEL = {
    "schema_version",
    "experiment_id",
    "sources",
    "frames",
    "representations",
    "preprocessing",
    "gamma_ls_grid",
    "cfar",
    "screen",
    "efficiency",
    "ica_search",
    "resources",
    "scientific_audit",
    "outputs",
}

_SOURCE_KEYS = {
    "movie",
    "protected_labels_v1",
    "latest_labels_v7",
    "two_frame_cs_parzen_fit",
    "multilag_v5_surface",
    "multilag_config_id",
}

_REPRESENTATIONS = (
    "raw",
    "difference_signed",
    "difference_energy_normalized",
    "pca_whitened_derivative",
    "cs_parzen_two_frame",
    "difference_multilag_energy_normalized",
    "pca_whitened_delay_total_energy",
    "cs_parzen_delay_residual",
)


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], scope: str
) -> None:
    actual = set(mapping)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unknown:
            details.append("unknown=" + ",".join(sorted(unknown)))
        raise GammaLSDifferenceConfigError(f"invalid {scope} fields: {'; '.join(details)}")


def _strictly_increasing(values: list[float], scope: str) -> None:
    if not values or any(b <= a for a, b in zip(values, values[1:])):
        raise GammaLSDifferenceConfigError(f"{scope} must be strictly increasing")


def _resolve_uri(value: str, *, repository: Path, authority: Path) -> Path:
    if value.startswith("repo://"):
        return (repository / value.removeprefix("repo://")).resolve()
    if value.startswith("data://"):
        return (authority / value.removeprefix("data://")).resolve()
    raise GammaLSDifferenceConfigError(
        f"paths must use repo:// or data:// portable identifiers, got {value!r}"
    )


@dataclass(frozen=True)
class GammaLSDifferenceConfig:
    """Validated manifest plus runtime-resolved resource paths."""

    manifest_path: Path
    repository: Path
    authority: Path
    payload: dict[str, Any]
    source_paths: dict[str, Path]
    output_root: Path

    @classmethod
    def load(cls, path: str | Path) -> "GammaLSDifferenceConfig":
        manifest = Path(path).expanduser().resolve()
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise GammaLSDifferenceConfigError("manifest must be a JSON object")
        _require_exact_keys(raw, _TOP_LEVEL, "top-level")

        repository = Path(__file__).resolve().parents[3]
        authority = data_root(repository)
        sources = raw["sources"]
        if not isinstance(sources, dict):
            raise GammaLSDifferenceConfigError("sources must be an object")
        _require_exact_keys(sources, _SOURCE_KEYS, "sources")
        source_paths = {
            key: _resolve_uri(str(sources[key]), repository=repository, authority=authority)
            for key in _SOURCE_KEYS - {"multilag_config_id"}
        }
        output_root = _resolve_uri(
            str(raw["outputs"].get("root", "")),
            repository=repository,
            authority=authority,
        )
        config = cls(
            manifest_path=manifest,
            repository=repository,
            authority=authority,
            payload=deepcopy(raw),
            source_paths=source_paths,
            output_root=output_root,
        )
        config.validate()
        return config

    def validate(self) -> None:
        raw = self.payload
        if raw["schema_version"] != 1:
            raise GammaLSDifferenceConfigError("schema_version must be exactly 1")
        if raw["experiment_id"] != "spon_ca_burst_gamma_ls_difference_ablation_v1":
            raise GammaLSDifferenceConfigError("unexpected experiment_id")
        if tuple(raw["representations"]) != _REPRESENTATIONS:
            raise GammaLSDifferenceConfigError(
                "representations must match the frozen ordered comparison set"
            )

        frames = raw["frames"]
        _require_exact_keys(
            frames,
            {
                "axes",
                "ui_one_based",
                "frame_interval_ms",
                "review_interval_ui",
                "quiet_interval_ui",
                "burst_intervals_ui",
            },
            "frames",
        )
        if frames["axes"] != "TYX" or frames["ui_one_based"] is not True:
            raise GammaLSDifferenceConfigError("version 1 requires one-based UI TYX frames")
        if float(frames["frame_interval_ms"]) != 20.0:
            raise GammaLSDifferenceConfigError("frozen source cadence must be 20 ms")
        review = [int(value) for value in frames["review_interval_ui"]]
        quiet = [int(value) for value in frames["quiet_interval_ui"]]
        if review != [1800, 2359] or quiet != [1800, 1899]:
            raise GammaLSDifferenceConfigError("review and quiet intervals changed")
        expected_bursts = {
            "1": [2003, 2026],
            "2": [2040, 2063],
            "3": [2122, 2149],
            "4": [2254, 2300],
        }
        if frames["burst_intervals_ui"] != expected_bursts:
            raise GammaLSDifferenceConfigError("burst intervals changed")

        preprocessing = raw["preprocessing"]
        _require_exact_keys(
            preprocessing,
            {
                "primary_input",
                "gaussian_sigma_px",
                "ema_alpha",
                "raw_input_energy_difference_sensitivity",
            },
            "preprocessing",
        )
        if (
            preprocessing["primary_input"]
            != "causal_gaussian_sigma1_then_ema_alpha0p4"
            or float(preprocessing["gaussian_sigma_px"]) != 1.0
            or float(preprocessing["ema_alpha"]) != 0.4
            or preprocessing["raw_input_energy_difference_sensitivity"] is not True
        ):
            raise GammaLSDifferenceConfigError("common causal preprocessing changed")

        grid = raw["gamma_ls_grid"]
        required_grid = {
            "g1_half_width_px",
            "g1_guard_radius_px",
            "g1_shape",
            "g1_mode_fraction_of_half_width",
            "g1_retain_radius_guard_pairs",
            "g2_shape",
            "g2_mode_fraction_of_half_width",
            "circular_support",
            "padding",
            "screen_scale_floor_percentile",
            "finalist_scale_floor_percentiles",
            "include_legacy_exact_anchor",
            "include_square_box_control",
            "square_box_control",
            "legacy_exact",
            "fold_local_common_context_finalists",
        }
        _require_exact_keys(grid, required_grid, "gamma_ls_grid")
        half_widths = [int(value) for value in grid["g1_half_width_px"]]
        guards = [int(value) for value in grid["g1_guard_radius_px"]]
        if half_widths != [7, 11, 15] or guards != [1, 3, 5]:
            raise GammaLSDifferenceConfigError("G1 radius/guard grid changed")
        if (
            float(grid["g1_shape"]) != 5.0
            or float(grid["g1_mode_fraction_of_half_width"]) != 0.75
            or int(grid["g1_retain_radius_guard_pairs"]) != 2
            or [float(value) for value in grid["g2_shape"]] != [2.0, 5.0, 9.0]
            or [float(value) for value in grid["g2_mode_fraction_of_half_width"]]
            != [0.5, 0.75, 1.0]
            or float(grid["screen_scale_floor_percentile"]) != 10.0
            or [float(value) for value in grid["finalist_scale_floor_percentiles"]]
            != [10.0]
        ):
            raise GammaLSDifferenceConfigError("Gamma-LS successive-halving grid changed")
        if grid["circular_support"] is not True:
            raise GammaLSDifferenceConfigError("primary Gamma-LS support must be radial disk")
        if grid["padding"] != "valid_renormalized_zero":
            raise GammaLSDifferenceConfigError("modern border contract changed")
        if not grid["include_legacy_exact_anchor"] or not grid["include_square_box_control"]:
            raise GammaLSDifferenceConfigError("diagnostic controls must remain enabled")
        legacy = grid["legacy_exact"]
        if legacy != {
            "half_width_px": 11,
            "guard_radius_px": 0,
            "shape": 9.0,
            "mode_radius_px": 35.0,
            "circular_support": False,
            "additive_epsilon": 64.0,
        }:
            raise GammaLSDifferenceConfigError("legacy_exact anchor changed")
        if grid["square_box_control"] != {
            "outer_half_width_px": 11,
            "guard_half_width_px": 3,
        }:
            raise GammaLSDifferenceConfigError("square-box diagnostic changed")

        cfar = raw["cfar"]
        _require_exact_keys(
            cfar,
            {
                "quiet_nms_peaks_per_pseudo_burst",
                "candidate_budgets_per_burst",
                "primary_budget_per_burst",
                "nms_distance_px",
                "nms_sensitivity_px",
                "match_radius_px",
                "burst_aggregation",
                "unmatched_candidates",
            },
            "cfar",
        )
        _strictly_increasing(
            [float(value) for value in cfar["quiet_nms_peaks_per_pseudo_burst"]],
            "quiet CFAR burden grid",
        )
        if [float(value) for value in cfar["quiet_nms_peaks_per_pseudo_burst"]] != [
            0.25,
            0.5,
            1.0,
            2.0,
            5.0,
        ]:
            raise GammaLSDifferenceConfigError("quiet CFAR burden grid changed")
        budgets = [int(value) for value in cfar["candidate_budgets_per_burst"]]
        _strictly_increasing([float(value) for value in budgets], "candidate budgets")
        if budgets != [20, 40, 58, 80, 100]:
            raise GammaLSDifferenceConfigError("candidate-budget grid changed")
        if int(cfar["primary_budget_per_burst"]) != 58:
            raise GammaLSDifferenceConfigError("B58 must remain the primary per-burst budget")
        if (
            int(cfar["nms_distance_px"]) != 6
            or [int(value) for value in cfar["nms_sensitivity_px"]] != [4, 8]
            or int(cfar["match_radius_px"]) != 6
            or cfar["unmatched_candidates"] != "unknown"
        ):
            raise GammaLSDifferenceConfigError("sparse-positive spatial contract changed")
        if cfar["burst_aggregation"] != "temporal_threshold_occupancy":
            raise GammaLSDifferenceConfigError("temporal max pooling is excluded in version 1")

        screen = raw["screen"]
        _require_exact_keys(
            screen,
            {
                "outer_folds",
                "quiet_calibration",
                "heldout_guard_frames",
                "burst_windows_used",
                "selection_uses_positive_coordinates",
                "selection_uses_positive_identities",
                "fold_selection",
                "scale_floor_fit",
                "gamma_context_selection_representations",
                "positive_tail_quantile",
                "positive_tail_spatial_population",
                "event_tail_aggregation",
                "quiet_tail_aggregation",
                "selection_score",
            },
            "screen",
        )
        if (
            screen["outer_folds"] != "leave_one_burst_out"
            or screen["quiet_calibration"]
            != "split_contiguous_halves_and_reverse_roles"
            or screen["burst_windows_used"] is not True
            or screen["selection_uses_positive_coordinates"] is not False
            or screen["selection_uses_positive_identities"] is not False
            or int(screen["heldout_guard_frames"]) != 10
        ):
            raise GammaLSDifferenceConfigError("protected selection contract changed")
        if screen.get("gamma_context_selection_representations") != [
            "raw",
            "difference_signed",
            "difference_energy_normalized",
        ]:
            raise GammaLSDifferenceConfigError(
                "Gamma context selection must use only the three frozen nonlearned arms"
            )
        if float(screen["positive_tail_quantile"]) != 0.999:
            raise GammaLSDifferenceConfigError("positive-tail quantile changed")
        if screen["positive_tail_spatial_population"] != "all_pixels":
            raise GammaLSDifferenceConfigError("positive-tail spatial population changed")
        if (
            screen["fold_selection"] != "independent_per_outer_heldout_burst"
            or screen["scale_floor_fit"]
            != "per_context_representation_outer_fold_training_quiet_half"
            or screen["event_tail_aggregation"]
            != "mean_of_three_training_burst_quantiles_and_two_quiet_half_swaps"
            or screen["quiet_tail_aggregation"]
            != "mean_of_two_heldout_quiet_half_quantiles"
        ):
            raise GammaLSDifferenceConfigError("Gamma context fold construction changed")
        if (
            screen["selection_score"]
            != "training_burst_vs_heldout_quiet_q0p999_contrast_with_runtime_pareto"
        ):
            raise GammaLSDifferenceConfigError("Gamma context selection score changed")
        if int(grid["fold_local_common_context_finalists"]) != 1:
            raise GammaLSDifferenceConfigError(
                "each outer fold must retain exactly one common Gamma context"
            )

        efficiency = raw["efficiency"]
        _require_exact_keys(
            efficiency,
            {
                "gpu_required",
                "workers_per_gpu",
                "frame_chunks",
                "warmup_iterations",
                "timed_iterations",
                "streaming_deadline_ms",
                "separate_fit_from_inference",
                "include_transfer_and_sync",
            },
            "efficiency",
        )
        if efficiency["gpu_required"] is not True:
            raise GammaLSDifferenceConfigError("the version-1 experiment is GPU-only")
        chunks = [int(value) for value in efficiency["frame_chunks"]]
        if chunks != [1, 8, 32, 64]:
            raise GammaLSDifferenceConfigError("timing chunk grid changed")
        if float(efficiency["streaming_deadline_ms"]) != 1.0:
            raise GammaLSDifferenceConfigError("1-kHz deadline must remain one millisecond")
        if (
            int(efficiency["workers_per_gpu"]) != 1
            or int(efficiency["warmup_iterations"]) != 50
            or int(efficiency["timed_iterations"]) != 2000
            or not efficiency["separate_fit_from_inference"]
            or not efficiency["include_transfer_and_sync"]
        ):
            raise GammaLSDifferenceConfigError("timing boundary changed")

        resources = raw["resources"]
        _require_exact_keys(
            resources,
            {
                "device",
                "cpu_threads",
                "max_peak_vram_gib",
                "max_peak_ram_gib",
                "minimum_free_disk_gib",
            },
            "resources",
        )
        device = resources["device"]
        cuda_selector = isinstance(device, str) and re.fullmatch(
            r"cuda(?::(?:0|[1-9][0-9]*))?", device
        )
        if cuda_selector is None or int(resources["cpu_threads"]) != 4:
            raise GammaLSDifferenceConfigError("version 1 requires one worker on CUDA")
        if (
            float(resources["max_peak_vram_gib"]) != 8.0
            or float(resources["max_peak_ram_gib"]) != 24.0
            or float(resources["minimum_free_disk_gib"]) != 20.0
        ):
            raise GammaLSDifferenceConfigError("resource caps changed")
        if min(
            float(resources["max_peak_vram_gib"]),
            float(resources["max_peak_ram_gib"]),
            float(resources["minimum_free_disk_gib"]),
        ) <= 0:
            raise GammaLSDifferenceConfigError("resource bounds must be positive")
        if raw["scientific_audit"] != {"enabled": True}:
            raise GammaLSDifferenceConfigError("scientific audit is mandatory")
        _require_exact_keys(
            raw["ica_search"],
            {
                "fit_scope",
                "frozen_source_fits_role",
                "multilag_refit_stage",
                "two_frame_cs_parzen_bandwidth",
                "paired_sample_seeds",
                "pairwise_total_outer_fold_fits",
                "screen_samples",
                "confirmation_samples",
                "coarse_step_degrees",
                "refine_half_width_degrees",
                "refine_step_degrees",
                "kernel_block_rows",
            },
            "ica_search",
        )
        if raw["ica_search"]["two_frame_cs_parzen_bandwidth"] != [0.25, 0.35, 0.5]:
            raise GammaLSDifferenceConfigError("two-frame CS-Parzen bandwidth grid changed")
        if (
            raw["ica_search"]["fit_scope"]
            != "per_outer_fold_training_frames_excluding_heldout_burst_and_guard"
            or raw["ica_search"]["frozen_source_fits_role"]
            != "implementation_parity_and_transductive_diagnostic_only"
            or raw["ica_search"]["multilag_refit_stage"]
            != "deferred_until_adjacent_or_fixed_multilag_gate"
            or int(raw["ica_search"]["pairwise_total_outer_fold_fits"]) != 36
        ):
            raise GammaLSDifferenceConfigError("ICA fit-scope contract changed")
        if [int(value) for value in raw["ica_search"]["paired_sample_seeds"]] != [
            20260805,
            20260806,
            20260807,
        ]:
            raise GammaLSDifferenceConfigError("paired ICA seed set changed")
        if (
            int(raw["ica_search"]["screen_samples"]) != 1024
            or int(raw["ica_search"]["confirmation_samples"]) != 4096
            or float(raw["ica_search"]["coarse_step_degrees"]) != 3.0
            or float(raw["ica_search"]["refine_half_width_degrees"]) != 3.0
            or float(raw["ica_search"]["refine_step_degrees"]) != 0.25
            or int(raw["ica_search"]["kernel_block_rows"]) != 256
        ):
            raise GammaLSDifferenceConfigError("ICA search grid changed")
        if set(raw["outputs"]) != {"root"}:
            raise GammaLSDifferenceConfigError("outputs accepts only root")

    @property
    def experiment_id(self) -> str:
        return str(self.payload["experiment_id"])

    def portable_dict(self) -> dict[str, Any]:
        """Return the original portable manifest without host-local paths."""
        return deepcopy(self.payload)


__all__ = [
    "GammaLSDifferenceConfig",
    "GammaLSDifferenceConfigError",
]
