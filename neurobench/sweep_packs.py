"""Parameter sweep-pack builders for Architecture Lab review packs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


SWEEP_PRESETS: dict[str, dict[str, Any]] = {
    "permissive": {"event_threshold_z": 2.0, "seed_z": 1.6, "grow_z": 0.8, "min_area_px": 4},
    "balanced": {"event_threshold_z": 2.4, "seed_z": 2.0, "grow_z": 1.1, "min_area_px": 4},
    "strict": {"event_threshold_z": 2.8, "seed_z": 2.4, "grow_z": 1.4, "min_area_px": 12},
    "artifact_suppression": {"event_threshold_z": 2.6, "seed_z": 2.2, "grow_z": 1.3, "min_area_px": 8},
    "high_recall": {"event_threshold_z": 1.8, "seed_z": 1.4, "grow_z": 0.7, "min_area_px": 3},
}


def _stage(stage_id: str, index: int, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": f"{stage_id}_{index:02d}",
        "stage_id": stage_id,
        "enabled": True,
        "params": dict(params or {}),
    }


def _pipeline(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    event_threshold_z = params["event_threshold_z"]
    return [
        _stage("temporal_highpass_gaussian", 1, {"sigma_frames": 6.0}),
        _stage("event_preserving_noise_suppression", 2, {"spatial_sigma_px": 0.8, "temporal_window_frames": 3}),
        _stage("robust_positive_local_z", 3, {"local_radius_px": 11, "epsilon": 1.0}),
        _stage(
            "component_filter",
            4,
            {
                "seed_z": params["seed_z"],
                "grow_z": params["grow_z"],
                "min_area_px": params["min_area_px"],
                "max_area_px": 260,
            },
        ),
        _stage("local_background_ring", 5, {"outer_radius_px": 15, "neuropil_weight": 0.7}),
        _stage(
            "robust_kalman_positive_innovation",
            6,
            {"event_threshold_z": event_threshold_z, "kalman_gain": 0.06, "spike_gain": 0.008},
        ),
        _stage("heuristic_priority_v1", 7),
        _stage("generate_neuron_review_app", 8, {"include_discovery": True}),
    ]


def build_sweep_pack(
    *,
    dataset_id: str,
    pack_id: str = "review_pack_v1",
    label: str = "Review parameter pack v1",
    source_manifest: str | None = None,
) -> dict[str, Any]:
    """Return grouped planned architecture runs for review-pack comparison."""
    runs = []
    for preset_name, params in SWEEP_PRESETS.items():
        run_id = f"{pack_id}_{preset_name}"
        runs.append(
            {
                "schema_version": 1,
                "run_id": run_id,
                "dataset_id": dataset_id,
                "label": preset_name.replace("_", " ").title(),
                "method_family": "current_localz_review_pack",
                "purpose": "candidate_proposal",
                "pipeline": _pipeline(params),
                "parameters": deepcopy(params),
                "execution": {"status": "planned"},
                "review_pack": {
                    "pack_id": pack_id,
                    "label": label,
                    "preset": preset_name,
                    "source_manifest": source_manifest,
                    "interpretation": _interpretation(preset_name),
                },
                "summary": {"roi_count": 0, "event_count": 0, "suggestion_count": 0},
                "artifacts": {},
            }
        )
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "review_pack": {
            "pack_id": pack_id,
            "label": label,
            "goal": "Compare recall, artifact burden, and candidate stability before label-driven tuning.",
            "presets": list(SWEEP_PRESETS),
        },
        "runs": runs,
    }


def _interpretation(preset_name: str) -> str:
    if preset_name == "permissive":
        return "Higher recall; expect more artifacts and review burden."
    if preset_name == "balanced":
        return "Current baseline-style settings for comparison."
    if preset_name == "strict":
        return "Lower review burden; useful for checking which candidates remain robust."
    if preset_name == "artifact_suppression":
        return "Slightly stricter morphology/thresholds to reduce impulse and background artifacts."
    if preset_name == "high_recall":
        return "Most permissive missed-neuron audit pack; use for discovery, not final labels."
    return "Planned comparison preset."
