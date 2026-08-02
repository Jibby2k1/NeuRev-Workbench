"""Re-evaluation with exact CPU reconstruction of promoted causal features."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neurobench.algorithms.scientific_feature_audit import (
    causal_local_correlation_feature,
)
from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_program import (
    _quiet_calibrate,
)

from .adjudication import label_view, load_tsv
from .config import HardRoiAdjudicationConfig
from .reevaluate import (
    PRIMARY_FAILURE_BUDGET,
    REVIEW_START_UI,
    _atomic_json,
    _atomic_tsv,
    _evaluate_map,
    _load_feature,
)


EXACT_CAUSAL_SPECIFICATIONS = {
    "coherence_w15": {"window_frames": 15, "lag_frames": 0},
    "propagation_lag2_w15": {"window_frames": 15, "lag_frames": 2},
}


def _exact_panel_values(
    config: HardRoiAdjudicationConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    rows = {str(row["feature_id"]): row for row in config.frozen_panel}
    carrier, carrier_provenance = _load_feature(rows["carrier_signed"])
    values: dict[str, np.ndarray] = {"carrier_signed": carrier}
    provenance: dict[str, dict[str, Any]] = {
        "carrier_signed": {
            **dict(rows["carrier_signed"]), **carrier_provenance,
            "quantitative_reproduction": "exact_preserved_tensor",
        }
    }
    for feature_id, specification in EXACT_CAUSAL_SPECIFICATIONS.items():
        reconstructed = causal_local_correlation_feature(
            carrier,
            window_frames=int(specification["window_frames"]),
            lag_frames=int(specification["lag_frames"]),
            spatial_sigma_px=2.0,
            activity_qualified=True,
        )
        reconstructed = _quiet_calibrate(reconstructed, 100)
        values[feature_id] = reconstructed
        provenance[feature_id] = {
            **dict(rows[feature_id]),
            "storage": "deterministic_cpu_reconstruction",
            "quantized": False,
            "display_clipped": False,
            "quantitative_reproduction": "exact_algorithmic_reconstruction",
            "parameters": {**specification, "spatial_sigma_px": 2.0,
                           "activity_qualified": True, "quiet_count": 100},
            "model_fit": False,
            "interpretation": (
                "Recomputed from the immutable carrier with the frozen deterministic "
                "causal operator; no model fitting or parameter selection occurred."
            ),
        }
    for feature_id in ("radial_cs_shell", "noise_vst_residual"):
        diagnostic, diagnostic_provenance = _load_feature(rows[feature_id])
        values[feature_id] = diagnostic
        provenance[feature_id] = {
            **dict(rows[feature_id]), **diagnostic_provenance,
            "quantitative_reproduction": "diagnostic_only_not_exact",
        }
    return values, provenance


def reevaluate_exact_causal(
    config: HardRoiAdjudicationConfig,
    *,
    adjudication_tsv: Path,
    output_dir: Path,
    allow_provisional: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    partial = Path(str(output_dir) + ".partial")
    if output_dir.exists() or partial.exists():
        raise FileExistsError("completed or partial re-evaluation output already exists")
    required = set(map(str, config.review["target_roi_ids"]))
    adjudication = load_tsv(
        adjudication_tsv.resolve(),
        require_adjudicated_targets=None if allow_provisional else required,
    )
    partial.mkdir(parents=True, exist_ok=False)
    panel, provenance = _exact_panel_values(config)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for feature_id, values in panel.items():
        if values.shape != (560, 340, 573):
            raise ValueError(f"frozen feature geometry differs for {feature_id}")
        for view in ("original", "confirmed", "inclusive"):
            for timing in ("original", "adjudicated"):
                labels = label_view(adjudication, view, timing)
                result, detail = _evaluate_map(
                    feature_id, values, labels, config,
                    label_view_id=view, timing_view_id=timing,
                )
                results.append(result)
                failures.extend(detail)
    payload = {
        "schema_version": 1,
        "status": "provisional_preview" if allow_provisional else "completed",
        "experiment_id": config.experiment_id,
        "adjudication_tsv": str(adjudication_tsv.resolve()),
        "adjudication_is_final": not allow_provisional,
        "model_tuning_performed": False,
        "gpu_used": False,
        "original_labels_overwritten": False,
        "unmatched_candidates_are_negatives": False,
        "review_start_frame_ui": REVIEW_START_UI,
        "primary_failure_budget": PRIMARY_FAILURE_BUDGET,
        "feature_provenance": provenance,
        "quantitative_feature_ids": [
            "carrier_signed", "coherence_w15", "propagation_lag2_w15"
        ],
        "diagnostic_only_feature_ids": [
            "radial_cs_shell", "noise_vst_residual"
        ],
        "results": results,
        "failure_class_counts": {
            reason: sum(row["failure_class_at_budget_58"] == reason for row in failures)
            for reason in sorted({row["failure_class_at_budget_58"] for row in failures})
        },
        "interpretation": (
            "Carrier, coherence, and lagged recurrence are quantitative. Radial-shell "
            "and VST residual results are diagnostic-only because only display-clipped "
            "TIFFs were preserved. Original/confirmed/inclusive label views and "
            "original/adjudicated timing are separate estimands. Precision remains "
            "unidentified outside an exhaustive field."
        ),
    }
    _atomic_json(partial / "metrics.json", payload)
    _atomic_tsv(partial / "observation_failure_audit.tsv", failures)
    _atomic_json(partial / "config.resolved.json", config.to_dict())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(output_dir)
    return payload
