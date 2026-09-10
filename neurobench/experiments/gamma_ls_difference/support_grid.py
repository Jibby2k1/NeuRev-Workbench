"""Frozen, coordinate-free design for the Gamma-LS support sufficiency study.

The original G1 screen ended at half-width 15.  This module defines a separate
extension rather than mutating that completed experiment.  Modern radial
contexts are the only primary-eligible rows.  Historical and square controls
are represented by a different type and cannot enter any selector here.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Any, Mapping, Sequence

from .grid import GammaContext, SCREEN_REPRESENTATIONS, TRAINING_FOLDS


STAGE_A_RADIUS_GUARD_PAIRS = (
    (11, 5),
    (15, 5),
    (15, 7),
    (19, 5),
    (19, 7),
    (23, 5),
    (23, 9),
    (31, 7),
    (31, 11),
)
STAGE_B_RADIUS_GUARD_PAIRS = (
    (39, 9),
    (39, 15),
    (47, 11),
    (47, 19),
)
SHAPES = (2.0, 5.0, 9.0)
MODE_FRACTIONS = (0.5, 0.75, 1.0)
PRIMARY_MAX_HALF_WIDTH = 23
STAGE_A_ENDPOINT = 31
STAGE_B_ENDPOINT = 47


class GammaSupportGridError(ValueError):
    """Raised when the support study design or evidence rows drift."""


@dataclass(frozen=True)
class ControlSpec:
    """A named diagnostic that is structurally excluded from primary selection."""

    control_id: str
    family: str
    semantics: str
    eligible_primary: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "family": self.family,
            "semantics": self.semantics,
            "eligible_primary": False,
        }


CONTROLS = (
    ControlSpec(
        "legacy_exact_n9_mode35_w23_eps64",
        "legacy_exact_gamma_local_standardization",
        "signed_square_truncated_gamma_reflect_center_excluded_additive_epsilon64",
    ),
    ControlSpec(
        "signed_square_annulus_ls_h11_g3",
        "signed_square_annulus_ls",
        "signed_uniform_square_outer_minus_square_guard_valid_reference_renormalized",
    ),
    ControlSpec(
        "maintained_positive_box_cfar_h11_g3",
        "maintained_positive_clipped_box_cfar",
        "positive_clipped_square_outer_minus_square_guard_replicate_boundary",
    ),
)


def _token(value: float) -> str:
    text = format(Decimal(str(float(value))).normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def _context(stage: str, half_width: int, guard: int, shape: float, mode: float) -> GammaContext:
    if half_width <= 0 or guard <= 0 or guard >= half_width:
        raise GammaSupportGridError("every radial support pair requires 0 < guard < half-width")
    if shape not in SHAPES or mode not in MODE_FRACTIONS:
        raise GammaSupportGridError("shape/mode left the frozen support grid")
    return GammaContext(
        context_id=(
            f"support_{stage}_h{half_width}_g{guard}"
            f"_n{_token(shape)}_m{_token(mode)}"
        ),
        stage=stage,
        half_width_px=half_width,
        guard_radius_px=guard,
        shape=shape,
        mode_fraction_of_half_width=mode,
        mode_radius_px=mode * half_width,
        support="radial_disk",
        padding="valid_renormalized_zero",
        eligible_primary=True,
    )


def enumerate_stage_contexts(stage: str) -> tuple[GammaContext, ...]:
    """Enumerate the fixed stage-A or conditional stage-B radial contexts."""

    if stage == "support_a":
        pairs = STAGE_A_RADIUS_GUARD_PAIRS
    elif stage == "support_b":
        pairs = STAGE_B_RADIUS_GUARD_PAIRS
    else:
        raise GammaSupportGridError("stage must be support_a or support_b")
    contexts = tuple(
        _context(stage, half_width, guard, shape, mode)
        for half_width, guard in pairs
        for shape in SHAPES
        for mode in MODE_FRACTIONS
    )
    expected = len(pairs) * len(SHAPES) * len(MODE_FRACTIONS)
    if len(contexts) != expected or len({row.context_id for row in contexts}) != expected:
        raise AssertionError("support contexts must be complete and uniquely named")
    return contexts


def design_counts(*, stage_b_run: bool) -> dict[str, int]:
    eligible = len(enumerate_stage_contexts("support_a"))
    if stage_b_run:
        eligible += len(enumerate_stage_contexts("support_b"))
    return {
        "eligible_contexts": eligible,
        "eligible_unique_context_arm_maps": eligible * len(SCREEN_REPRESENTATIONS),
        "eligible_fold_metric_cells": (
            eligible * len(SCREEN_REPRESENTATIONS) * len(TRAINING_FOLDS)
        ),
        "eligible_quiet_swap_rows": (
            eligible * len(SCREEN_REPRESENTATIONS) * len(TRAINING_FOLDS) * 2
        ),
        "diagnostic_controls": len(CONTROLS),
        "diagnostic_control_arm_maps": len(CONTROLS) * len(SCREEN_REPRESENTATIONS),
        "diagnostic_control_fold_metric_cells": (
            len(CONTROLS) * len(SCREEN_REPRESENTATIONS) * len(TRAINING_FOLDS)
        ),
        "diagnostic_control_quiet_swap_rows": (
            len(CONTROLS) * len(SCREEN_REPRESENTATIONS) * len(TRAINING_FOLDS) * 2
        ),
    }


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[GammaContext]
) -> tuple[dict[str, Any], ...]:
    allowed = {row.context_id: row for row in contexts if row.eligible_primary}
    if len(allowed) != len(contexts):
        raise GammaSupportGridError("primary selector received an ineligible context")
    expected = len(contexts) * len(SCREEN_REPRESENTATIONS) * len(TRAINING_FOLDS)
    if len(rows) != expected:
        raise GammaSupportGridError(
            f"incomplete support grid: expected {expected} fold cells, got {len(rows)}"
        )
    normalized: list[dict[str, Any]] = []
    keys: set[tuple[str, str, int]] = set()
    for source in rows:
        forbidden = set(source) & {
            "positive_coordinates",
            "positive_identities",
            "protected_labels",
            "known_positive_recall",
        }
        if forbidden:
            raise GammaSupportGridError(
                "support selection rows contain forbidden protected fields: "
                + ",".join(sorted(forbidden))
            )
        context_id = str(source.get("context_id", ""))
        representation = str(source.get("representation", ""))
        fold = int(source.get("training_fold", -1))
        key = (context_id, representation, fold)
        if context_id not in allowed:
            raise GammaSupportGridError("row contains a noneligible or unknown context")
        if representation not in SCREEN_REPRESENTATIONS or fold not in TRAINING_FOLDS:
            raise GammaSupportGridError("row left the fixed representation/fold design")
        if key in keys:
            raise GammaSupportGridError("duplicate context/representation/fold cell")
        keys.add(key)
        if source.get("positive_coordinates_used") is not False or source.get(
            "positive_identities_used"
        ) is not False:
            raise GammaSupportGridError("protected coordinates or identities entered selection")
        if source.get("burst_windows_used") is not True:
            raise GammaSupportGridError("support screen must disclose burst-window supervision")
        event = float(source["event_positive_tail"])
        quiet = float(source["quiet_positive_tail"])
        runtime = float(source["runtime_ms_per_frame"])
        if not all(math.isfinite(value) for value in (event, quiet, runtime)) or runtime <= 0:
            raise GammaSupportGridError("nonfinite support metric or nonpositive runtime")
        normalized.append(
            {
                "context": allowed[context_id],
                "context_id": context_id,
                "representation": representation,
                "training_fold": fold,
                "contrast": event - quiet,
                "runtime_ms_per_frame": runtime,
            }
        )
    return tuple(normalized)


def best_context_by_fold_and_width(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[GammaContext]
) -> dict[int, dict[int, dict[str, Any]]]:
    """Select shape/mode/guard separately inside each fold and width.

    Ranking is descending mean training-window contrast, descending worst-arm
    contrast, ascending measured batch runtime, then stable context id.
    """

    values = _validate_rows(rows, contexts)
    result: dict[int, dict[int, dict[str, Any]]] = {}
    for fold in TRAINING_FOLDS:
        by_width: dict[int, dict[str, Any]] = {}
        widths = sorted(
            {row.half_width_px for row in contexts if row.eligible_primary}
        )
        for width in widths:
            candidates = []
            for context in contexts:
                if context.half_width_px != width:
                    continue
                cells = [
                    row
                    for row in values
                    if row["training_fold"] == fold
                    and row["context_id"] == context.context_id
                ]
                if len(cells) != len(SCREEN_REPRESENTATIONS):
                    raise GammaSupportGridError("context is not common to all fixed arms")
                contrasts = [float(row["contrast"]) for row in cells]
                runtimes = [float(row["runtime_ms_per_frame"]) for row in cells]
                candidates.append(
                    {
                        "context": context,
                        "mean_positive_tail_contrast": sum(contrasts) / len(contrasts),
                        "minimum_representation_positive_tail_contrast": min(contrasts),
                        "mean_runtime_ms_per_frame": sum(runtimes) / len(runtimes),
                    }
                )
            candidates.sort(
                key=lambda row: (
                    -row["mean_positive_tail_contrast"],
                    -row["minimum_representation_positive_tail_contrast"],
                    row["mean_runtime_ms_per_frame"],
                    row["context"].context_id,
                )
            )
            winner = candidates[0]
            by_width[width] = {
                "context": winner["context"],
                "mean_positive_tail_contrast": winner["mean_positive_tail_contrast"],
                "minimum_representation_positive_tail_contrast": winner[
                    "minimum_representation_positive_tail_contrast"
                ],
                "mean_runtime_ms_per_frame": winner["mean_runtime_ms_per_frame"],
            }
        result[fold] = by_width
    return result


def _tolerance(best: float, *, absolute: float, relative: float) -> float:
    return max(float(absolute), abs(float(best)) * float(relative))


def stage_b_required(
    stage_a_best: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[bool, list[dict[str, Any]]]:
    """Trigger expansion if h31 is best or near-optimal in any fold."""

    diagnostics = []
    trigger = False
    for fold in TRAINING_FOLDS:
        by_width = stage_a_best[fold]
        best_score = max(float(row["mean_positive_tail_contrast"]) for row in by_width.values())
        endpoint_score = float(by_width[STAGE_A_ENDPOINT]["mean_positive_tail_contrast"])
        tolerance = _tolerance(
            best_score, absolute=absolute_tolerance, relative=relative_tolerance
        )
        near = endpoint_score >= best_score - tolerance
        trigger = trigger or near
        diagnostics.append(
            {
                "training_fold": fold,
                "best_training_contrast": best_score,
                "h31_training_contrast": endpoint_score,
                "near_optimal_tolerance": tolerance,
                "h31_is_best_or_near_optimal": near,
            }
        )
    return trigger, diagnostics


def support_boundary_status(
    all_best: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    stage_b_run: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Describe whether the predeclared largest support closes the search."""

    endpoint = STAGE_B_ENDPOINT if stage_b_run else STAGE_A_ENDPOINT
    rows = []
    unresolved = False
    for fold in TRAINING_FOLDS:
        by_width = all_best[fold]
        best_score = max(float(row["mean_positive_tail_contrast"]) for row in by_width.values())
        endpoint_score = float(by_width[endpoint]["mean_positive_tail_contrast"])
        tolerance = _tolerance(
            best_score, absolute=absolute_tolerance, relative=relative_tolerance
        )
        near = endpoint_score >= best_score - tolerance
        unresolved = unresolved or near
        rows.append(
            {
                "training_fold": fold,
                "endpoint_half_width_px": endpoint,
                "endpoint_training_contrast": endpoint_score,
                "best_training_contrast": best_score,
                "near_optimal_tolerance": tolerance,
                "endpoint_is_best_or_near_optimal": near,
            }
        )
    return {
        "stage_b_run": stage_b_run,
        "largest_tested_half_width_px": endpoint,
        "status": "unresolved_at_predeclared_maximum" if unresolved else "closed_by_clear_endpoint_inferiority",
        "requires_further_nonlabel_screen_before_sufficiency_claim": unresolved,
        "folds": rows,
    }


__all__ = [
    "CONTROLS",
    "MODE_FRACTIONS",
    "PRIMARY_MAX_HALF_WIDTH",
    "SHAPES",
    "STAGE_A_ENDPOINT",
    "STAGE_A_RADIUS_GUARD_PAIRS",
    "STAGE_B_ENDPOINT",
    "STAGE_B_RADIUS_GUARD_PAIRS",
    "ControlSpec",
    "GammaSupportGridError",
    "best_context_by_fold_and_width",
    "design_counts",
    "enumerate_stage_contexts",
    "stage_b_required",
    "support_boundary_status",
]
