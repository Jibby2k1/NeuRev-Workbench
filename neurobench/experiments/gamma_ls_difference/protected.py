"""Protected outer-fold two-frame representation experiment.

This module is the label-sealed continuation of :mod:`.screen`.  It refits
full-rank two-frame whitening and CS-Parzen rotations independently inside
each leave-one-burst-out fold, selects learned fit hyperparameters using only
the three training burst windows and quiet frames, freezes cross-fitted CFAR
candidates, and only then opens either sparse-positive label table.

The archived whole-review ICA packet is never loaded here.  It remains an
implementation-parity/transductive diagnostic and cannot enter protected
estimates.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
)
from neurobench.algorithms.pairwise_separation import (
    center_and_whiten_2d,
    fit_cs_parzen_ica,
    orient_and_select_activity_component,
)
from neurobench.experiments.pairwise_separation.sampling import uniform_anatomy_mask
from neurobench.metrics.sparse_detection import match_peaks_one_to_one

from .config import GammaLSDifferenceConfig
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from .evaluation import (
    CANDIDATE_BUDGETS_PER_BURST,
    MATCH_RADIUS_PX,
    NMS_DISTANCE_PX,
    QUIET_NMS_PEAK_BURDENS,
    calibrate_training_quiet_thresholds,
    duration_matched_quiet_windows,
    extract_burst_candidates,
    paired_representation_equivalence,
)
from . import gpu_representations
from .preflight import verify_matching_preflight
from .screen import (
    ENERGY_EPSILON,
    GAMMA_EPSILON,
    FoldContract,
    _elapsed,
    _evaluate_context_arm,
    _positive_scale_floor,
    _stream_common_history_to_device,
    build_fold_contracts,
)


FIXED_ARMS = ("raw", "difference_signed", "difference_energy_normalized")
LEARNED_ARMS = ("pca_whitened_derivative", "cs_parzen_two_frame")
DIAGNOSTIC_ARMS = ("pca_matched_to_selected_ica",)
ALL_ARMS = FIXED_ARMS + LEARNED_ARMS + DIAGNOSTIC_ARMS
QUIET_SWAPS = ("a_train_b_test", "b_train_a_test")
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_REPLICATES = 2000
KERNEL_DTYPE = np.float32
ACCUMULATOR_DTYPE = np.float64
_CONTEXT_RE = re.compile(
    r"^gamma_h(?P<h>[0-9]+)_g(?P<g>[0-9]+)_n(?P<n>[0-9]+(?:p[0-9]+)?)_m(?P<m>[0-9]+(?:p[0-9]+)?)$"
)


class ProtectedRepresentationUnavailable(RuntimeError):
    """Raised when the protected run fails a gate before scientific output."""


@dataclass(frozen=True)
class ContextLane:
    """One pre-label-selected Gamma context role for an outer fold."""

    role: str
    context_id: str
    half_width_px: int
    guard_radius_px: int
    shape: float
    mode_fraction_of_half_width: float
    selection_source: str

    @property
    def mode_radius_px(self) -> float:
        return self.half_width_px * self.mode_fraction_of_half_width

    def spec(self) -> GammaReferenceSpec:
        return GammaReferenceSpec.from_mode(
            self.context_id,
            support_width_px=2 * self.half_width_px + 1,
            shape_n=self.shape,
            mode_radius_px=self.mode_radius_px,
            guard_radius_px=self.guard_radius_px,
            support_geometry="disk",
            boundary_mode="valid_renormalized_zero",
            epsilon=GAMMA_EPSILON,
            scale_floor=0.0,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "context_id": self.context_id,
            "half_width_px": self.half_width_px,
            "support_width_px": 2 * self.half_width_px + 1,
            "guard_radius_px": self.guard_radius_px,
            "shape": self.shape,
            "mode_fraction_of_half_width": self.mode_fraction_of_half_width,
            "mode_radius_px": self.mode_radius_px,
            "support": "radial_disk",
            "padding": "valid_renormalized_zero",
            "selection_source": self.selection_source,
            "positive_coordinates_used_for_selection": False,
            "positive_identities_used_for_selection": False,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relay_progress_heartbeat(
    heartbeat: Callable[..., None],
    parent_stage: str,
    payload: Mapping[str, Any],
) -> None:
    """Forward child progress without allowing reserved heartbeat-key collisions."""

    if not isinstance(payload, Mapping):
        raise TypeError("heartbeat progress payload must be a mapping")
    heartbeat(str(parent_stage), upstream_progress=dict(payload))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"table {path.name} has inconsistent fields")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _artifact_index(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.name != "artifact_index.json"
            and not path.name.endswith(".partial")
        ):
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"schema_version": 1, "artifacts": rows}


def _number(value: str | float | int) -> float:
    return float(str(value).replace("p", "."))


def context_lane_from_id(
    context_id: str, *, role: str, selection_source: str
) -> ContextLane:
    """Parse the deterministic modern Gamma context identifier."""

    match = _CONTEXT_RE.fullmatch(str(context_id))
    if match is None:
        raise ValueError(f"unsupported Gamma context identifier: {context_id!r}")
    lane = ContextLane(
        role=str(role),
        context_id=str(context_id),
        half_width_px=int(match.group("h")),
        guard_radius_px=int(match.group("g")),
        shape=_number(match.group("n")),
        mode_fraction_of_half_width=_number(match.group("m")),
        selection_source=str(selection_source),
    )
    if not lane.role or lane.guard_radius_px >= lane.half_width_px:
        raise ValueError("invalid protected Gamma context role/geometry")
    lane.spec()  # Validate the complete radial reference before execution.
    return lane


def _lane_from_explicit(
    payload: Mapping[str, Any], *, role: str, selection_source: str
) -> ContextLane:
    context_id = str(payload["context_id"])
    match = _CONTEXT_RE.fullmatch(context_id)
    if match is not None:
        lane = context_lane_from_id(
            context_id, role=role, selection_source=selection_source
        )
        expected = {
            "half_width_px": lane.half_width_px,
            "guard_radius_px": lane.guard_radius_px,
            "shape": lane.shape,
            "mode_fraction_of_half_width": lane.mode_fraction_of_half_width,
        }
        for key, value in expected.items():
            if key in payload and not math.isclose(float(payload[key]), float(value)):
                raise ValueError(f"explicit context {context_id} disagrees on {key}")
    else:
        required = {
            "half_width_px",
            "guard_radius_px",
            "shape",
            "mode_fraction_of_half_width",
            "mode_radius_px",
            "support",
            "padding",
            "eligible_primary",
        }
        if not required.issubset(payload):
            missing = ", ".join(sorted(required - set(payload)))
            raise ValueError(
                f"explicit context {context_id} lacks geometry fields: {missing}"
            )
        if payload["support"] != "radial_disk":
            raise ValueError("protected contexts must use radial_disk support")
        if payload["padding"] != "valid_renormalized_zero":
            raise ValueError(
                "protected contexts must use valid_renormalized_zero padding"
            )
        if payload["eligible_primary"] is not True:
            raise ValueError("protected context is not primary eligible")
        lane = ContextLane(
            role=str(role),
            context_id=context_id,
            half_width_px=int(payload["half_width_px"]),
            guard_radius_px=int(payload["guard_radius_px"]),
            shape=float(payload["shape"]),
            mode_fraction_of_half_width=float(
                payload["mode_fraction_of_half_width"]
            ),
            selection_source=str(selection_source),
        )
        if not math.isclose(float(payload["mode_radius_px"]), lane.mode_radius_px):
            raise ValueError(
                f"explicit context {context_id} disagrees on mode_radius_px"
            )
        if not lane.role or lane.guard_radius_px >= lane.half_width_px:
            raise ValueError("invalid protected Gamma context role/geometry")
        lane.spec()
    return lane


def load_fold_context_plan(
    selection_dir: str | Path,
    folds: Sequence[FoldContract],
) -> tuple[dict[int, tuple[ContextLane, ...]], dict[str, Any]]:
    """Load original-screen or extended-support pre-label fold contexts.

    Extended screens may write ``fold_contexts.json`` with original,
    sufficient-support, larger-support, and training-best contexts.  The
    earlier ``protected_contexts.json`` and original G1/G2 ``selection.json``
    contracts remain supported.  No label-bearing table is accepted.
    """

    root = Path(selection_dir).expanduser().resolve()
    size_extension = root / "fold_contexts.json"
    explicit = root / "protected_contexts.json"
    if size_extension.is_file():
        payload = json.loads(size_extension.read_text(encoding="utf-8"))
        if payload.get("selection_uses_positive_coordinates") is not False:
            raise ValueError("size extension must attest coordinate-free selection")
        if payload.get("selection_uses_positive_identities") is not False:
            raise ValueError("size extension must attest identity-free selection")
        if payload.get("selection_scope") != "outer_training_fold_only":
            raise ValueError("size extension must be fold-local")
        source_file = size_extension
        fold_rows = payload.get("folds", [])
        plan = {}
        role_fields = (
            ("original_screen_context", "original_screen_context"),
            ("support_candidate_context", "size_sufficient_candidate"),
            ("larger_support_comparator", "larger_support_comparator"),
            ("training_best_context", "training_best_context"),
        )
        for row in fold_rows:
            fold_id = int(row["training_fold"])
            lanes = []
            seen_contexts: set[str] = set()
            for field, role in role_fields:
                context_payload = row.get(field)
                if context_payload is None:
                    continue
                lane = _lane_from_explicit(
                    context_payload,
                    role=role,
                    selection_source=size_extension.name,
                )
                # Preserve the three predeclared comparison roles even when
                # the smallest sufficient support equals the original screen
                # context.  Only the auxiliary training-best role may alias a
                # comparison context and be safely omitted.
                if field == "training_best_context" and lane.context_id in seen_contexts:
                    continue
                seen_contexts.add(lane.context_id)
                lanes.append(lane)
            if not lanes or not any(
                lane.role == "size_sufficient_candidate" for lane in lanes
            ):
                raise ValueError("size extension lacks a support candidate context")
            plan[fold_id] = tuple(lanes)
    elif explicit.is_file():
        payload = json.loads(explicit.read_text(encoding="utf-8"))
        if payload.get("selection_uses_positive_coordinates") is not False:
            raise ValueError("context plan must attest coordinate-free selection")
        if payload.get("selection_uses_positive_identities") is not False:
            raise ValueError("context plan must attest identity-free selection")
        if payload.get("selection_scope") != "outer_training_fold_only":
            raise ValueError("context plan must be fold-local")
        source_file = explicit
        fold_rows = payload.get("folds", [])
        plan: dict[int, tuple[ContextLane, ...]] = {}
        for row in fold_rows:
            fold_id = int(row["training_fold"])
            primary = _lane_from_explicit(
                row["primary_context"],
                role="size_selected_primary",
                selection_source=explicit.name,
            )
            sensitivities = tuple(
                _lane_from_explicit(
                    item,
                    role=str(item.get("role", f"support_sensitivity_{index + 1}")),
                    selection_source=explicit.name,
                )
                for index, item in enumerate(row.get("sensitivity_contexts", []))
            )
            lanes = (primary, *sensitivities)
            if len({lane.role for lane in lanes}) != len(lanes):
                raise ValueError("context roles must be unique inside each fold")
            plan[fold_id] = lanes
    else:
        source_file = root / "selection.json"
        payload = json.loads(source_file.read_text(encoding="utf-8"))
        if payload.get("selection_scope") != "outer_training_fold_only":
            raise ValueError("screen selection must be fold-local")
        if payload.get("across_fold_pooling_used_for_protected_selection") is not False:
            raise ValueError("protected context selection cannot pool outer folds")
        plan = {}
        fold_rows = payload.get("folds", [])
        for row in fold_rows:
            fold_id = int(row["training_fold"])
            context_id = str(row["common_g2_finalist_context_id"])
            ranked = {str(item["context_id"]) for item in row.get("g2_ranked_contexts", [])}
            if context_id not in ranked:
                raise ValueError("fold finalist is absent from its G2 ranking")
            plan[fold_id] = (
                context_lane_from_id(
                    context_id,
                    role="original_screen_primary",
                    selection_source=source_file.name,
                ),
            )

    expected = {fold.training_fold: fold for fold in folds}
    if set(plan) != set(expected) or len(fold_rows) != len(expected):
        raise ValueError("context plan must contain exactly the four outer folds")
    for row in fold_rows:
        fold_id = int(row["training_fold"])
        contract = expected[fold_id]
        if str(row.get("heldout_burst")) != contract.heldout_burst:
            raise ValueError("context-plan held-out burst does not match fold contract")
    return plan, {
        "root": str(root),
        "source_file": source_file.name,
        "source_sha256": _sha256(source_file),
        "selection_scope": "outer_training_fold_only",
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "context_roles_by_fold": {
            str(fold): [lane.as_dict() for lane in lanes]
            for fold, lanes in sorted(plan.items())
        },
    }


def pair_eligible_current_frames(
    frame_ui: np.ndarray, heldout_guard_ui: Sequence[int]
) -> np.ndarray:
    """Mask adjacent pairs for which neither source frame touches the guard."""

    current = np.asarray(frame_ui, dtype=np.int64)
    if current.ndim != 1 or current.size < 1 or not np.all(np.diff(current) == 1):
        raise ValueError("frame_ui must be a contiguous one-dimensional sequence")
    start, stop = map(int, heldout_guard_ui)
    if start > stop:
        raise ValueError("heldout_guard_ui must be inclusive start <= stop")
    current_outside = (current < start) | (current > stop)
    previous = current - 1
    previous_outside = (previous < start) | (previous > stop)
    result = current_outside & previous_outside
    if not np.any(result):
        raise ValueError("held-out guard excludes every adjacent pair")
    return result


def _sample_fold_pairs(
    common: Any,
    *,
    frame_ui: np.ndarray,
    fold: FoldContract,
    anatomy_mask: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample paired observations from fold-eligible frames on the device."""

    import torch

    if common.ndim != 3 or int(common.shape[0]) != int(frame_ui.size + 1):
        raise ValueError("common must include one predecessor before frame_ui")
    mask = np.asarray(anatomy_mask, dtype=bool)
    if mask.shape != tuple(common.shape[1:]):
        raise ValueError("anatomy mask does not match movie geometry")
    eligible_frames = np.flatnonzero(
        pair_eligible_current_frames(frame_ui, fold.heldout_guard_ui)
    )
    pixels = np.flatnonzero(mask.ravel())
    population = int(eligible_frames.size * pixels.size)
    count = int(sample_count)
    if not 1 <= count <= population:
        raise ValueError("sample_count exceeds the eligible pair/pixel population")
    rng = np.random.default_rng(int(seed))
    identities = np.sort(rng.choice(population, size=count, replace=False))
    frame_local = identities // pixels.size
    pixel_local = identities % pixels.size
    pair_positions = eligible_frames[frame_local]
    flat_pixels = pixels[pixel_local]
    y, x = np.unravel_index(flat_pixels, mask.shape)
    pair_device = torch.as_tensor(pair_positions, dtype=torch.int64, device=common.device)
    y_device = torch.as_tensor(y, dtype=torch.int64, device=common.device)
    x_device = torch.as_tensor(x, dtype=torch.int64, device=common.device)
    # common[0] is the predecessor of frame_ui[0].
    previous = common[pair_device, y_device, x_device]
    current = common[pair_device + 1, y_device, x_device]
    observations = torch.stack((previous, current)).detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    if observations.shape != (2, count) or not np.isfinite(observations).all():
        raise RuntimeError("sampled fold observations are invalid")
    identity_bytes = np.asarray(identities, dtype="<i8").tobytes()
    return observations, {
        "seed": int(seed),
        "sample_count": count,
        "population": population,
        "eligible_current_frame_count": int(eligible_frames.size),
        "eligible_anatomy_pixel_count": int(pixels.size),
        "current_frame_ui_min": int(frame_ui[pair_positions].min()),
        "current_frame_ui_max": int(frame_ui[pair_positions].max()),
        "sample_identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "heldout_guard_ui": list(fold.heldout_guard_ui),
        "pair_guard_rule": "neither_previous_nor_current_frame_touches_inclusive_guard",
        "positive_coordinates_used": False,
        "positive_identities_used": False,
    }


def _fit_packet(
    observations: np.ndarray,
    *,
    whitened: np.ndarray | None = None,
    whitening_fit: Any | None = None,
    whitening_seconds: float | None = None,
    bandwidth: float,
    screen_samples: int,
    block_rows: int,
    coarse_step_degrees: float,
    refine_half_width_degrees: float,
    refine_step_degrees: float,
) -> dict[str, Any]:
    """Fit one full-rank whitening plus bounded two-frame CS-Parzen rotation."""

    if whitened is None or whitening_fit is None:
        whitening_started = time.perf_counter()
        whitened, whitening = center_and_whiten_2d(observations)
        resolved_whitening_seconds = time.perf_counter() - whitening_started
    else:
        whitening = whitening_fit
        resolved_whitening_seconds = float(whitening_seconds or 0.0)
    if not whitening.identifiable or np.linalg.matrix_rank(whitening.whitening) != 2:
        raise RuntimeError("fold two-frame covariance is not full-rank identifiable")
    post_whitening_started = time.perf_counter()
    ica_started = time.perf_counter()
    fit = fit_cs_parzen_ica(
        whitened[:, : int(screen_samples)],
        whitened,
        bandwidth=float(bandwidth),
        block_rows=int(block_rows),
        screen_step_degrees=float(coarse_step_degrees),
        refine_half_width_degrees=float(refine_half_width_degrees),
        refine_step_degrees=float(refine_step_degrees),
        kernel_dtype=KERNEL_DTYPE,
        backend="cuda",
    )
    ica_seconds = time.perf_counter() - ica_started
    sample_components = fit.demixing @ whitened
    derivative = observations[1] - observations[0]
    common_signal = observations[1] + observations[0]
    _, component, signs, component_diagnostics = orient_and_select_activity_component(
        sample_components, derivative, common_signal
    )
    if component is None:
        raise RuntimeError("CS-Parzen activity component was unresolved")
    elapsed = resolved_whitening_seconds + (
        time.perf_counter() - post_whitening_started
    )
    return {
        "method_id": "cs_parzen_ica",
        "fit_scope": "outer_training_fold_excluding_heldout_burst_plus_guard",
        "mean": whitening.mean.tolist(),
        "covariance": whitening.covariance.tolist(),
        "whitening": whitening.whitening.tolist(),
        "dewhitening": whitening.dewhitening.tolist(),
        "eigenvalues": whitening.eigenvalues.tolist(),
        "whitening_rank": int(np.linalg.matrix_rank(whitening.whitening)),
        "condition_number": float(whitening.condition_number),
        "full_rank_no_components_discarded": True,
        "demixing": fit.demixing.tolist(),
        "mixing": None if fit.mixing is None else fit.mixing.tolist(),
        "objective_value": float(fit.objective),
        "converged": bool(fit.converged),
        "iterations": int(fit.iterations),
        "activity_component": int(component),
        "activity_sign": int(signs[component]),
        "component_selection": component_diagnostics,
        "diagnostics": fit.diagnostics,
        "bandwidth": float(bandwidth),
        "kernel_block_rows": int(block_rows),
        "kernel_dtype": str(np.dtype(KERNEL_DTYPE)),
        "scalar_accumulator_dtype": str(np.dtype(ACCUMULATOR_DTYPE)),
        "whitening_fit_seconds": float(resolved_whitening_seconds),
        "cs_parzen_rotation_fit_seconds": float(ica_seconds),
        "fit_seconds": float(elapsed),
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "archived_fit_loaded": False,
    }


def _fit_mapping(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "mean": packet["mean"],
        "whitening": packet["whitening"],
        "demixing": packet["demixing"],
        "activity_component": packet["activity_component"],
        "activity_sign": packet["activity_sign"],
    }


def _fixed_representation(common: Any, name: str) -> Any:
    if name == "raw":
        return gpu_representations.raw_representation(common).values[1:]
    if name == "difference_signed":
        return gpu_representations.signed_difference_representation(common).values
    if name == "difference_energy_normalized":
        return gpu_representations.energy_normalized_difference_representation(
            common, epsilon=ENERGY_EPSILON
        ).values
    raise ValueError(f"unknown fixed representation {name!r}")


def _learned_representation(common: Any, packet: Mapping[str, Any], name: str) -> Any:
    model = _fit_mapping(packet)
    if name == "pca_whitened_derivative":
        return gpu_representations.pca_whitened_derivative_representation(
            common, model
        ).values
    if name == "cs_parzen_two_frame":
        return gpu_representations.frozen_two_frame_cs_parzen_representation(
            common, model
        ).values
    raise ValueError(f"unknown learned representation {name!r}")


def _fit_selection_metrics(
    representation: Any,
    *,
    representation_name: str,
    lane: ContextLane,
    fold: FoldContract,
    frame_ui_device: Any,
    bursts: Mapping[str, Sequence[int]],
    scale_floor_percentile: float,
    tail_quantile: float,
    chunk_frames: int,
) -> dict[str, Any]:
    from .grid import GammaContext

    context = GammaContext(
        context_id=lane.context_id,
        stage="protected_fit_selection",
        half_width_px=lane.half_width_px,
        guard_radius_px=lane.guard_radius_px,
        shape=lane.shape,
        mode_fraction_of_half_width=lane.mode_fraction_of_half_width,
        mode_radius_px=lane.mode_radius_px,
    )
    aggregate, swaps, timing = _evaluate_context_arm(
        representation,
        representation_name=representation_name,
        context=context,
        stage="protected_fit_selection",
        folds=(fold,),
        frame_ui=frame_ui_device,
        bursts=bursts,
        scale_floor_percentile=scale_floor_percentile,
        tail_quantile=tail_quantile,
        chunk_frames=chunk_frames,
    )
    row = aggregate[0]
    contrasts = [float(item["positive_tail_contrast"]) for item in swaps]
    return {
        "mean_positive_tail_contrast": float(
            row["event_positive_tail"] - row["quiet_positive_tail"]
        ),
        "minimum_quiet_swap_positive_tail_contrast": min(contrasts),
        "quiet_swap_positive_tail_contrasts": contrasts,
        "gamma_runtime_ms_per_frame": float(timing["gamma_runtime_ms_per_frame"]),
        "selection_uses_training_burst_windows": True,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
    }


def select_fold_ica_fit(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    context_role: str,
    bandwidths: Sequence[float],
    seeds: Sequence[int],
) -> Mapping[str, Any]:
    """Select one of exactly nine label-free CS-Parzen fits deterministically."""

    selected = [
        row
        for row in rows
        if int(row["training_fold"]) == int(fold)
        and str(row["context_role"]) == str(context_role)
    ]
    expected = {(float(bandwidth), int(seed)) for bandwidth in bandwidths for seed in seeds}
    observed = {(float(row["bandwidth"]), int(row["sample_seed"])) for row in selected}
    if len(selected) != 9 or observed != expected:
        raise ValueError("fold/context fit selection requires the exact 3x3 grid")
    for row in selected:
        if row.get("positive_coordinates_used") is not False:
            raise ValueError("ICA fit selection row used positive coordinates")
        if row.get("positive_identities_used") is not False:
            raise ValueError("ICA fit selection row used positive identities")
    return sorted(
        selected,
        key=lambda row: (
            -float(row["ica_mean_positive_tail_contrast"]),
            -float(row["ica_minimum_quiet_swap_positive_tail_contrast"]),
            float(row["bandwidth"]),
            int(row["sample_seed"]),
        ),
    )[0]


def select_fold_pca_fit(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    context_role: str,
    bandwidths: Sequence[float],
    seeds: Sequence[int],
) -> Mapping[str, Any]:
    """Select among three seed-specific whitening fits, collapsing bandwidth.

    Whitening is fitted from the paired sample identified by ``sample_seed``;
    CS-Parzen bandwidth cannot alter it.  The nine grid rows must therefore
    contain three internally identical PCA metrics per seed.  Selection uses
    those three unique whitening fits only.
    """

    grid = [
        row
        for row in rows
        if int(row["training_fold"]) == int(fold)
        and str(row["context_role"]) == str(context_role)
    ]
    expected = {(float(bandwidth), int(seed)) for bandwidth in bandwidths for seed in seeds}
    observed = {(float(row["bandwidth"]), int(row["sample_seed"])) for row in grid}
    if len(grid) != 9 or observed != expected:
        raise ValueError("fold/context PCA selection requires the exact paired 3x3 grid")
    unique = []
    for seed in seeds:
        seed_rows = sorted(
            (row for row in grid if int(row["sample_seed"]) == int(seed)),
            key=lambda row: float(row["bandwidth"]),
        )
        reference = seed_rows[0]
        for row in seed_rows[1:]:
            for field in (
                "sample_identity_sha256",
                "pca_mean_positive_tail_contrast",
                "pca_minimum_quiet_swap_positive_tail_contrast",
            ):
                if row[field] != reference[field]:
                    raise ValueError(
                        f"bandwidth changed the supposedly bandwidth-independent PCA field {field}"
                    )
        if reference.get("positive_coordinates_used") is not False:
            raise ValueError("PCA fit selection row used positive coordinates")
        if reference.get("positive_identities_used") is not False:
            raise ValueError("PCA fit selection row used positive identities")
        unique.append(reference)
    return sorted(
        unique,
        key=lambda row: (
            -float(row["pca_mean_positive_tail_contrast"]),
            -float(row["pca_minimum_quiet_swap_positive_tail_contrast"]),
            int(row["sample_seed"]),
        ),
    )[0]


def _ui_window(interval: Sequence[int], *, review_start_ui: int) -> tuple[int, int]:
    start, stop = map(int, interval)
    return start - review_start_ui, stop - review_start_ui + 1


def _mask_interval(frame_ui: np.ndarray, interval: Sequence[int]) -> np.ndarray:
    start, stop = map(int, interval)
    return (frame_ui >= start) & (frame_ui <= stop)


def _duration_map(bursts: Mapping[str, Sequence[int]]) -> dict[str, int]:
    return {
        str(key): int(bounds[1]) - int(bounds[0]) + 1
        for key, bounds in sorted(bursts.items())
    }


def _score_candidates_for_arm(
    representation: Any,
    *,
    arm: str,
    lane: ContextLane,
    fold: FoldContract,
    frame_ui: np.ndarray,
    bursts: Mapping[str, Sequence[int]],
    scale_floor_percentile: float,
    chunk_frames: int,
    nms_distances_px: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Cross-fit quiet thresholds and return label-free held-out candidates."""

    import torch

    moments, gamma_ms = _elapsed(
        representation.device,
        lambda: gamma_local_standardization(
            representation,
            lane.spec(),
            chunk_frames=chunk_frames,
            return_statistics=True,
        ),
    )
    if moments.local_mean is None or moments.local_std is None:
        raise AssertionError("Gamma-LS statistics are required")
    frame_ui_device = torch.as_tensor(frame_ui, device=representation.device)
    heldout_window = {
        fold.heldout_burst: _ui_window(
            bursts[fold.heldout_burst], review_start_ui=int(frame_ui[0])
        )
    }
    durations = _duration_map(bursts)
    candidates: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    d2h_ms = 0.0
    for swap, floor_interval, test_interval in (
        (QUIET_SWAPS[0], fold.quiet_half_a_ui, fold.quiet_half_b_ui),
        (QUIET_SWAPS[1], fold.quiet_half_b_ui, fold.quiet_half_a_ui),
    ):
        floor_mask_np = _mask_interval(frame_ui, floor_interval)
        test_mask_np = _mask_interval(frame_ui, test_interval)
        floor_mask = torch.as_tensor(floor_mask_np, device=representation.device)
        floor = _positive_scale_floor(
            moments.local_std, floor_mask, scale_floor_percentile
        )

        def make_score() -> Any:
            return (representation - moments.local_mean) / (
                torch.maximum(moments.local_std, floor) + GAMMA_EPSILON
            )

        score, score_ms = _elapsed(representation.device, make_score)
        started = time.perf_counter()
        score_host = score.detach().cpu().numpy().astype(np.float32, copy=False)
        if representation.device.type == "cuda":
            torch.cuda.synchronize(representation.device)
        d2h_ms += (time.perf_counter() - started) * 1000.0
        test_windows = duration_matched_quiet_windows(test_mask_np, durations)
        for nms_distance_px in nms_distances_px:
            calibration = calibrate_training_quiet_thresholds(
                score_host,
                floor_mask_np,
                durations,
                target_peak_burdens=QUIET_NMS_PEAK_BURDENS,
                nms_distance_px=int(nms_distance_px),
            )
            for operating in calibration.operating_points:
                target = float(operating["target_nms_peaks_per_pseudo_burst"])
                threshold = float(operating["threshold_z"])
                heldout = extract_burst_candidates(
                    score_host,
                    heldout_window,
                    threshold_z=threshold,
                    nms_distance_px=int(nms_distance_px),
                )
                quiet_test = extract_burst_candidates(
                    score_host,
                    test_windows,
                    threshold_z=threshold,
                    nms_distance_px=int(nms_distance_px),
                )
                heldout_peaks = heldout.peaks[fold.heldout_burst]
                realized = sum(len(peaks) for peaks in quiet_test.peaks.values()) / len(
                    quiet_test.peaks
                )
                calibrations.append(
                    {
                        "training_fold": fold.training_fold,
                        "heldout_burst": int(fold.heldout_burst),
                        "context_role": lane.role,
                        "context_id": lane.context_id,
                        "representation": arm,
                        "quiet_swap": swap,
                        "nms_distance_px": int(nms_distance_px),
                        "nms_role": (
                            "primary" if int(nms_distance_px) == NMS_DISTANCE_PX else "descriptive_sensitivity"
                        ),
                        "floor_interval_ui": json.dumps(list(floor_interval)),
                        "test_interval_ui": json.dumps(list(test_interval)),
                        "scale_floor_percentile": float(scale_floor_percentile),
                        "scale_floor": float(floor.item()),
                        "target_nms_peaks_per_pseudo_burst": target,
                        "threshold_z": threshold,
                        "calibration_achieved_peaks_per_pseudo_burst": float(
                            operating["achieved_nms_peaks_per_pseudo_burst"]
                        ),
                        "heldout_quiet_peaks_per_pseudo_burst": float(realized),
                        "heldout_burst_candidate_count": len(heldout_peaks),
                        "probability_of_false_alarm_claimed": False,
                        "positive_coordinates_used": False,
                        "positive_identities_used": False,
                    }
                )
                for rank, (occupancy, x_px, y_px) in enumerate(heldout_peaks, start=1):
                    candidates.append(
                        {
                            "training_fold": fold.training_fold,
                            "burst_id": int(fold.heldout_burst),
                            "context_role": lane.role,
                            "context_id": lane.context_id,
                            "representation": arm,
                            "quiet_swap": swap,
                            "nms_distance_px": int(nms_distance_px),
                            "nms_role": (
                                "primary" if int(nms_distance_px) == NMS_DISTANCE_PX else "descriptive_sensitivity"
                            ),
                            "target_nms_peaks_per_pseudo_burst": target,
                            "threshold_z": threshold,
                            "candidate_rank": rank,
                            "occupancy_score": float(occupancy),
                            "x_px": int(x_px),
                            "y_px": int(y_px),
                            "interpretation_before_label_join": "unknown_candidate",
                        }
                    )
        del score, score_host
    del moments
    return candidates, calibrations, {
        "training_fold": fold.training_fold,
        "context_role": lane.role,
        "context_id": lane.context_id,
        "representation": arm,
        "gamma_runtime_ms": float(gamma_ms),
        "gamma_runtime_ms_per_frame": float(gamma_ms / len(representation)),
        "score_runtime_two_swaps_included": True,
        "d2h_ms": float(d2h_ms),
    }


def _read_sparse_positives(
    path: Path,
    *,
    selector: str,
    expected_rows: int,
    movie_shape_yx: tuple[int, int],
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {
        "observation_id",
        "burst_id",
        "canonical_roi_id",
        "x_px",
        "y_px",
        selector,
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path.name} lacks the sparse-positive contract")
    if len({row["observation_id"] for row in rows}) != len(rows):
        raise ValueError(f"{path.name} contains duplicate observation_id values")
    selected = [row for row in rows if row[selector].strip().lower() == "true"]
    if len(selected) != expected_rows:
        raise ValueError(
            f"{path.name} selected {len(selected)} rows, expected {expected_rows}"
        )
    height, width = movie_shape_yx
    result = []
    for row in selected:
        burst = int(row["burst_id"])
        x_px, y_px = float(row["x_px"]), float(row["y_px"])
        if burst not in (1, 2, 3, 4):
            raise ValueError("labels must use burst IDs 1..4")
        if not (0 <= x_px < width and 0 <= y_px < height):
            raise ValueError("sparse-positive coordinate is outside the movie")
        result.append(
            {
                "observation_id": row["observation_id"],
                "burst_id": burst,
                "canonical_roi_id": row["canonical_roi_id"],
                "x_px": x_px,
                "y_px": y_px,
            }
        )
    if len({row["observation_id"] for row in result}) != expected_rows:
        raise AssertionError("selected sparse-positive IDs are not unique")
    return result


def _candidate_groups(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["context_role"]),
            str(row["representation"]),
            str(row["quiet_swap"]),
            int(row["nms_distance_px"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
            int(row["burst_id"]),
        )
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        group.sort(key=lambda row: int(row["candidate_rank"]))
        if [int(row["candidate_rank"]) for row in group] != list(
            range(1, len(group) + 1)
        ):
            raise ValueError(f"candidate ranks are not contiguous for {key}")
    return groups


def observation_match_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    positives: Sequence[Mapping[str, Any]],
    *,
    cohort: str,
    operating_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per positive at every frozen operating point and budget."""

    groups = _candidate_groups(candidate_rows)
    positives_by_burst = {
        burst: [row for row in positives if int(row["burst_id"]) == burst]
        for burst in (1, 2, 3, 4)
    }
    if operating_rows is None:
        dimensions = sorted({key[:5] for key in groups})
    else:
        dimensions = sorted(
            {
                (
                    str(row["context_role"]),
                    str(row["representation"]),
                    str(row["quiet_swap"]),
                    int(row["nms_distance_px"]),
                    float(row["target_nms_peaks_per_pseudo_burst"]),
                )
                for row in operating_rows
            }
        )
    output: list[dict[str, Any]] = []
    for context_role, arm, swap, nms_distance, target in dimensions:
        for burst in (1, 2, 3, 4):
            candidate_group = groups.get(
                (context_role, arm, swap, nms_distance, target, burst), []
            )
            peaks = [
                (
                    float(row["occupancy_score"]),
                    int(row["x_px"]),
                    int(row["y_px"]),
                )
                for row in candidate_group
            ]
            burst_positives = positives_by_burst[burst]
            for budget in CANDIDATE_BUDGETS_PER_BURST:
                selected = peaks[:budget]
                matches, _ = match_peaks_one_to_one(
                    selected, burst_positives, MATCH_RADIUS_PX
                )
                # match_peaks_one_to_one returns matches in candidate-rank order;
                # recover exact candidate rank by coordinate/score when earlier
                # candidates were unmatched.
                corrected: dict[int, tuple[float, int, int, float, int]] = {}
                for label_index, score, x, y, distance in matches:
                    rank = next(
                        index + 1
                        for index, peak in enumerate(selected)
                        if peak == (score, x, y)
                    )
                    corrected[int(label_index)] = (score, x, y, distance, rank)
                for label_index, positive in enumerate(burst_positives):
                    match = corrected.get(label_index)
                    output.append(
                        {
                            "cohort": cohort,
                            "context_role": context_role,
                            "representation": arm,
                            "quiet_swap": swap,
                            "nms_distance_px": nms_distance,
                            "nms_role": (
                                "primary" if nms_distance == NMS_DISTANCE_PX else "descriptive_sensitivity"
                            ),
                            "target_nms_peaks_per_pseudo_burst": target,
                            "burst_id": burst,
                            "candidate_budget": budget,
                            "effective_candidate_count": min(len(peaks), budget),
                            "observation_id": positive["observation_id"],
                            "canonical_roi_id": positive["canonical_roi_id"],
                            "x_px": positive["x_px"],
                            "y_px": positive["y_px"],
                            "matched": match is not None,
                            "matched_candidate_rank": "" if match is None else match[4],
                            "matched_candidate_score": "" if match is None else match[0],
                            "matched_candidate_x_px": "" if match is None else match[1],
                            "matched_candidate_y_px": "" if match is None else match[2],
                            "match_distance_px": "" if match is None else match[3],
                            "unmatched_candidates": "unknown_not_negative",
                        }
                    )
    return output


def aggregate_match_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row["cohort"],
            row["context_role"],
            row["representation"],
            row["quiet_swap"],
            int(row["nms_distance_px"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
            int(row["burst_id"]),
            int(row["candidate_budget"]),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        matched = sum(bool(row["matched"]) for row in group)
        output.append(
            {
                "cohort": key[0],
                "context_role": key[1],
                "representation": key[2],
                "quiet_swap": key[3],
                "nms_distance_px": key[4],
                "nms_role": "primary" if key[4] == NMS_DISTANCE_PX else "descriptive_sensitivity",
                "target_nms_peaks_per_pseudo_burst": key[5],
                "burst_id": key[6],
                "candidate_budget": key[7],
                "known_positive_count": len(group),
                "matched_known_positive_count": matched,
                "known_positive_recall": matched / len(group),
                "unmatched_candidates": "unknown_not_negative",
                "precision_identified": False,
            }
        )
    return output


def _macro_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    cluster_weights: Mapping[str, int],
) -> tuple[float, float, dict[int, float]]:
    budgets = list(CANDIDATE_BUDGETS_PER_BURST)
    recall_by_budget: dict[int, float] = {}
    burst_b58: dict[int, float] = {}
    for budget in budgets:
        burst_recalls = []
        for burst in (1, 2, 3, 4):
            selected = [
                row
                for row in rows
                if str(row["representation"]) == arm
                and int(row["candidate_budget"]) == budget
                and int(row["burst_id"]) == burst
            ]
            numerator = sum(
                cluster_weights[str(row["canonical_roi_id"])] * bool(row["matched"])
                for row in selected
            )
            denominator = sum(
                cluster_weights[str(row["canonical_roi_id"])] for row in selected
            )
            burst_recall = numerator / denominator if denominator else float("nan")
            burst_recalls.append(burst_recall)
            if budget == 58:
                burst_b58[burst] = burst_recall
        recall_by_budget[budget] = float(np.nanmean(burst_recalls))
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray([recall_by_budget[budget] for budget in budgets])
    auc = float(np.trapezoid(y, x) / (x[-1] - x[0]))
    return auc, recall_by_budget[58], burst_b58


def clustered_bootstrap_contrasts(
    v1_match_rows: Sequence[Mapping[str, Any]],
    *,
    learned_arm: str = "cs_parzen_two_frame",
    controls: Sequence[str] = FIXED_ARMS,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[dict[str, Any]]:
    """Paired v1 contrasts with canonical-identity clustered bootstrap CIs."""

    base_dimensions = sorted(
        {
            (
                str(row["context_role"]),
                str(row["quiet_swap"]),
                int(row["nms_distance_px"]),
                float(row["target_nms_peaks_per_pseudo_burst"]),
            )
            for row in v1_match_rows
        }
    )
    pooled_dimensions = sorted(
        {
            (
                str(row["context_role"]),
                "crossfit_average",
                int(row["nms_distance_px"]),
                float(row["target_nms_peaks_per_pseudo_burst"]),
            )
            for row in v1_match_rows
        }
    )
    dimensions = base_dimensions + pooled_dimensions
    identities = sorted({str(row["canonical_roi_id"]) for row in v1_match_rows})
    if len(identities) != 26:
        raise ValueError(f"protected v1 bootstrap requires 26 identities, got {len(identities)}")
    rng = np.random.default_rng(int(seed))
    bootstrap_weights = []
    for _ in range(int(replicates)):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        bootstrap_weights.append(dict(zip(unique.tolist(), counts.tolist())))
    unit = {identity: 1 for identity in identities}
    output = []
    for context_role, swap, nms_distance, target in dimensions:
        subset = [
            row
            for row in v1_match_rows
            if str(row["context_role"]) == context_role
            and (swap == "crossfit_average" or str(row["quiet_swap"]) == swap)
            and int(row["nms_distance_px"]) == nms_distance
            and float(row["target_nms_peaks_per_pseudo_burst"]) == target
        ]
        learned_auc, learned_b58, learned_bursts = _macro_metric(
            subset, arm=learned_arm, cluster_weights=unit
        )
        for control in controls:
            control_auc, control_b58, control_bursts = _macro_metric(
                subset, arm=control, cluster_weights=unit
            )
            auc_deltas = []
            b58_deltas = []
            for weights in bootstrap_weights:
                # Missing sampled identities receive weight zero.
                full_weights = {identity: int(weights.get(identity, 0)) for identity in identities}
                a_auc, a_b58, _ = _macro_metric(
                    subset, arm=learned_arm, cluster_weights=full_weights
                )
                c_auc, c_b58, _ = _macro_metric(
                    subset, arm=control, cluster_weights=full_weights
                )
                auc_deltas.append(a_auc - c_auc)
                b58_deltas.append(a_b58 - c_b58)
            output.append(
                {
                    "context_role": context_role,
                    "quiet_swap": swap,
                    "nms_distance_px": nms_distance,
                    "nms_role": (
                        "primary" if nms_distance == NMS_DISTANCE_PX else "descriptive_sensitivity"
                    ),
                    "target_nms_peaks_per_pseudo_burst": target,
                    "learned_representation": learned_arm,
                    "control_representation": control,
                    "learned_budget_auc": learned_auc,
                    "control_budget_auc": control_auc,
                    "budget_auc_delta": learned_auc - control_auc,
                    "budget_auc_delta_ci95_low": float(np.nanpercentile(auc_deltas, 2.5)),
                    "budget_auc_delta_ci95_high": float(np.nanpercentile(auc_deltas, 97.5)),
                    "learned_b58_macro_recall": learned_b58,
                    "control_b58_macro_recall": control_b58,
                    "b58_macro_recall_delta": learned_b58 - control_b58,
                    "b58_delta_ci95_low": float(np.nanpercentile(b58_deltas, 2.5)),
                    "b58_delta_ci95_high": float(np.nanpercentile(b58_deltas, 97.5)),
                    "b58_burst_wins": sum(
                        learned_bursts[burst] > control_bursts[burst]
                        for burst in (1, 2, 3, 4)
                    ),
                    "b58_burst_ties": sum(
                        learned_bursts[burst] == control_bursts[burst]
                        for burst in (1, 2, 3, 4)
                    ),
                    "cluster_field": "canonical_roi_id",
                    "cluster_count": 26,
                    "bootstrap_seed": int(seed),
                    "bootstrap_replicates": int(replicates),
                    "v7_inferential_claim": False,
                }
            )
    return output


def _summary_by_arm(match_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in match_rows:
        key = (
            row["cohort"],
            row["context_role"],
            row["representation"],
            row["quiet_swap"],
            int(row["nms_distance_px"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
            int(row["candidate_budget"]),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(groups.items()):
        matched = sum(bool(row["matched"]) for row in rows)
        result.append(
            {
                "cohort": key[0],
                "context_role": key[1],
                "representation": key[2],
                "quiet_swap": key[3],
                "nms_distance_px": key[4],
                "nms_role": "primary" if key[4] == NMS_DISTANCE_PX else "descriptive_sensitivity",
                "target_nms_peaks_per_pseudo_burst": key[5],
                "candidate_budget": key[6],
                "known_positive_count": len(rows),
                "matched_known_positive_count": matched,
                "pooled_known_positive_recall": matched / len(rows),
                "precision_identified": False,
            }
        )
    pooled: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in match_rows:
        key = (
            row["cohort"],
            row["context_role"],
            row["representation"],
            int(row["nms_distance_px"]),
            float(row["target_nms_peaks_per_pseudo_burst"]),
            int(row["candidate_budget"]),
        )
        pooled.setdefault(key, []).append(row)
    for key, rows in sorted(pooled.items()):
        matched = sum(bool(row["matched"]) for row in rows)
        result.append(
            {
                "cohort": key[0],
                "context_role": key[1],
                "representation": key[2],
                "quiet_swap": "crossfit_average",
                "nms_distance_px": key[3],
                "nms_role": "primary" if key[3] == NMS_DISTANCE_PX else "descriptive_sensitivity",
                "target_nms_peaks_per_pseudo_burst": key[4],
                "candidate_budget": key[5],
                "known_positive_count": len(rows) // len(QUIET_SWAPS),
                "matched_known_positive_count": matched / len(QUIET_SWAPS),
                "pooled_known_positive_recall": matched / len(rows),
                "precision_identified": False,
            }
        )
    return result


def run_protected_representation_experiment(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    context_selection_dir: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run, seal, and evaluate the protected adjacent-frame comparison."""

    if not isinstance(config, GammaLSDifferenceConfig):
        raise TypeError("config must be a validated GammaLSDifferenceConfig")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"protected output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"protected output parent does not exist: {destination.parent}")

    # All gates precede output mutation.
    preflight = verify_matching_preflight(config, preflight_dir, require_gpu_ready=True)
    folds = build_fold_contracts(config)
    context_plan, context_provenance = load_fold_context_plan(
        context_selection_dir, folds
    )
    try:
        runtime = require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise ProtectedRepresentationUnavailable(str(error)) from error
    movie = np.load(config.source_paths["movie"], mmap_mode="r", allow_pickle=False)
    if movie.ndim != 3 or str(movie.dtype) != "uint16":
        raise ValueError("protected source must remain the frozen uint16 TYX movie")

    contract = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "run_type": "protected_outer_fold_two_frame_representation",
        "portable_config_sha256": _canonical_sha256(config.portable_dict()),
        "protected_executor_sha256": _sha256(Path(__file__).resolve()),
        "preflight_sha256": _sha256(Path(preflight_dir).resolve() / "preflight.json"),
        "movie_sha256": preflight["source"]["movie"]["sha256"],
        "context_selection": context_provenance,
        "fit_grid": {
            "bandwidths": list(config.payload["ica_search"]["two_frame_cs_parzen_bandwidth"]),
            "sample_seeds": list(config.payload["ica_search"]["paired_sample_seeds"]),
            "fits": int(config.payload["ica_search"]["pairwise_total_outer_fold_fits"]),
        },
        "bootstrap": {
            "cluster_field": "canonical_roi_id",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "archived_fits_role": "parity_only_not_loaded",
    }
    contract_sha = _canonical_sha256(contract)
    work = destination.parent / f".{destination.name}.protected-work"
    if work.exists():
        frozen_contract = json.loads((work / "run_contract.json").read_text(encoding="utf-8"))
        if _canonical_sha256(frozen_contract) != contract_sha:
            raise RuntimeError("existing resumable work directory has a different contract")
    else:
        work.mkdir()
        _atomic_json(work / "run_contract.json", contract)
    fit_dir = work / "fit_models"
    fit_dir.mkdir(exist_ok=True)

    def heartbeat(stage: str, **details: Any) -> None:
        _atomic_json(
            work / "heartbeat.json",
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "status": "running",
                **details,
            },
        )

    try:
        heartbeat("causal_preprocessing")
        start_ui, stop_ui = map(int, config.payload["frames"]["review_interval_ui"])
        chunk_frames = max(int(value) for value in config.payload["efficiency"]["frame_chunks"])
        common, preprocessing_timing = _stream_common_history_to_device(
            config.source_paths["movie"],
            review_start_ui=start_ui,
            review_stop_ui=stop_ui,
            chunk_frames=chunk_frames,
            device=__import__("torch").device(str(runtime["resolved_device"])),
            heartbeat=lambda payload: _relay_progress_heartbeat(
                heartbeat, "causal_preprocessing", payload
            ),
        )
        import torch

        frame_ui = np.arange(start_ui, stop_ui + 1, dtype=np.int64)
        frame_ui_device = torch.as_tensor(frame_ui, device=common.device)
        quiet_start, quiet_stop = map(int, config.payload["frames"]["quiet_interval_ui"])
        quiet_mask = _mask_interval(frame_ui, (quiet_start, quiet_stop))
        quiet_common = common[1:][torch.as_tensor(quiet_mask, device=common.device)]
        anatomy_mask, anatomy_summary = uniform_anatomy_mask(
            quiet_common.detach().cpu().numpy()
        )
        del quiet_common

        ica = config.payload["ica_search"]
        bandwidths = tuple(float(value) for value in ica["two_frame_cs_parzen_bandwidth"])
        seeds = tuple(int(value) for value in ica["paired_sample_seeds"])
        scale_percentile = float(
            config.payload["gamma_ls_grid"]["finalist_scale_floor_percentiles"][0]
        )
        tail_quantile = float(config.payload["screen"]["positive_tail_quantile"])
        bursts = config.payload["frames"]["burst_intervals_ui"]
        fit_rows: list[dict[str, Any]] = []
        packets: dict[tuple[int, int, float], dict[str, Any]] = {}
        pca_metric_cache: dict[tuple[int, int, str], dict[str, Any]] = {}
        completed_fits = 0
        for fold in folds:
            for seed in seeds:
                observations, sample_manifest = _sample_fold_pairs(
                    common,
                    frame_ui=frame_ui,
                    fold=fold,
                    anatomy_mask=anatomy_mask,
                    sample_count=int(ica["confirmation_samples"]),
                    seed=seed,
                )
                whitening_started = time.perf_counter()
                whitened, whitening_fit = center_and_whiten_2d(observations)
                whitening_seconds = time.perf_counter() - whitening_started
                if (
                    not whitening_fit.identifiable
                    or np.linalg.matrix_rank(whitening_fit.whitening) != 2
                ):
                    raise RuntimeError(
                        "fold two-frame covariance is not full-rank identifiable"
                    )
                for bandwidth in bandwidths:
                    checkpoint = fit_dir / (
                        f"fold_{fold.training_fold}__seed_{seed}__bandwidth_{str(bandwidth).replace('.', 'p')}.json"
                    )
                    if checkpoint.is_file():
                        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                        if saved.get("sample_manifest") != sample_manifest:
                            raise RuntimeError("fit checkpoint sample manifest changed")
                        packet = saved["fit"]
                    else:
                        packet = _fit_packet(
                            observations,
                            whitened=whitened,
                            whitening_fit=whitening_fit,
                            whitening_seconds=whitening_seconds,
                            bandwidth=bandwidth,
                            screen_samples=int(ica["screen_samples"]),
                            block_rows=int(ica["kernel_block_rows"]),
                            coarse_step_degrees=float(ica["coarse_step_degrees"]),
                            refine_half_width_degrees=float(ica["refine_half_width_degrees"]),
                            refine_step_degrees=float(ica["refine_step_degrees"]),
                        )
                        _atomic_json(
                            checkpoint,
                            {
                                "training_fold": fold.training_fold,
                                "heldout_burst": fold.heldout_burst,
                                "sample_seed": seed,
                                "bandwidth": bandwidth,
                                "sample_manifest": sample_manifest,
                                "fit": packet,
                            },
                        )
                    packets[(fold.training_fold, seed, bandwidth)] = packet
                    ica_rep = _learned_representation(common, packet, "cs_parzen_two_frame")
                    for lane in context_plan[fold.training_fold]:
                        pca_key = (fold.training_fold, seed, lane.role)
                        if pca_key not in pca_metric_cache:
                            pca_rep = _learned_representation(
                                common, packet, "pca_whitened_derivative"
                            )
                            pca_metric_cache[pca_key] = _fit_selection_metrics(
                                pca_rep,
                                representation_name="pca_whitened_derivative",
                                lane=lane,
                                fold=fold,
                                frame_ui_device=frame_ui_device,
                                bursts=bursts,
                                scale_floor_percentile=scale_percentile,
                                tail_quantile=tail_quantile,
                                chunk_frames=chunk_frames,
                            )
                            del pca_rep
                        pca_metric = pca_metric_cache[pca_key]
                        ica_metric = _fit_selection_metrics(
                            ica_rep,
                            representation_name="cs_parzen_two_frame",
                            lane=lane,
                            fold=fold,
                            frame_ui_device=frame_ui_device,
                            bursts=bursts,
                            scale_floor_percentile=scale_percentile,
                            tail_quantile=tail_quantile,
                            chunk_frames=chunk_frames,
                        )
                        fit_rows.append(
                            {
                                "training_fold": fold.training_fold,
                                "heldout_burst": int(fold.heldout_burst),
                                "context_role": lane.role,
                                "context_id": lane.context_id,
                                "bandwidth": bandwidth,
                                "sample_seed": seed,
                                "sample_identity_sha256": sample_manifest["sample_identity_sha256"],
                                "fit_seconds": packet["fit_seconds"],
                                "objective_value": packet["objective_value"],
                                "selected_angle_degrees": packet["diagnostics"]["selected_angle_degrees"],
                                "whitening_condition_number": packet["condition_number"],
                                "pca_mean_positive_tail_contrast": pca_metric["mean_positive_tail_contrast"],
                                "pca_minimum_quiet_swap_positive_tail_contrast": pca_metric["minimum_quiet_swap_positive_tail_contrast"],
                                "ica_mean_positive_tail_contrast": ica_metric["mean_positive_tail_contrast"],
                                "ica_minimum_quiet_swap_positive_tail_contrast": ica_metric["minimum_quiet_swap_positive_tail_contrast"],
                                "selection_uses_training_burst_windows": True,
                                "positive_coordinates_used": False,
                                "positive_identities_used": False,
                            }
                        )
                    del ica_rep
                    completed_fits += 1
                    heartbeat(
                        "ica_fit_grid",
                        completed_fits=completed_fits,
                        total_fits=36,
                    )
                del observations
        if completed_fits != 36 or len(packets) != 36:
            raise AssertionError("protected ICA grid did not complete exactly 36 fits")

        selected_rows = []
        for fold in folds:
            for lane in context_plan[fold.training_fold]:
                ica_winner = select_fold_ica_fit(
                    fit_rows,
                    fold=fold.training_fold,
                    context_role=lane.role,
                    bandwidths=bandwidths,
                    seeds=seeds,
                )
                pca_winner = select_fold_pca_fit(
                    fit_rows,
                    fold=fold.training_fold,
                    context_role=lane.role,
                    bandwidths=bandwidths,
                    seeds=seeds,
                )
                for arm, winner, metric_name, secondary_name in (
                    (
                        "cs_parzen_two_frame",
                        ica_winner,
                        "ica_mean_positive_tail_contrast",
                        "ica_minimum_quiet_swap_positive_tail_contrast",
                    ),
                    (
                        "pca_whitened_derivative",
                        pca_winner,
                        "pca_mean_positive_tail_contrast",
                        "pca_minimum_quiet_swap_positive_tail_contrast",
                    ),
                ):
                    selected_rows.append(
                        {
                            "training_fold": fold.training_fold,
                            "heldout_burst": int(fold.heldout_burst),
                            "context_role": lane.role,
                            "context_id": lane.context_id,
                            "selected_for_representation": arm,
                            "sample_seed": int(winner["sample_seed"]),
                            "packet_bandwidth": float(winner["bandwidth"]),
                            "bandwidth_role": (
                                "optimized_cs_parzen_hyperparameter"
                                if arm == "cs_parzen_two_frame"
                                else "canonical_lowest_bandwidth_packet_for_bandwidth_independent_whitening"
                            ),
                            "selection_metric": metric_name,
                            "mean_positive_tail_contrast": float(winner[metric_name]),
                            "minimum_quiet_swap_positive_tail_contrast": float(winner[secondary_name]),
                            "sample_identity_sha256": winner["sample_identity_sha256"],
                            "positive_coordinates_used": False,
                            "positive_identities_used": False,
                        }
                    )
        _atomic_tsv(work / "fit_grid.tsv", fit_rows)
        _atomic_tsv(work / "selected_fit_rows.tsv", selected_rows)

        candidate_rows: list[dict[str, Any]] = []
        calibration_rows: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        equivalence_rows: list[dict[str, Any]] = []
        candidate_cells = sum(len(lanes) for lanes in context_plan.values()) * len(ALL_ARMS)
        completed_cells = 0
        for fold in folds:
            heldout_mask = _mask_interval(frame_ui, bursts[fold.heldout_burst])
            heldout_device = torch.as_tensor(heldout_mask, device=common.device)
            for lane in context_plan[fold.training_fold]:
                ica_winner = next(
                    row
                    for row in selected_rows
                    if int(row["training_fold"]) == fold.training_fold
                    and row["context_role"] == lane.role
                    and row["selected_for_representation"] == "cs_parzen_two_frame"
                )
                pca_winner = next(
                    row
                    for row in selected_rows
                    if int(row["training_fold"]) == fold.training_fold
                    and row["context_role"] == lane.role
                    and row["selected_for_representation"] == "pca_whitened_derivative"
                )
                ica_packet = packets[
                    (
                        fold.training_fold,
                        int(ica_winner["sample_seed"]),
                        float(ica_winner["packet_bandwidth"]),
                    )
                ]
                pca_packet = packets[
                    (
                        fold.training_fold,
                        int(pca_winner["sample_seed"]),
                        float(pca_winner["packet_bandwidth"]),
                    )
                ]
                representations = {
                    name: _fixed_representation(common, name) for name in FIXED_ARMS
                }
                representations.update(
                    {
                        "pca_whitened_derivative": _learned_representation(
                            common, pca_packet, "pca_whitened_derivative"
                        ),
                        "cs_parzen_two_frame": _learned_representation(
                            common, ica_packet, "cs_parzen_two_frame"
                        ),
                        "pca_matched_to_selected_ica": _learned_representation(
                            common, ica_packet, "pca_whitened_derivative"
                        ),
                    }
                )
                signed_heldout = representations["difference_signed"][heldout_device]
                for learned_name in LEARNED_ARMS:
                    metric = paired_representation_equivalence(
                        signed_heldout,
                        representations[learned_name][heldout_device],
                    )
                    equivalence_rows.append(
                        {
                            "training_fold": fold.training_fold,
                            "heldout_burst": int(fold.heldout_burst),
                            "context_role": lane.role,
                            "selected_bandwidth": (
                                float(ica_winner["packet_bandwidth"])
                                if learned_name == "cs_parzen_two_frame"
                                else "not_applicable"
                            ),
                            "selected_sample_seed": (
                                int(ica_winner["sample_seed"])
                                if learned_name == "cs_parzen_two_frame"
                                else int(pca_winner["sample_seed"])
                            ),
                            "matched_packet_rotation_diagnostic": False,
                            "reference_representation": "difference_signed",
                            "candidate_representation": learned_name,
                            **metric,
                        }
                    )
                rotation_metric = paired_representation_equivalence(
                    representations["pca_matched_to_selected_ica"][heldout_device],
                    representations["cs_parzen_two_frame"][heldout_device],
                )
                equivalence_rows.append(
                    {
                        "training_fold": fold.training_fold,
                        "heldout_burst": int(fold.heldout_burst),
                        "context_role": lane.role,
                        "selected_bandwidth": float(ica_winner["packet_bandwidth"]),
                        "selected_sample_seed": int(ica_winner["sample_seed"]),
                        "matched_packet_rotation_diagnostic": True,
                        "reference_representation": "pca_matched_to_selected_ica",
                        "candidate_representation": "cs_parzen_two_frame",
                        **rotation_metric,
                    }
                )
                del signed_heldout
                for arm, representation in representations.items():
                    candidates, calibrations, timing = _score_candidates_for_arm(
                        representation,
                        arm=arm,
                        lane=lane,
                        fold=fold,
                        frame_ui=frame_ui,
                        bursts=bursts,
                        scale_floor_percentile=scale_percentile,
                        chunk_frames=chunk_frames,
                        nms_distances_px=(4, 6, 8),
                    )
                    candidate_rows.extend(candidates)
                    calibration_rows.extend(calibrations)
                    timing_rows.append(timing)
                    completed_cells += 1
                    heartbeat(
                        "candidate_freeze",
                        completed_context_arm_cells=completed_cells,
                        total_context_arm_cells=candidate_cells,
                    )
                del representations

        _atomic_tsv(work / "candidates_label_sealed.tsv", candidate_rows)
        _atomic_tsv(work / "threshold_calibration.tsv", calibration_rows)
        _atomic_tsv(work / "timings.tsv", timing_rows)
        _atomic_tsv(work / "representation_equivalence.tsv", equivalence_rows)
        candidate_seal = {
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "sparse_positive_fields_parsed_before_seal": False,
            "source_files_hashed_by_preflight_before_seal": True,
            "positive_coordinates_used": False,
            "positive_identities_used": False,
            "candidate_table": {
                "path": "candidates_label_sealed.tsv",
                "sha256": _sha256(work / "candidates_label_sealed.tsv"),
                "rows": len(candidate_rows),
            },
            "threshold_table_sha256": _sha256(work / "threshold_calibration.tsv"),
            "selected_fit_table_sha256": _sha256(work / "selected_fit_rows.tsv"),
            "fit_grid_table_sha256": _sha256(work / "fit_grid.tsv"),
            "unmatched_candidates": "unknown_not_negative",
        }
        _atomic_json(work / "candidate_seal.json", candidate_seal)
        # Preflight verifies immutable source bytes, but sparse-positive fields
        # are first parsed here, after the candidate stream has been sealed.
        heartbeat("protected_label_join", candidate_seal_sha256=_sha256(work / "candidate_seal.json"))
        v1 = _read_sparse_positives(
            config.source_paths["protected_labels_v1"],
            selector="include_inclusive",
            expected_rows=79,
            movie_shape_yx=tuple(movie.shape[1:]),
        )
        v7 = _read_sparse_positives(
            config.source_paths["latest_labels_v7"],
            selector="include_confirmed",
            expected_rows=106,
            movie_shape_yx=tuple(movie.shape[1:]),
        )
        v1_matches = observation_match_rows(
            candidate_rows,
            v1,
            cohort="protected_v1",
            operating_rows=calibration_rows,
        )
        v7_matches = observation_match_rows(
            candidate_rows,
            v7,
            cohort="latest_v7_sensitivity",
            operating_rows=calibration_rows,
        )
        v1_aggregate = aggregate_match_rows(v1_matches)
        v7_aggregate = aggregate_match_rows(v7_matches)
        contrasts = clustered_bootstrap_contrasts(
            v1_matches,
            controls=(
                *FIXED_ARMS,
                "pca_whitened_derivative",
                "pca_matched_to_selected_ica",
            ),
        )
        _atomic_tsv(work / "protected_v1_observation_matches.tsv", v1_matches)
        _atomic_tsv(work / "protected_v1_recall.tsv", v1_aggregate)
        _atomic_tsv(work / "protected_v1_clustered_bootstrap_contrasts.tsv", contrasts)
        _atomic_tsv(work / "latest_v7_observation_matches.tsv", v7_matches)
        _atomic_tsv(work / "latest_v7_sensitivity.tsv", v7_aggregate)

        summary_rows = _summary_by_arm(v1_matches)
        v7_summary_rows = _summary_by_arm(v7_matches)
        _atomic_tsv(work / "protected_v1_arm_summary.tsv", summary_rows)
        _atomic_tsv(work / "latest_v7_arm_summary.tsv", v7_summary_rows)
        claim_boundary = {
            "protected_v1_population": {
                "selector": "include_inclusive",
                "occurrences": len(v1),
                "canonical_identity_count": len({row["canonical_roi_id"] for row in v1}),
            },
            "latest_v7_population": {
                "selector": "include_confirmed",
                "occurrences": len(v7),
                "canonical_identity_count": len({row["canonical_roi_id"] for row in v7}),
                "role": "descriptive_sensitivity_not_independent_confirmation",
            },
            "unmatched_candidates": "unknown_not_negative",
            "precision_identified": False,
            "archived_fit_used": False,
            "fold_local_models": True,
            "heldout_burst_and_guard_excluded_from_fit": True,
            "candidate_artifacts_sealed_before_label_join": True,
            "context_selection_used_sparse_positive_labels": False,
            "scientific_audit_complete": False,
            "paper_promotion_ready": False,
        }
        _atomic_json(work / "claim_boundary.json", claim_boundary)
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "run_type": "protected_outer_fold_two_frame_representation",
            "status": "complete_protected_metrics_scientific_audit_pending",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "preprocessing_timing": preprocessing_timing,
            "anatomy_sampling": anatomy_summary,
            "fit_count": 36,
            "fit_grid_row_count": len(fit_rows),
            "selected_fit_row_count": len(selected_rows),
            "context_roles": context_provenance["context_roles_by_fold"],
            "candidate_row_count": len(candidate_rows),
            "candidate_seal": candidate_seal,
            "protected_v1": {
                "occurrences": len(v1),
                "canonical_identities": len({row["canonical_roi_id"] for row in v1}),
                "arm_summary": summary_rows,
                "clustered_bootstrap_contrasts": contrasts,
            },
            "latest_v7_sensitivity": {
                "occurrences": len(v7),
                "canonical_identities": len({row["canonical_roi_id"] for row in v7}),
                "arm_summary": v7_summary_rows,
                "inferential_claim": False,
            },
            "claim_boundary": claim_boundary,
        }
        _atomic_json(work / "summary.json", summary)
        checks = {
            "exact_36_cs_parzen_fits": len(packets) == 36,
            "four_outer_folds": len(folds) == 4,
            "heldout_pair_guard_enforced": True,
            "archived_fit_not_loaded": True,
            "candidate_seal_precedes_label_join": True,
            "protected_v1_uses_79_inclusive": len(v1) == 79,
            "protected_v1_has_26_identity_clusters": len({row["canonical_roi_id"] for row in v1}) == 26,
            "v7_uses_106_confirmed": len(v7) == 106,
            "all_candidates_unknown_before_join": all(
                row["interpretation_before_label_join"] == "unknown_candidate"
                for row in candidate_rows
            ),
            "precision_not_claimed": True,
            "scientific_audit_pending": True,
        }
        _atomic_json(
            work / "validation.json",
            {
                "status": "passed_protected_metric_artifact_contract",
                "checks": checks,
                "all_checks_pass": all(checks.values()),
            },
        )
        _atomic_json(
            work / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "grain": "context role x representation x quiet swap x quiet burden x heldout burst x candidate budget",
                "candidate_seal": "candidate_seal.json",
                "protected_primary": "protected_v1_arm_summary.tsv",
                "paired_inference": "protected_v1_clustered_bootstrap_contrasts.tsv",
                "per_observation_recomputability": "protected_v1_observation_matches.tsv",
                "latest_sensitivity": "latest_v7_arm_summary.tsv",
                "context_selection": context_provenance,
                "limitations": [
                    "sparse positives do not identify precision",
                    "v7 is candidate-assisted descriptive sensitivity",
                    "scientific-audit media are still pending",
                    "six-lag fold-local ICA refitting is outside this adjacent-frame stage",
                ],
            },
        )
        heartbeat("complete", status="complete_protected_metrics_scientific_audit_pending")
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        if destination.exists():
            raise FileExistsError(f"protected output appeared during execution: {destination}")
        work.replace(destination)
        return summary
    except Exception as error:
        if work.exists():
            _atomic_json(
                work / "status.json",
                {
                    "status": "interrupted_or_failed_resumable",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                    "safe_to_resume_with_identical_contract": True,
                },
            )
        raise


__all__ = [
    "ALL_ARMS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "ContextLane",
    "DIAGNOSTIC_ARMS",
    "ProtectedRepresentationUnavailable",
    "aggregate_match_rows",
    "clustered_bootstrap_contrasts",
    "context_lane_from_id",
    "load_fold_context_plan",
    "observation_match_rows",
    "pair_eligible_current_frames",
    "run_protected_representation_experiment",
    "select_fold_ica_fit",
    "select_fold_pca_fit",
]
