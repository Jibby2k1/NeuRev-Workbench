"""Coordinate-free CUDA screen for the guarded radial Gamma-LS grid.

The screen consumes human-declared burst *time windows*, so it is not called
fully label-free.  It has no API for, and never opens, sparse-positive
coordinates or identities.  Each outer fold excludes one burst (plus a
ten-frame guard), fits its local-standard-deviation floors on one quiet half,
scores the other quiet half, reverses those roles, and averages the two
contrasts.  Context selection remains local to that outer fold.

Only the three fixed, nonlearned representations participate in context
selection.  A common context is frozen per outer fold for later fair comparison
of every representation.  No across-fold deployment context is selected here.
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
import shutil
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
)

from .config import GammaLSDifferenceConfig
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from . import gpu_representations
from .grid import (
    SCREEN_REPRESENTATIONS,
    TRAINING_FOLDS,
    ContextAggregate,
    GammaContext,
    enumerate_g1_contexts,
    enumerate_g2_contexts,
    select_g1_radius_guard_pairs,
    select_g2_common_finalist,
)
from .preflight import _implementation_status


TAIL_QUANTILE_METHOD = "higher_order_statistic_k_equals_ceil_q_times_n"
QUIET_SWAPS = ("a_floor_b_tail", "b_floor_a_tail")
ENERGY_EPSILON = 1e-8
GAMMA_EPSILON = 1e-6


class CudaScreenUnavailable(RuntimeError):
    """Raised when a ready CUDA device cannot be used before output mutation."""


@dataclass(frozen=True)
class FoldContract:
    """One leakage-safe outer-fold split in one-based UI coordinates."""

    training_fold: int
    heldout_burst: str
    training_bursts: tuple[str, str, str]
    heldout_guard_ui: tuple[int, int]
    quiet_half_a_ui: tuple[int, int]
    quiet_half_b_ui: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "training_fold": self.training_fold,
            "heldout_burst": self.heldout_burst,
            "training_bursts": list(self.training_bursts),
            "heldout_guard_ui": list(self.heldout_guard_ui),
            "quiet_half_a_ui": list(self.quiet_half_a_ui),
            "quiet_half_b_ui": list(self.quiet_half_b_ui),
        }


@dataclass(frozen=True)
class ScreenExecution:
    """Small host-resident result returned after all device work completes."""

    g1_rows: tuple[Mapping[str, Any], ...]
    g1_swap_rows: tuple[Mapping[str, Any], ...]
    g2_rows: tuple[Mapping[str, Any], ...]
    g2_swap_rows: tuple[Mapping[str, Any], ...]
    fold_selections: tuple[Mapping[str, Any], ...]
    runtime: Mapping[str, Any]
    timings: Mapping[str, Any]
    peak_memory: Mapping[str, Any]


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


def _require_cuda(device: str) -> dict[str, Any]:
    try:
        return require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise CudaScreenUnavailable(str(error)) from error


def _verify_ready_preflight(
    config: GammaLSDifferenceConfig, artifact_dir: str | Path
) -> dict[str, Any]:
    """Verify readiness/config/code/movie without reopening either label table."""

    root = Path(artifact_dir).expanduser().resolve()
    preflight_path = root / "preflight.json"
    portable_path = root / "config.portable.json"
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    if portable != config.portable_dict():
        raise RuntimeError("preflight config does not match the requested config")
    if not payload.get("data_ready") or not payload.get("gpu_run_ready"):
        raise RuntimeError(f"preflight is not GPU-ready: {payload.get('status')}")
    current_implementation = _implementation_status(config)
    if (
        not current_implementation["complete"]
        or current_implementation["files"]
        != payload.get("implementation", {}).get("files")
    ):
        raise RuntimeError("implementation fingerprints changed after preflight")
    movie_path = config.source_paths["movie"]
    frozen_movie_hash = payload.get("source", {}).get("movie", {}).get("sha256")
    if not frozen_movie_hash or _sha256(movie_path) != frozen_movie_hash:
        raise RuntimeError("movie fingerprint changed after preflight")
    return {
        "artifact_dir": str(root),
        "preflight_sha256": _sha256(preflight_path),
        "portable_config_sha256": _sha256(portable_path),
        "status": payload["status"],
        "data_ready": True,
        "gpu_run_ready": True,
        "movie_sha256": frozen_movie_hash,
        "label_files_reopened": False,
    }


def _source_contract(config: GammaLSDifferenceConfig) -> dict[str, Any]:
    """Inspect the memory-mapped source without materializing its full history."""

    start_ui, stop_ui = map(int, config.payload["frames"]["review_interval_ui"])
    movie_path = config.source_paths["movie"]
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(movie, np.memmap) or movie.ndim != 3:
        raise ValueError("configured movie must be a memory-mappable TYX .npy")
    if not 1 < start_ui <= stop_ui <= int(movie.shape[0]):
        raise ValueError("configured review interval exceeds the movie")
    return {
        "source_id": config.payload["sources"]["movie"],
        "load_mode": "numpy_memmap_chunked_full_causal_history",
        "movie_shape_tyx": [int(value) for value in movie.shape],
        "movie_dtype": str(movie.dtype),
        "causal_history_processed_ui": [1, stop_ui],
        "review_interval_ui": [start_ui, stop_ui],
        "retained_common_state_interval_ui": [start_ui - 1, stop_ui],
        "aligned_representation_interval_ui": [start_ui, stop_ui],
        "ema_initialization": "first_acquisition_frame_then_exact_state_carry",
        "full_history_dense_host_materialization": False,
        "full_history_dense_device_materialization": False,
        "labels_opened": False,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
        "burst_windows_used": True,
    }


def _stream_common_history_to_device(
    movie_path: Path,
    *,
    review_start_ui: int,
    review_stop_ui: int,
    chunk_frames: int,
    device: Any,
    heartbeat: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[Any, dict[str, Any]]:
    """Causally preprocess frames 1..stop and retain only predecessor+review.

    Spatial filtering is frame-local.  The exact final EMA state of each chunk
    seeds the next chunk, so this equals one dense causal pass without storing
    the pre-review movie on host or device.
    """

    import torch

    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    stop_zero = int(review_stop_ui)
    retain_start_zero = int(review_start_ui) - 2  # UI review_start - 1
    if not 0 <= retain_start_zero < stop_zero <= int(movie.shape[0]):
        raise ValueError("invalid causal history/review bounds")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    retained = []
    state = None
    h2d_ms = 0.0
    preprocess_ms = 0.0
    transfer_count = 0
    expected_chunks = int(math.ceil(stop_zero / chunk_frames))
    for chunk_index, start in enumerate(range(0, stop_zero, chunk_frames), start=1):
        stop = min(start + chunk_frames, stop_zero)
        host_array = np.array(
            movie[start:stop], dtype=np.float32, order="C", copy=True
        )
        if not np.isfinite(host_array).all():
            raise ValueError("causal source chunk contains non-finite values")
        host_tensor = torch.from_numpy(host_array)
        device_chunk, elapsed_h2d = _elapsed(
            device,
            lambda source=host_tensor: source.to(
                device=device, dtype=torch.float32, non_blocking=False
            ),
        )
        h2d_ms += elapsed_h2d
        transfer_count += 1

        def preprocess_chunk() -> tuple[Any, list[Any]]:
            nonlocal state
            spatial = gpu_representations._spatial_gaussian_reflect(device_chunk)
            kept = []
            for offset in range(int(spatial.shape[0])):
                absolute = start + offset
                if state is None:
                    state = spatial[offset]
                else:
                    state = (
                        gpu_representations.EMA_ALPHA * spatial[offset]
                        + (1.0 - gpu_representations.EMA_ALPHA) * state
                    )
                if absolute >= retain_start_zero:
                    kept.append(state.clone())
            return state, kept

        (_, kept), elapsed_preprocess = _elapsed(device, preprocess_chunk)
        preprocess_ms += elapsed_preprocess
        retained.extend(kept)
        del device_chunk, host_tensor, host_array
        if heartbeat is not None:
            heartbeat(
                {
                    "stage": "causal_preprocessing",
                    "completed_history_chunks": chunk_index,
                    "total_history_chunks": expected_chunks,
                }
            )
    common = torch.stack(retained)
    expected_retained = review_stop_ui - review_start_ui + 2
    if int(common.shape[0]) != expected_retained:
        raise AssertionError("causal history retained an unexpected frame count")
    return common, {
        "h2d_ms": h2d_ms,
        "movie_h2d_transfer_count": transfer_count,
        "preprocessing_ms": preprocess_ms,
        "source_chunk_frames": chunk_frames,
        "history_chunk_count": expected_chunks,
        "causal_history_processed_ui": [1, review_stop_ui],
        "retained_common_state_interval_ui": [review_start_ui - 1, review_stop_ui],
        "ema_state_carried_across_chunks": True,
        "pre_review_outputs_discarded": True,
    }


def build_fold_contracts(config: Any) -> tuple[FoldContract, ...]:
    """Build four outer folds and the declared 50/50 quiet-window splits."""

    payload = getattr(config, "payload", config)
    frames = payload["frames"]
    screen = payload["screen"]
    bursts = {str(key): tuple(map(int, value)) for key, value in frames["burst_intervals_ui"].items()}
    if tuple(sorted(bursts)) != ("1", "2", "3", "4"):
        raise ValueError("exactly four numbered burst windows are required")
    quiet_start, quiet_stop = map(int, frames["quiet_interval_ui"])
    quiet_count = quiet_stop - quiet_start + 1
    if quiet_count != 100:
        raise ValueError("version 1 requires a 100-frame quiet interval")
    quiet_a = (quiet_start, quiet_start + 49)
    quiet_b = (quiet_start + 50, quiet_stop)
    guard = int(screen["heldout_guard_frames"])
    if guard < 10:
        raise ValueError("heldout burst guard must be at least ten frames")
    rows = []
    for fold, heldout in zip(TRAINING_FOLDS, sorted(bursts)):
        start, stop = bursts[heldout]
        training = tuple(key for key in sorted(bursts) if key != heldout)
        rows.append(
            FoldContract(
                training_fold=fold,
                heldout_burst=heldout,
                training_bursts=training,  # type: ignore[arg-type]
                heldout_guard_ui=(start - guard, stop + guard),
                quiet_half_a_ui=quiet_a,
                quiet_half_b_ui=quiet_b,
            )
        )
    return tuple(rows)


def _interval_mask(frame_ui: Any, interval: Sequence[int]) -> Any:
    start, stop = map(int, interval)
    return (frame_ui >= start) & (frame_ui <= stop)


def _positive_tail(values: Any, quantile: float) -> Any:
    """Return the frozen upper order statistic over each frame's positive map."""

    import torch

    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("tail input must be non-empty TYX")
    q = float(quantile)
    if not math.isfinite(q) or not 0.5 < q < 1.0:
        raise ValueError("positive_tail_quantile must be strictly between 0.5 and 1")
    flat = torch.clamp_min(values, 0.0).reshape(values.shape[0], -1)
    count = int(flat.shape[1])
    rank = min(count, max(1, int(math.ceil(q * count))))
    return torch.kthvalue(flat, rank, dim=1).values


def _positive_scale_floor(local_std: Any, mask: Any, percentile: float) -> Any:
    import torch

    selected = local_std[mask]
    positive = selected[selected > 0]
    if positive.numel() == 0:
        raise ValueError("training quiet local standard deviation has no positive values")
    requested = float(percentile)
    if not math.isfinite(requested) or not 0.0 <= requested <= 100.0:
        raise ValueError("scale-floor percentile must be in [0,100]")
    floor = torch.quantile(positive, requested / 100.0)
    if not bool(torch.isfinite(floor)) or float(floor.item()) <= 0.0:
        raise ValueError("training quiet scale floor is not finite and positive")
    return floor


def _context_spec(context: GammaContext) -> GammaReferenceSpec:
    return GammaReferenceSpec.from_mode(
        context.context_id,
        support_width_px=2 * int(context.half_width_px) + 1,
        shape_n=float(context.shape),
        mode_radius_px=float(context.mode_radius_px),
        guard_radius_px=float(context.guard_radius_px),
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=GAMMA_EPSILON,
        # Cell-specific floors are fitted after the reference moments exist.
        scale_floor=0.0,
    )


def _elapsed(device: Any, operation: Callable[[], Any]) -> tuple[Any, float]:
    import torch

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        stop.record()
        stop.synchronize()
        return result, float(start.elapsed_time(stop))
    started = time.perf_counter()
    result = operation()
    return result, (time.perf_counter() - started) * 1000.0


def _representation(common: Any, name: str) -> Any:
    if name == "raw":
        return gpu_representations.raw_representation(common).values[1:]
    if name == "difference_signed":
        return gpu_representations.signed_difference_representation(common).values
    if name == "difference_energy_normalized":
        return gpu_representations.energy_normalized_difference_representation(
            common, epsilon=ENERGY_EPSILON
        ).values
    raise ValueError(f"screen does not accept representation {name!r}")


def _tail_metrics(
    score: Any,
    *,
    frame_ui: Any,
    fold: FoldContract,
    bursts: Mapping[str, Sequence[int]],
    quiet_tail_interval: Sequence[int],
    tail_quantile: float,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    guard_mask = _interval_mask(frame_ui, fold.heldout_guard_ui)
    burst_values = []
    burst_counts: dict[str, int] = {}
    for burst_id in fold.training_bursts:
        mask = _interval_mask(frame_ui, bursts[burst_id]) & ~guard_mask
        count = int(torch.count_nonzero(mask).item())
        if count < 1:
            raise ValueError(f"training burst {burst_id} has no guard-safe frames")
        burst_counts[burst_id] = count
        burst_values.append(_positive_tail(score[mask], tail_quantile).mean())
    event_tail = torch.stack(burst_values).mean()
    quiet_mask = _interval_mask(frame_ui, quiet_tail_interval) & ~guard_mask
    quiet_count = int(torch.count_nonzero(quiet_mask).item())
    if quiet_count < 1:
        raise ValueError("quiet-tail interval has no guard-safe aligned frames")
    quiet_tail = _positive_tail(score[quiet_mask], tail_quantile).mean()
    return event_tail, quiet_tail, {
        "training_event_frame_counts": burst_counts,
        "quiet_tail_frames": quiet_count,
    }


def _evaluate_context_arm(
    representation: Any,
    *,
    representation_name: str,
    context: GammaContext,
    stage: str,
    folds: Sequence[FoldContract],
    frame_ui: Any,
    bursts: Mapping[str, Sequence[int]],
    scale_floor_percentile: float,
    tail_quantile: float,
    chunk_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    spec = _context_spec(context)
    moments, gamma_ms = _elapsed(
        representation.device,
        lambda: gamma_local_standardization(
            representation,
            spec,
            chunk_frames=chunk_frames,
            return_statistics=True,
        ),
    )
    if moments.local_mean is None or moments.local_std is None:
        raise AssertionError("Gamma-LS moments were not returned")
    if moments.values.device != representation.device:
        raise RuntimeError("Gamma-LS left the representation device")
    mean = moments.local_mean
    std = moments.local_std
    aggregate_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    for fold in folds:
        swap_payloads = []
        for swap_name, floor_interval, tail_interval in (
            (QUIET_SWAPS[0], fold.quiet_half_a_ui, fold.quiet_half_b_ui),
            (QUIET_SWAPS[1], fold.quiet_half_b_ui, fold.quiet_half_a_ui),
        ):
            guard_mask = _interval_mask(frame_ui, fold.heldout_guard_ui)
            floor_mask = _interval_mask(frame_ui, floor_interval) & ~guard_mask
            floor_count = int(torch.count_nonzero(floor_mask).item())
            if floor_count < 1:
                raise ValueError("quiet floor interval has no guard-safe aligned frames")

            def score_and_measure() -> tuple[Any, Any, Any, dict[str, Any]]:
                floor = _positive_scale_floor(std, floor_mask, scale_floor_percentile)
                score = (representation - mean) / (torch.maximum(std, floor) + GAMMA_EPSILON)
                event_tail, quiet_tail, counts = _tail_metrics(
                    score,
                    frame_ui=frame_ui,
                    fold=fold,
                    bursts=bursts,
                    quiet_tail_interval=tail_interval,
                    tail_quantile=tail_quantile,
                )
                return floor, event_tail, quiet_tail, counts

            measured, postprocess_ms = _elapsed(representation.device, score_and_measure)
            floor, event_tail, quiet_tail, counts = measured
            row = {
                "stage": stage,
                "context_id": context.context_id,
                "representation": representation_name,
                "training_fold": fold.training_fold,
                "heldout_burst": fold.heldout_burst,
                "training_bursts": json.dumps(list(fold.training_bursts)),
                "heldout_guard_ui": json.dumps(list(fold.heldout_guard_ui)),
                "quiet_swap": swap_name,
                "floor_interval_ui": json.dumps(list(floor_interval)),
                "tail_interval_ui": json.dumps(list(tail_interval)),
                "floor_eligible_aligned_frames": floor_count,
                "scale_floor_percentile": float(scale_floor_percentile),
                "scale_floor": float(floor.item()),
                "positive_tail_quantile": float(tail_quantile),
                "positive_tail_method": TAIL_QUANTILE_METHOD,
                "event_positive_tail": float(event_tail.item()),
                "quiet_positive_tail": float(quiet_tail.item()),
                "positive_tail_contrast": float((event_tail - quiet_tail).item()),
                "gamma_runtime_ms_per_frame": gamma_ms / int(representation.shape[0]),
                "postprocess_runtime_ms": postprocess_ms,
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "burst_windows_used": True,
                "training_event_frame_counts": json.dumps(
                    counts["training_event_frame_counts"], sort_keys=True
                ),
                "quiet_tail_frames": counts["quiet_tail_frames"],
            }
            swap_rows.append(row)
            swap_payloads.append(row)
        event_average = sum(row["event_positive_tail"] for row in swap_payloads) / 2.0
        quiet_average = sum(row["quiet_positive_tail"] for row in swap_payloads) / 2.0
        aggregate_rows.append(
            {
                "stage": stage,
                "context_id": context.context_id,
                "representation": representation_name,
                "training_fold": fold.training_fold,
                "event_positive_tail": event_average,
                "quiet_positive_tail": quiet_average,
                "runtime_ms_per_frame": gamma_ms / int(representation.shape[0]),
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "burst_windows_used": True,
            }
        )
    del moments, mean, std
    return aggregate_rows, swap_rows, {
        "context_id": context.context_id,
        "representation": representation_name,
        "gamma_runtime_ms": gamma_ms,
        "gamma_runtime_ms_per_frame": gamma_ms / int(representation.shape[0]),
        "fold_moment_reuse": True,
        "quiet_role_swaps_per_cell": 2,
    }


def _aggregate_dict(row: ContextAggregate) -> dict[str, Any]:
    return row.as_dict()


def _fold_rows(
    rows: Sequence[Mapping[str, Any]], fold: int
) -> list[Mapping[str, Any]]:
    return [row for row in rows if int(row["training_fold"]) == fold]


def _execute_device_screen(
    movie_path: Path,
    *,
    config: GammaLSDifferenceConfig,
    runtime: Mapping[str, Any],
    heartbeat: Callable[[Mapping[str, Any]], None] | None = None,
) -> ScreenExecution:
    import torch

    device = torch.device(str(runtime["resolved_device"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_ui, stop_ui = map(int, config.payload["frames"]["review_interval_ui"])
    chunk_frames = max(
        int(value) for value in config.payload["efficiency"]["frame_chunks"]
    )
    common, history_timing = _stream_common_history_to_device(
        movie_path,
        review_start_ui=start_ui,
        review_stop_ui=stop_ui,
        chunk_frames=chunk_frames,
        device=device,
        heartbeat=heartbeat,
    )
    if device.type == "cuda" and not common.is_cuda:
        raise RuntimeError("common preprocessing left CUDA")

    # The retained predecessor makes all arms align to the complete review
    # interval, including UI frame 1800 and both 50-frame quiet halves.
    frame_ui = torch.arange(start_ui, stop_ui + 1, device=device, dtype=torch.int64)
    folds = build_fold_contracts(config)
    bursts = config.payload["frames"]["burst_intervals_ui"]
    grid_payload = config.payload["gamma_ls_grid"]
    scale_percentile = float(grid_payload["screen_scale_floor_percentile"])
    tail_quantile = float(config.payload["screen"]["positive_tail_quantile"])

    g1_contexts = enumerate_g1_contexts(config)
    g1_rows: list[dict[str, Any]] = []
    g1_swaps: list[dict[str, Any]] = []
    timing_cells: list[dict[str, Any]] = []
    representation_timings: dict[str, float] = {}
    completed = 0
    for representation_name in SCREEN_REPRESENTATIONS:
        representation, representation_ms = _elapsed(
            device, lambda name=representation_name: _representation(common, name)
        )
        representation_timings[representation_name] = representation_ms
        if tuple(representation.shape) != (len(frame_ui), *common.shape[1:]):
            raise RuntimeError("screen representations are not commonly aligned")
        for context in g1_contexts:
            rows, swaps, timing = _evaluate_context_arm(
                representation,
                representation_name=representation_name,
                context=context,
                stage="g1",
                folds=folds,
                frame_ui=frame_ui,
                bursts=bursts,
                scale_floor_percentile=scale_percentile,
                tail_quantile=tail_quantile,
                chunk_frames=chunk_frames,
            )
            g1_rows.extend(rows)
            g1_swaps.extend(swaps)
            timing_cells.append({"stage": "g1", **timing})
            completed += 1
            if heartbeat is not None:
                heartbeat({"stage": "g1", "completed_context_arm_maps": completed, "total": 27})
        del representation

    if len(g1_rows) != 108 or len(g1_swaps) != 216:
        raise AssertionError("G1 did not produce 108 cells and 216 quiet-swap rows")
    g1_selections = {}
    g2_by_fold: dict[int, tuple[GammaContext, ...]] = {}
    for fold in TRAINING_FOLDS:
        selection = select_g1_radius_guard_pairs(
            _fold_rows(g1_rows, fold), g1_contexts, training_fold=fold
        )
        g1_selections[fold] = selection
        g2_by_fold[fold] = enumerate_g2_contexts(config, selection.retained_pairs)

    # Reuse an identical context/arm moment map across folds when G1 retained
    # the same geometry, while still fitting two cell-specific quiet floors.
    contexts_by_id: dict[str, GammaContext] = {}
    folds_by_context: dict[str, list[FoldContract]] = {}
    for fold in folds:
        for context in g2_by_fold[fold.training_fold]:
            contexts_by_id[context.context_id] = context
            folds_by_context.setdefault(context.context_id, []).append(fold)

    g2_rows: list[dict[str, Any]] = []
    g2_swaps: list[dict[str, Any]] = []
    completed = 0
    total_maps = len(contexts_by_id) * len(SCREEN_REPRESENTATIONS)
    for representation_name in SCREEN_REPRESENTATIONS:
        representation, representation_ms = _elapsed(
            device, lambda name=representation_name: _representation(common, name)
        )
        representation_timings[f"g2_rebuild_{representation_name}"] = representation_ms
        for context_id in sorted(contexts_by_id):
            rows, swaps, timing = _evaluate_context_arm(
                representation,
                representation_name=representation_name,
                context=contexts_by_id[context_id],
                stage="g2",
                folds=folds_by_context[context_id],
                frame_ui=frame_ui,
                bursts=bursts,
                scale_floor_percentile=scale_percentile,
                tail_quantile=tail_quantile,
                chunk_frames=chunk_frames,
            )
            g2_rows.extend(rows)
            g2_swaps.extend(swaps)
            timing_cells.append({"stage": "g2", **timing})
            completed += 1
            if heartbeat is not None:
                heartbeat({"stage": "g2", "completed_context_arm_maps": completed, "total": total_maps})
        del representation
    if len(g2_rows) != 216 or len(g2_swaps) != 432:
        raise AssertionError("G2 did not produce 216 cells and 432 quiet-swap rows")

    fold_selections = []
    for fold in TRAINING_FOLDS:
        fold_contract = next(row for row in folds if row.training_fold == fold)
        g1_selection = g1_selections[fold]
        g2_selection = select_g2_common_finalist(
            _fold_rows(g2_rows, fold), g2_by_fold[fold], training_fold=fold
        )
        fold_selections.append(
            {
                "training_fold": fold,
                "heldout_burst": fold_contract.heldout_burst,
                "training_bursts": list(fold_contract.training_bursts),
                "heldout_guard_ui": list(fold_contract.heldout_guard_ui),
                "g1_retained_context_ids": list(g1_selection.retained_context_ids),
                "g1_retained_radius_guard_pairs": [
                    {
                        "half_width_px": pair.half_width_px,
                        "guard_radius_px": pair.guard_radius_px,
                    }
                    for pair in g1_selection.retained_pairs
                ],
                "g1_ranked_contexts": [
                    _aggregate_dict(row) for row in g1_selection.ranked_contexts
                ],
                "common_g2_finalist_context_id": g2_selection.finalist_context_id,
                "g2_ranked_contexts": [
                    _aggregate_dict(row) for row in g2_selection.ranked_contexts
                ],
            }
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    else:  # used only by focused fixtures
        peak_memory = {"max_memory_allocated_bytes": 0, "max_memory_reserved_bytes": 0}
    return ScreenExecution(
        g1_rows=tuple(g1_rows),
        g1_swap_rows=tuple(g1_swaps),
        g2_rows=tuple(g2_rows),
        g2_swap_rows=tuple(g2_swaps),
        fold_selections=tuple(fold_selections),
        runtime=dict(runtime),
        timings={
            **history_timing,
            "representation_ms": representation_timings,
            "context_arm_maps": timing_cells,
            "cuda_synchronization": "event stop synchronized for every timed device operation",
        },
        peak_memory=peak_memory,
    )


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
        if path.is_file() and path.name != "artifact_index.json" and not path.name.endswith(".partial"):
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"schema_version": 1, "artifacts": rows}


def run_gpu_screen(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run G1/G2 and atomically commit a coordinate-free screen artifact."""

    if not isinstance(config, GammaLSDifferenceConfig):
        raise TypeError("config must be a validated GammaLSDifferenceConfig")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"screen output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"screen output parent does not exist: {destination.parent}")
    preflight = _verify_ready_preflight(config, preflight_dir)
    # Required before either the requested output or a sibling partial exists.
    runtime = _require_cuda(device)
    source = _source_contract(config)

    partial = destination.parent / f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    if partial.exists():
        raise FileExistsError(f"screen partial collision: {partial}")
    try:
        partial.mkdir()

        def heartbeat(payload: Mapping[str, Any]) -> None:
            _atomic_json(
                partial / "heartbeat.json",
                {"updated_at_utc": datetime.now(timezone.utc).isoformat(), **payload},
            )

        heartbeat({"stage": "starting", "status": "running"})
        execution = _execute_device_screen(
            config.source_paths["movie"],
            config=config,
            runtime=runtime,
            heartbeat=heartbeat,
        )
        _atomic_tsv(partial / "g1_metric_rows.tsv", execution.g1_rows)
        _atomic_tsv(partial / "g1_quiet_swap_rows.tsv", execution.g1_swap_rows)
        _atomic_tsv(partial / "g2_metric_rows.tsv", execution.g2_rows)
        _atomic_tsv(partial / "g2_quiet_swap_rows.tsv", execution.g2_swap_rows)
        selection = {
            "selection_scope": "outer_training_fold_only",
            "folds": list(execution.fold_selections),
            "deployment_context_id": None,
            "deployment_context_status": "deferred_until_after_protected_cross_validation",
            "across_fold_pooling_used_for_protected_selection": False,
        }
        _atomic_json(partial / "selection.json", selection)
        _atomic_json(partial / "timings.json", execution.timings)
        _atomic_json(
            partial / "provenance_hashes.json",
            {
                "portable_config_sha256": _canonical_sha256(config.portable_dict()),
                "preflight": preflight,
                "movie_sha256": preflight.get("movie_sha256"),
                "implementation": {
                    relative: _sha256(config.repository / relative)
                    for relative in (
                        "neurobench/algorithms/gamma_local_standardization.py",
                        "neurobench/experiments/gamma_ls_difference/gpu_representations.py",
                        "neurobench/experiments/gamma_ls_difference/grid.py",
                        "neurobench/experiments/gamma_ls_difference/screen.py",
                    )
                },
            },
        )
        claim_boundary = {
            "positive_coordinates_used": False,
            "positive_identities_used": False,
            "burst_windows_used": True,
            "burst_windows_source": "human_declared_manifest_intervals",
            "fully_label_free_claimed": False,
            "sparse_positive_performance_computed": False,
            "cfar_candidates_generated": False,
            "scientific_audit_complete": False,
            "multiscale_inference_claimed": False,
            "interpretation": (
                "This screen selects one guarded radial Gamma-LS context inside each "
                "outer training fold. It cannot establish recall, precision, ICA benefit, "
                "a deployment context, or 1-kHz readiness."
            ),
        }
        _atomic_json(partial / "claim_boundary.json", claim_boundary)
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "run_type": "coordinate_free_burst_window_gamma_ls_gpu_screen",
            "status": "complete_screen_only",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "runtime": dict(execution.runtime),
            "peak_gpu_memory": dict(execution.peak_memory),
            "positive_tail": {
                "quantile": float(config.payload["screen"]["positive_tail_quantile"]),
                "method": TAIL_QUANTILE_METHOD,
                "aggregation": "mean_frame_tail_then_equal_mean_across_three_training_bursts",
            },
            "quiet_role_swaps": list(QUIET_SWAPS),
            "g1_metric_cell_count": len(execution.g1_rows),
            "g1_quiet_swap_row_count": len(execution.g1_swap_rows),
            "g2_metric_cell_count": len(execution.g2_rows),
            "g2_quiet_swap_row_count": len(execution.g2_swap_rows),
            "selection": selection,
            "claim_boundary": claim_boundary,
        }
        _atomic_json(partial / "summary.json", summary)
        _atomic_json(
            partial / "validation.json",
            {
                "status": "passed_screen_artifact_contract",
                "checks": {
                    "cuda_required_and_used": True,
                    "chunked_h2d_matches_history_chunk_count": (
                        execution.timings["movie_h2d_transfer_count"]
                        == execution.timings["history_chunk_count"]
                    ),
                    "ema_state_carried_across_chunks": execution.timings[
                        "ema_state_carried_across_chunks"
                    ],
                    "g1_has_108_metric_cells": len(execution.g1_rows) == 108,
                    "g2_has_216_metric_cells": len(execution.g2_rows) == 216,
                    "two_quiet_role_swaps_per_cell": (
                        len(execution.g1_swap_rows) == 216
                        and len(execution.g2_swap_rows) == 432
                    ),
                    "four_fold_local_finalists": len(execution.fold_selections) == 4,
                    "deployment_context_deferred": True,
                    "positive_coordinates_used": False,
                    "positive_identities_used": False,
                    "burst_windows_used": True,
                },
            },
        )
        _atomic_json(
            partial / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "grain": "Gamma context x fixed representation x outer training fold",
                "selection": "selection.json",
                "metrics": ["g1_metric_rows.tsv", "g2_metric_rows.tsv"],
                "quiet_swap_evidence": [
                    "g1_quiet_swap_rows.tsv",
                    "g2_quiet_swap_rows.tsv",
                ],
                "labels": (
                    "human burst windows used; sparse-positive coordinates and identities "
                    "not opened or supplied"
                ),
                "claim_boundary": "claim_boundary.json",
            },
        )
        heartbeat({"stage": "complete", "status": "complete_screen_only"})
        _atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        if destination.exists():
            raise FileExistsError(f"screen output appeared during execution: {destination}")
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return summary


__all__ = [
    "CudaScreenUnavailable",
    "FoldContract",
    "QUIET_SWAPS",
    "ScreenExecution",
    "TAIL_QUANTILE_METHOD",
    "build_fold_contracts",
    "run_gpu_screen",
]
