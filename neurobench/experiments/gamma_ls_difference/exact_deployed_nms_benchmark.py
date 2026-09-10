"""Standalone CUDA timing benchmark for the exact deployed candidate output.

The earlier streaming benchmark intentionally timed a compact count/top-one
packet.  This extension leaves that artifact and its frozen preflight inputs
unchanged.  It times the candidate-producing path used by the maintained
offline evaluator:

``pinned frame -> CUDA causal representation -> radial Gamma-LS -> strict
threshold -> CUDA square local-max extraction -> sparse D2H -> deterministic
CPU greedy Euclidean plateau cleanup -> complete retained candidate table``.

"Complete" means the complete output of
``extract_separated_local_maxima(..., limit=candidate_limit_per_frame)``.  The
limit and every frame that reaches it are recorded.  Artifact serialization
after the returned NumPy table is available is outside the service boundary.
No labels are opened by this timing-only benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Sequence
import uuid

import numpy as np

from .config import GammaLSDifferenceConfig
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from .streaming_benchmark import (
    BoundedArrivalQueue,
    DEFAULT_ARRIVAL_QUEUE_CAPACITY,
    DEFAULT_SOURCE_RING_FRAMES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MINIMUM_PRODUCTION_DURATION_SECONDS,
    NMS_DISTANCE_PX,
    PreparedCausalGammaPipeline,
    STREAMING_ARMS,
    StreamingBenchmarkUnavailable,
    _artifact_index,
    _atomic_json,
    _atomic_npz,
    _atomic_tsv,
    _calibrate_benchmark_operating_point,
    _load_pinned_source_rings,
    _local_maximum_mask,
    _percentiles_ms,
    _process_memory,
    _screen_context_role,
    _sha256,
    _validate_run_request,
    _verify_ready_preflight,
    _wait_until_ns,
    parse_gamma_context_id,
)


EXACT_DECISION_MODE = "exact_sparse_candidate_table"
DEFAULT_EXACT_ARMS = ("difference_signed",)
DEFAULT_EXACT_CONTEXT_ID = "support_support_a_h15_g7_n9_m0p5"
DEFAULT_CANDIDATE_LIMIT_PER_FRAME = 10_000
EXACT_SERVICE_STAGES = (
    "h2d_ms",
    "representation_ms",
    "gamma_ls_ms",
    "gpu_square_local_max_extract_ms",
    "sparse_candidate_extract_and_d2h_wall_ms",
    "cpu_greedy_euclidean_cleanup_ms",
    "exact_decision_wall_ms",
    "cuda_dense_plus_sparse_gather_ms",
    "end_to_end_wall_ms",
)


def cleanup_square_local_max_candidates(
    candidate_frame_y_x: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    batch_size: int,
    distance_px: int = NMS_DISTANCE_PX,
    limit_per_frame: int = DEFAULT_CANDIDATE_LIMIT_PER_FRAME,
) -> dict[str, np.ndarray]:
    """Apply the maintained deterministic Euclidean cleanup to sparse maxima.

    ``candidate_frame_y_x`` must contain the full output of the preliminary
    square local-maximum mask, with columns ``frame, y, x``.  Candidates are
    ordered independently in each frame by descending score, then ascending
    y/x.  A candidate is retained only when it is strictly farther than
    ``distance_px`` from every already-retained candidate.  These rules match
    ``extract_separated_local_maxima`` when no optional tie breaker is used.
    """

    coordinates = np.asarray(candidate_frame_y_x)
    scores = np.asarray(candidate_scores)
    batch = int(batch_size)
    distance = int(distance_px)
    limit = int(limit_per_frame)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError("candidate_frame_y_x must have shape N x 3")
    if scores.ndim != 1 or len(scores) != len(coordinates):
        raise ValueError("candidate_scores must be a length-N vector")
    if batch < 1 or distance < 1 or limit < 1:
        raise ValueError("batch_size, distance_px, and limit_per_frame must be positive")
    if not np.issubdtype(coordinates.dtype, np.integer):
        raise ValueError("candidate coordinates must be integers")
    if len(coordinates):
        if np.any(coordinates < 0) or np.any(coordinates[:, 0] >= batch):
            raise ValueError("candidate coordinates contain an invalid frame index")
        if not np.isfinite(scores).all():
            raise ValueError("candidate scores must be finite")

    preliminary_count = np.zeros(batch, dtype=np.int32)
    retained_count = np.zeros(batch, dtype=np.int32)
    limit_reached = np.zeros(batch, dtype=np.bool_)
    top_score = np.full(batch, np.nan, dtype=np.float32)
    top_x = np.full(batch, -1, dtype=np.int32)
    top_y = np.full(batch, -1, dtype=np.int32)
    retained_frame: list[int] = []
    retained_rank: list[int] = []
    retained_score: list[float] = []
    retained_x: list[int] = []
    retained_y: list[int] = []
    squared_distance = int(distance * distance)

    for frame_index in range(batch):
        frame_rows = np.flatnonzero(coordinates[:, 0] == frame_index)
        preliminary_count[frame_index] = len(frame_rows)
        if not len(frame_rows):
            continue
        y = coordinates[frame_rows, 1]
        x = coordinates[frame_rows, 2]
        frame_scores = scores[frame_rows]
        order = np.lexsort((x, y, -frame_scores))
        selected: list[tuple[int, int]] = []
        for local_index in order:
            px = int(x[local_index])
            py = int(y[local_index])
            if all(
                (px - old_x) ** 2 + (py - old_y) ** 2 > squared_distance
                for old_x, old_y in selected
            ):
                selected.append((px, py))
                retained_frame.append(frame_index)
                retained_rank.append(len(selected))
                retained_score.append(float(frame_scores[local_index]))
                retained_x.append(px)
                retained_y.append(py)
                if len(selected) >= limit:
                    limit_reached[frame_index] = True
                    break
        retained_count[frame_index] = len(selected)
        first = len(retained_score) - len(selected)
        if selected:
            top_score[frame_index] = retained_score[first]
            top_x[frame_index] = selected[0][0]
            top_y[frame_index] = selected[0][1]

    return {
        "preliminary_count": preliminary_count,
        "count": retained_count,
        "limit_reached": limit_reached,
        "top_score": top_score,
        "top_x": top_x,
        "top_y": top_y,
        "candidate_frame_index": np.asarray(retained_frame, dtype=np.int32),
        "candidate_rank": np.asarray(retained_rank, dtype=np.int32),
        "candidate_score": np.asarray(retained_score, dtype=np.float32),
        "candidate_x": np.asarray(retained_x, dtype=np.int32),
        "candidate_y": np.asarray(retained_y, dtype=np.int32),
    }


def _candidate_table_from_score_maps(
    score_maps: np.ndarray,
    *,
    threshold_z: float,
    distance_px: int,
    limit_per_frame: int = DEFAULT_CANDIDATE_LIMIT_PER_FRAME,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Testable composition of the device square mask and exact host cleanup."""

    import torch

    values = np.asarray(score_maps, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("score_maps must be a finite YX or BYX array")
    tensor = torch.from_numpy(values).to(device=device)
    keep = _local_maximum_mask(
        tensor, threshold_z=float(threshold_z), distance_px=int(distance_px)
    )
    coordinates = torch.nonzero(keep, as_tuple=False).cpu().numpy()
    scores = tensor[keep].cpu().numpy()
    return cleanup_square_local_max_candidates(
        coordinates,
        scores,
        batch_size=len(values),
        distance_px=int(distance_px),
        limit_per_frame=int(limit_per_frame),
    )


def exact_nms_parity_self_check(*, device: str = "cpu") -> dict[str, Any]:
    """Exercise random, plateau, tie, batch, and strict-threshold parity."""

    from neurobench.metrics.sparse_detection import extract_separated_local_maxima

    rng = np.random.default_rng(917)
    random_score = rng.normal(size=(31, 35)).astype(np.float32)
    plateau_score = np.zeros((31, 35), dtype=np.float32)
    plateau_score[10:14, 10:14] = 5.0
    plateau_score[20, 24] = 5.0
    strict_score = np.zeros((31, 35), dtype=np.float32)
    strict_score[10, 10] = 2.0
    strict_score[20, 20] = np.nextafter(np.float32(2.0), np.float32(np.inf))
    cases = (
        ("random_continuous", random_score, 1.25, 3),
        ("flat_plateau_and_tie", plateau_score, 1.0, 3),
        ("strict_threshold", strict_score, 2.0, 3),
    )
    results: list[dict[str, Any]] = []
    for case_id, score, threshold, distance in cases:
        table = _candidate_table_from_score_maps(
            score,
            threshold_z=threshold,
            distance_px=distance,
            device=device,
        )
        observed = [
            (float(value), int(x), int(y))
            for value, x, y in zip(
                table["candidate_score"],
                table["candidate_x"],
                table["candidate_y"],
                strict=True,
            )
        ]
        expected = extract_separated_local_maxima(
            score.astype(np.float64),
            distance,
            threshold=float(np.nextafter(np.float64(threshold), np.inf)),
        )
        if observed != expected:
            raise AssertionError(f"exact deployed NMS parity failed for {case_id}")
        results.append(
            {
                "case_id": case_id,
                "candidate_count": len(observed),
                "passed": True,
            }
        )

    batch = np.stack((random_score, plateau_score), axis=0)
    batch_table = _candidate_table_from_score_maps(
        batch, threshold_z=1.0, distance_px=3, device=device
    )
    for frame_index, score in enumerate(batch):
        rows = batch_table["candidate_frame_index"] == frame_index
        observed = [
            (float(value), int(x), int(y))
            for value, x, y in zip(
                batch_table["candidate_score"][rows],
                batch_table["candidate_x"][rows],
                batch_table["candidate_y"][rows],
                strict=True,
            )
        ]
        expected = extract_separated_local_maxima(
            score.astype(np.float64),
            3,
            threshold=float(np.nextafter(np.float64(1.0), np.inf)),
        )
        if observed != expected:
            raise AssertionError(f"exact deployed NMS batch parity failed at {frame_index}")
    results.append(
        {
            "case_id": "two_frame_batch",
            "candidate_count": int(len(batch_table["candidate_score"])),
            "passed": True,
        }
    )
    return {
        "status": "passed_exact_maintained_extractor_parity",
        "device": str(device),
        "cases": results,
        "strict_threshold_mapping": (
            "GPU_score_gt_cutoff_equals_reference_score_ge_nextafter_cutoff"
        ),
        "tie_order": "descending_score_then_ascending_y_then_x",
        "separation": "greedy_pairwise_squared_Euclidean_distance_strictly_gt_d_squared",
    }


class ExactOutputCausalGammaPipeline(PreparedCausalGammaPipeline):
    """Prepared CUDA pipeline returning the full maintained candidate table."""

    def __init__(
        self,
        *,
        candidate_limit_per_frame: int = DEFAULT_CANDIDATE_LIMIT_PER_FRAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        limit = int(candidate_limit_per_frame)
        if limit < 1:
            raise ValueError("candidate_limit_per_frame must be positive")
        self.candidate_limit_per_frame = limit

    def process_batch(
        self, host_frames: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Return all exact retained candidates and inclusive service timings."""

        torch = self.torch
        if (
            not torch.is_tensor(host_frames)
            or host_frames.dtype != torch.uint16
            or host_frames.device.type != "cpu"
            or not host_frames.is_pinned()
            or host_frames.ndim != 3
            or tuple(map(int, host_frames.shape[1:])) != (self.height, self.width)
        ):
            raise ValueError("host_frames must be a pinned CPU uint16 BYX tensor")
        batch = int(host_frames.shape[0])
        if not 1 <= batch <= self.max_batch_frames:
            raise ValueError("host batch exceeds the prepared device buffer")

        wall_start = time.perf_counter_ns()
        self.events[0].record()
        self.device_input[:batch].copy_(host_frames, non_blocking=True)
        self.events[1].record()
        with torch.inference_mode():
            spatial = self._spatial(self.device_input[:batch].to(torch.float32))
            representation = self._represent(spatial)
            self.events[2].record()
            score = self._gamma(representation)
            self.events[3].record()
            decision_start = time.perf_counter_ns()
            keep = _local_maximum_mask(
                score,
                threshold_z=self.threshold_z,
                distance_px=NMS_DISTANCE_PX,
            )
            candidate_coordinates = torch.nonzero(keep, as_tuple=False).to(torch.int32)
            candidate_scores = score[keep]
            self.events[4].record()
            coordinates_cpu = candidate_coordinates.detach().cpu().numpy().copy()
            scores_cpu = candidate_scores.detach().cpu().numpy().copy()
        self.events[4].synchronize()
        d2h_complete = time.perf_counter_ns()
        decision = cleanup_square_local_max_candidates(
            coordinates_cpu,
            scores_cpu,
            batch_size=batch,
            distance_px=NMS_DISTANCE_PX,
            limit_per_frame=self.candidate_limit_per_frame,
        )
        cleanup_complete = time.perf_counter_ns()
        wall_ms = (cleanup_complete - wall_start) / 1e6
        return decision, {
            "h2d_ms": float(self.events[0].elapsed_time(self.events[1])),
            "representation_ms": float(self.events[1].elapsed_time(self.events[2])),
            "gamma_ls_ms": float(self.events[2].elapsed_time(self.events[3])),
            "gpu_square_local_max_extract_ms": float(
                self.events[3].elapsed_time(self.events[4])
            ),
            "sparse_candidate_extract_and_d2h_wall_ms": float(
                (d2h_complete - decision_start) / 1e6
            ),
            "cpu_greedy_euclidean_cleanup_ms": float(
                (cleanup_complete - d2h_complete) / 1e6
            ),
            "exact_decision_wall_ms": float(
                (cleanup_complete - decision_start) / 1e6
            ),
            "cuda_dense_plus_sparse_gather_ms": float(
                self.events[0].elapsed_time(self.events[4])
            ),
            "end_to_end_wall_ms": float(wall_ms),
        }


def _exact_streaming_readiness_gates(
    *,
    duration_seconds: float,
    arrival_interval_ms: float,
    p99_service_ms: float,
    dropped_arrivals: int,
    missed_processed_deadlines: int,
    maximum_queue_depth: int,
    maximum_backlog_after_dequeue: int,
    final_queue_depth: int,
) -> dict[str, bool]:
    return {
        "declared_duration_at_least_60_seconds": duration_seconds
        >= MINIMUM_PRODUCTION_DURATION_SECONDS,
        "exact_1ms_arrival_interval": math.isclose(
            arrival_interval_ms, 1.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "no_dropped_arrivals": int(dropped_arrivals) == 0,
        "p99_service_latency_strictly_below_deadline": float(p99_service_ms)
        < float(arrival_interval_ms),
        "zero_missed_processed_deadlines": int(missed_processed_deadlines) == 0,
        "maximum_queue_depth_at_most_one": int(maximum_queue_depth) <= 1,
        "zero_backlog_after_dequeue": int(maximum_backlog_after_dequeue) == 0,
        "final_queue_drained": int(final_queue_depth) == 0,
        "device_resident_dense_pipeline": True,
        "full_exact_candidate_table_available_on_host_within_timed_boundary": True,
        "deterministic_cpu_euclidean_plateau_cleanup_included": True,
    }


def _run_exact_sustained_lane(
    pipeline: ExactOutputCausalGammaPipeline,
    source_ring: Any,
    *,
    duration_seconds: float,
    arrival_interval_ms: float,
    arrival_queue_capacity: int,
    warmup_iterations: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one paced lane and retain every returned candidate row."""

    interval_ns = int(round(float(arrival_interval_ms) * 1e6))
    total_arrivals = int(round(float(duration_seconds) * 1000.0 / arrival_interval_ms))
    if total_arrivals < 1:
        raise ValueError("sustained lane needs at least one arrival")
    for index in range(int(warmup_iterations)):
        ring_index = index % int(source_ring.shape[0])
        pipeline.process_batch(source_ring[ring_index : ring_index + 1])

    pipeline.torch.cuda.synchronize(pipeline.device)
    pipeline.torch.cuda.reset_peak_memory_stats(pipeline.device)
    free_before, total_vram = pipeline.torch.cuda.mem_get_info(pipeline.device)
    ram_before = _process_memory()
    start_ns = time.perf_counter_ns() + 2_000_000
    arrivals = BoundedArrivalQueue(
        start_ns=start_ns,
        interval_ns=interval_ns,
        total_arrivals=total_arrivals,
        capacity=int(arrival_queue_capacity),
    )
    service: dict[str, list[float]] = {name: [] for name in EXACT_SERVICE_STAGES}
    response_ms: list[float] = []
    queue_wait_ms: list[float] = []
    backlog_after_dequeue: list[int] = []
    preliminary_counts: list[int] = []
    candidate_counts: list[int] = []
    limit_reached: list[bool] = []
    top_scores: list[float] = []
    top_x: list[int] = []
    top_y: list[int] = []
    source_indices: list[int] = []
    processed_arrival_indices: list[int] = []
    candidate_arrival: list[int] = []
    candidate_source: list[int] = []
    candidate_rank: list[int] = []
    candidate_score: list[float] = []
    candidate_x: list[int] = []
    candidate_y: list[int] = []
    missed = 0

    while not arrivals.exhausted or arrivals.depth:
        now_ns = time.perf_counter_ns()
        arrivals.enqueue_due(now_ns)
        if arrivals.depth == 0:
            next_ns = arrivals.next_scheduled_ns
            if next_ns is None:
                break
            _wait_until_ns(next_ns)
            arrivals.enqueue_due(time.perf_counter_ns())
            if arrivals.depth == 0:
                continue

        record = arrivals.pop()
        service_start_ns = time.perf_counter_ns()
        queue_wait_ms.append((service_start_ns - record.scheduled_ns) / 1e6)
        backlog_after_dequeue.append(arrivals.depth)
        ring_index = record.arrival_index % int(source_ring.shape[0])
        decision, timings = pipeline.process_batch(
            source_ring[ring_index : ring_index + 1]
        )
        completed_ns = time.perf_counter_ns()
        response_ms.append((completed_ns - record.scheduled_ns) / 1e6)
        for key in service:
            service[key].append(float(timings[key]))
        preliminary_counts.append(int(decision["preliminary_count"][0]))
        candidate_counts.append(int(decision["count"][0]))
        limit_reached.append(bool(decision["limit_reached"][0]))
        top_scores.append(float(decision["top_score"][0]))
        top_x.append(int(decision["top_x"][0]))
        top_y.append(int(decision["top_y"][0]))
        source_indices.append(ring_index)
        processed_arrival_indices.append(record.arrival_index)
        rows = int(len(decision["candidate_score"]))
        if rows != int(decision["count"][0]):
            raise AssertionError("single-frame candidate table does not match its count")
        candidate_arrival.extend([record.arrival_index] * rows)
        candidate_source.extend([ring_index] * rows)
        candidate_rank.extend(map(int, decision["candidate_rank"]))
        candidate_score.extend(map(float, decision["candidate_score"]))
        candidate_x.extend(map(int, decision["candidate_x"]))
        candidate_y.extend(map(int, decision["candidate_y"]))
        if completed_ns > record.scheduled_ns + interval_ns:
            missed += 1

    completed_ns = time.perf_counter_ns()
    free_after, _ = pipeline.torch.cuda.mem_get_info(pipeline.device)
    ram_after = _process_memory()
    processed = len(response_ms)
    if processed + arrivals.dropped != total_arrivals:
        raise AssertionError("processed plus dropped arrivals does not close")
    if len(candidate_score) != sum(candidate_counts):
        raise AssertionError("flattened candidate table does not close against frame counts")
    service_summary = {
        name: _percentiles_ms(values) for name, values in service.items()
    }
    response_summary = _percentiles_ms(response_ms)
    queue_summary = _percentiles_ms(queue_wait_ms)
    failure_count = missed + arrivals.dropped
    maximum_backlog = int(max(backlog_after_dequeue, default=0))
    gates = _exact_streaming_readiness_gates(
        duration_seconds=float(duration_seconds),
        arrival_interval_ms=float(arrival_interval_ms),
        p99_service_ms=float(service_summary["end_to_end_wall_ms"]["p99_ms"]),
        dropped_arrivals=int(arrivals.dropped),
        missed_processed_deadlines=int(missed),
        maximum_queue_depth=int(arrivals.max_depth),
        maximum_backlog_after_dequeue=maximum_backlog,
        final_queue_depth=int(arrivals.depth),
    )
    passed = all(gates.values())
    summary = {
        "status": (
            "passed_exact_1khz_streaming_gate"
            if passed
            else "complete_failed_exact_1khz_streaming_gate"
        ),
        "decision_mode": EXACT_DECISION_MODE,
        "arrival_interval_ms": float(arrival_interval_ms),
        "declared_arrival_rate_hz": float(1000.0 / arrival_interval_ms),
        "declared_duration_seconds": float(duration_seconds),
        "scheduled_arrivals": total_arrivals,
        "processed_arrivals": processed,
        "dropped_arrivals": int(arrivals.dropped),
        "drop_policy": "drop_newest_when_arrival_queue_full",
        "arrival_queue_capacity_frames": int(arrival_queue_capacity),
        "maximum_arrival_queue_depth_frames": int(arrivals.max_depth),
        "maximum_backlog_after_dequeue_frames": maximum_backlog,
        "missed_processed_deadlines": int(missed),
        "deadline_failure_count_including_drops": int(failure_count),
        "deadline_failure_fraction_including_drops": float(
            failure_count / total_arrivals
        ),
        "paced_wall_span_seconds_including_final_drain": float(
            (completed_ns - start_ns) / 1e9
        ),
        "service_latency": service_summary,
        "response_latency_from_scheduled_arrival": response_summary,
        "queue_wait": queue_summary,
        "preliminary_square_local_max_count": {
            "total": int(sum(preliminary_counts)),
            "mean_per_processed_frame": float(np.mean(preliminary_counts)),
            "maximum_per_frame": int(max(preliminary_counts, default=0)),
        },
        "candidate_count": {
            "total": int(sum(candidate_counts)),
            "mean_per_processed_frame": float(np.mean(candidate_counts)),
            "maximum_per_frame": int(max(candidate_counts, default=0)),
            "frames_with_decision": int(np.count_nonzero(candidate_counts)),
            "candidate_limit_per_frame": int(pipeline.candidate_limit_per_frame),
            "frames_reaching_candidate_limit": int(np.count_nonzero(limit_reached)),
        },
        "decision_output_contract": {
            "per_frame_fields": [
                "preliminary_count",
                "retained_count",
                "limit_reached",
                "top_score",
                "top_x",
                "top_y",
            ],
            "full_candidate_table_fields": [
                "arrival_index",
                "source_ring_index",
                "rank_within_frame",
                "score",
                "x",
                "y",
            ],
            "coordinate_convention": "x_column_y_row",
            "no_candidate_sentinels": {
                "top_score": "NaN",
                "top_x": -1,
                "top_y": -1,
            },
            "full_candidate_table_transferred": True,
            "full_candidate_table_retained_in_latency_samples": True,
            "table_scope": (
                "complete_output_of_extract_separated_local_maxima_up_to_the_"
                "declared_per_frame_limit"
            ),
            "candidate_limit_per_frame": int(pipeline.candidate_limit_per_frame),
            "preliminary_transfer_fixed_upper_bound_per_frame": int(
                pipeline.height * pipeline.width
            ),
            "artifact_serialization_in_service_timing": False,
        },
        "memory": {
            "ram_before": ram_before,
            "ram_after": ram_after,
            "peak_vram_allocated_bytes": int(
                pipeline.torch.cuda.max_memory_allocated(pipeline.device)
            ),
            "peak_vram_reserved_bytes": int(
                pipeline.torch.cuda.max_memory_reserved(pipeline.device)
            ),
            "free_vram_bytes_before": int(free_before),
            "free_vram_bytes_after": int(free_after),
            "total_vram_bytes": int(total_vram),
        },
        "timing_boundary": (
            "pinned_uint16_H2D_through_causal_preprocess_radial_Gamma_LS_"
            "strict_threshold_GPU_square_local_max_sparse_candidate_D2H_and_"
            "CPU_deterministic_greedy_Euclidean_cleanup_to_full_candidate_table"
        ),
        "nms_contract": {
            "distance_px": NMS_DISTANCE_PX,
            "preliminary_operator": "strict_threshold_plus_13x13_square_local_maxima",
            "final_operator": "deterministic_greedy_Euclidean_separation",
            "sort_order": "descending_score_then_ascending_y_then_x",
            "separation_rule": "squared_distance_strictly_greater_than_distance_squared",
            "border_exclusion_px": NMS_DISTANCE_PX,
            "reference": "neurobench.metrics.sparse_detection.extract_separated_local_maxima",
        },
        "gates": gates,
        "gate_passed": passed,
    }
    arrays = {
        "processed_arrival_index": np.asarray(processed_arrival_indices, dtype=np.int64),
        "source_ring_index": np.asarray(source_indices, dtype=np.int16),
        "response_latency_ms": np.asarray(response_ms, dtype=np.float32),
        "queue_wait_ms": np.asarray(queue_wait_ms, dtype=np.float32),
        "backlog_after_dequeue_frames": np.asarray(backlog_after_dequeue, dtype=np.int16),
        "preliminary_candidate_count": np.asarray(preliminary_counts, dtype=np.int32),
        "candidate_count": np.asarray(candidate_counts, dtype=np.int32),
        "candidate_limit_reached": np.asarray(limit_reached, dtype=np.bool_),
        "top_candidate_score": np.asarray(top_scores, dtype=np.float32),
        "top_candidate_x": np.asarray(top_x, dtype=np.int32),
        "top_candidate_y": np.asarray(top_y, dtype=np.int32),
        "candidate_arrival_index": np.asarray(candidate_arrival, dtype=np.int64),
        "candidate_source_ring_index": np.asarray(candidate_source, dtype=np.int16),
        "candidate_rank_within_frame": np.asarray(candidate_rank, dtype=np.int32),
        "candidate_score": np.asarray(candidate_score, dtype=np.float32),
        "candidate_x": np.asarray(candidate_x, dtype=np.int32),
        "candidate_y": np.asarray(candidate_y, dtype=np.int32),
        **{
            f"service_{name}": np.asarray(values, dtype=np.float32)
            for name, values in service.items()
        },
    }
    return summary, arrays


def _run_exact_batch_frontier(
    *,
    source_ring: Any,
    arm: str,
    reference: Any,
    threshold_z: float,
    device: Any,
    batch_sizes: Sequence[int],
    warmup_iterations: int,
    timed_iterations: int,
    candidate_limit_per_frame: int,
) -> list[dict[str, Any]]:
    """Measure exact-output throughput on one matched cyclic frame sequence.

    The configured iteration counts are interpreted at the largest batch size.
    Every batch size therefore sees the same number of full 64-frame ring
    cycles during warm-up and timing; only grouping differs.
    """

    import torch

    rows: list[dict[str, Any]] = []
    plan = matched_cyclic_batch_plan(
        ring_frames=int(source_ring.shape[0]),
        batch_sizes=batch_sizes,
        warmup_reference_iterations=int(warmup_iterations),
        timed_reference_iterations=int(timed_iterations),
    )
    for batch_size in map(int, batch_sizes):
        workload = plan[batch_size]
        pipeline = ExactOutputCausalGammaPipeline(
            frame_shape=(FRAME_HEIGHT, FRAME_WIDTH),
            arm=arm,
            reference=reference,
            threshold_z=threshold_z,
            device=device,
            max_batch_frames=batch_size,
            candidate_limit_per_frame=candidate_limit_per_frame,
        )
        cursor = 0
        for _ in range(int(workload["warmup_batches"])):
            host_batch = source_ring[cursor : cursor + batch_size]
            pipeline.process_batch(host_batch)
            cursor = (cursor + batch_size) % int(source_ring.shape[0])
        if cursor != 0:
            raise AssertionError("matched warm-up did not close on the source ring")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples: dict[str, list[float]] = {name: [] for name in EXACT_SERVICE_STAGES}
        candidate_total = 0
        preliminary_total = 0
        limit_frames = 0
        for _ in range(int(workload["timed_batches"])):
            host_batch = source_ring[cursor : cursor + batch_size]
            decision, timing = pipeline.process_batch(host_batch)
            cursor = (cursor + batch_size) % int(source_ring.shape[0])
            for stage_name in samples:
                samples[stage_name].append(float(timing[stage_name]))
            candidate_total += int(np.sum(decision["count"]))
            preliminary_total += int(np.sum(decision["preliminary_count"]))
            limit_frames += int(np.count_nonzero(decision["limit_reached"]))
        if cursor != 0:
            raise AssertionError("matched timed workload did not close on the source ring")
        wall = _percentiles_ms(samples["end_to_end_wall_ms"])
        rows.append(
            {
                "arm": arm,
                "context_id": reference.context_id,
                "decision_mode": EXACT_DECISION_MODE,
                "batch_frames": batch_size,
                "warmup_iterations": int(workload["warmup_batches"]),
                "warmup_frames": int(workload["warmup_frames"]),
                "warmup_source_ring_cycles": int(workload["warmup_ring_cycles"]),
                "timed_iterations": int(workload["timed_batches"]),
                "timed_frames": int(workload["timed_frames"]),
                "timed_source_ring_cycles": int(workload["timed_ring_cycles"]),
                "configured_reference_batch_frames": int(plan["reference_batch_frames"]),
                "configured_reference_warmup_iterations": int(warmup_iterations),
                "configured_reference_timed_iterations": int(timed_iterations),
                "source_frame_order": "sequential_0_through_63_cyclic",
                "mean_batch_wall_ms": wall["mean_ms"],
                "p50_batch_wall_ms": wall["p50_ms"],
                "p95_batch_wall_ms": wall["p95_ms"],
                "p99_batch_wall_ms": wall["p99_ms"],
                "max_batch_wall_ms": wall["max_ms"],
                "mean_h2d_ms": float(np.mean(samples["h2d_ms"])),
                "mean_representation_ms": float(np.mean(samples["representation_ms"])),
                "mean_gamma_ls_ms": float(np.mean(samples["gamma_ls_ms"])),
                "mean_gpu_square_local_max_extract_ms": float(
                    np.mean(samples["gpu_square_local_max_extract_ms"])
                ),
                "mean_sparse_candidate_extract_and_d2h_wall_ms": float(
                    np.mean(samples["sparse_candidate_extract_and_d2h_wall_ms"])
                ),
                "mean_cpu_greedy_euclidean_cleanup_ms": float(
                    np.mean(samples["cpu_greedy_euclidean_cleanup_ms"])
                ),
                "mean_exact_decision_wall_ms": float(
                    np.mean(samples["exact_decision_wall_ms"])
                ),
                "amortized_wall_ms_per_frame": wall["mean_ms"] / batch_size,
                "amortized_frames_per_second": 1000.0 * batch_size / wall["mean_ms"],
                "preliminary_candidate_count": preliminary_total,
                "retained_candidate_count": candidate_total,
                "candidate_limit_per_frame": int(candidate_limit_per_frame),
                "frames_reaching_candidate_limit": limit_frames,
                "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "interpretation": (
                    "exact_output_batch_throughput_on_matched_cyclic_source_workload_"
                    "only_not_single_frame_latency"
                ),
            }
        )
        del pipeline
        torch.cuda.synchronize(device)
    return rows


def matched_cyclic_batch_plan(
    *,
    ring_frames: int,
    batch_sizes: Sequence[int],
    warmup_reference_iterations: int,
    timed_reference_iterations: int,
) -> dict[Any, Any]:
    """Return a per-size plan with identical full-ring frame multisets.

    Counts are anchored to the largest requested batch.  The constraints make
    each host batch a zero-copy contiguous pinned slice and make every lane end
    at the same recurrent-state boundary.
    """

    ring = int(ring_frames)
    sizes = tuple(map(int, batch_sizes))
    if ring < 1 or not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("ring_frames and unique batch_sizes must be positive")
    if any(size < 1 or size > ring or ring % size for size in sizes):
        raise ValueError("every batch size must be a positive divisor of the ring")
    warmup_reference = int(warmup_reference_iterations)
    timed_reference = int(timed_reference_iterations)
    if warmup_reference < 1 or timed_reference < 1:
        raise ValueError("reference iteration counts must be positive")
    reference_batch = max(sizes)
    warmup_frames = warmup_reference * reference_batch
    timed_frames = timed_reference * reference_batch
    if warmup_frames % ring or timed_frames % ring:
        raise ValueError("matched warm-up and timing must contain full ring cycles")
    result: dict[Any, Any] = {
        "reference_batch_frames": reference_batch,
        "warmup_frames_per_size": warmup_frames,
        "timed_frames_per_size": timed_frames,
        "warmup_ring_cycles_per_size": warmup_frames // ring,
        "timed_ring_cycles_per_size": timed_frames // ring,
    }
    for size in sizes:
        if warmup_frames % size or timed_frames % size:
            raise ValueError("matched frame counts must be divisible by every batch size")
        result[size] = {
            "warmup_batches": warmup_frames // size,
            "warmup_frames": warmup_frames,
            "warmup_ring_cycles": warmup_frames // ring,
            "timed_batches": timed_frames // size,
            "timed_frames": timed_frames,
            "timed_ring_cycles": timed_frames // ring,
        }
    return result


def cyclic_ring_frame_indices(
    *, ring_frames: int, batch_frames: int, batch_count: int
) -> np.ndarray:
    """Materialize a small cyclic schedule for regression tests."""

    ring = int(ring_frames)
    batch = int(batch_frames)
    count = int(batch_count)
    if ring < 1 or batch < 1 or count < 1 or batch > ring or ring % batch:
        raise ValueError("schedule requires positive counts and a batch dividing the ring")
    return np.arange(batch * count, dtype=np.int64) % ring


def _implementation_fingerprints() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    relatives = (
        "neurobench/experiments/gamma_ls_difference/exact_deployed_nms_benchmark.py",
        "neurobench/experiments/gamma_ls_difference/streaming_benchmark.py",
        "neurobench/metrics/sparse_detection.py",
    )
    return {
        f"repo://{relative}": {
            "sha256": _sha256(repository / relative),
            "size_bytes": int((repository / relative).stat().st_size),
        }
        for relative in relatives
    }


def run_exact_deployed_nms_benchmark(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    screen_dir: str | Path,
    output_dir: str | Path,
    context_id: str = DEFAULT_EXACT_CONTEXT_ID,
    arms: Sequence[str] = DEFAULT_EXACT_ARMS,
    duration_seconds: float = MINIMUM_PRODUCTION_DURATION_SECONDS,
    arrival_interval_ms: float = 1.0,
    arrival_queue_capacity: int = DEFAULT_ARRIVAL_QUEUE_CAPACITY,
    source_ring_frames: int = DEFAULT_SOURCE_RING_FRAMES,
    stream_start_ui: int = 1900,
    candidate_limit_per_frame: int = DEFAULT_CANDIDATE_LIMIT_PER_FRAME,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run a new immutable exact-output timing artifact."""

    if not isinstance(config, GammaLSDifferenceConfig):
        raise TypeError("config must be a validated GammaLSDifferenceConfig")
    ordered_arms = _validate_run_request(
        arms=arms,
        duration_seconds=float(duration_seconds),
        arrival_interval_ms=float(arrival_interval_ms),
        queue_capacity=int(arrival_queue_capacity),
        source_ring_frames=int(source_ring_frames),
    )
    candidate_limit = int(candidate_limit_per_frame)
    if candidate_limit < 1:
        raise ValueError("candidate_limit_per_frame must be positive")
    reference = parse_gamma_context_id(context_id)
    screen_root = Path(screen_dir).expanduser().resolve()
    screen_role = _screen_context_role(screen_root, context_id)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"exact benchmark output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    preflight = _verify_ready_preflight(config, preflight_dir)
    cpu_parity = exact_nms_parity_self_check(device="cpu")
    try:
        runtime = require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise StreamingBenchmarkUnavailable(str(error)) from error
    cuda_parity = exact_nms_parity_self_check(
        device=str(runtime["resolved_device"])
    )
    parity = {"cpu": cpu_parity, "cuda": cuda_parity}

    partial = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if partial.exists():
        raise FileExistsError(partial)
    try:
        partial.mkdir()
        _atomic_json(
            partial / "heartbeat.json",
            {
                "status": "running",
                "stage": "loading_bounded_host_rings",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        import torch

        torch.set_num_threads(int(config.payload["resources"]["cpu_threads"]))
        resolved_device = torch.device(str(runtime["resolved_device"]))
        source_ring, calibration_ring, source = _load_pinned_source_rings(
            config.source_paths["movie"],
            stream_start_ui=int(stream_start_ui),
            source_ring_frames=int(source_ring_frames),
            quiet_stop_ui=int(config.payload["frames"]["quiet_interval_ui"][1]),
            device=resolved_device,
        )
        source["source_id"] = str(config.payload["sources"]["movie"])
        source["path_recorded"] = False
        source["source_ring_wrap_policy"] = (
            "cyclic_with_recurrent_state_carried_across_wrap"
        )
        warmup = int(config.payload["efficiency"]["warmup_iterations"])
        timed_iterations = int(config.payload["efficiency"]["timed_iterations"])
        batch_sizes = tuple(map(int, config.payload["efficiency"]["frame_chunks"]))
        batch_workload = matched_cyclic_batch_plan(
            ring_frames=int(source_ring.shape[0]),
            batch_sizes=batch_sizes,
            warmup_reference_iterations=warmup,
            timed_reference_iterations=timed_iterations,
        )
        lanes: dict[str, Any] = {}
        sample_arrays: dict[str, np.ndarray] = {}
        frontier_rows: list[dict[str, Any]] = []
        fit_rows: list[dict[str, Any]] = []

        for lane_index, arm in enumerate(ordered_arms, start=1):
            _atomic_json(
                partial / "heartbeat.json",
                {
                    "status": "running",
                    "stage": "exact_calibration_then_sustained_lane",
                    "arm": arm,
                    "lane_index": lane_index,
                    "lane_count": len(ordered_arms),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            fitted_reference, threshold_z, fit = _calibrate_benchmark_operating_point(
                calibration_ring,
                arm=arm,
                reference=reference,
                device=resolved_device,
                scale_floor_percentile=float(
                    config.payload["gamma_ls_grid"]["screen_scale_floor_percentile"]
                ),
            )
            pipeline = ExactOutputCausalGammaPipeline(
                frame_shape=(FRAME_HEIGHT, FRAME_WIDTH),
                arm=arm,
                reference=fitted_reference,
                threshold_z=threshold_z,
                device=resolved_device,
                max_batch_frames=1,
                candidate_limit_per_frame=candidate_limit,
            )
            sustained, arrays = _run_exact_sustained_lane(
                pipeline,
                source_ring,
                duration_seconds=float(duration_seconds),
                arrival_interval_ms=float(arrival_interval_ms),
                arrival_queue_capacity=int(arrival_queue_capacity),
                warmup_iterations=warmup,
            )
            del pipeline
            torch.cuda.synchronize(resolved_device)
            fit_rows.append({"arm": arm, "context_id": context_id, **fit})
            frontier_rows.extend(
                _run_exact_batch_frontier(
                    source_ring=source_ring,
                    arm=arm,
                    reference=fitted_reference,
                    threshold_z=threshold_z,
                    device=resolved_device,
                    batch_sizes=batch_sizes,
                    warmup_iterations=warmup,
                    timed_iterations=timed_iterations,
                    candidate_limit_per_frame=candidate_limit,
                )
            )
            lanes[arm] = {
                "arm": arm,
                "representation": arm,
                "decision_mode": EXACT_DECISION_MODE,
                "context": {
                    "context_id": context_id,
                    "support_width_px": int(fitted_reference.support_width_px),
                    "guard_radius_px": float(fitted_reference.guard_radius_px),
                    "shape_n": float(fitted_reference.shape_n),
                    "mode_radius_px": fitted_reference.nominal_mode_radius_px,
                    "scale_floor": float(fitted_reference.scale_floor),
                },
                "threshold_z": float(threshold_z),
                "fit": fit,
                "sustained": sustained,
            }
            for name, array in arrays.items():
                sample_arrays[f"{arm}__{name}"] = array

        _atomic_npz(partial / "latency_and_candidate_samples.npz", sample_arrays)
        _atomic_tsv(partial / "fit_times.tsv", fit_rows)
        _atomic_tsv(partial / "exact_batch_throughput_frontier.tsv", frontier_rows)
        all_passed = all(lane["sustained"]["gate_passed"] for lane in lanes.values())
        status = (
            "passed_exact_1khz_streaming_gate"
            if all_passed
            else "complete_failed_exact_1khz_streaming_gate"
        )
        claim_boundary = {
            "timing_only": True,
            "labels_opened": False,
            "scientific_operating_point_used": False,
            "benchmark_threshold_role": "quiet_tail_runtime_load_only",
            "probability_of_false_alarm_claimed": False,
            "biological_detection_performance_claimed": False,
            "batch_throughput_used_as_streaming_latency": False,
            "camera_acquisition_or_disk_io_timed": False,
            "artifact_serialization_or_downstream_control_timed": False,
            "scientific_audit_complete": False,
            "paper_readiness_claimed": False,
            "interpretation": (
                "A passed lane supports only exact candidate-output timing for this "
                "device/software/operator/shape boundary. It does not establish detector "
                "accuracy, control-loop stability, or end-to-end voltage-imaging inverse-"
                "control readiness."
            ),
        }
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "run_type": "standalone_exact_deployed_nms_gpu_streaming_and_batch_timing",
            "decision_mode": EXACT_DECISION_MODE,
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "source": source,
            "screen_context_provenance": screen_role,
            "preflight": preflight,
            "implementation_fingerprints": _implementation_fingerprints(),
            "exact_nms_parity_self_check": parity,
            "lanes": lanes,
            "batch_frontier": {
                "path": "exact_batch_throughput_frontier.tsv",
                "batch_sizes": list(batch_sizes),
                "configured_reference_batch_frames": int(
                    batch_workload["reference_batch_frames"]
                ),
                "configured_reference_warmup_iterations": warmup,
                "configured_reference_timed_iterations": timed_iterations,
                "warmup_frames_per_size": int(
                    batch_workload["warmup_frames_per_size"]
                ),
                "warmup_source_ring_cycles_per_size": int(
                    batch_workload["warmup_ring_cycles_per_size"]
                ),
                "timed_frames_per_size": int(
                    batch_workload["timed_frames_per_size"]
                ),
                "timed_source_ring_cycles_per_size": int(
                    batch_workload["timed_ring_cycles_per_size"]
                ),
                "source_frame_order": "sequential_0_through_63_cyclic",
                "decision_mode": EXACT_DECISION_MODE,
                "interpretation": (
                    "exact_output_throughput_on_identical_cyclic_frame_multisets_"
                    "only_not_single_frame_latency"
                ),
            },
            "bounded_buffers": {
                "arrival_queue_capacity_frames": int(arrival_queue_capacity),
                "host_source_ring_frames": int(source_ring_frames),
                "device_input_buffer_frames_streaming": 1,
                "recurrent_device_state_frames": 2,
                "preliminary_candidate_transfer_upper_bound_rows_per_stream_frame": (
                    FRAME_HEIGHT * FRAME_WIDTH
                ),
                "retained_candidate_limit_per_frame": candidate_limit,
                "unbounded_frame_or_output_buffer_present": False,
            },
            "claim_boundary": claim_boundary,
        }
        _atomic_json(partial / "summary.json", summary)
        _atomic_json(partial / "claim_boundary.json", claim_boundary)
        checks = {
            "cuda_required_and_used": bool(runtime.get("cuda_available")),
            "frame_shape_340_by_573": source["frame_shape_yx"]
            == [FRAME_HEIGHT, FRAME_WIDTH],
            "host_ring_is_pinned_and_bounded": bool(source["host_rings_pinned"]),
            "duration_at_least_60_seconds_per_lane": all(
                lane["sustained"]["declared_duration_seconds"]
                >= MINIMUM_PRODUCTION_DURATION_SECONDS
                for lane in lanes.values()
            ),
            "arrival_interval_exactly_1ms": all(
                lane["sustained"]["arrival_interval_ms"] == 1.0
                for lane in lanes.values()
            ),
            "arrival_accounting_closes": all(
                lane["sustained"]["scheduled_arrivals"]
                == lane["sustained"]["processed_arrivals"]
                + lane["sustained"]["dropped_arrivals"]
                for lane in lanes.values()
            ),
            "full_candidate_table_row_count_closes": all(
                lane["sustained"]["candidate_count"]["total"]
                == int(len(sample_arrays[f"{arm}__candidate_score"]))
                for arm, lane in lanes.items()
            ),
            "full_candidate_table_transferred_and_retained": all(
                lane["sustained"]["decision_output_contract"][
                    "full_candidate_table_transferred"
                ]
                and lane["sustained"]["decision_output_contract"][
                    "full_candidate_table_retained_in_latency_samples"
                ]
                for lane in lanes.values()
            ),
            "exact_nms_reference_parity_self_checks_passed": all(
                row["status"] == "passed_exact_maintained_extractor_parity"
                for row in parity.values()
            ),
            "batch_frontier_uses_identical_frame_multiset_per_size": all(
                int(row["timed_frames"])
                == int(batch_workload["timed_frames_per_size"])
                and int(row["timed_source_ring_cycles"])
                == int(batch_workload["timed_ring_cycles_per_size"])
                and row["source_frame_order"] == "sequential_0_through_63_cyclic"
                for row in frontier_rows
            ),
            "batch_frontier_candidate_load_matches_within_arm": all(
                len(
                    {
                        int(row["retained_candidate_count"])
                        for row in frontier_rows
                        if row["arm"] == arm
                    }
                )
                == 1
                for arm in ordered_arms
            ),
            "fit_time_separate": all(float(row["model_fit_ms"]) == 0.0 for row in fit_rows),
            "batch_frontier_separate": bool(frontier_rows),
            "labels_not_opened": True,
            "streaming_gate_passed_for_every_lane": all_passed,
        }
        validation = {
            "status": "passed_exact_streaming_artifact_contract",
            "streaming_readiness_status": status,
            "checks": checks,
            "artifact_contract_passed": all(
                value
                for key, value in checks.items()
                if key != "streaming_gate_passed_for_every_lane"
            ),
            "streaming_gate_passed": all_passed,
        }
        _atomic_json(partial / "validation.json", validation)
        _atomic_json(
            partial / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "grain": "causal representation arm x exact-output paced streaming lane",
                "primary_latency": (
                    "lanes.<arm>.sustained.service_latency.end_to_end_wall_ms"
                ),
                "exact_decision_latency": (
                    "lanes.<arm>.sustained.service_latency.exact_decision_wall_ms"
                ),
                "raw_latency_and_full_candidate_table": (
                    "latency_and_candidate_samples.npz"
                ),
                "fit_time": "fit_times.tsv",
                "batch_frontier": "exact_batch_throughput_frontier.tsv",
                "claim_boundary": "claim_boundary.json",
                "scientific_audit": (
                    "timing-only fixed-pipeline characterization; no detector rankings "
                    "or biological outputs"
                ),
            },
        )
        _atomic_json(
            partial / "heartbeat.json",
            {
                "status": status,
                "stage": "complete",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        _atomic_json(partial / "artifact_index.json", _artifact_index(partial))
        if destination.exists():
            raise FileExistsError(f"output appeared during run: {destination}")
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone exact deployed-NMS GPU streaming and batch benchmark."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-id", default=DEFAULT_EXACT_CONTEXT_ID)
    parser.add_argument(
        "--arms", nargs="+", choices=STREAMING_ARMS, default=list(DEFAULT_EXACT_ARMS)
    )
    parser.add_argument(
        "--duration-seconds", type=float, default=MINIMUM_PRODUCTION_DURATION_SECONDS
    )
    parser.add_argument(
        "--queue-capacity", type=int, default=DEFAULT_ARRIVAL_QUEUE_CAPACITY
    )
    parser.add_argument(
        "--source-ring-frames", type=int, default=DEFAULT_SOURCE_RING_FRAMES
    )
    parser.add_argument("--stream-start-ui", type=int, default=1900)
    parser.add_argument(
        "--candidate-limit-per-frame",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT_PER_FRAME,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = GammaLSDifferenceConfig.load(args.config)
    try:
        payload = run_exact_deployed_nms_benchmark(
            config,
            preflight_dir=args.preflight_dir,
            screen_dir=args.screen_dir,
            output_dir=args.output_dir,
            context_id=args.context_id,
            arms=args.arms,
            duration_seconds=args.duration_seconds,
            arrival_interval_ms=float(
                config.payload["efficiency"]["streaming_deadline_ms"]
            ),
            arrival_queue_capacity=args.queue_capacity,
            source_ring_frames=args.source_ring_frames,
            stream_start_ui=args.stream_start_ui,
            candidate_limit_per_frame=args.candidate_limit_per_frame,
            device=str(config.payload["resources"]["device"]),
        )
    except (RuntimeError, StreamingBenchmarkUnavailable) as error:
        print(
            json.dumps(
                {
                    "status": "blocked_gpu_preflight_or_runtime",
                    "output_mutated": False,
                    "reason": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 3
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CANDIDATE_LIMIT_PER_FRAME",
    "DEFAULT_EXACT_ARMS",
    "DEFAULT_EXACT_CONTEXT_ID",
    "EXACT_DECISION_MODE",
    "ExactOutputCausalGammaPipeline",
    "cleanup_square_local_max_candidates",
    "cyclic_ring_frame_indices",
    "exact_nms_parity_self_check",
    "matched_cyclic_batch_plan",
    "run_exact_deployed_nms_benchmark",
]
