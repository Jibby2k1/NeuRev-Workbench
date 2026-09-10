"""Sustained CUDA latency benchmark for the causal Gamma-LS detector.

This module measures the operational boundary requested by the paper:

``pinned host frame -> H2D -> causal representation -> radial Gamma-LS
   -> threshold -> spatial NMS -> bounded decision packet on the host``.

The paced experiment and the batch-throughput frontier are deliberately
separate.  Batch amortization is never used as evidence for single-frame
latency.  The source is a bounded pinned-memory ring filled from a real movie;
disk I/O and camera acquisition are outside the timing boundary and are
reported as such.  No labels are opened and the calibration threshold is a
benchmark-only quiet-tail operating point, not a scientific CFAR operating
point or a probability-of-false-alarm claim.
"""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
    gamma_reference_kernel,
)

from . import gpu_representations
from .config import GammaLSDifferenceConfig
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device
from .screen import _verify_ready_preflight


STREAMING_ARMS = ("raw", "difference_signed", "difference_energy_normalized")
MINIMUM_PRODUCTION_DURATION_SECONDS = 60.0
FRAME_HEIGHT = 340
FRAME_WIDTH = 573
DEFAULT_SOURCE_RING_FRAMES = 64
DEFAULT_ARRIVAL_QUEUE_CAPACITY = 8
CALIBRATION_PRE_ROLL_FRAMES = 64
CALIBRATION_QUIET_FRAMES = 100
CALIBRATION_TAIL_QUANTILE = 0.9999
CALIBRATION_SAMPLE_STRIDE = 16
NMS_DISTANCE_PX = 6
ENERGY_EPSILON = 1e-8
_CONTEXT_PATTERN = re.compile(
    r"^(?:(?P<canonical>gamma)|support_support_(?P<support_stage>[ab]))_h"
    r"(?P<half>[1-9][0-9]*)_g(?P<guard>[0-9]+)_n"
    r"(?P<shape>[1-9][0-9]*)_m(?P<mode>[0-9]+(?:p[0-9]+)?)$"
)


class StreamingBenchmarkUnavailable(RuntimeError):
    """Raised when the required CUDA runtime or frozen evidence is unavailable."""


@dataclass(frozen=True)
class ArrivalRecord:
    """One scheduled arrival retained by the bounded queue."""

    arrival_index: int
    scheduled_ns: int


@dataclass
class BoundedArrivalQueue:
    """Deterministic drop-newest queue for an independent periodic producer."""

    start_ns: int
    interval_ns: int
    total_arrivals: int
    capacity: int

    def __post_init__(self) -> None:
        if self.start_ns < 0 or self.interval_ns < 1:
            raise ValueError("arrival start and interval must be nonnegative/positive")
        if self.total_arrivals < 1 or self.capacity < 1:
            raise ValueError("arrival count and queue capacity must be positive")
        self._next_index = 0
        self._queue: deque[ArrivalRecord] = deque()
        self.dropped = 0
        self.max_depth = 0

    @property
    def next_index(self) -> int:
        return self._next_index

    @property
    def depth(self) -> int:
        return len(self._queue)

    @property
    def exhausted(self) -> bool:
        return self._next_index >= self.total_arrivals

    @property
    def next_scheduled_ns(self) -> int | None:
        if self.exhausted:
            return None
        return self.start_ns + self._next_index * self.interval_ns

    def enqueue_due(self, now_ns: int) -> int:
        """Materialize every arrival due by ``now_ns`` without exceeding capacity."""

        if now_ns < self.start_ns or self.exhausted:
            return 0
        due_last = min(
            self.total_arrivals - 1,
            (int(now_ns) - self.start_ns) // self.interval_ns,
        )
        added = 0
        while self._next_index <= due_last:
            record = ArrivalRecord(
                arrival_index=self._next_index,
                scheduled_ns=self.start_ns + self._next_index * self.interval_ns,
            )
            self._next_index += 1
            if len(self._queue) < self.capacity:
                self._queue.append(record)
                added += 1
                self.max_depth = max(self.max_depth, len(self._queue))
            else:
                self.dropped += 1
        return added

    def pop(self) -> ArrivalRecord:
        if not self._queue:
            raise IndexError("arrival queue is empty")
        return self._queue.popleft()


def parse_gamma_context_id(context_id: str, *, scale_floor: float = 0.0) -> GammaReferenceSpec:
    """Reconstruct a guarded radial context while preserving its exact source ID.

    Both the original ``gamma_h...`` IDs and the support experiment's
    ``support_support_a_h...``/``support_support_b_h...`` IDs encode the same
    numeric stencil fields.  The namespace is retained in ``context_id`` so a
    support-screen row cannot silently alias an original-screen row.
    """

    match = _CONTEXT_PATTERN.fullmatch(str(context_id))
    if match is None:
        raise ValueError(f"unsupported Gamma-LS context id: {context_id!r}")
    half = int(match.group("half"))
    guard = int(match.group("guard"))
    shape = float(match.group("shape"))
    mode_fraction = float(match.group("mode").replace("p", "."))
    return GammaReferenceSpec.from_mode(
        str(context_id),
        support_width_px=2 * half + 1,
        shape_n=shape,
        mode_radius_px=mode_fraction * half,
        guard_radius_px=float(guard),
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=1e-6,
        scale_floor=float(scale_floor),
    )


def _context_namespace_and_stage(context_id: str) -> tuple[str, str]:
    """Return provenance namespace and required fold-row stage for a valid ID."""

    match = _CONTEXT_PATTERN.fullmatch(str(context_id))
    if match is None:
        raise ValueError(f"unsupported Gamma-LS context id: {context_id!r}")
    support_stage = match.group("support_stage")
    if support_stage is None:
        return "original_gamma_grid", "base_g2"
    return f"support_support_{support_stage}", f"support_{support_stage}"


def _percentiles_ms(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("latency samples must be a non-empty finite vector")
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50.0)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "p99_ms": float(np.percentile(array, 99.0)),
        "max_ms": float(np.max(array)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError(f"inconsistent fields in {path.name}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("rb") as stream:
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


def _process_memory() -> dict[str, int | None]:
    result: dict[str, int | None] = {"rss_bytes": None, "high_water_bytes": None}
    status = Path("/proc/self/status")
    if not status.is_file():
        return result
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            result["rss_bytes"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            result["high_water_bytes"] = int(line.split()[1]) * 1024
    return result


def _wait_until_ns(target_ns: int) -> None:
    """Sleep coarsely, then yield/spin over the last 100 microseconds."""

    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 250_000:
            time.sleep((remaining - 100_000) / 1e9)
        elif remaining > 50_000:
            time.sleep(0)


def _load_pinned_source_rings(
    movie_path: Path,
    *,
    stream_start_ui: int,
    source_ring_frames: int,
    quiet_stop_ui: int,
    device: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load bounded source/calibration windows into page-locked host tensors."""

    import torch

    started = time.perf_counter_ns()
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(movie, np.memmap) or movie.ndim != 3:
        raise ValueError("stream source must be a memory-mapped TYX .npy array")
    if tuple(map(int, movie.shape[1:])) != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            f"production benchmark requires {FRAME_HEIGHT}x{FRAME_WIDTH} frames"
        )
    if movie.dtype != np.uint16:
        raise ValueError("production benchmark source must preserve uint16 camera frames")
    stream_start_zero = int(stream_start_ui) - 1
    stream_stop_zero = stream_start_zero + int(source_ring_frames)
    if not 0 <= stream_start_zero < stream_stop_zero <= int(movie.shape[0]):
        raise ValueError("stream source ring exceeds the configured movie")
    calibration_stop_zero = int(quiet_stop_ui)
    calibration_start_zero = (
        calibration_stop_zero
        - CALIBRATION_QUIET_FRAMES
        - CALIBRATION_PRE_ROLL_FRAMES
    )
    if calibration_start_zero < 0:
        raise ValueError("movie is too short for calibration pre-roll")

    stream_numpy = np.array(
        movie[stream_start_zero:stream_stop_zero], dtype=np.uint16, order="C", copy=True
    )
    calibration_numpy = np.array(
        movie[calibration_start_zero:calibration_stop_zero],
        dtype=np.uint16,
        order="C",
        copy=True,
    )
    stream_ring = torch.empty(stream_numpy.shape, dtype=torch.uint16, pin_memory=True)
    calibration_ring = torch.empty(
        calibration_numpy.shape, dtype=torch.uint16, pin_memory=True
    )
    stream_ring.copy_(torch.from_numpy(stream_numpy))
    calibration_ring.copy_(torch.from_numpy(calibration_numpy))
    del stream_numpy, calibration_numpy
    if not stream_ring.is_pinned() or not calibration_ring.is_pinned():
        raise RuntimeError("host source rings are not page locked")
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return stream_ring, calibration_ring, {
        "source_id": str(movie_path),
        "movie_shape_tyx": [int(value) for value in movie.shape],
        "movie_dtype": str(movie.dtype),
        "frame_shape_yx": [FRAME_HEIGHT, FRAME_WIDTH],
        "stream_source_interval_ui": [stream_start_ui, stream_stop_zero],
        "source_ring_frames": int(source_ring_frames),
        "source_ring_bytes": int(stream_ring.numel() * stream_ring.element_size()),
        "calibration_interval_ui": [calibration_start_zero + 1, calibration_stop_zero],
        "calibration_pre_roll_frames": CALIBRATION_PRE_ROLL_FRAMES,
        "calibration_quiet_frames": CALIBRATION_QUIET_FRAMES,
        "calibration_ring_bytes": int(
            calibration_ring.numel() * calibration_ring.element_size()
        ),
        "host_rings_pinned": True,
        "load_and_pin_ms_outside_timed_boundary": float(elapsed_ms),
        "disk_and_camera_acquisition_in_timed_boundary": False,
        "device": str(device),
        "labels_opened": False,
    }


def _representation(common: Any, arm: str) -> Any:
    if arm == "raw":
        return common
    if arm == "difference_signed":
        return common[1:] - common[:-1]
    if arm == "difference_energy_normalized":
        difference = common[1:] - common[:-1]
        return difference / (
            torch_sqrt(common[1:].square() + common[:-1].square() + ENERGY_EPSILON)
        )
    raise ValueError(f"unsupported streaming arm {arm!r}")


def torch_sqrt(values: Any) -> Any:
    """Tiny indirection that keeps module import CPU-only for artifact tests."""

    import torch

    return torch.sqrt(values)


def _local_maximum_mask(score: Any, *, threshold_z: float, distance_px: int) -> Any:
    """Return strict-threshold square-window maxima with a frozen border exclusion."""

    import torch.nn.functional as functional

    distance = int(distance_px)
    if score.ndim != 3 or distance < 1:
        raise ValueError("score must be BYX and distance_px must be positive")
    if min(map(int, score.shape[-2:])) <= 2 * distance:
        raise ValueError("score field is too small for the NMS border exclusion")
    pooled = functional.max_pool2d(
        score.unsqueeze(1),
        kernel_size=2 * distance + 1,
        stride=1,
        padding=distance,
    ).squeeze(1)
    keep = (score > float(threshold_z)) & (score == pooled)
    keep[:, :distance] = False
    keep[:, -distance:] = False
    keep[:, :, :distance] = False
    keep[:, :, -distance:] = False
    return keep


def _calibrate_benchmark_operating_point(
    calibration_ring: Any,
    *,
    arm: str,
    reference: GammaReferenceSpec,
    device: Any,
    scale_floor_percentile: float,
) -> tuple[GammaReferenceSpec, float, dict[str, Any]]:
    """Fit a benchmark-only quiet-tail floor/threshold; never use sparse labels."""

    import torch

    if arm not in STREAMING_ARMS:
        raise ValueError(f"unsupported streaming arm {arm!r}")
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_vram = torch.cuda.mem_get_info(device)
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start_event.record()
    with torch.inference_mode():
        values = calibration_ring.to(device=device, non_blocking=True).to(torch.float32)
        common = gpu_representations.causal_preprocess_common_input(values).values
        representation = _representation(common, arm)[-CALIBRATION_QUIET_FRAMES:]
        zero_floor = replace(reference, scale_floor=0.0)
        moments = gamma_local_standardization(
            representation,
            zero_floor,
            chunk_frames=16,
            return_statistics=True,
        )
        if moments.local_std is None or moments.local_mean is None:
            raise AssertionError("calibration requested but did not receive moments")
        std_sample = moments.local_std.reshape(-1)[::CALIBRATION_SAMPLE_STRIDE]
        floor = torch.quantile(
            std_sample,
            float(scale_floor_percentile) / 100.0,
        )
        score = (representation - moments.local_mean) / (
            torch.maximum(moments.local_std, floor) + float(reference.epsilon)
        )
        score_sample = score.reshape(-1)[::CALIBRATION_SAMPLE_STRIDE]
        threshold = torch.quantile(score_sample, CALIBRATION_TAIL_QUANTILE)
    stop_event.record()
    stop_event.synchronize()
    floor_value = float(floor.item())
    threshold_value = float(threshold.item())
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    free_after, _ = torch.cuda.mem_get_info(device)
    fitted = replace(reference, scale_floor=floor_value)
    return fitted, threshold_value, {
        "model_fit_ms": 0.0,
        "model_fit_reason": "fixed causal difference and fixed Gamma kernel have no learned parameters",
        "benchmark_calibration_wall_ms": float(wall_ms),
        "benchmark_calibration_cuda_ms": float(start_event.elapsed_time(stop_event)),
        "scale_floor_percentile": float(scale_floor_percentile),
        "scale_floor": floor_value,
        "threshold_tail_quantile": CALIBRATION_TAIL_QUANTILE,
        "threshold_z": threshold_value,
        "calibration_sample_stride": CALIBRATION_SAMPLE_STRIDE,
        "calibration_sample_count": int(score_sample.numel()),
        "threshold_role": "benchmark_only_quiet_tail_not_scientific_operating_point",
        "probability_of_false_alarm_claimed": False,
        "sparse_positive_coordinates_used": False,
        "sparse_positive_identities_used": False,
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "free_vram_bytes_before": int(free_before),
        "free_vram_bytes_after": int(free_after),
        "total_vram_bytes": int(total_vram),
    }


class PreparedCausalGammaPipeline:
    """Prepared single-device operator with persistent bounded recurrent state."""

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int],
        arm: str,
        reference: GammaReferenceSpec,
        threshold_z: float,
        device: Any,
        max_batch_frames: int,
    ) -> None:
        import torch
        import torch.nn.functional as functional

        if arm not in STREAMING_ARMS:
            raise ValueError(f"unsupported streaming arm {arm!r}")
        if frame_shape != (FRAME_HEIGHT, FRAME_WIDTH):
            raise ValueError("prepared production pipeline requires the frozen frame shape")
        if max_batch_frames < 1:
            raise ValueError("max_batch_frames must be positive")
        self.torch = torch
        self.functional = functional
        self.device = torch.device(device)
        self.arm = arm
        self.reference = reference
        self.threshold_z = float(threshold_z)
        self.max_batch_frames = int(max_batch_frames)
        self.height, self.width = frame_shape
        self.device_input = torch.empty(
            (self.max_batch_frames, self.height, self.width),
            dtype=torch.uint16,
            device=self.device,
        )
        self.host_counts = torch.empty(
            (self.max_batch_frames,), dtype=torch.int64, pin_memory=True
        )
        self.host_top_indices = torch.empty(
            (self.max_batch_frames,), dtype=torch.int64, pin_memory=True
        )
        self.host_top_scores = torch.empty(
            (self.max_batch_frames,), dtype=torch.float32, pin_memory=True
        )

        radius = int(
            gpu_representations.GAUSSIAN_TRUNCATE
            * gpu_representations.SPATIAL_SIGMA_PX
            + 0.5
        )
        coordinates = torch.arange(
            -radius, radius + 1, dtype=torch.float32, device=self.device
        )
        gaussian = torch.exp(
            -0.5 * (coordinates / gpu_representations.SPATIAL_SIGMA_PX).square()
        )
        self.gaussian = gaussian / gaussian.sum()
        self.y_indices = gpu_representations._scipy_reflect_indices(
            self.height, radius, device=self.device
        )
        self.x_indices = gpu_representations._scipy_reflect_indices(
            self.width, radius, device=self.device
        )

        kernel = gamma_reference_kernel(
            reference, device=self.device, dtype=torch.float32
        )
        self.gamma_weight = kernel.reshape(1, 1, *kernel.shape).repeat(2, 1, 1, 1)
        self.gamma_half = int(reference.support_width_px) // 2
        mass_weight = kernel.reshape(1, 1, *kernel.shape)
        self.reference_mass = functional.conv2d(
            torch.ones(
                (1, 1, self.height, self.width),
                dtype=torch.float32,
                device=self.device,
            ),
            mass_weight,
            padding=self.gamma_half,
        )
        if bool((self.reference_mass <= 0).any()):
            raise RuntimeError("prepared Gamma reference has zero border mass")
        self.scale_floor = torch.as_tensor(
            float(reference.scale_floor), dtype=torch.float32, device=self.device
        )
        self.ema_state = torch.empty(
            (self.height, self.width), dtype=torch.float32, device=self.device
        )
        self.previous_common = torch.empty_like(self.ema_state)
        self.state_ready = False
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

    def reset_state(self) -> None:
        self.state_ready = False

    def _spatial(self, values: Any) -> Any:
        frames = values.unsqueeze(1)
        spatial = self.functional.conv2d(
            self.torch.index_select(frames, -2, self.y_indices),
            self.gaussian.reshape(1, 1, -1, 1),
        )
        return self.functional.conv2d(
            self.torch.index_select(spatial, -1, self.x_indices),
            self.gaussian.reshape(1, 1, 1, -1),
        ).squeeze(1)

    def _represent(self, spatial: Any) -> Any:
        outputs = []
        for index in range(int(spatial.shape[0])):
            if not self.state_ready:
                common = spatial[index]
                representation = (
                    common if self.arm == "raw" else self.torch.zeros_like(common)
                )
                self.state_ready = True
            else:
                common = (
                    gpu_representations.EMA_ALPHA * spatial[index]
                    + (1.0 - gpu_representations.EMA_ALPHA) * self.ema_state
                )
                difference = common - self.previous_common
                if self.arm == "raw":
                    representation = common
                elif self.arm == "difference_signed":
                    representation = difference
                else:
                    representation = difference / self.torch.sqrt(
                        common.square()
                        + self.previous_common.square()
                        + ENERGY_EPSILON
                    )
            self.ema_state.copy_(common)
            self.previous_common.copy_(common)
            outputs.append(representation)
        return self.torch.stack(outputs)

    def _gamma(self, representation: Any) -> Any:
        stacked = self.torch.stack(
            (representation, representation.square()), dim=1
        )
        moments = self.functional.conv2d(
            stacked,
            self.gamma_weight,
            padding=self.gamma_half,
            groups=2,
        ) / self.reference_mass
        local_mean = moments[:, 0]
        local_second = moments[:, 1]
        local_std = self.torch.sqrt(
            (local_second - local_mean.square()).clamp_min(0.0)
        )
        return (representation - local_mean) / (
            self.torch.maximum(local_std, self.scale_floor)
            + float(self.reference.epsilon)
        )

    def _decision(self, score: Any) -> tuple[Any, Any, Any]:
        keep = _local_maximum_mask(
            score, threshold_z=self.threshold_z, distance_px=NMS_DISTANCE_PX
        )
        flat_keep = keep.reshape(int(score.shape[0]), -1)
        flat_score = score.reshape(int(score.shape[0]), -1)
        counts = flat_keep.count_nonzero(dim=1)
        retained = self.torch.where(
            flat_keep,
            flat_score,
            self.torch.full_like(flat_score, -self.torch.inf),
        )
        top_scores, top_indices = retained.max(dim=1)
        top_indices = self.torch.where(
            counts > 0, top_indices, self.torch.full_like(top_indices, -1)
        )
        top_scores = self.torch.where(
            counts > 0, top_scores, self.torch.full_like(top_scores, self.torch.nan)
        )
        return counts, top_scores, top_indices

    def process_batch(
        self, host_frames: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Run a pinned batch and return a bounded usable decision per frame.

        The decision packet contains the retained-candidate count plus the
        score/x/y of the highest-scoring retained candidate.  Frames without a
        candidate use ``top_score=NaN`` and ``top_x=top_y=-1``.  Every field is
        copied to pinned host memory before the terminal synchronization.
        """

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
            counts, top_scores, top_indices = self._decision(score)
            self.host_counts[:batch].copy_(counts, non_blocking=True)
            self.host_top_scores[:batch].copy_(top_scores, non_blocking=True)
            self.host_top_indices[:batch].copy_(top_indices, non_blocking=True)
            self.events[4].record()
        self.events[4].synchronize()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        host_counts = self.host_counts[:batch].numpy().copy()
        host_top_indices = self.host_top_indices[:batch].numpy().copy()
        host_top_scores = self.host_top_scores[:batch].numpy().copy()
        valid = host_top_indices >= 0
        top_x = np.where(valid, host_top_indices % self.width, -1).astype(np.int32)
        top_y = np.where(valid, host_top_indices // self.width, -1).astype(np.int32)
        decision = {
            "count": host_counts,
            "top_score": host_top_scores,
            "top_x": top_x,
            "top_y": top_y,
        }
        return decision, {
            "h2d_ms": float(self.events[0].elapsed_time(self.events[1])),
            "representation_ms": float(self.events[1].elapsed_time(self.events[2])),
            "gamma_ls_ms": float(self.events[2].elapsed_time(self.events[3])),
            "threshold_nms_decision_d2h_ms": float(
                self.events[3].elapsed_time(self.events[4])
            ),
            "cuda_total_ms": float(self.events[0].elapsed_time(self.events[4])),
            "end_to_end_wall_ms": float(wall_ms),
        }


def _stage_summaries(samples: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    return {name: _percentiles_ms(values) for name, values in samples.items()}


def _streaming_readiness_gates(
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
    """Return the strict predeclared 1-kHz paper-readiness gates."""

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
        "h2d_and_bounded_decision_packet_d2h_included": True,
    }


def _run_sustained_lane(
    pipeline: PreparedCausalGammaPipeline,
    source_ring: Any,
    *,
    duration_seconds: float,
    arrival_interval_ms: float,
    arrival_queue_capacity: int,
    warmup_iterations: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one independently paced lane for a fixed number of scheduled arrivals."""

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
    service: dict[str, list[float]] = {
        "h2d_ms": [],
        "representation_ms": [],
        "gamma_ls_ms": [],
        "threshold_nms_decision_d2h_ms": [],
        "cuda_total_ms": [],
        "end_to_end_wall_ms": [],
    }
    response_ms: list[float] = []
    queue_wait_ms: list[float] = []
    backlog_after_dequeue: list[int] = []
    candidate_counts: list[int] = []
    top_scores: list[float] = []
    top_x: list[int] = []
    top_y: list[int] = []
    source_indices: list[int] = []
    processed_arrival_indices: list[int] = []
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
            if arrivals.depth == 0:  # defensive against unusual clock granularity
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
        response = (completed_ns - record.scheduled_ns) / 1e6
        response_ms.append(response)
        for key in service:
            service[key].append(float(timings[key]))
        candidate_counts.append(int(decision["count"][0]))
        top_scores.append(float(decision["top_score"][0]))
        top_x.append(int(decision["top_x"][0]))
        top_y.append(int(decision["top_y"][0]))
        source_indices.append(ring_index)
        processed_arrival_indices.append(record.arrival_index)
        if completed_ns > record.scheduled_ns + interval_ns:
            missed += 1

    completed_ns = time.perf_counter_ns()
    free_after, _ = pipeline.torch.cuda.mem_get_info(pipeline.device)
    ram_after = _process_memory()
    processed = len(response_ms)
    if processed + arrivals.dropped != total_arrivals:
        raise AssertionError("processed plus dropped arrivals does not close")
    service_summary = _stage_summaries(service)
    response_summary = _percentiles_ms(response_ms)
    queue_summary = _percentiles_ms(queue_wait_ms)
    paced_span_seconds = (completed_ns - start_ns) / 1e9
    failure_count = missed + arrivals.dropped
    maximum_backlog = int(max(backlog_after_dequeue, default=0))
    gates = _streaming_readiness_gates(
        duration_seconds=float(duration_seconds),
        arrival_interval_ms=float(arrival_interval_ms),
        p99_service_ms=float(
            service_summary["end_to_end_wall_ms"]["p99_ms"]
        ),
        dropped_arrivals=int(arrivals.dropped),
        missed_processed_deadlines=int(missed),
        maximum_queue_depth=int(arrivals.max_depth),
        maximum_backlog_after_dequeue=maximum_backlog,
        final_queue_depth=int(arrivals.depth),
    )
    passed = all(gates.values())
    summary = {
        "status": "passed_1khz_streaming_gate" if passed else "complete_failed_1khz_streaming_gate",
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
        "diagnostic_failure_fraction_at_or_below_1pct": bool(
            failure_count / total_arrivals <= 0.01
        ),
        "paced_wall_span_seconds_including_final_drain": float(paced_span_seconds),
        "service_latency": service_summary,
        "response_latency_from_scheduled_arrival": response_summary,
        "queue_wait": queue_summary,
        "candidate_count": {
            "total": int(sum(candidate_counts)),
            "mean_per_processed_frame": float(np.mean(candidate_counts)),
            "maximum_per_frame": int(max(candidate_counts, default=0)),
            "frames_with_decision": int(np.count_nonzero(candidate_counts)),
        },
        "decision_output_contract": {
            "fields_per_frame": ["retained_count", "top_score", "top_x", "top_y"],
            "coordinate_convention": "x_column_y_row",
            "no_candidate_sentinels": {"top_score": "NaN", "top_x": -1, "top_y": -1},
            "bounded": True,
            "full_candidate_table_transferred": False,
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
            "pinned_uint16_H2D_through_preprocess_representation_radial_Gamma_LS_"
            "threshold_square_local_max_NMS_and_bounded_decision_packet_D2H_sync"
        ),
        "nms_contract": {
            "distance_px": NMS_DISTANCE_PX,
            "operator": "strict_threshold_plus_13x13_square_local_maximum_suppression",
            "border_exclusion_px": NMS_DISTANCE_PX,
            "plateau_note": (
                "all equal plateau maxima survive; continuous Gamma scores are expected "
                "to avoid material ties, but this is not the offline greedy plateau cleanup"
            ),
        },
        "gates": gates,
        "gate_passed": passed,
    }
    arrays = {
        "processed_arrival_index": np.asarray(processed_arrival_indices, dtype=np.int64),
        "source_ring_index": np.asarray(source_indices, dtype=np.int16),
        "response_latency_ms": np.asarray(response_ms, dtype=np.float32),
        "queue_wait_ms": np.asarray(queue_wait_ms, dtype=np.float32),
        "backlog_after_dequeue_frames": np.asarray(
            backlog_after_dequeue, dtype=np.int16
        ),
        "candidate_count": np.asarray(candidate_counts, dtype=np.int32),
        "top_candidate_score": np.asarray(top_scores, dtype=np.float32),
        "top_candidate_x": np.asarray(top_x, dtype=np.int32),
        "top_candidate_y": np.asarray(top_y, dtype=np.int32),
        **{
            f"service_{name}": np.asarray(values, dtype=np.float32)
            for name, values in service.items()
        },
    }
    return summary, arrays


def _run_batch_frontier(
    *,
    source_ring: Any,
    arm: str,
    reference: GammaReferenceSpec,
    threshold_z: float,
    device: Any,
    batch_sizes: Sequence[int],
    warmup_iterations: int,
    timed_iterations: int,
) -> list[dict[str, Any]]:
    """Measure amortized throughput separately from the paced latency evidence."""

    import torch

    rows = []
    maximum = max(map(int, batch_sizes))
    if int(source_ring.shape[0]) < maximum:
        raise ValueError("source ring must cover the largest batch size")
    for batch_size in map(int, batch_sizes):
        pipeline = PreparedCausalGammaPipeline(
            frame_shape=(FRAME_HEIGHT, FRAME_WIDTH),
            arm=arm,
            reference=reference,
            threshold_z=threshold_z,
            device=device,
            max_batch_frames=batch_size,
        )
        host_batch = source_ring[:batch_size]
        for _ in range(int(warmup_iterations)):
            pipeline.process_batch(host_batch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        wall_samples = []
        cuda_samples = []
        stage_samples: dict[str, list[float]] = {
            "h2d_ms": [],
            "representation_ms": [],
            "gamma_ls_ms": [],
            "threshold_nms_decision_d2h_ms": [],
        }
        candidate_total = 0
        for _ in range(int(timed_iterations)):
            decision, timing = pipeline.process_batch(host_batch)
            wall_samples.append(float(timing["end_to_end_wall_ms"]))
            cuda_samples.append(float(timing["cuda_total_ms"]))
            for stage_name in stage_samples:
                stage_samples[stage_name].append(float(timing[stage_name]))
            candidate_total += int(np.sum(decision["count"]))
        wall = _percentiles_ms(wall_samples)
        cuda = _percentiles_ms(cuda_samples)
        rows.append(
            {
                "arm": arm,
                "context_id": reference.context_id,
                "batch_frames": batch_size,
                "warmup_iterations": int(warmup_iterations),
                "timed_iterations": int(timed_iterations),
                "timed_frames": int(batch_size * timed_iterations),
                "mean_batch_wall_ms": wall["mean_ms"],
                "p50_batch_wall_ms": wall["p50_ms"],
                "p95_batch_wall_ms": wall["p95_ms"],
                "p99_batch_wall_ms": wall["p99_ms"],
                "max_batch_wall_ms": wall["max_ms"],
                "mean_batch_cuda_ms": cuda["mean_ms"],
                "mean_h2d_ms": float(np.mean(stage_samples["h2d_ms"])),
                "mean_representation_ms": float(
                    np.mean(stage_samples["representation_ms"])
                ),
                "mean_gamma_ls_ms": float(np.mean(stage_samples["gamma_ls_ms"])),
                "mean_threshold_nms_decision_d2h_ms": float(
                    np.mean(stage_samples["threshold_nms_decision_d2h_ms"])
                ),
                "amortized_wall_ms_per_frame": wall["mean_ms"] / batch_size,
                "amortized_frames_per_second": 1000.0 * batch_size / wall["mean_ms"],
                "candidate_count": candidate_total,
                "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "interpretation": "batch_throughput_only_not_single_frame_latency",
            }
        )
        del pipeline
        torch.cuda.synchronize(device)
    return rows


def _screen_context_role(screen_dir: Path, context_id: str) -> dict[str, Any]:
    selection_path = screen_dir / "selection.json"
    summary_path = screen_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = summary.get("status")
    if status == "complete_screen_only":
        context_namespace, expected_stage = _context_namespace_and_stage(context_id)
        if expected_stage != "base_g2":
            raise RuntimeError(
                "original Gamma screen requires an original gamma_h context ID"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        finalists = [
            str(row["common_g2_finalist_context_id"])
            for row in selection.get("folds", [])
        ]
        if len(finalists) != 4 or context_id not in finalists:
            raise RuntimeError("benchmark context is not a fold-local screen finalist")
        count = finalists.count(context_id)
        return {
            "screen_kind": "original_g1_g2_context_screen",
            "screen_dir": str(screen_dir),
            "benchmark_context_id": context_id,
            "benchmark_context_namespace": context_namespace,
            "screen_summary_sha256": _sha256(summary_path),
            "screen_selection_sha256": _sha256(selection_path),
            "fold_local_finalists": finalists,
            "benchmark_context_selected_in_fold_count": count,
            "benchmark_context_role": (
                "majority_fold_local_finalist_not_deployment_context"
                if count > len(finalists) / 2
                else "fold_local_finalist_not_deployment_context"
            ),
            "deployment_context_claimed": False,
        }
    if status != "complete_support_screen_only":
        raise RuntimeError("Gamma screen is not complete under a supported schema")

    validation_path = screen_dir / "validation.json"
    contexts_path = screen_dir / "fold_contexts.json"
    index_path = screen_dir / "artifact_index.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("status")
        != "passed_support_screen_artifact_contract_scientific_audit_pending"
    ):
        raise RuntimeError("Gamma support screen validation did not pass")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = {
        str(row["path"]): str(row["sha256"])
        for row in index.get("artifacts", [])
        if isinstance(row, Mapping) and "path" in row and "sha256" in row
    }
    for relative, path in (
        ("summary.json", summary_path),
        ("validation.json", validation_path),
        ("fold_contexts.json", contexts_path),
    ):
        if indexed.get(relative) != _sha256(path):
            raise RuntimeError(f"Gamma support screen index mismatch for {relative}")

    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    if contexts.get("selection_scope") != "outer_training_fold_only":
        raise RuntimeError("Gamma support screen is not outer-fold-local")
    if contexts.get("selection_uses_positive_coordinates") is not False:
        raise RuntimeError("Gamma support screen used positive coordinates")
    if contexts.get("selection_uses_positive_identities") is not False:
        raise RuntimeError("Gamma support screen used positive identities")
    if contexts.get("burst_windows_used") is not True:
        raise RuntimeError("Gamma support screen did not disclose burst-window use")
    folds = contexts.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise RuntimeError("Gamma support screen must contain four folds")
    context_fields = (
        "original_screen_context",
        "support_candidate_context",
        "larger_support_comparator",
        "training_best_context",
        "max_support_endpoint",
    )
    parsed = parse_gamma_context_id(context_id)
    context_namespace, expected_stage = _context_namespace_and_stage(context_id)
    appearances: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        if not isinstance(fold, Mapping):
            raise RuntimeError("Gamma support screen fold must be an object")
        for field in context_fields:
            row = fold.get(field)
            if not isinstance(row, Mapping) or str(row.get("context_id")) != context_id:
                continue
            if row.get("eligible_primary") is not True:
                raise RuntimeError("requested support context is not primary-eligible")
            expected = {
                "half_width_px": int(parsed.support_width_px // 2),
                "guard_radius_px": int(parsed.guard_radius_px),
                "shape": float(parsed.shape_n),
                "mode_fraction_of_half_width": float(
                    parsed.nominal_mode_radius_px
                    / (parsed.support_width_px // 2)
                ),
                "mode_radius_px": float(parsed.nominal_mode_radius_px),
            }
            if any(
                not math.isclose(
                    float(row.get(key, float("nan"))),
                    float(value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for key, value in expected.items()
            ):
                raise RuntimeError("Gamma support context metadata disagrees with its ID")
            if (
                str(row.get("stage")) != expected_stage
                or str(row.get("support")) != "radial_disk"
                or str(row.get("padding")) != "valid_renormalized_zero"
            ):
                raise RuntimeError("Gamma support context metadata disagrees with its ID")
            if (field == "original_screen_context") != (expected_stage == "base_g2"):
                raise RuntimeError("Gamma support context role disagrees with its ID namespace")
            appearances.append({"fold": fold_index, "field": field})
    if not appearances:
        raise RuntimeError("benchmark context is absent from the validated support screen")
    return {
        "screen_kind": "extended_support_sufficiency_screen",
        "screen_dir": str(screen_dir),
        "benchmark_context_id": context_id,
        "benchmark_context_namespace": context_namespace,
        "screen_summary_sha256": _sha256(summary_path),
        "screen_validation_sha256": _sha256(validation_path),
        "screen_fold_contexts_sha256": _sha256(contexts_path),
        "screen_artifact_index_sha256": _sha256(index_path),
        "benchmark_context_appearances": appearances,
        "benchmark_context_role": "eligible_fold_local_support_screen_context_not_deployment_context",
        "deployment_context_claimed": False,
        "scientific_audit_complete": False,
    }


def _validate_run_request(
    *,
    arms: Sequence[str],
    duration_seconds: float,
    arrival_interval_ms: float,
    queue_capacity: int,
    source_ring_frames: int,
) -> tuple[str, ...]:
    ordered = tuple(str(value) for value in arms)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("arms must be non-empty and unique")
    if any(arm not in STREAMING_ARMS for arm in ordered):
        raise ValueError(f"arms must be drawn from {STREAMING_ARMS}")
    if not math.isfinite(duration_seconds) or duration_seconds < MINIMUM_PRODUCTION_DURATION_SECONDS:
        raise ValueError("production sustained duration must be at least 60 seconds per lane")
    if not math.isclose(arrival_interval_ms, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("production arrival interval must be exactly 1 ms")
    if queue_capacity < 1:
        raise ValueError("arrival queue capacity must be positive")
    if source_ring_frames < max(64, 1):
        raise ValueError("source ring must retain at least 64 real frames")
    return ordered


def run_streaming_benchmark(
    config: GammaLSDifferenceConfig,
    *,
    preflight_dir: str | Path,
    screen_dir: str | Path,
    output_dir: str | Path,
    context_id: str = "gamma_h11_g5_n9_m1",
    arms: Sequence[str] = STREAMING_ARMS,
    duration_seconds: float = MINIMUM_PRODUCTION_DURATION_SECONDS,
    arrival_interval_ms: float = 1.0,
    arrival_queue_capacity: int = DEFAULT_ARRIVAL_QUEUE_CAPACITY,
    source_ring_frames: int = DEFAULT_SOURCE_RING_FRAMES,
    stream_start_ui: int = 1900,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run sustained paced lanes and atomically commit a timing-only artifact."""

    if not isinstance(config, GammaLSDifferenceConfig):
        raise TypeError("config must be a validated GammaLSDifferenceConfig")
    ordered_arms = _validate_run_request(
        arms=arms,
        duration_seconds=float(duration_seconds),
        arrival_interval_ms=float(arrival_interval_ms),
        queue_capacity=int(arrival_queue_capacity),
        source_ring_frames=int(source_ring_frames),
    )
    reference = parse_gamma_context_id(context_id)
    screen_root = Path(screen_dir).expanduser().resolve()
    screen_role = _screen_context_role(screen_root, context_id)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"streaming benchmark output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    preflight = _verify_ready_preflight(config, preflight_dir)
    try:
        runtime = require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise StreamingBenchmarkUnavailable(str(error)) from error

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
        source["source_ring_wrap_policy"] = "cyclic_with_recurrent_state_carried_across_wrap"
        warmup = int(config.payload["efficiency"]["warmup_iterations"])
        timed_iterations = int(config.payload["efficiency"]["timed_iterations"])
        batch_sizes = tuple(map(int, config.payload["efficiency"]["frame_chunks"]))
        lanes: dict[str, Any] = {}
        sample_arrays: dict[str, np.ndarray] = {}
        frontier_rows: list[dict[str, Any]] = []
        fit_rows: list[dict[str, Any]] = []

        for lane_index, arm in enumerate(ordered_arms, start=1):
            _atomic_json(
                partial / "heartbeat.json",
                {
                    "status": "running",
                    "stage": "calibration_then_sustained_lane",
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
            pipeline = PreparedCausalGammaPipeline(
                frame_shape=(FRAME_HEIGHT, FRAME_WIDTH),
                arm=arm,
                reference=fitted_reference,
                threshold_z=threshold_z,
                device=resolved_device,
                max_batch_frames=1,
            )
            sustained, arrays = _run_sustained_lane(
                pipeline,
                source_ring,
                duration_seconds=float(duration_seconds),
                arrival_interval_ms=float(arrival_interval_ms),
                arrival_queue_capacity=int(arrival_queue_capacity),
                warmup_iterations=warmup,
            )
            del pipeline
            torch.cuda.synchronize(resolved_device)
            fit_rows.append(
                {
                    "arm": arm,
                    "context_id": context_id,
                    **fit,
                }
            )
            frontier_rows.extend(
                _run_batch_frontier(
                    source_ring=source_ring,
                    arm=arm,
                    reference=fitted_reference,
                    threshold_z=threshold_z,
                    device=resolved_device,
                    batch_sizes=batch_sizes,
                    warmup_iterations=warmup,
                    timed_iterations=timed_iterations,
                )
            )
            lanes[arm] = {
                "arm": arm,
                "representation": arm,
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

        _atomic_npz(partial / "latency_samples.npz", sample_arrays)
        _atomic_tsv(partial / "fit_times.tsv", fit_rows)
        _atomic_tsv(partial / "batch_throughput_frontier.tsv", frontier_rows)
        all_passed = all(lane["sustained"]["gate_passed"] for lane in lanes.values())
        status = (
            "passed_1khz_streaming_gate"
            if all_passed
            else "complete_failed_1khz_streaming_gate"
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
            "scientific_audit_complete": False,
            "paper_readiness_claimed": False,
            "interpretation": (
                "A passed lane supports only this device/software/operator/shape timing "
                "boundary. It does not establish detector accuracy, control-loop stability, "
                "or end-to-end voltage-imaging inverse-control readiness."
            ),
        }
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "run_type": "sustained_gpu_streaming_latency_and_separate_batch_frontier",
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "source": source,
            "screen_context_provenance": screen_role,
            "preflight": preflight,
            "lanes": lanes,
            "batch_frontier": {
                "path": "batch_throughput_frontier.tsv",
                "batch_sizes": list(batch_sizes),
                "timed_iterations_per_size": timed_iterations,
                "interpretation": "throughput_only_not_single_frame_latency",
            },
            "bounded_buffers": {
                "arrival_queue_capacity_frames": int(arrival_queue_capacity),
                "host_source_ring_frames": int(source_ring_frames),
                "device_input_buffer_frames_streaming": 1,
                "recurrent_device_state_frames": 2,
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
            "latency_samples_close_for_every_lane": all(
                lane["sustained"]["scheduled_arrivals"]
                == lane["sustained"]["processed_arrivals"]
                + lane["sustained"]["dropped_arrivals"]
                for lane in lanes.values()
            ),
            "fit_time_separate": all(float(row["model_fit_ms"]) == 0.0 for row in fit_rows),
            "batch_frontier_separate": bool(frontier_rows),
            "labels_not_opened": True,
            "streaming_gate_passed_for_every_lane": all_passed,
        }
        validation = {
            "status": "passed_streaming_artifact_contract",
            "streaming_readiness_status": status,
            "checks": checks,
            "artifact_contract_passed": all(
                value for key, value in checks.items() if key != "streaming_gate_passed_for_every_lane"
            ),
            "streaming_gate_passed": all_passed,
        }
        _atomic_json(partial / "validation.json", validation)
        _atomic_json(
            partial / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "grain": "causal representation arm x paced streaming lane",
                "primary_latency": "lanes.<arm>.sustained.service_latency.end_to_end_wall_ms",
                "deadline_evidence": "lanes.<arm>.sustained response/backlog/drop fields",
                "raw_samples": "latency_samples.npz",
                "fit_time": "fit_times.tsv",
                "batch_frontier": "batch_throughput_frontier.tsv",
                "claim_boundary": "claim_boundary.json",
                "scientific_audit": (
                    "timing-only subexperiment; no detector rankings or biological outputs; "
                    "overall paper experiment audit remains incomplete"
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


__all__ = [
    "ArrivalRecord",
    "BoundedArrivalQueue",
    "MINIMUM_PRODUCTION_DURATION_SECONDS",
    "PreparedCausalGammaPipeline",
    "STREAMING_ARMS",
    "StreamingBenchmarkUnavailable",
    "parse_gamma_context_id",
    "run_streaming_benchmark",
]
