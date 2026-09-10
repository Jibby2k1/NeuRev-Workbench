"""Bounded, label-free CUDA smoke test for temporal representations + Gamma-LS.

This module is intentionally smaller than the protected experiment runner.  It
loads one short window from the real ``.npy`` movie with memory mapping, copies
that window to CUDA once, and keeps every representation on CUDA through the
guarded radial-Gamma local-standardization operator.  It does not read labels,
fit ICA, calibrate CFAR, emit candidates, or establish scientific performance.

The destination is committed with one directory rename.  CUDA is checked
before a partial directory is created, so an unavailable GPU leaves the caller's
requested output path untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping
import uuid

import numpy as np

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
)

from . import gpu_representations
from .cuda_runtime import CudaRuntimeUnavailable, require_cuda_device


SMOKE_ARMS = (
    "raw",
    "difference_signed",
    "difference_energy_normalized",
)
MAX_SMOKE_FRAMES = 64
PREVIEW_FRAMES = 2
PREVIEW_SIZE_PX = 64
ENERGY_EPSILON = 1e-8


class CudaSmokeUnavailable(RuntimeError):
    """Raised when the required CUDA runtime is not usable."""


@dataclass(frozen=True)
class SmokeExecution:
    """Host-side payload returned only after the CUDA pipeline finishes."""

    runtime: Mapping[str, Any]
    timings_ms: Mapping[str, Any]
    arm_diagnostics: Mapping[str, Mapping[str, Any]]
    arrays: Mapping[str, np.ndarray]
    peak_memory: Mapping[str, Any]


def default_smoke_reference() -> GammaReferenceSpec:
    """Return the single guarded radial context frozen for this smoke test."""

    half_width = 11
    return GammaReferenceSpec.from_mode(
        "gamma_h11_g3_n5_m0p75",
        support_width_px=2 * half_width + 1,
        shape_n=5.0,
        mode_radius_px=0.75 * half_width,
        guard_radius_px=3.0,
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=1e-6,
        scale_floor=0.0,
    )


def _validate_reference(spec: GammaReferenceSpec) -> None:
    if spec.support_geometry != "disk":
        raise ValueError("the CUDA smoke reference must use radial disk support")
    if float(spec.guard_radius_px) <= 0.0:
        raise ValueError("the CUDA smoke reference must have an explicit guard band")
    if spec.boundary_mode != "valid_renormalized_zero":
        raise ValueError("the CUDA smoke must use valid-weight border renormalization")


def _require_cuda(device: str) -> dict[str, Any]:
    """Resolve one CUDA device or fail without touching the output directory."""
    try:
        return require_cuda_device(device)
    except CudaRuntimeUnavailable as error:
        raise CudaSmokeUnavailable(str(error)) from error


def _load_memmap_window(
    movie_path: Path,
    *,
    start_frame_ui: int,
    frame_count: int,
    support_width_px: int,
    source_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Copy only a bounded, one-based frame interval from a TYX ``.npy`` memmap."""

    if not movie_path.is_file():
        raise FileNotFoundError(movie_path)
    if movie_path.suffix.lower() != ".npy":
        raise ValueError("the smoke source must be a memory-mappable .npy file")
    if not isinstance(start_frame_ui, int) or start_frame_ui < 1:
        raise ValueError("start_frame_ui must be a positive one-based integer")
    if not isinstance(frame_count, int) or not 2 <= frame_count <= MAX_SMOKE_FRAMES:
        raise ValueError(f"frame_count must be in [2,{MAX_SMOKE_FRAMES}]")

    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(movie, np.memmap):
        raise ValueError("np.load did not return a memory map for the movie")
    if movie.ndim != 3 or min(movie.shape) < 1:
        raise ValueError("the smoke source must have non-empty TYX dimensions")
    if not np.issubdtype(movie.dtype, np.number):
        raise ValueError("the smoke source must have a numeric dtype")
    if min(int(movie.shape[1]), int(movie.shape[2])) < int(support_width_px):
        raise ValueError("the movie is smaller than the Gamma reference support")

    start_zero = start_frame_ui - 1
    stop_zero = start_zero + frame_count
    if stop_zero > int(movie.shape[0]):
        raise ValueError(
            f"requested frames [{start_frame_ui},{stop_zero}] exceed {movie.shape[0]}"
        )
    # np.array(copy=True) materializes only the declared bounded slice.  The
    # entire source stays memory mapped and is never converted in one step.
    window = np.array(
        movie[start_zero:stop_zero], dtype=np.float32, order="C", copy=True
    )
    if not np.isfinite(window).all():
        raise ValueError("the selected source window contains non-finite values")
    digest = hashlib.sha256(window.tobytes(order="C")).hexdigest()
    if source_id is not None and not source_id.startswith(("data://", "repo://")):
        raise ValueError("source_id must be a portable data:// or repo:// identifier")
    source = {
        "load_mode": "numpy_memmap_then_bounded_copy",
        "movie_shape_tyx": [int(value) for value in movie.shape],
        "movie_dtype": str(movie.dtype),
        "window_start_frame_ui": int(start_frame_ui),
        "window_stop_frame_ui_inclusive": int(stop_zero),
        "window_start_index_zero_based": int(start_zero),
        "window_stop_index_zero_based_exclusive": int(stop_zero),
        "window_shape_tyx": [int(value) for value in window.shape],
        "window_float32_sha256": digest,
        "labels_read": False,
    }
    if source_id is None:
        source["path"] = str(movie_path)
        source["source_id"] = None
    else:
        source["source_id"] = source_id
        source["path_recorded"] = False
    return window, source


def _common_preprocess_on_device(
    values: Any,
    *,
    spatial_sigma_px: float,
    ema_alpha: float,
    gaussian_truncate: float,
) -> Any:
    """Apply the frozen common CUDA preprocessor without leaving device."""

    expected = (
        gpu_representations.SPATIAL_SIGMA_PX,
        gpu_representations.EMA_ALPHA,
        gpu_representations.GAUSSIAN_TRUNCATE,
    )
    requested = (float(spatial_sigma_px), float(ema_alpha), float(gaussian_truncate))
    if requested != expected:
        raise ValueError(
            "the smoke runner only accepts the frozen sigma=1, alpha=0.4, "
            "truncate=4 common preprocessor"
        )
    return gpu_representations.causal_preprocess_common_input(values).values


def _representation_on_device(common: Any, arm: str, *, epsilon: float) -> Any:
    """Build one commonly aligned representation as a tensor on the same device."""

    if arm not in SMOKE_ARMS:
        raise ValueError(f"unknown smoke arm {arm!r}")
    if arm == "raw":
        result = gpu_representations.raw_representation(common).values
        # Align raw to t=1..T-1 so all three arms have the same source frames.
        return result[1:]
    if arm == "difference_signed":
        return gpu_representations.signed_difference_representation(common).values
    return gpu_representations.energy_normalized_difference_representation(
        common, epsilon=epsilon
    ).values


def _cuda_elapsed(torch: Any, operation: Callable[[], Any]) -> tuple[Any, float]:
    """Time one operation on the current CUDA stream with explicit synchronization."""

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    stop.record()
    stop.synchronize()
    return result, float(start.elapsed_time(stop))


def _preview_on_device(values: Any) -> Any:
    height = int(values.shape[-2])
    width = int(values.shape[-1])
    crop_height = min(height, PREVIEW_SIZE_PX)
    crop_width = min(width, PREVIEW_SIZE_PX)
    y0 = (height - crop_height) // 2
    x0 = (width - crop_width) // 2
    count = min(int(values.shape[0]), PREVIEW_FRAMES)
    return values[:count, y0 : y0 + crop_height, x0 : x0 + crop_width].contiguous()


def _execute_cuda(
    window: np.ndarray,
    *,
    runtime: Mapping[str, Any],
    spec: GammaReferenceSpec,
    spatial_sigma_px: float,
    ema_alpha: float,
    gaussian_truncate: float,
    energy_epsilon: float,
) -> SmokeExecution:
    import torch

    device = torch.device(str(runtime["resolved_device"]))
    torch.cuda.reset_peak_memory_stats(device)
    host_tensor = torch.from_numpy(window)
    device_input, h2d_ms = _cuda_elapsed(
        torch,
        lambda: host_tensor.to(device=device, dtype=torch.float32, non_blocking=False),
    )
    if not device_input.is_cuda:
        raise RuntimeError("H2D stage did not produce a CUDA tensor")

    common, preprocessing_ms = _cuda_elapsed(
        torch,
        lambda: _common_preprocess_on_device(
            device_input,
            spatial_sigma_px=spatial_sigma_px,
            ema_alpha=ema_alpha,
            gaussian_truncate=gaussian_truncate,
        ),
    )
    if not common.is_cuda:
        raise RuntimeError("common preprocessing left CUDA")

    arm_diagnostics: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    timings: dict[str, Any] = {
        "h2d_ms": h2d_ms,
        "common_preprocessing_ms": preprocessing_ms,
        "arms": {},
        "synchronization": "CUDA events; stop event synchronized for every stage",
        "scope": "single cold bounded smoke; not a throughput or 1-kHz claim",
    }
    kernel_saved = False
    for arm in SMOKE_ARMS:
        representation, representation_ms = _cuda_elapsed(
            torch,
            lambda selected=arm: _representation_on_device(
                common, selected, epsilon=energy_epsilon
            ),
        )
        if not representation.is_cuda:
            raise RuntimeError(f"representation {arm} left CUDA")

        gamma_result, gamma_ms = _cuda_elapsed(
            torch,
            lambda current=representation: gamma_local_standardization(
                current,
                spec,
                chunk_frames=int(current.shape[0]),
                return_statistics=False,
            ),
        )
        if not gamma_result.values.is_cuda or not gamma_result.reference_kernel.is_cuda:
            raise RuntimeError(f"Gamma-LS result for {arm} left CUDA")

        representation_preview = _preview_on_device(representation)
        score_preview = _preview_on_device(gamma_result.values)
        device_previews = torch.stack(
            (representation_preview, score_preview), dim=0
        )
        host_previews, d2h_ms = _cuda_elapsed(
            torch,
            lambda payload=device_previews: payload.to(
                device="cpu", dtype=torch.float32, non_blocking=False
            ),
        )
        previews = host_previews.numpy()
        arrays[f"{arm}_representation_preview"] = previews[0].copy()
        arrays[f"{arm}_gamma_ls_preview"] = previews[1].copy()
        if not kernel_saved:
            kernel_host, kernel_d2h_ms = _cuda_elapsed(
                torch,
                lambda: gamma_result.reference_kernel.to(
                    device="cpu", dtype=torch.float32, non_blocking=False
                ),
            )
            arrays["gamma_reference_kernel"] = kernel_host.numpy().copy()
            kernel_saved = True
        else:
            kernel_d2h_ms = 0.0

        finite = bool(np.isfinite(previews).all())
        arm_diagnostics[arm] = {
            "source_frame_alignment": (
                "all arms correspond to common-input rows 1..T-1 (current frame t)"
            ),
            "representation_shape_tyx": [int(value) for value in representation.shape],
            "representation_device_before_gamma_ls": str(representation.device),
            "gamma_ls_device_before_d2h": str(gamma_result.values.device),
            "preview_shape_pair_tyx": [int(value) for value in previews.shape],
            "preview_finite": finite,
            "gamma_ls": dict(gamma_result.diagnostics),
        }
        if not finite:
            raise RuntimeError(f"non-finite preview generated for {arm}")
        timings["arms"][arm] = {
            "representation_ms": representation_ms,
            "gamma_ls_ms": gamma_ms,
            "diagnostic_d2h_ms": d2h_ms,
            "reference_kernel_d2h_ms": kernel_d2h_ms,
        }

    torch.cuda.synchronize(device)
    peak = {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    return SmokeExecution(
        runtime=dict(runtime),
        timings_ms=timings,
        arm_diagnostics=arm_diagnostics,
        arrays=arrays,
        peak_memory=peak,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values)
    if array.nbytes > 2 * 1024 * 1024:
        raise ValueError(f"diagnostic array exceeds the 2 MiB smoke bound: {path.name}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"diagnostic array is non-finite: {path.name}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _portable_artifact_index(root: Path) -> dict[str, Any]:
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(".partial")
    )
    paths.append("artifact_index.json")
    paths.sort()
    return {"schema_version": 1, "artifacts": paths}


def run_cuda_smoke(
    movie_path: str | Path,
    *,
    output_dir: str | Path,
    source_id: str | None = None,
    start_frame_ui: int = 1800,
    frame_count: int = 32,
    device: str = "cuda:0",
    spatial_sigma_px: float = 1.0,
    ema_alpha: float = 0.4,
    gaussian_truncate: float = 4.0,
    energy_epsilon: float = ENERGY_EPSILON,
    reference: GammaReferenceSpec | None = None,
) -> dict[str, Any]:
    """Run and atomically commit a bounded, unlabeled CUDA smoke artifact.

    The requested ``output_dir`` and its parent must not be created by the
    caller concurrently.  On any failure, ``output_dir`` remains absent.
    """

    source_path = Path(movie_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"smoke output already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"smoke output parent must already exist: {destination.parent}"
        )
    spec = default_smoke_reference() if reference is None else reference
    _validate_reference(spec)

    # These checks are deliberately before partial-directory construction.
    runtime = _require_cuda(device)
    window, source = _load_memmap_window(
        source_path,
        start_frame_ui=start_frame_ui,
        frame_count=frame_count,
        support_width_px=int(spec.support_width_px),
        source_id=source_id,
    )
    execution = _execute_cuda(
        window,
        runtime=runtime,
        spec=spec,
        spatial_sigma_px=spatial_sigma_px,
        ema_alpha=ema_alpha,
        gaussian_truncate=gaussian_truncate,
        energy_epsilon=energy_epsilon,
    )

    partial = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if partial.exists():  # practically unreachable; preserves collision contract
        raise FileExistsError(f"partial smoke output collision: {partial}")
    try:
        partial.mkdir()
        arrays_dir = partial / "arrays"
        diagnostics_dir = partial / "diagnostics"
        arrays_dir.mkdir()
        diagnostics_dir.mkdir()
        for name, values in execution.arrays.items():
            _atomic_npy(arrays_dir / f"{name}.npy", values)
        for arm, diagnostics in execution.arm_diagnostics.items():
            _atomic_json(diagnostics_dir / f"{arm}.json", diagnostics)

        completed_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "schema_version": 1,
            "experiment_id": "spon_ca_burst_gamma_ls_difference_ablation_v1",
            "run_type": "unlabeled_cuda_pipeline_smoke",
            "status": "complete_runtime_smoke_only",
            "completed_at_utc": completed_at,
            "source": source,
            "runtime": dict(execution.runtime),
            "preprocessing": {
                "stage_order": [
                    "bounded H2D",
                    "Gaussian spatial smoothing",
                    "causal EMA",
                    "temporal representation",
                    "guarded radial Gamma-LS",
                    "bounded diagnostic D2H",
                ],
                "spatial_sigma_px": float(spatial_sigma_px),
                "gaussian_truncate": float(gaussian_truncate),
                "spatial_boundary": "scipy_ndimage_reflect_equivalent",
                "ema_alpha": float(ema_alpha),
                "energy_normalization_epsilon": float(energy_epsilon),
            },
            "arms": list(SMOKE_ARMS),
            "common_aligned_output_frames": int(frame_count - 1),
            "gamma_reference": {
                "context_id": spec.context_id,
                "support_width_px": int(spec.support_width_px),
                "guard_radius_px": float(spec.guard_radius_px),
                "shape_n": float(spec.shape_n),
                "rate_mu": float(spec.rate_mu),
                "nominal_mode_radius_px": spec.nominal_mode_radius_px,
                "support_geometry": spec.support_geometry,
                "boundary_mode": spec.boundary_mode,
            },
            "timings_ms": dict(execution.timings_ms),
            "peak_gpu_memory": dict(execution.peak_memory),
            "scientific_scope": {
                "labels_read": False,
                "ica_fit_performed": False,
                "cfar_threshold_calibrated": False,
                "candidates_generated": False,
                "performance_metrics_computed": False,
                "scientific_audit_complete": False,
                "interpretation": (
                    "implementation/runtime smoke only; cannot support a detection, "
                    "accuracy, multiscale, or real-time claim"
                ),
            },
        }
        _atomic_json(partial / "summary.json", summary)
        _atomic_json(
            partial / "runtime.json",
            {
                "runtime": dict(execution.runtime),
                "timings_ms": dict(execution.timings_ms),
                "peak_gpu_memory": dict(execution.peak_memory),
            },
        )
        _atomic_json(
            partial / "validation.json",
            {
                "status": "passed_runtime_smoke_only",
                "checks": {
                    "source_was_memory_mapped": True,
                    "source_window_bounded": int(frame_count) <= MAX_SMOKE_FRAMES,
                    "cuda_required_and_used": True,
                    "representations_remained_on_cuda_through_gamma_ls": True,
                    "three_frozen_smoke_arms_present": list(execution.arm_diagnostics)
                    == list(SMOKE_ARMS),
                    "guarded_radial_gamma_reference": (
                        spec.support_geometry == "disk"
                        and float(spec.guard_radius_px) > 0.0
                    ),
                    "labels_read": False,
                    "all_previews_finite": all(
                        bool(row["preview_finite"])
                        for row in execution.arm_diagnostics.values()
                    ),
                },
            },
        )
        _atomic_json(
            partial / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "run_type": "unlabeled CUDA implementation smoke",
                "stage_sequence": (
                    "bounded movie window -> common causal preprocessing -> "
                    "raw or temporal difference -> guarded radial Gamma-LS"
                ),
                "frame_contract": "UI one-based inclusive; NumPy zero-based half-open",
                "coordinate_contract": "x=column, y=row",
                "primary_artifacts": [
                    "summary.json",
                    "runtime.json",
                    "validation.json",
                    "diagnostics/*.json",
                    "arrays/*_preview.npy",
                ],
                "labels": "not loaded",
                "scientific_audit": "incomplete by design; runtime smoke only",
                "claim_boundary": (
                    "No scientific, CFAR, multiscale, detection-efficiency, or "
                    "1-kHz conclusion follows from this cold smoke."
                ),
            },
        )
        # Write the index last so it inventories every already-committed file.
        _atomic_json(partial / "artifact_index.json", _portable_artifact_index(partial))
        if destination.exists():
            raise FileExistsError(
                f"smoke output appeared while the CUDA smoke was running: {destination}"
            )
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded label-free CUDA temporal-difference + Gamma-LS smoke."
    )
    parser.add_argument("--movie", required=True, help="Memory-mappable TYX .npy movie")
    parser.add_argument(
        "--source-id",
        help="Portable data:// or repo:// identifier recorded instead of the host path",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frame-ui", type=int, default=1800)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = run_cuda_smoke(
            args.movie,
            output_dir=args.output_dir,
            source_id=args.source_id,
            start_frame_ui=args.start_frame_ui,
            frame_count=args.frame_count,
            device=args.device,
        )
    except CudaSmokeUnavailable as error:
        # A machine-readable blocked state is emitted to stdout; the requested
        # output directory is intentionally not created.
        print(
            json.dumps(
                {
                    "status": "blocked_cuda_unavailable",
                    "output_mutated": False,
                    "reason": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 3
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CudaSmokeUnavailable",
    "MAX_SMOKE_FRAMES",
    "SMOKE_ARMS",
    "SmokeExecution",
    "default_smoke_reference",
    "run_cuda_smoke",
]
