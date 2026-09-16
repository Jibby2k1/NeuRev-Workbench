"""Reconstruct the historical quiet-calibrated carrier without test-frame fitting.

This module produces stages only. Candidate policies, labels, temporal pooling,
and scientific audits belong to the calling reconciliation experiment. The
first 100 source frames are setup data: causal inference starts after setup.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
from threadpoolctl import threadpool_limits

from neurobench.experiments.hierarchical_parzen_ica.architecture_lanes import (
    AffineICAReconstruction,
    InnovationCalibration,
    calibrate_reference_parzen_innovation,
)
from neurobench.experiments.hierarchical_parzen_ica.architecture_visuals import (
    _fit_raw_stochastic,
)
from neurobench.experiments.hierarchical_parzen_ica.signal_noise_split import (
    _quiet_standardization,
)


REPO = Path(__file__).resolve().parents[3]
HISTORICAL_CONFIG = REPO / "examples/spon_ca_burst_stage1_architecture_visuals.example.json"
SOURCE_DEPENDENCIES = (
    Path(__file__).resolve(),
    HISTORICAL_CONFIG,
    *(REPO / relative for relative in (
        "neurobench/experiments/hierarchical_parzen_ica/architecture_visuals.py",
        "neurobench/experiments/hierarchical_parzen_ica/architecture_lanes.py",
        "neurobench/experiments/hierarchical_parzen_ica/stage1.py",
        "neurobench/experiments/hierarchical_parzen_ica/safety.py",
        "neurobench/experiments/hierarchical_parzen_ica/signal_noise_split.py",
        "neurobench/algorithms/hierarchical_parzen_ica.py",
        "neurobench/algorithms/pairwise_separation.py",
    )),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    """Hash logical native-dtype C-order array bytes, not a backing NPY file."""
    digest = hashlib.sha256()
    for frame in values:
        digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _settings() -> dict[str, Any]:
    """Use the original fit configuration; resource/display settings are unused."""
    return json.loads(HISTORICAL_CONFIG.read_text(encoding="utf-8"))["stochastic"]


@dataclass(frozen=True)
class FrozenHistoricalCalibration:
    coefficients: AffineICAReconstruction
    innovation: InnovationCalibration
    center: np.ndarray
    scale: np.ndarray
    standardization: dict[str, float]
    fit: dict[str, Any]
    quiet_frame_count: int
    quiet_input_sha256: str


def _validate(values: np.ndarray, quiet_frame_count: int, frame_interval_s: float) -> None:
    if (
        values.ndim != 3
        or min(values.shape, default=0) < 1
        or values.dtype.kind not in "uif"
    ):
        raise ValueError("raw_view must be a nonempty real numeric TYX array")
    if isinstance(quiet_frame_count, bool) or int(quiet_frame_count) != quiet_frame_count:
        raise ValueError("quiet_frame_count must be an integer")
    if not 4 <= quiet_frame_count <= len(values):
        raise ValueError("quiet_frame_count must lie between 4 and the frame count")
    if not np.isfinite(frame_interval_s) or frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be finite and positive")


def iter_historical_residual(
    raw_view: np.ndarray,
    coefficients: AffineICAReconstruction,
    innovation: InnovationCalibration,
) -> Iterator[np.ndarray]:
    """Yield the legacy float32 residual, including its source-aligned first frame.

    Preserve the original arithmetic: raw observations enter as float32 while
    the recursive reference and affine correction use the legacy mixed-dtype
    operations. Converting the current frame to float64 before the reference
    update would subtly change the historical operator.
    """
    values = np.asarray(raw_view)
    if values.ndim != 3 or len(values) == 0:
        raise ValueError("raw_view must be a nonempty TYX array")
    coefficients.validated()
    base = innovation.quiet_background.astype(np.float64)
    if base.shape != values.shape[1:]:
        raise ValueError("frozen calibration and input spatial shapes differ")
    state = base.copy()
    previous = None
    for index in range(len(values)):
        current = np.asarray(values[index], dtype=np.float32)
        if not np.isfinite(current).all():
            raise ValueError(f"nonfinite raw input at view index {index}")
        if index == 0:
            residual = current - base
        else:
            state = (
                (1.0 - innovation.reference_refresh) * state
                + innovation.reference_refresh * current
            )
            correction = (
                coefficients.teacher_forced(previous, current)
                - state - innovation.correction_bias
            )
            background = state + innovation.correction_fraction * np.clip(
                correction, -innovation.correction_limit, innovation.correction_limit
            )
            residual = current - background
        result = np.asarray(residual, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError(f"nonfinite historical residual at view index {index}")
        yield result
        previous = current


def fit_historical_calibration(
    raw_quiet: np.ndarray,
    frame_interval_s: float = 0.02,
    *,
    fixture_coefficients: AffineICAReconstruction | None = None,
) -> FrozenHistoricalCalibration:
    """Fit only the explicitly supplied setup prefix, with one CPU thread.

    ``fixture_coefficients`` is solely a small-fixture numerical test hook.
    Production ``build_historical_stages`` always fits the original model.
    """
    quiet = np.array(raw_quiet, copy=True)
    _validate(quiet, len(quiet), frame_interval_s)
    if not np.isfinite(quiet).all():
        raise ValueError("quiet input must be finite")
    with threadpool_limits(limits=1):
        if fixture_coefficients is None:
            coefficients, fit = _fit_raw_stochastic(
                quiet, SimpleNamespace(stochastic=_settings())
            )
        else:
            coefficients = fixture_coefficients.validated()
            fit = {
                "kind": "fixed_fixture_coefficients_not_fitted",
                "affine_reconstruction": asdict(coefficients),
                "labels_used": False,
            }
        # The original feature utility runner converted observations to float32
        # before constructing this carrier, unlike its earlier architecture TIFFs.
        quiet_float = np.asarray(quiet, dtype=np.float32)
        innovation = calibrate_reference_parzen_innovation(
            quiet_float, len(quiet), coefficients,
            frame_period_ms=1000.0 * frame_interval_s,
            reference_half_life_seconds=10.0,
            correction_fraction=0.1,
            correction_clip_mad=4.0,
        )
        residual = np.stack(list(iter_historical_residual(quiet_float, coefficients, innovation)))
        center, scale, standardization = _quiet_standardization(residual, len(quiet), 10.0)
    return FrozenHistoricalCalibration(
        coefficients=coefficients, innovation=innovation, center=center, scale=scale,
        standardization=standardization, fit=fit, quiet_frame_count=len(quiet),
        quiet_input_sha256=_array_sha256(quiet),
    )


def _build(
    raw_view: np.ndarray,
    output_dir: str | Path,
    *,
    quiet_frame_count: int,
    source_start_ui: int,
    frame_interval_s: float,
    fixture_coefficients: AffineICAReconstruction | None = None,
    fixture: bool,
) -> dict[str, Any]:
    values = np.asarray(raw_view)
    _validate(values, quiet_frame_count, frame_interval_s)
    quiet_frame_count = int(quiet_frame_count)
    if isinstance(source_start_ui, bool) or int(source_start_ui) != source_start_ui or source_start_ui < 1:
        raise ValueError("source_start_ui must be a positive integer")
    source_start_ui = int(source_start_ui)
    output = Path(output_dir).resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"historical stage output collision: {output}")
    source_hashes = {str(path.relative_to(REPO)): _sha256(path) for path in SOURCE_DEPENDENCIES}
    frozen = fit_historical_calibration(
        values[:quiet_frame_count], frame_interval_s,
        fixture_coefficients=fixture_coefficients,
    )
    partial.mkdir(parents=True, exist_ok=False)
    raw_path = partial / "raw_residual.npy"
    carrier_path = partial / "carrier_signed.npy"
    residual_output = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=values.shape)
    carrier_output = np.lib.format.open_memmap(carrier_path, mode="w+", dtype=np.float16, shape=values.shape)
    input_digest = hashlib.sha256()
    quiet_digest = hashlib.sha256()
    with threadpool_limits(limits=1):
        for index, residual in enumerate(iter_historical_residual(values, frozen.coefficients, frozen.innovation)):
            native_bytes = np.ascontiguousarray(values[index]).tobytes()
            input_digest.update(native_bytes)
            if index < quiet_frame_count:
                quiet_digest.update(native_bytes)
            carrier = ((residual - frozen.center) / frozen.scale).astype(np.float16)
            if not np.isfinite(carrier).all():
                raise ValueError(f"historical float16 carrier overflow at view index {index}")
            residual_output[index] = residual
            carrier_output[index] = carrier
    residual_output.flush()
    carrier_output.flush()
    del residual_output, carrier_output
    if quiet_digest.hexdigest() != frozen.quiet_input_sha256:
        raise RuntimeError("quiet input changed after model fitting")
    np.savez(
        partial / "calibration.npz",
        quiet_background=frozen.innovation.quiet_background,
        correction_bias=frozen.innovation.correction_bias,
        center=frozen.center,
        scale=frozen.scale,
    )
    if any(_sha256(REPO / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("historical reconstruction source changed during execution")
    files = {
        name: {"path": filename, "sha256": _sha256(partial / filename)}
        for name, filename in (
            ("raw_residual", "raw_residual.npy"),
            ("carrier_signed", "carrier_signed.npy"),
            ("calibration", "calibration.npz"),
        )
    }
    metadata = {
        "schema_version": 1,
        "kind": "historical_carrier_technical_fixture" if fixture else "historical_carrier_reconstruction",
        "scientific_status": "stages_only_parent_scientific_audit_pending",
        "labels_used": False,
        "cpu_threads": 1,
        "numpy_version": np.__version__,
        "input": {
            "shape": list(values.shape), "dtype": str(values.dtype), "axes": "TYX",
            "logical_c_order_bytes_sha256": input_digest.hexdigest(),
            "hash_definition": "C-order bytes of the supplied view in its native dtype; excludes NPY header",
            "source_interval_ui_inclusive": [source_start_ui, source_start_ui + len(values) - 1],
            "quiet_interval_ui_inclusive": [source_start_ui, source_start_ui + quiet_frame_count - 1],
            "quiet_frame_count": quiet_frame_count,
            "quiet_logical_c_order_bytes_sha256": frozen.quiet_input_sha256,
            "frame_interval_s": frame_interval_s,
        },
        "causality": {
            "fit_and_scale_use": "supplied quiet prefix only",
            "online_eligible_from_source_ui": source_start_ui + quiet_frame_count,
            "setup_outputs_are_online_eligible": False,
            "future_test_frames_used": False,
            "candidate_policy_or_temporal_pooling_applied": False,
        },
        "settings": {
            "stochastic": _settings(),
            "reference_half_life_seconds": 10.0,
            "reference_refresh": frozen.innovation.reference_refresh,
            "correction_fraction": 0.1, "correction_clip_mad": 4.0,
            "quiet_correction_mad": frozen.innovation.quiet_correction_mad,
            "correction_limit": frozen.innovation.correction_limit,
            "quiet_scale_floor_percentile": 10.0,
            "mad_factor": 1.4826,
            "residual_dtype": "float32", "carrier_dtype": "float16",
        },
        "fit": frozen.fit,
        "standardization": frozen.standardization,
        "source_sha256": source_hashes,
        "files": files,
        "paths": {name: str(output / item["path"]) for name, item in files.items()},
    }
    (partial / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.rename(output)
    return json.loads((output / "metadata.json").read_text(encoding="utf-8"))


def build_historical_stages(
    raw_view: np.ndarray,
    output_dir: str | Path,
    frame_interval_s: float = 0.02,
) -> dict[str, Any]:
    """Reconstruct UI1800–2359, fitting only its first100 frames (UI1800–1899)."""
    if np.ndim(raw_view) != 3 or len(raw_view) != 560:
        raise ValueError("production historical view must have 560 TYX frames (UI1800–2359)")
    return _build(
        raw_view, output_dir, quiet_frame_count=100, source_start_ui=1800,
        frame_interval_s=frame_interval_s, fixture=False,
    )


def build_historical_fixture_stages(
    raw_view: np.ndarray,
    output_dir: str | Path,
    *,
    quiet_frame_count: int,
    source_start_ui: int = 1,
    frame_interval_s: float = 0.02,
    fixture_coefficients: AffineICAReconstruction | None = None,
) -> dict[str, Any]:
    """Small-array entrypoint with explicit setup length and source-frame origin."""
    return _build(
        raw_view, output_dir, quiet_frame_count=quiet_frame_count,
        source_start_ui=source_start_ui, frame_interval_s=frame_interval_s,
        fixture_coefficients=fixture_coefficients, fixture=True,
    )
