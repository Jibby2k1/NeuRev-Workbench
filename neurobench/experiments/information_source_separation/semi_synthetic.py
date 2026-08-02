"""Real-background semi-synthetic source injection without source-label claims."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .synthetic import make_spatiotemporal_fixture


@dataclass(frozen=True)
class SemiSyntheticFixture:
    fixture_id: str
    observation: np.ndarray
    native_background: np.ndarray
    injected_neural_signal: np.ndarray
    footprints: np.ndarray
    traces: np.ndarray
    metadata: dict[str, object]


def make_real_background_fixture(
    source_video: Path,
    *,
    quiet_start_ui: int,
    quiet_end_ui: int,
    crop_origin_xy: tuple[int, int],
    crop_size_px: int,
    amplitude: float,
    seed: int,
    morphology_case: str = "overlap",
) -> SemiSyntheticFixture:
    """Inject exact known sources into an untouched real quiet movie crop.

    The real crop is retained as one native-background channel because its
    biological structure, artifact, and measurement noise are not separately
    known. Only the injected neural signal is source truth.
    """
    path = Path(source_video).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if quiet_start_ui < 1 or quiet_end_ui < quiet_start_ui:
        raise ValueError("quiet UI interval is invalid")
    if crop_size_px < 8 or amplitude <= 0:
        raise ValueError("crop size and amplitude must be positive")
    movie = np.load(path, mmap_mode="r", allow_pickle=False)
    if movie.ndim != 3:
        raise ValueError("source movie must be frame-by-row-by-column")
    start = int(quiet_start_ui) - 1
    stop = int(quiet_end_ui)
    x, y = map(int, crop_origin_xy)
    if stop > movie.shape[0] or x < 0 or y < 0 or x + crop_size_px > movie.shape[2] or y + crop_size_px > movie.shape[1]:
        raise ValueError("quiet interval or crop lies outside source movie")
    native = np.asarray(
        movie[start:stop, y:y + crop_size_px, x:x + crop_size_px],
        dtype=np.float32,
    )
    generated = make_spatiotemporal_fixture(
        morphology_case,
        seed=int(seed),
        frame_count=len(native),
        shape=(crop_size_px, crop_size_px),
        snr=8.0,
    )
    differences = np.diff(native.astype(np.float64), axis=0)
    center = np.median(differences, axis=0, keepdims=True)
    temporal_noise = 1.4826 * np.median(np.abs(differences - center))
    scale = max(float(temporal_noise), 1.0) * float(amplitude)
    unit = generated.neural_signal.astype(np.float64)
    unit /= max(float(unit.max()), np.finfo(float).eps)
    injected = (scale * unit).astype(np.float32)
    observation = native + injected
    closure = observation.astype(np.float64) - native.astype(np.float64) - injected.astype(np.float64)
    return SemiSyntheticFixture(
        fixture_id=(
            f"real_quiet_ui{quiet_start_ui}-{quiet_end_ui}_x{x}_y{y}_"
            f"a{float(amplitude):g}_seed{int(seed)}"
        ),
        observation=observation,
        native_background=native,
        injected_neural_signal=injected,
        footprints=generated.footprints,
        traces=(scale * generated.traces / max(float(generated.neural_signal.max()), np.finfo(float).eps)).astype(np.float32),
        metadata={
            "source_video": str(path),
            "source_frames_ui_inclusive": [int(quiet_start_ui), int(quiet_end_ui)],
            "crop_origin_xy": [x, y],
            "crop_size_px": int(crop_size_px),
            "amplitude_multiplier": float(amplitude),
            "native_difference_mad": float(temporal_noise),
            "injected_peak_raw_units": float(injected.max()),
            "native_background_is_not_decomposed_truth": True,
            "maximum_closure_absolute": float(np.max(np.abs(closure))),
        },
    )
