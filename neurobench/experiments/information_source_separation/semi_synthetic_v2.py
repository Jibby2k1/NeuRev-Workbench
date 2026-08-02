"""Temporally rescaled real-background injection fixture, version 2."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .semi_synthetic import SemiSyntheticFixture
from .synthetic import make_spatiotemporal_fixture


def make_real_background_fixture_v2(
    source_video: Path, *, quiet_start_ui: int, quiet_end_ui: int,
    crop_origin_xy: tuple[int, int], crop_size_px: int, amplitude: float,
    seed: int, morphology_case: str = "overlap",
) -> SemiSyntheticFixture:
    """Inject sources after rescaling the 256-frame truth to the real crop length."""
    path = Path(source_video).resolve()
    movie = np.load(path, mmap_mode="r", allow_pickle=False)
    start, stop = int(quiet_start_ui)-1, int(quiet_end_ui)
    x, y = map(int, crop_origin_xy)
    if (quiet_start_ui < 1 or stop <= start or crop_size_px < 8 or amplitude <= 0
            or stop > movie.shape[0] or x < 0 or y < 0
            or x+crop_size_px > movie.shape[2] or y+crop_size_px > movie.shape[1]):
        raise ValueError("invalid real-background injection geometry")
    native = np.asarray(movie[start:stop, y:y+crop_size_px, x:x+crop_size_px], dtype=np.float32)
    generated = make_spatiotemporal_fixture(
        morphology_case, seed=int(seed), frame_count=256,
        shape=(crop_size_px, crop_size_px), snr=8.0)
    old_axis = np.linspace(0, 1, generated.traces.shape[1])
    new_axis = np.linspace(0, 1, len(native))
    traces = np.vstack([np.interp(new_axis, old_axis, trace) for trace in generated.traces])
    neural_unit = np.einsum("st,shw->thw", traces, generated.footprints)
    differences = np.diff(native.astype(np.float64), axis=0)
    center = np.median(differences, axis=0, keepdims=True)
    temporal_noise = 1.4826*np.median(np.abs(differences-center))
    scale = max(float(temporal_noise), 1.0)*float(amplitude)
    normalization = max(float(neural_unit.max()), np.finfo(float).eps)
    injected = (scale*neural_unit/normalization).astype(np.float32)
    scaled_traces = (scale*traces/normalization).astype(np.float32)
    observation = native+injected
    closure = observation.astype(np.float64)-native.astype(np.float64)-injected.astype(np.float64)
    return SemiSyntheticFixture(
        fixture_id=f"v2_ui{quiet_start_ui}-{quiet_end_ui}_x{x}_y{y}_a{amplitude:g}_seed{seed}_{morphology_case}",
        observation=observation, native_background=native,
        injected_neural_signal=injected,
        footprints=generated.footprints, traces=scaled_traces,
        metadata={"fixture_contract_version": 2, "source_video": str(path),
                  "source_frames_ui_inclusive": [int(quiet_start_ui), int(quiet_end_ui)],
                  "crop_origin_xy": [x,y], "crop_size_px": int(crop_size_px),
                  "amplitude_multiplier": float(amplitude),
                  "morphology_case": morphology_case,
                  "native_difference_mad": float(temporal_noise),
                  "injected_peak_raw_units": float(injected.max()),
                  "all_truth_sources_nonzero": bool(np.all(np.max(scaled_traces, axis=1) > 0)),
                  "maximum_closure_absolute": float(np.max(np.abs(closure))),
                  "native_background_is_not_decomposed_truth": True},
    )
