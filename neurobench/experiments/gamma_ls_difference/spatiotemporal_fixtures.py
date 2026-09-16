"""Frozen synthetic scenes for the small spatial/causal-temporal CFAR study.

These are fully specified technical simulations, not measured calcium kinetics
or a model of a particular indicator. The two seeds replicate each *template*;
the templates span selected scales but are not an orthogonal biological design.
All source truth starts after setup and common warmup. Raw movies are generated
without access to an operator, threshold, candidate stream, or model identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


FRAME_INTERVAL_S = 0.02
PIXEL_SIZE_UM = 0.5
DEFAULT_SEEDS = (20260914, 20260915)
TEMPLATE_IDS = (
    "isolated_compact_fast", "isolated_broad_slow", "isolated_intermediate",
    "sustained", "repeated", "crowded", "nuisance_variance_correlation",
    "nuisance_shared_brightness_motion",
)


@dataclass(frozen=True)
class SceneSpec:
    template_id: str
    seed: int

    @property
    def scene_id(self) -> str:
        return f"{self.template_id}__seed{self.seed}"


@dataclass
class SyntheticScene:
    raw: np.ndarray
    event_rows: list[dict[str, Any]]
    active_rows: list[dict[str, Any]]
    metadata: dict[str, Any]

    @property
    def expert_rows(self) -> list[dict[str, Any]]:
        """Audit-schema alias; these rows are synthetic truth, not human labels."""
        return self.event_rows


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    number = float(value)
    if not math.isfinite(number) or number != math.floor(number) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def scene_specs(seeds: Sequence[int] = DEFAULT_SEEDS) -> tuple[SceneSpec, ...]:
    """Eight fixed templates, each with the same two default random seeds."""
    values = tuple(_integer(seed, "seed") for seed in seeds)
    if not values or len(set(values)) != len(values):
        raise ValueError("seeds must be nonempty and distinct")
    return tuple(SceneSpec(template, seed) for template in TEMPLATE_IDS for seed in values)


def temporal_profile(
    rise_ms: float, decay_ms: float, *, hold_frames: int = 0,
    cutoff_fraction: float = 0.01, frame_interval_s: float = FRAME_INTERVAL_S,
) -> np.ndarray:
    """Positive finite-support profile; samples outside its returned span are zero.

    The sampled exponential rise ends at ceil(3*rise/dt), normalized to peak one.
    Then come ``hold_frames`` additional peak samples and an exponential decay.
    Decay samples below 1% are omitted. ``rise_ms=10`` is a sub-frame stress
    parameter, not a claim that 50 Hz sampling resolves a 10 ms biological rise.
    """
    rise, decay, dt = float(rise_ms) / 1000, float(decay_ms) / 1000, float(frame_interval_s)
    cutoff = float(cutoff_fraction)
    if not all(math.isfinite(x) and x > 0 for x in (rise, decay, dt)) or not 0 < cutoff < 1:
        raise ValueError("rise, decay, dt must be positive and cutoff must lie in (0,1)")
    hold = _integer(hold_frames, "hold_frames")
    n_rise = max(1, int(math.ceil(3 * rise / dt)))
    ascent = -np.expm1(-np.arange(1, n_rise + 1, dtype=np.float64) * dt / rise)
    ascent /= ascent[-1]
    n_decay = int(math.floor(-math.log(cutoff) * decay / dt))
    descent = np.exp(-np.arange(1, n_decay + 1, dtype=np.float64) * dt / decay)
    return np.concatenate((ascent, np.ones(hold), descent)).astype(np.float32)


def _event_definitions(template: str, shape: tuple[int, int]) -> list[dict[str, Any]]:
    y, x = ((size - 1) / 2 for size in shape)
    # Integer centers make generated peak pixels explicit. Matching still accepts
    # floating-point centers and the metadata retains x=column, y=row.
    x, y = float(math.floor(x + 0.5)), float(math.floor(y + 0.5))
    base = dict(x_px=x, y_px=y, peak_amplitude=24.0, onset_application_index=20,
                hold_frames=0, canonical_roi_id="synthetic_roi_01", burst_id=1)
    settings = {
        "isolated_compact_fast": (1.0, 10.0, 100.0),
        "isolated_broad_slow": (4.0, 100.0, 1000.0),
        "isolated_intermediate": (2.0, 40.0, 300.0),
        "sustained": (4.0, 40.0, 300.0),
        "repeated": (1.0, 10.0, 100.0),
        "crowded": (2.0, 100.0, 1000.0),
    }
    if template not in settings:
        return []
    sigma, rise, decay = settings[template]
    first = {**base, "sigma_px": sigma, "rise_ms": rise, "decay_ms": decay}
    if template == "sustained":
        first["hold_frames"] = 60
    if template == "repeated":
        return [first, {**first, "onset_application_index": 48, "burst_id": 2}]
    if template == "crowded":
        # Identical temporal extents give the two sources a shared audit burst.
        # Their 8px separation is fixed across kernels and exceeds the 6px NMS
        # radius; footprint overlap and unequal amplitude remain explicit stress.
        first["x_px"] = x - 4
        return [first, {**first, "x_px": x + 4, "sigma_px": 1.0,
                        "peak_amplitude": 18.0, "canonical_roi_id": "synthetic_roi_02"}]
    return [first]


def generate_scene(
    spec: SceneSpec, *, setup_frames: int = 100, warmup_frames: int = 64,
    application_frames: int = 300, shape: tuple[int, int] = (128, 128),
    spatial_halo_px: int = 0,
) -> SyntheticScene:
    """Generate one deterministic movie and exact application-only active truth.

    UI frame 1 is raw[0]. ``warmup_frames`` precedes setup and is excluded from
    calibration and application metrics, so setup has the common full history.
    Increasing warmup shifts the event times; the scene's
    random source, physical parameters and generation rules remain unchanged.
    ``shape`` denotes the evaluated image; raw includes ``spatial_halo_px`` on
    every side. Truth coordinates refer to this full raw image. The caller crops
    scores only after full-image filtering, and translates truth by the recorded
    evaluation origin. Generate once and give identical bytes to every spec.
    """
    if not isinstance(spec, SceneSpec) or spec.template_id not in TEMPLATE_IDS:
        raise ValueError("spec must identify a declared synthetic template")
    seed = _integer(spec.seed, "seed")
    setup = _integer(setup_frames, "setup_frames", 1)
    warmup = _integer(warmup_frames, "warmup_frames")
    application = _integer(application_frames, "application_frames", 1)
    if len(shape) != 2:
        raise ValueError("shape must contain Y and X")
    evaluation_shape = tuple(_integer(n, "spatial dimension", 40) for n in shape)
    halo = _integer(spatial_halo_px, "spatial_halo_px")
    spatial = tuple(n + 2 * halo for n in evaluation_shape)
    start = setup + warmup
    total = start + application
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(spatial, dtype=np.float32)
    background = 100 + 0.03 * (xx - spatial[1] / 2) + 0.02 * (yy - spatial[0] / 2)
    raw = np.empty((total, *spatial), dtype=np.float32)
    # Generate one frame at a time: bounded temporary memory and causal random
    # generation. No normalization uses future frames or event labels.
    correlated = np.zeros(spatial, dtype=np.float32)
    nuisance_path = []
    for frame in range(total):
        innovation = rng.standard_normal(spatial, dtype=np.float32)
        raw[frame] = background + 2.0 * innovation
        t = frame - start
        if spec.template_id == "nuisance_variance_correlation" and t >= 0:
            # Spatial correlation from a declared periodic 3x3 box; temporal AR1.
            spatial_noise = sum(np.roll(innovation, (dy, dx), axis=(0, 1))
                                for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 3.0
            correlated = 0.8 * correlated + 0.6 * spatial_noise
            scale = 2.0 if t < 100 else 5.0
            raw[frame] = background + scale * innovation + 3.0 * correlated
        elif spec.template_id == "nuisance_shared_brightness_motion" and t >= 0:
            shared = 12.0 if 40 <= t < 100 else (-8.0 if 160 <= t < 220 else 0.0)
            # A moving Gaussian artifact is deliberately NOT a neural source.
            # It has no expert/event/active truth row; its path is declared here.
            artifact_x = halo + 10 + (evaluation_shape[1] - 21) * t / max(1, application - 1)
            artifact_y = halo + evaluation_shape[0] * 0.35 + 5 * math.sin(2 * math.pi * t / 100)
            artifact = 20 * np.exp(-((xx - artifact_x) ** 2 + (yy - artifact_y) ** 2) / 18)
            raw[frame] += shared + artifact
            nuisance_path.append({"source_frame_ui": frame + 1, "x_px": artifact_x,
                                  "y_px": artifact_y, "shared_offset": shared})

    events: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    occupied: set[tuple[str, int]] = set()
    for index, definition in enumerate(_event_definitions(spec.template_id, evaluation_shape), 1):
        definition = {**definition, "x_px": definition["x_px"] + halo,
                      "y_px": definition["y_px"] + halo}
        profile = temporal_profile(definition["rise_ms"], definition["decay_ms"],
                                   hold_frames=definition["hold_frames"])
        first = start + definition["onset_application_index"]
        stop = first + len(profile)
        if stop > total:
            raise ValueError("application_frames truncates a declared event's finite active window")
        r2 = (xx - definition["x_px"]) ** 2 + (yy - definition["y_px"]) ** 2
        sigma = definition["sigma_px"]
        footprint = np.exp(-r2 / (2 * sigma ** 2)) * (r2 <= (4 * sigma) ** 2)
        event_id = f"{spec.scene_id}__event{index:02d}"
        roi = f"{spec.scene_id}__{definition['canonical_roi_id']}"
        event = {
            **definition, "scene_id": spec.scene_id, "template_id": spec.template_id,
            "seed": seed, "event_id": event_id, "observation_id": event_id,
            "canonical_roi_id": roi, "source_start_ui": first + 1,
            "source_stop_ui": stop, "onset_source_frame_ui": first + 1,
            "active_frame_count": len(profile), "truth_mode": "fully_synthetic",
            "annotation_source": "synthetic_generator_exact_active_support",
            "biological_identity_claimed": False,
            "spatial_fwhm_px": 2 * math.sqrt(2 * math.log(2)) * sigma,
            "spatial_sigma_um": sigma * PIXEL_SIZE_UM,
            "temporal_cutoff_fraction": 0.01,
        }
        events.append(event)
        for offset, amplitude in enumerate(profile):
            frame = first + offset
            key = (roi, frame + 1)
            if key in occupied:
                raise ValueError("overlapping events at the same canonical ROI duplicate active truth")
            occupied.add(key)
            raw[frame] += definition["peak_amplitude"] * amplitude * footprint
            active.append({"scene_id": spec.scene_id, "source_frame_ui": frame + 1,
                           "event_id": event_id, "observation_id": event_id,
                           "canonical_roi_id": roi, "burst_id": definition["burst_id"],
                           "x_px": definition["x_px"], "y_px": definition["y_px"],
                           "amplitude_at_center": float(definition["peak_amplitude"] * amplitude)})
    metadata = {
        "schema": "gamma_st_synthetic_scene_v1", "scene_id": spec.scene_id,
        "template_id": spec.template_id, "seed": seed, "shape_tyx": list(raw.shape),
        "dtype": "float32", "frame_interval_s": FRAME_INTERVAL_S,
        "evaluation_shape_yx": list(evaluation_shape), "spatial_halo_px": halo,
        "evaluation_box_yxyx": [halo, halo, halo + evaluation_shape[0], halo + evaluation_shape[1]],
        "evaluation_origin_yx": [halo, halo],
        "evaluation_box_convention": "[y0,x0,y1,x1], zero-based, half-open",
        "truth_coordinate_system": "full raw image; subtract evaluation_origin_yx after score cropping",
        "frame_rate_hz": 50.0, "pixel_size_um": PIXEL_SIZE_UM,
        "setup_frames": setup, "warmup_frames": warmup, "application_frames": application,
        "warmup_source_frames_ui": list(range(1, warmup + 1)),
        "setup_source_frames_ui": list(range(warmup + 1, warmup + setup + 1)),
        "application_source_frames_ui": list(range(start + 1, total + 1)),
        "application_source_start_ui": start + 1, "application_source_stop_ui": total,
        "source_frames_ui": list(range(1, total + 1)),
        "truth_mode": "fully_synthetic", "event_count": len(events),
        "active_region_frame_count": len(active), "spatial_truncation_sigma": 4.0,
        "background": {"offset": 100.0, "gradient_x_per_px": 0.03,
                       "gradient_y_per_px": 0.02, "base_noise_std": 2.0},
        "nuisance_parameters": (
            {"noise_std_before_application_offset100": 2.0, "noise_std_after": 5.0,
             "ar1_coefficient": 0.8, "ar1_innovation_coefficient": 0.6,
             "correlated_amplitude": 3.0, "spatial_filter": "periodic_3x3_sum_divided_by3"}
            if spec.template_id == "nuisance_variance_correlation" else
            {"artifact_peak_amplitude": 20.0, "artifact_sigma_px": 3.0,
             "artifact_is_neural_truth": False, "path": nuisance_path}
            if spec.template_id == "nuisance_shared_brightness_motion" else {}),
        "known_truth_scope": "all generated neural active-region frames in application",
        "kinetic_parameters_are_provisional_stress_settings": True,
        "rise_definition": "sampled exponential rise to normalized peak at ceil(3*rise/dt)",
        "decay_definition": "exponential decay, samples below 1% peak omitted; outside support exactly zero",
        "event_metric_definition": "coverage of each declared active window; not distinct onset classification",
        "frames_outside_application_in_primary_metrics": False,
        "scientific_promotion": False,
    }
    return SyntheticScene(raw, events, active, metadata)
