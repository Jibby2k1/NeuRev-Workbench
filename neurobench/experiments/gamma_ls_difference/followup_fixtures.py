"""Paired synthetic crowding controls; no biological or outcome-derived fit.

All four conditions within a seed use exactly the same Gaussian innovations.
Only the declared strong neighbor is added. Noise/background arithmetic and
finite temporal/footprint support follow the preceding synthetic pilot.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .spatiotemporal_fixtures import SyntheticScene, temporal_profile


SEEDS = (20260916, 20260917, 20260918)
SEPARATIONS_PX = (None, 8, 12, 16)
EVALUATION_SHAPE_YX = (128, 128)
SPATIAL_HALO_PX = 49
WARMUP_FRAMES = 64
SETUP_FRAMES = 100
APPLICATION_FRAMES = 300


def _condition(seed: Any, separation_px: Any) -> tuple[int, int | None]:
    if isinstance(seed, (bool, np.bool_)) or seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if isinstance(separation_px, (bool, np.bool_)) or separation_px not in SEPARATIONS_PX:
        raise ValueError("separation_px must be None, 8, 12 or 16")
    return int(seed), None if separation_px is None else int(separation_px)


def case_id(seed: int, separation_px: int | None = None) -> str:
    seed, separation = _condition(seed, separation_px)
    condition = "alone" if separation is None else f"sep{separation}"
    return f"crowding_{condition}__seed{seed}"


def generate_crowding(seed: int, separation_px: int | None = None) -> SyntheticScene:
    """Return one of the fixed 12 raw226x226, 464-frame simulations.

    Weak center in evaluation coordinates is (x,y)=(68,64); optional strong
    neighbor is directly left. Both sources share a sampled fluorescence onset
    at application index20 (source UI185) and one inclusive active window.
    Returned coordinates are in the full raw image, including the49px halo.
    """
    seed, separation = _condition(seed, separation_px)
    identity = case_id(seed, separation)
    template = "crowding_alone" if separation is None else f"crowding_sep{separation}"
    halo = SPATIAL_HALO_PX
    shape = tuple(size + 2 * halo for size in EVALUATION_SHAPE_YX)
    total = WARMUP_FRAMES + SETUP_FRAMES + APPLICATION_FRAMES
    application_start = WARMUP_FRAMES + SETUP_FRAMES
    yy, xx = np.indices(shape, dtype=np.float32)
    background = 100 + 0.03 * (xx - shape[1] / 2) + 0.02 * (yy - shape[0] / 2)
    rng = np.random.default_rng(seed)
    raw = np.empty((total, *shape), dtype=np.float32)
    for frame in range(total):
        innovation = rng.standard_normal(shape, dtype=np.float32)
        raw[frame] = background + 2.0 * innovation

    sources = [dict(source_role="weak", x_px=68.0 + halo, y_px=64.0 + halo,
                    sigma_px=1.0, peak_amplitude=18.0)]
    if separation is not None:
        sources.append(dict(source_role="neighbor", x_px=68.0 + halo - separation,
                            y_px=64.0 + halo, sigma_px=2.0, peak_amplitude=24.0))
    profile = temporal_profile(100.0, 1000.0)
    first = application_start + 20
    stop = first + len(profile)
    if stop > total:
        raise RuntimeError("Declared crowding movie truncates its finite event window")
    events, active = [], []
    for source in sources:
        role = source["source_role"]
        event_id = f"{identity}__event_{role}"
        roi = f"{identity}__roi_{role}"
        sigma = source["sigma_px"]
        radius2 = (xx - source["x_px"]) ** 2 + (yy - source["y_px"]) ** 2
        footprint = np.exp(-radius2 / (2 * sigma ** 2)) * (radius2 <= (4 * sigma) ** 2)
        event = dict(source, scene_id=identity, template_id=template, seed=seed,
                     event_id=event_id, observation_id=event_id, canonical_roi_id=roi, burst_id=1,
                     rise_ms=100.0, decay_ms=1000.0, hold_frames=0, onset_application_index=20,
                     source_start_ui=first + 1, source_stop_ui=stop, onset_source_frame_ui=first + 1,
                     active_frame_count=len(profile), truth_mode="fully_synthetic",
                     annotation_source="synthetic_generator_exact_active_support",
                     biological_identity_claimed=False,
                     spatial_fwhm_px=2 * math.sqrt(2 * math.log(2)) * sigma,
                     spatial_sigma_um=sigma * 0.5, temporal_cutoff_fraction=0.01)
        events.append(event)
        for offset, amplitude in enumerate(profile):
            frame = first + offset
            raw[frame] += source["peak_amplitude"] * amplitude * footprint
            active.append(dict(scene_id=identity, source_frame_ui=frame + 1, source_role=role,
                               event_id=event_id, observation_id=event_id, canonical_roi_id=roi,
                               burst_id=1, x_px=source["x_px"], y_px=source["y_px"],
                               amplitude_at_center=float(source["peak_amplitude"] * amplitude)))
    metadata = dict(
        schema="gamma_crowding_paired_synthetic_v1", scene_id=identity, template_id=template, seed=seed,
        shape_tyx=list(raw.shape), dtype="float32", truth_mode="fully_synthetic",
        frame_interval_s=0.02, frame_rate_hz=50.0, pixel_size_um=0.5,
        evaluation_shape_yx=list(EVALUATION_SHAPE_YX), spatial_halo_px=halo,
        evaluation_box_yxyx=[halo, halo, halo + 128, halo + 128], evaluation_origin_yx=[halo, halo],
        evaluation_box_convention="[y0,x0,y1,x1], zero-based, half-open",
        truth_coordinate_system="full raw image; subtract evaluation_origin_yx after score cropping",
        warmup_frames=WARMUP_FRAMES, setup_frames=SETUP_FRAMES, application_frames=APPLICATION_FRAMES,
        warmup_source_frames_ui=list(range(1, WARMUP_FRAMES + 1)),
        setup_source_frames_ui=list(range(WARMUP_FRAMES + 1, application_start + 1)),
        application_source_frames_ui=list(range(application_start + 1, total + 1)),
        application_source_start_ui=application_start + 1, application_source_stop_ui=total,
        source_frames_ui=list(range(1, total + 1)), event_count=len(events), active_region_frame_count=len(active),
        spatial_truncation_sigma=4.0, source_roles=[s["source_role"] for s in sources],
        weak_evaluation_center_xy=[68, 64], neighbor_separation_px=separation,
        paired_noise_group=f"seed{seed}", paired_setup_identical_within_seed=True,
        background=dict(offset=100.0, gradient_x_per_px=0.03, gradient_y_per_px=0.02, base_noise_std=2.0),
        nuisance_parameters={}, known_truth_scope="all generated neural active-region frames in application",
        kinetic_parameters_are_provisional_stress_settings=True,
        rise_definition="sampled exponential rise to normalized peak at ceil(3*rise/dt)",
        decay_definition="exponential decay, samples below 1% peak omitted; outside support exactly zero",
        event_metric_definition="coverage of each declared active window; not distinct onset classification",
        frames_outside_application_in_primary_metrics=False, scientific_promotion=False,
    )
    return SyntheticScene(raw, events, active, metadata)
