"""Paired raw-variance/spatial/temporal covariance controls with empty truth.

The application factors are fully prescribed. No sample normalization, movie
fit, truth parsing or detector output is used by this generator.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from .followup_fixtures import (
    SEEDS, EVALUATION_SHAPE_YX, SPATIAL_HALO_PX, WARMUP_FRAMES,
    SETUP_FRAMES, APPLICATION_FRAMES,
)


SCHEMA = "gamma_noise_covariance_factorial_v1"
INITIALIZER_STREAM = (20260915, 1)
AR_RHO = .8
AR_INNOVATION = .6
VARIANCE_STEP_APPLICATION_INDEX = 100


def _condition(seed, v, s, t):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    factors = []
    for name, value in zip(("v", "s", "t"), (v, s, t)):
        if not isinstance(value, (bool, np.bool_, int, np.integer)) or value not in (0, 1):
            raise ValueError(f"{name} must be zero or one")
        factors.append(int(value))
    return int(seed), *factors


def case_id(seed, v, s, t):
    seed, v, s, t = _condition(seed, v, s, t)
    if not (v or s or t):
        return f"reference_null_stationary__seed{seed}"
    return f"noise_v{v}_s{s}_t{t}__seed{seed}"


def specs():
    return tuple(dict(case_id=case_id(seed, v, s, t), seed=seed, v=v, s=s, t=t,
                      kind="factorial", factors=dict(V=v, S=s, T=t))
                 for seed in SEEDS for v in (0, 1) for s in (0, 1) for t in (0, 1))


def _spatial(values, enabled):
    if not enabled:
        return values
    return sum(np.roll(values, (dy, dx), axis=(0, 1))
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 3.0


def _initial_state(seed, spatial, shape_yx):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), *INITIALIZER_STREAM]))
    return _spatial(rng.standard_normal(shape_yx, dtype=np.float32), spatial)


def _application_step(innovation, state, *, v, s, t, application_index, variance_step):
    spatial = _spatial(innovation, s)
    unit = AR_RHO * state + AR_INNOVATION * spatial if t else spatial
    sigma = 5.0 if v and application_index >= variance_step else 2.0
    return sigma * unit, unit if t else state


def _iter_config(seed, v, s, t, *, shape_yx, warmup_frames, setup_frames,
                 application_frames, variance_step=VARIANCE_STEP_APPLICATION_INDEX):
    """Parameterized small-array entry point; production geometry stays fixed."""
    seed, v, s, t = _condition(seed, v, s, t)
    if (len(shape_yx) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 5 for n in shape_yx)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                   for n in (warmup_frames, setup_frames, application_frames, variance_step))):
        raise ValueError("Invalid fixture geometry or frame counts")
    yy, xx = np.indices(shape_yx, dtype=np.float32)
    background = 100 + .03 * (xx-shape_yx[1]/2) + .02 * (yy-shape_yx[0]/2)
    rng = np.random.default_rng(seed)
    # This independent draw never advances the frozen main innovation stream.
    state = _initial_state(seed, s, shape_yx) if t else None
    start = warmup_frames + setup_frames
    for frame in range(start + application_frames):
        innovation = rng.standard_normal(shape_yx, dtype=np.float32)
        if frame < start:
            yield background + 2.0 * innovation
        else:
            noise, state = _application_step(innovation, state, v=v, s=s, t=t,
                application_index=frame-start, variance_step=variance_step)
            yield background + noise


def iter_noise_frames(seed, v, s, t):
    """Yield raw float32 frames on the fixed 226-square, 464-frame domain."""
    yield from _iter_config(seed, v, s, t,
        shape_yx=tuple(n + 2*SPATIAL_HALO_PX for n in EVALUATION_SHAPE_YX),
        warmup_frames=WARMUP_FRAMES, setup_frames=SETUP_FRAMES,
        application_frames=APPLICATION_FRAMES)


def unit_covariance(s, t, *, lag_frames=0, displacement_yx=(0, 0)):
    """Theoretical unit-field covariance within the application, away from wrap.

    The initial AR field is stationary and independent of visible setup. This
    formula concerns application pairs, not pairs crossing the setup boundary.
    """
    _, _, s, t = _condition(SEEDS[0], 0, s, t)
    if isinstance(lag_frames, bool) or not isinstance(lag_frames, int) or lag_frames < 0:
        raise ValueError("lag_frames must be a nonnegative integer")
    if len(displacement_yx) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in displacement_yx):
        raise ValueError("displacement_yx must contain two integers")
    dy, dx = displacement_yx
    spatial = max(0, 3-abs(dy))*max(0, 3-abs(dx))/9 if s else float(dy == dx == 0)
    temporal = AR_RHO**lag_frames if t else float(lag_frames == 0)
    return float(spatial*temporal)


def legacy_variance(application_index):
    """Pointwise variance of the frozen additive compound-noise bridge."""
    if isinstance(application_index, bool) or not isinstance(application_index, int) or application_index < 0:
        raise ValueError("application_index must be nonnegative")
    sigma = 2. if application_index < VARIANCE_STEP_APPLICATION_INDEX else 5.
    return sigma*sigma + 9*(1-AR_RHO**(2*(application_index+1))) + 1.2*sigma


def theory():
    """Finite-filter and stationary covariance identities, never fitted to data."""
    x = np.arange(-4, 5, dtype=np.float64)
    g = np.exp(-x*x/2); g /= g.sum()
    box = np.full(3, 1/math.sqrt(3), dtype=np.float64)
    composed = np.convolve(g, box)
    gaussian_energy = float(np.sum(g*g)**2)
    composed_energy = float(np.sum(composed*composed)**2)
    alpha, retention = .4, .6
    ema_white = alpha*alpha/(1-retention*retention)
    ema_ar = ema_white*(1+retention*AR_RHO)/(1-retention*AR_RHO)
    return dict(schema_version=1, mathematical_diagnostic_only=True,
        precision="float64 theoretical arithmetic; source movie recurrence is float32",
        raw_pointwise_variance=dict(V0=[4., 4.], V1=[4., 25.], application_split_index=100,
                                   theoretical_not_empirically_normalized=True),
        unit_covariance="C_S(dy,dx) * rho^abs(lag), rho=0 or .8; temporal factor is one at lag0",
        spatial_covariance_on="max(0,3-|dy|)*max(0,3-|dx|)/9 away from periodic wrap",
        gaussian=dict(sigma_px=1., truncate=4., mode="reflect; interior calculation", weights_1d=g.tolist(),
                      white_noise_energy=gaussian_energy, spatial_on_composed_energy=composed_energy,
                      spatial_on_off_variance_ratio=composed_energy/gaussian_energy),
        ema=dict(alpha=alpha, retention=retention, raw_ar_rho=AR_RHO,
                 white_noise_variance_factor=ema_white, ar_noise_variance_factor=ema_ar,
                 temporal_on_off_variance_ratio=ema_ar/ema_white),
        combined_conditioned_variance_ratio=(composed_energy/gaussian_energy)*(ema_ar/ema_white),
        validity="Interior and stationary constant-variance phase after conditioning transients. Excludes initial mixing of white setup with changed application covariance, and excludes the EMA transient after the variance step. Raw application variance is stationary from its first frame; conditioned variance need not be.",
        reference_variance_expectation="Var_r(background) + sum_i r_i Sigma_ii - r^T Sigma r; homogeneous diagonal reduces sum_i r_i Sigma_ii to pointwise variance",
        contrast_variance="(t-r)^T Sigma (t-r); t and r are target and reference weights on a common grid",
        no_score_distribution_claim="These identities do not establish a normal Z distribution or a constant application false-alarm probability.",
        legacy=dict(formula="sigma(t)^2 + 9*(1-.8^(2*(t+1))) + 1.2*sigma(t)",
                    sigma_before=2., sigma_after=5., application_step_index=100,
                    covariance_current_epsilon_with_AR=.2,
                    first_application_variance=legacy_variance(0),
                    before_step_asymptotic_variance=15.4, after_step_asymptotic_variance=40.,
                    explanation="Old sigma*epsilon +3h uses h=.8hprev+.6L(epsilon) from zero, so the same current innovation contributes a cross term. This bridge is not factorial111."))


def noise_metadata(seed, v, s, t):
    seed, v, s, t = _condition(seed, v, s, t)
    case = case_id(seed, v, s, t); start = WARMUP_FRAMES+SETUP_FRAMES
    total = start+APPLICATION_FRAMES; halo = SPATIAL_HALO_PX
    shape = [total, *(n+2*halo for n in EVALUATION_SHAPE_YX)]
    box = [halo, halo, halo+EVALUATION_SHAPE_YX[0], halo+EVALUATION_SHAPE_YX[1]]
    return dict(schema=SCHEMA, schema_version=1, scene_id=case, case_id=case, kind="factorial",
        template_id=f"noise_v{v}_s{s}_t{t}", null_kind=f"v{v}_s{s}_t{t}",
        seed=seed, v=v, s=s, t=t, factors=dict(V=v, S=s, T=t),
        shape_tyx=shape, dtype="float32", truth_mode="fully_synthetic", event_count=0,
        active_region_frame_count=0, source_roles=[], neural_sources_generated=False,
        frame_interval_s=.02, frame_rate_hz=50., pixel_size_um=.5,
        evaluation_shape_yx=list(EVALUATION_SHAPE_YX), spatial_halo_px=halo,
        evaluation_box_yxyx=box, evaluation_origin_yx=[halo, halo],
        evaluation_box_convention="[y0,x0,y1,x1], zero-based, half-open",
        original_source_offset_xy=[halo, halo], truth_coordinate_system="empty neural truth",
        coordinate_transform="audit/score coordinates = original source minus [x0,y0]",
        warmup_frames=WARMUP_FRAMES, setup_frames=SETUP_FRAMES, application_frames=APPLICATION_FRAMES,
        warmup_source_frames_ui=list(range(1,WARMUP_FRAMES+1)),
        setup_source_frames_ui=list(range(WARMUP_FRAMES+1,start+1)),
        application_source_frames_ui=list(range(start+1,total+1)),
        application_source_start_ui=start+1, application_source_stop_ui=total,
        source_frames_ui=list(range(1,total+1)), factors_start_source_frame_ui=start+1,
        variance_step_source_frame_ui=start+VARIANCE_STEP_APPLICATION_INDEX+1 if v else None,
        paired_noise_group=f"seed{seed}", paired_setup_identical_within_seed=True,
        baseline_stationary_case_id=f"reference_null_stationary__seed{seed}",
        background=dict(offset=100., gradient_x_per_px=.03, gradient_y_per_px=.02, base_noise_std=2.),
        noise_parameters=dict(spatial_filter="periodic 3x3 sum divided by3" if s else "identity",
            temporal_ar_rho=AR_RHO if t else 0., temporal_innovation_coefficient=AR_INNOVATION if t else 1.,
            stationary_initial_state="L_S eta; independent standard Gaussian eta" if t else "not used",
            initializer_seed_sequence=[seed,*INITIALIZER_STREAM], main_rng="default_rng(seed), one float32 standard_normal image per source frame",
            setup_innovations_not_used_for_AR_initialization=True, main_stream_not_advanced_by_initializer=True,
            variance_scaling_after_AR=True, empirical_variance_normalization=False,
            sigma_before_step=2., sigma_after_step=5. if v else 2.,
            variance_step_application_index=VARIANCE_STEP_APPLICATION_INDEX if v else None,
            raw_pointwise_variance_before_step=4., raw_pointwise_variance_after_step=25. if v else 4.,
            raw_variance_scope="theoretical ensemble variance, not the sample variance of a finite movie",
            application_covariance="sigma(t)*sigma(u)*C_S(delta)*rho^abs(t-u); independent unit-variance initialization; applies within application only"),
        known_truth_scope="No generated neural sources anywhere in application; all neural-source proposals are false under this synthetic truth.",
        empty_truth_semantics="Known synthetic null, not sparse unannotated real data.",
        expert_audit_applicability="No expert ROIs or occurrences; retain empty Expert/Comparison sections and all required model artifacts.",
        conditioning="Gaussian sigma1/truncate4 reflect and causal EMA alpha.4; level and signed adjacent difference files follow the frozen data contract.",
        calibration_uses_truth=False, activity_truth_parsed=False, evaluation_requires_all_candidates_sealed=True,
        scientific_promotion=False, legacy_bridge_generated=False)


def _binding(path, logical_path=None):
    path = Path(path); digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return dict(path=str(Path(logical_path or path).resolve()), sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def _source_bindings():
    here = Path(__file__).resolve()
    return [_binding(p) for p in (here, here.with_name("followup_fixtures.py"),
                                  here.with_name("control_study.py"))]


_DATA_FILES = ("raw_source.npy", "Raw.npy", "input_source.npy", "Input.npy",
               "level_source.npy", "level.npy", "metadata.json", "experts.json", "active.json")


def _resume(folder, seed, v, s, t):
    receipt = json.loads((folder/"noise_prepared.json").read_text())
    case = case_id(seed, v, s, t)
    if (receipt.get("status"),receipt.get("schema"),receipt.get("case_id"),receipt.get("seed"),receipt.get("factors")) != ("PASS",SCHEMA,case,seed,dict(V=v,S=s,T=t)):
        raise RuntimeError("Noise preparation identity/status differs")
    expected = set(_DATA_FILES)|{"prepared.json","level_prepared.json"}
    actual = [Path(b["path"]).resolve() for b in receipt["bindings"]]
    if len(actual) != len(expected) or set(actual) != {folder/name for name in expected}:
        raise RuntimeError("Noise preparation file inventory differs")
    if receipt["source_bindings"] != _source_bindings():
        raise RuntimeError("Noise recipe source changed; preserve existing dataset")
    for b in receipt["bindings"]:
        if _binding(b["path"]) != b:
            raise RuntimeError(f"Noise preparation artifact changed: {b['path']}")
    return dict(status="PASS",case_id=case,folder=str(folder),resumed=True,
                receipt=_binding(folder/"noise_prepared.json"),bindings=receipt["bindings"])


def build_noise_dataset(root, seed, v, s, t):
    """Create atomically or verify exact resumption; never parse activity truth.

    Empty expert/active files are written and byte-hashed as artifacts. Neither
    is parsed or used to generate noise, conditioning or any fitted quantity.
    The original stationary dataset may instead be reused by the parent runner;
    this function does not overwrite or retrofit legacy preparation receipts.
    """
    from .control_study import conditioned_frames
    seed, v, s, t = _condition(seed, v, s, t)
    root = Path(root).resolve()
    if (root/"completion_manifest.json").exists():
        raise RuntimeError("Completed study roots are immutable")
    folder = root/"datasets"/case_id(seed,v,s,t)
    if folder.exists():
        return _resume(folder,seed,v,s,t)
    folder.parent.mkdir(parents=True,exist_ok=True)
    meta=noise_metadata(seed,v,s,t); sources=_source_bindings()
    with tempfile.TemporaryDirectory(prefix=f".{folder.name}.",dir=folder.parent) as work:
        staging=Path(work)
        def array(name,shape):
            return np.lib.format.open_memmap(staging/name,mode="w+",dtype=np.float32,shape=shape)
        def bind(name):
            return _binding(staging/name,folder/name)
        raw=array("raw_source.npy",tuple(meta["shape_tyx"]));count=0
        for i,frame in enumerate(iter_noise_frames(seed,v,s,t)):
            if i>=len(raw) or frame.shape!=raw.shape[1:] or frame.dtype!=np.float32 or not np.isfinite(frame).all():
                raise RuntimeError("Noise generator emitted an invalid frame")
            raw[i]=frame;count+=1
        if count!=len(raw):
            raise RuntimeError("Noise generator frame count differs")
        raw.flush();y0,x0,y1,x1=meta["evaluation_box_yxyx"]
        np.save(staging/"Raw.npy",raw[:,y0:y1,x0:x1])
        level=array("level_source.npy",raw.shape);difference=array("input_source.npy",raw.shape)
        for i,(current,delta) in enumerate(conditioned_frames(raw,1.,.4)):
            level[i],difference[i]=current,delta
        level.flush();difference.flush()
        np.save(staging/"level.npy",level[:,y0:y1,x0:x1])
        np.save(staging/"Input.npy",difference[:,y0:y1,x0:x1])
        del raw,level,difference
        _write(staging/"metadata.json",meta)
        _write(staging/"experts.json",[]);_write(staging/"active.json",[])
        _write(staging/"prepared.json",dict(status="PASS",source=bind("raw_source.npy"),
            input=bind("input_source.npy"),metadata=bind("metadata.json"),experts=bind("experts.json"),active=bind("active.json")))
        _write(staging/"level_prepared.json",dict(status="PASS",bindings=[bind("level_source.npy"),bind("level.npy")]))
        _write(staging/"noise_prepared.json",dict(status="PASS",schema=SCHEMA,case_id=folder.name,
            seed=seed,factors=dict(V=v,S=s,T=t),source_bindings=sources,
            bindings=[bind(name) for name in (*_DATA_FILES,"prepared.json","level_prepared.json")],
            activity_truth_parsed=False,scientific_promotion=False))
        staging.rename(folder)
    return dict(status="PASS",case_id=folder.name,folder=str(folder),resumed=False,
                receipt=_binding(folder/"noise_prepared.json"),
                bindings=json.loads((folder/"noise_prepared.json").read_text())["bindings"])
