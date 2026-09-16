"""Paired null fixtures separating background slope and conditioned variance.

Only the zero-mean *application noise* receives the prescribed variance gain.
The background and white-noise setup are never scaled. Matching is an ensemble,
stationary-interior calculation for Gaussian sigma=1/truncate=4 plus EMA=.4;
it is neither a movie fit nor exact matching through either transition. No
activity truth, detector result, or empirical normalization enters generation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

import numpy as np

from . import noise_fixtures as _noise
from .noise_fixtures import (
    SEEDS, EVALUATION_SHAPE_YX, SPATIAL_HALO_PX, WARMUP_FRAMES,
    SETUP_FRAMES, APPLICATION_FRAMES, VARIANCE_STEP_APPLICATION_INDEX,
)


SCHEMA = "gamma_background_conditioned_variance_factorial_v1"
BACKGROUNDS = ("flat", "sloped")
NORMALIZATIONS = ("raw", "conditioned")
_DATA_FILES = _noise._DATA_FILES


def _condition(seed, v, s, t, background, normalization):
    seed, v, s, t = _noise._condition(seed, v, s, t)
    if not isinstance(background, str) or background not in BACKGROUNDS:
        raise ValueError(f"background must be exactly one of {BACKGROUNDS}")
    if not isinstance(normalization, str) or normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be exactly one of {NORMALIZATIONS}")
    # These two requested labels describe one identical source, not replicates.
    normalization = normalization if s or t else "raw"
    return seed, v, s, t, background, normalization


def case_id(seed, v, s, t, background, normalization):
    """Canonical identity; S=T=0 aliases conditioned to raw for both V values."""
    seed, v, s, t, background, normalization = _condition(
        seed, v, s, t, background, normalization)
    return f"bg_{background}__{normalization}_v{v}_s{s}_t{t}__seed{seed}"


def matching_parameters(s, t, normalization):
    """Exact finite-filter theory and the float32 gain used by the source."""
    _, _, s, t, _, normalization = _condition(
        SEEDS[0], 0, s, t, "flat", normalization)
    theory = _noise.theory()
    k_s = float(theory["gaussian"]["spatial_on_off_variance_ratio"]) if s else 1.0
    k_t = float(theory["ema"]["temporal_on_off_variance_ratio"]) if t else 1.0
    theoretical_gain = 1.0 / math.sqrt(k_s * k_t) if normalization == "conditioned" else 1.0
    applied_gain = float(np.float32(theoretical_gain))
    baseline = (theory["gaussian"]["white_noise_energy"]
                * theory["ema"]["white_noise_variance_factor"])
    return dict(
        canonical_normalization=normalization, k_S=k_s, k_T=k_t,
        theoretical_application_noise_gain=theoretical_gain,
        applied_float32_application_noise_gain=applied_gain,
        ideal_conditioned_variance_ratio_to_white=k_s*k_t*theoretical_gain**2,
        float32_gain_conditioned_variance_ratio_to_white=k_s*k_t*applied_gain**2,
        stationary_white_conditioning_variance_factor=baseline,
        stationary_conditioned_variance_factor=baseline*k_s*k_t*theoretical_gain**2,
        matching_scope=theory["validity"],
        matched_quantity="Ensemble noise variance after the fixed Gaussian and level EMA, at the same V/sigma; not signed-difference variance or total reference spread.",
        background_scaled=False, setup_scaled=False, empirical_variance_normalization=False,
        gaussian=theory["gaussian"], ema=theory["ema"],
        arithmetic="Float32 source recurrence and scalar multiplication; theoretical gains and moments use float64.")


def _background(shape_yx, background):
    if background == "flat":
        return np.full(shape_yx, 100.0, dtype=np.float32)
    yy, xx = np.indices(shape_yx, dtype=np.float32)
    # Preserve the original background's operations, dtype and half-size origin.
    return 100 + .03 * (xx-shape_yx[1]/2) + .02 * (yy-shape_yx[0]/2)


def _iter_config(seed, v, s, t, background, normalization, *, shape_yx,
                 warmup_frames, setup_frames, application_frames,
                 variance_step=VARIANCE_STEP_APPLICATION_INDEX):
    """Tiny-fixture entry point; the public iterator fixes the production grid."""
    seed, v, s, t, background, normalization = _condition(
        seed, v, s, t, background, normalization)
    if (len(shape_yx) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 5 for n in shape_yx)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                   for n in (warmup_frames, setup_frames, application_frames, variance_step))):
        raise ValueError("Invalid fixture geometry or frame counts")
    fixed_background = _background(shape_yx, background)
    gain = np.float32(matching_parameters(s, t, normalization)["applied_float32_application_noise_gain"])
    rng = np.random.default_rng(seed)
    # The independent eta stream and its spatial filtering are exactly frozen.
    state = _noise._initial_state(seed, s, shape_yx) if t else None
    start = warmup_frames + setup_frames
    for frame in range(start + application_frames):
        innovation = rng.standard_normal(shape_yx, dtype=np.float32)
        if frame < start:
            yield fixed_background + 2.0 * innovation
        else:
            noise, state = _noise._application_step(
                innovation, state, v=v, s=s, t=t,
                application_index=frame-start, variance_step=variance_step)
            if gain != 1.0:
                noise = noise * gain
            yield fixed_background + noise


def iter_background_frames(seed, v, s, t, background, normalization):
    """Yield 464 float32 226-square raw frames; UI165 starts application."""
    yield from _iter_config(seed, v, s, t, background, normalization,
        shape_yx=tuple(n+2*SPATIAL_HALO_PX for n in EVALUATION_SHAPE_YX),
        warmup_frames=WARMUP_FRAMES, setup_frames=SETUP_FRAMES,
        application_frames=APPLICATION_FRAMES)


def background_metadata(seed, v, s, t, background, normalization):
    seed, v, s, t, background, normalization = _condition(
        seed, v, s, t, background, normalization)
    meta = _noise.noise_metadata(seed, v, s, t)
    case = case_id(seed, v, s, t, background, normalization)
    start = WARMUP_FRAMES + SETUP_FRAMES
    total = start + APPLICATION_FRAMES
    halo = SPATIAL_HALO_PX
    gain = matching_parameters(s, t, normalization)
    scale = gain["theoretical_application_noise_gain"]
    applied_scale = gain["applied_float32_application_noise_gain"]
    meta.update(
        schema=SCHEMA, scene_id=case, case_id=case, kind="background_factorial",
        template_id=f"bg_{background}__{normalization}_v{v}_s{s}_t{t}",
        null_kind=f"bg_{background}__{normalization}_v{v}_s{s}_t{t}",
        background_kind=background, B=background, normalization=normalization,
        factors=dict(V=v, S=s, T=t, B=background, normalization=normalization),
        normalization_identity="S=T=0 uses canonical raw identity for either requested normalization.",
        shape_tyx=[total, *(n+2*halo for n in EVALUATION_SHAPE_YX)],
        evaluation_shape_yx=list(EVALUATION_SHAPE_YX), spatial_halo_px=halo,
        evaluation_box_yxyx=[halo,halo,halo+EVALUATION_SHAPE_YX[0],halo+EVALUATION_SHAPE_YX[1]],
        evaluation_origin_yx=[halo,halo], original_source_offset_xy=[halo,halo],
        warmup_frames=WARMUP_FRAMES, setup_frames=SETUP_FRAMES, application_frames=APPLICATION_FRAMES,
        warmup_source_frames_ui=list(range(1,WARMUP_FRAMES+1)),
        setup_source_frames_ui=list(range(WARMUP_FRAMES+1,start+1)),
        application_source_frames_ui=list(range(start+1,total+1)),
        source_frames_ui=list(range(1,total+1)),
        application_source_start_ui=start+1, application_source_stop_ui=total,
        factors_start_source_frame_ui=start+1,
        variance_step_source_frame_ui=start+VARIANCE_STEP_APPLICATION_INDEX+1 if v else None,
        paired_noise_group=f"seed{seed}", paired_setup_identical_within_seed=False,
        paired_setup_identical_within_seed_and_background=True,
        paired_setup_innovations_identical_across_backgrounds=True,
        calibration_scope="Each background calibrates from its own white sigma2 setup; S/T and gain begin only in application.",
        baseline_stationary_case_id=case_id(seed,0,0,0,background,"raw"),
        frozen_noise_case_id=_noise.case_id(seed,v,s,t),
        exact_frozen_raw_frame_parity=background == "sloped" and normalization == "raw",
        background=dict(kind=background, offset=100.,
            gradient_x_per_px=.03 if background == "sloped" else 0.,
            gradient_y_per_px=.02 if background == "sloped" else 0.,
            center_xy=[(EVALUATION_SHAPE_YX[1]+2*halo)/2,(EVALUATION_SHAPE_YX[0]+2*halo)/2],
            base_noise_std=2., scaled=False),
        variance_matching=gain,
        synthetic_null_scope="No generated neural sources; unknown biological proposals are not inferred from these synthetic nulls.")
    meta["noise_parameters"].update(
        setup_noise_std=2., application_noise_gain=scale,
        application_noise_gain_float32=applied_scale,
        noise_only_scaling_after_sigma=True,
        raw_pointwise_variance_before_step=4.*scale**2,
        raw_pointwise_variance_after_step=(25. if v else 4.)*scale**2,
        float32_gain_raw_pointwise_variance_before_step=4.*applied_scale**2,
        float32_gain_raw_pointwise_variance_after_step=(25. if v else 4.)*applied_scale**2,
        application_covariance="gain^2*sigma(t)*sigma(u)*C_S(delta)*rho^abs(t-u); independent unit-variance initialization; application pairs only",
        conditioning_transients_matched=False,
        background_is_not_scaled=True)
    return meta


def _source_bindings():
    here = Path(__file__).resolve()
    return [_noise._binding(path) for path in (
        here, here.with_name("noise_fixtures.py"),
        here.with_name("followup_fixtures.py"), here.with_name("control_study.py"))]


def _resume(folder, seed, v, s, t, background, normalization):
    receipt_path = folder/"background_prepared.json"
    if not receipt_path.is_file():
        raise RuntimeError("Existing background dataset has no completed preparation receipt")
    receipt = json.loads(receipt_path.read_text())
    case = case_id(seed,v,s,t,background,normalization)
    factors = dict(V=v,S=s,T=t,B=background,normalization=normalization)
    if (receipt.get("status"),receipt.get("schema"),receipt.get("case_id"),
            receipt.get("seed"),receipt.get("factors")) != ("PASS",SCHEMA,case,seed,factors):
        raise RuntimeError("Background preparation identity/status differs")
    if receipt.get("variance_matching") != matching_parameters(s,t,normalization):
        raise RuntimeError("Background preparation variance-matching definition differs")
    if receipt.get("activity_truth_parsed") is not False or receipt.get("scientific_promotion") is not False:
        raise RuntimeError("Background preparation scope differs")
    expected = set(_DATA_FILES)|{"prepared.json","level_prepared.json"}
    actual = [Path(b["path"]).resolve() for b in receipt["bindings"]]
    if len(actual) != len(expected) or set(actual) != {folder/name for name in expected}:
        raise RuntimeError("Background preparation file inventory differs")
    if receipt["source_bindings"] != _source_bindings():
        raise RuntimeError("Background recipe source changed; preserve existing dataset")
    for binding in receipt["bindings"]:
        if _noise._binding(binding["path"]) != binding:
            raise RuntimeError(f"Background preparation artifact changed: {binding['path']}")
    return dict(status="PASS",case_id=case,folder=str(folder),resumed=True,
                receipt=_noise._binding(receipt_path),bindings=receipt["bindings"])


def build_background_dataset(root, seed, v, s, t, background, normalization):
    """Atomically prepare source/cropped stages, or verify an exact resume.

    Empty truth files are generated and byte-hashed, never parsed. Preparation
    is not evaluation or scientific-audit completion. Existing completed roots
    and datasets with differing identities, artifacts or recipe code are kept.
    """
    from .control_study import conditioned_frames
    seed,v,s,t,background,normalization = _condition(seed,v,s,t,background,normalization)
    root = Path(root).resolve()
    if (root/"completion_manifest.json").exists():
        raise RuntimeError("Completed study roots are immutable")
    folder = root/"datasets"/case_id(seed,v,s,t,background,normalization)
    if folder.exists():
        return _resume(folder,seed,v,s,t,background,normalization)
    folder.parent.mkdir(parents=True,exist_ok=True)
    meta = background_metadata(seed,v,s,t,background,normalization)
    sources = _source_bindings()
    with tempfile.TemporaryDirectory(prefix=f".{folder.name}.",dir=folder.parent) as work:
        staging = Path(work)
        def array(name,shape):
            return np.lib.format.open_memmap(staging/name,mode="w+",dtype=np.float32,shape=shape)
        def bind(name):
            return _noise._binding(staging/name,folder/name)
        raw = array("raw_source.npy",tuple(meta["shape_tyx"]))
        count = 0
        for i,frame in enumerate(iter_background_frames(seed,v,s,t,background,normalization)):
            if i>=len(raw) or frame.shape!=raw.shape[1:] or frame.dtype!=np.float32 or not np.isfinite(frame).all():
                raise RuntimeError("Background generator emitted an invalid frame")
            raw[i] = frame
            count += 1
        if count != len(raw):
            raise RuntimeError("Background generator frame count differs")
        raw.flush()
        y0,x0,y1,x1 = meta["evaluation_box_yxyx"]
        np.save(staging/"Raw.npy",raw[:,y0:y1,x0:x1])
        level = array("level_source.npy",raw.shape)
        difference = array("input_source.npy",raw.shape)
        count = 0
        for i,(current,delta) in enumerate(conditioned_frames(raw,1.,.4)):
            if i>=len(raw) or not np.isfinite(current).all() or not np.isfinite(delta).all():
                raise RuntimeError("Background conditioning emitted an invalid frame")
            level[i],difference[i] = current,delta
            count += 1
        if count != len(raw):
            raise RuntimeError("Background conditioning frame count differs")
        level.flush(); difference.flush()
        np.save(staging/"level.npy",level[:,y0:y1,x0:x1])
        np.save(staging/"Input.npy",difference[:,y0:y1,x0:x1])
        del raw,level,difference
        _noise._write(staging/"metadata.json",meta)
        _noise._write(staging/"experts.json",[])
        _noise._write(staging/"active.json",[])
        _noise._write(staging/"prepared.json",dict(status="PASS",source=bind("raw_source.npy"),
            input=bind("input_source.npy"),metadata=bind("metadata.json"),
            experts=bind("experts.json"),active=bind("active.json")))
        _noise._write(staging/"level_prepared.json",dict(status="PASS",bindings=[bind("level_source.npy"),bind("level.npy")]))
        receipt = dict(status="PASS",schema=SCHEMA,case_id=folder.name,seed=seed,
            factors=meta["factors"],source_bindings=sources,
            variance_matching=meta["variance_matching"],
            bindings=[bind(name) for name in (*_DATA_FILES,"prepared.json","level_prepared.json")],
            activity_truth_parsed=False,scientific_promotion=False)
        _noise._write(staging/"background_prepared.json",receipt)
        staging.rename(folder)
    return dict(status="PASS",case_id=folder.name,folder=str(folder),resumed=False,
                receipt=_noise._binding(folder/"background_prepared.json"),bindings=receipt["bindings"])
