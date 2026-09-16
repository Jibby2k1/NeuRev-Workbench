"""Source-free, paired null movies for the reference-geometry investigation.

The frozen follow-up supplies geometry/noise conventions; the frozen spatial-
temporal fixtures supply nuisance arithmetic. No neural sources are generated
or subtracted. Neither preparation nor resumption opens any label file.
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
from .spatiotemporal_fixtures import SyntheticScene


KINDS = ("stationary", "variance_correlation", "shared_brightness_motion")
SCHEMA = "gamma_reference_paired_null_v1"


def _condition(seed, kind):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    return int(seed), kind


def null_case_id(seed, kind):
    seed, kind = _condition(seed, kind)
    return f"reference_null_{kind}__seed{seed}"


def null_specs():
    return tuple(dict(case_id=null_case_id(seed, kind), seed=seed, kind=kind)
                 for seed in SEEDS for kind in KINDS)


def _motion(t):
    x = SPATIAL_HALO_PX + 10 + (EVALUATION_SHAPE_YX[1] - 21) * t / max(1, APPLICATION_FRAMES - 1)
    y = SPATIAL_HALO_PX + EVALUATION_SHAPE_YX[0] * .35 + 5 * math.sin(2 * math.pi * t / 100)
    shared = 12. if 40 <= t < 100 else (-8. if 160 <= t < 220 else 0.)
    return x, y, shared


def iter_null_frames(seed, kind):
    """Yield every full-halo raw float32 frame, using only one-frame workspace.

    Draw one identical innovation image at every frame for every kind. UI1..164
    remains stationary; AR state is zero until application UI165, as in the
    frozen nuisance generator. No source truth or detector output is an input.
    """
    seed, kind = _condition(seed, kind)
    shape = tuple(size + 2 * SPATIAL_HALO_PX for size in EVALUATION_SHAPE_YX)
    yy, xx = np.indices(shape, dtype=np.float32)
    background = 100 + .03 * (xx - shape[1] / 2) + .02 * (yy - shape[0] / 2)
    rng = np.random.default_rng(seed)
    correlated = np.zeros(shape, dtype=np.float32)
    start = WARMUP_FRAMES + SETUP_FRAMES
    for frame in range(start + APPLICATION_FRAMES):
        innovation = rng.standard_normal(shape, dtype=np.float32)
        field = background + 2.0 * innovation
        t = frame - start
        if kind == "variance_correlation" and t >= 0:
            spatial_noise = sum(np.roll(innovation, (dy, dx), axis=(0, 1))
                                for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 3.0
            correlated = .8 * correlated + .6 * spatial_noise
            scale = 2.0 if t < 100 else 5.0
            field = background + scale * innovation + 3.0 * correlated
        elif kind == "shared_brightness_motion" and t >= 0:
            x, y, shared = _motion(t)
            artifact = 20 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 18)
            field += shared + artifact
        yield field


def _metadata(seed, kind):
    seed, kind = _condition(seed, kind)
    case = null_case_id(seed, kind)
    halo = SPATIAL_HALO_PX
    start = WARMUP_FRAMES + SETUP_FRAMES
    total = start + APPLICATION_FRAMES
    shape = [total, *(n + 2 * halo for n in EVALUATION_SHAPE_YX)]
    box = [halo, halo, halo + EVALUATION_SHAPE_YX[0], halo + EVALUATION_SHAPE_YX[1]]
    nuisance = {}
    if kind == "variance_correlation":
        nuisance = dict(noise_std_before_application_offset100=2., noise_std_after=5.,
                        ar1_coefficient=.8, ar1_innovation_coefficient=.6, correlated_amplitude=3.,
                        spatial_filter="periodic_3x3_sum_divided_by3", ar_state_before_application="zero")
    elif kind == "shared_brightness_motion":
        nuisance = dict(artifact_peak_amplitude=20., artifact_sigma_px=3., artifact_is_neural_truth=False,
                        path=[dict(source_frame_ui=start+t+1, x_px=_motion(t)[0],
                                   y_px=_motion(t)[1], shared_offset=_motion(t)[2])
                              for t in range(APPLICATION_FRAMES)])
    return dict(schema=SCHEMA, schema_version=1, scene_id=case, case_id=case, template_id=f"null_{kind}",
        seed=seed, null_kind=kind, shape_tyx=shape, dtype="float32", truth_mode="fully_synthetic",
        frame_interval_s=.02, frame_rate_hz=50., pixel_size_um=.5,
        evaluation_shape_yx=list(EVALUATION_SHAPE_YX), spatial_halo_px=halo,
        evaluation_box_yxyx=box, evaluation_origin_yx=[halo, halo],
        evaluation_box_convention="[y0,x0,y1,x1], zero-based, half-open",
        truth_coordinate_system="full raw image; empty neural truth", coordinate_transform="audit/score coordinates = original source minus [x0,y0]",
        original_source_offset_xy=[halo, halo], warmup_frames=WARMUP_FRAMES, setup_frames=SETUP_FRAMES,
        application_frames=APPLICATION_FRAMES, warmup_source_frames_ui=list(range(1,WARMUP_FRAMES+1)),
        setup_source_frames_ui=list(range(WARMUP_FRAMES+1,start+1)),
        application_source_frames_ui=list(range(start+1,total+1)),
        application_source_start_ui=start+1, application_source_stop_ui=total, source_frames_ui=list(range(1,total+1)),
        event_count=0, active_region_frame_count=0, source_roles=[], neural_sources_generated=False,
        paired_noise_group=f"seed{seed}", paired_setup_identical_within_seed=True,
        setup_pairing_scope="Same stationary warmup/setup arithmetic and full-field innovation sequence as frozen followup_crowding; nuisance starts only in application.",
        background=dict(offset=100., gradient_x_per_px=.03, gradient_y_per_px=.02, base_noise_std=2.),
        nuisance_parameters=nuisance, nuisance_path_coordinate_system="full raw including halo",
        recipe_origin=dict(geometry_noise="followup_fixtures.generate_crowding",
                           nuisances="spatiotemporal_fixtures.generate_scene: application noise/brightness/motion branches; no event-generation loop",
                           modifications="Neural sources disabled; use follow-up three seeds and 64/100/300 timing with common stationary setup."),
        known_truth_scope="No generated neural active regions anywhere in application; every neural-source proposal is false under this synthetic truth.",
        empty_truth_semantics="Known synthetic null, not sparse unannotated real data.",
        expert_audit_applicability="No expert ROIs or occurrences; preserve explicit empty Expert/Comparison sections and all required model artifacts.",
        calibration_uses_truth=False, evaluation_requires_all_candidates_sealed=True,
        conditioning="Gaussian sigma1 truncate4 reflect, causal EMA alpha.4 initialized from first spatial frame; no rectification. Input files retain signed adjacent difference; level files retain EMA current activity.",
        scientific_promotion=False)


def generate_null(seed, kind):
    """Convenience full movie; production preparation uses the bounded iterator."""
    return SyntheticScene(np.stack(list(iter_null_frames(seed, kind))), [], [], _metadata(seed, kind))


def _binding(path, logical_path=None):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(Path(logical_path or path).resolve()), sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _source_bindings():
    here = Path(__file__).resolve()
    return [_binding(path) for path in (here, here.with_name("followup_fixtures.py"),
                                        here.with_name("spatiotemporal_fixtures.py"), here.with_name("control_study.py"))]


def _resume(folder, case, seed, kind):
    receipt = json.loads((folder / "null_prepared.json").read_text())
    if (receipt.get("status"), receipt.get("schema"), receipt.get("case_id"), receipt.get("seed"), receipt.get("kind")) != ("PASS", SCHEMA, case, seed, kind):
        raise RuntimeError("Null preparation identity/status differs")
    expected = set(_DATA_FILES) | {"prepared.json", "level_prepared.json"}
    actual = [Path(b["path"]).resolve() for b in receipt["bindings"]]
    if len(actual) != len(expected) or set(actual) != {folder / name for name in expected}:
        raise RuntimeError("Null preparation file inventory differs")
    if receipt["source_bindings"] != _source_bindings():
        raise RuntimeError("Null recipe source changed; preserve the existing dataset")
    for record in receipt["bindings"]:
        if _binding(record["path"]) != record:
            raise RuntimeError(f"Null preparation file changed: {record['path']}")
    return dict(status="PASS", case_id=case, folder=str(folder), resumed=True,
                receipt=_binding(folder / "null_prepared.json"), bindings=receipt["bindings"])


_DATA_FILES = ("raw_source.npy", "Raw.npy", "input_source.npy", "Input.npy", "level_source.npy", "level.npy",
               "metadata.json", "experts.json", "active.json")


def build_null_dataset(root, seed, kind):
    """Atomically create the follow-up-compatible dataset; verify on resumption.

    ``root`` is the new campaign root, not its datasets directory. Array files
    are float32, (time,y,x); all source frames, including warmup, are retained.
    A failed build never installs a partial final directory. This helper does
    not read truth files, calibrate, emit candidates, join truth, or claim audit
    completion. Only the returned preparation receipt has status PASS.
    """
    from .control_study import conditioned_frames

    seed, kind = _condition(seed, kind)
    root = Path(root).resolve()
    if (root / "completion_manifest.json").exists():
        raise RuntimeError("Completed study roots are immutable")
    case = null_case_id(seed, kind)
    datasets = root / "datasets"
    folder = datasets / case
    if folder.exists():
        return _resume(folder, case, seed, kind)
    datasets.mkdir(parents=True, exist_ok=True)
    meta = _metadata(seed, kind)
    sources = _source_bindings()
    with tempfile.TemporaryDirectory(prefix=f".{case}.", dir=datasets) as work:
        staging = Path(work)
        def array(name, shape):
            return np.lib.format.open_memmap(staging / name, mode="w+", dtype=np.float32, shape=shape)
        def bind(name):
            return _binding(staging / name, folder / name)
        raw = array("raw_source.npy", tuple(meta["shape_tyx"]))
        count = 0
        for i, frame in enumerate(iter_null_frames(seed, kind)):
            if frame.shape != raw.shape[1:] or frame.dtype != np.float32 or not np.isfinite(frame).all():
                raise RuntimeError("Null generator emitted an invalid raw frame")
            raw[i] = frame
            count += 1
        if count != raw.shape[0]:
            raise RuntimeError("Null generator frame count differs from declared geometry")
        raw.flush()
        y0, x0, y1, x1 = meta["evaluation_box_yxyx"]
        np.save(staging / "Raw.npy", raw[:,y0:y1,x0:x1])
        level = array("level_source.npy", raw.shape)
        difference = array("input_source.npy", raw.shape)
        for i, (current, delta) in enumerate(conditioned_frames(raw, 1., .4)):
            level[i], difference[i] = current, delta
        level.flush(); difference.flush()
        np.save(staging / "level.npy", level[:,y0:y1,x0:x1])
        np.save(staging / "Input.npy", difference[:,y0:y1,x0:x1])
        del raw, level, difference
        _write(staging / "metadata.json", meta)
        _write(staging / "experts.json", [])
        _write(staging / "active.json", [])
        _write(staging / "prepared.json", dict(status="PASS", source=bind("raw_source.npy"),
            input=bind("input_source.npy"), metadata=bind("metadata.json"), experts=bind("experts.json"), active=bind("active.json")))
        _write(staging / "level_prepared.json", dict(status="PASS", bindings=[bind("level_source.npy"), bind("level.npy")]))
        _write(staging / "null_prepared.json", dict(status="PASS", schema=SCHEMA, case_id=case, seed=seed, kind=kind,
            source_bindings=sources, bindings=[bind(name) for name in (*_DATA_FILES, "prepared.json", "level_prepared.json")],
            preparation_scope="Generated synthetic null, conditioned inputs and empty truth created; no labels read or outcomes evaluated.",
            setup_pairing="Exact common innovations and stationary warmup/setup recipe; production parent verifies against frozen paired crowding bytes."))
        if folder.exists():
            raise FileExistsError(f"Preserve concurrently created dataset: {folder}")
        staging.rename(folder)
    receipt = json.loads((folder / "null_prepared.json").read_text())
    return dict(status="PASS", case_id=case, folder=str(folder), resumed=False,
                receipt=_binding(folder / "null_prepared.json"), bindings=receipt["bindings"])
