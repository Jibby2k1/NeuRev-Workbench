import json
from itertools import islice

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import reference_fixtures as fixtures
from neurobench.experiments.gamma_ls_difference.followup_fixtures import generate_crowding
from neurobench.experiments.gamma_ls_difference.spatiotemporal_fixtures import SceneSpec, generate_scene
from neurobench.experiments.gamma_ls_difference.control_study import conditioned_frames


def test_matrix_is_exact_nine_and_rejects_unsupported_identities():
    specs = fixtures.null_specs()
    assert len(specs) == len({r["case_id"] for r in specs}) == 9
    assert {(r["seed"], r["kind"]) for r in specs} == {
        (s, k) for s in (20260916, 20260917, 20260918)
        for k in ("stationary", "variance_correlation", "shared_brightness_motion")}
    for seed, kind in ((True, "stationary"), (20260916., "stationary"), (1, "stationary"),
                       (20260916, "../stationary"), (20260916, None)):
        with pytest.raises(ValueError):
            fixtures.null_case_id(seed, kind)


@pytest.mark.parametrize("seed", fixtures.SEEDS)
def test_every_null_warmup_and_setup_are_byte_equal_to_frozen_crowding(seed):
    old = generate_crowding(seed)
    for kind in fixtures.KINDS:
        for index, frame in enumerate(islice(fixtures.iter_null_frames(seed, kind), 164)):
            assert np.array_equal(frame, old.raw[index])
    meta = fixtures._metadata(seed, "stationary")
    assert meta["shape_tyx"] == [464, 226, 226]
    assert meta["evaluation_box_yxyx"] == [49, 49, 177, 177]
    assert meta["setup_source_frames_ui"] == list(range(65, 165))
    assert meta["application_source_frames_ui"] == list(range(165, 465))


@pytest.mark.parametrize("kind", ("variance_correlation", "shared_brightness_motion"))
def test_application_recipe_is_byte_equal_to_frozen_generator_outside_neural_footprints(kind):
    """Compare literal nuisance arithmetic over all times, not only setup.

    The reference test alone reads freshly generated synthetic geometry to mask
    source footprints. Production null generation never calls this generator
    or consumes its truth. The mask discards source support for every frame.
    """
    original = generate_scene(SceneSpec(f"nuisance_{kind}", 20260916),
        shape=(128, 128), warmup_frames=64, setup_frames=100,
        application_frames=300, spatial_halo_px=49)
    yy, xx = np.indices((226, 226))
    outside = np.ones((226, 226), dtype=bool)
    for event in original.event_rows:
        outside &= (xx-event["x_px"])**2 + (yy-event["y_px"])**2 > (4*event["sigma_px"])**2
    assert outside.any()
    count = 0
    for index, frame in enumerate(fixtures.iter_null_frames(20260916, kind)):
        assert frame.dtype == np.float32
        assert np.array_equal(frame[outside], original.raw[index][outside])
        count += 1
    assert count == 464


def test_nuisance_onset_offsets_and_null_metadata_are_explicit():
    streams = [fixtures.iter_null_frames(20260917, kind) for kind in fixtures.KINDS]
    for index, frames in enumerate(zip(*streams)):
        if index < 164:
            assert all(np.array_equal(frames[0], frame) for frame in frames[1:])
        elif index in (164, 204, 263, 264, 324, 383, 384):
            assert all(not np.array_equal(frames[0], frame) for frame in frames[1:])
    for kind in fixtures.KINDS:
        meta = fixtures._metadata(20260917, kind)
        assert meta["event_count"] == meta["active_region_frame_count"] == 0
        assert meta["source_roles"] == [] and meta["neural_sources_generated"] is False
        assert meta["truth_mode"] == "fully_synthetic"
        assert meta["calibration_uses_truth"] is False and meta["evaluation_requires_all_candidates_sealed"]
    path = fixtures._metadata(20260917, "shared_brightness_motion")["nuisance_parameters"]["path"]
    assert len(path) == 300 and path[0]["source_frame_ui"] == 165 and path[-1]["source_frame_ui"] == 464
    assert [path[i]["shared_offset"] for i in (39, 40, 99, 100, 159, 160, 219, 220)] == [0, 12, 12, 0, 0, -8, -8, 0]


@pytest.fixture
def tiny_geometry(monkeypatch):
    """Small file-contract fixture; production public matrix has no overrides."""
    for name, value in dict(EVALUATION_SHAPE_YX=(16, 16), SPATIAL_HALO_PX=4,
                            WARMUP_FRAMES=2, SETUP_FRAMES=2, APPLICATION_FRAMES=4).items():
        monkeypatch.setattr(fixtures, name, value)


def test_prepared_files_conditioning_empty_truth_and_verified_resume(tmp_path, tiny_geometry):
    result = fixtures.build_null_dataset(tmp_path, 20260916, "stationary")
    folder = tmp_path / "datasets" / result["case_id"]
    raw = np.load(folder / "raw_source.npy")
    level = np.load(folder / "level_source.npy")
    difference = np.load(folder / "input_source.npy")
    assert raw.shape == level.shape == difference.shape == (8, 24, 24)
    for index, (expected_level, expected_difference) in enumerate(conditioned_frames(raw, 1., .4)):
        assert np.array_equal(level[index], expected_level)
        assert np.array_equal(difference[index], expected_difference)
    assert not difference[0].any()
    for source, crop in ((raw, "Raw.npy"), (level, "level.npy"), (difference, "Input.npy")):
        assert np.array_equal(source[:,4:-4,4:-4], np.load(folder / crop))
    assert json.loads((folder / "experts.json").read_text()) == []
    assert json.loads((folder / "active.json").read_text()) == []
    receipt = json.loads((folder / "null_prepared.json").read_text())
    for record in receipt["bindings"]:
        assert fixtures._binding(record["path"]) == record
    for name in ("prepared.json", "level_prepared.json"):
        assert json.loads((folder / name).read_text())["status"] == "PASS"
    again = fixtures.build_null_dataset(tmp_path, 20260916, "stationary")
    assert not result["resumed"] and again["resumed"]
    assert result["receipt"] == again["receipt"]


def test_changed_array_and_incomplete_inventory_cannot_resume(tmp_path, tiny_geometry):
    result = fixtures.build_null_dataset(tmp_path, 20260916, "variance_correlation")
    folder = tmp_path / "datasets" / result["case_id"]
    with (folder / "level.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RuntimeError, match="file changed"):
        fixtures.build_null_dataset(tmp_path, 20260916, "variance_correlation")
    receipt_path = folder / "null_prepared.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["bindings"] = receipt["bindings"][:-1]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="inventory differs"):
        fixtures.build_null_dataset(tmp_path, 20260916, "variance_correlation")


def test_recipe_changes_or_completed_roots_cannot_be_reused(tmp_path, tiny_geometry, monkeypatch):
    fixtures.build_null_dataset(tmp_path, 20260918, "stationary")
    original = fixtures._source_bindings()
    monkeypatch.setattr(fixtures, "_source_bindings", lambda: [dict(original[0], sha256="0"*64)])
    with pytest.raises(RuntimeError, match="source changed"):
        fixtures.build_null_dataset(tmp_path, 20260918, "stationary")
    (tmp_path / "completion_manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="immutable"):
        fixtures.build_null_dataset(tmp_path, 20260918, "stationary")


def test_failed_generation_never_installs_partial_dataset(tmp_path, tiny_geometry, monkeypatch):
    monkeypatch.setattr(fixtures, "iter_null_frames", lambda seed, kind: iter([np.zeros((24,24), np.float32)]))
    with pytest.raises(RuntimeError, match="frame count"):
        fixtures.build_null_dataset(tmp_path, 20260918, "stationary")
    assert list((tmp_path / "datasets").iterdir()) == []
