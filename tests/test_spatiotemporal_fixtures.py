import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.spatiotemporal_fixtures import (
    DEFAULT_SEEDS, TEMPLATE_IDS, SceneSpec, generate_scene, scene_specs, temporal_profile,
)


def test_declared_16_scenes_share_two_seeds_and_have_stable_ids():
    specs = scene_specs()
    assert len(specs) == len({s.scene_id for s in specs}) == 16
    assert {s.template_id for s in specs} == set(TEMPLATE_IDS)
    for template in TEMPLATE_IDS:
        assert tuple(s.seed for s in specs if s.template_id == template) == DEFAULT_SEEDS


def test_determinism_halo_coordinate_binding_and_warmup_before_setup():
    spec = SceneSpec("isolated_compact_fast", 101)
    one = generate_scene(spec, shape=(40, 44), spatial_halo_px=3)
    two = generate_scene(spec, shape=(40, 44), spatial_halo_px=3)
    assert np.array_equal(one.raw, two.raw)
    assert one.event_rows == two.event_rows
    assert one.raw.shape == (464, 46, 50)
    assert one.raw.dtype == np.float32
    assert one.metadata["evaluation_box_yxyx"] == [3, 3, 43, 47]
    assert one.metadata["setup_source_frames_ui"] == list(range(65, 165))
    assert one.metadata["application_source_frames_ui"] == list(range(165, 465))
    event = one.event_rows[0]
    assert (event["y_px"], event["x_px"]) == (23.0, 25.0)
    assert event["source_start_ui"] == 185
    assert min(row["source_frame_ui"] for row in one.active_rows) > 164
    assert one.expert_rows is one.event_rows
    different = generate_scene(SceneSpec(spec.template_id, 102), shape=(40, 44), spatial_halo_px=3)
    assert not np.array_equal(one.raw[:1], different.raw[:1])


def test_repeated_roi_truth_has_two_nonoverlapping_active_windows():
    scene = generate_scene(SceneSpec("repeated", 1), shape=(40, 40), warmup_frames=7)
    a, b = scene.event_rows
    assert a["canonical_roi_id"] == b["canonical_roi_id"]
    assert a["source_stop_ui"] < b["source_start_ui"]
    assert a["burst_id"] != b["burst_id"]
    keys = {(r["canonical_roi_id"], r["source_frame_ui"]) for r in scene.active_rows}
    assert len(keys) == len(scene.active_rows)


def test_crowded_sources_share_exact_audit_extent_and_have_distinct_truth():
    scene = generate_scene(SceneSpec("crowded", 1), shape=(40, 40))
    a, b = scene.event_rows
    assert (a["burst_id"], a["source_start_ui"], a["source_stop_ui"]) == (
        b["burst_id"], b["source_start_ui"], b["source_stop_ui"])
    assert abs(a["x_px"] - b["x_px"]) == 8
    assert a["canonical_roi_id"] != b["canonical_roi_id"]
    assert len(scene.active_rows) == a["active_frame_count"] + b["active_frame_count"]


@pytest.mark.parametrize("template", TEMPLATE_IDS[-2:])
def test_nuisance_is_explicitly_non_neural_and_has_no_truth_rows(template):
    scene = generate_scene(SceneSpec(template, 1), shape=(40, 40), warmup_frames=2)
    assert scene.active_rows == scene.event_rows == []
    assert scene.metadata["truth_mode"] == "fully_synthetic"
    assert np.isfinite(scene.raw).all()
    if "motion" in template:
        assert len(scene.metadata["nuisance_parameters"]["path"]) == 300
        assert scene.metadata["nuisance_parameters"]["artifact_is_neural_truth"] is False


def test_temporal_support_and_declared_truncation_fail_closed():
    profile = temporal_profile(100, 1000)
    assert np.all(profile > 0)
    assert profile.max() == 1
    assert profile[-1] >= 0.01
    assert profile[-1] * np.exp(-0.02 / 1.0) < 0.01
    with pytest.raises(ValueError, match="truncates"):
        generate_scene(SceneSpec("isolated_broad_slow", 1), shape=(40, 40), application_frames=200)
    with pytest.raises(ValueError):
        scene_specs((1, 1))
