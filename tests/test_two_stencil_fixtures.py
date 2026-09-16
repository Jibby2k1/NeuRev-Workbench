from __future__ import annotations

import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import two_stencil_fixtures as fixtures


def test_fixtures_are_deterministic_static_scenes_with_a_signed_pair():
    first = fixtures.build_fixture_scenes()
    second = fixtures.build_fixture_scenes()
    changed_seed = fixtures.build_fixture_scenes(seed=fixtures.SEED + 1)
    assert len(first) == len({scene.scene_id for scene in first}) == 8
    for a, b, c in zip(first, second, changed_seed):
        assert a.values.shape == (64, 64)
        assert np.isfinite(a.values).all()
        np.testing.assert_array_equal(a.values, b.values)
        if a.scene_id == "noise":
            assert not np.array_equal(a.values, c.values)
        else:
            np.testing.assert_array_equal(a.values, c.values)
    scenes = {scene.scene_id: scene for scene in first}
    np.testing.assert_allclose(scenes["compact_positive"].values + scenes["compact_negative"].values,
                               2 * scenes["constant"].values, rtol=0, atol=1e-15)
    assert scenes["near_boundary"].components[0]["x_px"] == 2
    assert len(scenes["two_nearby"].components) == 2
    assert not scenes["gradient"].components and not scenes["noise"].components


def test_fixture_matrix_declares_stabilizer_without_a_detection_readout():
    cells = fixtures.fixture_cells()
    assert len(cells) == len({name for name, _ in cells}) == 6
    assert {spec.design for _, spec in cells} == {"point", "direct", "serial"}
    assert {spec.guard_radius_px for _, spec in cells} == {0, 7}
    assert all(spec.scale_floor == .1 and spec.support_geometry == "square" for _, spec in cells)
    assert len(fixtures.fixture_cells(include_uniform=True)) == 12


def test_small_numerical_packet_is_bound_and_does_not_claim_biological_accuracy(tmp_path, monkeypatch):
    # One genuine operator arm is enough to test packet generation; the operator
    # module separately tests the entire matrix against independent pixel sums.
    cell = fixtures.fixture_cells()[0]
    monkeypatch.setattr(fixtures, "fixture_cells", lambda **_: (cell,))
    root = tmp_path / "fixtures"
    report = fixtures.run_fixture_diagnostics(root, render=False, size=32)
    assert report["status"] == "NUMERICAL_DIAGNOSTICS_PASS"
    assert report["constant_field_contrast_pass"] is True
    assert report["metric_row_count"] == 8
    assert report["scientific_promotion"] is False
    assert report["biological_accuracy_claimed"] is False
    assert report["visual_qa"] == "not_rendered"
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["batch_axis"] == "independent_static_scenes_not_time"
    assert protocol["thresholding_or_nms_applied"] is False
    assert protocol["scale_floor"] == .1
    for artifact in report["artifacts"]:
        path = root / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert fixtures._sha256(path) == artifact["sha256"]
    with np.load(root / f"stages_{cell[0]}.npz", allow_pickle=False) as packet:
        assert set(packet.files) == {*fixtures.STAGES, "target_kernel", "reference_kernel"}
        assert packet["Z"].shape == (8, 32, 32)
        assert packet["Z"].dtype == np.float64
        assert np.isfinite(packet["Z"]).all()
    centers = json.loads((root / "constructed_center_metrics.json").read_text())
    assert len(centers) == 6
    for scene_id, sign in (("compact_positive", 1), ("compact_negative", -1)):
        row = next(row for row in centers if row["scene_id"] == scene_id)
        assert row["observed_Z_sign_at_center"] == sign
    with pytest.raises(FileExistsError):
        fixtures.run_fixture_diagnostics(root, render=False, size=32)


@pytest.mark.parametrize("kwargs", ({"size": 16}, {"size": True}, {"seed": -1}, {"seed": 1.5}))
def test_invalid_fixture_design_is_rejected(kwargs):
    with pytest.raises(ValueError):
        fixtures.build_fixture_scenes(**kwargs)
