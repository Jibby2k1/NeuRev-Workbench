"""Checks of complete paired reporting and shared scientific display scales."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.two_stencil_report import (
    INPUTS, METRICS, TARGETS, display_limits, factorial_contrasts,
    fixed_center_crop, normalize_results, readout_contrasts, signed_display_stage,
)


def fixture_matrix():
    cells = []
    for representation in INPUTS:
        for design in ("point", "direct", "serial"):
            for guard in (0., 7.):
                for family in ("uniform", "gamma"):
                    cells.append({
                        "cell_id": f"{representation}__{design}_{family}_g{guard:g}",
                        "input_representation": representation, "is_deployed_anchor": False,
                        "spec": {"design": design, "reference_family": family,
                                 "guard_radius_px": guard, "support_geometry": "square"},
                    })
    cells.append({"cell_id": "deployed_anchor", "input_representation": INPUTS[1],
                  "is_deployed_anchor": True,
                  "spec": {"design": "point", "reference_family": "gamma",
                           "guard_radius_px": 7., "support_geometry": "disk"}})
    protocol = {"cells": cells, "target_proposals_per_frame": list(TARGETS)}
    rows = []
    for index, cell in enumerate(cells):
        spec = cell["spec"]
        controls = cell["is_deployed_anchor"] or (
            spec["design"] in {"direct", "serial"} and spec["reference_family"] == "gamma"
            and spec["guard_radius_px"] == 0)
        for readout in (("Z", "A", "contrast") if controls else ("Z",)):
            for q in TARGETS:
                matches = (10 + {"point": 0, "direct": 2, "serial": 5}[spec["design"]]
                           + int(spec["reference_family"] == "gamma")
                           + int(spec["guard_radius_px"] == 7)
                           + {"Z": 0, "A": -2, "contrast": -1}[readout])
                count = 100 + 3 * index + int(q * 4)
                rows.append({
                    "operator_cell_id": cell["cell_id"], "input_representation": cell["input_representation"],
                    **spec, "is_deployed_anchor": cell["is_deployed_anchor"],
                    "readout": readout, "target_proposals_per_frame": q,
                    "known_positive_count": 79, "application_score_frame_count": 2259,
                    METRICS[0]: matches / 79, "matched_known_positive_count": matches,
                    "emitted_frame_proposal_count": count, "application_proposals_per_frame": count / 2259,
                    "spatial_representative_count_in_declared_windows": 40,
                })
    return protocol, rows


def test_all_fixed_factorial_pairs_hold_other_factors_and_exclude_disk_anchor():
    protocol, raw = fixture_matrix()
    normalized = normalize_results(list(reversed(raw)), protocol)
    assert len(normalized) == 175
    contrasts = factorial_contrasts(normalized)
    assert len(contrasts) == 200
    assert all("deployed_anchor" not in (row["left_cell_id"], row["right_cell_id"]) for row in contrasts)
    expected = {"direct_minus_point": 2, "serial_minus_direct": 3,
                "gamma_minus_uniform": 1, "guard7_minus_guard0": 1}
    for row in contrasts:
        assert row["delta_" + METRICS[0]] == pytest.approx(expected[row["comparison"]] / 79)
        assert row["left_readout"] == row["right_readout"] == "Z"
        assert row["difference_direction"] == "left_minus_right"


def test_readout_contrasts_include_all_five_predeclared_cells_and_targets():
    protocol, raw = fixture_matrix()
    normalized = normalize_results(raw, protocol)
    controls = readout_contrasts(normalized)
    assert len(controls) == 75
    assert len({row["left_cell_id"] for row in controls}) == 5
    assert "deployed_anchor" in {row["left_cell_id"] for row in controls}
    expected = {"contrast_minus_A": 1, "Z_minus_contrast": 1, "Z_minus_A": 2}
    for row in controls:
        assert row["left_cell_id"] == row["right_cell_id"]
        assert row["delta_" + METRICS[0]] == pytest.approx(expected[row["comparison"]] / 79)


def test_partial_duplicate_and_nonreconciling_results_are_rejected():
    protocol, rows = fixture_matrix()
    with pytest.raises(ValueError, match="all175"):
        normalize_results(rows[:-1], protocol)
    with pytest.raises(ValueError, match="duplicate"):
        normalize_results(rows + [rows[0]], protocol)
    broken = deepcopy(rows)
    broken[0][METRICS[0]] = .99
    with pytest.raises(ValueError, match="matched/79"):
        normalize_results(broken, protocol)
    broken = deepcopy(rows)
    broken[0]["support_geometry"] = "disk"
    with pytest.raises(ValueError, match="protocol"):
        normalize_results(broken, protocol)


def test_display_limits_pool_noncore_cells_and_keep_signed_zero_centered(tmp_path):
    raw = np.full((1, 4, 5), 100., dtype=np.float32)
    np.save(tmp_path / "raw.npy", raw)
    inputs = {}
    for representation, value in zip(INPUTS, (10., -1.)):
        path = tmp_path / f"{representation}.npy"
        np.save(path, np.full_like(raw, value))
        inputs[representation] = {"path": str(path)}
    cells = [
        {"cell_id": "current_core", "input_representation": INPUTS[0]},
        {"cell_id": "current_noncore", "input_representation": INPUTS[0]},
        {"cell_id": "difference", "input_representation": INPUTS[1]},
    ]
    for cell, value in zip(cells, (2., 20., -3.)):
        folder = tmp_path / "cells" / cell["cell_id"]
        folder.mkdir(parents=True)
        stages = {stage: np.full_like(raw, value) for stage in ("A", "M", "contrast", "Z")}
        stages["sigma"] = np.full_like(raw, 1.)
        np.savez(folder / "stage_snapshots.npz", source_frames_ui=[1], **stages)
    protocol = {"source_movie": {"path": str(tmp_path / "raw.npy")},
                "inputs": inputs, "cells": cells, "snapshot_source_ui": [1]}
    result = display_limits(tmp_path, protocol)
    current_a = result["limits"][INPUTS[0]]["A"]
    assert current_a["vmax"] == 20  # all arms, including a noncore arm, set the common scale
    assert current_a["cell_count"] == 2
    difference_a = result["limits"][INPUTS[1]]["A"]
    assert difference_a["vmin"] == -3 and difference_a["vmax"] == 3
    assert result["fullfield_and_crop_share_identical_limits"]
    assert not result["annotation_coordinates_used_for_display_selection"]
    assert signed_display_stage(INPUTS[0], "Z")
    assert not signed_display_stage(INPUTS[1], "sigma")
    assert fixed_center_crop(340, 573) == (222, 350, 106, 234)
