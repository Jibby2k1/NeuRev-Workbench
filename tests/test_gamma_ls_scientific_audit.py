from __future__ import annotations

import json

import numpy as np

from neurobench.experiments.gamma_ls_difference import scientific_audit as audit


def test_fixed_stage_crop_npz_uses_exact_source_and_aligned_previous(tmp_path) -> None:
    raw = np.arange(8 * 7 * 9, dtype=np.uint16).reshape(8, 7, 9)
    conditioned = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    difference = np.zeros_like(conditioned)
    difference[1:] = conditioned[1:] - conditioned[:-1]
    gamma = difference * 0.5
    occupancy = np.full((7, 9), 0.25, dtype=np.float32)
    candidate = {
        "candidate_id": "label_free_rank_1",
        "burst_id": 1,
        "peak_source_frame_ui": 4,
        "x_px": 4,
        "y_px": 3,
    }
    npz_path = tmp_path / "stage.npz"
    metadata_path = tmp_path / "stage.json"

    metadata = audit._write_fixed_stage_crop_npz(
        npz_path,
        metadata_path,
        tmp_path / "stage_preview.png",
        raw_movie=raw,
        conditioned=conditioned,
        difference=difference,
        gamma=gamma,
        occupancy=occupancy,
        candidate=candidate,
        review_start_ui=2,
        source_movie_sha256="frozen-source",
        threshold_z=5.0,
        nms_distance_px=6,
        crop_radius_px=2,
    )

    with np.load(npz_path, allow_pickle=False) as packet:
        assert set(packet.files) == {
            "raw",
            "conditioned_previous",
            "conditioned_current",
            "difference",
            "gamma_ls",
            "threshold_exceedance",
            "proposals",
            "burst_occupancy",
        }
        np.testing.assert_array_equal(packet["raw"], raw[3, 1:6, 2:7])
        np.testing.assert_allclose(
            packet["conditioned_current"] - packet["conditioned_previous"],
            packet["difference"],
        )
    assert metadata["candidate_id"] == "label_free_rank_1"
    assert metadata["source_frame_ui"] == 4
    assert metadata["raw_provenance"].startswith("exact uint16 acquired")
    assert metadata["framewise_diagnostic"]["nms_distance_px"] == 6
    assert metadata["framewise_diagnostic"]["threshold_z"] == 5.0
    assert metadata["framewise_diagnostic"]["threshold_calibration_unit"] == (
        "nms_peaks_per_duration_matched_pseudo_burst"
    )
    assert metadata["burst_occupancy_semantics"].startswith("sealed fraction")
    assert (tmp_path / "stage_preview.png").is_file()
    assert json.loads(metadata_path.read_text())["source_movie_sha256"] == (
        "frozen-source"
    )


def test_truthy_handles_tsv_boolean_values() -> None:
    assert audit._truthy(True)
    assert audit._truthy("True")
    assert not audit._truthy(False)
    assert not audit._truthy("False")
