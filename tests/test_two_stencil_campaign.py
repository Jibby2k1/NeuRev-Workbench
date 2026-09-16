"""Small provenance/label-join gate checks; no detector or GPU execution."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobench.experiments.gamma_ls_difference import two_stencil_campaign as campaign


def _sealed_matrix(tmp_path, monkeypatch, *, missing_last=False):
    cells = tuple(SimpleNamespace(cell_id=f"cell_{index:02d}") for index in range(25))
    monkeypatch.setattr(campaign, "build_two_stencil_factorial", lambda **_: cells)
    for cell in cells[:-1] if missing_last else cells:
        campaign.write_json(tmp_path / "cells" / cell.cell_id / "scoring_complete.json", {})
    label_path = tmp_path / "expert_occurrences.tsv"
    label_path.write_text("observation_id\tcanonical_roi_id\nexample\troi_001\n")
    monkeypatch.setattr(campaign, "LABEL_PATH", label_path)
    campaign.write_json(tmp_path / "preflight.json", {"label_sha256": campaign.sha256(label_path)})
    return cells, label_path


def _forbid_label_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("labels or proposals were parsed before the provenance gate passed")

    monkeypatch.setattr(campaign, "read_tsv", forbidden)


def test_repository_root_is_the_populated_checkout():
    expected = Path(__file__).resolve().parents[1]
    assert campaign.REPO == expected
    assert (campaign.REPO / "AGENTS.md").is_file()
    assert campaign.SOURCE_ROOT.is_relative_to(expected / "Outputs")


def test_protocol_freezes_24_cells_plus_anchor_and_disjoint_score_intervals(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "source_movie": {"path": "/frozen/raw.npy", "file_sha256": "raw-hash"},
        "arrays": {
            name: {"path": f"stage_arrays/{name}.npy", "shape_tyx": [2359, 340, 573],
                   "dtype": "float32", "file_sha256": name + "-hash"}
            for name in ("conditioned_current_frame", "difference_signed")
        },
    }
    campaign.write_json(source / "stage_manifest.json", manifest)
    monkeypatch.setattr(campaign, "SOURCE_ROOT", source)
    frozen = campaign.protocol()
    assert frozen["calibration_source_ui"] == [2, 100]
    assert frozen["application_source_ui"] == [101, 2359]
    assert frozen["cold_start_ui1_excluded_from_calibration"] is True
    assert frozen["calibration_assumed_event_free"] is False
    assert frozen["source_manifest_sha256"] == campaign.sha256(source / "stage_manifest.json")
    cells = frozen["cells"]
    assert len(cells) == len({cell["cell_id"] for cell in cells}) == 25
    assert sum(cell["is_deployed_anchor"] for cell in cells) == 1
    for name, entry in frozen["inputs"].items():
        assert entry["path"] == str(source / "stage_arrays" / f"{name}.npy")
        assert entry["file_sha256"] == name + "-hash"


def test_any_unsealed_matrix_cell_blocks_the_first_label_join(tmp_path, monkeypatch):
    cells, _ = _sealed_matrix(tmp_path, monkeypatch, missing_last=True)
    _forbid_label_reads(monkeypatch)
    with pytest.raises(ValueError, match="all25cells.*sealed"):
        campaign._evaluate_cell(tmp_path, cells[0])
    assert not (tmp_path / "cells" / cells[0].cell_id / "numeric_complete.json").exists()


def test_label_mutation_after_preflight_is_rejected_before_parsing(tmp_path, monkeypatch):
    cells, label_path = _sealed_matrix(tmp_path, monkeypatch)
    label_path.write_text(label_path.read_text() + "new\troi_002\n")
    _forbid_label_reads(monkeypatch)
    with pytest.raises(ValueError, match="label source changed"):
        campaign._evaluate_cell(tmp_path, cells[0])


def test_candidate_seal_mutation_is_rejected_before_label_join(tmp_path, monkeypatch):
    cells, _ = _sealed_matrix(tmp_path, monkeypatch)
    folder = tmp_path / "cells" / cells[0].cell_id
    campaign.write_json(folder / "candidate_seal.json", {"candidate_files": []})
    campaign.write_json(folder / "scoring_complete.json", {
        "candidate_seal_sha256": campaign.sha256(folder / "candidate_seal.json")
    })
    campaign.write_json(folder / "candidate_seal.json", {"candidate_files": [], "changed": True})
    _forbid_label_reads(monkeypatch)
    with pytest.raises(ValueError, match="candidate seal changed"):
        campaign._evaluate_cell(tmp_path, cells[0])


def test_candidate_bytes_mutation_is_rejected_before_label_join(tmp_path, monkeypatch):
    cells, _ = _sealed_matrix(tmp_path, monkeypatch)
    folder = tmp_path / "cells" / cells[0].cell_id
    candidate_path = folder / "candidates.tsv"
    candidate_path.write_text("proposal_id\tscore\np1\t2.0\n")
    seal = {"candidate_files": [{
        "path": str(candidate_path.relative_to(tmp_path)),
        "sha256": campaign.sha256(candidate_path), "count": 1,
    }]}
    campaign.write_json(folder / "candidate_seal.json", seal)
    campaign.write_json(folder / "scoring_complete.json", {
        "candidate_seal_sha256": campaign.sha256(folder / "candidate_seal.json")
    })
    candidate_path.write_text("proposal_id\tscore\np1\t9.0\n")
    _forbid_label_reads(monkeypatch)
    with pytest.raises(ValueError, match="candidate table changed"):
        campaign._evaluate_cell(tmp_path, cells[0])
