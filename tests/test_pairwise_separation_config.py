import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from neurobench.experiments.pairwise_separation.config import PairwiseConfigError, PairwiseSeparationConfig
from neurobench.experiments.pairwise_separation.preflight import preflight


ROOT = Path(__file__).resolve().parents[1]


def test_example_is_strict_and_preserves_six_pixel_primary_radius():
    config = PairwiseSeparationConfig.load(ROOT / "examples/spon_ca_burst_pairwise_separation.example.json")
    assert config.schema_version == 1
    assert config.evaluation.primary_match_radius_px == 6
    raw = json.loads((ROOT / "examples/spon_ca_burst_pairwise_separation.example.json").read_text())
    raw["evaluation"]["primary_match_radius_px"] = 8
    with pytest.raises(PairwiseConfigError, match="six pixels"):
        path = ROOT / "tests" / ".invalid_pairwise.json"
        try:
            path.write_text(json.dumps(raw)); PairwiseSeparationConfig.load(path)
        finally:
            path.unlink(missing_ok=True)


def test_unknown_manifest_field_is_rejected(tmp_path):
    raw = json.loads((ROOT / "examples/spon_ca_burst_pairwise_separation.example.json").read_text())
    raw["preprocessing"]["future_leak"] = True
    path = tmp_path / "bad.json"; path.write_text(json.dumps(raw))
    with pytest.raises(PairwiseConfigError, match="future_leak"):
        PairwiseSeparationConfig.load(path)


def test_preflight_requires_separate_new_artifact_destination(tmp_path):
    raw = json.loads((ROOT / "examples/spon_ca_burst_pairwise_separation.example.json").read_text())
    video = np.zeros((80, 12, 14), np.uint16); video[:, 3:9, 4:10] = 100
    np.save(tmp_path / "source.npy", video); tifffile.imwrite(tmp_path / "source.tif", video, photometric="minisblack")
    (tmp_path / "labels.tsv").write_text("burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\tstop_frame_zero_exclusive\tpoint_index\troi_identity\tx_px\ty_px\trecurrence_count\n1\t61\t65\t60\t65\t1\tr1\t7\t6\t1\n")
    (tmp_path / "design.md").write_text("# design\n")
    raw.update({"source_video":"source.npy", "source_tiff":"source.tif", "labels_tsv":"labels.tsv", "design_document":"design.md", "output_dir":"run"})
    raw["frames"] = {"review_start_ui":1,"review_end_ui":80,"quiet_start_ui":1,"quiet_end_ui":60,"frame_period_ms":20}
    raw["sampling"].update({"screen_samples":16,"confirm_samples":32,"pairwise_diagnostic_frames_ui":[20,40,60]})
    raw["resources"].update({"max_ram_mib":1,"min_free_disk_mib":1,"max_output_mib":10,"kernel_block_rows":16})
    raw["methods"]["cs_parzen_ica"]["kernel_block_rows"] = 16
    manifest = tmp_path / "config.json"; manifest.write_text(json.dumps(raw))
    config = PairwiseSeparationConfig.load(manifest)
    artifact = tmp_path / "preflight"
    payload = preflight(config, artifact_dir=artifact)
    assert payload["ready"] and (artifact / "label_projection_overlay.png").is_file()
    assert not config.output_dir.exists()
    with pytest.raises(FileExistsError): preflight(config, artifact_dir=artifact)
