from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference.config import (
    GammaLSDifferenceConfig,
    GammaLSDifferenceConfigError,
)
from neurobench.experiments.gamma_ls_difference.preflight import _gamma_search_rows


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json"


def test_example_manifest_is_strict_and_portable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    config = GammaLSDifferenceConfig.load(EXAMPLE)
    assert config.experiment_id.endswith("_v1")
    assert config.source_paths["movie"].is_relative_to(REPOSITORY)
    assert config.output_root.is_relative_to(REPOSITORY)
    assert config.portable_dict()["sources"]["movie"].startswith("data://")


def test_unknown_field_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["silent_typo"] = True
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GammaLSDifferenceConfigError, match="unknown=silent_typo"):
        GammaLSDifferenceConfig.load(manifest)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:12"])
def test_cuda_device_selector_accepts_generic_or_logical_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["resources"]["device"] = device
    manifest = tmp_path / "device.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert GammaLSDifferenceConfig.load(manifest).payload["resources"]["device"] == device


@pytest.mark.parametrize("device", ["cpu", "cuda:", "cuda:-1", "cuda:01", "cuda:0:0"])
def test_non_cuda_or_malformed_device_selector_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["resources"]["device"] = device
    manifest = tmp_path / "bad_device.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GammaLSDifferenceConfigError, match="one worker on CUDA"):
        GammaLSDifferenceConfig.load(manifest)


def test_primary_search_is_radial_and_controls_are_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    rows = _gamma_search_rows(GammaLSDifferenceConfig.load(EXAMPLE))
    primary = [row for row in rows if row["eligible_primary"]]
    controls = [row for row in rows if not row["eligible_primary"]]
    assert len(primary) == 9
    assert {row["support"] for row in primary} == {"disk"}
    assert len(controls) == 2
    assert any("square" in str(row["support"]) for row in controls)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("gamma_ls_grid", "g1_shape", 4.0),
        ("gamma_ls_grid", "screen_scale_floor_percentile", 5.0),
        ("cfar", "quiet_nms_peaks_per_pseudo_burst", [0.5, 1.0]),
        ("screen", "positive_tail_quantile", 0.99),
        ("screen", "fold_selection", "pooled_across_outer_folds"),
        ("efficiency", "frame_chunks", [1, 16]),
        ("ica_search", "screen_samples", 512),
    ],
)
def test_scientific_grid_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload[section][field] = replacement
    manifest = tmp_path / f"bad_{section}_{field}.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GammaLSDifferenceConfigError):
        GammaLSDifferenceConfig.load(manifest)
