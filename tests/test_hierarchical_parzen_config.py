import json
from pathlib import Path

import pytest

from neurobench.experiments.hierarchical_parzen_ica.config import (
    HierarchicalParzenICAConfig,
    HierarchicalParzenConfigError,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/spon_ca_burst_hierarchical_parzen_noisy_ica.example.json"


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_example_loads_with_frozen_stage_and_evaluation_contracts() -> None:
    config = HierarchicalParzenICAConfig.load(EXAMPLE)
    assert config.schema_version == 1
    assert config.source_video.is_absolute()
    assert config.labels_tsv.is_absolute()
    assert config.output_dir.is_absolute()
    assert set(config.stage1["methods"]) == {
        "fixed_common_difference_reference",
        "adaptive_gain_common_difference",
        "batch_cs_parzen_pairwise",
        "stochastic_parzen_score_pairwise",
    }
    assert set(config.stage2["methods"]) == {
        "ordinary_parzen_ica", "noisy_parzen_ica_posterior"
    }
    assert config.evaluation["primary_match_radius_px"] == 6
    assert config.evaluation["fixed_candidates_per_burst"] == 58
    assert config.stage1["safety"]["maximum_learned_fraction"] == 0.1
    assert config.stage1["safety"]["unsafe_policy"] == "reference_fallback"
    assert config.resources.cpu_threads == 2


def test_unknown_nested_fields_are_rejected(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["stage2"]["dictionary"]["future_unbounded_option"] = True
    with pytest.raises(HierarchicalParzenConfigError, match="future_unbounded_option"):
        HierarchicalParzenICAConfig.load(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["stage1"].update({"methods": ["stochastic_parzen_score_pairwise"]}), "four declared"),
        (lambda p: p["stage2"].update({"stride_y": 24}), "overlapping"),
        (lambda p: p["evaluation"].update({"primary_match_radius_px": 8}), "six pixels"),
        (lambda p: p["evaluation"].update({"fixed_candidates_per_burst": 100}), "frozen at 58"),
        (lambda p: p["realtime"].update({"adaptation_enabled": True}), "freezes adaptation"),
        (
            lambda p: p["stage1"]["safety"].update(
                {"maximum_current_observation_coefficient": 0.8}
            ),
            "recursive safety",
        ),
    ],
)
def test_scientific_and_resource_guards_are_enforced(
    tmp_path: Path, mutate, message: str
) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    mutate(payload)
    with pytest.raises(HierarchicalParzenConfigError, match=message):
        HierarchicalParzenICAConfig.load(_write(tmp_path, payload))


def test_to_dict_is_json_serializable_and_retains_channel_controls() -> None:
    config = HierarchicalParzenICAConfig.load(EXAMPLE)
    payload = config.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    assert "noisy_parzen_ica_posterior" in encoded
    assert payload["visualization"]["write_dense_structured_artifact"] is False
    assert payload["visualization"]["write_dense_measurement_noise"] is False
