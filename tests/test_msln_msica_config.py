import json
from pathlib import Path

import pytest

from neurobench.experiments.msln_msica.config import MSLNMSICAConfig, MSLNMSICAConfigError
from neurobench.experiments.msln_msica.context_bank import ordered_contexts


EXAMPLE = Path(__file__).parents[1] / "examples" / "spon_ca_burst_msln_msica_v1.example.json"


def test_standard_config_has_frozen_eight_context_order() -> None:
    config = MSLNMSICAConfig.load(EXAMPLE)
    assert [item.context_id for item in ordered_contexts(config)] == [
        "spatial_5_meanstd", "spatial_7_meanstd", "spatial_15_meanstd",
        "temporal_5_meanstd", "temporal_15_meanstd", "temporal_31_meanstd",
        "st_t15_s5_meanstd", "st_t15_s7_meanstd",
    ]


def test_config_rejects_unknown_keys_and_true_isa(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text())
    payload["unexpected"] = 1
    path = tmp_path / "bad.json"; path.write_text(json.dumps(payload))
    with pytest.raises(MSLNMSICAConfigError, match="Unknown top-level"):
        MSLNMSICAConfig.load(path)
    payload.pop("unexpected")
    payload["cross_context"]["true_isa_enabled"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(MSLNMSICAConfigError, match="true ISA"):
        MSLNMSICAConfig.load(path)
