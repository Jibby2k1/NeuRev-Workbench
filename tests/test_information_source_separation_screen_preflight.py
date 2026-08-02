from pathlib import Path

from neurobench.experiments.information_source_separation.config import (
    InformationSeparationConfig,
)
from neurobench.experiments.information_source_separation.screen_preflight import (
    audit_generated_screen,
)


def test_screen_preflight_is_read_only_and_never_self_authorizes(tmp_path: Path) -> None:
    config = InformationSeparationConfig.load(
        "examples/spon_ca_burst_information_source_separation_v1.example.json"
    )
    target = tmp_path / "screen"
    audit = audit_generated_screen(config, output_dir=target)
    assert audit["counts"]["screen_fit_count"] == 672
    assert audit["ready_for_explicit_user_selection"]
    assert not audit["run_authorized"]
    assert not target.exists()
