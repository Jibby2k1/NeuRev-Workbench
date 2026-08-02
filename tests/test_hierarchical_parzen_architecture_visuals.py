import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.hierarchical_parzen_ica.architecture_config import (
    ArchitectureVisualConfig,
    ArchitectureVisualConfigError,
)
from neurobench.experiments.hierarchical_parzen_ica.architecture_visuals import (
    preflight,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (
    ROOT
    / "examples/spon_ca_burst_stage1_architecture_visuals.example.json"
)


def test_architecture_visual_example_loads() -> None:
    config = ArchitectureVisualConfig.load(EXAMPLE)
    assert config.schema_version == 1
    assert config.resources["device"] == "cpu"
    assert config.stochastic["safety"]["maximum_learned_fraction"] == 1.0
    assert config.architectures["ids"] == [
        "teacher_forced_stochastic",
        "raw_stochastic_recurrence",
        "quiet_fixed_point_recurrence",
        "reference_parzen_innovation",
    ]


def test_unknown_architecture_visual_field_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["architectures"]["unbounded_future_control"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ArchitectureVisualConfigError,
        match="unbounded_future_control",
    ):
        ArchitectureVisualConfig.load(path)


def test_architecture_visual_preflight_is_read_only_and_collision_safe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "video.npy"
    np.save(source, np.ones((12, 8, 9), dtype=np.uint16))
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["source_video"] = "video.npy"
    payload["output_dir"] = "outputs/run"
    payload["frames"].update(
        {
            "review_start_ui": 1,
            "review_end_ui": 12,
            "quiet_start_ui": 1,
            "quiet_end_ui": 6,
        }
    )
    (tmp_path / "outputs").mkdir()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = ArchitectureVisualConfig.load(path)
    result = preflight(config)
    assert result["ready"] is True
    assert result["tiff_count"] == 9
    assert result["resolved_output_frames"] == 11
    assert not config.output_dir.exists()

    config.output_dir.mkdir()
    collided = preflight(config)
    assert collided["ready"] is False
    assert collided["gates"]["output_absent"] is False
