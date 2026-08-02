from pathlib import Path

from neurobench.experiments.hierarchical_parzen_ica.denoising_program_config import (
    DenoisingProgramConfig,
    FAMILY_IDS,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/spon_ca_burst_advanced_denoising_program.example.json"


def test_program_design_has_frozen_family_and_combination_counts() -> None:
    config = DenoisingProgramConfig.load(EXAMPLE)
    assert tuple(config.designs()) == FAMILY_IDS
    assert config.breadth_combination_count == 69
    assert config.full_field_combination_count == 20
    assert config.finalist_count == 10
    assert config.confirmation_evaluation_count == 120


def test_design_rows_have_stable_family_local_indices() -> None:
    config = DenoisingProgramConfig.load(EXAMPLE)
    for family, rows in config.designs().items():
        assert [row["variant_index"] for row in rows] == list(
            range(1, len(rows) + 1)
        )
        assert all(row["family_id"] == family for row in rows)
