from pathlib import Path

from neurobench.experiments.hierarchical_parzen_ica.innovation_denoising_config import (
    FAMILY_IDS,
    InnovationDenoisingConfig,
)
from neurobench.experiments.hierarchical_parzen_ica.innovation_denoising_program import (
    _mixture_specs,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/spon_ca_burst_innovation_denoising_v3.example.json"


def test_v3_frozen_design_counts_and_family_order() -> None:
    config = InnovationDenoisingConfig.load(EXAMPLE)
    assert tuple(config.designs()) == FAMILY_IDS
    assert config.breadth_combination_count == 96
    assert config.full_field_combination_count == 16
    assert config.family_finalist_count == 8
    assert config.mixture_combination_count == 8
    assert config.maximum_confirmation_refit_count == 6
    assert config.tiff_finalist_count == 10


def test_v3_mixture_design_has_six_pairs_and_two_four_way_settings() -> None:
    config = InnovationDenoisingConfig.load(EXAMPLE)
    sources = [
        {"variant_id": f"source_{index}"}
        for index in range(int(config.mixture["pareto_source_count"]))
    ]
    specs = _mixture_specs(sources, config)
    assert len(specs) == 8
    assert [len(row["source_indices"]) for row in specs] == [2] * 6 + [4] * 2
    assert all(sum(row["weights"]) <= 1 for row in specs)
