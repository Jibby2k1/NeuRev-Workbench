from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.information_source_separation.cnmf_adapter import (
    audit_caiman_backend,
    require_caiman_backend,
)
from neurobench.experiments.information_source_separation.config import (
    CASES,
    InformationSeparationConfig,
)
from neurobench.experiments.information_source_separation.semi_synthetic import (
    make_real_background_fixture,
)


def test_example_manifest_freezes_cases_methods_and_resources() -> None:
    config = InformationSeparationConfig.load(
        "examples/spon_ca_burst_information_source_separation_v1.example.json"
    )
    assert tuple(config.generated["case_ids"]) == CASES
    assert config.generated_fixture_count() == 13 * 5 * 3
    assert config.methods["multilag_sobi"]["enabled"]
    assert not config.methods["group_energy_isa"]["enabled"]
    assert config.source_video.is_absolute()


def test_caiman_backend_is_explicit_and_never_substitutes_nmf() -> None:
    audit = audit_caiman_backend("unfrozen_pending_install_authorization")
    assert audit["backend"] == "caiman"
    assert not audit["fallback_used"]
    assert not audit["fit_authorized"]
    with pytest.raises(RuntimeError, match="CaImAn"):
        require_caiman_backend("unfrozen_pending_install_authorization")


def test_real_background_injection_has_exact_truth_and_ui_contract(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    movie = rng.integers(300, 900, size=(120, 20, 22), dtype=np.uint16)
    path = tmp_path / "movie.npy"
    np.save(path, movie, allow_pickle=False)
    fixture = make_real_background_fixture(
        path,
        quiet_start_ui=1,
        quiet_end_ui=100,
        crop_origin_xy=(3, 4),
        crop_size_px=12,
        amplitude=1.0,
        seed=7,
    )
    closure = (
        fixture.observation
        - fixture.native_background
        - fixture.injected_neural_signal
    )
    assert fixture.observation.shape == (100, 12, 12)
    assert float(np.max(np.abs(closure))) < 1e-4
    assert fixture.metadata["source_frames_ui_inclusive"] == [1, 100]
    assert fixture.metadata["native_background_is_not_decomposed_truth"] is True
