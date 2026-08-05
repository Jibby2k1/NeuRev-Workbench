import json
from pathlib import Path
import numpy as np

from neurobench.algorithms.multilag_msica import project_temporal_fit, project_temporal_fit_chunked
from neurobench.experiments.msln_msica.multilag_program import (
    _fit_from_dict,
    _load,
    _parameter_grid,
    _protected_from_proposals,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "examples" / "spon_ca_burst_multilag_msica_v5.example.json"


def test_v5_manifest_and_factorial_counts() -> None:
    config = _load(CONFIG)
    count = sum(len(_parameter_grid(spec)) for spec in config["design"]["objective_grid"].values())
    assert count == 33
    assert len(config["design"]["multilag_profiles"]) == 3
    assert len(config["design"]["embedding_profiles"]) == 3
    assert config["evaluation"]["selection_labels_used"] is False
    assert config["compute"]["workers_per_gpu"] == 1


def test_fit_json_roundtrip_and_chunked_cpu_projection() -> None:
    surface_path = ROOT / "Outputs" / "HierarchicalParzenICA" / "spon_ca_burst_multilag_msica_v5" / "stage_a" / "surface.json"
    if not surface_path.is_file():
        return
    payload = json.loads(surface_path.read_text(encoding="utf-8"))
    fit = _fit_from_dict(payload["expansion_rows"][0]["fit"])
    rng = np.random.default_rng(9)
    movie = rng.normal(size=(30, 5, 7)).astype(np.float32)
    direct = project_temporal_fit(movie, fit, backend="cpu")
    chunked = project_temporal_fit_chunked(movie, fit, backend="cpu", frame_chunk=4)
    for name in direct:
        np.testing.assert_allclose(direct[name], chunked[name], rtol=2e-5, atol=2e-5)


def test_protected_matching_keeps_unmatched_candidates_unknown() -> None:
    config = _load(CONFIG)
    proposals = {str(burst): [[10.0, 10, 10], [9.0, 100, 100]] for burst in range(1, 5)}
    labels = [
        {"burst_id": burst, "x_px": 10, "y_px": 10, "roi_identity": f"r{burst}"}
        for burst in range(1, 5)
    ]
    result = _protected_from_proposals(proposals, labels, config)
    assert result["matched_by_budget"]["20"] == 4
    assert result["unmatched_candidates_are"] == "unknown"
