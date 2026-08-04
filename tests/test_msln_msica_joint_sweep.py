import json
from pathlib import Path

import numpy as np

from neurobench.experiments.msln_msica.joint_sweep import (
    _contexts,
    _load,
    _recall,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "spon_ca_burst_joint_msln_residual_sweep_v2.example.json"


def test_joint_sweep_has_frozen_thirty_context_grid() -> None:
    config = _load(CONFIG)
    contexts = _contexts(config)
    assert len(contexts) == 30
    assert contexts[0].context_id == "joint_s5_g1_t5_g1"
    assert contexts[-1].context_id == "joint_s15_g5_t31_g1"
    assert all(item.temporal_guard_frames == 1 for item in contexts)


def test_joint_sweep_manifest_is_visual_primary_and_resource_bounded() -> None:
    payload = json.loads(CONFIG.read_text())
    assert payload["evaluation"]["winner_basis"] == "visual_primary_recall_guardrail"
    assert payload["compute"]["workers"] == 1
    assert payload["compute"]["max_peak_ram_gb"] <= 12
    assert payload["sweep"]["ica_finalists"] < payload["sweep"]["shortlist_contexts"]


def test_recall_keeps_unmatched_candidates_unknown(tmp_path: Path) -> None:
    config = _load(CONFIG)
    config["source"]["review_interval_ui"] = [1, 8]
    config["source"]["burst_intervals_ui"] = {"1": [2, 4]}
    config["evaluation"]["candidate_budgets"] = [1]
    config["evaluation"]["nms_distance_px"] = 1
    labels = [{"burst_id": 1, "x_px": 4, "y_px": 4}]
    evidence = np.zeros((8, 11, 11), dtype=np.float32)
    evidence[1:4, 4, 4] = 10
    result = _recall(evidence, labels, config)
    assert result["matched_by_budget"]["1"] == 1
    assert result["unmatched_candidates_are"] == "unknown"
