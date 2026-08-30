from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

from neurobench.experiments.neuron_identifiability.jepa_evaluation import (
    hierarchical_strongest_comparator_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "NREV-EXP-0028"
RUN_ID = "NREV-RUN-EXP-0028-SCREEN-20260829-B"
PROVENANCE = ROOT / "research" / "run-provenance" / RUN_ID


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_screen_run_is_engineering_success_not_scientific_completion() -> None:
    experiment = _yaml(
        ROOT / "research" / "registry" / "experiments" / f"{EXPERIMENT_ID}.yaml"
    )
    decision = _yaml(ROOT / "research" / "registry" / "decisions" / "NREV-DEC-0020.yaml")
    run = _yaml(ROOT / "research" / "registry" / "runs" / f"{RUN_ID}.yaml")

    assert run["lifecycle"] == "succeeded"
    assert run["validation"]["preflight"] == "passed"
    assert run["validation"]["execution"] == "passed"
    assert run["validation"]["outputs"] == "partial"

    check_rows = {row["id"]: row for row in run["validation"]["checks"]}
    checks = {key: row["status"] for key, row in check_rows.items()}
    assert checks["engineering_completion"] == "passed"
    assert checks["primary_gain"] == "partial"
    assert checks["collapse_gate"] == "partial"
    assert checks["scientific_audit"] == "partial"
    assert checks["scientific_promotion"] == "passed"
    assert "does not formally resolve" in check_rows["primary_gain"]["summary"]
    assert "complete cross-seed gate remains unresolved" in check_rows[
        "collapse_gate"
    ]["summary"]

    assert experiment["lifecycle"] == "draft"
    assert experiment["outcome"] == "not_evaluated"
    assert experiment["evidence_tier"] == "none"
    assert experiment["claim_ids"] == []
    assert "evidence_capsule_id" not in experiment
    assert RUN_ID in experiment["run_ids"]
    assert decision["lifecycle"] == "draft"
    assert decision["action"] == "hold"
    assert decision["claim_ids"] == []
    assert decision["evidence_capsule_ids"] == []
    assert not (ROOT / "research" / "evidence" / f"{EXPERIMENT_ID}.json").exists()


def test_screen_provenance_hashes_match_committed_exact_copies() -> None:
    run = _yaml(ROOT / "research" / "registry" / "runs" / f"{RUN_ID}.yaml")
    resolved = PROVENANCE / "resolved_config.json"
    artifact_index_path = PROVENANCE / "artifact_index.json"

    assert _sha256(resolved) == run["configuration"]["sha256"]
    assert _sha256(artifact_index_path) == run["artifacts"]["manifest_sha256"]

    resolved_payload = json.loads(resolved.read_text(encoding="utf-8"))
    assert resolved_payload["registered_config_sha256"] == _sha256(
        ROOT / "examples" / "spatiotemporal_jepa_representation_v1.example.json"
    )
    assert resolved_payload["execution"]["mode"] == "screen"
    assert resolved_payload["execution"]["scientific_execution"] is False
    assert resolved_payload["execution"]["overrides"]["max_training_seeds"] == 1
    assert resolved_payload["execution"]["overrides"]["steps"] == 500
    assert resolved_payload["execution"]["overrides"]["verify_hashes"] is True

    artifact_index = json.loads(artifact_index_path.read_text(encoding="utf-8"))
    assert len(artifact_index["artifacts"]) == 25
    assert sum(row["bytes"] for row in artifact_index["artifacts"]) == 389_292_601
    indexed = {row["path"]: row for row in artifact_index["artifacts"]}
    for name in (
        "summary.json",
        "training_metrics.json",
        "validation.json",
        "nuisance_audit.json",
        "paired_injection_results.tsv",
    ):
        copy = PROVENANCE / name
        assert _sha256(copy) == indexed[name]["sha256"]
        assert copy.stat().st_size == indexed[name]["bytes"]


def test_screen_result_remains_outside_supported_claim_ledger() -> None:
    ledger = (ROOT / "research" / "generated" / "CLAIM_LEDGER.md").read_text(
        encoding="utf-8"
    )
    result = (
        ROOT / "docs" / "research" / "SPATIOTEMPORAL_JEPA_SCREEN_V1_RESULTS.md"
    ).read_text(encoding="utf-8")

    assert EXPERIMENT_ID not in ledger
    assert "1.85%" in result
    assert "18.75%" in result
    assert "-31.71 to -3.70 percentage points" in result
    assert "not escalate JEPA v1 automatically" in result


def test_registered_5000_draw_bootstrap_recheck_preserves_negative_result() -> None:
    primary = "compact_jepa_latent_temporal_change"
    comparators = (
        "masked_pixel_autoencoder_latent_temporal_change",
        "frozen_random_encoder_latent_temporal_change",
        "frozen_handcrafted_stack",
    )
    methods = (primary, *comparators)
    grouped: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with (PROVENANCE / "paired_injection_results.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["training_seed"]) != 1001 or row["method"] not in methods:
                continue
            key = (
                row["background_recording_id"],
                row["background_window_id"],
                int(row["injection_seed"]),
            )
            grouped[key][row["method"]].append(float(row["source_on_recall"]))

    cluster_rows = []
    for (recording, window, injection_seed), values in sorted(grouped.items()):
        assert set(values) == set(methods)
        assert all(len(values[method]) == 3 for method in methods)
        cluster_rows.append(
            {
                "background_recording_id": recording,
                "background_window_id": window,
                "injection_seed": injection_seed,
                "method_values": {
                    method: sum(values[method]) / len(values[method])
                    for method in methods
                },
            }
        )

    assert len(cluster_rows) == 36
    result = hierarchical_strongest_comparator_bootstrap(
        cluster_rows,
        primary_method=primary,
        comparator_methods=comparators,
        draws=5_000,
        seed=5_102,
    )
    assert result["observed_strongest_comparator"] == "frozen_handcrafted_stack"
    assert result["observed_mean"] == -0.16898148148148148
    assert result["confidence_interval_95"] == [
        -0.31712962962962965,
        -0.037037037037037035,
    ]
    assert result["bootstrap_strongest_selection_fraction"] == {
        "masked_pixel_autoencoder_latent_temporal_change": 0.0024,
        "frozen_random_encoder_latent_temporal_change": 0.0002,
        "frozen_handcrafted_stack": 0.9974,
    }
