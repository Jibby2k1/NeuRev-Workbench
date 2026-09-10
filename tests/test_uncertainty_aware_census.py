from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from neurobench.algorithms.proposal_ranking import CandidateTable
from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_config import (
    InnovationRankerConfig,
)
from neurobench.experiments.neuron_identifiability import uncertainty_aware_census as census


ROOT = Path(__file__).resolve().parents[1]


def _table(
    positions: list[list[int]],
    features: list[list[float]],
    source_count: list[int],
) -> CandidateTable:
    return CandidateTable(
        positions=np.asarray(positions, dtype=np.int32).reshape(-1, 2),
        features=np.asarray(features, dtype=np.float32).reshape(len(positions), 2),
        source_count=np.asarray(source_count, dtype=np.int16),
    )


def test_candidate_table_rows_are_stable_flat_and_unlabeled() -> None:
    table = _table([[5, 7], [2, 3]], [[0.25, 0.5], [0.75, 1.0]], [2, 1])

    first = census.candidate_table_rows(
        table,
        partition="event",
        partition_id=2,
        feature_ids=("alpha", "beta"),
    )
    second = census.candidate_table_rows(
        table,
        partition="event",
        partition_id=2,
        feature_ids=("alpha", "beta"),
    )

    assert first == second
    assert first[0] == {
        "candidate_id": "irv5_event_p02_x00005_y00007",
        "partition": "event",
        "partition_id": 2,
        "proposal_order": 1,
        "burst_id": 2,
        "quiet_map_id": None,
        "x_px": 5,
        "y_px": 7,
        "source_count": 2,
        "feature__alpha": 0.25,
        "feature__beta": 0.5,
    }
    assert all("label" not in key for row in first for key in row)


def test_candidate_rows_fail_closed_on_schema_nonfinite_and_duplicate_position() -> None:
    table = _table([[1, 2]], [[0.1, 0.2]], [1])
    with pytest.raises(census.CensusContractError, match="feature schema"):
        census.candidate_table_rows(
            table,
            partition="event",
            partition_id=1,
            feature_ids=("only_one",),
        )

    table.features[0, 0] = np.nan
    with pytest.raises(census.CensusContractError, match="finite numeric"):
        census.candidate_table_rows(
            table,
            partition="event",
            partition_id=1,
            feature_ids=("alpha", "beta"),
        )

    duplicate = _table(
        [[1, 2], [1, 2]], [[0.1, 0.2], [0.3, 0.4]], [1, 2]
    )
    with pytest.raises(census.CensusContractError, match="duplicate coordinates"):
        census.candidate_table_rows(
            duplicate,
            partition="quiet",
            partition_id=1,
            feature_ids=("alpha", "beta"),
        )


def test_materialization_keeps_partitions_separate_and_enforces_counts() -> None:
    two = _table([[1, 2], [3, 4]], [[0.1, 0.2], [0.3, 0.4]], [1, 2])
    one = _table([[5, 6]], [[0.5, 0.6]], [1])
    result = census.materialize_candidate_rows(
        quiet_tables=(one, two),
        event_tables={1: two, 2: one},
        feature_ids=("alpha", "beta"),
        expected_event_counts={1: 2, 2: 1},
        expected_quiet_counts=(1, 2),
    )

    assert result["counts"] == {
        "event": {"1": 2, "2": 1},
        "quiet": [1, 2],
        "event_total": 3,
        "quiet_total": 3,
    }
    assert {row["burst_id"] for row in result["event_rows"]} == {1, 2}
    assert all(row["quiet_map_id"] is None for row in result["event_rows"])
    assert all(row["burst_id"] is None for row in result["quiet_rows"])

    with pytest.raises(census.CensusContractError, match="counts differ"):
        census.materialize_candidate_rows(
            quiet_tables=(one, two),
            event_tables={1: two, 2: one},
            feature_ids=("alpha", "beta"),
            expected_event_counts={1: 2, 2: 2},
            expected_quiet_counts=(1, 2),
        )


def test_feature_dictionary_records_full_v5_schema_and_unknown_semantics() -> None:
    value = census.census_feature_dictionary()
    assert len(value["feature_order"]) == 34
    assert len(value["feature_columns"]) == 34
    assert value["candidate_label_state"] == "unlabeled"
    assert "not verified non-neurons" in value["prohibited_interpretation"]
    assert value["feature_dictionary_sha256"] == census.stable_hash(
        {key: item for key, item in value.items() if key != "feature_dictionary_sha256"}
    )


def test_hash_verifier_reports_exact_bytes_and_rejects_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen-input")
    expected = hashlib.sha256(b"frozen-input").hexdigest()

    result = census.hash_and_verify_files({"source": source}, {"source": expected})
    assert result == {
        "source": {"bytes": len(b"frozen-input"), "sha256": expected}
    }

    source.write_bytes(b"changed")
    with pytest.raises(census.CensusContractError, match="hash differs"):
        census.hash_and_verify_files({"source": source}, {"source": expected})


def test_checked_in_v5_contract_and_scientific_primitive_hashes_are_exact() -> None:
    config = InnovationRankerConfig.load(
        ROOT / "examples/spon_ca_burst_innovation_ranker_v5.example.json"
    )
    census.validate_frozen_contract(config)
    assert len(census.EXPECTED_PROPOSAL_SOURCE_IDS) == 22

    changed = replace(
        config,
        proposals={**config.proposals, "dedupe_radius_px": 4.0},
    )
    with pytest.raises(census.CensusContractError, match="dedupe_radius_px"):
        census.validate_frozen_contract(changed)

    files = {
        role: ROOT / relative
        for role, relative in census.CODE_RELATIVE_PATHS.items()
    }
    result = census.hash_and_verify_files(files, census.EXPECTED_CODE_SHA256)
    assert set(result) == set(census.EXPECTED_CODE_SHA256)


def test_preflight_is_read_only_and_returns_portable_exact_hash_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    data = tmp_path / "data"
    repository.mkdir()
    data.mkdir()
    (repository / "AGENTS.md").write_text("# fixture\n", encoding="utf-8")
    materializer = (
        repository
        / "neurobench"
        / "experiments"
        / "neuron_identifiability"
        / "uncertainty_aware_census.py"
    )
    materializer.parent.mkdir(parents=True)
    materializer.write_text("# fixture materializer\n", encoding="utf-8")
    monkeypatch.setattr(census, "__file__", str(materializer))
    config_path = data / "config.resolved.json"
    config_path.write_text("{}\n", encoding="utf-8")

    base = InnovationRankerConfig.load(
        ROOT / "examples/spon_ca_burst_innovation_ranker_v5.example.json"
    )
    feature_root = data / "Outputs" / "feature_bank"
    config = replace(
        base,
        feature_root=feature_root,
        feature_manifest=feature_root / "feature_manifest.json",
        source_video=data / "Inputs" / "video.npy",
        labels_tsv=data / "Inputs" / "event_windows.tsv",
        output_dir=data / "Outputs" / "ranker_v5",
        preflight_dir=data / "Outputs" / "ranker_v5_preflight",
    )
    monkeypatch.setattr(census.InnovationRankerConfig, "load", lambda _path: config)
    portable = census.portableize_paths(
        config.to_dict(), repository=repository, data=data
    )
    monkeypatch.setattr(
        census, "EXPECTED_PORTABLE_CONFIG_SHA256", census.stable_hash(portable)
    )

    def fake_hashes(
        files: dict[str, Path], expected: dict[str, str]
    ) -> dict[str, dict[str, object]]:
        assert set(files) == set(expected)
        return {
            role: {"bytes": 1, "sha256": expected[role]}
            for role in sorted(files)
        }

    monkeypatch.setattr(census, "hash_and_verify_files", fake_hashes)
    monkeypatch.setattr(census, "sha256_file", lambda _path: "0" * 64)
    output = repository / "census"
    progress = repository / "census.progress.jsonl"
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    result = census.preflight_uncertainty_aware_census(
        repository_root=repository,
        data_root=data,
        config_path=config_path,
        output_root=output,
        progress=progress,
    )

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert after == before
    assert result["ready"] is True
    assert result["paths"] == {
        "config": "data://config.resolved.json",
        "output": "repo://census",
        "progress": "repo://census.progress.jsonl",
    }
    assert result["config"]["portable_semantic_sha256"] == census.stable_hash(
        portable
    )
    assert result["materializer_code"]["path"] == (
        "repo://neurobench/experiments/neuron_identifiability/"
        "uncertainty_aware_census.py"
    )
