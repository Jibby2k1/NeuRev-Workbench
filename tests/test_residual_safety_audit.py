from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neurobench.experiments.neuron_identifiability.residual_safety import (
    JEPA_RESIDUAL_METHOD,
    RANDOM_RESIDUAL_METHOD,
    RAW_METHOD,
)
from neurobench.experiments.neuron_identifiability.residual_safety_audit import (
    run_residual_safety_audit,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic(method: str, fixture: str) -> dict[str, object]:
    return {
        "method": method,
        "background_recording_id": "recording",
        "background_window_id": "window",
        "fixture_id": fixture,
        "background": {"background_rms_ratio": 1.5, "dynamic_mad_ratio": 1.2},
        "seams": {"seam_to_interior_jump_ratio": 9.0},
        "decoder_reference": {
            "seams": {
                "signed_residual_off": {"boundary_to_within_jump_ratio": 1.5}
            }
        },
    }


def _evaluation(method: str, fixture: str) -> dict[str, object]:
    return {
        "method": method,
        "fixture_id": fixture,
        "background_recording_id": "recording",
        "background_window_id": "window",
        "source_on_recovery": {
            "recall": 0.1875 if method == RAW_METHOD else 1.0,
            "recovered_sources": 1,
            "injected_sources": 1,
        },
    }


def _parent(tmp_path: Path) -> Path:
    parent = tmp_path / "parent"
    parent.mkdir()
    diagnostics = {
        "rows": [
            _diagnostic(JEPA_RESIDUAL_METHOD, "fixture"),
            _diagnostic(RANDOM_RESIDUAL_METHOD, "fixture"),
        ]
    }
    evaluations = {
        "rows": [
            _evaluation(RAW_METHOD, "fixture"),
            _evaluation(JEPA_RESIDUAL_METHOD, "fixture"),
            _evaluation(RANDOM_RESIDUAL_METHOD, "fixture"),
        ]
    }
    for name, payload in (
        ("residual_diagnostics.json", diagnostics),
        ("paired_injection_results.json", evaluations),
    ):
        (parent / name).write_text(json.dumps(payload), encoding="utf-8")
    artifacts = [
        {"path": name, "sha256": _sha(parent / name), "bytes": (parent / name).stat().st_size}
        for name in ("residual_diagnostics.json", "paired_injection_results.json")
    ]
    (parent / "artifact_index.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    return parent


def _run_b_shaped_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "NREV-RUN-EXP-0029-SCREEN-20260830-B"
    parent.mkdir()
    diagnostics = []
    evaluations = []
    for window_index in range(12):
        window = f"window_{window_index:02d}"
        for fixture_index in range(9):
            fixture = f"{window}__fixture_{fixture_index:02d}"
            for method in (JEPA_RESIDUAL_METHOD, RANDOM_RESIDUAL_METHOD):
                row = _diagnostic(method, fixture)
                row["background_window_id"] = window
                diagnostics.append(row)
            for method in (RAW_METHOD, JEPA_RESIDUAL_METHOD, RANDOM_RESIDUAL_METHOD):
                row = _evaluation(method, fixture)
                row["background_window_id"] = window
                row["source_on_recovery"]["recall"] = 0.1875 if method == RAW_METHOD else 1.0
                evaluations.append(row)
    payloads = {
        "residual_diagnostics.json": {"rows": diagnostics},
        "paired_injection_results.json": {"rows": evaluations},
    }
    for name, payload in payloads.items():
        (parent / name).write_text(json.dumps(payload), encoding="utf-8")
    artifacts = [
        {"path": name, "sha256": _sha(parent / name), "bytes": (parent / name).stat().st_size}
        for name in payloads
    ]
    (parent / "artifact_index.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    return parent


def test_parent_hash_drift_fails_before_output_creation(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    (parent / "residual_diagnostics.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="parent input drifted"):
        run_residual_safety_audit(parent, output)

    assert not output.exists()
    assert not output.with_name("output.partial").exists()


def test_existing_output_or_partial_is_never_overwritten(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run_residual_safety_audit(parent, output)


def test_run_b_shaped_package_is_atomic_indexed_and_fail_closed(tmp_path: Path) -> None:
    parent = _run_b_shaped_parent(tmp_path)
    output = tmp_path / "output"

    summary = run_residual_safety_audit(parent, output)

    assert output.is_dir()
    assert not output.with_name("output.partial").exists()
    assert summary["methods"][JEPA_RESIDUAL_METHOD]["safe_window_count"] == 0
    assert summary["methods"][RANDOM_RESIDUAL_METHOD]["safe_window_count"] == 0
    assert summary["policy_results"][JEPA_RESIDUAL_METHOD]["fixture_count"] == 108
    assert summary["policy_results"][JEPA_RESIDUAL_METHOD]["macro_source_on_recall"] == 0.1875
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["all_engineering_checks_passed"] is True
    index = json.loads((output / "artifact_index.json").read_text(encoding="utf-8"))
    assert index["artifact_count"] == len(index["artifacts"])
    for row in index["artifacts"]:
        path = output / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha(path) == row["sha256"]
