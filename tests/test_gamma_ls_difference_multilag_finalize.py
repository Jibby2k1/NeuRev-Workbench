from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference.multilag_finalize import (
    EXPECTED_EXIT_CODE,
    _recorded_termination,
)
from neurobench.experiments.gamma_ls_difference.protected_finalize import (
    ProtectedFinalizeUnavailable,
)


def test_statusless_sealed_termination_requires_observed_sigsegv(
    tmp_path: Path,
) -> None:
    state = _recorded_termination(
        tmp_path, observed_exit_code=EXPECTED_EXIT_CODE
    )
    assert state == {
        "original_status_file_present": False,
        "operator_observed_exit_code": 139,
        "interpretation": "SIGSEGV_after_candidate_seal",
    }
    with pytest.raises(
        ProtectedFinalizeUnavailable, match="operator-observed exit 139"
    ):
        _recorded_termination(tmp_path, observed_exit_code=1)


def test_finalizer_failure_retains_the_initial_statusless_exit(
    tmp_path: Path,
) -> None:
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "status": "finalizer_failed_resumable",
                "error": "OSError('disk full')",
                "original_status_file_present": False,
                "original_operator_observed_exit_code": 139,
            }
        ),
        encoding="utf-8",
    )
    state = _recorded_termination(tmp_path, observed_exit_code=139)
    assert state["operator_observed_exit_code"] == 139
    assert state["prior_finalizer_error"] == "OSError('disk full')"


def test_unrecognized_status_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    with pytest.raises(
        ProtectedFinalizeUnavailable, match="not a recognized finalizer retry"
    ):
        _recorded_termination(tmp_path, observed_exit_code=139)
