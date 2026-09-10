from neurobench.experiments.ica_whitening_evaluation.completion_audit import (
    audit_ica_whitening_completion,
)


def test_completion_audit_fails_closed_when_artifacts_are_missing(tmp_path):
    result = audit_ica_whitening_completion(tmp_path / "missing")
    assert result["status"] == "incomplete"
    assert result["completion_claim_allowed"] is False
    assert set(result["incomplete_gates"]) == set(result["gates"])
