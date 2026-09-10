from neurobench.experiments.ica_whitening_evaluation.concluding_report import (
    build_concluding_report,
)


def test_concluding_report_keeps_missing_gates_pending(tmp_path):
    text = build_concluding_report(tmp_path / "missing")
    assert "Completion claim allowed:** false" in text
    assert "Pending: exact 30,891-fit factorial coverage has not passed" in text
    assert "Pending: promotion is false" in text
    assert "Pending: exact independent-recording confirmation has not passed" in text
