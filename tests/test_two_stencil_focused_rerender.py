from __future__ import annotations

import hashlib
import json

import pytest

from neurobench.experiments.gamma_ls_difference.two_stencil_focused_rerender import rerender_focused_cards


def test_companion_rejects_original_summary_drift_before_creating_output(tmp_path):
    source = tmp_path / "focused_diagnostics"
    source.mkdir()
    summary = source / "summary.json"
    summary.write_text(json.dumps({"artifacts": [], "card_count": 14}))
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    qa = source / "visual_qa.json"
    qa.write_text(json.dumps({"source_bindings": {"summary.json": digest}}))
    summary.write_text(json.dumps({"artifacts": [], "card_count": 13}))
    before = {path.name: path.read_bytes() for path in (summary, qa)}
    with pytest.raises(ValueError, match="sealed file hash mismatch"):
        rerender_focused_cards(tmp_path)
    assert not (tmp_path / "focused_diagnostics_readable_v2").exists()
    assert {path.name: path.read_bytes() for path in (summary, qa)} == before
