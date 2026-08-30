from pathlib import Path

from neurobench.experiments.neuron_identifiability.manuscript_exports import _macro


def test_macro_is_visibly_provisional_and_fully_source_mapped(tmp_path: Path):
    source = tmp_path / "source.json"; source.write_text('{"value": 79}\n')
    line, record = _macro("Count", 79, source, "/value", "original_site_original_timing", "abc", tmp_path)
    assert line == r"\newcommand{\Count}{\Provisional{79}}"
    assert record["json_pointer"] == "/value"
    assert len(record["source_sha256"]) == 64
    assert record["status"] == "provisional"
