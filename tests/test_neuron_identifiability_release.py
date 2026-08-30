from pathlib import Path

from neurobench.experiments.neuron_identifiability.reproducibility_release import inventory, sha256


def test_release_inventory_is_sorted_hashed_and_excludes_itself(tmp_path: Path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    rows = inventory(tmp_path, exclude={tmp_path / "b.txt"})
    assert [row["path"] for row in rows] == ["a.txt"]
    assert rows[0]["sha256"] == sha256(tmp_path / "a.txt")
