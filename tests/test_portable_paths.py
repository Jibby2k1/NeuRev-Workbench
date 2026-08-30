from pathlib import Path

from neurobench.portable_paths import configured_root, portable_path, portableize_paths


def test_portable_path_labels_repository_data_and_media_roots(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    data = tmp_path / "frozen-data"
    media = tmp_path / "review-media"

    assert portable_path(repository / "docs" / "result.json", repository=repository, data=data, media=media) == "repo://docs/result.json"
    assert portable_path(data / "Outputs" / "run" / "summary.json", repository=repository, data=data, media=media) == "data://Outputs/run/summary.json"
    assert portable_path(media / "v7_panel" / "manifest.json", repository=repository, data=data, media=media) == "media://v7_panel/manifest.json"


def test_portableize_paths_rewrites_nested_values_and_mapping_keys(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    source = repository / "paper" / "summary.json"
    payload = {str(source): {"source": str(source), "scientific_value": 0.887}}

    portable = portableize_paths(payload, repository=repository)

    assert portable == {
        "repo://paper/summary.json": {
            "source": "repo://paper/summary.json",
            "scientific_value": 0.887,
        }
    }


def test_configured_root_honors_runtime_environment(monkeypatch, tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    monkeypatch.setenv("NEUROBENCH_TEST_ROOT", str(selected))
    assert configured_root("NEUROBENCH_TEST_ROOT", tmp_path / "fallback") == selected.resolve()
