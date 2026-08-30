import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAVIGATION = ROOT / "docs" / "navigation.json"


def _local_markdown_links(path: Path) -> list[Path]:
    links = []
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", path.read_text()):
        target = target.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        links.append((path.parent / target).resolve())
    return links


def test_machine_navigation_routes_exist_and_preserve_boundaries() -> None:
    payload = json.loads(NAVIGATION.read_text())
    assert payload["schema_version"] == 2
    assert payload["generated_from"] == "research/registry"
    assert payload["generated_policy"] == "do_not_edit"
    assert payload["scientific_boundaries"]["recordings"] == 1
    assert payload["scientific_boundaries"]["sites"] == 50
    assert payload["scientific_boundaries"]["occurrences"] == 106
    assert {
        "Full-field precision is unresolved.",
        "One-to-one biological source identity is unresolved.",
        "Generalization to an independent recording is unresolved.",
    } <= set(payload["scientific_boundaries"]["unchanged_claims"])
    assert len({route["id"] for route in payload["routes"]}) == len(payload["routes"])
    assert {
        "project_story",
        "local_candidate_review",
        "maintained_codebase",
        "publication_release",
    } <= {route["id"] for route in payload["routes"]}

    for route in payload["routes"]:
        assert (ROOT / route["human_start"]).is_file(), route["id"]
        for field in ("authority", "machine_context", "implementation"):
            if field in route:
                assert (ROOT / route[field]).exists(), f"{route['id']}: {field}"
        for field in ("human_start", "authority", "machine_context"):
            if field in route:
                assert Path(route[field]).parts[0] not in {"Inputs", "Outputs"}, (
                    route["id"],
                    field,
                )
    for generated in payload["generated_files"]["paths"]:
        assert (ROOT / generated).is_file(), generated
        assert Path(generated).parts[0] not in {"Inputs", "Outputs"}, generated
    assert "README.md" not in payload["generated_files"]["paths"]
    regions = {row["path"]: row for row in payload["generated_regions"]["regions"]}
    assert set(regions) == {"README.md", "docs/REPOSITORY_GUIDE.md"}
    for path, region in regions.items():
        text = (ROOT / path).read_text(encoding="utf-8")
        assert text.count(region["begin_marker"]) == 1
        assert text.count(region["end_marker"]) == 1

    story_route = next(route for route in payload["routes"] if route["id"] == "project_story")
    assert story_route["authority"] == "research/registry"
    assert story_route["compiled_view"] == "research/generated/canonical.json"


def test_navigation_exposes_portable_evidence_separately_from_local_outputs() -> None:
    payload = json.loads(NAVIGATION.read_text())
    repository = payload["repository"]
    assert repository["portable_evidence"] == "research/evidence"
    assert (ROOT / repository["portable_evidence"]).is_dir()
    assert repository["local_artifacts"] == "Outputs"

    local_routes = [route for route in payload["routes"] if "local_artifacts" in route]
    assert local_routes
    for route in local_routes:
        assert Path(route["local_artifacts"]).parts[0] == "Outputs"
        assert (ROOT / route["authority"]).is_file()


def test_high_value_document_links_are_not_stale() -> None:
    documents = [
        ROOT / "README.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "REPOSITORY_GUIDE.md",
        ROOT / "docs" / "research" / "README.md",
        ROOT / "docs" / "workflows" / "README.md",
        ROOT / "neurobench" / "experiments" / "README.md",
    ]
    missing = [str(target) for document in documents for target in _local_markdown_links(document) if not target.exists()]
    assert missing == []


def test_agent_entrypoint_distinguishes_checkout_from_external_data() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    assert "Repository checkout: use the directory containing this `AGENTS.md`" in agents
    assert "external data authority" in agents
    assert "docs/REPOSITORY_GUIDE.md" in agents


def test_llm_entrypoint_declares_its_generated_boundary() -> None:
    text = (ROOT / "llms.txt").read_text(encoding="utf-8")
    assert "Generated from `research/registry/`; do not edit by hand." in text
