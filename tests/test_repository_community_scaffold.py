from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_citation_matches_package_and_public_repository():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == "NeuRev Workbench"
    assert citation["version"] == project["version"]
    assert citation["license"] == project["license"]
    assert citation["repository-code"] == "https://github.com/Jibby2k1/NeuRev-Workbench"
    assert citation["authors"]


def test_issue_forms_are_parseable_and_have_unique_field_ids():
    issue_root = ROOT / ".github" / "ISSUE_TEMPLATE"
    forms = sorted(path for path in issue_root.glob("*.yml") if path.name != "config.yml")
    assert {path.name for path in forms} >= {
        "bug_report.yml",
        "documentation.yml",
        "research_experiment.yml",
        "scientific_evidence.yml",
    }
    for path in forms:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["name"]
        assert payload["description"]
        assert isinstance(payload["body"], list) and payload["body"]
        ids = [item["id"] for item in payload["body"] if "id" in item]
        assert len(ids) == len(set(ids)), path
        assert not re.search(r"/home/|/Users/|[A-Za-z]:\\\\", path.read_text(encoding="utf-8"))


def test_release_and_contribution_policies_preserve_scientific_boundaries():
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    release = (ROOT / "docs" / "RELEASE_PROCESS.md").read_text(encoding="utf-8")
    pull_request = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")

    for text in (contributing, release, pull_request):
        assert "publication" in text.lower()
        assert "scientific" in text.lower()
        assert "Inputs/" in text
        assert "Outputs/" in text
    assert "does not by itself establish a claim" in re.sub(r"\s+", " ", contributing)
    assert "Publication is not promotion" in re.sub(r"\s+", " ", release)
    assert "Generated research and manuscript views" in re.sub(r"\s+", " ", pull_request)


def test_research_templates_are_tokenized_and_planning_safe():
    templates = ROOT / "research" / "templates"
    experiment = yaml.safe_load((templates / "native-experiment.template.yaml").read_text())
    decision = yaml.safe_load((templates / "draft-decision.template.yaml").read_text())
    run = yaml.safe_load((templates / "planned-run.template.yaml").read_text())

    assert experiment["origin"] == {"kind": "native"}
    assert experiment["lifecycle"] == "draft"
    assert experiment["outcome"] == "not_evaluated"
    assert experiment["id"] == "__EXPERIMENT_ID__"
    assert decision["action"] == "hold"
    assert "does not authorize execution" in decision["constraints"][0]
    assert run["lifecycle"] == "planned"
    assert run["validation"]["execution"] == "not_run"
    assert (templates / "experiment-protocol.template.md").is_file()
    assert (templates / "evidence-capsule.after-validation.template.json").is_file()
