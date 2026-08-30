from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import jsonschema
import pytest
import yaml

from tools import new_research_experiment as scaffold


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture()
def registry_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "checkout"
    shutil.copytree(ROOT / "research" / "schemas", root / "research" / "schemas")
    shutil.copytree(ROOT / "research" / "templates", root / "research" / "templates")
    for group in ("programs", "claims", "experiments", "decisions", "runs"):
        (root / "research" / "registry" / group).mkdir(parents=True, exist_ok=True)
    _write(
        root / "research" / "registry" / "programs" / "test.yaml",
        "schema_version: 1\nrecord_type: program\nid: NREV-PRG-9000\nslug: test-program\n"
        "title: Test program\nsummary: Test only.\nlifecycle: draft\nscientific_scope: Synthetic test scope.\n"
        "objectives: [Test scaffolding.]\nboundaries: [No scientific claim.]\nvisibility: public\n",
    )
    _write(
        root / "research" / "registry" / "claims" / "NREV-CLM-9000.yaml",
        "schema_version: 1\nrecord_type: claim\nid: NREV-CLM-9000\nprogram_id: NREV-PRG-9000\n"
        "title: Test claim\nstatement: Test-only claim.\nscope: Synthetic fixture.\n"
        "limitations: [No scientific bearing.]\nclaim_state: proposed\nevidence_tier: none\n"
        "review_state: not_requested\nevidence_capsule_ids: []\nvisibility: public\n",
    )
    index = {
        "schema_version": 1,
        "flagship_program_id": "NREV-PRG-9000",
        "programs": ["NREV-PRG-9000"],
        "claims": ["NREV-CLM-9000"],
        "experiments": [],
        "completed_experiments": [],
        "planned_experiments": [],
        "decisions": [],
        "runs": [],
        "evidence_capsules": [],
    }
    _write(root / "research" / "registry" / "index.yaml", yaml.safe_dump(index, sort_keys=False))
    _write(root / "docs" / "workflows" / "test_protocol.md", "# Frozen test protocol\n")
    _write(root / "examples" / "test_config.json", '{"scientific_audit": {"enabled": true}}\n')
    _write(root / "pyproject.toml", "[project]\nname = \"fixture\"\nversion = \"0.0.0\"\n")
    descriptor = root / "research" / "data-registry" / "synthetic-fixture.yaml"
    _write(descriptor, "id: SYNTHETIC-FIXTURE\nclassification: public-synthetic\n")
    descriptor_sha = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "NeuRev Test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "remote", "add", "origin", "git@github.com:Jibby2k1/NeuRev-Workbench.git")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root, descriptor_sha


def _args(root: Path, descriptor_sha: str, *extra: str):
    argv = [
        "--root",
        str(root),
        "--experiment-id",
        "NREV-EXP-9001",
        "--decision-id",
        "NREV-DEC-9001",
        "--run-id",
        "NREV-RUN-EXP-9001-PILOT-01",
        "--program-id",
        "NREV-PRG-9000",
        "--claim-id",
        "NREV-CLM-9000",
        "--title",
        "Synthetic scaffold integration",
        "--question",
        "Does the scaffold preserve the native registry contract?",
        "--objective",
        "Validate safe planning-record creation.",
        "--rationale",
        "Future experiments need reproducible and non-overwriting setup.",
        "--design-summary",
        "Validate three linked records in an isolated synthetic checkout.",
        "--protocol-path",
        "docs/workflows/test_protocol.md",
        "--unit-of-analysis",
        "one synthetic registry package",
        "--comparison",
        "generated records versus the canonical schemas",
        "--data-scope",
        "one public synthetic descriptor",
        "--grouping-or-blocking",
        "No grouping is applicable to this structural test.",
        "--sample-plan",
        "One deterministic synthetic fixture.",
        "--randomization-plan",
        "Use the registered deterministic seed.",
        "--blinding-plan",
        "Blinding is not applicable to this structural test.",
        "--analysis-plan",
        "Require all schema validators to pass.",
        "--falsifier",
        "Any emitted record fails its canonical schema.",
        "--gate",
        "schema_contract",
        "analysis",
        "Every emitted record validates against its canonical schema.",
        "--config",
        "examples/test_config.json",
        "--input",
        "SYNTHETIC-FIXTURE",
        "data",
        "research/data-registry/synthetic-fixture.yaml",
        descriptor_sha,
        "--seed",
        "1001",
        "--output-root",
        "Outputs/Test/NREV-EXP-9001/runs/NREV-RUN-EXP-9001-PILOT-01",
        *extra,
    ]
    return scaffold.build_parser().parse_args(argv)


def test_dry_run_builds_three_schema_valid_records_without_mutation(registry_checkout):
    root, descriptor_sha = registry_checkout
    index_before = (root / "research" / "registry" / "index.yaml").read_bytes()

    records, next_index, captured_index = scaffold.build_records(_args(root, descriptor_sha))

    assert set(records) == {"experiment", "decision", "run"}
    assert records["experiment"]["lifecycle"] == "draft"
    assert records["experiment"]["outcome"] == "not_evaluated"
    assert records["decision"]["action"] == "hold"
    assert records["run"]["lifecycle"] == "planned"
    assert records["run"]["code"]["dirty"] is False
    assert next_index["planned_experiments"] == ["NREV-EXP-9001"]
    assert captured_index == index_before
    assert (root / "research" / "registry" / "index.yaml").read_bytes() == index_before
    assert not any(
        path
        for group in ("experiments", "decisions", "runs")
        for path in (root / "research" / "registry" / group).glob("NREV-*-9001*.yaml")
    )

    for kind, record in records.items():
        schema = json.loads(
            (root / "research" / "schemas" / scaffold.SCHEMA_FILES[kind]).read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).validate(record)


def test_write_is_linked_indexed_and_never_overwrites(registry_checkout):
    root, descriptor_sha = registry_checkout
    records, next_index, index_before = scaffold.build_records(_args(root, descriptor_sha))

    written = scaffold.write_records(root, records, next_index, index_before)

    assert len(written) == 4
    index = yaml.safe_load((root / "research" / "registry" / "index.yaml").read_text())
    assert index["experiments"] == ["NREV-EXP-9001"]
    assert index["decisions"] == ["NREV-DEC-9001"]
    assert index["runs"] == ["NREV-RUN-EXP-9001-PILOT-01"]
    experiment_path = root / "research" / "registry" / "experiments" / "NREV-EXP-9001.yaml"
    original_experiment = experiment_path.read_bytes()
    current_index = (root / "research" / "registry" / "index.yaml").read_bytes()

    with pytest.raises(scaffold.ScaffoldError, match="overwrite"):
        scaffold.write_records(root, records, next_index, index_before)

    assert experiment_path.read_bytes() == original_experiment
    assert (root / "research" / "registry" / "index.yaml").read_bytes() == current_index
    assert not (root / "research" / "registry" / ".scaffold.lock").exists()


def test_public_input_rejects_local_inputs_tree(registry_checkout):
    root, descriptor_sha = registry_checkout
    private_path = root / "Inputs" / "raw-recording.tif"
    _write(private_path, "not public\n")
    private_sha = hashlib.sha256(private_path.read_bytes()).hexdigest()
    args = _args(root, descriptor_sha)
    args.input = [["RAW-RECORDING", "data", "Inputs/raw-recording.tif", private_sha]]

    with pytest.raises(scaffold.ScaffoldError, match="sanitized descriptors"):
        scaffold.build_records(args)


def test_dirty_checkout_requires_explicit_capture(registry_checkout):
    root, descriptor_sha = registry_checkout
    _write(root / "examples" / "test_config.json", '{"scientific_audit": {"enabled": true}, "changed": true}\n')

    with pytest.raises(scaffold.ScaffoldError, match="worktree is dirty"):
        scaffold.build_records(_args(root, descriptor_sha))

    records, _, _ = scaffold.build_records(_args(root, descriptor_sha, "--allow-dirty"))
    assert records["run"]["code"]["dirty"] is True
    assert len(records["run"]["code"]["diff_sha256"]) == 64


def test_output_root_must_encode_exact_experiment_and_run_ids(registry_checkout):
    root, descriptor_sha = registry_checkout
    args = _args(root, descriptor_sha)
    args.output_root = "Outputs/Test/NREV-EXP-9999/runs/NREV-RUN-WRONG-01"

    with pytest.raises(scaffold.ScaffoldError, match="exact registered IDs"):
        scaffold.build_records(args)
