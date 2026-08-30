#!/usr/bin/env python3
"""Safely scaffold one schema-valid native NeuRev experiment package.

The package contains a draft experiment, a planning-hold decision, and a
planned run. The default is read-only preview; ``--write`` uses exclusive file
creation and updates the registry index only after every record validates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import yaml
from jsonschema import Draft202012Validator, FormatChecker


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
ID_PATTERNS = {
    "experiment": re.compile(r"^NREV-EXP-[0-9]{4,}$"),
    "decision": re.compile(r"^NREV-DEC-[0-9]{4,}$"),
    "run": re.compile(r"^NREV-RUN-[A-Z0-9][A-Z0-9-]{2,63}$"),
    "program": re.compile(r"^NREV-PRG-[0-9]{4,}$"),
    "claim": re.compile(r"^NREV-CLM-[0-9]{4,}$"),
}
SCHEMA_FILES = {
    "experiment": "experiment.schema.json",
    "decision": "decision.schema.json",
    "run": "run.schema.json",
}
TEMPLATE_FILES = {
    "experiment": "native-experiment.template.yaml",
    "decision": "draft-decision.template.yaml",
    "run": "planned-run.template.yaml",
}
INPUT_ROLES = {"data", "labels", "model", "configuration", "reference", "other"}
GATE_STAGES = {"preflight", "execution", "analysis", "review", "publication"}
RESERVED_GATE_IDS = {"output_collision", "scientific_audit", "publication_boundary"}
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
PRIVATE_PATH_PARTS = {
    ".env",
    "inputs",
    "outputs",
    "identity-map",
    "identity_map",
    "private",
    "private-review",
    "randomization-key",
    "randomization-keys",
    "reviewer-response",
    "reviewer-responses",
    "secrets",
}


class ScaffoldError(ValueError):
    """Raised when a scaffold request is unsafe or structurally incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.decode(errors="replace").strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise ScaffoldError(f"Git provenance check failed: {detail}") from exc


def _normalize_repository_uri(raw: str) -> str:
    value = raw.strip()
    ssh_match = re.fullmatch(r"git@([^:]+):(.+?)(?:\.git)?", value)
    if ssh_match:
        value = f"https://{ssh_match.group(1)}/{ssh_match.group(2).removesuffix('.git')}"
    elif value.startswith("ssh://git@"):
        parsed = urlparse(value)
        value = f"https://{parsed.hostname}/{parsed.path.lstrip('/').removesuffix('.git')}"
    elif value.endswith(".git"):
        value = value[:-4]
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ScaffoldError(
            "The code repository must resolve to a credential-free HTTPS URI; "
            "configure origin or pass a public remote before scaffolding."
        )
    return value


def _dirty_state_sha256(root: Path, status: bytes) -> str:
    """Hash tracked changes plus names and content of non-ignored untracked files."""
    digest = hashlib.sha256()
    digest.update(b"NEUREV-DIRTY-STATE-V1\0")
    digest.update(status)
    digest.update(_git(root, "diff", "--binary", "HEAD", "--"))
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for encoded in sorted(item for item in untracked if item):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = root / relative
        digest.update(b"\0PATH\0")
        digest.update(encoded)
        if path.is_symlink():
            digest.update(b"\0SYMLINK\0")
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            digest.update(b"\0FILE\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def capture_code_provenance(root: Path, *, allow_dirty: bool) -> dict[str, Any]:
    worktree = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve(strict=True)
    if worktree != root.resolve(strict=True):
        raise ScaffoldError(
            f"--root must be the substantive Git checkout ({worktree}), not a nested or wrapper directory"
        )
    repository = _normalize_repository_uri(_git(root, "remote", "get-url", "origin").decode().strip())
    commit = _git(root, "rev-parse", "HEAD").decode().strip().lower()
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ScaffoldError(f"Expected a full Git commit SHA, found {commit!r}")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise ScaffoldError(
            "The worktree is dirty. Commit or stash the intended protocol/configuration first, "
            "or repeat with --allow-dirty to capture a working-state digest."
        )
    result: dict[str, Any] = {"repository": repository, "commit": commit, "dirty": dirty}
    if branch and branch != "HEAD":
        result["branch"] = branch
    if dirty:
        result["diff_sha256"] = _dirty_state_sha256(root, status)
    return result


def _safe_repository_file(root: Path, raw: str, *, purpose: str) -> tuple[str, Path]:
    if not raw or "\\" in raw:
        raise ScaffoldError(f"{purpose} must be a non-empty POSIX repository-relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ScaffoldError(f"{purpose} must be traversal-free and repository-relative: {raw!r}")
    path = root.joinpath(*pure.parts)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ScaffoldError(f"{purpose} must resolve to an existing file inside the checkout: {raw!r}") from exc
    if not resolved.is_file():
        raise ScaffoldError(f"{purpose} is not a file: {raw!r}")
    return pure.as_posix(), resolved


def _safe_output_root(root: Path, raw: str, *, experiment_id: str, run_id: str) -> str:
    if not raw or "\\" in raw:
        raise ScaffoldError("--output-root must use a POSIX repository-relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or not pure.parts:
        raise ScaffoldError("--output-root must be traversal-free and repository-relative")
    if pure.parts[0] != "Outputs" or len(pure.parts) < 5:
        raise ScaffoldError(
            "--output-root must be a specific new path below Outputs/<program>/<experiment>/runs/<run>"
        )
    if tuple(pure.parts[-3:]) != (experiment_id, "runs", run_id):
        raise ScaffoldError(
            "--output-root must end with the exact registered IDs: "
            f"{experiment_id}/runs/{run_id}"
        )
    candidate = root.joinpath(*pure.parts)
    if candidate.exists() or candidate.is_symlink():
        raise ScaffoldError(f"Refusing an existing output root: {pure.as_posix()}")
    existing_parent = candidate.parent
    while not existing_parent.exists() and existing_parent != root:
        existing_parent = existing_parent.parent
    try:
        existing_parent.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ScaffoldError("--output-root resolves through a path outside the checkout") from exc
    return pure.as_posix()


def _validate_public_input_location(root: Path, raw: str, expected_sha256: str) -> str:
    if raw.startswith("doi:"):
        if len(raw) <= 4 or any(char.isspace() for char in raw):
            raise ScaffoldError(f"Invalid DOI input locator: {raw!r}")
        return raw
    if raw.startswith("https://"):
        parsed = urlparse(raw)
        if not parsed.netloc or parsed.username or parsed.password:
            raise ScaffoldError("Input URLs must be credential-free HTTPS locators")
        return raw
    relative, path = _safe_repository_file(root, raw, purpose="input descriptor")
    lowered = {part.lower() for part in PurePosixPath(relative).parts}
    if lowered & PRIVATE_PATH_PARTS:
        raise ScaffoldError(
            "Public run records may reference only sanitized descriptors outside Inputs/, Outputs/, "
            "and private-review paths."
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ScaffoldError(
            f"Input descriptor checksum mismatch for {relative}: expected {expected_sha256}, observed {observed}"
        )
    return relative


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScaffoldError(f"Expected a YAML mapping: {path}")
    return payload


def _replace_tokens(value: Any, replacements: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return copy.deepcopy(replacements[value]) if value in replacements else value
    if isinstance(value, list):
        return [_replace_tokens(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_tokens(item, replacements) for key, item in value.items()}
    return value


def _find_unresolved_tokens(value: Any, location: str = "record") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and re.fullmatch(r"__[A-Z0-9_]+__", value):
        found.append(f"{location}={value}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_unresolved_tokens(item, f"{location}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_find_unresolved_tokens(item, f"{location}.{key}"))
    return found


def _schema_validate(root: Path, kind: str, record: Mapping[str, Any]) -> None:
    schema_path = root / "research" / "schemas" / SCHEMA_FILES[kind]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(record), key=lambda item: list(item.absolute_path))
    if errors:
        details = []
        for error in errors:
            field = ".".join(map(str, error.absolute_path)) or "<record>"
            details.append(f"{kind}.{field}: {error.message}")
        raise ScaffoldError("Schema validation failed:\n" + "\n".join(details))


def _known_records(root: Path, group: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "research" / "registry" / group).glob("*.yaml")):
        row = _load_yaml_mapping(path)
        identifier = row.get("id")
        if isinstance(identifier, str):
            records[identifier] = row
    return records


def _validate_id(kind: str, value: str) -> None:
    if not ID_PATTERNS[kind].fullmatch(value):
        raise ScaffoldError(f"Invalid {kind} ID: {value!r}")


def _ensure_unique_strings(values: Iterable[str], *, field: str) -> list[str]:
    result = list(values)
    if not result or any(not item.strip() for item in result):
        raise ScaffoldError(f"{field} requires at least one non-empty value")
    if len(result) != len(set(result)):
        raise ScaffoldError(f"{field} contains duplicates")
    return result


def _build_inputs(root: Path, raw_inputs: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    identifiers: set[str] = set()
    records: list[dict[str, Any]] = []
    for identifier, role, location, digest in raw_inputs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,79}", identifier):
            raise ScaffoldError(f"Invalid input identifier: {identifier!r}")
        if identifier in identifiers:
            raise ScaffoldError(f"Duplicate input identifier: {identifier}")
        identifiers.add(identifier)
        if role not in INPUT_ROLES:
            raise ScaffoldError(f"Invalid input role {role!r}; choose from {sorted(INPUT_ROLES)}")
        if not SHA256_RE.fullmatch(digest):
            raise ScaffoldError(f"Input {identifier} requires a lowercase SHA-256 digest")
        records.append(
            {
                "id": identifier,
                "role": role,
                "sha256": digest,
                "location": _validate_public_input_location(root, location, digest),
            }
        )
    return records


def _build_gates(raw_gates: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = [
        {
            "id": "output_collision",
            "criterion": "The registered output root is absent immediately before execution.",
            "stage": "preflight",
            "required": True,
        },
        {
            "id": "scientific_audit",
            "criterion": "The applicable scientific audit inventory and media validation pass before audit completion.",
            "stage": "review",
            "required": True,
        },
        {
            "id": "publication_boundary",
            "criterion": "The public evidence capsule and release candidates pass the publication-boundary audit.",
            "stage": "publication",
            "required": True,
        },
    ]
    seen = set(RESERVED_GATE_IDS)
    for identifier, stage, criterion in raw_gates:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", identifier):
            raise ScaffoldError(f"Invalid gate ID: {identifier!r}")
        if identifier in seen:
            raise ScaffoldError(f"Duplicate or reserved gate ID: {identifier}")
        if stage not in GATE_STAGES:
            raise ScaffoldError(f"Invalid gate stage {stage!r}; choose from {sorted(GATE_STAGES)}")
        if not criterion.strip():
            raise ScaffoldError(f"Gate {identifier} requires a non-empty criterion")
        seen.add(identifier)
        gates.append({"id": identifier, "criterion": criterion, "stage": stage, "required": True})
    return gates


def _build_randomization(seeds: Sequence[int], not_applicable: str | None) -> dict[str, Any]:
    if seeds and not_applicable:
        raise ScaffoldError("Use --seed or --randomization-not-applicable, not both")
    if not seeds and not not_applicable:
        raise ScaffoldError(
            "Register at least one deterministic --seed, or give an explicit --randomization-not-applicable reason"
        )
    if len(seeds) != len(set(seeds)):
        raise ScaffoldError("Seeds must be unique")
    if any(seed < 0 for seed in seeds):
        raise ScaffoldError("Seeds must be non-negative integers")
    if seeds:
        return {"deterministic": True, "seeds": list(seeds)}
    return {"deterministic": True, "seeds": [], "notes": not_applicable}


def _updated_index(index: dict[str, Any], *, experiment_id: str, decision_id: str, run_id: str) -> dict[str, Any]:
    result = copy.deepcopy(index)
    additions = {
        "experiments": experiment_id,
        "planned_experiments": experiment_id,
        "decisions": decision_id,
        "runs": run_id,
    }
    for field, identifier in additions.items():
        values = result.get(field)
        if not isinstance(values, list):
            raise ScaffoldError(f"research/registry/index.yaml field {field!r} must be a list")
        if identifier in values:
            raise ScaffoldError(f"Registry index already contains {identifier}")
        values.append(identifier)
        values.sort()
    return result


def build_records(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any], bytes]:
    root = args.root.resolve(strict=True)
    for kind, value in (
        ("experiment", args.experiment_id),
        ("decision", args.decision_id),
        ("run", args.run_id),
        ("program", args.program_id),
    ):
        _validate_id(kind, value)
    for claim_id in args.claim_id:
        _validate_id("claim", claim_id)
    for dependency_id in args.depends_on:
        _validate_id("experiment", dependency_id)

    programs = _known_records(root, "programs")
    claims = _known_records(root, "claims")
    experiments = _known_records(root, "experiments")
    if args.program_id not in programs:
        raise ScaffoldError(f"Unknown program ID: {args.program_id}")
    for claim_id in args.claim_id:
        claim = claims.get(claim_id)
        if claim is None:
            raise ScaffoldError(f"Unknown claim ID: {claim_id}")
        if claim.get("program_id") != args.program_id:
            raise ScaffoldError(f"Claim {claim_id} belongs to {claim.get('program_id')}, not {args.program_id}")
    for dependency_id in args.depends_on:
        if dependency_id not in experiments:
            raise ScaffoldError(f"Unknown dependency experiment ID: {dependency_id}")

    protocol_path, _ = _safe_repository_file(root, args.protocol_path, purpose="protocol path")
    config_path, config_file = _safe_repository_file(root, args.config, purpose="configuration path")
    lock_path, lock_file = _safe_repository_file(root, args.dependency_lock, purpose="dependency lock path")
    for purpose, relative in (("protocol", protocol_path), ("configuration", config_path), ("dependency lock", lock_path)):
        if PurePosixPath(relative).parts[0] in {"Inputs", "Outputs"}:
            raise ScaffoldError(f"{purpose} must be maintained source, not a local data or output path")

    inputs = _build_inputs(root, args.input)
    gates = _build_gates(args.gate)
    falsifiers = _ensure_unique_strings(args.falsifier, field="--falsifier")
    randomization = _build_randomization(args.seed, args.randomization_not_applicable)
    output_root = _safe_output_root(
        root,
        args.output_root,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
    )
    code = capture_code_provenance(root, allow_dirty=args.allow_dirty)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    date = now.date().isoformat()
    planned_at = now.isoformat().replace("+00:00", "Z")
    environment = {
        "platform": f"{sys.platform}/{platform.machine() or 'unknown-machine'}",
        "runtime": platform.python_implementation() + " " + platform.python_version(),
        "dependency_lock": {"path": lock_path, "sha256": sha256_file(lock_file)},
    }

    replacements: dict[str, Any] = {
        "__EXPERIMENT_ID__": args.experiment_id,
        "__DECISION_ID__": args.decision_id,
        "__RUN_ID__": args.run_id,
        "__PROGRAM_ID__": args.program_id,
        "__TITLE__": args.title,
        "__QUESTION__": args.question,
        "__OBJECTIVE__": args.objective,
        "__RATIONALE__": args.rationale,
        "__DESIGN_SUMMARY__": args.design_summary,
        "__PROTOCOL_PATH__": protocol_path,
        "__UNIT_OF_ANALYSIS__": args.unit_of_analysis,
        "__COMPARISON__": args.comparison,
        "__DATA_SCOPE__": args.data_scope,
        "__GROUPING_OR_BLOCKING__": args.grouping_or_blocking,
        "__SAMPLE_PLAN__": args.sample_plan,
        "__RANDOMIZATION_PLAN__": args.randomization_plan,
        "__BLINDING_PLAN__": args.blinding_plan,
        "__ANALYSIS_PLAN__": args.analysis_plan,
        "__FALSIFIERS__": falsifiers,
        "__GATES__": gates,
        "__CLAIM_IDS__": list(args.claim_id),
        "__DECISION_IDS__": [args.decision_id],
        "__RUN_IDS__": [args.run_id],
        "__DEPENDENCY_IDS__": list(args.depends_on),
        "__DATE__": date,
        "__DECISION_TITLE__": f"Planning hold for {args.title}",
        "__EXPERIMENT_IDS__": [args.experiment_id],
        "__PLANNED_AT__": planned_at,
        "__CODE_PROVENANCE__": code,
        "__CONFIGURATION__": {"path": config_path, "sha256": sha256_file(config_file)},
        "__INPUTS__": inputs,
        "__ENVIRONMENT__": environment,
        "__RANDOMIZATION__": randomization,
        "__OUTPUT_ROOT__": output_root,
    }
    records: dict[str, dict[str, Any]] = {}
    templates = root / "research" / "templates"
    for kind, filename in TEMPLATE_FILES.items():
        template = _load_yaml_mapping(templates / filename)
        record = _replace_tokens(template, replacements)
        unresolved = _find_unresolved_tokens(record)
        if unresolved:
            raise ScaffoldError(f"Unresolved tokens in {filename}: {', '.join(unresolved)}")
        _schema_validate(root, kind, record)
        records[kind] = record

    for path in _record_paths(root, records).values():
        if path.exists() or path.is_symlink():
            raise ScaffoldError(f"Refusing to overwrite existing registry record: {path.relative_to(root)}")

    index_path = root / "research" / "registry" / "index.yaml"
    index_bytes = index_path.read_bytes()
    index = _load_yaml_mapping(index_path)
    next_index = _updated_index(
        index,
        experiment_id=args.experiment_id,
        decision_id=args.decision_id,
        run_id=args.run_id,
    )
    return records, next_index, index_bytes


def _record_paths(root: Path, records: Mapping[str, Mapping[str, Any]]) -> dict[str, Path]:
    return {
        "experiment": root / "research" / "registry" / "experiments" / f"{records['experiment']['id']}.yaml",
        "decision": root / "research" / "registry" / "decisions" / f"{records['decision']['id']}.yaml",
        "run": root / "research" / "registry" / "runs" / f"{records['run']['id']}.yaml",
    }


def _yaml_bytes(payload: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(payload), sort_keys=False, allow_unicode=True, width=120
    ).encode("utf-8")


def write_records(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    next_index: Mapping[str, Any],
    expected_index_bytes: bytes,
) -> list[Path]:
    paths = _record_paths(root, records)
    index_path = root / "research" / "registry" / "index.yaml"
    lock_path = root / "research" / "registry" / ".scaffold.lock"
    created: list[Path] = []
    temporary_index: Path | None = None
    lock_fd: int | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.is_symlink():
                raise ScaffoldError(f"Refusing to overwrite existing registry record: {path.relative_to(root)}")
        if index_path.read_bytes() != expected_index_bytes:
            raise ScaffoldError("Registry index changed after validation; retry from a fresh preview")
        for kind in ("experiment", "decision", "run"):
            path = paths[kind]
            with path.open("xb") as handle:
                handle.write(_yaml_bytes(records[kind]))
                handle.flush()
                os.fsync(handle.fileno())
            created.append(path)
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".index.yaml.", suffix=".tmp", dir=index_path.parent, delete=False
        ) as handle:
            temporary_index = Path(handle.name)
            handle.write(_yaml_bytes(next_index))
            handle.flush()
            os.fsync(handle.fileno())
        if index_path.read_bytes() != expected_index_bytes:
            raise ScaffoldError("Registry index changed during write; no existing record was overwritten")
        os.replace(temporary_index, index_path)
        temporary_index = None
        return [*created, index_path]
    except FileExistsError as exc:
        raise ScaffoldError(
            "Another scaffold operation is active, or a stale research/registry/.scaffold.lock requires review"
        ) from exc
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary_index is not None:
            try:
                temporary_index.unlink()
            except FileNotFoundError:
                pass
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SCRIPT_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--write", action="store_true", help="write records after a schema-valid preview")
    parser.add_argument("--allow-dirty", action="store_true", help="capture a dirty-state digest instead of refusing")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--claim-id", action="append", default=[], help="existing atomic claim ID; repeat as needed")
    parser.add_argument("--depends-on", action="append", default=[], help="existing experiment dependency ID")
    parser.add_argument("--title", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--design-summary", required=True)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--unit-of-analysis", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--data-scope", required=True)
    parser.add_argument("--grouping-or-blocking", required=True)
    parser.add_argument("--sample-plan", required=True)
    parser.add_argument("--randomization-plan", required=True)
    parser.add_argument("--blinding-plan", required=True)
    parser.add_argument("--analysis-plan", required=True)
    parser.add_argument("--falsifier", action="append", required=True, help="repeat for each falsifier")
    parser.add_argument(
        "--gate",
        action="append",
        nargs=3,
        required=True,
        metavar=("ID", "STAGE", "CRITERION"),
        help="required experiment-specific gate; repeat as needed",
    )
    parser.add_argument("--config", required=True, help="maintained repository-relative configuration")
    parser.add_argument("--dependency-lock", default="pyproject.toml")
    parser.add_argument(
        "--input",
        action="append",
        nargs=4,
        required=True,
        metavar=("ID", "ROLE", "LOCATION", "SHA256"),
        help="sanitized input descriptor or durable URL; repeat as needed",
    )
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--randomization-not-applicable")
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        records, next_index, index_bytes = build_records(args)
        paths = _record_paths(root, records)
        if args.write:
            written = write_records(root, records, next_index, index_bytes)
            print("Created schema-valid planning records:")
            for path in written:
                print(f"  {path.relative_to(root)}")
            print("Next: review the records, then rebuild and check generated registry views.")
        else:
            print("Schema-valid dry run; no files changed:")
            for path in (*paths.values(), root / "research" / "registry" / "index.yaml"):
                print(f"  {path.relative_to(root)}")
            code = records["run"]["code"]
            print(
                f"Captured code target {code['commit']} (dirty={str(code['dirty']).lower()}); "
                f"output root {records['run']['artifacts']['output_root']}"
            )
            print("Review the request and repeat with --write to create it.")
        return 0
    except (OSError, ScaffoldError, json.JSONDecodeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
