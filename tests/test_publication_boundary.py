import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from tools.audit_publication_boundary import scan_repository, should_fail


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "tools" / "audit_publication_boundary.py"


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _init_repository(root: Path, *, include_ignore: bool = False) -> None:
    _git(root, "init", "--quiet")
    if include_ignore:
        shutil.copyfile(ROOT / ".gitignore", root / ".gitignore")


def _write(root: Path, relative_path: str, content: str | bytes = "fixture\n") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


def _is_ignored(root: Path, relative_path: str) -> bool:
    completed = _git(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        relative_path,
        check=False,
    )
    assert completed.returncode in {0, 1}
    return completed.returncode == 0


def _write_zip_container(
    root: Path,
    relative_path: str,
    members: dict[str, str | bytes],
) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_name, content in members.items():
            archive.writestr(member_name, content)
    return path


def test_gitignore_enforces_local_generated_and_private_boundaries(tmp_path: Path) -> None:
    _init_repository(tmp_path, include_ignore=True)
    os.symlink("/" + "opt" + "/external-environment", tmp_path / "env")

    ignored = [
        "Inputs",
        "Inputs/fish/raw_movie.tif",
        "Inputs/labels.tsv",
        "Outputs",
        "Outputs/experiment/run.json",
        ".venv-neurobench",
        ".venv",
        "env",
        "venv/python",
        "tmp/scratch.txt",
        "paper/deck_tmp/build.js",
        "paper/deck/node_modules/package/index.js",
        "paper/deck/build/bundle.js",
        "paper/build/main.aux",
        "paper/compiler.log",
        "paper/release.zip",
        "review/private_administrator/contract.json",
        "review/randomization_key.json",
        "review/reviewer_submission_a.json",
        ".env.local",
        "credentials.json",
    ]
    for relative_path in ignored:
        assert _is_ignored(tmp_path, relative_path), relative_path

    allowed = [
        "Inputs/README.md",
        "README.md",
        ".env.example",
        "docs/assets/curated_figure.pdf",
        "paper/overleaf_jnm/main.tex",
    ]
    for relative_path in allowed:
        assert not _is_ignored(tmp_path, relative_path), relative_path


def test_clean_candidate_set_passes_without_mutating_repository(tmp_path: Path) -> None:
    _init_repository(tmp_path, include_ignore=True)
    _write(tmp_path, "src/module.py", "VALUE = 3\n")
    _write(tmp_path, "README.md", "# Portable fixture\n")
    _write(tmp_path, ".env.example", "API_KEY=example\n")
    _write(tmp_path, "Inputs/private.tsv", "not publishable\n")
    _git(tmp_path, "add", "src/module.py")
    status_before = _git(tmp_path, "status", "--porcelain=v1", "-z").stdout

    report = scan_repository(tmp_path)

    status_after = _git(tmp_path, "status", "--porcelain=v1", "-z").stdout
    assert status_after == status_before
    assert report.errors == 0
    assert report.warnings == 0
    assert report.files_considered == 4  # ignore rules, README, env template, and source
    assert {finding.path for finding in report.findings} == set()


def test_auditor_finds_publication_risks_without_exposing_secret(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _write(tmp_path, "Inputs/raw.dat")
    _write(tmp_path, "Outputs/run/result.json", "{}\n")
    _write(tmp_path, "review/randomization_key.json", "{}\n")
    _write(tmp_path, "release.zip", b"PK fixture")
    _write(tmp_path, "src/large.txt", "x" * 80)
    home_path = "/" + "home" + "/alice/private/source.tif"
    fake_token = "ghp_" + "A" * 36
    _write(tmp_path, "src/config.txt", f"source={home_path}\ntoken={fake_token}\n")
    os.symlink("/" + "opt" + "/external-python", tmp_path / "env-link")
    _git(tmp_path, "add", "-f", ".")
    status_before = _git(tmp_path, "status", "--porcelain=v1", "-z").stdout

    report = scan_repository(tmp_path, max_file_bytes=32, max_text_bytes=4096)

    status_after = _git(tmp_path, "status", "--porcelain=v1", "-z").stdout
    assert status_after == status_before
    codes = {finding.code for finding in report.findings}
    assert {
        "absolute-home-path",
        "absolute-symlink",
        "archive-candidate",
        "likely-secret",
        "private-review-payload",
        "tracked-local-artifact",
        "unexpectedly-large-file",
    } <= codes
    assert report.errors >= 5
    assert report.warnings >= 2
    serialized = json.dumps(report.to_dict())
    assert fake_token not in serialized
    assert should_fail(report, "error")
    assert should_fail(report, "warning")
    assert not should_fail(report, "never")


def test_auditor_scans_supported_zip_containers_without_leaking_matches(
    tmp_path: Path,
) -> None:
    _init_repository(tmp_path)
    home_path = "/" + "home" + "/alice/private/source.tif"
    fake_token = "ghp_" + "A" * 36
    fixtures = {
        "paper/deck.pptx": "ppt/notesSlides/notesSlide1.xml",
        "paper/manuscript.docx": "word/document.xml",
        "paper/table.xlsx": "xl/sharedStrings.xml",
        "paper/arrays.npz": "paths.npy",
    }
    for container_path, member_name in fixtures.items():
        member_content = f"source={home_path}\ntoken={fake_token}\n"
        if container_path.endswith(".npz"):
            # Model the NUL-separated representation used by fixed-width NumPy
            # Unicode arrays without loading NumPy in this safety test.
            member_content = "\0".join(member_content)
        _write_zip_container(tmp_path, container_path, {member_name: member_content})
    _git(tmp_path, "add", ".")

    report = scan_repository(tmp_path)

    assert report.container_files_scanned == 4
    assert report.container_members_scanned == 4
    assert report.errors == 8
    assert report.warnings == 0
    for container_path in fixtures:
        codes = {
            finding.code
            for finding in report.findings
            if finding.path == container_path
        }
        assert codes == {"absolute-home-path", "likely-secret"}
    serialized = json.dumps(report.to_dict())
    assert home_path not in serialized
    assert fake_token not in serialized
    assert not any(member_name in serialized for member_name in fixtures.values())


def test_zip_container_scan_limits_are_bounded_and_block_release(
    tmp_path: Path,
) -> None:
    _init_repository(tmp_path)
    hidden_path = "/" + "home" + "/alice/private/source.tif"
    _write_zip_container(
        tmp_path,
        "paper/oversized.docx",
        {
            "word/a.xml": "safe\n" * 32,
            "word/z.xml": "x" * 256 + hidden_path,
        },
    )
    _git(tmp_path, "add", ".")

    report = scan_repository(
        tmp_path,
        max_container_bytes=64,
        max_container_member_bytes=32,
        max_container_members=1,
    )

    assert report.container_files_scanned == 1
    assert report.container_members_scanned == 1
    assert report.errors == 0
    assert {finding.code for finding in report.findings} == {"bounded-container-scan"}
    assert should_fail(report, "warning")
    assert hidden_path not in json.dumps(report.to_dict())


def test_ignored_generic_archive_is_not_a_container_candidate(tmp_path: Path) -> None:
    _init_repository(tmp_path, include_ignore=True)
    hidden_path = "/" + "home" + "/alice/private/source.tif"
    _write_zip_container(tmp_path, "paper/release.zip", {"notes.txt": hidden_path})
    _write(tmp_path, "README.md", "# Portable fixture\n")

    report = scan_repository(tmp_path)

    assert "paper/release.zip" not in {
        finding.path for finding in report.findings
    }
    assert report.container_files_scanned == 0
    assert report.errors == 0
    assert report.warnings == 0


def test_cli_json_and_ci_failure_thresholds(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _write(tmp_path, "release.zip", b"PK fixture")
    _git(tmp_path, "add", "release.zip")

    warning_allowed = subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--root",
            str(tmp_path),
            "--format",
            "json",
            "--fail-on",
            "error",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert warning_allowed.returncode == 0
    payload = json.loads(warning_allowed.stdout)
    assert payload["schema_version"] == 1
    assert payload["summary"]["errors"] == 0
    assert payload["summary"]["warnings"] == 1

    warning_blocked = subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--root",
            str(tmp_path),
            "--fail-on",
            "warning",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert warning_blocked.returncode == 1
    assert "WARNING [archive-candidate]" in warning_blocked.stdout
