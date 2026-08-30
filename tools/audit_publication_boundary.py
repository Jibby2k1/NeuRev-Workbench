#!/usr/bin/env python3
"""Read-only audit of files that are candidates for publication through Git.

The audit considers tracked files plus untracked files that are not excluded by
Git ignore rules. It never follows symlinks and deliberately reports finding
metadata without echoing matched content or possible credential values.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Iterable, Sequence
import zipfile


SCHEMA_VERSION = 1
DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024
# Fully inspect every text candidate that is still within the default public
# file-size envelope. Larger candidates receive both size and bounded-scan
# warnings and require explicit release review.
DEFAULT_MAX_TEXT_BYTES = DEFAULT_MAX_FILE_BYTES
DEFAULT_MAX_CONTAINER_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_CONTAINER_MEMBER_BYTES = DEFAULT_MAX_CONTAINER_BYTES
DEFAULT_MAX_CONTAINER_MEMBERS = 2048

SEVERITY_ORDER = {"error": 0, "warning": 1}
FAIL_ORDER = {"never": 2, "error": 0, "warning": 1}

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar")
ZIP_CONTAINER_SUFFIXES = (".docx", ".npz", ".pptx", ".xlsx")
ZIP_CONTAINER_TEXT_MEMBER_SUFFIXES = (
    ".csv",
    ".json",
    ".md",
    ".rels",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
)
LOCAL_ROOTS = {"Inputs", "Outputs"}
PRIVATE_PATH_MARKERS = (
    "private_administrator",
    "private_admin",
    "randomization_key",
    "reviewer_submission",
    "reviewer_response",
    "user_review",
    "reference_single_reviewer",
)

ABSOLUTE_HOME_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])/(?:home|Users)/[A-Za-z0-9._-]+/"),
    re.compile(r"\b[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+[\\/]"),
)

SECRET_PATTERNS = (
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    (
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    (
        "aws-access-key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (
        "google-api-key",
        re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"),
    ),
    (
        "slack-token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    ),
    (
        "credential-assignment",
        re.compile(
            r"(?ix)\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|"
            r"auth[_-]?token|password)\b\s*[:=]\s*[\"']?"
            r"(?!example\b|sample\b|test\b|dummy\b|replace\b|changeme\b|"
            r"your\b|<|\$\{|\{\{)[A-Za-z0-9/+=_.-]{12,}"
        ),
    ),
)


@dataclass(frozen=True)
class Candidate:
    path: str
    tracked: bool


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    path: str
    message: str
    line: int | None = None


@dataclass
class AuditReport:
    root: str
    files_considered: int
    tracked_files: int
    untracked_files: int
    text_files_scanned: int
    binary_files_skipped: int
    container_files_scanned: int
    container_members_scanned: int
    bytes_scanned: int
    findings: list[Finding]

    @property
    def errors(self) -> int:
        return sum(finding.severity == "error" for finding in self.findings)

    @property
    def warnings(self) -> int:
        return sum(finding.severity == "warning" for finding in self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "root": self.root,
            "summary": {
                "files_considered": self.files_considered,
                "tracked_files": self.tracked_files,
                "untracked_files": self.untracked_files,
                "text_files_scanned": self.text_files_scanned,
                "binary_files_skipped": self.binary_files_skipped,
                "container_files_scanned": self.container_files_scanned,
                "container_members_scanned": self.container_members_scanned,
                "bytes_scanned": self.bytes_scanned,
                "errors": self.errors,
                "warnings": self.warnings,
            },
            "findings": [
                {key: value for key, value in asdict(finding).items() if value is not None}
                for finding in self.findings
            ],
        }


class PublicationAuditError(RuntimeError):
    """Raised when Git cannot provide the publication candidate set."""


@dataclass(frozen=True)
class _ContainerScanResult:
    valid: bool
    bytes_scanned: int = 0
    members_scanned: int = 0
    home_path_found: bool = False
    secret_kind: str | None = None
    bounded: bool = False
    unreadable_member: bool = False


def _git_paths(root: Path, *arguments: str) -> set[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise PublicationAuditError("Git is not available") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise PublicationAuditError(f"Git candidate enumeration failed: {detail}") from exc

    return {
        os.fsdecode(raw_path)
        for raw_path in completed.stdout.split(b"\0")
        if raw_path
    }


def publication_candidates(root: Path) -> list[Candidate]:
    """Return tracked plus untracked/non-ignored paths without traversing trees."""

    tracked = _git_paths(root, "--cached")
    untracked = _git_paths(root, "--others", "--exclude-standard")
    return [
        Candidate(path=path, tracked=path in tracked)
        for path in sorted(tracked | untracked)
    ]


def _is_archive(path: str) -> bool:
    lowered = path.lower()
    return any(lowered.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _zip_container_suffix(path: str) -> str | None:
    suffix = PurePosixPath(path).suffix.lower()
    return suffix if suffix in ZIP_CONTAINER_SUFFIXES else None


def _is_private_payload(path: str) -> bool:
    lowered = path.lower()
    basename = PurePosixPath(path).name.lower()
    return any(marker in lowered for marker in PRIVATE_PATH_MARKERS) or bool(
        re.fullmatch(r"annotations_reviewer_[^/]+\.json", basename)
    )


def _is_credential_path(path: str) -> bool:
    basename = PurePosixPath(path).name.lower()
    if basename in {".env.example", ".env.template"}:
        return False
    return (
        basename in {".env", "credentials.json", "secrets.json"}
        or basename.startswith(".env.")
        or basename.startswith("service-account")
        or basename.startswith("service_account")
        or basename.endswith((".pem", ".key"))
    )


def _looks_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\0" in data:
        return True
    text_bytes = b"\n\r\t\f\b" + bytes(range(32, 127))
    non_text = sum(byte not in text_bytes for byte in data)
    return non_text / len(data) > 0.30


def _first_matching_line(text: str, patterns: Iterable[re.Pattern[str]]) -> int | None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in patterns):
            return line_number
    return None


def _first_secret(text: str) -> tuple[str, int] | None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                return name, line_number
    return None


def _sensitive_content_matches(data: bytes) -> tuple[bool, str | None]:
    """Return bounded sensitive-content classifications without matched values."""

    variants = [data.decode("utf-8", errors="replace")]
    # NumPy fixed-width Unicode arrays encode ASCII characters with intervening
    # NUL bytes. Inspecting a NUL-stripped variant catches paths and credentials
    # without importing NumPy or deserializing/pickling an array.
    if b"\0" in data:
        variants.append(data.replace(b"\0", b"").decode("utf-8", errors="replace"))

    home_path_found = False
    secret_kind: str | None = None
    for text in variants:
        if not home_path_found:
            home_path_found = _first_matching_line(text, ABSOLUTE_HOME_PATTERNS) is not None
        if secret_kind is None:
            secret = _first_secret(text)
            if secret is not None:
                secret_kind = secret[0]
        if home_path_found and secret_kind is not None:
            break
    return home_path_found, secret_kind


def _eligible_container_member(container_suffix: str, member_name: str) -> bool:
    pure = PurePosixPath(member_name)
    if not pure.name:
        return False
    if container_suffix == ".npz":
        # NPZ members are normally NPY arrays. Read their raw bounded bytes so
        # fixed-width strings and object metadata can be classified without
        # loading or executing them.
        return True
    return pure.suffix.lower() in ZIP_CONTAINER_TEXT_MEMBER_SUFFIXES


def _scan_zip_container(
    data: bytes,
    *,
    container_suffix: str,
    max_bytes: int,
    max_member_bytes: int,
    max_members: int,
) -> _ContainerScanResult:
    """Inspect bounded member bytes from a supported ZIP-container candidate."""

    try:
        archive = zipfile.ZipFile(io.BytesIO(data), mode="r")
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return _ContainerScanResult(valid=False)

    bytes_scanned = 0
    members_scanned = 0
    home_path_found = False
    secret_kind: str | None = None
    bounded = False
    unreadable_member = False

    with archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        eligible = [
            member
            for member in members
            if _eligible_container_member(container_suffix, member.filename)
        ]
        eligible.sort(key=lambda member: member.filename)
        if len(members) > max_members or len(eligible) > max_members:
            bounded = True
        eligible = eligible[:max_members]

        for member in eligible:
            remaining = max_bytes - bytes_scanned
            if remaining <= 0:
                bounded = True
                break

            # Member names are content too, but never include them in findings.
            name_home, name_secret = _sensitive_content_matches(
                member.filename.encode("utf-8", errors="replace")
            )
            home_path_found = home_path_found or name_home
            if secret_kind is None:
                secret_kind = name_secret

            limit = min(max_member_bytes, remaining)
            try:
                with archive.open(member, mode="r") as member_stream:
                    member_data = member_stream.read(limit + 1)
            except (
                EOFError,
                NotImplementedError,
                OSError,
                RuntimeError,
                ValueError,
                zipfile.BadZipFile,
            ):
                unreadable_member = True
                continue

            if len(member_data) > limit:
                member_data = member_data[:limit]
                bounded = True
            if member.file_size > limit:
                bounded = True

            bytes_scanned += len(member_data)
            members_scanned += 1
            member_home, member_secret = _sensitive_content_matches(member_data)
            home_path_found = home_path_found or member_home
            if secret_kind is None:
                secret_kind = member_secret

        if len(eligible) > members_scanned and bytes_scanned >= max_bytes:
            bounded = True

    return _ContainerScanResult(
        valid=True,
        bytes_scanned=bytes_scanned,
        members_scanned=members_scanned,
        home_path_found=home_path_found,
        secret_kind=secret_kind,
        bounded=bounded,
        unreadable_member=unreadable_member,
    )


def _safe_candidate_path(root: Path, candidate_path: str) -> Path | None:
    pure = PurePosixPath(candidate_path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        return None
    return root.joinpath(*pure.parts)


def _read_bounded(path: Path, limit: int) -> tuple[bytes, bool]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("candidate is not a regular file")
        data = bytearray()
        while len(data) <= limit:
            block = os.read(descriptor, min(64 * 1024, limit + 1 - len(data)))
            if not block:
                break
            data.extend(block)
        return bytes(data[:limit]), len(data) > limit
    finally:
        os.close(descriptor)


def scan_repository(
    root: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES,
    max_container_bytes: int = DEFAULT_MAX_CONTAINER_BYTES,
    max_container_member_bytes: int = DEFAULT_MAX_CONTAINER_MEMBER_BYTES,
    max_container_members: int = DEFAULT_MAX_CONTAINER_MEMBERS,
) -> AuditReport:
    """Audit publication candidates under ``root`` without changing them."""

    root = root.resolve()
    candidates = publication_candidates(root)
    findings: list[Finding] = []
    text_files_scanned = 0
    binary_files_skipped = 0
    container_files_scanned = 0
    container_members_scanned = 0
    bytes_scanned = 0

    def add(severity: str, code: str, path: str, message: str, line: int | None = None) -> None:
        findings.append(Finding(severity, code, path, message, line))

    for candidate in candidates:
        display_path = candidate.path
        pure = PurePosixPath(display_path)
        local_path = _safe_candidate_path(root, display_path)
        if local_path is None:
            add(
                "error",
                "unsafe-git-path",
                display_path,
                "Git returned a path outside the repository-relative path contract.",
            )
            continue

        if candidate.tracked and pure.parts and pure.parts[0] in LOCAL_ROOTS:
            add(
                "error",
                "tracked-local-artifact",
                display_path,
                "A local Inputs/ or Outputs/ artifact is already tracked by Git.",
            )
        if _is_private_payload(display_path):
            add(
                "error",
                "private-review-payload",
                display_path,
                "A reviewer response, identity mapping, or administrator-only payload is a publication candidate.",
            )
        if _is_credential_path(display_path):
            add(
                "error",
                "credential-file",
                display_path,
                "A credential-like filename is a publication candidate.",
            )
        if _is_archive(display_path):
            add(
                "warning",
                "archive-candidate",
                display_path,
                "An archive is tracked or not covered by ignore rules; publish it through a release store instead.",
            )

        try:
            metadata = local_path.lstat()
        except FileNotFoundError:
            add(
                "warning",
                "missing-worktree-path",
                display_path,
                "Git lists this path, but it is absent from the current worktree.",
            )
            continue
        except OSError:
            add(
                "warning",
                "unreadable-path",
                display_path,
                "The candidate metadata could not be read safely.",
            )
            continue

        if stat.S_ISLNK(metadata.st_mode):
            try:
                target = os.readlink(local_path)
            except OSError:
                target = ""
            if os.path.isabs(target):
                add(
                    "error",
                    "absolute-symlink",
                    display_path,
                    "A publication candidate symlink has an absolute, workstation-bound target.",
                )
            else:
                add(
                    "warning",
                    "relative-symlink",
                    display_path,
                    "A publication candidate is a symlink; verify its target exists in a clean checkout.",
                )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            add(
                "warning",
                "non-regular-candidate",
                display_path,
                "The publication candidate is not a regular file and was not inspected.",
            )
            continue

        if metadata.st_size > max_file_bytes:
            add(
                "warning",
                "unexpectedly-large-file",
                display_path,
                f"Candidate size exceeds the configured {max_file_bytes}-byte Git threshold.",
            )

        try:
            data, truncated = _read_bounded(local_path, max_text_bytes)
        except OSError:
            add(
                "warning",
                "unreadable-file",
                display_path,
                "The candidate content could not be opened without following symlinks.",
            )
            continue

        bytes_scanned += len(data)
        container_suffix = _zip_container_suffix(display_path)
        if container_suffix is not None:
            if truncated:
                add(
                    "warning",
                    "bounded-container-scan",
                    display_path,
                    "The compressed ZIP container exceeds the configured read limit and was not inspected.",
                )
                binary_files_skipped += 1
                continue

            container_result = _scan_zip_container(
                data,
                container_suffix=container_suffix,
                max_bytes=max_container_bytes,
                max_member_bytes=max_container_member_bytes,
                max_members=max_container_members,
            )
            if not container_result.valid:
                add(
                    "warning",
                    "invalid-zip-container",
                    display_path,
                    "The file uses a supported ZIP-container extension but is not a readable ZIP container.",
                )
                # A mislabeled plain-text file still receives the ordinary text
                # checks below; a malformed binary is counted as skipped there.
            else:
                container_files_scanned += 1
                container_members_scanned += container_result.members_scanned
                bytes_scanned += container_result.bytes_scanned
                if container_result.bounded:
                    add(
                        "warning",
                        "bounded-container-scan",
                        display_path,
                        "ZIP-container inspection reached a configured member or byte limit; matched content remains suppressed.",
                    )
                if container_result.unreadable_member:
                    add(
                        "warning",
                        "unreadable-container-member",
                        display_path,
                        "At least one eligible ZIP-container member could not be read safely; member details are suppressed.",
                    )
                if container_result.home_path_found:
                    add(
                        "error",
                        "absolute-home-path",
                        display_path,
                        "A workstation-specific user-home path appears in an inspected ZIP-container member; matched content and member name are suppressed.",
                    )
                if container_result.secret_kind is not None:
                    add(
                        "error",
                        "likely-secret",
                        display_path,
                        f"An inspected ZIP-container member matches the {container_result.secret_kind} credential signature; the value and member name are suppressed.",
                    )
                continue

        if _looks_binary(data[:8192]):
            binary_files_skipped += 1
            continue
        if truncated:
            add(
                "warning",
                "bounded-text-scan",
                display_path,
                f"Only the first {max_text_bytes} bytes were inspected; the text candidate is larger.",
            )

        text_files_scanned += 1
        text = data.decode("utf-8", errors="replace")
        home_line = _first_matching_line(text, ABSOLUTE_HOME_PATTERNS)
        if home_line is not None:
            add(
                "error",
                "absolute-home-path",
                display_path,
                "A workstation-specific user-home path appears in publishable text.",
                home_line,
            )
        secret = _first_secret(text)
        if secret is not None:
            secret_kind, secret_line = secret
            add(
                "error",
                "likely-secret",
                display_path,
                f"Content matches the {secret_kind} credential signature; the value is intentionally suppressed.",
                secret_line,
            )

    findings.sort(
        key=lambda finding: (
            SEVERITY_ORDER[finding.severity],
            finding.code,
            finding.path,
            finding.line or 0,
        )
    )
    return AuditReport(
        root=str(root),
        files_considered=len(candidates),
        tracked_files=sum(candidate.tracked for candidate in candidates),
        untracked_files=sum(not candidate.tracked for candidate in candidates),
        text_files_scanned=text_files_scanned,
        binary_files_skipped=binary_files_skipped,
        container_files_scanned=container_files_scanned,
        container_members_scanned=container_members_scanned,
        bytes_scanned=bytes_scanned,
        findings=findings,
    )


def _path_for_terminal(path: str) -> str:
    return json.dumps(path, ensure_ascii=True)


def render_text(report: AuditReport) -> str:
    lines = [
        "NeuRev publication-boundary audit",
        f"root: {report.root}",
        (
            "summary: "
            f"{report.files_considered} candidates, {report.errors} errors, "
            f"{report.warnings} warnings"
        ),
    ]
    for finding in report.findings:
        location = _path_for_terminal(finding.path)
        if finding.line is not None:
            location = f"{location}:{finding.line}"
        lines.append(
            f"{finding.severity.upper()} [{finding.code}] {location} - {finding.message}"
        )
    if not report.findings:
        lines.append("PASS: no publication-boundary findings")
    return "\n".join(lines)


def should_fail(report: AuditReport, fail_on: str) -> bool:
    if fail_on == "never":
        return False
    threshold = FAIL_ORDER[fail_on]
    return any(SEVERITY_ORDER[finding.severity] <= threshold for finding in report.findings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Git worktree to audit")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report format written to standard output",
    )
    parser.add_argument(
        "--fail-on",
        choices=("error", "warning", "never"),
        default="error",
        help="Lowest severity that produces exit status 1",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Warn when a candidate exceeds this size",
    )
    parser.add_argument(
        "--max-text-bytes",
        type=int,
        default=DEFAULT_MAX_TEXT_BYTES,
        help="Maximum bytes read from any one text candidate",
    )
    parser.add_argument(
        "--max-container-bytes",
        type=int,
        default=DEFAULT_MAX_CONTAINER_BYTES,
        help="Maximum expanded member bytes inspected per supported ZIP container",
    )
    parser.add_argument(
        "--max-container-member-bytes",
        type=int,
        default=DEFAULT_MAX_CONTAINER_MEMBER_BYTES,
        help="Maximum expanded bytes inspected from one ZIP-container member",
    )
    parser.add_argument(
        "--max-container-members",
        type=int,
        default=DEFAULT_MAX_CONTAINER_MEMBERS,
        help="Maximum ZIP-container members considered per candidate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.max_file_bytes <= 0
        or args.max_text_bytes <= 0
        or args.max_container_bytes <= 0
        or args.max_container_member_bytes <= 0
        or args.max_container_members <= 0
    ):
        parser.error("scan limits must be positive integers")

    try:
        report = scan_repository(
            args.root,
            max_file_bytes=args.max_file_bytes,
            max_text_bytes=args.max_text_bytes,
            max_container_bytes=args.max_container_bytes,
            max_container_member_bytes=args.max_container_member_bytes,
            max_container_members=args.max_container_members,
        )
    except PublicationAuditError as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "root": str(args.root.resolve()),
            "summary": {"errors": 1, "warnings": 0},
            "findings": [
                {
                    "severity": "error",
                    "code": "git-enumeration-failed",
                    "path": ".",
                    "message": str(exc),
                }
            ],
        }
        if args.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"ERROR [git-enumeration-failed] {_path_for_terminal('.')} - {exc}")
        return 1

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return int(should_fail(report, args.fail_on))


if __name__ == "__main__":
    sys.exit(main())
