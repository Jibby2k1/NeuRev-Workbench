"""Explicit optional adapter contract for the external CaImAn CNMF reference."""
from __future__ import annotations

from importlib import metadata, util
import json
import os
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_CAIMAN_PYTHON = Path(
    "/home/jibby2k1/.local/share/neurobench-caiman-1.13.1/bin/python"
)


def _external_audit(python_executable: Path) -> tuple[bool, str | None, str | None]:
    if not python_executable.is_file():
        return False, None, f"interpreter does not exist: {python_executable}"
    probe = (
        "import importlib.metadata as m,json; import caiman; "
        "from caiman.source_extraction.cnmf import cnmf; "
        "print(json.dumps({'version':m.version('caiman'),'cnmf':cnmf.CNMF.__name__}))"
    )
    environment = dict(os.environ)
    environment.update({
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
    completed = subprocess.run(
        [str(python_executable), "-c", probe], capture_output=True, text=True,
        timeout=60, check=False, env=environment,
    )
    if completed.returncode != 0:
        return False, None, completed.stderr.strip()[-1000:]
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return False, None, f"invalid probe output: {exc}"
    if payload.get("cnmf") != "CNMF":
        return False, str(payload.get("version")), "CNMF class import failed"
    return True, str(payload["version"]), None


def audit_caiman_backend(
    expected_version: str,
    *,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Report local or isolated CaImAn availability without fitting data."""
    requested = python_executable or os.environ.get("NEUROBENCH_CAIMAN_PYTHON")
    if requested is None and DEFAULT_CAIMAN_PYTHON.is_file():
        requested = DEFAULT_CAIMAN_PYTHON
    probe_error = None
    if requested is not None:
        interpreter = Path(requested).expanduser().resolve()
        available, installed_version, probe_error = _external_audit(interpreter)
        environment_kind = "isolated_external_python"
    else:
        interpreter = None
        available = util.find_spec("caiman") is not None
        installed_version = None
        if available:
            try:
                installed_version = metadata.version("caiman")
            except metadata.PackageNotFoundError:
                installed_version = "unknown"
        environment_kind = "current_python"
    frozen = bool(
        available
        and expected_version not in {"", "unfrozen_pending_install_authorization"}
        and installed_version == expected_version
    )
    return {
        "backend": "caiman", "available": available,
        "installed_version": installed_version, "expected_version": expected_version,
        "version_frozen": frozen, "fit_authorized": frozen,
        "fallback_used": False, "environment_kind": environment_kind,
        "python_executable": str(interpreter) if interpreter else None,
        "probe_error": probe_error,
        "interpretation": (
            "CaImAn CNMF is an external scientific reference. Absence or an "
            "unfrozen version is reported explicitly; ordinary NMF is not substituted."
        ),
    }


def require_caiman_backend(
    expected_version: str,
    *,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fit-ready audit or raise an actionable backend error."""
    audit = audit_caiman_backend(
        expected_version, python_executable=python_executable
    )
    if not audit["available"]:
        raise RuntimeError(
            "CaImAn is unavailable and no ordinary NMF fallback is scientifically equivalent."
        )
    if not audit["version_frozen"]:
        raise RuntimeError("CaImAn is present but its version is not frozen by the manifest.")
    return audit
