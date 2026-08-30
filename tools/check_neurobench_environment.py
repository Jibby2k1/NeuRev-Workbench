#!/usr/bin/env python3
"""Report local tools needed for the Neurobench/Fiji processing workflow."""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_FIJI = Path(os.environ.get("NEUROBENCH_FIJI", shutil.which("fiji") or "fiji"))


def status(label: str, ok: bool, detail: str) -> None:
    mark = "OK" if ok else "MISSING"
    print(f"{mark:7} {label}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Neurobench processing environment.")
    parser.add_argument("--fiji", type=Path, default=DEFAULT_FIJI)
    args = parser.parse_args()

    status("python", True, sys.executable)
    for module in ["numpy", "scipy", "tifffile", "pytest"]:
        spec = importlib.util.find_spec(module)
        status(f"python module {module}", spec is not None, "installed" if spec else "not installed")

    fiji = args.fiji if args.fiji.exists() else Path(shutil.which("fiji") or "")
    status("fiji", bool(fiji and fiji.exists()), str(fiji) if fiji else "not on PATH")
    if fiji and fiji.exists():
        try:
            result = subprocess.run(
                [str(fiji), "--headless", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            first_line = (result.stdout or result.stderr).strip().splitlines()
            detail = first_line[0] if first_line else f"exit code {result.returncode}"
            status("fiji version", result.returncode == 0, detail)
        except Exception as exc:
            status("fiji version", False, str(exc))


if __name__ == "__main__":
    main()
