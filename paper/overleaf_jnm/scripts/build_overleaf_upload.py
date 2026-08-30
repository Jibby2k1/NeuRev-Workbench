#!/usr/bin/env python3
"""Build a clean, deterministic Overleaf upload ZIP from this directory."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "NeuRev_JNM_Overleaf_Working_Package.zip"
INCLUDE_SUFFIXES = {".tex", ".bib", ".png", ".pdf", ".txt", ".md", ".json", ".yaml", ".tsv"}
EXCLUDE_NAMES = {"main.pdf", "supplement_main.pdf"}
EXCLUDE_PARTS = {".git", "__pycache__"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path.suffix.lower() in INCLUDE_SUFFIXES
        and path.name not in EXCLUDE_NAMES
        and not any(part in EXCLUDE_PARTS for part in relative.parts)
    )


def main() -> int:
    files = sorted((path for path in ROOT.rglob("*") if included(path)), key=lambda path: path.relative_to(ROOT).as_posix())
    temporary = OUTPUT.with_suffix(".partial.zip")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = ZipInfo(relative, date_time=(2026, 8, 23, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
    temporary.replace(OUTPUT)
    print(f"{OUTPUT}\t{len(files)} files\t{OUTPUT.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
