from __future__ import annotations

from pathlib import Path

from tools.check_built_site_links import unresolved_references


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_built_site_link_audit_accepts_local_and_external_targets(tmp_path: Path) -> None:
    _write(
        tmp_path / "index.html",
        '<a href="guide/">Guide</a><img src="assets/figure.svg">'
        '<a href="/NeuRev-Workbench/guide/">Root guide</a>'
        '<a href="https://example.org/evidence">External</a>',
    )
    _write(tmp_path / "guide" / "index.html", '<a href="../">Home</a>')
    _write(tmp_path / "assets" / "figure.svg", "<svg></svg>")

    assert unresolved_references(tmp_path, base_path="/NeuRev-Workbench/") == []


def test_built_site_link_audit_reports_missing_and_escaping_targets(tmp_path: Path) -> None:
    _write(
        tmp_path / "nested" / "index.html",
        '<a href="missing/">Missing</a><a href="../../../private.txt">Escape</a>',
    )

    errors = unresolved_references(tmp_path)

    assert len(errors) == 2
    assert any("missing/" in error for error in errors)
    assert any("private.txt" in error for error in errors)
