from __future__ import annotations

import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CONFIG = ROOT / "mkdocs.yml"
LANDING = DOCS / "index.md"


def _nav_targets(items: list[object]) -> list[str]:
    targets: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for value in item.values():
            if isinstance(value, str):
                targets.append(value)
            elif isinstance(value, list):
                targets.extend(_nav_targets(value))
    return targets


def _markdown_targets(text: str) -> list[str]:
    return re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", text)


def test_mkdocs_configuration_is_searchable_strict_and_local() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert payload["strict"] is True
    assert payload["docs_dir"] == "docs"
    assert payload["theme"]["name"] == "material"
    assert payload["theme"]["font"] is False
    assert payload["theme"]["palette"] == [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "custom",
            "accent": "custom",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "custom",
            "accent": "custom",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to light mode",
            },
        },
    ]
    assert "search.highlight" in payload["theme"]["features"]
    assert "search.suggest" in payload["theme"]["features"]
    assert any("search" in plugin for plugin in payload["plugins"])
    assert payload["validation"]["links"]["not_found"] == "warn"
    assert payload["validation"]["links"]["unrecognized_links"] == "warn"
    assert payload["validation"]["links"]["anchors"] == "warn"
    excluded = set(payload["exclude_docs"].splitlines())
    assert {"/README.md", "/CODEBASE_AUDIT.md", "/plan.md", "archive/plans/"} <= excluded

    for target in _nav_targets(payload["nav"]):
        if "://" in target:
            assert target.startswith("https://github.com/Jibby2k1/NeuRev-Workbench")
            continue
        path = PurePosixPath(target)
        assert ".." not in path.parts, target
        assert (DOCS / path).is_file(), target

    for stylesheet in payload["extra_css"]:
        assert (DOCS / stylesheet).is_file(), stylesheet


def test_landing_page_links_are_clean_clone_safe_and_images_have_alt_text() -> None:
    text = LANDING.read_text(encoding="utf-8")
    assert "../" not in text
    assert "https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/" in text

    for target in _markdown_targets(text):
        target = target.split("#", 1)[0]
        if not target:
            continue
        if "://" in target:
            assert target.startswith("https://github.com/Jibby2k1/NeuRev-Workbench")
            continue
        path = PurePosixPath(target)
        assert ".." not in path.parts, target
        assert (DOCS / path).exists(), target

    images = re.findall(r"!\[([^]]*)\]\(([^)]+)\)", text)
    assert images
    assert all(alt.strip() for alt, _ in images)


def test_site_styles_preserve_reserved_scientific_colors() -> None:
    stylesheet = (DOCS / "stylesheets" / "extra.css").read_text(encoding="utf-8")
    for reserved in ("#2f9d67", "#7bd7a5", "#d97706", "#ffad5c", "#f4e7a1"):
        assert reserved not in stylesheet.lower()
    assert "prefers-reduced-motion" in stylesheet
    assert ":focus-visible" in stylesheet
    assert ".neurev-hero .md-button" in stylesheet
    assert "color: var(--md-accent-fg-color)" in stylesheet
    assert "max-width: 88rem" in stylesheet


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    high, low = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_diagrams_are_accessible_and_keep_small_accent_text_above_aa_contrast() -> None:
    diagrams = DOCS / "assets" / "diagrams"
    for name in ("repository-evidence-flow.svg", "experiment-lifecycle.svg"):
        path = diagrams / name
        text = path.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        assert root.attrib["role"] == "img"
        assert root.attrib["aria-labelledby"]
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        assert root.find("{http://www.w3.org/2000/svg}desc") is not None
        assert "--accent-text: #0f6670" in text
        for reserved in ("#2f9d67", "#7bd7a5", "#d97706", "#ffad5c", "#f4e7a1"):
            assert reserved not in text.lower()

    for background in ("#ddf4f5", "#e8eefb", "#f7fafb"):
        assert _contrast("#0f6670", background) >= 4.5


def test_docs_extra_and_ci_build_only_contract() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]["docs"]
    assert any(requirement.startswith("mkdocs>=") for requirement in extras)
    assert any(requirement.startswith("mkdocs-material>=") for requirement in extras)

    workflow = (ROOT / ".github" / "workflows" / "documentation.yml").read_text(
        encoding="utf-8"
    )
    assert "mkdocs build --clean --strict" in workflow
    assert "check_built_site_links.py" in workflow
    assert "--base-path /NeuRev-Workbench/" in workflow
    assert "contents: read" in workflow
    assert "pages: write" not in workflow
    assert "deploy" not in workflow.lower()
    assert "peaceiris/actions-gh-pages" not in workflow
