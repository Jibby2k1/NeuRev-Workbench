#!/usr/bin/env python3
"""Fail when a built MkDocs site contains unresolved local links or assets."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class Reference:
    tag: str
    attribute: str
    target: str


class _ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[Reference] = []

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attribute = "href" if tag in {"a", "link"} else "src" if tag in {"img", "script"} else None
        if attribute is None:
            return
        value = dict(attributes).get(attribute)
        if value:
            self.references.append(Reference(tag, attribute, value))


def _candidate_targets(
    site: Path,
    source: Path,
    raw_path: str,
    *,
    base_path: str,
) -> list[Path]:
    decoded = unquote(raw_path)
    if decoded.startswith("/"):
        normalized_base = "/" + base_path.strip("/")
        normalized_base = normalized_base + "/" if normalized_base != "/" else normalized_base
        if not decoded.startswith(normalized_base):
            return []
        candidate = site / decoded[len(normalized_base):]
    else:
        candidate = source.parent / decoded
    candidates = [candidate]
    if decoded.endswith("/") or candidate.is_dir():
        candidates.append(candidate / "index.html")
    elif not candidate.suffix:
        candidates.extend((candidate / "index.html", candidate.with_suffix(".html")))
    return candidates


def unresolved_references(site_dir: Path, *, base_path: str = "/") -> list[str]:
    """Return deterministic diagnostics for missing local built-site targets."""
    site = site_dir.resolve(strict=True)
    errors: list[str] = []
    for source in sorted(site.rglob("*.html")):
        parser = _ReferenceParser()
        parser.feed(source.read_text(encoding="utf-8", errors="replace"))
        for reference in parser.references:
            parsed = urlsplit(reference.target)
            if parsed.scheme or parsed.netloc or reference.target.startswith(("mailto:", "javascript:", "data:")):
                continue
            if not parsed.path:
                continue
            candidates = _candidate_targets(
                site,
                source,
                parsed.path,
                base_path=base_path,
            )
            safe_candidates = []
            for candidate in candidates:
                resolved = candidate.resolve(strict=False)
                try:
                    resolved.relative_to(site)
                except ValueError:
                    continue
                safe_candidates.append(resolved)
            if not safe_candidates or not any(candidate.is_file() for candidate in safe_candidates):
                relative = source.relative_to(site).as_posix()
                errors.append(
                    f"{relative}: <{reference.tag} {reference.attribute}> unresolved local target {reference.target!r}"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument(
        "--base-path",
        default="/",
        help="URL path prefix configured by site_url (for example /NeuRev-Workbench/)",
    )
    args = parser.parse_args(argv)
    try:
        errors = unresolved_references(args.site_dir, base_path=args.base_path)
    except FileNotFoundError:
        parser.error(f"built site directory does not exist: {args.site_dir}")
    if errors:
        print("Built documentation contains unresolved local references:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"built documentation links valid: {args.site_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
