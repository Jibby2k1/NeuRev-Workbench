#!/usr/bin/env python3
"""Build a dataset index page for processed neuron workbenches."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from neurobench.data.catalog import discover_dataset_catalog

def _relative(root: Path, value: str) -> str:
    path = Path(value)
    absolute = path if path.is_absolute() else root / path
    return absolute.resolve().relative_to(root.resolve()).as_posix()


def dataset_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    records = discover_dataset_catalog(root, search_roots=[Path(".")], max_depth=5)
    for record in records:
        paths = record.get("paths") or {}
        if not (record.get("capabilities") or {}).get("review_app"):
            continue
        app_index = (root / paths["entrypoint"]).resolve()
        app_html = app_index.read_text(encoding="utf-8", errors="ignore") if app_index.exists() else ""
        has_review_subpages = "reviewStencilPage" in app_html and "reviewOverlapPage" in app_html
        video = record.get("video") or {}
        rows.append(
            {
                "dataset_id": record.get("dataset_id", ""),
                "name": record.get("name") or record.get("dataset_id", ""),
                "frames": video.get("frames", ""),
                "width": video.get("width", ""),
                "height": video.get("height", ""),
                "rois": record.get("roi_count", ""),
                "suggestions": record.get("suggestion_count", ""),
                "median_area": record.get("median_roi_area", ""),
                "app": _relative(root, paths["entrypoint"]),
                "review_data": _relative(root, paths["review_data"]),
                "annotations": _relative(root, paths["annotations"]),
                "architecture_runs": _relative(root, paths["architecture_runs"]),
                "review_stencil": _relative(root, paths["entrypoint"]) + "#review-stencil" if has_review_subpages else "",
                "review_overlap": _relative(root, paths["entrypoint"]) + "#review-overlap" if has_review_subpages else "",
            }
        )
    return rows


def render(rows: list[dict]) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
      <article class="dataset">
        <div>
          <h2>{html.escape(str(row['dataset_id']))}</h2>
          <p>{html.escape(str(row['name']))}</p>
        </div>
        <dl>
          <div><dt>Frames</dt><dd>{html.escape(str(row['frames']))}</dd></div>
          <div><dt>Size</dt><dd>{html.escape(str(row['width']))} x {html.escape(str(row['height']))}</dd></div>
          <div><dt>ROIs</dt><dd>{html.escape(str(row['rois']))}</dd></div>
          <div><dt>Suggestions</dt><dd>{html.escape(str(row['suggestions']))}</dd></div>
          <div><dt>Median area</dt><dd>{html.escape(str(row['median_area']))} px</dd></div>
        </dl>
        <nav>
          <a class="primary" href="{html.escape(row['app'])}">Open dashboard</a>
          <a href="{html.escape(row['review_data'])}">Review data</a>
          <a href="{html.escape(row['annotations'])}">Annotations</a>
          <a href="{html.escape(row['architecture_runs'])}">Pipeline runs</a>
          {f'<a href="{html.escape(row["review_stencil"])}">Review stencil</a>' if row.get('review_stencil') else ''}
          {f'<a href="{html.escape(row["review_overlap"])}">Sweep overlap</a>' if row.get('review_overlap') else ''}
        </nav>
      </article>"""
        )
    empty = """
      <section class="empty">
        <h2>No processed datasets found</h2>
        <p>Run <code>tools/run_neuron_review_pipeline.py</code> with a dataset manifest to populate this index.</p>
      </section>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neuron Workbench Datasets</title>
<style>
:root {{ color-scheme: light; --bg:#edf2f7; --panel:#fff; --ink:#102033; --muted:#667085; --line:#cbd5e1; --accent:#0369a1; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Arial, Helvetica, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px; margin-bottom:18px; }}
h1 {{ margin:0; font-size:24px; letter-spacing:0; }}
.count {{ color:var(--muted); font-weight:700; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(330px, 1fr)); gap:14px; }}
.dataset, .empty {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 8px 22px rgba(15,23,42,.07); }}
h2 {{ margin:0; font-size:17px; letter-spacing:0; }}
p {{ margin:5px 0 0; color:var(--muted); }}
dl {{ display:grid; grid-template-columns:repeat(5, 1fr); gap:8px; margin:16px 0; }}
dt {{ color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; }}
dd {{ margin:3px 0 0; font-weight:800; }}
nav {{ display:flex; flex-wrap:wrap; gap:8px; }}
a {{ color:var(--accent); border:1px solid var(--line); border-radius:6px; padding:8px 10px; text-decoration:none; font-weight:700; font-size:13px; background:#fff; }}
a.primary {{ color:#fff; background:var(--accent); border-color:var(--accent); }}
code {{ background:#eef2ff; padding:2px 5px; border-radius:4px; }}
@media (max-width:720px) {{ main {{ padding:14px; }} header {{ display:block; }} dl {{ grid-template-columns:repeat(2, 1fr); }} }}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Neuron Workbench Datasets</h1>
      <p>Open any processed sample, continue annotation, or inspect run artifacts.</p>
    </div>
    <div class="count">{len(rows)} dataset{'s' if len(rows) != 1 else ''}</div>
  </header>
  <section class="grid">
    {''.join(cards) if cards else empty}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the neuron workbench dataset index.")
    parser.add_argument("--root", type=Path, default=Path("Outputs/NeuronReview"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or (root / "index.html")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = dataset_rows(root)
    out.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {out} with {len(rows)} dataset(s)")


if __name__ == "__main__":
    main()
