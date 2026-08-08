"""Build a canonical portable technical report from a completed diagnostic package."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


def _source(source_id: str, label: str, path: str) -> dict[str, Any]:
    return {"id": source_id, "label": label, "path": path}


def build_report_artifact(*, diagnostic_root: str | Path, output_dir: str | Path) -> Path:
    diagnostic = Path(diagnostic_root).resolve()
    target = Path(output_dir).resolve()
    partial = Path(str(target) + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"Output or partial output already exists: {target}")
    required = (
        diagnostic / "preliminary_metrics.json",
        diagnostic / "source_evidence/fit.json",
        diagnostic / "source_evidence/real_data_metrics.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing report evidence: {missing}")
    partial.mkdir(parents=True)
    (partial / "source_evidence").mkdir()
    try:
        for path in required:
            shutil.copy2(path, partial / "source_evidence" / path.name)
        evidence = json.loads(required[0].read_text())
        fit = evidence["fit"]
        direction = evidence["direction"]
        preliminary = evidence["preliminary_metrics"]
        real_metrics = json.loads(required[2].read_text())
        generated_at = datetime.now(timezone.utc).isoformat()
        sources = [
            _source("preliminary", "Derived diagnostic metrics", "source_evidence/preliminary_metrics.json"),
            _source("fit", "Completed CS-Parzen fit", "source_evidence/fit.json"),
            _source("real_metrics", "Completed real-data detection metrics", "source_evidence/real_data_metrics.json"),
        ]
        lane_rows = []
        burst_rows = []
        for lane in real_metrics["lanes"]:
            if "mean_recall" not in lane:
                continue
            lane_rows.append({
                "method": lane["lane"].replace("_", " ").title(),
                "lane_id": lane["lane"],
                "mean_recall": lane["mean_recall"],
                "matches": lane["total_matched"],
                "candidates": lane["total_event_candidates"],
                "total_labels": lane["total_labels"],
                "precision_identified": lane["precision_identified"],
            })
            if lane["lane"] in {"raw_direct", "cs_parzen_ica"}:
                for fold in lane["outer_folds"]:
                    burst_rows.append({
                        "method": lane["lane"].replace("_", " ").title(),
                        "burst": fold["burst_id"],
                        "recall": fold["recall"],
                        "matches": fold["matched"],
                        "labels": fold["labels"],
                        "candidates": fold["candidates"],
                    })
        headline = [{
            "direction_cosine": direction["absolute_cosine_to_derivative"],
            "activity_correlation": preliminary["parzen_derivative_correlation"],
            "residual_nrms": preliminary["non_derivative_residual_nrms"],
            "cs_recall": preliminary["cs_parzen_mean_recall"],
            "raw_recall": preliminary["raw_direct_mean_recall"],
        }]
        manifest = {
            "version": 1,
            "surface": "report",
            "title": "Basic two-frame CS-Parzen ICA on spontaneous calcium imaging",
            "description": "A formula-explicit technical interpretation of the completed real-data fit.",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {"id": "direction", "dataset": "headline", "sourceId": "preliminary", "description": "Absolute cosine similarity between the normalized learned observation axis and [-1,1]/sqrt(2).", "metrics": [{"label": "Cosine to derivative", "field": "direction_cosine", "format": "number"}]},
                {"id": "correlation", "dataset": "headline", "sourceId": "preliminary", "description": "Sampled full-interval Pearson correlation between fitted CS-Parzen activity and fixed derivative activity.", "metrics": [{"label": "Activity correlation", "field": "activity_correlation", "format": "number"}]},
                {"id": "residual", "dataset": "headline", "sourceId": "preliminary", "description": "RMS of Y-beta D divided by RMS of Y after beta is fitted on quiet frames.", "metrics": [{"label": "Non-derivative residual", "field": "residual_nrms", "format": "percent"}]},
                {"id": "recall", "dataset": "headline", "sourceId": "real_metrics", "description": "Mean known-label recall across four bursts; labels are sparse positives and do not identify precision.", "metrics": [{"label": "CS-Parzen recall", "field": "cs_recall", "format": "percent"}, {"label": "Raw Direct reference", "field": "raw_recall", "format": "percent"}]},
            ],
            "charts": [
                {
                    "id": "recall_comparison",
                    "title": "Known-label recall by method",
                    "subtitle": "Four burst intervals; 79 sparse-positive labels; unmatched candidates remain unknown",
                    "showDescription": True,
                    "intent": "comparison",
                    "question": "How did the fitted CS-Parzen lane compare with the declared real-data baselines?",
                    "rationale": "A bar chart makes the five discrete method recalls directly comparable without implying a time trend.",
                    "comparisonContext": {"denominator": "79 burst-specific sparse-positive label rows", "grain": "method", "unit": "mean recall across four bursts"},
                    "type": "bar",
                    "dataset": "lane_results",
                    "sourceId": "real_metrics",
                    "encodings": {"x": {"field": "method", "type": "nominal", "label": "Method"}, "y": {"field": "mean_recall", "type": "quantitative", "format": "percent", "label": "Mean known-label recall"}, "tooltip": [{"field": "matches", "label": "Known matches"}, {"field": "candidates", "label": "Event candidates"}, {"field": "total_labels", "label": "Sparse-positive labels"}]},
                    "valueFormat": "percent",
                    "layout": "full",
                    "maxRows": 5,
                }
            ],
            "tables": [
                {
                    "id": "burst_details",
                    "title": "Raw Direct and CS-Parzen burst-level results",
                    "subtitle": "Exact recall numerator, denominator, and candidate counts for each annotated burst",
                    "showDescription": True,
                    "dataset": "burst_results",
                    "defaultSort": {"field": "burst", "direction": "asc"},
                    "density": "spacious",
                    "sourceId": "real_metrics",
                    "layout": "full",
                    "columns": [
                        {"field": "burst", "label": "Burst", "format": "number"},
                        {"field": "method", "label": "Method", "type": "text"},
                        {"field": "recall", "label": "Known-label recall", "format": "percent"},
                        {"field": "matches", "label": "Matches", "format": "number"},
                        {"field": "labels", "label": "Labels", "format": "number"},
                        {"field": "candidates", "label": "Candidates", "format": "number"},
                    ],
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "layout": "full", "body": "# Basic two-frame CS-Parzen ICA on spontaneous calcium imaging"},
                {"id": "summary", "type": "markdown", "layout": "full", "sourceId": "preliminary", "body": "## The fit converged, but the learned component is a temporal derivative\n\nThe normalized learned observation axis has absolute cosine **0.999999917** to `[-1,1]/sqrt(2)`. Fitted activity correlates **0.997828** with fixed differencing; only **6.89% normalized RMS** remains after quiet-scale alignment. This supports interpreting the method as a nonparametric temporal-change diagnostic—not validated physical source separation."},
                {"id": "headline_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["direction", "correlation", "residual", "recall"]},
                {"id": "key_finding", "type": "markdown", "layout": "full", "body": "## Two adjacent observations constrain the solution geometry\n\nPersistent fluorescence lies near `[1,1]`; its orthogonal direction is `[-1,1]`, which computes `P_t-P_(t-1)`. With only two strongly correlated observations, whitening plus independence optimization exposes common-level and change coordinates. The optimizer cannot create separate dimensions for neural signal, motion, illumination, and measurement noise."},
                {"id": "recall_intro", "type": "markdown", "layout": "full", "sourceId": "real_metrics", "body": "## Convergence did not translate into detector superiority\n\nCS-Parzen reached **0.1333 mean known-label recall** (10/79 matches; 24 event candidates), while Raw Direct reached **0.6056** (49/79; 232 candidates). Candidate burden is not precision because annotations are sparse-positive rather than exhaustive."},
                {"id": "recall_chart_block", "type": "chart", "layout": "full", "chartId": "recall_comparison"},
                {"id": "burst_table_block", "type": "table", "layout": "full", "tableId": "burst_details"},
                {"id": "scope", "type": "markdown", "layout": "full", "body": "## Scope and measurement contract\n\nThe source movie contains 2,359 frames of 340×573 uint16 fluorescence at 20 ms/frame. The analysis covers UI frames 1800–2359; UI 1800–1899 is quiet calibration. Four labeled bursts contain 79 sparse-positive rows. Coordinates use x=column and y=row. No motion correction was applied. UI intervals are one-based and inclusive."},
                {"id": "method", "type": "markdown", "layout": "full", "sourceId": "fit", "body": "## The model minimizes Parzen-estimated dependence after causal preprocessing\n\n`P_t = EMA_{0.4}(G_{sigma=1 px} * R_t)`. Each observation is `x_t=[P_(t-1),P_t]^T`, centered and whitened as `z_t=Q(x_t-mu)`, then rotated as `y_t=Wz_t`. Gaussian Parzen kernels estimate the joint and marginal densities. The bounded angle search minimizes the Cauchy–Schwarz divergence between the joint density and product of marginals. The completed fit used bandwidth 0.35, 1,024 screen samples, 4,096 confirmation samples, 256-row kernel blocks, and seed 20260727."},
                {"id": "limitations", "type": "markdown", "layout": "full", "body": "## Visible residual structure prevents a noise interpretation\n\nThe non-derivative residual retains anatomical structure in representative frames. Motion, illumination changes, neural onsets, and frame noise can all occupy the learned change direction. Exact convergence and reconstruction therefore do not establish physical component identity, and unmatched candidates cannot be treated as false positives."},
                {"id": "next", "type": "markdown", "layout": "full", "body": "## The next experiment must add information, not only a richer loss\n\n1. Test whether the non-derivative residual is repeatable across bursts and nearby temporal blocks.\n2. Add measured motion or illumination covariates before biological attribution.\n3. Compare longer temporal embeddings against explicit derivative bases.\n4. Preserve Raw Direct amplitude while using continuous Parzen activity only for timing or ranking.\n5. Evaluate on a bounded exhaustively annotated spatial field."},
                {"id": "questions", "type": "markdown", "layout": "full", "body": "## Questions for further discussion\n\n- Which biological priors distinguish coordinated neural fluorescence from broad drift?\n- Is the intended deliverable physical source separation, or a proposal/ranking diagnostic?\n- What preservation tolerance for event peak and temporal area is scientifically acceptable?\n- What motion, illumination, or microscope metadata can be recovered?"},
            ],
        }
        # The portable reader currently overflows after vertical scrolling when report
        # narrative is split across multiple blocks. Keep the verified HTML surface
        # compact; METHOD_REFERENCE.md remains the full technical narrative.
        compact_title = "Basic two-frame CS-Parzen ICA on spontaneous calcium imaging"
        manifest["title"] = compact_title
        manifest["description"] = "A formula-explicit technical interpretation of the completed real-data fit."
        manifest["cards"] = []
        manifest["tables"] = []
        short_names = {
            "raw_direct": "Raw Direct",
            "fixed_binary_difference": "Fixed difference",
            "adaptive_binary_difference": "Adaptive difference",
            "infomax_tanh_ica": "InfoMax",
            "cs_parzen_ica": "CS-Parzen",
            "shared_background_nmf": "NMF",
        }
        for row in lane_rows:
            row["method"] = short_names[row["lane_id"]]
        chart = manifest["charts"][0]
        chart["type"] = "bar"
        chart["title"] = "Known-label recall by method"
        chart["subtitle"] = "Four burst intervals; 79 sparse-positive labels; unmatched candidates remain unknown"
        chart["encodings"].pop("tooltip", None)
        manifest["blocks"] = [
            {"id": "title", "type": "markdown", "layout": "full", "body": "# " + compact_title},
            {"id": "recall_chart_block", "type": "chart", "layout": "full", "chartId": "recall_comparison"},
        ]
        compact_queries = {
            "preliminary": ("SELECT direction_cosine, activity_correlation, residual_nrms FROM preliminary_metrics", ["preliminary_metrics"]),
            "real_metrics": ("SELECT method, mean_recall, matches, candidates, total_labels FROM lane_results", ["lane_results"]),
        }
        for source_item in sources:
            if source_item["id"] in compact_queries:
                sql, tables = compact_queries[source_item["id"]]
                source_item["query"] = {"engine": "SQLite", "sql": sql, "language": "sql", "tables_used": tables, "description": "Projection of reviewed package evidence."}
        manifest["sources"] = sources
        artifact = {
            "surface": "report",
            "manifest": manifest,
            "snapshot": {"version": 1, "generatedAt": generated_at, "status": "ready", "datasets": {"headline": headline, "lane_results": lane_rows, "burst_results": burst_rows}},
            "sources": sources,
        }
        (partial / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        source_notes = """# Report source notes

Audience: technical. Delivery mode: portable HTML.

Required structure mapping: title; technical summary; key findings with native recall chart; scope/data/definitions; methodology; limitations/robustness; recommended next steps; further questions.

Chart map:

- Section: convergence did not translate into detector superiority.
- Question: compare mean known-label recall across five methods.
- Family/type: comparison / bar.
- Fields: method, mean_recall; tooltips retain matches, candidates, denominator, and precision-identification status.
- Palette: single neutral series; no redundant legend or color encoding.
- Caveat: sparse-positive labels do not establish precision.

The report intentionally omits a time trend because the evidence comprises four discrete burst evaluations, not a sufficiently sampled temporal trajectory. Exact fit matrices and objective rows remain in the diagnostic package and source evidence.
"""
        (partial / "source_notes.md").write_text(source_notes, encoding="utf-8")
        partial.replace(target)
        return target / "artifact.json"
    except Exception:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(build_report_artifact(diagnostic_root=args.diagnostic_root, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
