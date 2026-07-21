"""Latent-state interpretation reports for grid autoencoders."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.workbench.intermediates import normalize_array_frame, write_png_gray8


def build_latent_interpretation_report(
    *,
    autoencoder_run: Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    max_frame_points: int = 4000,
    nearest_neighbors: int = 3,
    title: str = "Latent State Interpretation Report",
) -> dict[str, Any]:
    """Build JSON, Markdown, HTML, and PNG summaries for saved latent codes."""
    run = _load_json(autoencoder_run) if isinstance(autoencoder_run, (str, Path)) else dict(autoencoder_run)
    latent_path = Path(str(run.get("latent_codes_path") or Path(str(run["checkpoint_path"])).with_name("latent_codes.npz")))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    latent = _load_latent(latent_path)
    codes = latent["codes"]
    video_ids = latent["video_ids"]
    labels = latent["labels"]
    centered = codes - codes.mean(axis=0, keepdims=True)
    components, explained = _principal_components(centered, n_components=4)
    frame_coords = centered @ components[:2].T if components.size else np.zeros((codes.shape[0], 2), dtype=np.float32)
    video_records = _video_records(codes, frame_coords, video_ids, labels)
    label_summary = _label_summary(video_records)
    separability = _label_separability(video_records)
    neighbors = _nearest_neighbors(video_records, k=nearest_neighbors)
    top_dims = _top_label_dimensions(codes, labels, limit=10)
    sampled_frames = _sample_frame_records(frame_coords, video_ids, labels, max_points=max_frame_points)
    embedding_png = out / "latent_video_embedding.png"
    trajectory_png = out / "latent_trajectory_preview.png"
    _write_video_embedding_png(embedding_png, video_records)
    _write_trajectory_png(trajectory_png, frame_coords, video_ids, labels, max_points=max_frame_points)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "autoencoder_run_path": str(autoencoder_run) if isinstance(autoencoder_run, (str, Path)) else None,
        "latent_codes_path": str(latent_path),
        "source_dataset": run.get("source_dataset"),
        "latent_dim": int(codes.shape[1]) if codes.ndim == 2 else 0,
        "frame_count": int(codes.shape[0]),
        "video_count": len(video_records),
        "label_counts": dict(Counter(str(label) for label in labels.tolist())),
        "pca": {
            "explained_variance_ratio": [float(v) for v in explained],
            "components_preview": components[:4, : min(8, components.shape[1])].round(6).tolist() if components.size else [],
        },
        "label_summary": label_summary,
        "label_separability": separability,
        "top_label_separating_latent_dims": top_dims,
        "nearest_neighbors": neighbors,
        "video_records": video_records,
        "sampled_frame_records": sampled_frames,
        "artifacts": {
            "embedding_png": str(embedding_png),
            "trajectory_png": str(trajectory_png),
        },
        "limitations": _limitations(),
    }
    json_path = out / "latent_interpretation_report.json"
    md_path = out / "latent_interpretation_report.md"
    html_path = out / "latent_interpretation_report.html"
    report["report_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    report["html_path"] = str(html_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_latent_interpretation_markdown(report), encoding="utf-8")
    html_path.write_text(render_latent_interpretation_html(report), encoding="utf-8")
    return report


def render_latent_interpretation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report.get('title', 'Latent State Interpretation Report')}",
        "",
        f"Generated: `{report.get('created_at')}`",
        f"Latent codes: `{report.get('latent_codes_path')}`",
        f"Frames: `{report.get('frame_count')}`",
        f"Videos: `{report.get('video_count')}`",
        f"Latent dim: `{report.get('latent_dim')}`",
        "",
        "## PCA Summary",
        "",
        f"Explained variance ratio: `{', '.join(_fmt(v) for v in report.get('pca', {}).get('explained_variance_ratio', [])[:4])}`",
        "",
        "## Label Separability",
        "",
    ]
    sep = report.get("label_separability", {})
    lines.extend(
        [
            f"- Nearest-centroid leave-one-video-out accuracy: `{_fmt(sep.get('nearest_centroid_leave_one_video_accuracy'))}`.",
            f"- Between/within centroid distance ratio: `{_fmt(sep.get('between_within_distance_ratio'))}`.",
            f"- Mean within-label distance: `{_fmt(sep.get('mean_within_label_distance'))}`.",
            f"- Mean between-label distance: `{_fmt(sep.get('mean_between_label_distance'))}`.",
        ]
    )
    lines.extend(["", "## Label Summary", "", "| Label | Videos | Frames | Mean speed | P95 speed |", "|---|---:|---:|---:|---:|"])
    for row in report.get("label_summary", []):
        lines.append(f"| {row.get('label')} | {row.get('video_count')} | {row.get('frame_count')} | {_fmt(row.get('mean_latent_velocity'))} | {_fmt(row.get('p95_latent_velocity'))} |")
    lines.extend(["", "## Top Label-Separating Latent Dimensions", "", "| Dim | Eta squared | Label means |", "|---:|---:|---|"])
    for row in report.get("top_label_separating_latent_dims", [])[:10]:
        lines.append(f"| {row.get('dimension')} | {_fmt(row.get('eta_squared'))} | `{row.get('label_means')}` |")
    lines.extend(["", "## Nearest Video Neighbors", "", "| Video | Label | Neighbors |", "|---|---|---|"])
    for item in report.get("nearest_neighbors", [])[:20]:
        neigh = ", ".join(f"{n.get('video_id')} ({n.get('label')}, d={_fmt(n.get('distance'))})" for n in item.get("neighbors", []))
        lines.append(f"| `{item.get('video_id')}` | {item.get('label')} | {neigh} |")
    lines.extend(["", "## Artifacts", ""])
    for key, path in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- {key}: `{path}`")
    lines.extend(["", "## Limitations", ""])
    for item in report.get("limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def render_latent_interpretation_html(report: Mapping[str, Any]) -> str:
    title = str(report.get("title") or "Latent State Interpretation Report")
    sep = report.get("label_separability", {}) if isinstance(report.get("label_separability"), Mapping) else {}
    artifacts = report.get("artifacts", {}) if isinstance(report.get("artifacts"), Mapping) else {}
    label_rows = "".join(
        f"<tr><td>{_e(r.get('label'))}</td><td>{_e(r.get('video_count'))}</td><td>{_e(r.get('frame_count'))}</td><td>{_fmt(r.get('mean_latent_velocity'))}</td><td>{_fmt(r.get('p95_latent_velocity'))}</td></tr>"
        for r in report.get("label_summary", [])
    )
    dim_rows = "".join(
        f"<tr><td>{_e(r.get('dimension'))}</td><td>{_fmt(r.get('eta_squared'))}</td><td><code>{_e(r.get('label_means'))}</code></td></tr>"
        for r in report.get("top_label_separating_latent_dims", [])[:10]
    )
    neighbor_rows = "".join(
        f"<tr><td><code>{_e(item.get('video_id'))}</code></td><td>{_e(item.get('label'))}</td><td>{_e(', '.join(str(n.get('video_id')) + ' (' + str(n.get('label')) + ', d=' + _fmt(n.get('distance')) + ')' for n in item.get('neighbors', [])))}</td></tr>"
        for item in report.get("nearest_neighbors", [])[:20]
    )
    limitations = "".join(f"<li>{_e(item)}</li>" for item in report.get("limitations", []))
    emb = Path(str(artifacts.get("embedding_png", ""))).name if artifacts.get("embedding_png") else ""
    traj = Path(str(artifacts.get("trajectory_png", ""))).name if artifacts.get("trajectory_png") else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_e(title)}</title>
<style>
body {{ margin: 0; background: #f7f8fb; color: #1f2933; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
header {{ background: #111827; color: white; padding: 28px 32px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px 32px 44px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.metric, section {{ background: white; border: 1px solid #d8dee6; border-radius: 8px; }}
.metric {{ padding: 12px 14px; }}
.metric div:first-child {{ color: #5b6675; font-size: 12px; text-transform: uppercase; }}
.metric div:last-child {{ font-size: 20px; margin-top: 4px; }}
section {{ margin-top: 16px; padding: 16px; }}
h2 {{ font-size: 17px; margin: 0 0 12px; }}
.figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
img {{ max-width: 100%; image-rendering: pixelated; border: 1px solid #e5e7eb; background: #111; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ color: #4b5563; }} code {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<header>
<h1>{_e(title)}</h1>
<div>Latent codes: {_e(report.get('latent_codes_path'))}</div>
</header>
<main>
<div class="metrics">
  <div class="metric"><div>Frames</div><div>{_e(report.get('frame_count'))}</div></div>
  <div class="metric"><div>Videos</div><div>{_e(report.get('video_count'))}</div></div>
  <div class="metric"><div>Latent dim</div><div>{_e(report.get('latent_dim'))}</div></div>
  <div class="metric"><div>Centroid accuracy</div><div>{_fmt(sep.get('nearest_centroid_leave_one_video_accuracy'))}</div></div>
</div>
<section>
<h2>Embedding Previews</h2>
<div class="figure-grid">
  <figure><img src="{_e(emb)}" alt="Video-level latent embedding"><figcaption>Video-level mean-latent PCA embedding.</figcaption></figure>
  <figure><img src="{_e(traj)}" alt="Frame-level latent trajectory preview"><figcaption>Sampled frame-level trajectory PCA preview.</figcaption></figure>
</div>
</section>
<section><h2>Label Summary</h2><table><thead><tr><th>Label</th><th>Videos</th><th>Frames</th><th>Mean speed</th><th>P95 speed</th></tr></thead><tbody>{label_rows}</tbody></table></section>
<section><h2>Top Label-Separating Latent Dimensions</h2><table><thead><tr><th>Dim</th><th>Eta squared</th><th>Label means</th></tr></thead><tbody>{dim_rows}</tbody></table></section>
<section><h2>Nearest Video Neighbors</h2><table><thead><tr><th>Video</th><th>Label</th><th>Neighbors</th></tr></thead><tbody>{neighbor_rows}</tbody></table></section>
<section><h2>Limitations</h2><ul>{limitations}</ul></section>
</main>
</body>
</html>
"""


def build_latent_objective_plan(
    *,
    interpretation_report: Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    title: str = "Latent Objective Follow-Up Plan",
) -> dict[str, Any]:
    """Plan supervised/contrastive latent follow-ups from an interpretation report."""
    report_path = Path(interpretation_report) if isinstance(interpretation_report, (str, Path)) else None
    report = _load_json(report_path) if report_path is not None else dict(interpretation_report)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    separability = report.get("label_separability") if isinstance(report.get("label_separability"), Mapping) else {}
    label_count = len(separability.get("labels") or report.get("label_counts") or {})
    chance = (1.0 / label_count) if label_count else None
    accuracy = _num(separability.get("nearest_centroid_leave_one_video_accuracy"))
    ratio = _num(separability.get("between_within_distance_ratio"))
    top_dims = [dict(row) for row in report.get("top_label_separating_latent_dims", [])[:8] if isinstance(row, Mapping)]
    diagnosis = _latent_objective_diagnosis(accuracy=accuracy, chance=chance, ratio=ratio)
    evidence = {
        "interpretation_report_path": str(report_path) if report_path is not None else report.get("report_path"),
        "frame_count": report.get("frame_count"),
        "video_count": report.get("video_count"),
        "latent_dim": report.get("latent_dim"),
        "label_counts": report.get("label_counts"),
        "nearest_centroid_leave_one_video_accuracy": accuracy,
        "chance_accuracy": chance,
        "between_within_distance_ratio": ratio,
        "mean_within_label_distance": separability.get("mean_within_label_distance"),
        "mean_between_label_distance": separability.get("mean_between_label_distance"),
        "top_label_separating_latent_dims": top_dims,
    }
    objectives = _latent_objective_candidates(diagnosis=diagnosis, top_dims=top_dims)
    plan = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "evidence": evidence,
        "diagnosis": diagnosis,
        "recommended_objectives": objectives,
        "acceptance_gates": _latent_objective_acceptance_gates(chance=chance),
        "non_goals": [
            "Do not claim left/right/neutral state separation from the current unsupervised autoencoder latent space alone.",
            "Do not select a latent objective using frame-level random splits; all claims should use held-out videos.",
            "Do not trade away reconstruction or forecasting baselines without reporting the regression explicitly.",
        ],
        "recommended_next_step": "Prototype a held-out-video supervised latent-head smoke test before spending GPU time on a full auxiliary-objective sweep.",
    }
    json_path = out / "latent_objective_plan.json"
    markdown_path = out / "latent_objective_plan.md"
    plan["plan_path"] = str(json_path)
    plan["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_latent_objective_plan_markdown(plan), encoding="utf-8")
    return plan


def render_latent_objective_plan_markdown(plan: Mapping[str, Any]) -> str:
    evidence = plan.get("evidence") if isinstance(plan.get("evidence"), Mapping) else {}
    diagnosis = plan.get("diagnosis") if isinstance(plan.get("diagnosis"), Mapping) else {}
    lines = [
        f"# {plan.get('title', 'Latent Objective Follow-Up Plan')}",
        "",
        f"Generated: `{plan.get('created_at')}`",
        f"Source interpretation report: `{evidence.get('interpretation_report_path')}`",
        "",
        "## Evidence",
        "",
        f"- Frames: `{evidence.get('frame_count')}`; videos: `{evidence.get('video_count')}`; latent dim: `{evidence.get('latent_dim')}`.",
        f"- Label counts: `{evidence.get('label_counts')}`.",
        f"- Leave-one-video nearest-centroid accuracy: `{_fmt(evidence.get('nearest_centroid_leave_one_video_accuracy'))}`; chance baseline: `{_fmt(evidence.get('chance_accuracy'))}`.",
        f"- Between/within centroid distance ratio: `{_fmt(evidence.get('between_within_distance_ratio'))}`.",
        f"- Diagnosis: `{diagnosis.get('status')}` - {diagnosis.get('summary')}",
        "",
        "## Objective Candidates",
        "",
        "| Priority | Objective | Why | First check |",
        "|---:|---|---|---|",
    ]
    for idx, row in enumerate(plan.get("recommended_objectives", []), start=1):
        lines.append(f"| {idx} | {row.get('name')} | {row.get('rationale')} | {row.get('first_check')} |")
    lines.extend(["", "## Acceptance Gates", ""])
    for item in plan.get("acceptance_gates", []):
        lines.append(f"- {item}")
    dims = evidence.get("top_label_separating_latent_dims") or []
    if dims:
        lines.extend(["", "## Candidate Latent Dimensions To Inspect", "", "| Dim | Eta squared | Label means |", "|---:|---:|---|"])
        for row in dims[:8]:
            lines.append(f"| {row.get('dimension')} | {_fmt(row.get('eta_squared'))} | `{row.get('label_means')}` |")
    lines.extend(["", "## Non-Goals", ""])
    for item in plan.get("non_goals", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Next Step", "", str(plan.get("recommended_next_step"))])
    return "\n".join(lines).rstrip() + "\n"


def _latent_objective_diagnosis(*, accuracy: float | None, chance: float | None, ratio: float | None) -> dict[str, Any]:
    if accuracy is None and ratio is None:
        return {"status": "insufficient_evidence", "summary": "The interpretation report did not contain separability metrics."}
    weak_accuracy = accuracy is not None and chance is not None and accuracy <= chance
    weak_ratio = ratio is not None and ratio < 1.0
    if weak_accuracy and weak_ratio:
        return {
            "status": "weak_label_separability",
            "summary": "Held-out video labels are not cleanly separated by the current latent space; between-label centroids are closer than within-label variation.",
        }
    if weak_accuracy:
        return {
            "status": "weak_centroid_prediction",
            "summary": "Held-out video centroid classification is at or below the label-count chance baseline.",
        }
    if weak_ratio:
        return {
            "status": "overlapping_label_geometry",
            "summary": "Between-label centroids are closer than within-label variation, even if centroid classification is not below chance.",
        }
    return {
        "status": "separability_present_but_unvalidated",
        "summary": "Latent labels show some separability; follow-up objectives should still verify held-out-video generalization and forecasting impact.",
    }


def _latent_objective_candidates(*, diagnosis: Mapping[str, Any], top_dims: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    top_dim_text = ", ".join(str(row.get("dimension")) for row in top_dims[:5]) or "none"
    return [
        {
            "name": "Held-out-video supervised latent head",
            "rationale": "The current unsupervised latent space has weak label separability, so a small auxiliary label head is the lowest-cost test of whether behavior labels are recoverable from the encoder.",
            "first_check": "Train only a linear/logistic head on frozen latent summaries with leave-one-video validation; compare against chance before changing the autoencoder.",
        },
        {
            "name": "Supervised contrastive latent regularizer",
            "rationale": "Nearest-video neighbors are often cross-label, so same-label positives and different-label negatives may reduce label overlap without requiring dense frame labels.",
            "first_check": "Use video-level labels with held-out-video batches; verify centroid accuracy and between/within ratio improve without collapsing reconstruction variance.",
        },
        {
            "name": "Multi-task reconstruction plus h2/h5 prediction",
            "rationale": "Forecasting is the core task, and shared-horizon models already expose per-horizon failures; coupling reconstruction with short-horizon prediction may shape latents toward dynamics-relevant state.",
            "first_check": "Run a small frozen/partially-frozen latent dynamics probe and require positive h2/h5 test improvement over persistence plus no active-cell regression relative to current controls.",
        },
        {
            "name": "Targeted latent-dimension audit",
            "rationale": f"A few dimensions carry the strongest label association ({top_dim_text}), but they are hypotheses rather than claims.",
            "first_check": "Plot these dimensions by video and time, then test whether masking or emphasizing them changes held-out label accuracy or forecast error.",
        },
    ]


def _latent_objective_acceptance_gates(*, chance: float | None) -> list[str]:
    chance_text = _fmt(chance)
    return [
        f"Held-out videos: label accuracy should beat the label-count chance baseline (`{chance_text}`) by a clear margin before claiming label-state encoding.",
        "Between/within centroid distance ratio should move above `1.0` or the report should state that labels remain geometrically overlapped.",
        "Any auxiliary objective must report reconstruction MSE and h2/h5 forecasting metrics against the current autoencoder plus persistence controls.",
        "Active-cell, top-activity, and high-change test metrics must be reported separately; a global-MSE gain alone is not enough.",
    ]


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _load_latent(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        codes = data["latent_codes"].astype(np.float32)
        video_ids = data["frame_video_ids"].astype(str)
        labels = data["frame_labels"].astype(str)
    if codes.ndim != 2:
        raise ValueError("latent_codes must be a 2-D array.")
    if video_ids.shape[0] != codes.shape[0] or labels.shape[0] != codes.shape[0]:
        raise ValueError("latent code, video ID, and label arrays must have matching frame counts.")
    return {"codes": codes, "video_ids": video_ids, "labels": labels}


def _principal_components(centered: np.ndarray, *, n_components: int) -> tuple[np.ndarray, list[float]]:
    if centered.size == 0:
        return np.zeros((0, 0), dtype=np.float32), []
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(singular * singular))
    explained = ((singular * singular) / total).astype(float).tolist() if total > 0 else [0.0 for _ in singular]
    return vt[:n_components].astype(np.float32), explained[:n_components]


def _video_records(codes: np.ndarray, frame_coords: np.ndarray, video_ids: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    records = []
    for video_id in _ordered_unique(video_ids):
        mask = video_ids == video_id
        z = codes[mask]
        coords = frame_coords[mask]
        velocity = np.linalg.norm(np.diff(z, axis=0), axis=1) if z.shape[0] > 1 else np.zeros(0, dtype=np.float32)
        label = str(labels[mask][0]) if mask.any() else "unknown"
        records.append(
            {
                "video_id": str(video_id),
                "label": label,
                "frame_count": int(mask.sum()),
                "latent_mean": z.mean(axis=0).round(6).tolist(),
                "latent_std": z.std(axis=0).round(6).tolist(),
                "pca_mean": coords.mean(axis=0).round(6).tolist() if coords.size else [0.0, 0.0],
                "pca_start": coords[0].round(6).tolist() if coords.size else [0.0, 0.0],
                "pca_end": coords[-1].round(6).tolist() if coords.size else [0.0, 0.0],
                "mean_latent_velocity": float(np.mean(velocity)) if velocity.size else 0.0,
                "p95_latent_velocity": float(np.percentile(velocity, 95)) if velocity.size else 0.0,
            }
        )
    return records


def _label_summary(video_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in video_records:
        by_label[str(record.get("label", "unknown"))].append(record)
    rows = []
    for label, records in sorted(by_label.items()):
        rows.append(
            {
                "label": label,
                "video_count": len(records),
                "frame_count": int(sum(int(r.get("frame_count", 0)) for r in records)),
                "mean_latent_velocity": float(np.mean([float(r.get("mean_latent_velocity", 0.0)) for r in records])) if records else 0.0,
                "p95_latent_velocity": float(np.mean([float(r.get("p95_latent_velocity", 0.0)) for r in records])) if records else 0.0,
                "mean_pca": np.asarray([r.get("pca_mean", [0.0, 0.0]) for r in records], dtype=np.float32).mean(axis=0).round(6).tolist() if records else [0.0, 0.0],
            }
        )
    return rows


def _label_separability(video_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not video_records:
        return {}
    means = np.asarray([record.get("latent_mean", []) for record in video_records], dtype=np.float32)
    labels = np.asarray([str(record.get("label", "unknown")) for record in video_records])
    label_values = sorted(set(labels.tolist()))
    centroids = {label: means[labels == label].mean(axis=0) for label in label_values}
    within = []
    for label in label_values:
        rows = means[labels == label]
        if rows.size:
            within.extend(np.linalg.norm(rows - centroids[label], axis=1).astype(float).tolist())
    between = []
    for i, a in enumerate(label_values):
        for b in label_values[i + 1 :]:
            between.append(float(np.linalg.norm(centroids[a] - centroids[b])))
    preds = []
    for index, row in enumerate(means):
        train_labels = [label for j, label in enumerate(labels) if j != index]
        train_rows = np.asarray([other for j, other in enumerate(means) if j != index], dtype=np.float32)
        pred = _nearest_centroid_predict(row, train_rows, np.asarray(train_labels, dtype=str))
        preds.append(pred)
    acc = float(np.mean(np.asarray(preds, dtype=str) == labels)) if len(preds) else None
    mean_within = float(np.mean(within)) if within else None
    mean_between = float(np.mean(between)) if between else None
    return {
        "labels": label_values,
        "nearest_centroid_leave_one_video_accuracy": acc,
        "mean_within_label_distance": mean_within,
        "mean_between_label_distance": mean_between,
        "between_within_distance_ratio": (mean_between / mean_within) if mean_between is not None and mean_within not in (None, 0.0) else None,
        "centroids": {label: centroids[label].round(6).tolist() for label in label_values},
    }


def _nearest_centroid_predict(row: np.ndarray, train_rows: np.ndarray, train_labels: np.ndarray) -> str:
    if train_rows.size == 0:
        return "unknown"
    best_label = None
    best_dist = float("inf")
    for label in sorted(set(train_labels.tolist())):
        centroid = train_rows[train_labels == label].mean(axis=0)
        dist = float(np.linalg.norm(row - centroid))
        if dist < best_dist:
            best_label = label
            best_dist = dist
    return str(best_label)


def _nearest_neighbors(video_records: Sequence[Mapping[str, Any]], *, k: int) -> list[dict[str, Any]]:
    if not video_records:
        return []
    means = np.asarray([record.get("latent_mean", []) for record in video_records], dtype=np.float32)
    out = []
    for i, record in enumerate(video_records):
        distances = np.linalg.norm(means - means[i], axis=1)
        order = [idx for idx in np.argsort(distances).tolist() if idx != i][: max(int(k), 0)]
        out.append(
            {
                "video_id": record.get("video_id"),
                "label": record.get("label"),
                "neighbors": [
                    {
                        "video_id": video_records[j].get("video_id"),
                        "label": video_records[j].get("label"),
                        "distance": float(distances[j]),
                    }
                    for j in order
                ],
            }
        )
    return out


def _top_label_dimensions(codes: np.ndarray, labels: np.ndarray, *, limit: int) -> list[dict[str, Any]]:
    label_values = sorted(set(labels.astype(str).tolist()))
    if len(label_values) < 2:
        return []
    global_mean = codes.mean(axis=0)
    total_ss = np.sum((codes - global_mean) ** 2, axis=0)
    between_ss = np.zeros(codes.shape[1], dtype=np.float64)
    label_means: dict[str, np.ndarray] = {}
    for label in label_values:
        rows = codes[labels == label]
        if rows.size == 0:
            continue
        mean = rows.mean(axis=0)
        label_means[label] = mean
        between_ss += rows.shape[0] * ((mean - global_mean) ** 2)
    eta = np.divide(between_ss, total_ss, out=np.zeros_like(between_ss), where=total_ss > 0)
    order = np.argsort(-eta)[: max(int(limit), 0)]
    rows = []
    for dim in order:
        rows.append(
            {
                "dimension": int(dim),
                "eta_squared": float(eta[dim]),
                "label_means": {label: float(label_means[label][dim]) for label in label_values if label in label_means},
            }
        )
    return rows


def _sample_frame_records(frame_coords: np.ndarray, video_ids: np.ndarray, labels: np.ndarray, *, max_points: int) -> list[dict[str, Any]]:
    n = frame_coords.shape[0]
    if n == 0 or max_points <= 0:
        return []
    if n <= max_points:
        indices = np.arange(n)
    else:
        indices = np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))
    return [
        {
            "frame_index": int(i),
            "video_id": str(video_ids[i]),
            "label": str(labels[i]),
            "pc1": float(frame_coords[i, 0]),
            "pc2": float(frame_coords[i, 1]),
        }
        for i in indices.tolist()
    ]


def _write_video_embedding_png(path: Path, video_records: Sequence[Mapping[str, Any]]) -> None:
    coords = np.asarray([record.get("pca_mean", [0.0, 0.0]) for record in video_records], dtype=np.float32)
    labels = [str(record.get("label", "unknown")) for record in video_records]
    _write_scatter_png(path, coords, labels, size=192)


def _write_trajectory_png(path: Path, frame_coords: np.ndarray, video_ids: np.ndarray, labels: np.ndarray, *, max_points: int) -> None:
    n = frame_coords.shape[0]
    if n == 0:
        _write_scatter_png(path, np.zeros((0, 2), dtype=np.float32), [], size=192)
        return
    if n > max_points and max_points > 0:
        idx = np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))
        coords = frame_coords[idx]
        labs = labels[idx].astype(str).tolist()
    else:
        coords = frame_coords
        labs = labels.astype(str).tolist()
    _write_scatter_png(path, coords, labs, size=192)


def _write_scatter_png(path: Path, coords: np.ndarray, labels: Sequence[str], *, size: int) -> None:
    img = np.zeros((size, size), dtype=np.float32)
    if coords.size:
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        pix = np.round((coords - mins) / span * (size - 8) + 4).astype(int)
        label_values = {label: idx for idx, label in enumerate(sorted(set(str(v) for v in labels)))}
        denom = max(len(label_values) - 1, 1)
        for (px, py), label in zip(pix, labels):
            y = int(np.clip(py, 0, size - 1))
            x = int(np.clip(px, 0, size - 1))
            value = 0.35 + 0.6 * (label_values.get(str(label), 0) / denom)
            img[y, x] = max(img[y, x], float(value))
            if y + 1 < size:
                img[y + 1, x] = max(img[y + 1, x], float(value) * 0.7)
            if x + 1 < size:
                img[y, x + 1] = max(img[y, x + 1], float(value) * 0.7)
    write_png_gray8(path, size, size, normalize_array_frame(img))


def _ordered_unique(values: np.ndarray) -> list[str]:
    seen = set()
    out = []
    for value in values.astype(str).tolist():
        if value not in seen:
            out.append(str(value))
            seen.add(str(value))
    return out


def _limitations() -> list[str]:
    return [
        "PCA coordinates are descriptive and should not be treated as a biological axis without follow-up validation.",
        "Nearest-neighbor and centroid analyses use video-level mean latent codes, so they summarize videos rather than individual events.",
        "Latent dimensions are encoder features; dimension-label associations are hypotheses for review, not causal claims.",
    ]


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "n/a"


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
