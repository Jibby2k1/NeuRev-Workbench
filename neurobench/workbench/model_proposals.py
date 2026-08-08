"""Build label-free model-proposal review packages from frozen Workbench candidates."""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from neurobench.annotations import default_annotations_v3
from neurobench.workbench.annotation_revisions import initialize_revision_root
from neurobench.workbench.builder import build_workbench


DETAIL_COLUMNS = [
    "proposal_id",
    "occurrence_id",
    "burst_id",
    "start_frame_ui",
    "end_frame_ui",
    "rank",
    "score",
    "x_px",
    "y_px",
    "recurrence_count",
    "event_source",
    "model_state",
    "review_state",
    "corrected_x_px",
    "corrected_y_px",
    "reviewer_notes",
]
IDENTITY_COLUMNS = [
    "proposal_id",
    "median_x_px",
    "median_y_px",
    "occurrence_count",
    "first_frame_ui",
    "last_frame_ui",
    "maximum_score",
    "mean_score",
    "event_source",
    "model_state",
    "review_state",
    "reviewer_notes",
]
_EXPERT_DERIVED_FIELDS = {
    "expert_supported",
    "linked_expert_id",
    "matched_expert_roi",
    "match_distance_px",
    "match_status",
}


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(f".{path.name}.partial")
    target.write_bytes(_json_bytes(payload))
    os.replace(target, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value: str, fallback: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_.-")
    return result or fallback


def _copy_or_link(source: str, target: str) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _strip_expert_fields(model_roi: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(model_roi))
    for field in _EXPERT_DERIVED_FIELDS:
        result.pop(field, None)
    result["status"] = "unknown"
    result["linked_expert_id"] = ""
    members = []
    for member in result.get("members") or []:
        cleaned = {key: deepcopy(value) for key, value in dict(member).items() if key not in _EXPERT_DERIVED_FIELDS}
        members.append(cleaned)
    result["members"] = members
    return result


def _occurrence_rows(models: Sequence[Mapping[str, Any]], *, event_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        proposal_id = str(model.get("id") or "")
        members = list(model.get("members") or [])
        if not members:
            intervals = model.get("event_intervals") or model.get("eventIntervals") or []
            members = [
                {
                    "burst": index,
                    "x": (model.get("source_xy") or [0, 0])[0],
                    "y": (model.get("source_xy") or [0, 0])[1],
                    "start_ui": interval[0],
                    "end_ui": interval[1],
                    "rank": index,
                    "score": model.get("score", ""),
                }
                for index, interval in enumerate(intervals, 1)
            ]
        recurrence = len(members)
        for index, member in enumerate(members, 1):
            burst = int(member.get("burst") or member.get("burst_id") or index)
            start = int(member.get("start_ui") or member.get("start_frame_ui") or model.get("ui_frame") or 1)
            end = int(member.get("end_ui") or member.get("end_frame_ui") or start)
            rows.append(
                {
                    "proposal_id": proposal_id,
                    "occurrence_id": f"{proposal_id}__burst_{burst:03d}__occurrence_{index:03d}",
                    "burst_id": burst,
                    "start_frame_ui": start,
                    "end_frame_ui": end,
                    "rank": int(member.get("rank") or index),
                    "score": member.get("score", model.get("score", "")),
                    "x_px": float(member.get("x", (model.get("source_xy") or [0, 0])[0])),
                    "y_px": float(member.get("y", (model.get("source_xy") or [0, 0])[1])),
                    "recurrence_count": recurrence,
                    "event_source": event_source,
                    "model_state": "unknown",
                    "review_state": "",
                    "corrected_x_px": "",
                    "corrected_y_px": "",
                    "reviewer_notes": "",
                }
            )
    rows.sort(key=lambda row: (int(row["burst_id"]), int(row["rank"]), -_number(row["score"]), str(row["proposal_id"])))
    return rows


def _identity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["proposal_id"]), []).append(row)
    result = []
    for proposal_id, members in sorted(grouped.items()):
        xs = sorted(float(item["x_px"]) for item in members)
        ys = sorted(float(item["y_px"]) for item in members)
        scores = [_number(item["score"]) for item in members if str(item.get("score", "")) != ""]
        result.append(
            {
                "proposal_id": proposal_id,
                "median_x_px": _median(xs),
                "median_y_px": _median(ys),
                "occurrence_count": len(members),
                "first_frame_ui": min(int(item["start_frame_ui"]) for item in members),
                "last_frame_ui": max(int(item["end_frame_ui"]) for item in members),
                "maximum_score": max(scores) if scores else "",
                "mean_score": sum(scores) / len(scores) if scores else "",
                "event_source": members[0]["event_source"],
                "model_state": "unknown",
                "review_state": "",
                "reviewer_notes": "",
            }
        )
    return result


def _median(values: Sequence[float]) -> float:
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _write_tsv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _column_name(index: int) -> str:
    result = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _cell_xml(row: int, column: int, value: Any, style: int = 0) -> str:
    reference = f"{_column_name(column)}{row}"
    style_attribute = f' s="{style}"' if style else ""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style_attribute}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f'<c r="{reference}"{style_attribute}><v>{value}</v></c>'
    escaped = html.escape(str(value), quote=False)
    return f'<c r="{reference}" t="inlineStr"{style_attribute}><is><t xml:space="preserve">{escaped}</t></is></c>'


def _sheet_xml(rows: Sequence[Sequence[Any]], *, widths: Sequence[float] | None = None, freeze_row: int | None = None) -> str:
    maximum_columns = max((len(row) for row in rows), default=1)
    dimension = f"A1:{_column_name(maximum_columns)}{max(1, len(rows))}"
    cols = ""
    if widths:
        cols = "<cols>" + "".join(
            f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
            for index, width in enumerate(widths, 1)
        ) + "</cols>"
    pane = ""
    if freeze_row:
        pane = (
            '<sheetViews><sheetView workbookViewId="0">'
            f'<pane ySplit="{freeze_row}" topLeftCell="A{freeze_row + 1}" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>'
        )
    body = []
    for row_index, values in enumerate(rows, 1):
        cells = []
        for column_index, value in enumerate(values, 1):
            style = 1 if row_index == 1 or (row_index == 5 and "proposal/rank" in {str(item) for item in values}) else 0
            cell = _cell_xml(row_index, column_index, value, style)
            if cell:
                cells.append(cell)
        body.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>{pane}{cols}<sheetData>{"".join(body)}</sheetData>'
        '<pageMargins left="0.4" right="0.4" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>'
        '</worksheet>'
    )


def _workbook_rows(
    occurrences: Sequence[Mapping[str, Any]],
    *,
    event_source: str,
    include_candidates: bool,
) -> list[list[Any]]:
    groups: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {}
    for row in occurrences:
        key = (int(row["burst_id"]), int(row["start_frame_ui"]), int(row["end_frame_ui"]))
        groups.setdefault(key, []).append(row)
    ordered = sorted(groups)
    columns = max(1, len(ordered) * 4 - 1)
    rows: list[list[Any]] = [[""] * columns for _ in range(5)]
    rows[0][0] = "MODEL PROPOSALS FOR REVIEW — UNREVIEWED, NOT EXPERT TRUTH" if include_candidates else "BLINDED EXPERT ANNOTATION TEMPLATE"
    rows[1][0] = "Event windows (UI frames, one-based inclusive):"
    rows[3][0] = f"Event source: {event_source}"
    maximum = 0
    for block, key in enumerate(ordered):
        burst, start, end = key
        offset = block * 4
        rows[1][offset] = f"{start}-{end}"
        rows[2][offset] = f"Burst {burst}"
        rows[4][offset:offset + 3] = ["proposal/rank", "x", "y"]
        maximum = max(maximum, len(groups[key]))
    if include_candidates:
        for index in range(maximum):
            output = [""] * columns
            for block, key in enumerate(ordered):
                members = sorted(groups[key], key=lambda item: (int(item["rank"]), str(item["proposal_id"])))
                if index >= len(members):
                    continue
                member = members[index]
                offset = block * 4
                output[offset:offset + 3] = [member["rank"], member["x_px"], member["y_px"]]
            rows.append(output)
    return rows


def _write_xlsx(path: Path, sheets: Sequence[tuple[str, Sequence[Sequence[Any]], Sequence[float] | None, int | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides) + '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
        + "".join(
            f'<sheet name="{html.escape(name, quote=True)}" sheetId="{index}" r:id="rId{index}"/>'
            for index, (name, _, _, _) in enumerate(sheets, 1)
        ) + '</sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(sheets) + 1)
        )
        + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        + '</Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF3A5F52"/><bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1"/></cellXfs>'
        '</styleSheet>'
    )
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        for index, (_, rows, widths, freeze_row) in enumerate(sheets, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows, widths=widths, freeze_row=freeze_row))
    os.replace(temporary, path)


def _table_rows(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [list(columns)] + [[row.get(column, "") for column in columns] for row in rows]


def _event_windows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    unique = {
        (int(row["burst_id"]), int(row["start_frame_ui"]), int(row["end_frame_ui"]))
        for row in rows
    }
    return [
        {"burst_id": burst, "start_frame_ui": start, "end_frame_ui": end}
        for burst, start, end in sorted(unique)
    ]


def build_model_proposal_package(
    *,
    source_app_dir: str | Path,
    output_root: str | Path,
    reviewer_id: str = "reviewer_local_1",
    event_source: str = "supplied",
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Create a model-only dashboard and proposal workbook without changing the source app."""
    if event_source not in {"supplied", "model_proposed"}:
        raise ValueError("event_source must be 'supplied' or 'model_proposed'")
    source_app = Path(source_app_dir).expanduser().resolve()
    source_review = source_app / "review_data.json"
    if not source_review.is_file():
        raise FileNotFoundError(source_review)
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing model-proposal package collision: {output}")
    review_data = json.loads(source_review.read_text(encoding="utf-8"))
    correction = dict(review_data.get("annotationCorrection") or review_data.get("annotation_correction") or {})
    models = [_strip_expert_fields(item) for item in correction.get("model_rois") or correction.get("modelRois") or []]
    if not models:
        raise ValueError("source review_data has no annotationCorrection model_rois")
    contracts = list(correction.get("view_contracts") or correction.get("viewContracts") or [])
    if not contracts:
        raise ValueError("source review_data has no annotationCorrection view_contracts")
    for contract in contracts:
        pattern = str(contract.get("frame_pattern") or contract.get("framePattern") or "")
        if not pattern:
            continue
        bounds = contract.get("frame_mapping") or contract.get("frameMapping") or {}
        shape = contract.get("shape_tyx") or contract.get("shapeTyx") or [1]
        first = int(bounds.get("offset") or 0) + 1
        last = first + int(shape[0]) - 1
        for frame in {first, last}:
            relative = pattern.replace("%04d", f"{frame:04d}").replace("%03d", f"{frame:03d}")
            if not (source_app / relative).is_file():
                raise FileNotFoundError(source_app / relative)
    source_dataset = dict(review_data.get("dataset") or {})
    resolved_dataset_id = str(dataset_id or source_dataset.get("dataset_id") or correction.get("source_video_id") or "model_proposal_dataset")
    frozen_run = str((correction.get("revision") or {}).get("frozenRunId") or (review_data.get("parameters") or {}).get("frozen_lane") or "frozen_model_run")
    revision_id = _safe_id(f"{resolved_dataset_id}_model_proposal_draft_v1", "model_proposal_draft_v1")
    now = datetime.now(timezone.utc).isoformat()
    empty_annotations = default_annotations_v3()
    empty_source_sha = hashlib.sha256(b"NO_EXPERT_ANNOTATIONS\n").hexdigest()
    revision = {
        "schema_version": 1,
        "revisionId": revision_id,
        "parentRevisionId": "unlabeled_seed",
        "state": "draft",
        "reviewerId": str(reviewer_id),
        "frozenRunId": frozen_run,
        "sourceAnnotationsSha256": empty_source_sha,
        "createdAt": now,
        "updatedAt": now,
        "revisionToken": 0,
        "operationCount": 0,
    }
    occurrence_rows = _occurrence_rows(models, event_source=event_source)
    identity_rows = _identity_rows(occurrence_rows)
    sanitized = deepcopy(review_data)
    sanitized["dataset"] = {
        **source_dataset,
        "dataset_id": resolved_dataset_id,
        "name": str(source_dataset.get("name") or resolved_dataset_id) + " · model proposal review",
    }
    sanitized["rois"] = []
    sanitized["parameters"] = {
        **dict(review_data.get("parameters") or {}),
        "purpose": "label-free model proposal review",
        "expert_annotations": "not_applicable_pending_labels",
        "event_source": event_source,
    }
    sanitized["annotationCorrection"] = {
        **correction,
        "schema_version": 1,
        "mode": "model_only",
        "read_only": False,
        "revision": revision,
        "expert_annotation_state": "not_applicable_pending_labels",
        "comparison_state": "not_applicable_pending_labels",
        "event_source": event_source,
        "expert_rois": [],
        "model_rois": models,
        "matches": [],
    }
    sanitized.pop("annotation_correction", None)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=parent))
    try:
        app = staging / "app"
        app.mkdir()
        frames_source = source_app / "frames"
        if frames_source.is_dir():
            shutil.copytree(frames_source, app / "frames", copy_function=_copy_or_link)
        _write_json(app / "review_data.json", sanitized)
        architecture_source = source_app / "architecture_runs.json"
        if architecture_source.is_file():
            shutil.copy2(architecture_source, app / "architecture_runs.json")
        build_workbench(app_dir=app, review_data_path=app / "review_data.json", dataset_id=resolved_dataset_id)
        revision_root = initialize_revision_root(
            app / "annotation_revisions",
            revision=revision,
            annotations=empty_annotations,
        )
        shutil.copy2(revision_root / "annotations.json", app / "annotations.json")

        exports = staging / "proposal_exports"
        _write_tsv(exports / "model_proposals_long.tsv", DETAIL_COLUMNS, occurrence_rows)
        _write_tsv(exports / "model_proposal_identities.tsv", IDENTITY_COLUMNS, identity_rows)
        provenance_rows = [
            ["field", "value"],
            ["dataset_id", resolved_dataset_id],
            ["frozen_run_id", frozen_run],
            ["source_app", str(source_app)],
            ["source_review_data_sha256", _sha256(source_review)],
            ["expert_annotations", "not_applicable_pending_labels"],
            ["event_source", event_source],
            ["coordinate_convention", "x=column, y=row"],
            ["frame_convention", "UI frames are one-based and inclusive"],
            ["candidate_interpretation", "unreviewed model proposal; unknown, not false positive"],
            ["generated_at", now],
        ]
        proposal_workbook = exports / "MODEL_PROPOSALS_FOR_REVIEW.xlsx"
        _write_xlsx(
            proposal_workbook,
            [
                ("Expert-compatible layout", _workbook_rows(occurrence_rows, event_source=event_source, include_candidates=True), [18, 13, 13, 3] * max(1, len(_event_windows(occurrence_rows))), 5),
                ("Model proposal details", _table_rows(DETAIL_COLUMNS, occurrence_rows), [22, 28, 10, 15, 15, 9, 14, 12, 12, 14, 18, 14, 14, 15, 15, 30], 1),
                ("Stable model identities", _table_rows(IDENTITY_COLUMNS, identity_rows), [22, 14, 14, 15, 15, 15, 14, 14, 18, 14, 14, 30], 1),
                ("Provenance", provenance_rows, [30, 95], 1),
            ],
        )
        blinded_workbook = exports / "BLINDED_EXPERT_TEMPLATE.xlsx"
        _write_xlsx(
            blinded_workbook,
            [
                ("Expert annotation template", _workbook_rows(occurrence_rows, event_source=event_source, include_candidates=False), [18, 13, 13, 3] * max(1, len(_event_windows(occurrence_rows))), 5),
                ("Instructions", [
                    ["BLINDED EXPERT ANNOTATION TEMPLATE"],
                    ["Enter point number, x, and y under each event-window block."],
                    ["Do not open the model-proposal workbook first if independent annotation is required."],
                    ["Coordinates", "x=column, y=row"],
                    ["Frames", "one-based and inclusive"],
                    ["Event source", event_source],
                ], [30, 95], None),
            ],
        )
        audit = staging / "audit"
        _write_json(audit / "1_Expert_Annotations" / "status.json", {
            "status": "not_applicable",
            "reason": "No expert annotations were supplied.",
        })
        _write_json(audit / "2_Model_Annotations" / "status.json", {
            "status": "review_ready",
            "model_identity_count": len(identity_rows),
            "model_occurrence_count": len(occurrence_rows),
            "dashboard": "../../app/index.html#annotation-correction",
            "candidate_trace_source": "annotationCorrection.model_rois[*].traces",
            "full_media_audit": "pending_new_data_run_or_explicit_regeneration",
        })
        _write_json(audit / "3_Comparison" / "status.json", {
            "status": "not_applicable",
            "reason": "Comparison requires a later expert annotation revision.",
        })
        report = f"""# Label-Free Model Proposal Review Package

Status: review ready; expert annotations are not available.

- Dataset: `{resolved_dataset_id}`
- Frozen model run: `{frozen_run}`
- Model identities: {len(identity_rows)}
- Model occurrences: {len(occurrence_rows)}
- Event source: `{event_source}`
- Expert audit: `not_applicable_pending_labels`
- Comparison audit: `not_applicable_pending_labels`

Every proposal is unreviewed and remains unknown, not a false positive or an
expert annotation. Use `BLINDED_EXPERT_TEMPLATE.xlsx` for independent
annotation. Use `MODEL_PROPOSALS_FOR_REVIEW.xlsx` only for explicitly
model-assisted review.

The dashboard defaults to the model-only queue. Promoting a proposal creates a
separate expert annotation with model-proposal provenance; it never mutates the
frozen model proposal.
"""
        (staging / "REPORT.md").write_text(report, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "review_ready",
            "dataset_id": resolved_dataset_id,
            "frozen_run_id": frozen_run,
            "source_app": str(source_app),
            "source_review_data_sha256": _sha256(source_review),
            "expert_annotations": "not_applicable_pending_labels",
            "comparison": "not_applicable_pending_labels",
            "event_source": event_source,
            "event_windows": _event_windows(occurrence_rows),
            "model_identity_count": len(identity_rows),
            "model_occurrence_count": len(occurrence_rows),
            "dashboard_route": "app/index.html#annotation-correction",
            "files": {
                "model_proposals_workbook": "proposal_exports/MODEL_PROPOSALS_FOR_REVIEW.xlsx",
                "blinded_expert_template": "proposal_exports/BLINDED_EXPERT_TEMPLATE.xlsx",
                "model_proposals_long": "proposal_exports/model_proposals_long.tsv",
                "model_proposal_identities": "proposal_exports/model_proposal_identities.tsv",
            },
            "generated_at": now,
        }
        _write_json(staging / "proposal_manifest.json", manifest)
        validation = {
            **manifest,
            "status": "complete",
            "source_unchanged": True,
            "expert_fields_removed": True,
            "empty_annotation_revision": revision_id,
            "checksums": {
                path.name: _sha256(path)
                for path in [
                    proposal_workbook,
                    blinded_workbook,
                    exports / "model_proposals_long.tsv",
                    exports / "model_proposal_identities.tsv",
                    app / "review_data.json",
                ]
            },
        }
        _write_json(staging / "validation.json", validation)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": "complete",
        "output_root": str(output),
        "app_dir": str(output / "app"),
        "proposal_workbook": str(output / "proposal_exports" / "MODEL_PROPOSALS_FOR_REVIEW.xlsx"),
        "blinded_template": str(output / "proposal_exports" / "BLINDED_EXPERT_TEMPLATE.xlsx"),
        "model_identity_count": len(identity_rows),
        "model_occurrence_count": len(occurrence_rows),
        "expert_annotations": "not_applicable_pending_labels",
    }
