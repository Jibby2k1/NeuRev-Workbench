"""Build and score a staged, portable, detector-blind bounded-field review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contracts import atomic_json, atomic_text


PACKAGE_ID = "external_blinded_bounded_review_v2"
SCHEMA_VERSION = 2
RANDOMIZATION_SEED = 240829
ASSETS = Path(__file__).with_name("external_review_assets")
PHASE_A_REQUIRED_CLASSES = (
    "distinct_source", "overlapping_source", "identity_uncertain",
    "non_neuronal", "noise_artifact", "unresolved",
)
PHASE_B_CALLS = ("definite_neuron", "probable_neuron", "uncertain", "unlikely_neuron")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_to_global(
    display_x: float,
    display_y: float,
    *,
    header_px: float,
    media_scale: float,
    region_x0: float,
    region_y0: float,
) -> tuple[float, float]:
    """Convert intrinsic rendered-video pixels to global image coordinates."""
    if display_y < header_px:
        raise ValueError("header clicks are not annotation coordinates")
    return (
        region_x0 + display_x / media_scale,
        region_y0 + (display_y - header_px) / media_scale,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, "window.REVIEW_CONFIG = " + json.dumps(payload, sort_keys=True) + ";\n")


def _copy_common_assets(target: Path, phase: str) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for name in ("index.html", "style.css", f"phase_{phase.lower()}.js", "serve_review.py"):
        shutil.copy2(ASSETS / name, target / ("app.js" if name.startswith("phase_") else name))
    shutil.copy2(target / "index.html", target / "START_HERE.html")


def _zip_deterministic(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(str(path.relative_to(source)), date_time=(2026, 8, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def _reblind_video(source: Path, destination: Path, blind_id: str) -> None:
    """Replace the embedded candidate ID and coordinates with an opaque ID."""
    temporary = destination.with_suffix(".partial.mp4")
    text = f"Independent review | {blind_id}"
    vf = (
        "drawbox=x=0:y=0:w=iw:h=58:color=0x070b10:t=fill,"
        f"drawtext=text='{text}':x=10:y=13:fontcolor=white:fontsize=25"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(destination)


def candidate_randomization(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    shuffled = [dict(item) for item in items]
    random.Random(RANDOMIZATION_SEED).shuffle(shuffled)
    return [dict(item, blind_review_id=f"Q{index:03d}") for index, item in enumerate(shuffled, 1)]


def _phase_a(
    source_media: Path,
    target: Path,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    _copy_common_assets(target, "a")
    media_dir = target / "media"
    media_dir.mkdir()
    clips = []
    for index, item in enumerate(source_manifest["media"], 1):
        clip_id = f"R{index:02d}"
        video_name = f"{clip_id}_raw.mp4"
        shutil.copy2(source_media / item["path"], media_dir / video_name)
        burst = int(item["burst_id"])
        mean_name = f"{clip_id}_mean.png"
        max_name = f"{clip_id}_max.png"
        shutil.copy2(source_media / f"burst_{burst}_mean_projection.png", media_dir / mean_name)
        shutil.copy2(source_media / f"burst_{burst}_max_projection.png", media_dir / max_name)
        clips.append({
            "clip_id": clip_id,
            "video": f"media/{video_name}",
            "mean_projection": f"media/{mean_name}",
            "max_projection": f"media/{max_name}",
            "first_frame_ui": int(item["first_frame_ui"]),
            "last_frame_ui": int(item["last_frame_ui"]),
            "frame_count": int(item["frame_count"]),
            "fps": int(item["fps"]),
        })
    region = source_manifest["region"]
    config = {
        "schema_version": SCHEMA_VERSION,
        "package_id": PACKAGE_ID,
        "phase": "A_raw_first",
        "title": "Independent Raw-first bounded-field review",
        "instructions": "Mark every visible source in every clip before viewing the assisted package.",
        "classes": list(PHASE_A_REQUIRED_CLASSES),
        "confidence_values": [1, 2, 3, 4, 5],
        "coordinate_contract": "x=column, y=row; UI frames are one-based and inclusive",
        "display_geometry": {"intrinsic_width": 576, "intrinsic_height": 630, "header_px": 54, "crop_scale": 3},
        "region": {key: region[key] for key in ("x0", "y0", "width", "height")},
        "clips": clips,
        "blinding": "raw_only; no detector proposals, labels, scores, ranks, or candidate identities",
        "completion": "all clips individually signed off plus full-region certification",
    }
    _write_config(target / "config.js", config)
    atomic_text(target / "README.md", _phase_a_readme())
    return config


def _phase_b(
    candidate_media: Path,
    target: Path,
    randomized: list[dict[str, Any]],
) -> dict[str, Any]:
    _copy_common_assets(target, "b")
    media_dir = target / "media"
    media_dir.mkdir()
    public_items = []
    for item in randomized:
        blind_id = item["blind_review_id"]
        source = candidate_media / item["video"]
        destination = media_dir / f"{blind_id}_six_panel.mp4"
        _reblind_video(source, destination, blind_id)
        public_items.append({"blind_id": blind_id, "video": f"media/{destination.name}"})
    config = {
        "schema_version": SCHEMA_VERSION,
        "package_id": PACKAGE_ID,
        "phase": "B_assisted",
        "title": "Independent processed-evidence review",
        "instructions": "Complete only after the Raw-first export has been submitted and locked.",
        "calls": list(PHASE_B_CALLS),
        "identity_relations": ["distinct_source", "overlapping_or_multiple", "duplicate_or_neighbor_capture", "other_biological_feature", "artifact_or_noise", "unresolved"],
        "stage_visibility": ["visible", "not_visible", "uncertain"],
        "morphology_flags": ["small", "discreet", "crescent", "non_circular", "possible_multiple", "other"],
        "limitation_flags": ["low_snr", "artifact_adjacent", "strong_neighbor", "off_center", "skin_surface", "none"],
        "confidence_values": [1, 2, 3, 4, 5],
        "items": public_items,
        "blinding": "opaque IDs; no current labels, detector sites, scores, ranks, or randomization key",
        "completion": "one complete rating per item plus phase-order certification",
    }
    _write_config(target / "config.js", config)
    atomic_text(target / "README.md", _phase_b_readme())
    return config


def _phase_a_readme() -> str:
    return """# Phase A: Raw-first bounded-field review

Open `START_HERE.html` in a modern browser. If local video playback is blocked,
run `python serve_review.py` and open the printed local address.

Enter only the opaque reviewer ID supplied by the administrator. Review every
clip and click every visible source. Complete the source class, confidence,
identity-across-clips, timing, and notes. Sign off each clip even when no source
is visible. Export the locked JSON and return it to the administrator.

Do not open Phase B until the administrator confirms receipt of Phase A. Do not
discuss annotations with another reviewer before both submissions are locked.
"""


def _phase_b_readme() -> str:
    return """# Phase B: processed-evidence review

Open `START_HERE.html` only after the administrator has accepted your Phase A
submission. Rate every opaque candidate using the synchronized Raw, ICA, local-
standardized, and complete-trace panels. Record neuron likelihood, identity
relation, stage visibility, morphology, limitations, possible center offset,
confidence, and notes. Export the locked JSON and return it to the administrator.

The package contains no current decisions, detector ranks, or scoring key.
"""


def _private_bundle(
    target: Path,
    randomized: list[dict[str, Any]],
    review_rows: dict[str, dict[str, Any]],
    phase_a: dict[str, Any],
    phase_b: dict[str, Any],
) -> None:
    target.mkdir(parents=True, exist_ok=False)
    key_rows = []
    for item in randomized:
        reference = review_rows[item["blind_id"]]
        key_rows.append({
            "blind_review_id": item["blind_review_id"],
            "source_blind_id": item["blind_id"],
            "detection_site_id": item["detection_site_id"],
            "reference_single_reviewer_label": reference["normalized_label"],
            "reference_confidence": int(reference["confidence_1_to_5"]),
        })
    atomic_json(target / "phase_B_randomization_key.json", {
        "schema_version": SCHEMA_VERSION,
        "private": True,
        "seed": RANDOMIZATION_SEED,
        "items": key_rows,
    })
    atomic_json(target / "scoring_contract.json", {
        "schema_version": SCHEMA_VERSION,
        "package_id": PACKAGE_ID,
        "phase_A_clip_ids": [item["clip_id"] for item in phase_a["clips"]],
        "phase_B_item_ids": [item["blind_id"] for item in phase_b["items"]],
        "phase_A_matching_radius_px": 6.0,
        "minimum_reviewers": 2,
        "adjudication_required": True,
        "permitted_claim_scope": "frozen enriched bounded region in one recording",
        "prohibited_claims": ["whole-recording precision", "independent-recording generalization", "histological identity", "causal connectivity"],
    })
    shutil.copy2(ASSETS / "score_submissions.py", target / "score_submissions.py")
    atomic_text(target / "ADMIN_README.md", _admin_readme())


def _admin_readme() -> str:
    return """# Private administrator bundle

Do not share this directory or archive with reviewers.

1. Give each reviewer an opaque ID.
2. Send only the Phase A ZIP.
3. Validate and retain each locked Phase A JSON.
4. Send Phase B only after Phase A receipt.
5. Retain the randomization key privately.
6. After at least two complete reviewers, run:

```bash
python score_submissions.py --private-dir . --output analysis reviewer_A_phase_A.json reviewer_B_phase_A.json reviewer_A_phase_B.json reviewer_B_phase_B.json
```

The automated output is an agreement and disagreement package, not final truth.
An adjudicator must resolve source identities and classifications before detector
precision or recall is computed.
"""


def _submission_phase(payload: dict[str, Any]) -> str:
    return str(payload.get("phase", ""))


def validate_submission(payload: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if payload.get("package_id") != PACKAGE_ID:
        errors.append("package_id mismatch")
    if not str(payload.get("reviewer_id", "")).strip():
        errors.append("reviewer_id missing")
    if not payload.get("locked"):
        errors.append("submission is not locked")
    phase = _submission_phase(payload)
    if phase == "A_raw_first":
        expected = set(contract["phase_A_clip_ids"])
        coverage = {key for key, value in (payload.get("coverage") or {}).items() if value}
        if coverage != expected:
            errors.append("Phase A coverage incomplete")
        if not payload.get("full_region_certified"):
            errors.append("Phase A full-region certification missing")
        for mark in payload.get("marks", []):
            if mark.get("clip_id") not in expected:
                errors.append("Phase A mark has unknown clip")
            if mark.get("class") not in PHASE_A_REQUIRED_CLASSES:
                errors.append("Phase A mark has unsupported class")
    elif phase == "B_assisted":
        expected = set(contract["phase_B_item_ids"])
        ratings = payload.get("ratings") or {}
        if set(ratings) != expected:
            errors.append("Phase B ratings incomplete")
        if not payload.get("phase_order_certified"):
            errors.append("Phase B phase-order certification missing")
        for rating in ratings.values():
            if rating.get("neuron_call") not in PHASE_B_CALLS:
                errors.append("Phase B rating has unsupported neuron_call")
            if rating.get("identity_relation") not in ("distinct_source", "overlapping_or_multiple", "duplicate_or_neighbor_capture", "other_biological_feature", "artifact_or_noise", "unresolved"):
                errors.append("Phase B rating has unsupported identity_relation")
            for field in ("raw_visibility", "ica_visibility", "ls_visibility"):
                if rating.get(field) not in ("visible", "not_visible", "uncertain"):
                    errors.append(f"Phase B rating has unsupported {field}")
    else:
        errors.append("unsupported phase")
    return sorted(set(errors))


def _match_marks(a: list[dict[str, Any]], b: list[dict[str, Any]], radius: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not a or not b:
        return [], list(range(len(a))), list(range(len(b)))
    distance = np.asarray([
        [np.hypot(float(x["x_px_global"]) - float(y["x_px_global"]), float(x["y_px_global"]) - float(y["y_px_global"])) for y in b]
        for x in a
    ])
    rows, cols = linear_sum_assignment(distance)
    matches = [(int(i), int(j), float(distance[i, j])) for i, j in zip(rows, cols) if distance[i, j] <= radius]
    used_a = {i for i, _, _ in matches}
    used_b = {j for _, j, _ in matches}
    return matches, [i for i in range(len(a)) if i not in used_a], [j for j in range(len(b)) if j not in used_b]


def analyze_submissions(paths: list[Path], private_dir: Path, output: Path) -> dict[str, Any]:
    contract = _read_json(private_dir / "scoring_contract.json")
    key = _read_json(private_dir / "phase_B_randomization_key.json")
    payloads = [_read_json(path) for path in paths]
    validation = {path.name: validate_submission(payload, contract) for path, payload in zip(paths, payloads)}
    if any(validation.values()):
        raise ValueError("invalid submissions: " + json.dumps(validation, sort_keys=True))
    output.mkdir(parents=True, exist_ok=False)
    phase_a = [p for p in payloads if _submission_phase(p) == "A_raw_first"]
    phase_b = [p for p in payloads if _submission_phase(p) == "B_assisted"]
    if len({p["reviewer_id"] for p in phase_a}) < 2 or len({p["reviewer_id"] for p in phase_b}) < 2:
        raise ValueError("at least two independent reviewers are required in each phase")

    spatial_rows = []
    disagreement_rows = []
    a0, a1 = phase_a[:2]
    for clip_id in contract["phase_A_clip_ids"]:
        marks0 = [m for m in a0.get("marks", []) if m["clip_id"] == clip_id]
        marks1 = [m for m in a1.get("marks", []) if m["clip_id"] == clip_id]
        matches, only0, only1 = _match_marks(marks0, marks1, float(contract["phase_A_matching_radius_px"]))
        for i, j, distance in matches:
            same_class = marks0[i].get("class") == marks1[j].get("class")
            spatial_rows.append({"clip_id": clip_id, "reviewer_A_mark": marks0[i]["mark_id"], "reviewer_B_mark": marks1[j]["mark_id"], "distance_px": distance, "same_class": int(same_class)})
            if not same_class:
                disagreement_rows.append({"phase": "A", "item_id": clip_id, "reviewer_A": marks0[i]["mark_id"], "reviewer_B": marks1[j]["mark_id"], "reason": "class_disagreement"})
        for index in only0:
            disagreement_rows.append({"phase": "A", "item_id": clip_id, "reviewer_A": marks0[index]["mark_id"], "reviewer_B": "", "reason": "unmatched_reviewer_A_mark"})
        for index in only1:
            disagreement_rows.append({"phase": "A", "item_id": clip_id, "reviewer_A": "", "reviewer_B": marks1[index]["mark_id"], "reason": "unmatched_reviewer_B_mark"})

    reference = {row["blind_review_id"]: row for row in key["items"]}
    phase_b_rows = []
    b0, b1 = phase_b[:2]
    for item_id in contract["phase_B_item_ids"]:
        rating0 = b0["ratings"][item_id]
        rating1 = b1["ratings"][item_id]
        same_call = rating0["neuron_call"] == rating1["neuron_call"]
        phase_b_rows.append({
            "blind_review_id": item_id,
            "reviewer_A_call": rating0["neuron_call"],
            "reviewer_B_call": rating1["neuron_call"],
            "same_call": int(same_call),
            "reference_single_reviewer_label": reference[item_id]["reference_single_reviewer_label"],
        })
        if not same_call:
            disagreement_rows.append({"phase": "B", "item_id": item_id, "reviewer_A": rating0["neuron_call"], "reviewer_B": rating1["neuron_call"], "reason": "neuron_call_disagreement"})

    _write_tsv(output / "phase_A_spatial_matches.tsv", spatial_rows)
    _write_tsv(output / "phase_B_call_agreement.tsv", phase_b_rows)
    _write_tsv(output / "adjudication_queue.tsv", disagreement_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "agreement_complete_adjudication_required",
        "reviewers": sorted({p["reviewer_id"] for p in payloads}),
        "phase_A": {
            "reviewer_mark_counts": [len(p.get("marks", [])) for p in phase_a[:2]],
            "spatial_matches": len(spatial_rows),
            "median_match_distance_px": float(np.median([r["distance_px"] for r in spatial_rows])) if spatial_rows else None,
            "matched_class_agreement": float(np.mean([r["same_class"] for r in spatial_rows])) if spatial_rows else None,
        },
        "phase_B": {
            "items": len(phase_b_rows),
            "exact_call_agreement": float(np.mean([r["same_call"] for r in phase_b_rows])),
            "reviewer_call_counts": [dict(Counter(p["ratings"][i]["neuron_call"] for i in contract["phase_B_item_ids"])) for p in phase_b[:2]],
        },
        "adjudication_items": len(disagreement_rows),
        "claim_boundary": contract["permitted_claim_scope"],
        "prohibited_claims": contract["prohibited_claims"],
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "validation.json", {"status": "passed", "submissions": len(payloads), "submission_errors": validation, "adjudication_required": True})
    atomic_text(output / "REPORT.md", "# External bounded review agreement\n\nAutomated agreement analysis completed. Final truth and detector metrics remain unavailable until the adjudication queue is resolved. See `summary.json` and the TSV tables.\n")
    return summary


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["phase", "item_id", "reviewer_A", "reviewer_B", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _public_blinding_scan(root: Path, forbidden: Iterable[str]) -> list[dict[str, str]]:
    findings = []
    text_suffixes = {".html", ".js", ".css", ".json", ".md", ".py", ".txt", ".tsv"}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = str(path.relative_to(root))
        haystacks = [relative.lower()]
        if path.suffix.lower() in text_suffixes:
            haystacks.append(path.read_text(errors="replace").lower())
        for token in forbidden:
            if any(token.lower() in haystack for haystack in haystacks):
                findings.append({"path": relative, "token": token})
    return findings


def build_package(source_run: Path, candidate_media: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    source_media = source_run / "review_packet" / "bounded_field_media"
    source_manifest = _read_json(source_media / "media_manifest.json")
    candidate_manifest = _read_json(candidate_media / "manifest.json")
    review_rows = {row["blind_id"]: row for row in csv.DictReader((candidate_media / "user_review_v1.tsv").open(), delimiter="\t")}
    randomized = candidate_randomization(candidate_manifest["items"])
    phase_a = _phase_a(source_media, partial / "public_phase_A_raw_first", source_manifest)
    phase_b = _phase_b(candidate_media, partial / "public_phase_B_assisted", randomized)
    _private_bundle(partial / "private_administrator", randomized, review_rows, phase_a, phase_b)

    forbidden_a = ["dsite_", "priority_score", "known_positive", "NC001", "randomization_key", "reference_single_reviewer"]
    forbidden_b = ["dsite_", "priority_score", "known_positive", "randomization_key", "reference_single_reviewer", "user_review_v1"]
    blinding_a = _public_blinding_scan(partial / "public_phase_A_raw_first", forbidden_a)
    blinding_b = _public_blinding_scan(partial / "public_phase_B_assisted", forbidden_b)
    if blinding_a or blinding_b:
        raise ValueError(f"public blinding scan failed: {blinding_a + blinding_b}")

    archives = []
    for directory, name in (
        (partial / "public_phase_A_raw_first", "NeuRev_external_review_v2_PHASE_A_RAW_FIRST.zip"),
        (partial / "public_phase_B_assisted", "NeuRev_external_review_v2_PHASE_B_ASSISTED.zip"),
        (partial / "private_administrator", "NeuRev_external_review_v2_PRIVATE_ADMIN.zip"),
    ):
        destination = partial / name
        _zip_deterministic(directory, destination)
        archives.append({"path": name, "bytes": destination.stat().st_size, "sha256": sha256(destination)})
    contract = {
        "schema_version": SCHEMA_VERSION,
        "package_id": PACKAGE_ID,
        "status": "ready_for_two_independent_reviewers",
        "region": source_manifest["region"],
        "phases": ["A_raw_first", "B_assisted_after_locked_A"],
        "reviewers_required": 2,
        "adjudication_required": True,
        "phase_A_clips": len(phase_a["clips"]),
        "phase_B_items": len(phase_b["items"]),
        "public_private_separation": True,
        "claim_scope": "independent annotation validation within one frozen enriched bounded region",
        "not_external_biological_replication": True,
        "archives": archives,
    }
    atomic_json(partial / "experiment_contract.json", contract)
    atomic_json(partial / "validation.json", {
        "status": "passed",
        "phase_A_media": len(phase_a["clips"]),
        "phase_B_media": len(phase_b["items"]),
        "public_phase_A_blinding_findings": blinding_a,
        "public_phase_B_blinding_findings": blinding_b,
        "archives": len(archives),
        "source_manifest_sha256": sha256(source_media / "media_manifest.json"),
        "candidate_manifest_sha256": sha256(candidate_media / "manifest.json"),
    })
    atomic_text(partial / "README.md", "# External blinded bounded review v2\n\nShare Phase A first. Share Phase B only after receiving a locked Phase A JSON. Never share the private administrator ZIP. See `private_administrator/ADMIN_README.md`.\n")
    artifacts = []
    for path in sorted(p for p in partial.rglob("*") if p.is_file() and p.name != "artifact_index.json"):
        artifacts.append({"path": str(path.relative_to(partial)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(partial / "artifact_index.json", {"schema_version": 1, "artifacts": artifacts})
    partial.replace(output)
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source-run", type=Path, required=True)
    build.add_argument("--candidate-media", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--private-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("submissions", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build_package(args.source_run, args.candidate_media, args.output), indent=2))
    else:
        print(json.dumps(analyze_submissions(args.submissions, args.private_dir, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
