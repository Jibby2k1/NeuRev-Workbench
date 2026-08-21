"""Collision-safe builder and read-only audit for exhaustive truth-set packages."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from neurobench.models.truth_set import TruthSetManifest, validate_event_record, validate_object_record
from neurobench.review.truth_set import assert_blinded_payload, build_candidate_union, deterministic_second_review_sample, sha256_payload


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_json_text(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_input(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def preflight_truth_set(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = TruthSetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    payload = manifest.to_dict()
    checks: list[dict[str, Any]] = []
    source = _resolve_input(path, payload["source_video"]["path"])
    source_ok = source.is_file() and _sha256_file(source) == payload["source_video"]["sha256"]
    checks.append({"name": "source_video_checksum", "passed": source_ok, "path": str(source)})
    revision_refs = payload["annotation_revisions"]
    base_revision = _resolve_input(path, revision_refs["base_revision_path"])
    checks.append({"name": "base_annotation_revision", "passed": base_revision.is_file() and _sha256_file(base_revision) == revision_refs["base_revision_sha256"], "path": str(base_revision)})
    for revision in revision_refs["published_revisions"]:
        revision_path = _resolve_input(path, revision["path"])
        valid = revision_path.is_file() and _sha256_file(revision_path) == revision["sha256"]
        if valid:
            revision_payload = json.loads(revision_path.read_text(encoding="utf-8"))
            valid = revision_payload.get("revisionId") == revision["revision_id"] and revision_payload.get("state") == "published"
        checks.append({"name": f"published_annotation_revision:{revision['revision_id']}", "passed": valid, "path": str(revision_path)})
    for region in payload["regions"]:
        mask = _resolve_input(path, region["coverage_mask"]["path"])
        passed = mask.is_file() and _sha256_file(mask) == region["coverage_mask"]["sha256"]
        checks.append({"name": f"coverage_mask:{region['region_id']}", "passed": passed, "path": str(mask)})
    for lane in payload["frozen_candidate_panel"]["lane_artifacts"]:
        artifact = _resolve_input(path, lane["artifact_path"])
        scores = _resolve_input(path, lane["score_array_path"])
        checks.append({"name": f"lane_artifact:{lane['lane_id']}", "passed": artifact.is_file() and _sha256_file(artifact) == lane["artifact_sha256"], "path": str(artifact)})
        checks.append({"name": f"score_array:{lane['lane_id']}", "passed": scores.is_file() and _sha256_file(scores) == lane["score_array_sha256"], "path": str(scores)})
    lane_ids = sorted({str(item["source_lane_id"]) for item in payload["candidates"]})
    checks.append({"name": "frozen_candidate_lanes", "passed": bool(lane_ids), "lane_ids": lane_ids})
    checks.append({"name": "candidate_panel_checksum", "passed": sha256_payload(payload["candidates"]) == payload["frozen_candidate_panel"]["sha256"]})
    policy_for_checksum = dict(payload["frozen_analysis_policy"])
    declared_policy_checksum = policy_for_checksum.pop("sha256")
    checks.append({"name": "analysis_policy_checksum", "passed": sha256_payload(policy_for_checksum) == declared_policy_checksum})
    checks.append({"name": "protected_region_label_free_selection", "passed": all(item["selection_mode"] in {"deterministic_tissue_mask", "random_label_free", "synthetic_fixture"} for item in payload["regions"] if item["role"] == "protected")})
    return {"manifest": str(path), "truth_set_id": payload["truth_set_id"], "passed": all(item["passed"] for item in checks), "checks": checks, "resolved_manifest_sha256": sha256_payload(payload)}


def build_truth_set_package(manifest_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Build an initial raw-first package atomically; never overwrite a target."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    preflight = preflight_truth_set(manifest_file)
    if not preflight["passed"]:
        raise ValueError("truth-set preflight failed: " + ", ".join(item["name"] for item in preflight["checks"] if not item["passed"]))
    manifest = TruthSetManifest.from_dict(json.loads(manifest_file.read_text(encoding="utf-8"))).to_dict()
    target = Path(output_root).expanduser().resolve()
    outputs_root = (PROJECT_ROOT / "Outputs").resolve()
    if PROJECT_ROOT in target.parents and outputs_root not in target.parents:
        raise ValueError("truth-set generated outputs inside the repository must live under ignored Outputs/")
    if target.exists():
        raise FileExistsError(f"Refusing truth-set output collision: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    try:
        policy = manifest["frozen_analysis_policy"]
        blinded, source_key = build_candidate_union(
            manifest["candidates"],
            spatial_radius_px=float(policy["matching_radius_px"]),
            temporal_radius_frames=int(policy.get("temporal_matching_radius_frames", 0)),
            random_seed=int(policy.get("candidate_randomization_seed", 0)),
        )
        assert_blinded_payload(blinded)
        resolved = {key: value for key, value in manifest.items() if key != "candidates"}
        resolved["candidate_count"] = len(blinded)
        resolved["resolved_manifest_sha256"] = sha256_payload(manifest)
        _write_json(staging / "truth_set_manifest.json", manifest)
        _write_json(staging / "resolved_manifest.json", resolved)
        _write_json(staging / "input_fingerprints.json", {"source_video": manifest["source_video"]["sha256"], "coverage_masks": {item["region_id"]: item["coverage_mask"]["sha256"] for item in manifest["regions"]}, "annotation_revisions": {"base": manifest["annotation_revisions"]["base_revision_sha256"], **{item["revision_id"]: item["sha256"] for item in manifest["annotation_revisions"]["published_revisions"]}}})
        for region in manifest["regions"]:
            region_root = staging / "regions" / region["region_id"]
            for pass_name in ("raw_first", "candidate_assisted", "second_review", "adjudication"):
                (region_root / pass_name).mkdir(parents=True, exist_ok=True)
            _write_json(region_root / "region.json", region)
            source_mask = _resolve_input(manifest_file, region["coverage_mask"]["path"])
            shutil.copy2(source_mask, region_root / f"coverage_mask{source_mask.suffix}")
        _write_json(staging / "candidate_panel" / "blinded_candidates.json", {"schema_version": 1, "candidates": blinded})
        _write_tsv(staging / "candidate_panel" / "blinded_candidates.tsv", blinded)
        _write_json(staging / "candidate_panel" / "candidate_panel_checksum.json", {"sha256": sha256_payload(blinded), "candidate_count": len(blinded)})
        _write_json(staging / "private" / "candidate_source_key.json", {"schema_version": 1, "sealed": True, "key": source_key})
        _write_tsv(staging / "annotations" / "objects.tsv", [])
        _write_tsv(staging / "annotations" / "events.tsv", [])
        _write_tsv(staging / "annotations" / "unresolved.tsv", [])
        revision_refs = json.loads(json.dumps(manifest["annotation_revisions"]))
        base_revision_target = staging / "annotations" / "revisions" / f"{revision_refs['base_revision_id']}.json"
        base_revision_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_resolve_input(manifest_file, revision_refs["base_revision_path"]), base_revision_target)
        revision_refs["base_revision_path"] = f"revisions/{base_revision_target.name}"
        for revision in revision_refs["published_revisions"]:
            target_revision = base_revision_target.parent / f"{revision['revision_id']}.json"
            shutil.copy2(_resolve_input(manifest_file, revision["path"]), target_revision)
            revision["path"] = f"revisions/{target_revision.name}"
        _write_json(staging / "annotations" / "revision_references.json", revision_refs)
        raw_payload = {
            "schema_version": 1,
            "truth_set_id": manifest["truth_set_id"],
            "mode": "truth_set",
            "review_pass": "raw_first",
            "blinding_state": "raw_first",
            "regions": [{key: region[key] for key in ("region_id", "role", "coverage_mode", "review_status", "spatial_bounds_px", "ui_frame_interval")} for region in manifest["regions"]],
            "candidates": [],
            "permitted_evidence": ["raw_video", "neutral_fixed_views", "region_boundary", "traces", "pixel_probes", "coverage_progress"],
        }
        assert_blinded_payload(raw_payload)
        _write_json(staging / "review" / "review_payload.json", raw_payload)
        _write_json(staging / "review" / "reviewer_provenance.json", manifest["review_provenance"])
        _write_json(staging / "review" / "agreement.json", {"status": "not_started"})
        _write_tsv(staging / "review" / "disagreement_queue.tsv", [])
        frozen = {
            "schema_version": 1,
            "candidate_panel_id": manifest["frozen_candidate_panel"]["panel_id"],
            "candidate_panel_sha256": sha256_payload(blinded),
            "lane_ids": sorted({item["source_lane_id"] for item in manifest["candidates"]}),
            "lane_artifacts": manifest["frozen_candidate_panel"]["lane_artifacts"],
            "source_candidates_sha256": sha256_payload(manifest["candidates"]),
            "analysis_policy": policy,
            "analysis_policy_sha256": sha256_payload(policy),
            "region_selection": {item["region_id"]: {"seed": item.get("selection_seed"), "mask_sha256": item["coverage_mask"]["sha256"]} for item in manifest["regions"]},
            "analysis_code_git_sha": os.environ.get("NEUROBENCH_GIT_SHA", "synthetic-fixture-no-git-sha"),
            "resolved_manifest_sha256": sha256_payload(resolved),
        }
        _write_json(staging / "freeze" / "frozen_lane_manifest.json", frozen)
        _write_json(staging / "freeze" / "protected_lock.json", {"state": "locked", "locked_at": manifest["timestamps"]["locked_at"], "frozen_lane_manifest_sha256": sha256_payload(frozen), "unsealed_at": None, "unsealed_by": None, "unseal_reason": None})
        _write_tsv(staging / "metrics" / "review_efficiency.csv", [])
        _write_json(staging / "metrics" / "exhaustive_object_metrics.json", {"status": "not_available_until_exhaustive_review"})
        _write_json(staging / "metrics" / "exhaustive_event_metrics.json", {"status": "not_available_until_exhaustive_review"})
        _write_json(staging / "metrics" / "grouped_intervals.json", {"grouping": ["persistent_object_identity", "burst"], "status": "not_available"})
        validation = audit_truth_set_root(staging)
        _write_json(staging / "validation.json", validation)
        _write_json(staging / "artifact_index.json", _artifact_index(staging))
        _write_json(staging / "llm_context.json", {"truth_set_id": manifest["truth_set_id"], "state": "raw_first", "a0_decision": validation["decision"], "limitations": ["synthetic fixture only" if "synthetic" in manifest["dataset_id"] else "human review incomplete", "no detector performance claim"]})
        (staging / "REPORT.md").write_text(f"# Truth Set {manifest['truth_set_id']}\n\nInitial raw-first package. A0: **{validation['decision']}**.\n\nNo detector evaluation has been opened.\n", encoding="utf-8")
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"truth_set_root": str(target), "truth_set_id": manifest["truth_set_id"], "candidate_count": len(blinded), "a0": audit_truth_set_root(target)}


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_tsv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(item) for item in csv.DictReader(handle, delimiter="\t") if not (set(item) == {"empty"} and not item.get("empty"))]


def _artifact_index(root: Path) -> dict[str, Any]:
    return {"schema_version": 1, "artifacts": [{"path": item.relative_to(root).as_posix(), "sha256": _sha256_file(item)} for item in sorted(root.rglob("*")) if item.is_file() and item.name != "artifact_index.json"]}


def reveal_candidate_assisted_payload(root: str | Path) -> dict[str, Any]:
    """Reveal only opaque candidates after every region's raw-first pass is locked."""
    package = Path(root).expanduser().resolve()
    if not (package / "freeze" / "frozen_lane_manifest.json").is_file():
        raise ValueError("candidate-assisted protected review requires a frozen lane manifest")
    manifest = json.loads((package / "truth_set_manifest.json").read_text(encoding="utf-8"))
    if any(not region["raw_first_complete"] or region["review_status"] not in {"raw_first_locked", "candidate_assisted_in_progress", "candidate_assisted_locked", "second_review", "adjudication", "adjudicated", "published"} for region in manifest["regions"]):
        raise ValueError("raw-first review must be complete and locked before candidate-assisted review")
    candidates = json.loads((package / "candidate_panel" / "blinded_candidates.json").read_text(encoding="utf-8"))["candidates"]
    payload = json.loads((package / "review" / "review_payload.json").read_text(encoding="utf-8"))
    payload.update({"review_pass": "candidate_assisted", "blinding_state": "candidate_blinded", "candidates": candidates})
    assert_blinded_payload(payload)
    _write_json(package / "review" / "review_payload.json", payload)
    return payload


def load_candidate_source_key(root: str | Path) -> dict[str, Any]:
    """Return the private source key only after the protected package is unsealed."""
    package = Path(root).expanduser().resolve()
    lock = json.loads((package / "freeze" / "protected_lock.json").read_text(encoding="utf-8"))
    if lock.get("state") != "unsealed":
        raise PermissionError("candidate source key is sealed until protected unsealing")
    return json.loads((package / "private" / "candidate_source_key.json").read_text(encoding="utf-8"))


def lock_raw_first_region(
    root: str | Path,
    *,
    region_id: str,
    published_revision_id: str,
    objects: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Lock one raw-first pass after a referenced published revision is supplied."""
    package = Path(root).expanduser().resolve()
    path = package / "truth_set_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if published_revision_id not in manifest["annotation_revisions"]["published_revision_ids"]:
        raise ValueError("raw-first lock requires a referenced published annotation revision")
    region = next((item for item in manifest["regions"] if item["region_id"] == region_id), None)
    if region is None:
        raise ValueError(f"unknown region: {region_id}")
    if region["raw_first_complete"]:
        raise ValueError("raw-first region is already locked")
    record_truth_set_annotations(package, objects=objects, events=events)
    region["raw_first_complete"] = True
    region["review_status"] = "raw_first_locked"
    region["raw_first_revision_id"] = published_revision_id
    # raw_first_revision_id is package state, not part of the immutable input schema.
    region.pop("raw_first_revision_id")
    _write_json(path, manifest)
    return region


def record_candidate_dispositions(root: str | Path, dispositions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    package = Path(root).expanduser().resolve()
    panel = json.loads((package / "candidate_panel" / "blinded_candidates.json").read_text(encoding="utf-8"))["candidates"]
    known = {item["candidate_id"]: item for item in panel}
    rows = [dict(item) for item in dispositions]
    ids = [str(item.get("candidate_id") or "") for item in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(known):
        raise ValueError("every blinded union candidate requires exactly one disposition")
    allowed = {"neuron", "artifact", "background", "unresolved"}
    if any(item.get("disposition") not in allowed for item in rows):
        raise ValueError("candidate disposition must be neuron, artifact, background, or unresolved")
    _write_tsv(package / "candidate_panel" / "candidate_dispositions.tsv", rows)
    unresolved_path = package / "annotations" / "unresolved.tsv"
    retained = [item for item in _read_tsv(unresolved_path) if item.get("record_type") != "candidate"]
    retained.extend({"record_type": "candidate", "subject_id": item["candidate_id"], "disposition": "unresolved"} for item in rows if item["disposition"] == "unresolved")
    _write_tsv(unresolved_path, retained)
    manifest_path = package / "truth_set_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_region = {region["region_id"]: [] for region in manifest["regions"]}
    for item in rows:
        by_region[known[item["candidate_id"]]["region_id"]].append(item)
    for region in manifest["regions"]:
        members = by_region[region["region_id"]]
        region["candidate_union_count"] = len(members)
        region["candidate_union_disposition_count"] = len(members)
        region["unresolved_count"] = sum(item["disposition"] == "unresolved" for item in members)
        region["review_status"] = "candidate_assisted_locked"
    _write_json(manifest_path, manifest)
    return {"candidate_count": len(rows), "unresolved_count": sum(item["disposition"] == "unresolved" for item in rows)}


def record_truth_set_annotations(root: str | Path, *, objects: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> None:
    package = Path(root).expanduser().resolve()
    object_rows = [validate_object_record(item) for item in objects]
    event_rows = [validate_event_record(item) for item in events]
    object_ids = {item["object_id"] for item in object_rows}
    if any(item["object_id"] not in object_ids for item in event_rows):
        raise ValueError("every event must reference a persistent object")
    _write_tsv(package / "annotations" / "objects.tsv", object_rows)
    _write_tsv(package / "annotations" / "events.tsv", event_rows)
    unresolved = [{"record_type": "object", **item} for item in object_rows if item["disposition"] == "unresolved"] + [{"record_type": "event", **item} for item in event_rows if item["disposition"] == "unresolved"]
    _write_tsv(package / "annotations" / "unresolved.tsv", unresolved)


def second_review_selection(root: str | Path, dispositions: Sequence[Mapping[str, Any]], *, seed: int) -> list[str]:
    selected = deterministic_second_review_sample(dispositions, seed=seed, fraction=0.20)
    _write_json(Path(root) / "review" / "second_review_sample.json", {"seed": seed, "fraction": 0.20, "subject_ids": selected})
    return selected


def record_second_review(
    root: str | Path,
    *,
    selected_subject_ids: Sequence[str],
    reviewer_a: Mapping[str, Any],
    reviewer_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist independent-review agreement using the existing agreement owner."""
    from neurobench.review.agreement import annotation_agreement_report, disagreement_tsv_rows

    package = Path(root).expanduser().resolve()
    expected = set(json.loads((package / "review" / "second_review_sample.json").read_text(encoding="utf-8"))["subject_ids"])
    if set(selected_subject_ids) != expected:
        raise ValueError("second-review records must cover the frozen sample exactly")
    report = annotation_agreement_report(reviewer_a, reviewer_b)
    _write_json(package / "review" / "agreement.json", report)
    _write_tsv(package / "review" / "disagreement_queue.tsv", disagreement_tsv_rows(report))
    manifest_path = package / "truth_set_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for region in manifest["regions"]:
        region["second_review_sample_count"] = len(expected)
        region["review_status"] = "adjudication" if report["disagreement_queue"] else "adjudicated"
    _write_json(manifest_path, manifest)
    return report


def record_adjudication(root: str | Path, decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    package = Path(root).expanduser().resolve()
    report = json.loads((package / "review" / "agreement.json").read_text(encoding="utf-8"))
    required = {(item["subject_group"], str(item["subject_id"])) for item in report.get("disagreement_queue", [])}
    supplied = {(str(item.get("subject_group")), str(item.get("subject_id"))) for item in decisions}
    if supplied != required:
        raise ValueError("adjudication must cover every disagreement exactly")
    if any(item.get("decision") not in {"accepted", "rejected", "neuron", "artifact", "background", "event", "unresolved"} for item in decisions):
        raise ValueError("invalid adjudication decision")
    _write_json(package / "review" / "adjudication.json", {"decisions": list(decisions)})
    manifest_path = package / "truth_set_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for region in manifest["regions"]:
        region["review_status"] = "adjudicated"
    manifest["coverage_status"] = "complete"
    manifest["review_pass_state"] = "adjudication"
    _write_json(manifest_path, manifest)
    return {"adjudicated_count": len(decisions), "unresolved_count": sum(item["decision"] == "unresolved" for item in decisions)}


def unseal_protected(root: str | Path, *, reviewer_id: str, reason: str, timestamp: str | None = None) -> dict[str, Any]:
    package = Path(root).expanduser().resolve()
    path = package / "freeze" / "protected_lock.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["state"] == "unsealed":
        raise ValueError("protected lock is already unsealed and immutable")
    frozen = json.loads((package / "freeze" / "frozen_lane_manifest.json").read_text(encoding="utf-8"))
    if sha256_payload(frozen) != lock["frozen_lane_manifest_sha256"]:
        raise ValueError("frozen lane manifest changed after protected lock")
    if not reviewer_id.strip() or not reason.strip():
        raise ValueError("unseal reviewer and reason are required")
    unsealed_at = timestamp or datetime.now(timezone.utc).isoformat()
    lock.update({"state": "unsealed", "unsealed_at": unsealed_at, "unsealed_by": reviewer_id, "unseal_reason": reason})
    _write_json(path, lock)
    manifest_path = package / "truth_set_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protected_lock_state"] = "unsealed"
    manifest["blinding_state"] = "unsealed"
    manifest["timestamps"]["unsealed_at"] = unsealed_at
    manifest["timestamps"]["updated_at"] = unsealed_at
    for region in manifest["regions"]:
        if region["role"] == "protected":
            region["protected_lock"].update({"locked": True, "unsealed": True, "unsealed_at": unsealed_at, "unsealed_by": reviewer_id, "unseal_reason": reason})
    _write_json(manifest_path, manifest)
    return lock


def publish_truth_set(root: str | Path, *, timestamp: str | None = None) -> dict[str, Any]:
    package = Path(root).expanduser().resolve()
    marker = package / "publication.json"
    if marker.exists():
        raise FileExistsError("truth set is already published and immutable")
    result = audit_truth_set_root(package)
    if result["decision"] != "advance":
        raise ValueError("A0 must advance before immutable truth-set publication")
    manifest_path = package / "truth_set_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    publication_time = timestamp or datetime.now(timezone.utc).isoformat()
    manifest["review_pass_state"] = "published"
    manifest["timestamps"]["published_at"] = publication_time
    manifest["timestamps"]["updated_at"] = publication_time
    for region in manifest["regions"]:
        region["review_status"] = "published"
    _write_json(manifest_path, manifest)
    payload = {"truth_set_id": manifest["truth_set_id"], "published_at": publication_time, "manifest_sha256": sha256_payload(manifest), "predecessor_truth_set_id": manifest.get("predecessor_truth_set_id")}
    _write_json(marker, payload)
    return payload


def audit_truth_set_root(root: str | Path) -> dict[str, Any]:
    """Read-only A0 audit. It never changes package state."""
    package = Path(root).expanduser().resolve()
    blockers: list[str] = []
    try:
        manifest = TruthSetManifest.from_dict(json.loads((package / "truth_set_manifest.json").read_text(encoding="utf-8"))).to_dict()
    except Exception as exc:
        return {"schema_version": 1, "decision": "stop", "blockers": [f"invalid manifest: {exc}"]}
    required = ["resolved_manifest.json", "input_fingerprints.json", "candidate_panel/blinded_candidates.json", "private/candidate_source_key.json", "freeze/frozen_lane_manifest.json", "freeze/protected_lock.json", "annotations/revision_references.json"]
    for relative in required:
        if not (package / relative).is_file():
            blockers.append(f"missing artifact: {relative}")
    try:
        public = json.loads((package / "review" / "review_payload.json").read_text(encoding="utf-8"))
        assert_blinded_payload(public)
    except Exception as exc:
        blockers.append(f"blinding audit failed: {exc}")
    candidates = json.loads((package / "candidate_panel" / "blinded_candidates.json").read_text(encoding="utf-8")).get("candidates", []) if (package / "candidate_panel" / "blinded_candidates.json").is_file() else []
    disposition_path = package / "candidate_panel" / "candidate_dispositions.tsv"
    disposition_rows: list[dict[str, Any]] = []
    if disposition_path.is_file():
        with disposition_path.open("r", encoding="utf-8", newline="") as handle:
            disposition_rows = list(csv.DictReader(handle, delimiter="\t"))
    candidate_ids = {item["candidate_id"] for item in candidates}
    disposed_ids = {item.get("candidate_id", "") for item in disposition_rows if item.get("disposition") in {"neuron", "artifact", "background", "unresolved"}}
    if disposed_ids != candidate_ids:
        blockers.append("candidate disposition table does not cover the blinded union exactly")
    unresolved_ids = {item["candidate_id"] for item in disposition_rows if item.get("disposition") == "unresolved"}
    if unresolved_ids and not (package / "annotations" / "unresolved.tsv").is_file():
        blockers.append("unresolved candidates are not retained explicitly")
    for region in manifest["regions"]:
        if not region["coverage_mode"]:
            blockers.append(f"region {region['region_id']} lacks explicit coverage")
        if not region["raw_first_complete"]:
            blockers.append(f"region {region['region_id']} raw-first review is not locked")
        if region.get("candidate_union_disposition_count", 0) < region.get("candidate_union_count", 0):
            blockers.append(f"region {region['region_id']} has undispositioned candidates")
        if region["review_status"] not in {"adjudicated", "published"}:
            blockers.append(f"region {region['region_id']} second review/adjudication is incomplete")
    lock_path = package / "freeze" / "protected_lock.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        frozen_path = package / "freeze" / "frozen_lane_manifest.json"
        if not frozen_path.is_file() or sha256_payload(json.loads(frozen_path.read_text(encoding="utf-8"))) != lock.get("frozen_lane_manifest_sha256"):
            blockers.append("frozen lane manifest checksum changed after protected lock")
        if lock.get("state") != "unsealed":
            blockers.append("protected region is not unsealed")
        if lock.get("unsealed_at") and not lock.get("locked_at"):
            blockers.append("protected freeze does not predate unsealing")
        if lock.get("unsealed_at") and lock.get("locked_at") and lock["unsealed_at"] < lock["locked_at"]:
            blockers.append("protected unseal timestamp predates the freeze")
    if manifest["coverage_status"] != "complete":
        blockers.append("coverage status is incomplete")
    revision_path = package / "annotations" / "revision_references.json"
    if revision_path.is_file():
        references = json.loads(revision_path.read_text(encoding="utf-8"))
        base = package / "annotations" / references["base_revision_path"]
        if not base.is_file() or _sha256_file(base) != references["base_revision_sha256"]:
            blockers.append("base annotation revision checksum is invalid")
        for revision in references["published_revisions"]:
            item = package / "annotations" / revision["path"]
            valid = item.is_file() and _sha256_file(item) == revision["sha256"]
            if valid:
                payload = json.loads(item.read_text(encoding="utf-8"))
                valid = payload.get("revisionId") == revision["revision_id"] and payload.get("state") == "published"
            if not valid:
                blockers.append(f"published annotation revision is invalid: {revision['revision_id']}")
    decision = "advance" if not blockers else "incomplete"
    return {"schema_version": 1, "truth_set_id": manifest["truth_set_id"], "decision": decision, "candidate_count": len(candidates), "blockers": blockers, "meaning": "A0 advance authorizes protected evaluation only; it is not a detector pass."}
