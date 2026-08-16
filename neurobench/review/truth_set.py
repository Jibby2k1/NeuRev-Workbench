"""Deterministic, detector-blinded truth-set review helpers."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def deterministic_region_from_mask(
    tissue_mask: Sequence[Sequence[bool | int]],
    *,
    width_px: int,
    height_px: int,
    seed: int,
    excluded_bounds: Sequence[Mapping[str, int]] = (),
) -> dict[str, int]:
    """Choose a tile from a label-free tissue mask using only mask occupancy and a seed."""
    if width_px < 1 or height_px < 1 or not tissue_mask or not tissue_mask[0]:
        raise ValueError("region dimensions and tissue mask must be non-empty")
    image_height, image_width = len(tissue_mask), len(tissue_mask[0])
    if any(len(row) != image_width for row in tissue_mask):
        raise ValueError("tissue mask rows must have equal width")
    if width_px > image_width or height_px > image_height:
        raise ValueError("region dimensions exceed tissue mask")
    candidates = []
    for y in range(image_height - height_px + 1):
        for x in range(image_width - width_px + 1):
            bounds = {"x_min": x, "y_min": y, "x_max_exclusive": x + width_px, "y_max_exclusive": y + height_px}
            if any(not (bounds["x_max_exclusive"] <= other["x_min"] or other["x_max_exclusive"] <= bounds["x_min"] or bounds["y_max_exclusive"] <= other["y_min"] or other["y_max_exclusive"] <= bounds["y_min"]) for other in excluded_bounds):
                continue
            occupancy = sum(bool(tissue_mask[py][px]) for py in range(y, y + height_px) for px in range(x, x + width_px))
            if occupancy == width_px * height_px:
                candidates.append(bounds)
    if not candidates:
        raise ValueError("no fully covered label-free tile satisfies the constraints")
    return candidates[random.Random(int(seed)).randrange(len(candidates))]


def _candidate_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(item["region_id"]), int(item["ui_frame_interval"][0]), int(item["ui_frame_interval"][1]),
        float(item["y_px"]), float(item["x_px"]), str(item["source_lane_id"]), str(item["source_candidate_id"]),
    )


def build_candidate_union(
    candidates: Sequence[Mapping[str, Any]],
    *,
    spatial_radius_px: float,
    temporal_radius_frames: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Deduplicate frozen candidates and return blinded rows plus a private key."""
    if spatial_radius_px < 0 or temporal_radius_frames < 0:
        raise ValueError("candidate reconciliation radii must be non-negative")
    groups: list[list[dict[str, Any]]] = []
    for raw in sorted((dict(item) for item in candidates), key=_candidate_sort_key):
        match = None
        center_frame = (int(raw["ui_frame_interval"][0]) + int(raw["ui_frame_interval"][1])) / 2.0
        for group in groups:
            anchor = group[0]
            anchor_frame = (int(anchor["ui_frame_interval"][0]) + int(anchor["ui_frame_interval"][1])) / 2.0
            if raw["region_id"] != anchor["region_id"]:
                continue
            if math.hypot(float(raw["x_px"]) - float(anchor["x_px"]), float(raw["y_px"]) - float(anchor["y_px"])) <= spatial_radius_px and abs(center_frame - anchor_frame) <= temporal_radius_frames:
                match = group
                break
        if match is None:
            groups.append([raw])
        else:
            match.append(raw)

    blinded: list[dict[str, Any]] = []
    private: dict[str, dict[str, Any]] = {}
    for group in groups:
        stable = sha256_payload([_candidate_sort_key(item) for item in group])[:16]
        opaque_id = f"candidate_{stable}"
        xs = [float(item["x_px"]) for item in group]
        ys = [float(item["y_px"]) for item in group]
        starts = [int(item["ui_frame_interval"][0]) for item in group]
        ends = [int(item["ui_frame_interval"][1]) for item in group]
        blinded.append({
            "candidate_id": opaque_id,
            "region_id": group[0]["region_id"],
            "x_px": sum(xs) / len(xs),
            "y_px": sum(ys) / len(ys),
            "ui_frame_interval": [min(starts), max(ends)],
            "disposition": None,
        })
        private[opaque_id] = {
            "sources": [
                {"lane_id": item["source_lane_id"], "candidate_id": item["source_candidate_id"], "score": item["score"]}
                for item in sorted(group, key=_candidate_sort_key)
            ]
        }
    random.Random(int(random_seed)).shuffle(blinded)
    for index, item in enumerate(blinded, 1):
        item["presentation_index"] = index
    return blinded, private


def deterministic_second_review_sample(
    dispositions: Sequence[Mapping[str, Any]], *, seed: int, fraction: float = 0.20
) -> list[str]:
    """Sample accepted/rejected independently and always include unresolved/disagreements."""
    if not 0.20 <= fraction <= 1.0:
        raise ValueError("second-review fraction must be at least 20%")
    forced = {
        str(item["subject_id"])
        for item in dispositions
        if item.get("disposition") == "unresolved" or bool(item.get("disagreement"))
    }
    rng = random.Random(int(seed))
    selected = set(forced)
    by_disposition: dict[str, list[str]] = defaultdict(list)
    for item in dispositions:
        if item.get("disposition") in {"accepted", "rejected", "neuron", "artifact", "background", "event"}:
            by_disposition[str(item["disposition"])].append(str(item["subject_id"]))
    for values in by_disposition.values():
        values = sorted(set(values))
        rng.shuffle(values)
        count = math.ceil(len(values) * fraction) if values else 0
        selected.update(values[:count])
    return sorted(selected)


def assert_blinded_payload(payload: Any) -> None:
    """Reject detector identity, scores, ranks, or source filenames anywhere in public JSON."""
    forbidden_keys = {"source_lane_id", "lane_id", "score", "rank", "source_candidate_id", "candidate_source_key"}
    forbidden_fragments = {"raw_direct", "fullrank_ica", "exponential_w5", "coherence_w15", "candidate_source_key.json"}

    def walk(value: Any, path: str = "root") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in forbidden_keys:
                    raise ValueError(f"blinding leak at {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            normalized = value.lower()
            if any(fragment in normalized for fragment in forbidden_fragments):
                raise ValueError(f"blinding leak at {path}")

    walk(payload)
