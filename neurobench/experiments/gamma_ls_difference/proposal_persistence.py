"""Causal, deterministic association of an unchanged frame-proposal ledger.

Immutable anchors define algorithmic locations, not biological identities.
Existing-anchor edges are greedily consumed by (squared distance, creation
index, proposal ID). This is not globally optimal tracking. All unmatched rows
create new anchors after existing-anchor matching; no input proposal is removed.
The association does not use the episode gap, so gap comparisons only change
segmentation. Every integer frame in the declared inclusive window is assumed
to have been scored; an unavailable frame must instead delimit a separate call.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite numeric data") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    result = _number(value, name)
    if not result.is_integer() or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def analyze(
    rows: Iterable[Mapping[str, Any]],
    *,
    first_ui: int,
    last_ui: int,
    radius_px: float,
    max_missing_frames: int,
    fps: float = 50,
    pixel_um: float = 0.5,
    offset_xy: tuple[float, float] = (0, 0),
) -> dict[str, Any]:
    """Return JSON-compatible ``sites/episodes/memberships/summary`` tables.

    Required row keys: proposal_id, source_frame_ui, x_px, y_px, score.
    candidate_rank_within_frame is optional. Finite numeric CSV strings are
    accepted; IDs must be unique nonempty strings. Coordinates are crop-local
    x=column/y=row; source coordinates add offset_xy, in pixels. Radius is
    inclusive Euclidean distance from the unchanged first-proposal anchor.

    A gap of g allows at most g scored frames without an assigned proposal;
    two assigned frames can therefore be at most g+1 frames apart. Episode
    confirmation occurs on its third observation; no earlier membership is
    relabeled using future qualification. All final episode summaries are
    descriptive and naturally depend on the observed window end.
    """
    first_ui = _integer(first_ui, "first_ui", 1)
    last_ui = _integer(last_ui, "last_ui", first_ui)
    max_missing_frames = _integer(max_missing_frames, "max_missing_frames", 0)
    radius_px = _number(radius_px, "radius_px")
    fps = _number(fps, "fps")
    pixel_um = _number(pixel_um, "pixel_um")
    if min(radius_px, fps, pixel_um) <= 0:
        raise ValueError("radius_px, fps and pixel_um must be positive")
    radius_squared = radius_px * radius_px
    if not math.isfinite(radius_squared) or radius_squared == 0:
        raise ValueError("radius_px must have a finite, nonzero squared radius")
    if len(offset_xy) != 2:
        raise ValueError("offset_xy must contain x and y pixel offsets")
    ox, oy = (_number(value, "offset_xy") for value in offset_xy)

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for original in rows:
        if not isinstance(original, Mapping):
            raise ValueError("each proposal must be a mapping")
        required = {"proposal_id", "source_frame_ui", "x_px", "y_px", "score"}
        if not required.issubset(original):
            raise ValueError(f"proposal is missing keys: {sorted(required - original.keys())}")
        proposal_id = original["proposal_id"]
        if not isinstance(proposal_id, str) or not proposal_id.strip():
            raise ValueError("proposal_id must be a nonempty string")
        if proposal_id in seen:
            raise ValueError(f"duplicate proposal_id: {proposal_id}")
        seen.add(proposal_id)
        frame = _integer(original["source_frame_ui"], "source_frame_ui", 1)
        if not first_ui <= frame <= last_ui:
            raise ValueError(f"proposal {proposal_id} falls outside the declared window")
        x = _number(original["x_px"], "x_px")
        y = _number(original["y_px"], "y_px")
        source_x, source_y = x + ox, y + oy
        if not all(math.isfinite(v) for v in (source_x, source_y, x / radius_px, y / radius_px)):
            raise ValueError("coordinate mapping or spatial-index coordinate overflow")
        rank = original.get("candidate_rank_within_frame")
        if rank is not None and rank != "":
            rank = _integer(rank, "candidate_rank_within_frame", 1)
        else:
            rank = None
        by_frame[frame].append({
            "proposal_id": proposal_id,
            "source_frame_ui": frame,
            "x_px": x,
            "y_px": y,
            "source_x_px": source_x,
            "source_y_px": source_y,
            "score": _number(original["score"], "score"),
            "candidate_rank_within_frame": rank,
        })

    sites: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    current_episode: dict[int, int] = {}
    consecutive_run: dict[int, int] = {}

    for frame in sorted(by_frame):
        proposals = sorted(by_frame[frame], key=lambda row: (
            -row["score"], row["y_px"], row["x_px"], row["proposal_id"]))
        edges: list[tuple[float, int, str]] = []
        eligible: dict[str, int] = {}
        for row in proposals:
            pid = row["proposal_id"]
            eligible[pid] = 0
            bx = math.floor(row["x_px"] / radius_px)
            by = math.floor(row["y_px"] / radius_px)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for site_index in grid.get((bx + dx, by + dy), ()):
                        site = sites[site_index]
                        delta_x = row["x_px"] - site["anchor_x_px"]
                        delta_y = row["y_px"] - site["anchor_y_px"]
                        d2 = delta_x * delta_x + delta_y * delta_y
                        if d2 <= radius_squared:
                            eligible[pid] += 1
                            edges.append((d2, site_index, pid))
        assignments: dict[str, tuple[int, float]] = {}
        used_sites: set[int] = set()
        for d2, site_index, pid in sorted(edges):
            if pid not in assignments and site_index not in used_sites:
                assignments[pid] = (site_index, d2)
                used_sites.add(site_index)

        for row in proposals:
            pid = row["proposal_id"]
            created = pid not in assignments
            if created:
                site_index = len(sites)
                site = {
                    "site_id": f"site_{site_index + 1:06d}",
                    "creation_index": site_index + 1,
                    "anchor_proposal_id": pid,
                    "anchor_frame_ui": frame,
                    "anchor_x_px": row["x_px"],
                    "anchor_y_px": row["y_px"],
                    "anchor_source_x_px": row["source_x_px"],
                    "anchor_source_y_px": row["source_y_px"],
                    "first_ui": frame,
                    "last_ui": frame,
                    "observed_frames": 0,
                    "episode_ids": [],
                    "inter_episode_missing_frames": [],
                    "maximum_anchor_distance_px": 0.0,
                }
                sites.append(site)
                grid[(math.floor(row["x_px"] / radius_px),
                      math.floor(row["y_px"] / radius_px))].append(site_index)
                d2 = 0.0
            else:
                site_index, d2 = assignments[pid]
                site = sites[site_index]

            episode_index = current_episode.get(site_index)
            if episode_index is None or frame - episodes[episode_index]["last_ui"] > max_missing_frames + 1:
                if episode_index is not None:
                    site["inter_episode_missing_frames"].append(frame - episodes[episode_index]["last_ui"] - 1)
                episode_index = len(episodes)
                episode = {
                    "episode_id": f"episode_{episode_index + 1:06d}",
                    "site_id": site["site_id"],
                    "episode_index_within_site": len(site["episode_ids"]) + 1,
                    "anchor_x_px": site["anchor_x_px"],
                    "anchor_y_px": site["anchor_y_px"],
                    "anchor_source_x_px": site["anchor_source_x_px"],
                    "anchor_source_y_px": site["anchor_source_y_px"],
                    "first_ui": frame,
                    "last_ui": frame,
                    "observed_frames": 0,
                    "max_consecutive_run": 0,
                    "max_missing_frames_between_observations": 0,
                    "confirmed_at_ui": None,
                    "proposal_ids": [],
                    "source_frames_ui": [],
                }
                episodes.append(episode)
                current_episode[site_index] = episode_index
                site["episode_ids"].append(episode["episode_id"])
                consecutive_run[episode_index] = 0
            episode = episodes[episode_index]
            if episode["observed_frames"]:
                missing = frame - episode["last_ui"] - 1
                episode["max_missing_frames_between_observations"] = max(
                    episode["max_missing_frames_between_observations"], missing)
                consecutive_run[episode_index] = consecutive_run[episode_index] + 1 if missing == 0 else 1
            else:
                consecutive_run[episode_index] = 1
            episode["observed_frames"] += 1
            episode["last_ui"] = frame
            episode["max_consecutive_run"] = max(episode["max_consecutive_run"], consecutive_run[episode_index])
            episode["proposal_ids"].append(pid)
            episode["source_frames_ui"].append(frame)
            if episode["observed_frames"] == 3:
                episode["confirmed_at_ui"] = frame
            site["observed_frames"] += 1
            site["last_ui"] = frame
            site["maximum_anchor_distance_px"] = max(site["maximum_anchor_distance_px"], math.sqrt(d2))
            memberships.append({
                **row,
                "site_id": site["site_id"],
                "episode_id": episode["episode_id"],
                "distance_to_anchor_px": math.sqrt(d2),
                "distance_to_anchor_um": math.sqrt(d2) * pixel_um,
                "eligible_existing_anchor_count": eligible[pid],
                "created_new_site": created,
                "created_despite_eligible_anchor_collision": created and eligible[pid] > 0,
                "episode_observations_so_far": episode["observed_frames"],
                "episode_confirmed_so_far": episode["observed_frames"] >= 3,
            })

    for episode in episodes:
        span = episode["last_ui"] - episode["first_ui"] + 1
        episode.update({
            "span_frames": span,
            "elapsed_ms": (span - 1) * 1000 / fps,
            "span_ms": span * 1000 / fps,
            "occupancy": episode["observed_frames"] / span,
            "missing_frames_within_span": span - episode["observed_frames"],
            "left_censored": episode["first_ui"] <= first_ui + max_missing_frames,
            "right_censored": episode["last_ui"] + max_missing_frames + 1 > last_ui,
            "closed_at_ui": (episode["last_ui"] + max_missing_frames + 1
                             if episode["last_ui"] + max_missing_frames + 1 <= last_ui else None),
            "qualifies_3_observations": episode["observed_frames"] >= 3,
            "qualifies_5_observations": episode["observed_frames"] >= 5,
            "qualifies_10_observations": episode["observed_frames"] >= 10,
            "confirmation_elapsed_ms": ((episode["confirmed_at_ui"] - episode["first_ui"]) * 1000 / fps
                                        if episode["confirmed_at_ui"] is not None else None),
        })
    for site in sites:
        site["episode_count"] = len(site["episode_ids"])
        site["recurrence_count"] = max(0, site["episode_count"] - 1)
    persistence_counts = {}
    for n in (3, 5, 10):
        qualifying = [episode for episode in episodes if episode["observed_frames"] >= n]
        persistence_counts[f"persistent_episode_count_{n}"] = len(qualifying)
        persistence_counts[f"persistent_site_count_{n}"] = len({episode["site_id"] for episode in qualifying})
        persistence_counts[f"persistent_proposal_count_{n}"] = sum(episode["observed_frames"] for episode in qualifying)
    if (len(memberships) != len(seen)
            or sum(episode["observed_frames"] for episode in episodes) != len(seen)
            or sum(site["observed_frames"] for site in sites) != len(seen)):
        raise RuntimeError("internal proposal conservation failure")
    return {
        "sites": sites,
        "episodes": episodes,
        "memberships": memberships,
        "summary": {
            "schema_version": 1,
            "first_ui": first_ui,
            "last_ui": last_ui,
            "window_frames": last_ui - first_ui + 1,
            "window_seconds": (last_ui - first_ui + 1) / fps,
            "radius_px": radius_px,
            "radius_um": radius_px * pixel_um,
            "max_missing_frames": max_missing_frames,
            "fps": fps,
            "pixel_um": pixel_um,
            "offset_xy": [ox, oy],
            "proposal_count": len(memberships),
            "frames_with_proposals": len(by_frame),
            "site_count": len(sites),
            "episode_count": len(episodes),
            "recurrent_site_count": sum(site["recurrence_count"] > 0 for site in sites),
            **persistence_counts,
            "episodes_with_at_least_observations": {
                str(n): sum(episode["observed_frames"] >= n for episode in episodes) for n in (3, 5, 10)},
            "proposals_in_episodes_with_at_least_observations": {
                str(n): sum(episode["observed_frames"] for episode in episodes if episode["observed_frames"] >= n)
                for n in (3, 5, 10)},
            "ambiguous_proposal_count": sum(row["eligible_existing_anchor_count"] > 1 for row in memberships),
            "new_sites_due_to_anchor_collision": sum(row["created_despite_eligible_anchor_collision"] for row in memberships),
            "association": "causal greedy edges ordered by distance_squared, site_creation_index, proposal_id; immutable anchors; one proposal/site/frame",
            "creation_order": "descending score, y_px, x_px, proposal_id; prior-frame anchors only",
            "episode_definition": "same-anchor assigned observations with no more than max_missing_frames intervening scored frames",
            "recurrence_definition": "a later separate episode assigned to the same unchanged algorithmic anchor",
            "duration_definition": "span_frames is inclusive; elapsed_ms is last minus first sample time; occupancy is observations/span_frames",
            "confirmation_definition": "third observed frame, not third elapsed frame; neither a biological label nor controller trigger",
            "persistence_count_definition": "retrospective counts of episodes with at least N observations, distinct sites having such episodes, and all proposals within those episodes",
            "frame_inventory_assumption": "every integer frame in the inclusive window was scored; unavailable intervals require separate calls",
            "biological_identity_established": False,
            "global_assignment_optimality_claimed": False,
            "independent_frame_inference_claimed": False,
        },
    }
