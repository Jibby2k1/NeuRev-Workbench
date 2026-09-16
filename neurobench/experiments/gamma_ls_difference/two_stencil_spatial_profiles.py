"""Bounded spatial profiles and descriptive distributions from fixed snapshots.

The same frozen 14 occurrences and nearest deployed-anchor burst sites define
locations shared by every arm. No full-stage array is opened, and no NMS,
calibration, matching, classification, or operator fit is rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import shutil

import numpy as np

from . import two_stencil_focused_diagnostics as focused


STAGES = ("Raw", "X", "A", "M", "sigma", "contrast", "Z")
PROFILE_FRAME_BY_BURST = {1: 2014, 2: 2051, 3: 2135, 4: 2277}
PROFILE_RADIUS_PX = 24
QUANTILES = (0, .01, .05, .25, .5, .75, .95, .99, 1)
SNAPSHOT_GROUPS = {
    "calibration_time_snapshots": (50, 100),
    "configured_burst_window_snapshots": (2003, 2014, 2026, 2051, 2135, 2277),
    "unannotated_time_snapshots": (1840, 2359),
}
ANCHOR_ID = "deployed_signed_point_gamma_g7_disk"


def snapshot_frame_index(source_frames_ui, source_frame_ui):
    """Resolve the original UI frame through stored metadata, never position."""
    frames = tuple(int(v) for v in source_frames_ui)
    if (not frames or len(set(frames)) != len(frames)
            or any(int(v) != v or v < 1 for v in source_frames_ui)):
        raise ValueError("snapshot UI frames must be distinct positive integers")
    target = int(source_frame_ui)
    if target != source_frame_ui or target < 1 or target not in frames:
        raise ValueError("requested source UI frame is absent from snapshot metadata")
    return frames.index(target)


def extract_profiles(frame, *, x_px, y_px, radius_px=PROFILE_RADIUS_PX):
    """Extract two integer-pixel lines, preserving unavailable edge offsets."""
    values = np.asarray(frame)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("a nonempty YX image is required")
    x, y = float(x_px), float(y_px)
    if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < values.shape[1] and 0 <= y < values.shape[0]):
        raise ValueError("profile center is outside the source image")
    if isinstance(radius_px, bool) or int(radius_px) != radius_px or radius_px < 1:
        raise ValueError("profile radius must be a positive integer")
    px, py = min(values.shape[1]-1, math.floor(x+.5)), min(values.shape[0]-1, math.floor(y+.5))
    offsets = np.arange(-int(radius_px), int(radius_px)+1, dtype=np.int32)
    answer = {}
    for orientation in ("horizontal", "vertical"):
        xx = px+offsets if orientation == "horizontal" else np.full_like(offsets, px)
        yy = py+offsets if orientation == "vertical" else np.full_like(offsets, py)
        valid = (xx >= 0) & (xx < values.shape[1]) & (yy >= 0) & (yy < values.shape[0])
        samples = np.full(len(offsets), np.nan, dtype=np.float64)
        samples[valid] = values[yy[valid], xx[valid]]
        if not np.isfinite(samples[valid]).all():
            raise ValueError("profile includes nonfinite source pixels")
        answer[orientation] = {"offset_px": offsets.copy(), "x_px": xx, "y_px": yy,
                               "valid": valid, "values": samples}
    return {"requested_x_px": x, "requested_y_px": y, "trace_x_px": px, "trace_y_px": py}, answer


def _core_cells(protocol):
    cells = [c for c in protocol["cells"] if c["is_deployed_anchor"] or (
        c["spec"]["reference_family"] == "gamma" and c["spec"]["guard_radius_px"] == 0
        and c["spec"]["support_geometry"] == "square")]
    if len(cells) != 7:
        raise ValueError("expected the six core Gamma0 arms and one separate disk anchor")
    return cells


def _locations(selection, labels, sites):
    by_id = {r["observation_id"]: r for r in labels}
    locations = []
    for selected in selection:
        label = by_id[selected["observation_id"]]
        burst = int(label["burst_id"])
        frame = PROFILE_FRAME_BY_BURST[burst]
        if not int(label["source_start_ui"]) <= frame <= int(label["source_stop_ui"]):
            raise ValueError("fixed profile frame must lie inside the configured burst window")
        x, y = float(label["x_px"]), float(label["y_px"])
        common = {"observation_id": label["observation_id"], "canonical_roi_id": label["canonical_roi_id"],
                  "burst_id": burst, "source_frame_ui": frame,
                  "source_start_ui": int(label["source_start_ui"]), "source_stop_ui": int(label["source_stop_ui"]),
                  "expert_original_x_px": x, "expert_original_y_px": y}
        locations.append({**common, "location_kind": "expert", "available": True,
                          "x_px": x, "y_px": y, "anchor_site_id": None,
                          "anchor_representative_proposal_id": None, "distance_from_expert_px": 0.0})
        candidates = [r for r in sites if int(r["burst_id"]) == burst]
        nearest = min(candidates, key=lambda r: (math.hypot(float(r["x_px"])-x, float(r["y_px"])-y),
                      int(r["site_rank"]), r["site_id"]), default=None)
        locations.append({**common, "location_kind": "nearest_anchor_burst_site", "available": nearest is not None,
            "x_px": None if nearest is None else float(nearest["x_px"]),
            "y_px": None if nearest is None else float(nearest["y_px"]),
            "anchor_site_id": None if nearest is None else nearest["site_id"],
            "anchor_representative_proposal_id": None if nearest is None else nearest["representative_proposal_id"],
            "anchor_representative_source_frame_ui": None if nearest is None else int(nearest["source_frame_ui"]),
            "distance_from_expert_px": None if nearest is None else math.hypot(float(nearest["x_px"])-x, float(nearest["y_px"])-y),
            "location_is_one_to_one_assignment": False,
            "location_rule": "nearest frozen q1 deployed-anchor site within same burst, no distance cutoff; ties site_rank then site_id"})
    return locations


def _profile_limits(profiles, cells):
    limits = {}
    for representation in ("conditioned_current_frame", "difference_signed"):
        ids = {c["cell_id"] for c in cells if c["input_representation"] == representation}
        limits[representation] = {}
        for stage in STAGES:
            samples = [packet[stage][packet["valid"]] for key, packet in profiles.items() if key[0] in ids]
            lo, hi = min(float(v.min()) for v in samples), max(float(v.max()) for v in samples)
            if representation == "difference_signed" and stage not in ("Raw", "sigma"):
                extent = max(abs(lo), abs(hi), 1e-12)*1.05
                bounds = [-extent, extent]
            else:
                padding = max(hi-lo, abs(hi)*.01, 1e-12)*.05
                bounds = [min(0.0, lo) if stage in ("Raw", "sigma") else lo-padding, hi+padding]
            limits[representation][stage] = bounds
    return limits


def _render(destination, locations, cells, profiles, limits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    style = {"point": ("#4169a6", "-"), "direct": ("#a45b28", "--"),
             "serial": ("#80529b", ":"), "disk anchor": ("#303030", "-.")}
    for location in locations:
        name = f"{location['observation_id']}__{location['location_kind']}"
        if not location["available"]:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axis("off")
            ax.text(.5, .5, "Not applicable: no frozen q1 deployed-anchor burst representative", ha="center")
            fig.suptitle(name)
        else:
            fig, axes = plt.subplots(4, 7, figsize=(24, 12), sharex=True)
            for row, (representation, orientation) in enumerate((
                ("conditioned_current_frame", "horizontal"), ("conditioned_current_frame", "vertical"),
                ("difference_signed", "horizontal"), ("difference_signed", "vertical"))):
                arms = [c for c in cells if c["input_representation"] == representation]
                for column, stage in enumerate(STAGES):
                    axis = axes[row, column]
                    for cell in arms[:1] if stage in ("Raw", "X") else arms:
                        key = (cell["cell_id"], location["observation_id"], location["location_kind"], orientation)
                        packet = profiles[key]
                        label = "shared input" if stage in ("Raw", "X") else ("disk anchor" if cell["is_deployed_anchor"] else cell["spec"]["design"])
                        color, linestyle = ("#303030", "-") if label == "shared input" else style[label]
                        axis.plot(packet["offset_px"], packet[stage], color=color, ls=linestyle, lw=1.2, label=label)
                    axis.axvline(0, color=".75", lw=.7)
                    axis.set_ylim(limits[representation][stage])
                    axis.set_xlim(-PROFILE_RADIUS_PX, PROFILE_RADIUS_PX)
                    axis.grid(alpha=.15)
                    axis.tick_params(labelsize=7)
                    if row == 0:
                        axis.set_title(stage)
                    if column == 0:
                        axis.set_ylabel(("Current" if row < 2 else "Signed") + " / " + orientation)
                    if row == 3:
                        axis.set_xlabel("pixel offset from rounded center", fontsize=8)
                axes[row, 2].legend(fontsize=7, loc="best")
            candidate = "expert center" if location["location_kind"] == "expert" else f"nearest anchor site {location['anchor_site_id']}, distance {location['distance_from_expert_px']:.2f}px"
            fig.suptitle(f"{location['observation_id']} | UI {location['source_frame_ui']} | {candidate}\n"
                f"Original center ({location['x_px']:.3f}, {location['y_px']:.3f}); sampled pixel ({location['trace_x_px']}, {location['trace_y_px']}). "
                "Shared stage scales within each input across all 14 cards and both location types.", fontsize=11)
            fig.text(.5, .015, "Current = conditioned current frame; Signed = adjacent difference. Raw and X are shared inputs within each row. Candidate site need not be a proposal at this snapshot frame.", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .045, 1, .94))
        path = destination / "figures" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=125)
        plt.close(fig)
        files.append(str(path.relative_to(destination)))
    return files


def run_spatial_profile_diagnostics(campaign_root, *, render=True):
    root = Path(campaign_root).resolve()
    destination = root / "spatial_profile_diagnostics"
    if destination.exists():
        raise FileExistsError(f"new spatial supplement output directory required: {destination}")
    protocol, preflight, _ = focused._campaign_gate(root)
    frames = tuple(int(v) for v in protocol["snapshot_source_ui"])
    if set(frames) != {v for group in SNAPSHOT_GROUPS.values() for v in group}:
        raise ValueError("the fixed snapshot groups must partition the declared source frames")
    label_path = Path(preflight["label_source"])
    focused._verify_file(label_path, preflight["label_sha256"])
    labels = focused._read_tsv(label_path)
    original = root / "focused_diagnostics"
    original_summary = focused._json(original/"summary.json")
    original_manifest = {r["path"]: r["sha256"] for r in original_summary["artifacts"]}
    focused._verify_file(original/"provenance.json", original_manifest["provenance.json"])
    original_provenance = focused._json(original/"provenance.json")
    focused._verify_file(root/"protocol.json", original_provenance["campaign_protocol_sha256"])
    focused._verify_file(original/"card_selection.tsv", focused.FROZEN_SELECTION_SHA256)
    selection = focused._read_tsv(original/"card_selection.tsv")
    cells = _core_cells(protocol)
    anchor_folder = root/"cells"/ANCHOR_ID
    sites_path = anchor_folder/"readouts/Z/q1/site_rows.tsv"
    site_relative = str(sites_path.relative_to(root))
    site_sha = original_provenance["verified_cells"][ANCHOR_ID][site_relative]
    focused._verify_file(sites_path, site_sha)
    sites = focused._read_tsv(sites_path)
    locations = _locations(selection, labels, sites)
    destination.mkdir()
    provenance = {"campaign_protocol_sha256": focused._sha256(root/"protocol.json"),
        "campaign_candidate_seal_sha256": focused._sha256(root/"campaign_candidate_seal.json"),
        "original_focused_summary_sha256": focused._sha256(original/"summary.json"),
        "original_focused_provenance_sha256": focused._sha256(original/"provenance.json"),
        "label_source": str(label_path), "label_sha256": preflight["label_sha256"],
        "selection_sha256": focused.FROZEN_SELECTION_SHA256, "anchor_site_path": str(sites_path),
        "anchor_site_sha256": site_sha, "snapshot_sources": {}, "raw_input_sources": {},
        "code_sha256": {Path(__file__).name: focused._sha256(Path(__file__)),
                        Path(focused.__file__).name: focused._sha256(Path(focused.__file__))}}
    for path in (Path(__file__), Path(focused.__file__)):
        shutil.copy2(path, destination/path.name)
    # Bind Raw/X bytes with one serial hash per distinct source file, then
    # read only the ten declared frames. Full derived stage arrays stay closed.
    inputs = {}
    verified_input_files = {}
    preflight_sources = {r["stage"]: r for r in preflight["sources"]}
    for name, item in {"Raw": protocol["source_movie"], **protocol["inputs"]}.items():
        if preflight_sources[name]["sha256"] != item["file_sha256"]:
            raise ValueError("protocol and preflight source hashes disagree")
        source_path = Path(item["path"]).resolve()
        if source_path not in verified_input_files:
            verified_input_files[source_path] = focused._verify_file(source_path, item["file_sha256"])
        elif verified_input_files[source_path] != item["file_sha256"]:
            raise ValueError("one input source file has inconsistent expected hashes")
        video = np.load(item["path"], mmap_mode="r", allow_pickle=False)
        if list(video.shape) != item["shape_tyx"] or str(video.dtype) != item["dtype"]:
            raise ValueError("Raw/input header differs from campaign source binding")
        inputs[name] = np.asarray(video[np.asarray(frames)-1]).copy()
        if not np.isfinite(inputs[name]).all():
            raise ValueError("nonfinite Raw/input snapshot values")
        provenance["raw_input_sources"][name] = {"path": item["path"],
            "verified_whole_file_sha256": verified_input_files[source_path], "whole_file_rescanned": True,
            "source_frames_ui": list(frames), "snapshot_shape": list(inputs[name].shape),
            "snapshot_dtype": str(inputs[name].dtype),
            "snapshot_c_order_value_sha256": hashlib.sha256(inputs[name].tobytes(order="C")).hexdigest()}
        del video
    input_packet = destination/"input_snapshots.npz"
    np.savez_compressed(input_packet, source_frames_ui=np.asarray(frames), **inputs)
    provenance["input_snapshot_artifact_sha256"] = focused._sha256(input_packet)
    profiles, distribution_rows, profile_rows, distribution_cache = {}, [], [], {}
    for cell in cells:
        cell_id, representation = cell["cell_id"], cell["input_representation"]
        folder = root/"cells"/cell_id
        complete = focused._json(folder/"numeric_complete.json")
        path = folder/"stage_snapshots.npz"
        snapshot_sha = focused._verify_file(path, complete["stage_snapshots_sha256"])
        provenance["snapshot_sources"][cell_id] = {"path": str(path), "sha256": snapshot_sha,
            "numeric_complete_sha256": focused._sha256(folder/"numeric_complete.json")}
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive["source_frames_ui"]) != frames:
                raise ValueError("saved snapshot source frames differ from protocol")
            arrays = {"Raw": inputs["Raw"], "X": inputs[representation], **{s: archive[s] for s in STAGES[2:]}}
        if any(a.shape != arrays["Raw"].shape or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("stage snapshot shape/finite-value mismatch")
        for stage, array in arrays.items():
            for group, group_frames in SNAPSHOT_GROUPS.items():
                cache_key = (("shared_Raw" if stage == "Raw" else representation) if stage in ("Raw", "X") else cell_id, stage, group)
                if cache_key not in distribution_cache:
                    values = array[[snapshot_frame_index(frames, f) for f in group_frames]].reshape(-1)
                    quantiles = np.quantile(values, QUANTILES, method="linear")
                    distribution_cache[cache_key] = {"pixel_frame_count": int(values.size),
                        "negative_fraction": float(np.mean(values < 0)), "zero_fraction": float(np.mean(values == 0)),
                        **{f"quantile_{q:g}": float(v) for q, v in zip(QUANTILES, quantiles)}}
                distribution_rows.append({"operator_cell_id": cell_id, "input_representation": representation,
                    "stage": stage, "snapshot_group": group, "source_frames_ui": ",".join(map(str, group_frames)),
                    "snapshot_count": len(group_frames), **distribution_cache[cache_key]})
        for location in locations:
            if not location["available"]:
                continue
            frame_index = snapshot_frame_index(frames, location["source_frame_ui"])
            stage_profiles = {}
            for stage, array in arrays.items():
                center, stage_profiles[stage] = extract_profiles(array[frame_index], x_px=location["x_px"], y_px=location["y_px"])
                location.update(center)
            for orientation in ("horizontal", "vertical"):
                geometry = stage_profiles["Raw"][orientation]
                packet = {key: value for key, value in geometry.items() if key != "values"}
                packet.update({stage: stage_profiles[stage][orientation]["values"] for stage in STAGES})
                key = (cell_id, location["observation_id"], location["location_kind"], orientation)
                profiles[key] = packet
                for offset_index in range(len(packet["offset_px"])):
                    valid = bool(packet["valid"][offset_index])
                    profile_rows.append({"operator_cell_id": cell_id, "observation_id": location["observation_id"],
                        "location_kind": location["location_kind"], "source_frame_ui": location["source_frame_ui"],
                        "orientation": orientation, "offset_px": int(packet["offset_px"][offset_index]),
                        "x_px": int(packet["x_px"][offset_index]), "y_px": int(packet["y_px"][offset_index]),
                        "valid_image_pixel": valid, **{s: float(packet[s][offset_index]) if valid else None for s in STAGES}})
        del arrays
    limits = _profile_limits(profiles, cells)
    focused._write_tsv(destination/"locations.tsv", locations)
    focused._write_tsv(destination/"profiles.tsv", profile_rows)
    focused._write_tsv(destination/"snapshot_distribution_quantiles.tsv", distribution_rows)
    focused._write_json(destination/"display_limits.json", {"limits": limits,
        "population": "all fixed profile offsets, both locations, both orientations and all14 cards; shared per stage within input"})
    definitions = {"source_frames_ui": list(frames), "profile_frame_by_burst": PROFILE_FRAME_BY_BURST,
        "profile_radius_px": PROFILE_RADIUS_PX, "snapshot_groups": SNAPSHOT_GROUPS,
        "rounding": "floor(coord+0.5), clipped to final valid pixel; offset0 denotes that rounded pixel",
        "edge_handling": "retain all49 offsets; out-of-image offsets are explicit unavailable values, never reflected or zero-filled",
        "candidate_location": "same nearest frozen q1 deployed-anchor burst representative coordinate for all arms; not necessarily emitting at chosen snapshot or assigned to expert",
        "distribution_population": "all image pixels pooled over each fixed group, including unlabeled locations; correlated pixels are not independent replicates",
        "distribution_quantile_estimator": "NumPy linear sample quantiles", "quantile_probabilities": QUANTILES,
        "time_group_semantics": "calibration-time and configured burst-window labels describe intervals only; unannotated-time snapshots are not negative or quiet truth",
        "source_hash_boundary": "Raw/X whole-file hashes are freshly verified once per distinct file; only ten declared image frames are loaded; full A/M/sigma/contrast/Z arrays remain unopened",
        "scope": "spatial support and descriptive distributions, not biological accuracy or causal attribution",
        "fixed_selection_count": 14, "fixed_arm_count": 7, "nms_refit_or_model_selection": False}
    focused._write_json(destination/"definitions.json", definitions)
    focused._write_json(destination/"provenance.json", provenance)
    figures = _render(destination, locations, cells, profiles, limits) if render else []
    (destination/"README.md").write_text(
        "# Spatial profile supplement\n\nProfiles compare all seven fixed core arms at the same 14 expert centers and the "
        "geometrically nearest frozen deployed-anchor q1 burst-site centers. The shared location need not emit at the chosen snapshot "
        "and is not necessarily the expert's one-to-one assignment. Missing sites are explicitly not applicable.\n\n"
        "Read locations.tsv for exact source UI frames and coordinates, profiles.tsv for native stage samples, and "
        "snapshot_distribution_quantiles.tsv for full-image descriptive quantiles. Each figure shows horizontal and vertical "
        "profiles separately for each input. Stage scales are shared across arms and cards within each input.\n\n"
        "Calibration-time snapshots do not establish event-free data. Configured burst windows do not provide per-neuron onset. "
        "The other fixed frames are unannotated-time snapshots, not known negatives. Pixel distributions are descriptive and "
        "do not estimate precision or a false-positive rate. No model, threshold, NMS, or candidate selection is rerun.\n")
    result = {"status": "SPATIAL_PROFILE_SUPPLEMENT_COMPLETE_VISUAL_QA_PENDING", "expert_occurrence_count": len(selection),
        "arm_count": len(cells), "location_count": len(locations),
        "unavailable_candidate_location_count": sum(not r["available"] for r in locations),
        "profile_sample_row_count": len(profile_rows), "distribution_row_count": len(distribution_rows),
        "figure_count": len(figures), "figures": figures,
        "scientific_audit_complete": False, "scientific_promotion": False,
        "artifacts": [{"path": str(p.relative_to(destination)), "sha256": focused._sha256(p)}
                      for p in sorted(destination.rglob("*")) if p.is_file()]}
    focused._write_json(destination/"summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="completed campaign root")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    print(run_spatial_profile_diagnostics(args.output, render=not args.no_render), flush=True)


if __name__ == "__main__":
    main()
