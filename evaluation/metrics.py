# evaluation/metrics.py
"""Functions for computing core performance metrics (FROC, AUC, etc.)."""
import numpy as np
from typing import Tuple, Dict, List
from collections import defaultdict
from pathlib import Path # <-- ADD THIS IMPORT

try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None

from utils import extract_pixel_detections

def calculate_froc_point_metrics(
    gt_data: Dict, detections: Dict, video_shape: Tuple, distance_tolerance: float,
    return_fp_coords: bool = False, return_matches: bool = False
) -> Dict:
    """Calculates metrics for a single point on a FROC curve."""
    if cdist is None:
        raise ImportError("scipy is required for FROC calculation.")
        
    num_frames = video_shape[0]
    all_gt_events = [(frame_idx, *pt) for frame_idx, points in gt_data.items() for pt in points]
    all_gt_events_count = len(all_gt_events)
    
    if all_gt_events_count == 0:
        total_fp = sum(len(v) for v in detections.values())
        metrics = {"TP": 0, "FP": total_fp, "FN": 0, "TPR": 0.0, "FPPI": total_fp / num_frames if num_frames > 0 else 0.0}
        if return_fp_coords: metrics["fp_detections"] = detections
        return metrics

    gt_matched_status = np.zeros(all_gt_events_count, dtype=bool)
    matched_det_in_frame = {frame_idx: set() for frame_idx in detections.keys()}

    for i, (frame_idx, x, y, gt_id) in enumerate(all_gt_events):
        if frame_idx not in detections: continue
        det_in_frame = detections[frame_idx]
        if not det_in_frame: continue
        
        det_coords = np.array([[d['x'], d['y']] for d in det_in_frame])
        distances = cdist(np.array([[x, y]]), det_coords)
        
        matched_det_indices = np.where(distances[0] <= distance_tolerance)[0]
        if len(matched_det_indices) > 0:
            gt_matched_status[i] = True
            matched_det_in_frame[frame_idx].add(matched_det_indices[0])

    tp_count = np.sum(gt_matched_status)
    all_matched = {f: matched_det_in_frame.get(f, set()) for f in detections}
    fp_count = sum(len(detections[f]) - len(all_matched[f]) for f in detections)
    
    metrics = {
        "TP": int(tp_count), "FP": int(fp_count), "FN": int(all_gt_events_count - tp_count),
        "TPR": tp_count / all_gt_events_count if all_gt_events_count > 0 else 0.0,
        "FPPI": fp_count / num_frames if num_frames > 0 else 0.0,
    }
    
    if return_fp_coords:
        fp_detections = defaultdict(list)
        for frame_idx, dets in detections.items():
            for i, det in enumerate(dets):
                if i not in all_matched.get(frame_idx, set()):
                    fp_detections[frame_idx].append(det)
        metrics["fp_detections"] = dict(fp_detections)
        
    if return_matches:
        # Existing:
        metrics["gt_match_status"] = gt_matched_status.tolist()

        # NEW: helper fields for unique-ID coverage
        gt_event_ids = [ev[3] for ev in all_gt_events]            # id for each GT event
        matched_ids = {gt_event_ids[i] for i, m in enumerate(gt_matched_status) if m}
        all_ids     = set(gt_event_ids)
        metrics["matched_unique_ids"]       = sorted(list(matched_ids))
        metrics["unique_id_coverage"]       = (len(matched_ids) / max(1, len(all_ids)))
        metrics["full_unique_id_coverage"]  = (len(matched_ids) == len(all_ids))
        
    return metrics

def calculate_truncated_auc(froc_points: List[Dict], truncation_limit: float) -> float:
    """Calculates the Area Under the FROC Curve up to a given FPPI limit."""
    if not froc_points or truncation_limit <= 0: return 0.0
    
    points = sorted([p for p in froc_points if p.get('FPPI') is not None and p['FPPI'] <= truncation_limit], key=lambda p: p['FPPI'])
    if not points: return 0.0
    
    fppi = [p['FPPI'] for p in points]
    tpr = [p['TPR'] for p in points]
    
    if not fppi or fppi[0] > 0:
        fppi.insert(0, 0.0)
        tpr.insert(0, 0.0)
    if fppi and fppi[-1] < truncation_limit:
        fppi.append(truncation_limit)
        tpr.append(tpr[-1])
        
    unique_fppi, indices = np.unique(fppi, return_index=True)
    unique_tpr = np.array(tpr)[indices]
    
    return np.trapz(unique_tpr, unique_fppi) if len(unique_fppi) >= 2 else 0.0

def calculate_detection_latency(final_mask_np: np.ndarray,
                                gt_data: Dict,
                                distance_tolerance: float,
                                output_dir: Path):
    """
    For each unique GT id, compute (detect_frame - onset_frame) in frames, where
    detect_frame is the first frame with a detection within the distance_tolerance
    of any GT coordinate for that id at that frame or later.
    Saves:
      - detection_latency_per_event.csv
      - detection_latency_summary.csv
    """
    if cdist is None:
        raise ImportError("scipy is required for detection latency calculation.")

    # Build per-frame detection coordinate lists
    dets = extract_pixel_detections(final_mask_np)
    # Index GT by id with ordered time
    id_to_frames = defaultdict(list)  # id -> list[(frame, x, y)]
    for f, pts in gt_data.items():
        for x, y, gid in pts:
            id_to_frames[int(gid)].append((int(f), float(x), float(y)))
    for gid in id_to_frames:
        id_to_frames[gid] = sorted(id_to_frames[gid], key=lambda t: t[0])

    rows = []
    for gid, samples in id_to_frames.items():
        if not samples:
            continue
        onset_frame = samples[0][0]

        detect_frame = None
        # scan frames from onset onward until we find a matching detection
        search_frames = sorted([f for f in dets.keys() if f >= onset_frame])
        for f in search_frames:
            det_coords = np.array([[d['x'], d['y']] for d in dets[f]]) if dets.get(f) else None
            if det_coords is None or len(det_coords) == 0:
                continue
            # Compare to any GT coord for this id at this frame (or nearest future frame if sparse)
            # First, try current-frame matches
            gt_now = [(x, y) for (ff, x, y) in samples if ff == f]
            # if none at this frame, allow comparing to latest available GT position so far
            if not gt_now:
                # use the most recent GT sample at or before f
                past = [(ff, x, y) for (ff, x, y) in samples if ff <= f]
                if past:
                    x, y = past[-1][1], past[-1][2]
                    gt_now = [(x, y)]
            if not gt_now:
                continue
            gt_coords = np.array(gt_now)
            d = cdist(gt_coords, det_coords).min()
            if d <= distance_tolerance:
                detect_frame = f
                break

        latency = None if detect_frame is None else (detect_frame - onset_frame)
        rows.append({
            "id": gid,
            "onset_frame": onset_frame,
            "detect_frame": detect_frame if detect_frame is not None else -1,
            "latency_frames": latency if latency is not None else np.nan
        })

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(output_dir / "detection_latency_per_event.csv", index=False)

    if not df.empty and df["latency_frames"].notna().any():
        vals = df["latency_frames"].dropna().values
        summary = pd.DataFrame([{
            "count": int(len(vals)),
            "mean_latency": float(np.mean(vals)),
            "median_latency": float(np.median(vals)),
            "min_latency": float(np.min(vals)),
            "max_latency": float(np.max(vals))
        }])
        summary.to_csv(output_dir / "detection_latency_summary.csv", index=False)