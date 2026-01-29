# evaluation/analysis.py
"""
High-level analysis functions for evaluating model performance, running statistical
tests, and generating data for reports.
"""
import logging
import random
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist
import config
from scipy.stats import beta, chi2

from . import metrics
from utils import extract_pixel_detections
from core.detection import CFAR, apply_neighborhood_filter
from core.filters import get_feature_map
from evaluation.metrics import calculate_froc_point_metrics

# --- add near the top of analysis.py (after imports) ---
def _TPR(point):   return point.get("TPR",  point.get("tpr"))
def _FPPI(point):  return point.get("FPPI", point.get("fppi"))

def find_top_k_models(all_results: List[Dict], k: int) -> List[Dict]:
    """
    Finds top K models by selecting configurations that achieve 100% TPR
    and ranking them by the minimum FPPI.
    """
    logging.info("🎯 Optimizing for 100% TPR with minimum FPPI...")
    best_points_at_100_tpr = []

    for result in all_results:
        roc_points = result.get('roc_points', [])
        if not roc_points:
            continue

        points_at_100_tpr = [
            p for p in roc_points
            if _TPR(p) is not None and np.isclose(_TPR(p), 1.0)
        ]

        if not points_at_100_tpr:
            continue

        min_fppi_point = min(points_at_100_tpr, key=lambda p: (_FPPI(p) if _FPPI(p) is not None else float('inf')))

        best_points_at_100_tpr.append({
            'params': result['params'],
            'point': min_fppi_point,
            'fppi': _FPPI(min_fppi_point)
        })

    if not best_points_at_100_tpr:
        logging.warning("No hyperparameter configurations achieved 100% TPR. Cannot rank models.")
        return []

    sorted_models = sorted(best_points_at_100_tpr, key=lambda p: p['fppi'])
    top_k_count = min(k, len(sorted_models))
    logging.info(f"🏆 Found {len(sorted_models)} models that achieved 100% TPR. Reporting top {top_k_count}.")
    if sorted_models:
        best_model = sorted_models[0]
        logging.info(f"   - Best model achieves 100% TPR with FPPI = {best_model['fppi']:.3f} and params: {best_model['params']['varied_param']}")

    return sorted_models[:k]

def find_best_params_per_tp_level(all_results: List[Dict], num_levels: int) -> List[Dict]:
    """Finds the best hyperparameter configuration for various True Positive counts."""
    logging.info("🏆 Searching for best hyperparameter configurations per TP level...")
    best_finds_by_tp = defaultdict(lambda: {'fppi': float('inf')})
    for result in all_results:
        for point in result.get('roc_points', []):
            tp, fppi = point.get('TP'), point.get('FPPI')
            if tp is not None and fppi is not None and fppi < best_finds_by_tp[tp]['fppi']:
                best_finds_by_tp[tp] = {'target_tp': tp, 'params': result['params'], 'point': point, 'fppi': fppi}
    
    top_tp_keys = sorted([k for k in best_finds_by_tp if k is not None], reverse=True)[:num_levels]
    final_list = sorted([best_finds_by_tp[key] for key in top_tp_keys], key=lambda x: x['target_tp'], reverse=True)
    logging.info(f"Found {len(final_list)} optimal configurations across different TP levels.")
    return final_list

def calculate_and_save_paper_metrics(video_np, features_np, neuron_gt_data, model_config, output_dir):
    """Calculates and saves a specific set of metrics for manuscript figures."""
    logging.info("🔬 Calculating summary metrics for manuscript...")
    T, H, W = features_np.shape
    total_voxels = video_np.size
    event_voxels = sum(len(points) for points in neuron_gt_data.values())
    voxel_occupancy = (event_voxels / total_voxels) * 100 if total_voxels > 0 else 0

    gt_df = pd.DataFrame(
        [(frame, id, x, y) for frame, points in neuron_gt_data.items() for x, y, id in points],
        columns=['frame', 'id', 'x', 'y']
    )
    if not gt_df.empty:
        duration_df = gt_df.groupby('id')['frame'].agg(['min', 'max']).reset_index()
        duration_df['duration'] = duration_df['max'] - duration_df['min'] + 1
        duration_df[['duration']].to_csv(output_dir / 'event_duration_stats.csv', index=False)

        first_pos = gt_df.loc[gt_df.groupby('id')['frame'].idxmin()][['id', 'x', 'y']].set_index('id')
        last_pos  = gt_df.loc[gt_df.groupby('id')['frame'].idxmax()][['id', 'x', 'y']].set_index('id')
        merged_pos = first_pos.join(last_pos, lsuffix='_first', rsuffix='_last')
        merged_pos['dist'] = np.sqrt(
            (merged_pos['x_last'] - merged_pos['x_first'])**2 +
            (merged_pos['y_last'] - merged_pos['y_first'])**2
        )
        max_drift_pixels = float(merged_pos['dist'].max())
        max_drift_um = max_drift_pixels * 0.5  # pixel size assumption
    else:
        max_drift_pixels, max_drift_um = 0.0, 0.0

    # --- Ablation study (No Spatio-Temporal Filter) ---
    # Robustly pick a CFAR config: prefer cfar1_params (new), else legacy cfar_params, else cfar2_params.
    params = model_config.get('params', {})
    cfar_dict = params.get('cfar1_params') or params.get('cfar_params') or params.get('cfar2_params')
    z_thresh = float(model_config.get('point', {}).get('z_score', 0.0))

    metrics_no_st = {'TPR': np.nan, 'FPPI': np.nan}
    try:
        if cfar_dict is None:
            logging.warning("No CFAR params found in model_config; skipping 'No ST' ablation.")
        else:
            # choose device safely (string or torch.device accepted)
            dev = model_config.get('device')
            if dev is None:
                dev = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

            raw_video_features = torch.from_numpy(video_np.astype(np.float32)).to(dev)
            cfar_params = dict(cfar_dict)
            cfar_params['T'] = z_thresh

            cfar_detector = CFAR(cfar_params)
            _, cfar_mask_no_st, _, _ = cfar_detector(raw_video_features.unsqueeze(1).float())

            neigh_size, neigh_k = params['neighborhood_config']
            final_mask_no_st = apply_neighborhood_filter(cfar_mask_no_st.squeeze(1), neigh_k, neigh_size)
            dets_no_st = extract_pixel_detections(final_mask_no_st.detach().cpu().numpy())

            metrics_no_st = metrics.calculate_froc_point_metrics(
                neuron_gt_data, dets_no_st, video_np.shape, params['distance_tolerance']
            )
    except Exception as e:
        logging.exception(f"Ablation (No ST) failed: {e}")

    paper_metrics = {
        "Voxel Occupancy (%)":             [f"{voxel_occupancy:.4f}"],
        "Optimal Z-Score (T)":             [f"{z_thresh:.2f}"],
        "True Positive Rate":              [_TPR(model_config['point'])],
        "False Positives Per Image":       [_FPPI(model_config['point'])],
        "Total Ground Truth Events":       [len(gt_df['id'].unique()) if not gt_df.empty else 0],
        "Max Sample Drift (pixels)":       [f"{max_drift_pixels:.2f}"],
        "Max Sample Drift (um)":           [f"{max_drift_um:.2f}"],
        "Ablation (No ST Filter) TPR":     [f"{metrics_no_st.get('TPR', np.nan):.3f}" if metrics_no_st.get('TPR') is not None else "nan"],
        "Ablation (No ST Filter) FPPI":    [f"{metrics_no_st.get('FPPI', np.nan):.3f}" if metrics_no_st.get('FPPI') is not None else "nan"],
    }
    pd.DataFrame(paper_metrics).to_csv(output_dir / "manuscript_metrics_summary.csv", index=False)
    logging.info(f"✅ Manuscript-specific metrics saved to {output_dir / 'manuscript_metrics_summary.csv'}")

def generate_error_taxonomy_report(fp_detections: Dict, vessel_gt_data: Dict, output_dir: Path):
    """Categorizes false positives and saves the taxonomy to a CSV."""
    logging.info("🔬 Generating error taxonomy for false positives...")
    fp_list = [(frame, coord['x'], coord['y']) for frame, coords in fp_detections.items() for coord in coords]
    if not fp_list:
        logging.info("No false positives to analyze for error taxonomy.")
        return

    vascular_fp_count = 0
    for frame, fp_x, fp_y in tqdm(fp_list, desc="Analyzing FPs", ncols=80, leave=False):
        vessels_in_frame = vessel_gt_data.get(frame, [])
        if not vessels_in_frame: continue
        vessel_coords = np.array([[x, y] for x, y, _ in vessels_in_frame])
        fp_coord = np.array([[fp_x, fp_y]])
        if np.min(cdist(fp_coord, vessel_coords)) <= (4.5 * 1.5): # Example distance threshold
            vascular_fp_count += 1
            
    total_fp = len(fp_list)
    other_fp_count = total_fp - vascular_fp_count
    taxonomy_data = {
        "Category": ["Vascular Signals", "Other (e.g., Debris)"],
        "Count": [vascular_fp_count, other_fp_count],
        "Percentage": [f"{(vascular_fp_count/total_fp)*100:.2f}%" if total_fp > 0 else "0.00%", f"{(other_fp_count/total_fp)*100:.2f}%" if total_fp > 0 else "0.00%"]
    }
    pd.DataFrame(taxonomy_data).to_csv(output_dir / "error_taxonomy.csv", index=False)
    logging.info(f"✅ Error taxonomy report saved to {output_dir / 'error_taxonomy.csv'}")

def log_performance_profile(optimal_config, video_np, device, output_dir):
    """Profiles and saves the processing speed and memory usage."""
    logging.info("🔬 Profiling performance...")
    if not (device.type == 'cuda' and torch.cuda.is_available()):
        logging.warning("GPU not available, skipping performance profiling.")
        return
        
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    _, _ = get_feature_map(video_np, optimal_config['params'], device)
    end_time = time.time()
    
    duration = end_time - start_time
    num_frames = video_np.shape[0]
    fps = num_frames / duration
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    gpu_model = torch.cuda.get_device_name(device)
    perf_data = {
        "GPU Model": [gpu_model],
        "Frames Per Second (fps)": [f"{fps:.2f}"],
        "Peak VRAM Usage (GB)": [f"{peak_vram_gb:.3f}"],
        "Total Processing Time (s)": [f"{duration:.2f}"],
    }
    pd.DataFrame(perf_data).to_csv(output_dir / "performance_profile.csv", index=False)
    logging.info(f"✅ Performance profile saved to {output_dir / 'performance_profile.csv'}")

def calculate_bootstrap_ci(final_mask_np, gt_data, video_shape, distance_tolerance, output_dir, n_bootstraps=1000):
    """Calculates and saves bootstrapped 95% confidence intervals for TPR and FPPI."""
    logging.info(f"🔬 Calculating 95% confidence intervals with {n_bootstraps} bootstraps...")
    
    # Get original metrics and match status
    base_metrics = metrics.calculate_froc_point_metrics(gt_data, extract_pixel_detections(final_mask_np), video_shape, distance_tolerance, return_matches=True, return_fp_coords=True)
    if 'gt_match_status' not in base_metrics:
        logging.warning("Could not get match status for bootstrap CI calculation.")
        return
        
    gt_match_status = base_metrics['gt_match_status']
    fp_detections = base_metrics.get('fp_detections', {})
    
    tp_indices = np.where(gt_match_status)[0]
    fp_coords = [coord for frame_dets in fp_detections.values() for coord in frame_dets]
    total_gt = len(gt_match_status)

    if total_gt == 0:
        logging.warning("No ground truth events for bootstrap CI calculation.")
        return
        
    tpr_boot, fppi_boot = [], []
    for _ in tqdm(range(n_bootstraps), desc="Bootstrap CI", ncols=80, leave=False):
        # Resample True Positives
        tp_resampled_indices = np.random.choice(tp_indices, size=len(tp_indices), replace=True) if len(tp_indices) > 0 else []
        tp_boot_count = len(np.unique(tp_resampled_indices))
        tpr_boot.append(tp_boot_count / total_gt)
        
        # Resample False Positives
        fp_resampled = random.choices(fp_coords, k=len(fp_coords)) if fp_coords else []
        fppi_boot.append(len(fp_resampled) / video_shape[0])
        
    tpr_ci = np.percentile(tpr_boot, [2.5, 97.5])
    fppi_ci = np.percentile(fppi_boot, [2.5, 97.5])
    
    ci_data = {
        "Metric": ["TPR", "FPPI"],
        "Value": [base_metrics.get('TPR'), base_metrics.get('FPPI')],
        "95% CI Lower": [tpr_ci[0], fppi_ci[0]],
        "95% CI Upper": [tpr_ci[1], fppi_ci[1]],
    }
    pd.DataFrame(ci_data).to_csv(output_dir / "confidence_intervals.csv", index=False)
    logging.info(f"✅ Confidence intervals saved to {output_dir / 'confidence_intervals.csv'}")

def calculate_confidence_intervals(final_mask_np,
                                   gt_data,
                                   video_shape,
                                   distance_tolerance,
                                   output_dir):
    """
    Fast CIs for TPR (Clopper–Pearson) and FPPI (Poisson rate).
    Falls back to bootstrap when config.CI_METHOD == "bootstrap".
    """
    logging.info("🔬 Computing confidence intervals (%s)...", config.CI_METHOD)

    base = metrics.calculate_froc_point_metrics(
        gt_data,
        extract_pixel_detections(final_mask_np),
        video_shape,
        distance_tolerance,
        return_fp_coords=False,
        return_matches=True,
    )

    TP   = int(base.get("TP", 0))
    FN   = int(base.get("FN", 0))
    FP   = int(base.get("FP", 0))
    Ngt  = TP + FN
    F    = int(video_shape[0])
    TPR  = base.get("TPR", 0.0)
    FPPI = base.get("FPPI", 0.0)

    if config.CI_METHOD.lower() == "bootstrap":
        n_boot = int(getattr(config, "BOOTSTRAP_N", 200))
        return calculate_bootstrap_ci(
            final_mask_np, gt_data, video_shape, distance_tolerance,
            output_dir, n_bootstraps=n_boot
        )

    # ---- Parametric (default) ----
    alpha = float(getattr(config, "CI_ALPHA", 0.05))

    # TPR CI (binomial Clopper–Pearson)
    if Ngt > 0:
        # Handle edge cases (TP=0 or TP=Ngt) to avoid beta(nans)
        tpr_lo = 0.0 if TP == 0   else beta.ppf(alpha/2, TP,   Ngt-TP+1)
        tpr_hi = 1.0 if TP == Ngt else beta.ppf(1-alpha/2, TP+1, Ngt-TP)
    else:
        tpr_lo, tpr_hi = 0.0, 0.0

    # FPPI CI (Poisson exact CI for rate per frame)
    # FP ~ Poisson(λ*F)  => λ = FPPI
    if F > 0:
        # chi-square quantile method
        # lower bound (0 when FP=0), upper bound well-defined always
        from math import isfinite
        if FP == 0:
            fppi_lo = 0.0
        else:
            fppi_lo = 0.5 * chi2.ppf(alpha/2,  2*FP)     / F
        fppi_hi = 0.5 * chi2.ppf(1-alpha/2, 2*(FP+1)) / F

        # numeric safety
        fppi_lo = float(fppi_lo) if isfinite(fppi_lo) else 0.0
        fppi_hi = float(fppi_hi) if isfinite(fppi_hi) else FPPI
    else:
        fppi_lo, fppi_hi = 0.0, 0.0

    import pandas as pd
    ci_data = pd.DataFrame([
        {"Metric": "TPR",  "Value": TPR,  "95% CI Lower": tpr_lo,  "95% CI Upper": tpr_hi},
        {"Metric": "FPPI", "Value": FPPI, "95% CI Lower": fppi_lo, "95% CI Upper": fppi_hi},
    ])
    out = output_dir / "confidence_intervals.csv"
    ci_data.to_csv(out, index=False)
    logging.info(f"✅ Confidence intervals saved to {out}")

def generate_and_save_stage_wise_samples(raw_np, gamma_np, cfar_np, gt_data, output_dir):
    """Samples and saves foreground/background pixel values from pipeline stages."""
    logging.info("🔬 Generating pixel samples for stage-wise histograms...")
    
    gt_locations_set = set()
    for t, points in gt_data.items():
        for x, y, _ in points:
            if 0 <= t < raw_np.shape[0] and 0 <= int(y) < raw_np.shape[1] and 0 <= int(x) < raw_np.shape[2]:
                gt_locations_set.add((t, int(y), int(x)))

    if not gt_locations_set:
        logging.warning("No ground truth data available to generate histogram samples.")
        return

    raw_samples, gamma_samples, cfar_samples = [], [], []
    H, W = raw_np.shape[1], raw_np.shape[2]

    for t, y, x in tqdm(list(gt_locations_set), desc="Sampling Pixels", ncols=80, leave=False):
        # Sample foreground
        raw_samples.append({'value': raw_np[t, y, x], 'label': 'FG'})
        gamma_samples.append({'value': gamma_np[t, y, x], 'label': 'FG'})
        cfar_samples.append({'value': cfar_np[t, y, x], 'label': 'FG'})

        # Sample background
        while True:
            y_rand, x_rand = np.random.randint(0, H), np.random.randint(0, W)
            if (t, y_rand, x_rand) not in gt_locations_set:
                raw_samples.append({'value': raw_np[t, y_rand, x_rand], 'label': 'BG'})
                gamma_samples.append({'value': gamma_np[t, y_rand, x_rand], 'label': 'BG'})
                cfar_samples.append({'value': cfar_np[t, y_rand, x_rand], 'label': 'BG'})
                break
    
    pd.DataFrame(raw_samples).to_csv(output_dir / "stage_samples_raw.csv", index=False)
    pd.DataFrame(gamma_samples).to_csv(output_dir / "stage_samples_gamma.csv", index=False)
    pd.DataFrame(cfar_samples).to_csv(output_dir / "stage_samples_cfarz.csv", index=False)
    logging.info(f"✅ Stage-wise sample data saved to {output_dir}")

# In evaluation/analysis.py

def find_balanced_operating_points(all_results: List[Dict], k: int, fppi_normalization_max: float) -> List[Dict]:
    """
    Finds top K models by selecting the operating point that maximizes the Youden's J
    statistic (TPR - normalized FPPI), representing a balance between sensitivity
    and specificity. This aligns with finding a point of 'equal risk' or a slope
    of ~45 degrees on a normalized FROC curve.

    Args:
        all_results: The list of results from the grid search.
        k: The number of top models to return.
        fppi_normalization_max: The FPPI value to consider as 'maximum' for normalization (e.g., the truncation limit).

    Returns:
        A list of the top K models sorted by their balanced score.
    """
    logging.info(f"🎯 Optimizing for a balanced trade-off (Youden's J) with FPPI normalized to {fppi_normalization_max}...")
    best_balanced_points = []

    for result in all_results:
        roc_points = result.get('roc_points', [])
        if not roc_points:
            continue

        # Find the point with the best Youden's J score for this parameter set
        best_point = max(
            roc_points,
            key=lambda p: (_TPR(p) or 0.0) - ((_FPPI(p) or 0.0) / fppi_normalization_max)
        )
        best_score = (_TPR(best_point) or 0.0) - ((_FPPI(best_point) or 0.0) / fppi_normalization_max)

        best_balanced_points.append({
            'params': result['params'],
            'point': best_point,
            'youden_j_score': best_score
        })

    if not best_balanced_points:
        logging.warning("Could not find any balanced operating points.")
        return []

    # Sort all configurations by their best Youden's J score
    sorted_models = sorted(best_balanced_points, key=lambda p: p['youden_j_score'], reverse=True)

    top_k_count = min(k, len(sorted_models))
    logging.info(f"🏆 Found {len(sorted_models)} balanced models. Reporting top {top_k_count}.")
    if sorted_models:
        best_model = sorted_models[0]
        logging.info(
    f"   - Best balanced model achieves score={best_model['youden_j_score']:.3f} "
    f"(TPR={_TPR(best_model['point']):.3f}, FPPI={_FPPI(best_model['point']):.3f})"
)

    return sorted_models[:k]

# In evaluation/analysis.py

def calculate_and_save_unique_neuron_stats(gt_data: Dict, matched_gt_events: np.ndarray, all_gt_events_df: pd.DataFrame, output_dir: Path):
    """
    Calculates the detection rate of unique neurons and saves it.
    """
    logging.info("🔬 Calculating unique neuron detection rate...")
    all_neuron_ids = set(all_gt_events_df['id'].unique())
    if not all_neuron_ids:
        logging.warning("No unique neuron IDs found in ground truth.")
        return

    # Find the IDs of neurons that were matched at least once
    matched_indices = np.where(matched_gt_events)[0]
    detected_neuron_ids = set(all_gt_events_df.iloc[matched_indices]['id'].unique())

    detection_rate = len(detected_neuron_ids) / len(all_neuron_ids) if all_neuron_ids else 0.0
    
    stats = {
        "Total Unique Neurons": [len(all_neuron_ids)],
        "Detected Unique Neurons": [len(detected_neuron_ids)],
        "Unique Neuron Detection Rate": [f"{detection_rate:.2%}"]
    }
    
    pd.DataFrame(stats).to_csv(output_dir / "unique_neuron_detection_stats.csv", index=False)
    logging.info(f"✅ Unique neuron stats saved. Rate: {detection_rate:.2%}")

def build_gt_events_df(neuron_gt_data: Dict[int, List[Tuple[float,float,int]]]) -> pd.DataFrame:
    """
    Build a flat DataFrame of ground-truth events with columns [frame, id, x, y]
    from the repo's neuron_gt_data structure: {frame: [(x, y, id), ...], ...}.
    """
    rows = []
    for f, points in neuron_gt_data.items():
        for x, y, id_ in points:
            rows.append((int(f), int(id_), float(x), float(y)))
    df = pd.DataFrame(rows, columns=["frame", "id", "x", "y"])
    if not df.empty:
        df = df.sort_values(["id", "frame"]).reset_index(drop=True)
    return df

def compute_unique_id_coverage(
    detections: Dict[int, List[Dict[str, float]]],
    gt_df: pd.DataFrame,
    distance_tolerance: float,
) -> float:
    """
    Fraction of unique GT neuron IDs that are detected at least once.
    A GT event (frame,id,x,y) is considered detected if any detection in that frame
    lies within 'distance_tolerance' in Euclidean distance. The ID is counted once.
    """
    if gt_df.empty:
        return 0.0
    try:
        from scipy.spatial.distance import cdist
    except Exception:
        cdist = None

    detected_ids = set()
    for f, df_f in gt_df.groupby("frame"):
        dets = detections.get(int(f), [])
        if not len(dets):
            continue
        det_arr = np.array([[d["x"], d["y"]] for d in dets], dtype=float)
        gt_arr = df_f[["x", "y"]].to_numpy(dtype=float)
        if det_arr.size == 0 or gt_arr.size == 0:
            continue

        if cdist is not None:
            D = cdist(gt_arr, det_arr)
            matched = (D <= float(distance_tolerance)).any(axis=1)
        else:
            matched = []
            for gx, gy in gt_arr:
                ok = False
                for dx, dy in det_arr:
                    if (dx - gx) ** 2 + (dy - gy) ** 2 <= float(distance_tolerance) ** 2:
                        ok = True
                        break
                matched.append(ok)
            matched = np.asarray(matched, dtype=bool)

        if matched.any():
            ids_here = df_f.loc[matched, "id"].astype(int).tolist()
            detected_ids.update(ids_here)

    total_ids = int(gt_df["id"].nunique())
    return (len(detected_ids) / total_ids) if total_ids > 0 else 0.0

def find_max_threshold_with_full_id_coverage(
    z_score_map: np.ndarray,
    neighborhood_config: Tuple[int, int],
    neuron_gt_data: Dict[int, List[Tuple[float, float, int]]],
    video_shape: Tuple[int, int, int],
    distance_tolerance: float,
    sweep: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Given a (T,H,W) z-score map, find the largest threshold T* in 'sweep'
    such that unique-id coverage is exactly 1.0 (i.e., every neuron ID is detected at least once).
    Returns a metrics dict (compatible with calculate_froc_point_metrics output) augmented with 'z_score'.
    If no threshold achieves full coverage, returns None.
    """
    assert z_score_map.ndim == 3, f"Expected (T,H,W), got {tuple(z_score_map.shape)}"
    ksize, k = int(neighborhood_config[0]), int(neighborhood_config[1])
    if sweep is None:
        sweep = getattr(config, "RESTRICTIVE_Z_SWEEP", getattr(config, "Z_SCORE_SWEEP"))
    sweep = np.asarray(sweep, dtype=float)
    if sweep.size == 0:
        logging.warning("Empty threshold sweep for restrictive ID-coverage experiment.")
        return None

    # Build GT table once
    gt_df = build_gt_events_df(neuron_gt_data)
    if gt_df.empty:
        logging.warning("No ground-truth neuron events found; skipping restrictive experiment.")
        return None

    T, H, W = video_shape
    best_thr = None
    best_metrics = None

    # Torch tensor once to avoid re-wrapping in the loop
    z_thw = torch.from_numpy(z_score_map).to(torch.float32)
    for thr in sweep:
        # Threshold, neighborhood filter, detections
        mask_thw = (z_thw > float(thr)).to(torch.uint8)
        mask_filtered = apply_neighborhood_filter(mask_thw, k=k, kernel_size=ksize)  # (T,H,W) uint8
        detections = extract_pixel_detections(mask_filtered.cpu().numpy())

        # Compute unique-id coverage
        uic = compute_unique_id_coverage(detections, gt_df, distance_tolerance)
        if uic >= 1.0 - 1e-9:  # allow for floating tolerances
            # Compute standard metrics at this threshold so downstream ranking is consistent
            metrics = calculate_froc_point_metrics(
                gt_data=neuron_gt_data,
                detections=detections,
                video_shape=video_shape,
                distance_tolerance=distance_tolerance,
            )
            metrics["z_score"] = float(thr)
            metrics["UniqueID_Coverage"] = float(uic)

            # Keep the most restrictive (i.e., the largest thr)
            best_thr = float(thr)
            best_metrics = metrics

    return best_metrics