# reporting/generators.py
"""Functions for generating output reports, both detailed and summary CSVs."""
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import pandas as pd
import tifffile
import tqdm
import tqdm.auto as tqdma

# Import from our other project modules
import config
from utils import NumpyEncoder, extract_pixel_detections
from core.pipelines import run_single_stage, run_two_stage  # NEW
from core.filters import get_feature_map
from core.detection import CFAR, apply_neighborhood_filter
from evaluation import metrics, analysis
from evaluation.analysis import find_max_threshold_with_full_id_coverage
from core.filters import full_kalman_mcc_filter, full_kalman_mcc_filter_gpu_batches


# --- Functions to Generate Summary CSV Reports ---

def save_full_results_csv(all_results: List[Dict], output_dir: Path):
    """Saves all raw results from the grid search into a single CSV."""
    logging.info("Saving all grid search results to CSV...")
    if not all_results:
        logging.warning("No results to save.")
        return

    flat_results = []
    for res in all_results:
        # Copy to avoid mutating input
        res_copy = {**res}
        params = res_copy.get('params', {})
        varied_params = dict(params.get('varied_param', {}))  # shallow copy

        # Normalize some fields for readability
        if 'neighborhood_config' in varied_params:
            varied_params['neighborhood_config'] = str(varied_params['neighborhood_config'])

        # CFAR type can live under cfar1_params (gamma) or cfar_params (legacy)
        cfar_dict = params.get('cfar1_params', params.get('cfar_params', {}))
        cfar_type = cfar_dict.get('type', 'unknown')

        # NEW: backfill eps1/eps2 into varied_param for CSVs (if missing)
        if 'eps1' not in varied_params and 'cfar1_params' in params:
            varied_params['eps1'] = params['cfar1_params'].get('eps')
        if 'cfar2_params' in params and 'eps2' not in varied_params:
            varied_params['eps2'] = params['cfar2_params'].get('eps')


        # Include architecture when available
        arch = params.get('arch', 'single')

        params_row = {
            **varied_params,
            'arch': arch,
            'cfar_type': cfar_type,
        }

        for point in res_copy.get('roc_points', []):
            flat_results.append({**params_row, **point})

    full_df = pd.DataFrame(flat_results)
    output_path = output_dir / "all_grid_search_results.csv"
    full_df.to_csv(output_path, index=False)
    logging.info(f"✅ Full results saved to {output_path}")

def save_top_models_summary(top_k_models: List[Dict], output_dir: Path):
    """Saves a summary of the top K models (optimized for 100% TPR) to a CSV."""
    if not top_k_models:
        return
        
    summary_data = []
    for rank, model in enumerate(top_k_models):
        entry = {'Rank': rank + 1, **model['params']['varied_param'], **model['point']}
        entry['neighborhood_config'] = str(entry.get('neighborhood_config'))
        summary_data.append(entry)
        
    df = pd.DataFrame(summary_data)
    output_path = output_dir / "top_models_at_100_tpr.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"✅ Top models at 100% TPR saved to {output_path}")

def save_top_tp_configs_summary(top_tp_configs: List[Dict], output_dir: Path):
    """Saves a summary of the best configurations for different TP levels."""
    if not top_tp_configs:
        return
        
    summary_data = []
    for c in top_tp_configs:
        params_to_add = c['params']['varied_param']
        params_to_add['neighborhood_config'] = str(params_to_add.get('neighborhood_config', 'N/A'))
        row = {
            'TPs': c['target_tp'],
            'Min FPPI': f"{c['fppi']:.4f}",
            'Z-Score': f"{c['point']['z_score']:.2f}",
            **params_to_add
        }
        summary_data.append(row)
        
    df = pd.DataFrame(summary_data)
    output_path = output_dir / "top_configurations_summary.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"✅ Top configurations per TP level saved to {output_path}")

# --- Function to Generate Detailed Per-Model Report ---

def generate_detailed_report(
    model_config: Dict, rank: int, video_np: np.ndarray, device: torch.device,
    neuron_gt_data: Dict, vessel_gt_data: Dict, base_output_dir: Path
):
    """Generates a comprehensive report folder for a single model configuration."""
    # Local imports to avoid changing module-level imports
    from core.pipelines import run_single_stage, run_two_stage
    from core.detection import CFAR, apply_neighborhood_filter
    import tifffile

    output_dir = base_output_dir / f"Rank_{rank}_Report"
    logging.info(f"--- Generating detailed report for Rank #{rank} model ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    params, point = model_config['params'], model_config['point']
    z_score_thresh = float(point.get('z_score', 3.0))
    T, H, W = video_np.shape
    distance_tolerance = params['distance_tolerance']

    # Save model config
    with open(output_dir / "model_config.json", 'w') as f:
        json.dump(model_config, f, cls=NumpyEncoder, indent=4)

    # ---- Re-run pipeline on GPU to get intermediates & final mask (ONE threshold) ----
    ftype = params['varied_param']['filter_type']
    arch  = params.get('arch', 'single')

    panels_np: List[np.ndarray] = []
    final_mask_np: np.ndarray

    if ftype == 'gamma':
        if arch == 'single':
            out = run_single_stage(video_np, {
                "st1": params["st1"],
                "cfar1": params["cfar1_params"],
                "z_thresh": z_score_thresh
            })
            # Panels: raw | features1 | z1 | final
            panels_np = [
                video_np,
                out["features1"],
                out["z1"],
                out["final_mask"].astype(np.float32),
            ]
        else:  # two_stage
            out = run_two_stage(video_np, {
                "st1":  params["st1"],
                "cfar1": params["cfar1_params"],
                "st2":   params["st2"],
                "cfar2": params["cfar2_params"],
                "z_thresh": z_score_thresh
            })
            # Panels: raw | features1 | z1 | features2 | z2 | final
            panels_np = [
                video_np,
                out["features1"],
                out["z1"],
                out["features2"],
                out["z2"],
                out["final_mask"].astype(np.float32),
            ]

        # Neighborhood smoothing (display/metrics) on the final mask
        neigh_size, neigh_k = params['neighborhood_config']
        final_mask_t = torch.from_numpy(panels_np[-1] > 0.5)  # (T,H,W) bool
        final_mask_t = apply_neighborhood_filter(final_mask_t, neigh_k, neigh_size)
        final_mask_np = final_mask_t.cpu().numpy().astype(np.uint8)
        panels_np[-1] = final_mask_np.astype(np.float32)

        # Choose the "middle" features for paper metrics:
        # - single: features1
        # - two-stage: features1 (keep consistent with prior analyses)
        features_np = panels_np[1]

    else:
        # ---- Kalman legacy path (unchanged) ----
        from core.filters import get_feature_map
        features_np, _ = get_feature_map(video_np, params, device)
        features_gpu = torch.from_numpy(features_np.astype(np.float32)).to(device)

        # robust to new/old keys
        cfar_params = (params.get('cfar1_params') or params.get('cfar_params')).copy()
        cfar_params['T'] = z_score_thresh
        cfar_detector = CFAR(cfar_params)
        z_score_map, cfar_mask, _, _ = cfar_detector(features_gpu.unsqueeze(1).float())

        neigh_size, neigh_k = params['neighborhood_config']
        final_mask = apply_neighborhood_filter(cfar_mask.squeeze(1), neigh_k, neigh_size)
        final_mask_np = final_mask.cpu().numpy().astype(np.uint8)

        # Panels: raw | features | z | (raw CFAR mask) | final
        panels_np = [
            video_np,
            features_np,
            z_score_map.squeeze(1).detach().cpu().numpy(),
            cfar_mask.squeeze(1).detach().cpu().numpy().astype(np.float32),
            final_mask_np.astype(np.float32),
        ]

    # --- Sanitize panels (NaN/Inf -> 0) & write per-panel stats for debugging ---
    def _sanitize(a: np.ndarray) -> np.ndarray:
        x = a.astype(np.float32, copy=False)
        x[~np.isfinite(x)] = 0.0
        return x

    panels_np = [_sanitize(p) for p in panels_np]

    # Label panels for stats
    if ftype == 'gamma' and arch == 'single' and len(panels_np) == 4:
        panel_labels = ["raw", "features1", "z1", "final_mask"]
    elif ftype == 'gamma' and arch != 'single' and len(panels_np) == 6:
        panel_labels = ["raw", "features1", "z1", "features2", "z2", "final_mask"]
    elif ftype != 'gamma' and len(panels_np) == 5:
        panel_labels = ["raw", "features", "z", "cfar_mask_raw", "final_mask"]
    else:
        panel_labels = [f"panel_{i}" for i in range(len(panels_np))]

    stats = []
    for i, (lbl, p) in enumerate(zip(panel_labels, panels_np)):
        finite = np.isfinite(p)
        fin = int(finite.sum())
        total = int(p.size)
        if fin > 0:
            vals = p[finite]
            s = {
                "index": i, "label": lbl,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "finite": fin, "total": total
            }
        else:
            s = {"index": i, "label": lbl, "min": None, "max": None, "mean": None, "finite": 0, "total": total}
        stats.append(s)
        logging.info(f"[diag] panel {i} ({lbl}): "
                     f"finite={s['finite']}/{s['total']} "
                     f"min={s['min'] if s['min'] is not None else 'NA'} "
                     f"max={s['max'] if s['max'] is not None else 'NA'} "
                     f"mean={s['mean'] if s['mean'] is not None else 'NA'}")

    with open(output_dir / "diagnostic_panels_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- Save diagnostic video (normalize each panel to [0,1]) ---
    tensor_list = [torch.from_numpy(p.astype(np.float32)) if not torch.is_tensor(p) else p for p in panels_np]
    panels = []
    for p in tensor_list:
        p_cpu = p.cpu().float()
        min_val, max_val = p_cpu.min(), p_cpu.max()
        rng = max_val - min_val
        normalized_p = (p_cpu - min_val) / rng if rng > 1e-9 else torch.zeros_like(p_cpu)
        panels.append(normalized_p)

    # Concatenate along width and write ImageJ-compatible TIF
    try:
        tifffile.imwrite(
            output_dir / "diagnostic_video.tif",
            (torch.cat(panels, dim=2).numpy() * 65535).astype(np.uint16),
            imagej=True
        )
    except Exception as e:
        logging.warning(f"Failed to write diagnostic_video.tif: {e}")

    # --- Run and Save Analyses ---
    analysis.calculate_and_save_paper_metrics(video_np, features_np, neuron_gt_data, model_config, output_dir)

    # Get FP coordinates for error taxonomy
    metrics_with_fp = metrics.calculate_froc_point_metrics(
        neuron_gt_data,
        extract_pixel_detections(final_mask_np),
        (T, H, W),
        distance_tolerance,
        return_fp_coords=True
    )
    fp_detections = metrics_with_fp.get("fp_detections", {})
    if fp_detections:
        fp_coords_list = [{'frame': f, 'x': c['x'], 'y': c['y']} for f, coords in fp_detections.items() for c in coords]
        pd.DataFrame(fp_coords_list).to_csv(output_dir / "fp_coords.csv", index=False)
        if vessel_gt_data:
            analysis.generate_error_taxonomy_report(fp_detections, vessel_gt_data, output_dir)

    # Get FN data for per-ID analysis
    metrics_with_matches = metrics.calculate_froc_point_metrics(
        neuron_gt_data,
        extract_pixel_detections(final_mask_np),
        (T, H, W),
        distance_tolerance,
        return_matches=True
    )
    if 'gt_match_status' in metrics_with_matches:
        all_gt_events = pd.DataFrame(
            [(frame, pt[2]) for frame, points in neuron_gt_data.items() for pt in points],
            columns=['frame', 'id']
        )
        all_gt_events['matched'] = metrics_with_matches['gt_match_status']
        fn_counts = all_gt_events[~all_gt_events['matched']].groupby('id').size().reset_index(name='count')
        fn_counts.to_csv(output_dir / "fn_per_id.csv", index=False)

        # Unique neuron detection stats
        analysis.calculate_and_save_unique_neuron_stats(
            gt_data=neuron_gt_data,
            matched_gt_events=metrics_with_matches['gt_match_status'],
            all_gt_events_df=all_gt_events,
            output_dir=output_dir
        )

    # Remaining analyses
    analysis.log_performance_profile(model_config, video_np, device, output_dir)
    analysis.calculate_confidence_intervals(final_mask_np, neuron_gt_data, (T, H, W), distance_tolerance, output_dir)
    metrics.calculate_detection_latency(final_mask_np, neuron_gt_data, distance_tolerance, output_dir)

    logging.info(f"✅ Detailed report for Rank #{rank} saved to: {output_dir}")

# In reporting/generators.py

def save_balanced_summary(balanced_models: List[Dict], output_dir: Path):
    """Saves a summary of the top models based on the Youden's J balanced score."""
    if not balanced_models:
        return
        
    summary_data = []
    for rank, model in enumerate(balanced_models):
        entry = {
            'Rank': rank + 1,
            'Youden_J_Score': model['youden_j_score'],
            **model['params']['varied_param'],
            **model['point']
        }
        entry['neighborhood_config'] = str(entry.get('neighborhood_config'))
        summary_data.append(entry)
        
    df = pd.DataFrame(summary_data)
    output_path = output_dir / "top_models_balanced.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"✅ Top balanced models saved to {output_path}")

def _run_kalman_mcc(frames_np: np.ndarray, sigma: float, mu: float, max_frames: int):
    """Run MCC on first max_frames; prefer GPU if CuPy is available."""
    T = frames_np.shape[0]
    use_T = min(T, int(max_frames)) if max_frames and max_frames > 0 else T
    sub = frames_np[:use_T].copy()

    try:
        import cupy as cp  # noqa: F401
        logging.info("Kalman–MCC: using GPU (CuPy).")
        bg_u16, diff_u16 = full_kalman_mcc_filter_gpu_batches(
            sub, sigma=sigma, mu=mu, num_batches=4
        )
        return bg_u16, diff_u16
    except Exception as e:
        logging.info(f"Kalman–MCC: falling back to CPU ({e}).")
        bg_u16, diff_u16 = full_kalman_mcc_filter(sub, sigma=sigma, mu=mu)
        return bg_u16, diff_u16

def generate_kalman_background(
    video_np: np.ndarray,
    output_dir: Path,
    sigma: float,
    mu: float,
    max_frames: int = 300,
    write_tiffs: bool = False,  # NEW
) -> tuple[Path | None, Path | None]:
    """
    Runs Kalman–MCC and (optionally) writes kalman_bg.tif + kalman_diff.tif.
    Returns their paths or (None, None) if not written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Kalman–MCC: computing background%s...",
                 " and writing TIFFs" if write_tiffs else "")

    bg_u16, diff_u16 = _run_kalman_mcc(video_np, sigma=sigma, mu=mu, max_frames=max_frames)

    bg_path = output_dir / "kalman_bg.tif"
    diff_path = output_dir / "kalman_diff.tif"

    if write_tiffs:
        tifffile.imwrite(bg_path, bg_u16,   imagej=True)
        tifffile.imwrite(diff_path, diff_u16, imagej=True)
        logging.info(f"✅ Saved {bg_path} and {diff_path}")
        return bg_path, diff_path

    # No writes
    logging.info("ℹ️ Skipped writing large TIFFs (write_tiffs=False).")
    return None, None

def _compute_z_map_for_params(video_np: np.ndarray, params: Dict, device: Optional[torch.device]):
    """
    Compute a non-binarized z-score map (T,H,W) for a given model configuration.
    Supports:
      - Gamma single-stage  -> returns out["z1"]
      - Gamma two-stage     -> returns out["z2"]
      - Kalman–MCC + CFAR   -> residual features -> CFAR(T=0) -> z map
    """
    p = dict(params)  # shallow copy

    # Default, but Gamma pipelines ignore it for z output
    p.setdefault("z_thresh", 0.0)

    # What family are we in?
    # In your grid, this lives under params["varied_param"]["filter_type"]
    fam = (p.get("varied_param", {}).get("filter_type") or "").lower()

    # ---- Gamma path ----
    if fam == "gamma":
        arch = str(p.get("arch", "single")).lower()

        if arch in ("single", "single_stage", "single-stage"):
            out = run_single_stage(
                video_np=video_np,
                p={
                    "st1":   p["st1"],
                    "cfar1": p["cfar1_params"],  # map *_params -> pipeline key
                    "z_thresh": p["z_thresh"],
                },
                device=device,
            )
            return out["z1"]

        elif "two" in arch:
            out = run_two_stage(
                video_np=video_np,
                p={
                    "st1":   p["st1"],
                    "cfar1": p["cfar1_params"],
                    "st2":   p["st2"],
                    "cfar2": p["cfar2_params"],
                    "z_thresh": p["z_thresh"],
                },
                device=device,
            )
            return out["z2"]

        else:
            logging.info(f"[restrictive] Skipping unsupported arch={p.get('arch')}")
            return None

    # ---- Kalman–MCC path ----
    if fam == "kalman_mcc":
        # Build residual features, then CFAR (T=0) to get a z-score map,
        # mirroring your worker's legacy branch.
        features_np, _ = get_feature_map(video_np, p, device)
        x_tchw = torch.from_numpy(features_np).to(device).float().unsqueeze(1)  # (T,1,H,W)
        cfar_detector = CFAR({**p["cfar1_params"], "T": 0})
        with torch.no_grad():
            z, *_ = cfar_detector(x_tchw)
        return z.squeeze(1).detach().cpu().numpy()

    logging.info(f"[restrictive] Unknown filter family for params: {fam}")
    return None

def _results_row_from_point(params: Dict, point: Dict) -> Dict:
    row = {**params.get("varied_param", {}), **point}
    # Normalize tuple-like fields to strings for CSV safety
    if "neighborhood_config" in row and not isinstance(row["neighborhood_config"], str):
        row["neighborhood_config"] = str(row["neighborhood_config"])
    return row

def run_restrictive_id_coverage_experiment(
    all_results: List[Dict],
    video_np: np.ndarray,
    neuron_gt_data: Dict[int, List],
    device: torch.device,
    base_output_dir: Path,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    For each hyperparameter config, compute the CFAR z-map (no threshold) and then
    find the **largest** z-threshold whose detections achieve UniqueID_Coverage == 1.0
    (every GT neuron ID seen at least once). Return the top-K configs ranked by:

        1) Highest z* (most restrictive),
        2) Then lowest FPPI,
        3) Then highest TPR.

    Also saves a summary CSV under:
        <base_output_dir>/Restrictive_UniqueID_Coverage/restrictive_summary.csv

    Returns:
        A list of dicts with keys: {'params', 'point'} suitable for
        generators.generate_detailed_report / the existing reporting stack.
    """
    out_dir = base_output_dir / "Restrictive_UniqueID_Coverage"
    out_dir.mkdir(exist_ok=True, parents=True)

    logging.info("🎯 Running 'Restrictive Unique-ID Coverage' experiment (most-restrictive z* with full ID coverage)...")

    qualifying: List[Dict] = []
    summary_rows = []

    # Helper: compute z-map (T,H,W) for a given params/config
    def _compute_z_map(params: Dict) -> np.ndarray:
        ftype = params["varied_param"]["filter_type"]
        arch  = params.get("arch", "single")

        if ftype == "gamma":
            if arch == "single":
                out = run_single_stage(video_np, {
                    "st1": params["st1"],
                    "cfar1": params["cfar1_params"],
                    "z_thresh": 0.0,  # return raw z1
                })
                return out["z1"].astype(np.float32)
            else:
                out = run_two_stage(video_np, {
                    "st1": params["st1"],
                    "cfar1": params["cfar1_params"],
                    "st2": params["st2"],
                    "cfar2": params["cfar2_params"],
                    "z_thresh": 0.0,  # return raw z2
                })
                return out["z2"].astype(np.float32)

        # Legacy Kalman–MCC path: compute features then CFAR z with T=0
        feat_np, _ = get_feature_map(video_np, params, device)
        features = torch.from_numpy(feat_np.astype(np.float32)).to(device).unsqueeze(1)
        cfar_dict = params.get("cfar1_params", params.get("cfar_params", {}))
        cfar = CFAR({**cfar_dict, "T": 0.0})
        z_map, *_ = cfar(features)
        return z_map.squeeze().detach().cpu().numpy().astype(np.float32)

    # Iterate through every config once
    for res in tqdma.tqdm(all_results, desc="Restrictive ID coverage", ncols=80, leave=False):
        params = res["params"]
        try:
            z_map = _compute_z_map(params)

            metrics_at_best = analysis.find_max_threshold_with_full_id_coverage(
                z_score_map=z_map,
                neighborhood_config=params["neighborhood_config"],
                neuron_gt_data=neuron_gt_data,
                video_shape=video_np.shape,
                distance_tolerance=params["distance_tolerance"],
                sweep=getattr(config, "RESTRICTIVE_Z_SWEEP", getattr(config, "Z_SCORE_SWEEP", None)),
            )

            if metrics_at_best is None:
                continue

            # Store in a structure compatible with the rest of the pipeline
            qualifying.append({
                "params": params,
                "point": metrics_at_best,  # contains TPR, FPPI, z_score, UniqueID_Coverage
            })

            row = dict(params.get("varied_param", {}))
            row.update({
                "arch": params.get("arch", "single"),
                "z_star": metrics_at_best.get("z_score", np.nan),
                "TPR": metrics_at_best.get("TPR", np.nan),
                "FPPI": metrics_at_best.get("FPPI", np.nan),
                "UniqueID_Coverage": metrics_at_best.get("UniqueID_Coverage", np.nan),
            })
            summary_rows.append(row)

        except Exception as e:
            logging.exception(f"[Restrictive] Failed for params={params.get('varied_param', params)}: {e}")

    # Save CSV summary
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "restrictive_summary.csv", index=False)
        logging.info(f"✅ Restrictive summary saved to {out_dir / 'restrictive_summary.csv'}")
    else:
        logging.warning("No configurations achieved full unique-ID coverage; no summary CSV written.")

    if not qualifying:
        logging.info("🏁 'Restrictive unique-ID coverage' experiment complete (no qualifying configs).")
        return []

    # Rank by (z* DESC, FPPI ASC, TPR DESC)
    qualifying.sort(
        key=lambda m: (
            float(m["point"].get("z_score", -1e9)),
            float(m["point"].get("FPPI", np.inf)),
            -float(m["point"].get("TPR", 0.0)),
        ),
        reverse=False,  # because we invert signs appropriately below
    )
    # The sort above is a bit tricky; simpler is:
    qualifying.sort(
        key=lambda m: (
            -float(m["point"].get("z_score", -1e9)),   # highest z* first
             float(m["point"].get("FPPI", np.inf)),     # then lowest FPPI
            -float(m["point"].get("TPR", 0.0)),         # then highest TPR
        )
    )

    if top_k is not None and top_k > 0:
        qualifying = qualifying[:min(top_k, len(qualifying))]

    logging.info(f"🏁 'Restrictive unique-ID coverage' experiment complete. {len(qualifying)} configs selected.")
    return qualifying