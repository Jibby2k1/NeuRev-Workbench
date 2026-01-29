# main.py
"""Main execution script to run the grid search, analysis, and reporting."""

import os
import sys
import logging
import random
import multiprocessing as mp
from itertools import product
from typing import List, Dict, Generator, Tuple
from pathlib import Path

import numpy as np
import torch
import psutil
import pandas as pd
from tqdm import tqdm

# Import from our new modules
import config
from data_loader import load_video, parse_ground_truth_csv
from worker import process_full_task_for_worker
from evaluation import analysis
from reporting import generators, plotters
from core.filters import get_feature_map
from core.detection import CFAR
from core.pipelines import run_single_stage, run_two_stage  # NEW
from reporting.generators import run_restrictive_id_coverage_experiment

def setup_environment():
    """Configures logging, seeding, and environment variables."""
    os.environ["OMP_NUM_THREADS"] = str(max(1, psutil.cpu_count(logical=False) // config.NUM_WORKERS))
    torch.set_float32_matmul_precision('high')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    logging.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

def create_param_configs():
    """
    Produces parameter dicts understood by the worker for:
      - Gamma (single-stage)  : ST1 + CFAR1
      - Gamma (two-stage)     : ST1 + CFAR1 -> ST2 + CFAR2   (single global threshold later)
      - Kalman–MCC (legacy)   : unchanged
    """
    from itertools import product

    def _grid(param_dict):
        keys = list(param_dict.keys())
        for vals in product(*(param_dict[k] for k in keys)):
            yield dict(zip(keys, vals))

    all_configs = []

    # --- Gamma family using the new architecture knobs ---
    if "gamma" in getattr(config, "ENABLED_FILTER_FAMILIES", ["gamma"]):
        for arch in getattr(config, "ARCHITECTURES", ["single"]):
            for st1 in _grid(config.ST1_PARAMS):
                for shared in _grid({
                    "distance_tolerance": config.SHARED_PARAMS["distance_tolerance"],
                    "neighborhood_config": config.SHARED_PARAMS["neighborhood_config"],
                }):
                    if arch == "single":
                        for c1 in config.CFAR1_PARAMS:
                            all_configs.append({
                                # For reporting/CSV & grouping
                                "varied_param": {
                                    "filter_type": "gamma",
                                    "arch": "single",
                                    "t_decay": st1["t_decay"],
                                    "s_decay": st1["s_decay"],
                                    "neighborhood_config": shared["neighborhood_config"],
                                    "eps1": c1.get("eps"),
                                },
                                # For the worker/pipeline
                                "arch": "single",
                                "st1": st1,
                                "cfar1_params": c1,
                                "neighborhood_config": shared["neighborhood_config"],
                                "distance_tolerance": shared["distance_tolerance"],
                            })
                    else:  # two_stage
                        for st2 in _grid(config.ST2_PARAMS):
                            for c1 in config.CFAR1_PARAMS:
                                for c2 in config.CFAR2_PARAMS:
                                    all_configs.append({
                                        "varied_param": {
                                            "filter_type": "gamma",
                                            "arch": "two_stage",
                                            "t_decay": st1["t_decay"],
                                            "s_decay": st1["s_decay"],
                                            "t2_decay": st2["t_decay"],
                                            "s2_decay": st2["s_decay"],
                                            "neighborhood_config": shared["neighborhood_config"],
                                            "eps1": c1.get("eps"),
                                            "eps2": c2.get("eps"),
                                        },
                                        "arch": "two_stage",
                                        "st1": st1,
                                        "st2": st2,
                                        "cfar1_params": c1,
                                        "cfar2_params": c2,
                                        "neighborhood_config": shared["neighborhood_config"],
                                        "distance_tolerance": shared["distance_tolerance"],
                                    })

    # --- Kalman–MCC family (legacy path preserved) ---
    grids = getattr(config, "PREPROCESSOR_GRIDS", {})
    if "kalman_mcc" in getattr(config, "ENABLED_FILTER_FAMILIES", []) and "kalman_mcc" in grids:
        p_set = grids["kalman_mcc"]
        filter_keys = list(p_set.keys())
        shared_keys = list(config.SHARED_PARAMS.keys())
        for filter_vals in product(*(p_set[k] for k in filter_keys)):
            filter_config = dict(zip(filter_keys, filter_vals))
            for c1 in config.CFAR1_PARAMS:  # NEW: iterate CFAR1
                for shared_vals in product(*(config.SHARED_PARAMS[k] for k in shared_keys)):
                    shared_config = dict(zip(shared_keys, shared_vals))
                    all_configs.append({
                        "varied_param": {
                            "filter_type": "kalman_mcc",
                            "sigma": filter_config["sigma"],
                            "mu":    filter_config["mu"],
                            "neighborhood_config": shared_config["neighborhood_config"],
                            "eps1": c1.get("eps"),   # NEW
                        },
                        "arch": "single",
                        "kalman_params": filter_config,
                        "cfar1_params": c1,       # NEW (was shared_config['cfar_config'])
                        "neighborhood_config": shared_config["neighborhood_config"],
                        "distance_tolerance": shared_config["distance_tolerance"],
                    })


    return all_configs

def create_task_generator(configs: List[Dict], video_np, gt_data, video_shape) -> Generator[Tuple, None, None]:
    """
    Yields (video_np, params, Z_SCORE_SWEEP, gt_data, video_shape)
    Params always include:
      - 'varied_param' (for CSV & grouping)
      - 'arch', 'neighborhood_config', 'distance_tolerance'
      - for gamma single:  'st1', 'cfar1_params'
      - for gamma two_stage:'st1','st2','cfar1_params','cfar2_params'
      - for kalman_mcc:    'kalman_params','cfar1_params'
    """
    for p in configs:
        yield (video_np, p, config.Z_SCORE_SWEEP, gt_data, video_shape)

def generate_reports_for_model_list(
    model_list: List[Dict], report_category_name: str,
    video_np, device, neuron_gt_data, vessel_gt_data, base_output_dir: Path
):
    """
    Helper function to generate a full suite of detailed reports for a given
    list of models, saving them into a dedicated sub-folder.
    """
    if not model_list:
        logging.warning(f"No models found for category '{report_category_name}'. Skipping detailed report generation.")
        return

    logging.info(f"--- Generating Detailed Reports for '{report_category_name}' Category ---")
    category_dir = base_output_dir / f"{report_category_name}_Reports"
    category_dir.mkdir(exist_ok=True)

    for i, model_config in enumerate(model_list):
        generators.generate_detailed_report(
            model_config, rank=i + 1, video_np=video_np, device=device,
            neuron_gt_data=neuron_gt_data, vessel_gt_data=vessel_gt_data,
            base_output_dir=category_dir
        )

def main():
    """Main execution workflow (with Kalman–MCC and expanded figure suite)."""
    setup_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Data ---
    video_np = load_video(config.INPUT_VIDEO)
    T, H, W = video_np.shape
    neuron_gt_data = parse_ground_truth_csv(config.NEURON_GT_CSV)
    vessel_gt_data = parse_ground_truth_csv(config.VESSEL_GT_CSV)
    config.BASE_OUTPUT_DIR.mkdir(exist_ok=True)

    # --- 2. Run Grid Search ---
    all_param_configs = create_param_configs()
    tasks = create_task_generator(all_param_configs, video_np, neuron_gt_data, (T, H, W))
    all_results = []
    logging.info(f"Starting grid search with {len(all_param_configs)} combinations...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=config.NUM_WORKERS) as pool:
        pbar = tqdm(total=len(all_param_configs), desc="Processing Combinations")
        for result in pool.imap_unordered(process_full_task_for_worker, tasks):
            if result:
                all_results.append(result)
            pbar.update(1)
        pbar.close()
    if not all_results:
        logging.critical("Grid search completed with no results. Exiting.")
        sys.exit(1)

    # --- 3. Analyze Results (family-wise + combined) ---
    logging.info("--- Analyzing Grid Search Results ---")
    generators.save_full_results_csv(all_results, config.BASE_OUTPUT_DIR)

    from collections import defaultdict
    by_family = defaultdict(list)
    for r in all_results:
        ftype = r['params']['varied_param']['filter_type']
        by_family[ftype].append(r)

    def tag(ft: str) -> str:
        return {"gamma": "Gamma", "kalman_mcc": "KalmanMCC"}.get(ft, ft)

    # Per-family reports
    for ftype, fam in by_family.items():
        fam_dir = config.BASE_OUTPUT_DIR / tag(ftype)
        fam_dir.mkdir(exist_ok=True)
        logging.info(f"=== Family: {tag(ftype)} ({len(fam)} configs) ===")

        top_models = analysis.find_top_k_models(fam, k=config.TOP_K_MODELS_TO_REPORT)
        balanced   = analysis.find_balanced_operating_points(
            fam, k=config.TOP_K_MODELS_TO_REPORT,
            fppi_normalization_max=config.TRUNCATION_FPPI_LIMIT
        )

        generators.save_top_models_summary(top_models, fam_dir)
        generators.save_balanced_summary(balanced, fam_dir)

        # full suites per family
        generate_reports_for_model_list(top_models,   f"{tag(ftype)}_Top_100_TPR", video_np, device, neuron_gt_data, vessel_gt_data, fam_dir)
        generate_reports_for_model_list(balanced,     f"{tag(ftype)}_Top_Balanced", video_np, device, neuron_gt_data, vessel_gt_data, fam_dir)

        # FROC + Youden per family
        plotters.plot_range_truncated_froc(fam, top_models, config.TRUNCATION_FPPI_LIMIT, fam_dir)
        if hasattr(plotters, "plot_youden_vs_fppi"):
            plotters.plot_youden_vs_fppi(balanced, config.TRUNCATION_FPPI_LIMIT, fam_dir)

        # Sensitivity: Gamma(t_decay,s_decay) or Kalman(sigma,mu)
        if ftype == "gamma":
            t_vals = sorted({p['params']['varied_param'].get('t_decay') for p in fam if 't_decay' in p['params']['varied_param']})
            s_vals = sorted({p['params']['varied_param'].get('s_decay') for p in fam if 's_decay' in p['params']['varied_param']})
            if len(t_vals) > 1 or len(s_vals) > 1:
                plotters.generate_sensitivity_plots(
                    results=fam, x_vals=t_vals, y_vals=s_vals,
                    output_dir=fam_dir, fppi_limit=config.TRUNCATION_FPPI_LIMIT,
                    x_name='t_decay', y_name='s_decay', filter_type="Gamma ST Filter"
                )
        elif ftype == "kalman_mcc":
            sig_vals = sorted({p['params']['varied_param']['sigma'] for p in fam})
            mu_vals  = sorted({p['params']['varied_param']['mu'] for p in fam})
            if len(sig_vals) > 1 or len(mu_vals) > 1:
                plotters.generate_sensitivity_plots(
                    results=fam, x_vals=sig_vals, y_vals=mu_vals,
                    output_dir=fam_dir, fppi_limit=config.TRUNCATION_FPPI_LIMIT,
                    x_name='sigma', y_name='mu', filter_type="Kalman–MCC"
                )

        # --- Gamma: diagnostics for #1 balanced model ---
        logging.info("--- Gamma: diagnostics for #1 balanced model ---")
        if balanced:
            top_params = balanced[0]["params"]

            if top_params["varied_param"]["filter_type"] != "gamma":
                logging.warning("Top balanced model is not a Gamma configuration; skipping Gamma diagnostics.")
            else:
                arch = top_params.get("arch", "single")

                # Recompute intermediates/z ONCE via the GPU pipelines (no thresholding for analysis)
                if arch == "single":
                    out = run_single_stage(video_np, {
                        "st1":     top_params["st1"],
                        "cfar1":   top_params["cfar1_params"],
                        "z_thresh": 0.0,  # we want raw z1 for plots
                    })
                    gamma_np = out["features1"]   # ST1 output
                    cfar_np  = out["z1"]          # z1 is the CFAR z-map for single-stage
                else:
                    out = run_two_stage(video_np, {
                        "st1":     top_params["st1"],
                        "cfar1":   top_params["cfar1_params"],
                        "st2":     top_params["st2"],
                        "cfar2":   top_params["cfar2_params"],
                        "z_thresh": 0.0,           # we want raw z2 for plots
                    })
                    gamma_np = out["features1"]   # keep ST1 as “middle-stage” for consistency
                    cfar_np  = out["z2"]          # z2 is the final CFAR z-map for two-stage

                # Diagnostics directory
                diag_dir = fam_dir / "Diagnostics"
                diag_dir.mkdir(exist_ok=True)

                # Generate stage-wise samples & plots (robust histogram/power functions already patched)
                analysis.generate_and_save_stage_wise_samples(video_np, gamma_np, cfar_np, neuron_gt_data, diag_dir)
                plotters.plot_stage_wise_histograms(diag_dir)
                plotters.plot_frame_wise_power(video_np, neuron_gt_data, "Raw_Video", diag_dir)
                plotters.plot_frame_wise_power(gamma_np, neuron_gt_data, tag(ftype), diag_dir)
                plotters.plot_frame_wise_power(cfar_np, neuron_gt_data, "CFAR_Z-Score", diag_dir)
        else:
            logging.warning("No balanced models available; skipping Gamma diagnostics.")

    # Combined “true best across both families”
    combined_dir = config.BASE_OUTPUT_DIR / "Combined"
    combined_dir.mkdir(exist_ok=True)
    top_all = analysis.find_top_k_models(all_results, k=config.TOP_K_MODELS_TO_REPORT)
    bal_all = analysis.find_balanced_operating_points(
        all_results, k=config.TOP_K_MODELS_TO_REPORT,
        fppi_normalization_max=config.TRUNCATION_FPPI_LIMIT
    )
    generators.save_top_models_summary(top_all, combined_dir)
    generators.save_balanced_summary(bal_all, combined_dir)
    generate_reports_for_model_list(top_all, "Combined_Top_100_TPR", video_np, device, neuron_gt_data, vessel_gt_data, combined_dir)
    generate_reports_for_model_list(bal_all, "Combined_Top_Balanced", video_np, device, neuron_gt_data, vessel_gt_data, combined_dir)
    plotters.plot_range_truncated_froc(all_results, top_all, config.TRUNCATION_FPPI_LIMIT, combined_dir)
    if hasattr(plotters, "plot_youden_vs_fppi"):
        plotters.plot_youden_vs_fppi(bal_all, config.TRUNCATION_FPPI_LIMIT, combined_dir)


    # --- 4. Global static figures (once) ---
    logging.info("--- Generating global static figures ---")
    plotters.plot_pipeline_schematic(config.BASE_OUTPUT_DIR / "fig_pipeline.png")
    plotters.plot_cfar_ring_illustration(config.BASE_OUTPUT_DIR / "fig_cfar_ring.png")
    plotters.plot_cfar_heatmap_2d(config.BASE_OUTPUT_DIR / "fig_cfar_heatmap_2d.png")
    def _pick_cfar_kernel_size():
    # Prefer the first CFAR1 config, then CFAR2, then (legacy) SHARED_PARAMS['cfar_config'], else default.
        for arr_name in ("CFAR1_PARAMS", "CFAR2_PARAMS"):
            arr = getattr(config, arr_name, None)
            if isinstance(arr, (list, tuple)) and arr:
                ks = arr[0].get("kernel_size")
                if isinstance(ks, (list, tuple)) and len(ks) == 2:
                    return tuple(ks)
        cf_shared = getattr(config, "SHARED_PARAMS", {}).get("cfar_config")
        if isinstance(cf_shared, (list, tuple)) and cf_shared:
            ks = cf_shared[0].get("kernel_size")
            if isinstance(ks, (list, tuple)) and len(ks) == 2:
                return tuple(ks)
        return (23, 23)

    kernel_size_for_plot = _pick_cfar_kernel_size()

    plotters.plot_cfar_grid_heatmap(
        config.BASE_OUTPUT_DIR / "fig_cfar_grid_heatmap.png",
        kernel_size=kernel_size_for_plot,
    )
    if hasattr(plotters, "plot_cfar_guard_ring_cartoon"):
        plotters.plot_cfar_guard_ring_cartoon(
            config.BASE_OUTPUT_DIR / "fig_cfar_guard_ring.png"
        )

    # --- 5. Family-wise + combined analysis (reports + figs) ---
    from collections import defaultdict

    def tag(ft: str) -> str:
        return {"gamma": "Gamma", "kalman_mcc": "KalmanMCC"}.get(ft, ft)

    # Bucket results by preprocessor family
    by_family = defaultdict(list)
    for r in all_results:
        ftype = r["params"]["varied_param"]["filter_type"]
        by_family[ftype].append(r)

    for ftype, fam in by_family.items():
        fam_name = tag(ftype)
        fam_dir = config.BASE_OUTPUT_DIR / fam_name
        fam_dir.mkdir(exist_ok=True)
        logging.info(f"=== Family: {fam_name} ({len(fam)} configs) ===")

        # Rank within this family
        top_models = analysis.find_top_k_models(fam, k=config.TOP_K_MODELS_TO_REPORT)
        balanced   = analysis.find_balanced_operating_points(
            fam, k=config.TOP_K_MODELS_TO_REPORT,
            fppi_normalization_max=config.TRUNCATION_FPPI_LIMIT
        )

        # Save CSVs and full report suites (inside family dir)
        generators.save_top_models_summary(top_models, fam_dir)
        generators.save_balanced_summary(balanced, fam_dir)
        generate_reports_for_model_list(
            model_list=top_models,
            report_category_name=f"{fam_name}_Top_100_TPR",
            video_np=video_np, device=device,
            neuron_gt_data=neuron_gt_data, vessel_gt_data=vessel_gt_data,
            base_output_dir=fam_dir,
        )
        generate_reports_for_model_list(
            model_list=balanced,
            report_category_name=f"{fam_name}_Top_Balanced",
            video_np=video_np, device=device,
            neuron_gt_data=neuron_gt_data, vessel_gt_data=vessel_gt_data,
            base_output_dir=fam_dir,
        )

        # Family FROC + Youden
        plotters.plot_range_truncated_froc(
            fam, top_models, config.TRUNCATION_FPPI_LIMIT, fam_dir
        )
        if hasattr(plotters, "plot_youden_vs_fppi"):
            plotters.plot_youden_vs_fppi(
                balanced, config.TRUNCATION_FPPI_LIMIT, fam_dir
            )

        # Family sensitivity heatmaps
        if ftype == "gamma":
            t_vals = sorted({p['params']['varied_param'].get('t_decay') for p in fam if 't_decay' in p['params']['varied_param']})
            s_vals = sorted({p['params']['varied_param'].get('s_decay') for p in fam if 's_decay' in p['params']['varied_param']})
            if len(t_vals) > 1 or len(s_vals) > 1:
                plotters.generate_sensitivity_plots(
                    results=fam, x_vals=t_vals, y_vals=s_vals,
                    output_dir=fam_dir, fppi_limit=config.TRUNCATION_FPPI_LIMIT,
                    x_name="t_decay", y_name="s_decay", filter_type="Gamma ST Filter",
                )
                if hasattr(plotters, "plot_gamma_profiles"):
                    plotters.plot_gamma_profiles(fam_dir, s_vals, t_vals)
        elif ftype == "kalman_mcc":
            sig_vals = sorted({p['params']['varied_param']['sigma'] for p in fam})
            mu_vals  = sorted({p['params']['varied_param']['mu']    for p in fam})
            if len(sig_vals) > 1 or len(mu_vals) > 1:
                plotters.generate_sensitivity_plots(
                    results=fam, x_vals=sig_vals, y_vals=mu_vals,
                    output_dir=fam_dir, fppi_limit=config.TRUNCATION_FPPI_LIMIT,
                    x_name="sigma", y_name="mu", filter_type="Kalman–MCC",
                )

                # Top-performer diagnostics for this family (use best 'balanced' model if available)
        if balanced:
            logging.info(f"--- {fam_name}: diagnostics for #1 balanced model ---")
            top_params = balanced[0]["params"]

            if top_params["varied_param"]["filter_type"] == "gamma":
                # Use GPU pipelines; no built-in CFAR threshold so we get raw z1/z2 for plots
                arch = top_params.get("arch", "single")
                if arch == "single":
                    out = run_single_stage(video_np, {
                        "st1":     top_params["st1"],
                        "cfar1":   top_params["cfar1_params"],
                        "z_thresh": 0.0,
                    })
                    feat_np = out["features1"]     # ST1
                    cfar_np = out["z1"]           # z1 (single-stage)
                    middle_label = "Gamma ST"
                else:
                    out = run_two_stage(video_np, {
                        "st1":     top_params["st1"],
                        "cfar1":   top_params["cfar1_params"],
                        "st2":     top_params["st2"],
                        "cfar2":   top_params["cfar2_params"],
                        "z_thresh": 0.0,
                    })
                    feat_np = out["features1"]     # keep ST1 as middle for consistency
                    cfar_np = out["z2"]           # z2 (two-stage)
                    middle_label = "Gamma ST"

            else:
                # Kalman–MCC legacy path (still supported)
                feat_np, feat_label = get_feature_map(video_np, top_params, device)
                features_gpu = torch.from_numpy(feat_np.astype(np.float32)).to(device).unsqueeze(1)
                cfar_dict = top_params.get("cfar1_params", top_params.get("cfar_params", {}))
                cfar = CFAR({**cfar_dict, "T": 0})
                z_map, *_ = cfar(features_gpu)
                cfar_np = z_map.squeeze().cpu().numpy()
                middle_label = "MCC Residual"

            # Diagnostics directory
            diag_dir = fam_dir / "Diagnostics"
            diag_dir.mkdir(exist_ok=True)

            # Panels & plots (histogram/power helpers already made robust)
            analysis.generate_and_save_stage_wise_samples(video_np, feat_np, cfar_np, neuron_gt_data, diag_dir)
            plotters.plot_stage_wise_histograms(diag_dir)
            plotters.plot_frame_wise_power(video_np, neuron_gt_data, "Raw_Video", diag_dir)
            plotters.plot_frame_wise_power(feat_np, neuron_gt_data, middle_label, diag_dir)
            plotters.plot_frame_wise_power(cfar_np, neuron_gt_data, "CFAR_Z-Score", diag_dir)

            # PSD comparison only makes sense for Gamma
            if top_params["varied_param"]["filter_type"] == "gamma" and hasattr(plotters, "plot_psd_comparison"):
                plotters.plot_psd_comparison(video_np, feat_np, neuron_gt_data, fam_dir)

            # Error maps & qualitative panels (use report already generated for rank #1)
            rank1_report_dir = fam_dir / f"{fam_name}_Top_Balanced_Reports" / "Rank_1_Report"
            if rank1_report_dir.exists():
                plotters.plot_event_geometry(rank1_report_dir, fam_dir)
            plotters.plot_error_maps(fam_dir / f"{fam_name}_Top_Balanced_Reports", video_np.shape)
            plotters.plot_qualitative_montage(fam_dir, neuron_gt_data)
            plotters.plot_full_frame_detections(fam_dir, neuron_gt_data)



    # Combined “true best” across both families
    combined_dir = config.BASE_OUTPUT_DIR / "Combined"
    combined_dir.mkdir(exist_ok=True)
    top_all = analysis.find_top_k_models(all_results, k=config.TOP_K_MODELS_TO_REPORT)
    bal_all = analysis.find_balanced_operating_points(
        all_results, k=config.TOP_K_MODELS_TO_REPORT,
        fppi_normalization_max=config.TRUNCATION_FPPI_LIMIT
    )
    generators.save_top_models_summary(top_all, combined_dir)
    generators.save_balanced_summary(bal_all, combined_dir)
    generate_reports_for_model_list(top_all, "Combined_Top_100_TPR", video_np, device, neuron_gt_data, vessel_gt_data, combined_dir)
    generate_reports_for_model_list(bal_all, "Combined_Top_Balanced",   video_np, device, neuron_gt_data, vessel_gt_data, combined_dir)
    plotters.plot_range_truncated_froc(all_results, top_all, config.TRUNCATION_FPPI_LIMIT, combined_dir)
    if hasattr(plotters, "plot_youden_vs_fppi"):
        plotters.plot_youden_vs_fppi(bal_all, config.TRUNCATION_FPPI_LIMIT, combined_dir)

    # === After the existing combined summaries/plots are written ===
    restrictive_top = run_restrictive_id_coverage_experiment(
    all_results=all_results,
    video_np=video_np,
    neuron_gt_data=neuron_gt_data,
    device=device,
    base_output_dir=combined_dir,
    top_k=config.TOP_K_MODELS_TO_REPORT,  # reuse your existing K
    )

    # Generate the SAME detailed diagnostics you get for other leaderboards
    generate_reports_for_model_list(
        model_list=restrictive_top,
        report_category_name="Restrictive_UniqueID_Coverage",
        video_np=video_np,
        device=device,
        neuron_gt_data=neuron_gt_data,
        vessel_gt_data=vessel_gt_data,
        base_output_dir=combined_dir,
    )



    logging.info("--- All reporting phases complete. ---")

if __name__ == "__main__":
    main()