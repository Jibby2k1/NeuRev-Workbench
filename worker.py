# worker.py
"""Legacy grid-search worker preserved for reproducibility.

New maintained work belongs under :mod:`neurobench`; see
``docs/CODEBASE_NAVIGATION.md`` before extending this compatibility surface.
"""
import gc
import logging
import numpy as np
from typing import Tuple, List, Dict
import torch

from core.pipelines import run_single_stage, run_two_stage  # NEW
from core.filters import get_feature_map
from core.detection import CFAR, apply_neighborhood_filter
from evaluation.metrics import calculate_froc_point_metrics, metric_value
from evaluation import metrics
from utils import extract_pixel_detections

def process_full_task_for_worker(task_tuple: Tuple) -> Dict:
    """
    Args (unchanged):
      task_tuple = (video_np, params, z_sweep, gt_data, video_shape)

    Returns (schema unchanged, with one new field):
      {
        "params": params,
        "roc_points": [ { "z_score": float, "TPR": float, "FPPI": float, ...}, ... ],
        "restrictive_point":  # NEW: dict or None
            { "z_score": float, "TPR": float, "FPPI": float,
              "matched_unique_ids": [ids...], "unique_id_coverage": float }
      }
    """
    video_np, params, z_sweep, gt_data, video_shape = task_tuple
    T, H, W = video_shape

    result = {"params": params, "roc_points": []}
    restrictive_point = None

    try:
        # -------- 1) Build a z-score map ONCE per config (no threshold applied) --------
        fam  = str(params.get("varied_param", {}).get("filter_type", "")).lower()
        arch = str(params.get("arch", "single")).lower()

        if fam == "gamma":
            if arch in ("single", "single_stage", "single-stage"):
                # get raw z1 (no threshold)
                out = run_single_stage(
                    video_np=video_np,
                    p={"st1": params["st1"], "cfar1": params["cfar1_params"], "z_thresh": 0.0},
                    device=None,
                )
                z_map = out["z1"]  # (T,H,W)
            elif "two" in arch:
                # get raw z2 (no threshold)
                out = run_two_stage(
                    video_np=video_np,
                    p={
                        "st1": params["st1"],
                        "cfar1": params["cfar1_params"],
                        "st2": params["st2"],
                        "cfar2": params["cfar2_params"],
                        "z_thresh": 0.0,
                    },
                    device=None,
                )
                z_map = out["z2"]  # (T,H,W)
            else:
                logging.info(f"[worker] Unsupported Gamma arch={arch}; skipping config.")
                return result

        elif fam == "kalman_mcc":
            # residual features -> CFAR(T=0) to produce a z-score map
            features_np, _ = get_feature_map(video_np, params, device=None)  # (T,H,W)
            x_tchw = torch.from_numpy(features_np.astype(np.float32)).unsqueeze(1)  # (T,1,H,W)
            cfar_params = params.get("cfar1_params", params.get("cfar_params", {}))
            cfar = CFAR({**cfar_params, "T": 0})
            with torch.no_grad():
                z, *_ = cfar(x_tchw.float())
            z_map = z.squeeze(1).cpu().numpy()  # (T,H,W)

        else:
            logging.info(f"[worker] Unknown filter family '{fam}'; skipping config.")
            return result

        # -------- 2) Sweep thresholds high->low; record ROC and first full-ID point --------
        neigh_size, neigh_k = params["neighborhood_config"]
        distance_tol = float(params["distance_tolerance"])

        coverage_found = False
        sweep = sorted([float(z) for z in z_sweep], reverse=True)

        for z in sweep:
            # binarize
            mask_bool = (z_map >= z)

            # apply neighborhood filter (same order as elsewhere: (k, size))
            mask_t = torch.from_numpy(mask_bool)
            mask_t = apply_neighborhood_filter(mask_t, neigh_k, neigh_size)
            mask_np = mask_t.cpu().numpy().astype(np.uint8)

            # detections at this operating point
            detections = extract_pixel_detections(mask_np)

            # we only ask for per-event matches until we've found full coverage
            need_matches = not coverage_found
            m = metrics.calculate_froc_point_metrics(
                gt_data=gt_data,
                detections=detections,
                video_shape=(T, H, W),
                distance_tolerance=distance_tol,
                return_matches=need_matches,
            )

            # append standard ROC point (unchanged fields you already use downstream)
            pt = {
                "z_score": float(z),
                "TPR": float(metric_value(m, "TPR", 0.0)),
                "FPPI": float(metric_value(m, "FPPI", 0.0)),
                "TP": int(metric_value(m, "TP", 0)),
                "FP": int(metric_value(m, "FP", 0)),
                "FN": int(metric_value(m, "FN", 0)),
            }
            # Legacy lowercase copies keep older plotting/reporting code readable
            # while canonical uppercase metrics propagate through new outputs.
            pt["tpr"] = pt["TPR"]
            pt["fppi"] = pt["FPPI"]
            result["roc_points"].append(pt)

            # record the first (max-z) full unique-ID coverage point
            if need_matches and m.get("full_unique_id_coverage", False):
                restrictive_point = {
                    **pt,
                    "matched_unique_ids": m.get("matched_unique_ids", []),
                    "unique_id_coverage": float(m.get("unique_id_coverage", 0.0)),
                }
                coverage_found = True

        # add the new field (harvested later by generators.run_restrictive_id_coverage_experiment)
        result["restrictive_point"] = restrictive_point
        return result

    except Exception as e:
        logging.exception(f"[worker] Exception for params={params.get('varied_param', params)}: {e}")
        result["restrictive_point"] = None
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
