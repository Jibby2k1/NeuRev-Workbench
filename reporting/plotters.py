# reporting/plotters.py
"""
Functions for creating and saving all visualizations and figures for the analysis.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import scipy.stats
import numpy as np
import pandas as pd
import tifffile
import config
import json

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, FancyBboxPatch, ArrowStyle, Circle
    from matplotlib.lines import Line2D
except ImportError:
    plt = None
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

from evaluation.metrics import calculate_truncated_auc
from core.filters import full_kalman_mcc_filter, full_kalman_mcc_filter_gpu_batches, specify_gamma_kernel, generate_gamma_kernel

# Note: All functions check if 'plt' was imported successfully.

# --- robust metric access (upper/lower case) ---
def _TPR(p):
    v = p.get("TPR", p.get("tpr"))
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

def _FPPI(p):
    v = p.get("FPPI", p.get("fppi"))
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

def _resolve_rank1_report_dir(base_output_dir: Path) -> Optional[Path]:
    """
    Look for the Rank_1_Report directory under either:
      - <base>/Top_Balanced_Reports/Rank_1_Report
      - <base>/*Top_Balanced_Reports/Rank_1_Report   (family-prefixed)
    Return the first that exists, else None.
    """
    candidates = [
        base_output_dir / "Top_Balanced_Reports" / "Rank_1_Report",
        *[(p / "Rank_1_Report") for p in base_output_dir.glob("*Top_Balanced_Reports")]
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def plot_event_geometry(report_dir: Path, output_dir: Path):
    """
    Plots a histogram of the ground truth event durations with integer bins.
    """
    if not plt or not pd:
        return
    duration_csv_path = report_dir / "event_duration_stats.csv"
    if not duration_csv_path.exists():
        logging.warning(f"Event duration stats not found at {duration_csv_path}, skipping geometry plot.")
        return

    logging.info("📊 Generating event geometry plot (integer bins)...")
    df = pd.read_csv(duration_csv_path)
    if df.empty or 'duration' not in df.columns:
        logging.warning("Duration CSV is empty or malformed; skipping.")
        return

    durations = df['duration'].astype(int).to_numpy()
    vals, counts = np.unique(durations, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(vals, counts, width=0.9, edgecolor='black')
    plt.title("Distribution of Ground Truth Event Durations")
    plt.xlabel("Duration (frames)")
    plt.ylabel("Number of Events")

    max_d = int(vals.max())
    if max_d <= 50:
        xticks = np.arange(1, max_d + 1)
    else:
        step = max(1, max_d // 20)
        xticks = np.arange(1, max_d + 1, step)
    plt.xticks(xticks)
    plt.grid(True, axis='y', ls='--', alpha=0.6)
    plt.tight_layout()
    output_path = output_dir / "fig_event_duration_histogram.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"✅ Event duration histogram saved to {output_path}")

def plot_full_frame_detections(
    base_output_dir: Path, neuron_gt_data: Dict
):
    """
    Generates a full-scale image for each frame containing a ground truth event,
    overlaying all GT locations and all model detections for that frame.
    """
    if not plt: return
    rank1_report_dir = _resolve_rank1_report_dir(base_output_dir)
    if rank1_report_dir is None:
        logging.warning(f"Could not find Rank_1_Report under {base_output_dir}; skipping full-frame plots.")
        return
    diag_video_path = rank1_report_dir / "diagnostic_video.tif"


    if not diag_video_path.exists():
        logging.warning(f"Diagnostic video not found at {diag_video_path}, skipping full-frame plots.")
        return

    logging.info("🖼️  Generating full-frame detection plots...")
    diag_video = tifffile.imread(diag_video_path)

    # Accept 4, 5, or 6 panels (single-stage=4, legacy=5, two-stage=6)
    if diag_video.ndim != 3:
        logging.error(f"Diagnostic video has an unexpected shape {diag_video.shape}. Skipping full-frame plots.")
        return
    W_total = diag_video.shape[2]
    n_panels = 6 if (W_total % 6 == 0) else (5 if (W_total % 5 == 0) else (4 if (W_total % 4 == 0) else None))
    if n_panels is None:
        logging.error(f"Diagnostic video width {W_total} not divisible by 4, 5, or 6. Skipping.")
        return

    W = W_total // n_panels
    raw_panel        = diag_video[:, :, 0*W:1*W]
    final_mask_panel = diag_video[:, :, (n_panels-1)*W:n_panels*W]

    # Group ground truth events by frame
    gt_by_frame = defaultdict(list)
    for frame, points in neuron_gt_data.items():
        for x, y, _ in points:
            gt_by_frame[frame].append((x, y))

    output_folder = base_output_dir / "Full_Frame_Detections"
    output_folder.mkdir(exist_ok=True)

    # Generate a plot for each frame that has a ground truth event
    for frame_idx, gt_points in gt_by_frame.items():
        fig, ax = plt.subplots(figsize=(10, 10 * (W/W) if W > 0 else 1))
        
        raw_image = raw_panel[frame_idx]
        detection_mask = final_mask_panel[frame_idx]

        ax.imshow(raw_image, cmap='gray', vmin=np.percentile(raw_image, 5), vmax=np.percentile(raw_image, 99.5))

        # Plot all detections on the frame
        if np.any(detection_mask):
            ax.contour(detection_mask, levels=[0.5], colors='red', linewidths=0.8, alpha=0.9)

        # Plot all ground truth points on the frame
        for x, y in gt_points:
            gt_circle = Circle((x, y), radius=5, color='cyan', fill=False, lw=1.2)
            ax.add_patch(gt_circle)

        # Create a custom legend
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Model Detection'),
            Line2D([0], [0], marker='o', color='w', label='Ground Truth',
                   markerfacecolor='none', markeredgecolor='cyan', markersize=10, markeredgewidth=1.2)
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='black', edgecolor='white', labelcolor='white')

        ax.set_title(f"Full-Frame Detections - Frame {frame_idx}")
        ax.axis('off')
        plt.tight_layout()
        output_path = output_folder / f"fig_full_frame_detection_frame_{frame_idx}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    logging.info(f"✅ Full-frame detection plots saved to {output_folder}")

def plot_qualitative_montage(
    base_output_dir: Path, neuron_gt_data: Dict, patch_size: int = 48
):
    """
    Generates a montage of qualitative results for each ground truth event,
    ensuring all patches are of a uniform size.
    """
    if not plt: return
    rank1_report_dir = _resolve_rank1_report_dir(base_output_dir)
    if rank1_report_dir is None:
        logging.warning(f"Could not find Rank_1_Report under {base_output_dir}; skipping montage.")
        return
    diag_video_path = rank1_report_dir / "diagnostic_video.tif"


    if not diag_video_path.exists():
        logging.warning(f"Diagnostic video not found at {diag_video_path}, skipping montage.")
        return

    logging.info("🖼️  Generating qualitative detection montage...")
    diag_video = tifffile.imread(diag_video_path)

    if diag_video.ndim != 3:
        logging.error(f"Diagnostic video has an unexpected shape {diag_video.shape}. Skipping montage.")
        return

    W_total = diag_video.shape[2]
    n_panels = 6 if (W_total % 6 == 0) else (5 if (W_total % 5 == 0) else (4 if (W_total % 4 == 0) else None))
    if n_panels is None:
        logging.error(f"Diagnostic video width {W_total} not divisible by 4, 5, or 6. Skipping montage.")
        return

    W = W_total // n_panels
    raw_panel        = diag_video[:, :, 0*W:1*W]
    zscore_panel     = diag_video[:, :, (n_panels-2)*W:(n_panels-1)*W]  # z1 (single) or z2 (two-stage)
    final_mask_panel = diag_video[:, :, (n_panels-1)*W:n_panels*W]


    gt_events = {}
    for frame, points in sorted(neuron_gt_data.items()):
        for x, y, gt_id in points:
            if gt_id not in gt_events:
                gt_events[gt_id] = {'frame': frame, 'x': int(x), 'y': int(y)}

    num_events = len(gt_events)
    if num_events == 0:
        logging.warning("No ground truth events found to generate montage.")
        return

    fig, axes = plt.subplots(num_events, 3, figsize=(9, 2.5 * num_events),
                             gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    if num_events == 1:
        axes = np.array([axes])

    fig.suptitle("Qualitative Detections for Each Labeled Neuron", fontsize=16, y=0.98)

    for i, (gt_id, event) in enumerate(gt_events.items()):
        t, cx, cy = event['frame'], event['x'], event['y']
        half_patch = patch_size // 2
        
        y_slice = slice(max(0, cy - half_patch), min(raw_panel.shape[1], cy + half_patch))
        x_slice = slice(max(0, cx - half_patch), min(raw_panel.shape[2], cx + half_patch))
        
        raw_patch_data = raw_panel[t, y_slice, x_slice]

        # --- PATCH SIZE NORMALIZATION LOGIC ---
        # Create a full-sized black canvas for each patch
        h, w = raw_patch_data.shape
        y_paste = (patch_size - h) // 2
        x_paste = (patch_size - w) // 2
        
        # Function to create and paste patch onto a canvas
        def create_padded_patch(data, dtype, fill_value=0):
            canvas = np.full((patch_size, patch_size), fill_value, dtype=dtype)
            canvas[y_paste:y_paste+h, x_paste:x_paste+w] = data
            return canvas

        # Create padded patches
        raw_patch = create_padded_patch(raw_patch_data, raw_panel.dtype, fill_value=np.percentile(raw_patch_data, 5))
        zscore_patch = create_padded_patch(zscore_panel[t, y_slice, x_slice], zscore_panel.dtype)
        mask_patch = create_padded_patch(final_mask_panel[t, y_slice, x_slice], final_mask_panel.dtype)
        # --- END PATCH NORMALIZATION ---

        # Plot Raw + GT
        axes[i, 0].imshow(raw_patch, cmap='gray')
        gt_circle = Circle((patch_size/2, patch_size/2), radius=3, color='cyan', fill=False, lw=1.5)
        axes[i, 0].add_patch(gt_circle)
        # Change "Event ID" to "Neuron ID"
        axes[i, 0].set_ylabel(f"Neuron ID: {gt_id}\nFrame: {t}", rotation=0, labelpad=45, ha='right', va='center')

        # Plot Z-Score
        axes[i, 1].imshow(zscore_patch, cmap='magma', vmin=0)

        # Plot Raw + Detection
        axes[i, 2].imshow(raw_patch, cmap='gray')
        axes[i, 2].contour(mask_patch, levels=[0.5], colors='red', linewidths=1.5)

        if i == 0:
            axes[i, 0].set_title("Raw Video + GT")
            axes[i, 1].set_title("CFAR Z-Score Map")
            axes[i, 2].set_title("Raw Video + Detection")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = base_output_dir / "fig_qualitative_montage.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"✅ Qualitative montage saved to {output_path}")

def plot_psd_comparison(raw_np: np.ndarray, gamma_np: np.ndarray, gt_data: Dict, output_dir: Path, patch_size: int = 32, num_patches: int = 100):
    if not plt: return
    logging.info("🔬 Generating Power Spectral Density (PSD) comparison plot...")

    H, W = raw_np.shape[1], raw_np.shape[2]
    if H <= patch_size or W <= patch_size:
        logging.warning(
            f"Video dimensions ({H}x{W}) are too small for the requested patch size of {patch_size}. "
            f"Skipping PSD plot."
        )
        return

    results = {'raw': [], 'gamma': []}
    gt_coords_by_frame = defaultdict(list)
    
    for frame_idx, points in gt_data.items():
        for x, y, _ in points:
            if 0 <= int(y) < H and 0 <= int(x) < W:
                gt_coords_by_frame[frame_idx].append((int(x), int(y)))

    patches_collected = 0
    attempts = 0
    while patches_collected < num_patches and attempts < num_patches * 10:
        t = np.random.randint(0, raw_np.shape[0])
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)
        attempts += 1

        frame_gt_mask = np.zeros((H, W), dtype=bool)
        if t in gt_coords_by_frame:
            for x_gt, y_gt in gt_coords_by_frame[t]:
                min_y, max_y = max(0, y_gt - 5), min(H, y_gt + 6)
                min_x, max_x = max(0, x_gt - 5), min(W, x_gt + 6)
                frame_gt_mask[min_y:max_y, min_x:max_x] = True
        
        if not np.any(frame_gt_mask[y:y+patch_size, x:x+patch_size]):
            processed_patches = {}
            for key, data_np in [('raw', raw_np), ('gamma', gamma_np)]:
                patch = data_np[t, y:y+patch_size, x:x+patch_size].astype(np.float64)
                if patch.std() < 1e-9: continue  # Skip flat patches

                f_transform = np.fft.fft2(patch)
                f_transform_shifted = np.fft.fftshift(f_transform)
                psd_2d = np.abs(f_transform_shifted)**2

                cy, cx = psd_2d.shape[0] // 2, psd_2d.shape[1] // 2
                y_coords, x_coords = np.ogrid[-cy:cy, -cx:cx]
                radius_map = np.sqrt(x_coords**2 + y_coords**2)
                max_radius = int(radius_map.max())
                
                if max_radius < 1: continue

                freq_bins = np.arange(1, max_radius + 1)
                radial_psd, _, _ = scipy.stats.binned_statistic(
                    radius_map.ravel(), psd_2d.ravel(), statistic='mean', bins=freq_bins
                )
                processed_patches[key] = radial_psd

            if 'raw' in processed_patches and 'gamma' in processed_patches:
                results['raw'].append(processed_patches['raw'])
                results['gamma'].append(processed_patches['gamma'])
                patches_collected += 1

    if not results['raw'] or not results['gamma']:
        logging.warning("Could not generate sufficient data for PSD plot after %d attempts.", attempts)
        return

    min_len = min(len(r) for r in results['raw'] + results['gamma'])
    trimmed_raw = [r[:min_len] for r in results['raw']]
    trimmed_gamma = [r[:min_len] for r in results['gamma']]

    avg_psd_raw = np.nanmean(np.array(trimmed_raw), axis=0)
    avg_psd_gamma = np.nanmean(np.array(trimmed_gamma), axis=0)
    
    if np.all(np.isnan(avg_psd_raw)) or np.all(np.isnan(avg_psd_gamma)):
        logging.warning("PSD data is all NaN. Cannot generate plot.")
        return
        
    freqs = np.arange(1, len(avg_psd_raw) + 1)

    plt.figure(figsize=(7, 5))
    plt.loglog(freqs, avg_psd_raw, label="Raw Video Background", color='cornflowerblue')
    plt.loglog(freqs, avg_psd_gamma, label="After Gamma ST Filter", color='orangered')
    plt.title("Spatial Power Spectrum of Background")
    plt.xlabel("Spatial Frequency (cycles/patch)")
    plt.ylabel("Power (a.u.)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / "fig_psd_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"✅ PSD comparison plot saved to {output_path}")

def plot_pipeline_schematic(output_path: Path):
    """Generates and saves a schematic of the processing pipeline."""
    if not plt: return
    logging.info(f"Generating pipeline schematic...")
    
    def box(ax, xy, text):
        patch = FancyBboxPatch(xy, 1.9, 0.8, boxstyle="round,pad=0.02,rounding_size=0.06", ec='k', fc='white', zorder=2)
        ax.add_patch(patch)
        ax.text(xy[0] + 0.95, xy[1] + 0.4, text, ha='center', va='center', fontsize=10, zorder=3)

    def arrow(ax, x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=ArrowStyle("Simple", head_width=6, head_length=8), lw=1.2, zorder=1))

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis('off')
    xs = [0.2, 2.4, 4.6, 6.8, 9.0]
    labels = ["Raw Video", "Gamma ST Filter", "CFAR z-score", "Neighborhood\nFilter (opt.)", "Detections &\nMetrics"]
    
    for x, l in zip(xs, labels):
        box(ax, (x, 0.6), l)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + 1.9, 1.0, xs[i+1], 1.0)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight') # Use bbox_inches
    plt.close()
    logging.info(f"✅ Pipeline schematic saved to {output_path}")

def plot_range_truncated_froc(all_results: List[Dict], top_k_models: List[Dict], truncation_limit: float, output_dir: Path):
    """Plots the FROC curves for the top K models."""
    if not plt or not top_k_models: return
    logging.info(f"Generating Top-K FROC plot...")
    
    plt.figure(figsize=(7, 6))
    
    for model in top_k_models:
        full_result = next((r for r in all_results if r['params'] == model['params']), None)
        if full_result and full_result.get('roc_points'):
            roc_points = sorted(
                [p for p in full_result.get('roc_points', []) if _FPPI(p) is not None and _TPR(p) is not None],
                key=lambda p: _FPPI(p)
            )
            fppi = [_FPPI(p) for p in roc_points]
            tpr  = [_TPR(p)  for p in roc_points]
            plt.plot(fppi, tpr, alpha=0.4, lw=1.5)

    tops = [(_FPPI(m['point']), _TPR(m['point'])) for m in top_k_models if 'point' in m]
    tops = [(f, t) for f, t in tops if f is not None and t is not None]
    top_fppi = [f for f, _ in tops]
    top_tpr  = [t for _, t in tops]
    plt.scatter(top_fppi, top_tpr, s=40, zorder=3, c='red', label=f'Top {len(top_k_models)} Optimal Points')
    
    if top_fppi:
        plt.scatter(top_fppi[0], top_tpr[0], s=80, zorder=4, facecolors='none', edgecolors='gold', lw=2, label='Best Model')

    plt.xlabel("False Positives Per Image (FPPI)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"Range-Truncated FROC (Top {len(top_k_models)} Models)")
    plt.grid(True, ls='--', alpha=0.5)
    plt.xlim(0, truncation_limit)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / "fig_froc_top_k.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"✅ Top-K FROC plot saved to {output_path}")

def generate_sensitivity_plots(results: List[Dict], x_vals: List, y_vals: List, output_dir: Path, fppi_limit: float, x_name: str, y_name: str, filter_type: str):
    """Generates and saves sensitivity heatmaps for hyperparameter sweeps."""
    if not plt or not x_vals or not y_vals or not results: return
    logging.info(f"🔬 Generating sensitivity plots for {filter_type} ({y_name} vs {x_name})...")
    
    aucs = np.full((len(x_vals), len(y_vals)), np.nan)
    distances = np.full((len(x_vals), len(y_vals)), np.nan)
    
    for res in results:
        p = res['params']
        try:
            varied_p = p.get('varied_param', p)
            x_idx = x_vals.index(varied_p[x_name])
            y_idx = y_vals.index(varied_p[y_name])
            aucs[x_idx, y_idx] = calculate_truncated_auc(res['roc_points'], fppi_limit)
            def _d(pt):
                f = _FPPI(pt); t = _TPR(pt)
                return np.inf if (f is None or t is None) else np.sqrt(f*f + (1 - t)**2)
            min_dist = min((_d(pt) for pt in res.get('roc_points', [])), default=np.inf)

            distances[x_idx, y_idx] = min_dist
        except (ValueError, KeyError):
            continue
            
    # Plot AUC heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(aucs.T, origin='lower', aspect='auto', cmap='viridis', interpolation='bilinear')
    ax.set_xticks(np.arange(len(x_vals))); ax.set_xticklabels([f"{v:.3f}" for v in x_vals], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_vals))); ax.set_yticklabels([f"{v:.3f}" for v in y_vals])
    ax.set_xlabel(x_name); ax.set_ylabel(y_name)
    ax.set_title(f"AUC Sensitivity ({filter_type})")
    fig.colorbar(im, ax=ax, label=f"Truncated AUC (FPPI < {fppi_limit:.1f})")
    plt.tight_layout()
    plt.savefig(output_dir / f"sensitivity_{filter_type}_auc.png", dpi=300)
    plt.close()

def plot_stage_wise_histograms(diag_dir: Path):
    """
    Plot histograms for key stages using the normalized diagnostic video.
    Robust to 4/5/6 panels and to NaN/inf values.

    Panels (width-stacked) expected in diagnostic_video.tif:
      - single-stage (4): raw | ST1 | z1 | final
      - legacy (5):      raw | ST  | z  | CFARmask | final
      - two-stage (6):   raw | ST1 | z1 | ST2 | z2 | final

    Outputs:
      - stage_histograms.png (overlay of raw, ST1, [ST2 if present])
      - cfar_z_hist.png      (z1 or z2, whichever is penultimate)
    """
    import logging
    import numpy as np
    import tifffile
    import matplotlib.pyplot as plt

    path = diag_dir / "diagnostic_video.tif"
    if not path.exists():
        logging.warning(f"No diagnostic_video.tif found in {diag_dir}; skipping stage-wise histograms.")
        return

    try:
        vid = tifffile.imread(path)  # (T, H, W_total), uint16 in [0, 65535]
    except Exception as e:
        logging.warning(f"Failed to read {path}: {e}")
        return

    if vid.ndim != 3:
        logging.warning(f"diagnostic_video.tif has unexpected shape {vid.shape}; skipping histograms.")
        return

    T, H, Wtot = vid.shape
    # Determine panel count: prefer 6, else 5, else 4
    n_panels = 6 if (Wtot % 6 == 0) else (5 if (Wtot % 5 == 0) else (4 if (Wtot % 4 == 0) else None))
    if n_panels is None:
        logging.warning(f"Diagnostic video width {Wtot} not divisible by 4, 5, or 6; skipping histograms.")
        return

    W = Wtot // n_panels

    # Helper to extract a panel [0..n_panels-1], convert to float in [0,1], flatten and sanitize
    def _panel_vals(idx: int) -> np.ndarray:
        arr = vid[:, :, idx * W:(idx + 1) * W].astype(np.float32) / 65535.0
        vals = arr.ravel()
        # Sanitize: drop NaN/Inf
        vals = vals[np.isfinite(vals)]
        # If nothing usable, return empty
        return vals

    # Map panels
    raw_vals = _panel_vals(0)
    st1_vals = _panel_vals(1)

    # Penultimate is always z (z1 in single/legacy; z2 in two-stage)
    z_vals = _panel_vals(n_panels - 2)

    # ST2 exists only if we have 6 panels
    st2_vals = _panel_vals(3) if n_panels == 6 else None

    # ---- Plot overlay of “image-like” stages (raw, ST1, [ST2]) ----
    plt.figure(figsize=(8, 5), dpi=140)

    def _plot_hist(vals, label, alpha=0.6):
        if vals is None or vals.size == 0:
            logging.info(f"Skipping histogram for {label}: no finite data.")
            return False
        # guard against degenerate all-equal values
        vmin, vmax = np.min(vals), np.max(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            logging.info(f"Skipping histogram for {label}: degenerate or non-finite range.")
            return False
        plt.hist(vals, bins=80, density=True, alpha=alpha, label=label)
        return True

    any_plotted = False
    any_plotted |= _plot_hist(raw_vals, "Raw")
    any_plotted |= _plot_hist(st1_vals, "ST1")
    if st2_vals is not None:
        any_plotted |= _plot_hist(st2_vals, "ST2")

    if any_plotted:
        plt.xlabel("Normalized intensity")
        plt.ylabel("Density")
        plt.title("Stage-wise Histograms (Image-like Panels)")
        plt.legend()
        plt.tight_layout()
        out1 = diag_dir / "stage_histograms.png"
        plt.savefig(out1)
        plt.close()
        logging.info(f"Saved {out1}")
    else:
        plt.close()
        logging.info("No valid data for stage-wise (raw/ST) histograms; skipped.")

    # ---- Plot CFAR z histogram (penultimate panel) ----
    logging.info("Plotting histogram for CFAR z stage...")
    plt.figure(figsize=(8, 5), dpi=140)
    if z_vals is None or z_vals.size == 0:
        logging.info("No finite CFAR z values to plot; skipping.")
        plt.close()
        return

    vmin, vmax = np.min(z_vals), np.max(z_vals)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin == vmax:
        logging.info(f"CFAR z panel has degenerate or non-finite range [{vmin}, {vmax}]; skipping.")
        plt.close()
        return

    plt.hist(z_vals, bins=120, density=True, alpha=0.75, label="CFAR z (normalized)")
    plt.xlabel("Normalized z")
    plt.ylabel("Density")
    plt.title("Histogram — CFAR z Panel (penultimate)")
    plt.legend()
    plt.tight_layout()
    out2 = diag_dir / "cfar_z_hist.png"
    plt.savefig(out2)
    plt.close()
    logging.info(f"Saved {out2}")

def plot_error_maps(base_output_dir: Path, video_shape: Tuple):
    """Generates and saves visual reports for error analysis (FP density, FN counts)."""
    if not plt or not pd: return
    logging.info("Generating error analysis maps...")
    
    H, W = video_shape[1], video_shape[2]
    rank_1_dir = base_output_dir / "Rank_1_Report"
    fp_coords_csv = rank_1_dir / "fp_coords.csv"
    fn_per_id_csv = rank_1_dir / "fn_per_id.csv"

    # FP Density Map
    if fp_coords_csv.exists():
        fp_df = pd.read_csv(fp_coords_csv)
        heatmap = np.zeros((H, W), dtype=np.int32)
        for _, row in fp_df.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= y < H and 0 <= x < W:
                heatmap[y, x] += 1
        
        plt.figure(figsize=(6, 5))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.title("FP Density Map (Top Model)"); plt.axis('off')
        plt.colorbar(label="FP Count"); plt.tight_layout()
        output_path = base_output_dir / "fig_fp_density.png"
        plt.savefig(output_path, dpi=300); plt.close()
        logging.info(f"✅ FP density map saved to {output_path}")

    # FN per ID Bar Chart
    if fn_per_id_csv.exists():
        fn_df = pd.read_csv(fn_per_id_csv)
        if not fn_df.empty:
            fns_sorted = fn_df.sort_values(by='count', ascending=False).head(30)
            plt.figure(figsize=(12, 6))
            plt.bar(fns_sorted['id'].astype(str), fns_sorted['count'], color='tomato')
            plt.xlabel("Neuron ID"); plt.ylabel("False Negative Count")
            plt.title(f"Top {len(fns_sorted)} Neurons by False Negative Count (Top Model)")
            plt.grid(True, axis='y', ls='--', alpha=0.4)
            plt.xticks(rotation=90, fontsize=8)
            plt.tight_layout()
            output_path = base_output_dir / "fig_fn_per_id.png"
            plt.savefig(output_path, dpi=300); plt.close()
            logging.info(f"✅ FN per ID plot saved to {output_path}")

            # In reporting/plotters.py

def plot_cfar_ring_illustration(output_path: Path):
    """
    Generates and saves a pretty 3D illustration of the CFAR guard and reference rings.
    """
    if not plt: return
    logging.info("Generating 3D CFAR ring illustration...")

    H = W = 101
    cy, cx = H // 2, W // 2
    guard_radius = 9
    ref_radius = 35
    
    x = np.linspace(-cx, cx, W)
    y = np.linspace(-cy, cy, H)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X**2 + Y**2)

    Z = np.zeros_like(dist)
    sigma = 4.0
    Z += np.exp(-((dist - ref_radius)**2) / (2 * sigma**2))
    Z[dist <= guard_radius] = 0
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_title("3D CFAR Kernel Illustration", fontsize=16)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_zlabel("Weight")
    ax.view_init(elev=40, azim=-70)
    ax.grid(True)
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Reference Weight")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ 3D CFAR ring illustration saved to {output_path}")

def plot_cfar_heatmap_2d(output_path: Path):
    """
    Generates and saves a 2D heatmap of the CFAR guard and reference rings.
    """
    if not plt: return
    logging.info("Generating 2D CFAR heatmap illustration...")

    H = W = 101
    cy, cx = H // 2, W // 2
    guard_radius = 9
    ref_radius = 35
    
    x = np.linspace(-cx, cx, W)
    y = np.linspace(-cy, cy, H)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X**2 + Y**2)

    Z = np.zeros_like(dist)
    sigma = 4.0
    Z += np.exp(-((dist - ref_radius)**2) / (2 * sigma**2))
    Z[dist <= guard_radius] = 0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Z, cmap='viridis', origin='lower', extent=[-cx, cx, -cy, cy])
    ax.set_title("2D CFAR Kernel Heatmap", fontsize=16)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    fig.colorbar(im, ax=ax, label="Reference Weight")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"✅ 2D CFAR heatmap saved to {output_path}")

def plot_cfar_grid_heatmap(output_path: Path, kernel_size: Tuple[int, int] = (23, 23)):
    """
    Generates a discrete grid-based heatmap of the CFAR kernel.
    """
    if not plt: return
    logging.info(f"Generating discrete {kernel_size[0]}x{kernel_size[1]} CFAR grid heatmap...")

    H, W = kernel_size
    cy, cx = H // 2, W // 2

    # Scaled-down radii for illustration on a smaller grid
    guard_radius = 3
    ref_radius = 8
    
    y_coords, x_coords = np.ogrid[:H, :W]
    dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)

    # Use a diverging colormap where the center is negative, guard is zero, and reference is positive
    Z = np.zeros_like(dist, dtype=float)
    Z[dist > guard_radius] = np.exp(-((dist[dist > guard_radius] - ref_radius)**2) / (2 * 2.0**2))
    Z[dist <= guard_radius] = 0
    Z[cy, cx] = -1 # Mark the center pixel with a distinct negative value

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(Z, cmap='RdBu_r', interpolation='nearest')

    # Add grid lines
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    ax.set_xticks(np.arange(0, W, 2))
    ax.set_yticks(np.arange(0, H, 2))
    ax.tick_params(axis='x', rotation=90)
    
    ax.set_title(f"Discrete {H}x{W} CFAR Kernel", fontsize=16)
    fig.colorbar(im, ax=ax, label="Weight (Blue=Center, Red=Reference)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"✅ Discrete CFAR grid heatmap saved to {output_path}")

def plot_frame_wise_power(data, neuron_gt_data: Dict, label: str, diag_dir: Path):
    """
    Plot per-frame 'power' (mean of squared values) for a 3D volume (T,H,W).
    Robust to NaN/Inf and empty/degenerate inputs.
    Saves: diag_dir / f"frame_power_{label}.png"
    """
    import logging
    from pathlib import Path
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # Ensure output dir exists
    Path(diag_dir).mkdir(parents=True, exist_ok=True)

    # Accept torch or numpy; coerce to numpy float32
    if isinstance(data, torch.Tensor):
        vol = data.detach().cpu().numpy()
    else:
        vol = np.asarray(data)

    if vol.ndim != 3:
        logging.warning(f"[plot_frame_wise_power] Expected (T,H,W), got shape {vol.shape}; skipping.")
        return

    T, H, W = vol.shape
    vol = vol.astype(np.float32, copy=False)

    # Compute per-frame power = mean of squared finite values
    powers = np.empty(T, dtype=np.float64)
    powers.fill(np.nan)
    for t in range(T):
        frame = vol[t]
        finite = np.isfinite(frame)
        if finite.any():
            vals = frame[finite]
            powers[t] = float(np.mean(vals * vals))
        # else keep NaN

    finite_powers = powers[np.isfinite(powers)]
    if finite_powers.size == 0:
        logging.info(f"[plot_frame_wise_power] No finite values for '{label}'; skipping plot.")
        return

    # X axis: use only finite indices to avoid plotting NaNs
    idx = np.arange(T, dtype=np.int64)
    finite_idx = idx[np.isfinite(powers)]
    y = powers[np.isfinite(powers)]

    # Y-limits with padding; guard against constant series
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        logging.info(f"[plot_frame_wise_power] Non-finite range for '{label}'; skipping plot.")
        return
    if ymax == ymin:
        pad = 1e-6 if ymax == 0.0 else abs(ymax) * 0.05
        ymin, ymax = ymax - pad, ymax + pad
    else:
        pad = 0.05 * (ymax - ymin)
        ymin, ymax = ymin - pad, ymax + pad

    # Build figure
    plt.figure(figsize=(10, 4.5), dpi=140)
    plt.plot(finite_idx, y, linewidth=1.2, label=label)

    # Overlay GT frames (shaded bands for frames containing events)
    if isinstance(neuron_gt_data, dict) and len(neuron_gt_data) > 0:
        # collect frames with at least one GT point
        gt_frames = sorted([int(f) for f, pts in neuron_gt_data.items() if pts])
        # merge contiguous frames into spans to reduce clutter
        spans = []
        if gt_frames:
            start = prev = gt_frames[0]
            for f in gt_frames[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    spans.append((start, prev))
                    start = prev = f
            spans.append((start, prev))
        # draw spans
        for a, b in spans:
            # draw as vertical band; use full y-range
            plt.axvspan(a, b + 1, color='gray', alpha=0.08, linewidth=0)

        if gt_frames:
            plt.scatter(gt_frames, np.full(len(gt_frames), ymin + 0.02 * (ymax - ymin)),
                        s=6, marker='|', alpha=0.6, label='GT frames')

    plt.title(f"Frame-wise Power — {label}")
    plt.xlabel("Frame")
    plt.ylabel("Mean squared value")
    plt.ylim(ymin, ymax)
    plt.xlim(0, T - 1)
    plt.legend(loc='upper right')
    plt.tight_layout()

    out_path = Path(diag_dir) / f"frame_power_{label}.png"
    try:
        plt.savefig(out_path)
        logging.info(f"Saved {out_path}")
    except Exception as e:
        logging.warning(f"[plot_frame_wise_power] Failed to save figure: {e}")
    finally:
        plt.close()

def plot_kalman_mcc_trace(video_np: np.ndarray, output_dir: Path,
                          sigma: float, mu: float,
                          max_frames: int = 300, num_pixels: int = 3,
                          neuron_gt_data: Dict = None):
    """
    Runs a lightweight Kalman–MCC pass (CPU) on the first max_frames, then plots
    raw vs MCC background and residual/weights for a few representative pixels.
    """
    if not plt: return
    logging.info("🧮 Generating Kalman–MCC trace plot...")
    T = min(video_np.shape[0], max_frames)
    clip = video_np[:T]

    # Run MCC on the clip
    bg_u16, _ = full_kalman_mcc_filter(clip, sigma=sigma, mu=mu)
    bg = bg_u16.astype(np.float32)

    # Choose pixels: prefer GT centroids, else high-variance random pixels
    coords = []
    if neuron_gt_data:
        for f, pts in sorted(neuron_gt_data.items()):
            for x, y, _ in pts:
                if 0 <= int(y) < video_np.shape[1] and 0 <= int(x) < video_np.shape[2]:
                    coords.append((min(f, T-1), int(y), int(x)))
            if len(coords) >= num_pixels:
                break
    while len(coords) < num_pixels:
        y = np.random.randint(0, video_np.shape[1])
        x = np.random.randint(0, video_np.shape[2])
        coords.append((min(T//2, T-1), y, x))

    # Plot
    fig, axes = plt.subplots(num_pixels, 2, figsize=(10, 2.6 * num_pixels))
    if num_pixels == 1: axes = np.array([axes])

    max_val = np.iinfo(video_np.dtype).max if np.issubdtype(video_np.dtype, np.integer) else 1.0
    for i, (_, yy, xx) in enumerate(coords):
        raw_ts = video_np[:T, yy, xx].astype(np.float32)
        bg_ts  = bg[:T, yy, xx]
        # innovation at each step uses previous background estimate
        e = np.zeros_like(raw_ts)
        e[1:] = raw_ts[1:] - bg_ts[:-1]
        w = np.exp(- (e**2) / (2 * (sigma * max_val)**2))

        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.plot(raw_ts, label="Raw", alpha=0.7)
        ax0.plot(bg_ts, label="MCC Background", alpha=0.9)
        ax0.set_title(f"Pixel Trace @ (x={xx}, y={yy})")
        ax0.set_xlabel("Frame"); ax0.set_ylabel("Intensity (a.u.)")
        ax0.legend(loc="upper right"); ax0.grid(True, ls='--', alpha=0.4)

        ax1.plot(e, label="Innovation e_k")
        ax1.plot(w * np.max(np.abs(e)), label="Weight ~ exp(-e^2/2σ^2)", alpha=0.8)
        ax1.set_title("Innovation and MCC Weight")
        ax1.set_xlabel("Frame"); ax1.legend(loc="upper right")
        ax1.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    out = output_dir / "fig_kalman_trace.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logging.info(f"✅ Kalman–MCC trace saved to {out}")

def plot_innovation_histogram_mcc(video_np: np.ndarray, output_dir: Path,
                                  sigma: float, mu: float, max_frames: int = 300,
                                  sample_pixels: int = 5000):
    """
    Shows the distribution of MCC innovations (before reweighting) vs a Gaussian fit.
    """
    if not plt: return
    logging.info("📈 Generating innovation histogram (MCC)...")
    T = min(video_np.shape[0], max_frames)
    clip = video_np[:T]
    bg_u16, _ = full_kalman_mcc_filter(clip, sigma=sigma, mu=mu)
    bg = bg_u16.astype(np.float32)

    # Collect innovations e_k = y_k - x_{k-1}
    raw = clip.astype(np.float32)
    e = raw[1:] - bg[:-1]
    e = e.reshape(-1)
    if len(e) > sample_pixels:
        idx = np.random.choice(len(e), size=sample_pixels, replace=False)
        e = e[idx]

    mu_hat, std_hat = float(np.mean(e)), float(np.std(e) + 1e-9)

    plt.figure(figsize=(6,4))
    plt.hist(e, bins=120, density=True, alpha=0.7, color='gray', label='MCC Innovations')
    xs = np.linspace(mu_hat - 4*std_hat, mu_hat + 4*std_hat, 400)
    from scipy.stats import norm
    plt.plot(xs, norm.pdf(xs, mu_hat, std_hat), lw=2, label='Gaussian fit')
    plt.title("Innovation Distribution (Kalman–MCC)")
    plt.xlabel("Innovation"); plt.ylabel("Density"); plt.legend()
    plt.grid(True, ls='--', alpha=0.4); plt.tight_layout()
    out = output_dir / "fig_innov_hist_raw_vs_mcc.png"
    plt.savefig(out, dpi=300); plt.close()
    logging.info(f"✅ Innovation histogram saved to {out}")

def plot_gamma_profiles(output_dir: Path,
                        s_decay_vals: List[float],
                        t_decay_vals: List[float]):
    """
    Plots example 1D temporal and 2D spatial Gamma kernels used by the feature map.
    """
    if not plt: return
    logging.info("🌊 Plotting Gamma kernel profiles...")

    # Temporal (1D)
    fig, ax = plt.subplots(figsize=(6,3))
    for half_decay in sorted(set([v for v in t_decay_vals if v > 0]))[:5]:
        n, mu = specify_gamma_kernel('center-peaked', half_decay_radius=half_decay)
        k = generate_gamma_kernel(1, n, mu, 51).cpu().numpy()
        xs = np.arange(-25, 26)
        ax.plot(xs, k, label=f"t_half={half_decay:.3f}")
    ax.set_title("Temporal Gamma Kernels (1D)")
    ax.set_xlabel("Lag (frames)"); ax.set_ylabel("Weight"); ax.legend()
    ax.grid(True, ls='--', alpha=0.4); plt.tight_layout()
    plt.savefig(output_dir / "fig_gamma_temporal_profiles.png", dpi=300)
    plt.close()

    # Spatial (2D)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    vals = sorted(set([v for v in s_decay_vals if v > 0]))[:3]
    fig, axes = plt.subplots(1, len(vals), figsize=(4*len(vals), 3))
    if len(vals) == 1: axes = [axes]
    for ax, half_decay in zip(axes, vals):
        n, mu = specify_gamma_kernel('center-peaked', half_decay_radius=half_decay)
        k2 = generate_gamma_kernel(2, n, mu, (23,23)).cpu().numpy()
        im = ax.imshow(k2, cmap='viridis')
        ax.set_title(f"Spatial Gamma\ns_half={half_decay:.3f}")
        ax.axis('off')
        divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_gamma_spatial_profiles.png", dpi=300)
    plt.close()
    logging.info("✅ Gamma profiles saved")

def plot_cfar_guard_ring_cartoon(output_path: Path, guard_radius: int = 9, ref_radius: int = 35):
    """
    A simple 2D cartoon overlay of guard (hole) and reference ring.
    """
    if not plt: return
    logging.info("🟢 Drawing CFAR guard/reference cartoon...")
    H = W = 201
    cy, cx = H//2, W//2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    ref_mask = (R >= guard_radius) & (np.abs(R - ref_radius) <= 6)
    base = np.zeros((H, W), dtype=float)
    base[ref_mask] = 1.0

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(base, cmap='Greens', origin='lower')
    circ1 = plt.Circle((cx, cy), guard_radius, color='black', fill=False, lw=2)
    circ2 = plt.Circle((cx, cy), ref_radius, color='green', fill=False, lw=2, ls='--')
    ax.add_patch(circ1); ax.add_patch(circ2)
    ax.scatter([cx], [cy], s=50, c='r')
    ax.set_title("CFAR Guard (hole) and Reference Ring")
    ax.axis('off'); plt.tight_layout()
    plt.savefig(output_path, dpi=300); plt.close()
    logging.info(f"✅ CFAR cartoon saved to {output_path}")

def plot_event_timelines(base_output_dir: Path, neuron_gt_data: Dict, window: int = 12):
    """
    For the #1 balanced model, plot z(t) at each GT centroid over a ±window.
    """
    if not plt: return
    rdir = base_output_dir / "Top_Balanced_Reports" / "Rank_1_Report"
    diag = rdir / "diagnostic_video.tif"
    cfg  = rdir / "model_config.json"
    if not diag.exists() or not cfg.exists():
        logging.warning("Missing Rank_1_Report artifacts; skipping timelines.")
        return
    with open(cfg, "r") as f:
        mc = json.load(f)
    Tstar = float(mc["point"].get("z_score", 0.0))

    diag_video = tifffile.imread(diag)  # stacked panels
    W_total = diag_video.shape[2]
    if diag_video.ndim != 3 or W_total % 5 != 0:
        logging.error(f"Unexpected diagnostic stack shape {diag_video.shape}")
        return
    W = W_total // 5
    zscore_panel = diag_video[:, :, 2*W:3*W].astype(np.float32)

    # first appearance per id
    gt_events = {}
    for f, pts in sorted(neuron_gt_data.items()):
        for x, y, gid in pts:
            gt_events.setdefault(int(gid), (int(f), int(y), int(x)))
    if not gt_events:
        logging.warning("No GT events found; skipping timelines.")
        return

    rows = 3
    cols = int(np.ceil(len(gt_events) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.8*rows), squeeze=False)
    idx = 0
    for gid, (f0, yy, xx) in sorted(gt_events.items()):
        t0 = f0
        t1 = max(0, t0 - window)
        t2 = min(zscore_panel.shape[0]-1, t0 + window)
        ts = np.arange(t1, t2+1)
        zt = zscore_panel[t1:t2+1, yy, xx]
        ax = axes[idx // cols, idx % cols]
        ax.plot(ts, zt, lw=1.5)
        ax.axhline(Tstar, ls='--', color='r', label='Threshold')
        ax.axvline(t0, ls=':', color='k', label='Onset')
        ax.set_title(f"ID {gid} @ (x={xx}, y={yy})")
        ax.set_xlabel("Frame"); ax.set_ylabel("z-score")
        ax.grid(True, ls='--', alpha=0.4)
        idx += 1
    # de-dup legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0,0,0.98,0.95])
    out = base_output_dir / "fig_event_timelines.png"
    plt.savefig(out, dpi=300); plt.close()
    logging.info(f"✅ Event timelines saved to {out}")

def plot_youden_vs_fppi(balanced_models: List[Dict], truncation_limit: float, output_dir: Path):
    """
    Scatter plot of Youden's J (TPR - FPPI/limit) vs FPPI for the balanced set.
    """
    if not plt or not balanced_models: return
    logging.info("⚖️ Plotting Youden's J vs FPPI...")
    pts = [m.get("point", {}) for m in balanced_models if m.get("point")]
    pairs = [(_FPPI(p), _TPR(p)) for p in pts]
    pairs = [(f, t) for f, t in pairs if f is not None and t is not None]
    if not pairs:
        logging.warning("No valid points to plot (missing TPR/FPPI). Skipping.")
        return

    fppi = [f for f, _ in pairs]
    tpr  = [t for _, t in pairs]
    J    = [t - (f / float(truncation_limit)) for f, t in pairs]

    plt.figure(figsize=(6,5))
    plt.scatter(fppi, J, alpha=0.8)
    best = int(np.argmax(J))
    plt.scatter([fppi[best]], [J[best]], s=120, facecolors='none', edgecolors='gold', lw=2, label='Selected')
    plt.xlabel("FPPI"); plt.ylabel("Youden's J (TPR - FPPI/limit)")
    plt.title("Operating-Point Selection")
    plt.grid(True, ls='--', alpha=0.4); plt.legend()
    out = output_dir / "fig_youden_vs_fppi.png"
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    logging.info(f"✅ Youden vs FPPI saved to {out}")

def plot_postproc_before_after(base_output_dir: Path):
    """
    Show a frame where post-processing removes the most FPs (from Rank_1_Report).
    """
    if not plt: return
    rdir = _resolve_rank1_report_dir(base_output_dir)
    if rdir is None:
        logging.warning("No diagnostic video found; skipping postproc visualization.")
        return
    diag = rdir / "diagnostic_video.tif"

    if not diag.exists():
        logging.warning("No diagnostic video found; skipping postproc visualization.")
        return
    diag_video = tifffile.imread(diag)
    W_total = diag_video.shape[2]
    if diag_video.ndim != 3 or W_total % 5 != 0: return
    W = W_total // 5
    raw_panel    = diag_video[:, :, 0*W:1*W]
    cfar_panel   = diag_video[:, :, 3*W:4*W]
    final_panel  = diag_video[:, :, 4*W:5*W]

    # pick frame with largest FP reduction
    diffs = np.count_nonzero(cfar_panel, axis=(1,2)) - np.count_nonzero(final_panel, axis=(1,2))
    f = int(np.argmax(diffs))

    fig, axes = plt.subplots(1,3, figsize=(12,4))
    axes[0].imshow(raw_panel[f], cmap='gray'); axes[0].set_title(f"Raw (frame {f})")
    axes[1].imshow(cfar_panel[f], cmap='gray'); axes[1].set_title("CFAR mask (pre)")
    axes[2].imshow(final_panel[f], cmap='gray'); axes[2].set_title("Post-processed mask")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    out = base_output_dir / "fig_postproc_before_after.png"
    plt.savefig(out, dpi=300); plt.close()
    logging.info(f"✅ Post-processing before/after saved to {out}")

def _pick_trace_pixels(neuron_gt_data, H, W, k):
    chosen = []
    by_id = {}
    for frame, pts in neuron_gt_data.items():
        for x, y, nid in pts:
            if nid not in by_id:
                by_id[nid] = (int(y), int(x))
    for _, (yy, xx) in by_id.items():
        if 0 <= yy < H and 0 <= xx < W:
            chosen.append((yy, xx))
        if len(chosen) >= k:
            break
    rng = np.random.default_rng(123)
    while len(chosen) < k:
        yy = int(rng.integers(0, H))
        xx = int(rng.integers(0, W))
        if (yy, xx) not in chosen:
            chosen.append((yy, xx))
    return chosen

def plot_kalman_mcc_trace_from_tifs(
    video_np: np.ndarray,
    output_dir: Path,
    neuron_gt_data=None,
    num_pixels: int = 3,
):
    """Plot raw vs MCC background vs innovation traces using saved kalman_bg/diff TIFFs."""
    if plt is None:
        logging.warning("matplotlib not available; skipping MCC trace plot.")
        return
    bg_path = output_dir / "kalman_bg.tif"
    if not bg_path.exists():
        logging.warning(f"{bg_path} not found; run generators.generate_kalman_background first.")
        return

    bg_u16 = tifffile.imread(bg_path)
    T = min(video_np.shape[0], bg_u16.shape[0])
    H, W = bg_u16.shape[1], bg_u16.shape[2]
    coords = _pick_trace_pixels(neuron_gt_data or {}, H, W, num_pixels)

    vmax_raw = np.iinfo(video_np.dtype).max if np.issubdtype(video_np.dtype, np.integer) else 1.0
    vmax_bg = np.iinfo(bg_u16.dtype).max

    fig, axes = plt.subplots(len(coords), 1, figsize=(8, 2.4 * len(coords)), sharex=True)
    if len(coords) == 1:
        axes = [axes]

    for ax, (yy, xx) in zip(axes, coords):
        raw = video_np[:T, yy, xx].astype(np.float32) / vmax_raw
        bg  = bg_u16[:T, yy, xx].astype(np.float32) / vmax_bg
        resid = raw - bg
        ax.plot(raw, lw=1.1, label="Raw")
        ax.plot(bg, lw=1.1, label="MCC Background")
        ax.plot(resid, lw=1.1, label="Innovation (Raw−BG)")
        ax.set_ylabel(f"(y={yy}, x={xx})")
        ax.grid(True, ls="--", alpha=0.4)

    axes[-1].set_xlabel("Frame")
    axes[0].legend(ncol=3, fontsize=9)
    plt.tight_layout()
    outpath = output_dir / "fig_kalman_mcc_trace.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    logging.info(f"✅ Saved {outpath}")

def plot_innovation_histogram_from_tifs(
    output_dir: Path,
    sample_pixels: int = 8000,
):
    """Histogram of innovation (Raw−BG) using kalman_bg.tif and kalman_diff.tif."""
    if plt is None:
        logging.warning("matplotlib not available; skipping MCC innovation hist.")
        return
    diff_path = output_dir / "kalman_diff.tif"
    if not diff_path.exists():
        logging.warning(f"{diff_path} not found; run generators.generate_kalman_background first.")
        return

    diff_u16 = tifffile.imread(diff_path)
    vals = diff_u16.astype(np.float32).ravel()
    if len(vals) > sample_pixels:
        rng = np.random.default_rng(321)
        idx = rng.choice(len(vals), size=sample_pixels, replace=False)
        vals = vals[idx]

    # normalize to [0,1] range based on dtype
    vmax = np.iinfo(diff_u16.dtype).max
    innovations = vals / vmax

    mu_hat = float(np.mean(innovations))
    sigma_hat = float(np.std(innovations) + 1e-9)

    import numpy as np
    x = np.linspace(mu_hat - 5*sigma_hat, mu_hat + 5*sigma_hat, 400)
    gauss = (1.0/(np.sqrt(2*np.pi)*sigma_hat)) * np.exp(-0.5*((x-mu_hat)/sigma_hat)**2)

    plt.figure(figsize=(7,5))
    plt.hist(innovations, bins=120, density=True, alpha=0.75, label="Innovation (norm.)")
    plt.plot(x, gauss, lw=2, label=f"Gaussian Fit (μ={mu_hat:.3e}, σ={sigma_hat:.3e})")
    plt.title("Kalman–MCC Innovation Distribution")
    plt.xlabel("Innovation (normalized)")
    plt.ylabel("Density")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    outpath = output_dir / "fig_kalman_innovation_hist.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    logging.info(f"✅ Saved {outpath}")