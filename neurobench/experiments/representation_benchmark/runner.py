"""Resource-bounded execution for PCA/ICA/autoencoder representation fits."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import tifffile

from neurobench.algorithms.representation_benchmark import (
    matched_component_stability,
    orient_components,
)
from neurobench.experiments.frame_difference import _atomic_json, _sha256
from neurobench.experiments.learnable_contrast import core as v1
from neurobench.experiments.pairwise_separation.evaluation import (
    QUIET_DURATIONS,
    QUIET_STARTS,
    event_intervals,
)
from neurobench.metrics.sparse_detection import (
    extract_local_maxima,
    match_peaks_one_to_one,
    temporal_pool,
)

from .config import RepresentationBenchmarkConfig


RAW_EXPECTED = 0.6056159420289855


def _progress(path: Path, stage: str, **details: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_unix": time.time(), "stage": stage, **details}, sort_keys=True) + "\n")
        stream.flush()


def _atomic_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
    temporary.replace(path)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _matching_preflight(config: RepresentationBenchmarkConfig, directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "preflight.json").read_text(encoding="utf-8"))
    resolved = json.loads((directory / "config.resolved.json").read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(config.to_dict()))
    if not payload.get("ready") or resolved != expected:
        raise RuntimeError("Run requires a ready preflight from the identical resolved config")
    return payload


def _pca_full(
    pixel_traces: np.ndarray, rank: int, *, device: str, chunk_pixels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact uncentered temporal Gram eigendecomposition, projected in chunks."""
    import torch

    target = torch.device(device)
    frames = pixel_traces.shape[1]
    gram = torch.zeros((frames, frames), dtype=torch.float32, device=target)
    for start in range(0, len(pixel_traces), chunk_pixels):
        block = torch.as_tensor(
            np.ascontiguousarray(pixel_traces[start:start + chunk_pixels]),
            dtype=torch.float32, device=target,
        )
        gram.addmm_(block.T, block)
        del block
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    selected = torch.clamp(eigenvalues[order], min=0)
    basis_torch = eigenvectors[:, order].T.contiguous()
    basis = basis_torch.cpu().numpy().astype(np.float32)
    singular = torch.sqrt(selected).cpu().numpy().astype(np.float64)
    explained = (selected / torch.clamp(torch.trace(gram), min=1e-20)).cpu().numpy().astype(np.float64)
    scores = np.empty((len(pixel_traces), rank), dtype=np.float32)
    for start in range(0, len(pixel_traces), chunk_pixels):
        stop = min(len(pixel_traces), start + chunk_pixels)
        block = torch.as_tensor(
            np.ascontiguousarray(pixel_traces[start:stop]), dtype=torch.float32, device=target
        )
        scores[start:stop] = (block @ basis_torch.T).cpu().numpy()
    del gram, eigenvalues, eigenvectors, selected, basis_torch
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores, basis, singular, explained


def _reconstruct(
    spatial: np.ndarray, temporal: np.ndarray, *, device: str, chunk_pixels: int
) -> np.ndarray:
    import torch

    target = torch.device(device)
    traces = torch.as_tensor(temporal, dtype=torch.float32, device=target)
    output = np.empty((len(spatial), temporal.shape[1]), dtype=np.float32)
    for start in range(0, len(spatial), chunk_pixels):
        stop = min(len(spatial), start + chunk_pixels)
        block = torch.as_tensor(np.ascontiguousarray(spatial[start:stop]), device=target)
        output[start:stop] = (block @ traces).cpu().numpy()
    del traces
    if device == "cuda":
        torch.cuda.empty_cache()
    return output


def _torch_fastica(
    spatial_scores: np.ndarray,
    temporal_basis: np.ndarray,
    *,
    sample_indices: np.ndarray,
    seed: int,
    max_iterations: int,
    tolerance: float,
    device: str,
    chunk_pixels: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    target = torch.device(device)
    mean = spatial_scores.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(spatial_scores.std(axis=0, ddof=1, dtype=np.float64), 1e-8).astype(np.float32)
    sample = (spatial_scores[sample_indices] - mean) / scale
    z = torch.as_tensor(sample, dtype=torch.float32, device=target)
    rank = z.shape[1]
    generator = torch.Generator(device=target)
    generator.manual_seed(int(seed))
    w = torch.randn((rank, rank), generator=generator, device=target)

    def decorrelate(matrix):
        values, vectors = torch.linalg.eigh(matrix @ matrix.T)
        return (vectors * torch.rsqrt(torch.clamp(values, min=1e-8))) @ vectors.T @ matrix

    w = decorrelate(w)
    delta = float("inf")
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        projection = z @ w.T
        nonlinearity = torch.tanh(projection)
        derivative = (1.0 - nonlinearity.square()).mean(dim=0)
        updated = nonlinearity.T @ z / len(z) - derivative[:, None] * w
        updated = decorrelate(updated)
        delta = float(torch.max(torch.abs(torch.abs(torch.diag(updated @ w.T)) - 1.0)).item())
        w = updated
        if delta < tolerance:
            converged = True
            break
    unmixing = w.cpu().numpy().astype(np.float32)
    sources = np.empty_like(spatial_scores)
    for start in range(0, len(spatial_scores), chunk_pixels):
        stop = min(len(spatial_scores), start + chunk_pixels)
        normalized = (spatial_scores[start:stop] - mean) / scale
        sources[start:stop] = normalized @ unmixing.T
    traces = unmixing @ (scale[:, None] * temporal_basis)
    diagnostics = {
        "seed": int(seed), "iterations": iteration, "converged": converged,
        "final_delta": delta, "contrast": "logcosh", "algorithm": "symmetric_fastica",
        "fit_sample_pixels": len(sample_indices),
    }
    del z, w
    if device == "cuda":
        torch.cuda.empty_cache()
    return sources, traces.astype(np.float32), diagnostics


def _component_evidence(
    spatial: np.ndarray,
    temporal: np.ndarray,
    *,
    quiet_frames: int,
    events: list[tuple[int, int]],
    device: str,
    chunk_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    oriented_spatial, oriented_temporal, signs = orient_components(
        spatial, temporal, quiet_frames, events
    )
    spatial_center = np.median(oriented_spatial, axis=0)
    spatial_mad = 1.4826 * np.median(np.abs(oriented_spatial - spatial_center), axis=0)
    spatial_scale = np.maximum(spatial_mad, np.percentile(spatial_mad, 10))
    quiet = oriented_temporal[:, :quiet_frames]
    temporal_center = np.median(quiet, axis=1)
    temporal_mad = 1.4826 * np.median(np.abs(quiet - temporal_center[:, None]), axis=1)
    temporal_scale = np.maximum(temporal_mad, np.percentile(temporal_mad, 10))
    positive_temporal = np.clip(
        (oriented_temporal - temporal_center[:, None]) / np.maximum(temporal_scale[:, None], 1e-6),
        0, 10,
    ).astype(np.float32)
    import torch

    target = torch.device(device)
    temporal_t = torch.as_tensor(positive_temporal, device=target)
    evidence = np.empty((len(spatial), temporal.shape[1]), dtype=np.float32)
    for start in range(0, len(spatial), chunk_pixels):
        stop = min(len(spatial), start + chunk_pixels)
        positive_spatial = np.clip(
            (oriented_spatial[start:stop] - spatial_center) / np.maximum(spatial_scale, 1e-6),
            0, 10,
        ).astype(np.float32)
        block = torch.as_tensor(positive_spatial, device=target)
        evidence[start:stop] = (block @ temporal_t / math.sqrt(spatial.shape[1])).cpu().numpy()
    return evidence, {
        "signs": signs.astype(float).tolist(),
        "spatial_scale_median": float(np.median(spatial_scale)),
        "temporal_scale_median": float(np.median(temporal_scale)),
        "aggregation": "positive_spatial_z @ positive_quiet_temporal_z / sqrt(rank)",
    }


def _calibrate(quiet_maps: list[np.ndarray], distance: int, peaks_per_map: float) -> float:
    values = sorted(
        (
            peak[0]
            for score in quiet_maps
            for peak in extract_local_maxima(score, distance, limit=3000)
        ),
        reverse=True,
    )
    allowed = max(1, int(round(peaks_per_map * len(quiet_maps))))
    if len(values) <= allowed:
        raise RuntimeError("Too few quiet peaks for calibration")
    return float(np.nextafter(values[allowed], np.inf))


def _evaluate_stack(
    lane: str,
    stack: np.ndarray,
    labels: list[dict[str, Any]],
    config: RepresentationBenchmarkConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, np.ndarray]]:
    e = config.evaluation
    quiet_maps = [
        temporal_pool(stack[start:start + duration], "lme0.25")
        for start, duration in zip(QUIET_STARTS, QUIET_DURATIONS)
    ]
    cutoff = _calibrate(quiet_maps, e.nms_distance_px, e.quiet_false_peaks_per_map)
    intervals = event_intervals(labels, config.frames.review_start_ui)
    folds = []
    candidates: list[dict[str, Any]] = []
    maps: dict[int, np.ndarray] = {}
    for burst, (start, stop) in intervals.items():
        score = temporal_pool(stack[start:stop], "lme0.25")
        maps[burst] = score
        ranked = extract_local_maxima(score, e.nms_distance_px, limit=3000)
        threshold_peaks = [peak for peak in ranked if peak[0] >= cutoff]
        fixed_peaks = ranked[:e.fixed_candidates_per_burst]
        rows = [row for row in labels if int(row["burst_id"]) == burst]
        threshold_matches, matched_indices = match_peaks_one_to_one(
            threshold_peaks, rows, e.match_radius_px
        )
        fixed_matches, _ = match_peaks_one_to_one(fixed_peaks, rows, e.match_radius_px)
        folds.append({
            "burst_id": burst, "labels": len(rows),
            "quiet_threshold_matched": len(threshold_matches),
            "quiet_threshold_recall": len(threshold_matches) / len(rows),
            "quiet_threshold_candidates": len(threshold_peaks),
            "fixed_budget_matched": len(fixed_matches),
            "fixed_budget_recall": len(fixed_matches) / len(rows),
            "fixed_budget_candidates": len(fixed_peaks),
        })
        for index, (score_value, x, y) in enumerate(threshold_peaks):
            nearest = min(math.hypot(x - row["x_px"], y - row["y_px"]) for row in rows)
            candidates.append({
                "lane": lane, "burst_id": burst, "score": score_value,
                "x_px": x, "y_px": y, "matched_known_label": index in matched_indices,
                "nearest_known_label_px": nearest,
                "interpretation": "known_match" if index in matched_indices else "unknown_candidate",
            })
    return {
        "lane": lane, "status": "evaluated", "outer_folds": folds,
        "mean_recall": float(np.mean([row["quiet_threshold_recall"] for row in folds])),
        "pooled_recall": sum(row["quiet_threshold_matched"] for row in folds) / sum(row["labels"] for row in folds),
        "total_matched": sum(row["quiet_threshold_matched"] for row in folds),
        "total_labels": sum(row["labels"] for row in folds),
        "total_event_candidates": sum(row["quiet_threshold_candidates"] for row in folds),
        "fixed_budget_mean_recall": float(np.mean([row["fixed_budget_recall"] for row in folds])),
        "fixed_budget_pooled_recall": sum(row["fixed_budget_matched"] for row in folds) / sum(row["labels"] for row in folds),
        "fixed_budget_total_matched": sum(row["fixed_budget_matched"] for row in folds),
        "fixed_budget_total_candidates": sum(row["fixed_budget_candidates"] for row in folds),
        "quiet_rank_cutoff": cutoff, "precision_identified": False,
    }, candidates, maps


def _factor_metrics(
    spatial: np.ndarray, temporal: np.ndarray, quiet_frames: int, events: list[tuple[int, int]]
) -> dict[str, Any]:
    centered = spatial - np.median(spatial, axis=0)
    absolute = np.abs(centered)
    top_count = max(1, int(math.ceil(len(spatial) * 0.01)))
    sorted_energy = np.partition(absolute, len(spatial) - top_count, axis=0)[-top_count:]
    localization = sorted_energy.sum(axis=0) / np.maximum(absolute.sum(axis=0), 1e-12)
    quiet = temporal[:, :quiet_frames]
    quiet_center = np.median(quiet, axis=1)
    quiet_scale = np.maximum(
        1.4826 * np.median(np.abs(quiet - quiet_center[:, None]), axis=1), 1e-6
    )
    event_snr = np.max(
        np.stack([
            np.abs(temporal[:, start:stop].mean(axis=1) - quiet_center) / quiet_scale
            for start, stop in events
        ], axis=1),
        axis=1,
    )
    return {
        "mean_top_1pct_spatial_mass": float(localization.mean()),
        "median_top_1pct_spatial_mass": float(np.median(localization)),
        "mean_peak_event_snr": float(event_snr.mean()),
        "median_peak_event_snr": float(np.median(event_snr)),
        "component_event_snr": event_snr.astype(float).tolist(),
    }


def _nmse(reference: np.ndarray, estimate: np.ndarray, indices: np.ndarray) -> float:
    residual = reference[indices].astype(np.float64) - estimate[indices].astype(np.float64)
    denominator = np.square(reference[indices].astype(np.float64)).sum()
    return float(np.square(residual).sum() / max(denominator, 1e-12))


def _train_autoencoder(
    pixel_traces: np.ndarray,
    *,
    kind: str,
    rank: int,
    seed: int,
    config: RepresentationBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    a, r = config.autoencoder, config.resources
    device = torch.device(r.device)
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        len(pixel_traces), size=a.train_pixels + a.validation_pixels, replace=False
    )
    train_indices = selected[:a.train_pixels]
    validation_indices = selected[a.train_pixels:]
    torch.manual_seed(seed)
    if r.device == "cuda":
        torch.cuda.manual_seed_all(seed)

    class Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if kind == "linear":
                self.encoder = nn.Linear(pixel_traces.shape[1], rank, bias=False)
                self.decoder = nn.Linear(rank, pixel_traces.shape[1], bias=False)
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(pixel_traces.shape[1], a.hidden_width), nn.GELU(),
                    nn.Linear(a.hidden_width, rank), nn.Softplus(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(rank, a.hidden_width), nn.GELU(),
                    nn.Linear(a.hidden_width, pixel_traces.shape[1]),
                )

        def forward(self, value):
            return self.decoder(self.encoder(value))

    model = Autoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-5)
    train = np.ascontiguousarray(pixel_traces[train_indices])
    losses = []
    model.train()
    generator = np.random.default_rng(seed + 1000)
    for epoch in range(a.epochs):
        order = generator.permutation(len(train))
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(train), a.batch_size):
            batch_np = train[order[start:start + a.batch_size]]
            batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = torch.mean((prediction - batch) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch)
            seen += len(batch)
        losses.append(epoch_loss / seen)
    model.eval()
    with torch.inference_mode():
        validation = torch.as_tensor(
            np.ascontiguousarray(pixel_traces[validation_indices]), device=device
        )
        prediction = model(validation)
        validation_mse = float(torch.mean((prediction - validation) ** 2).item())
        validation_nmse = float(
            torch.sum((prediction - validation) ** 2).item()
            / max(torch.sum(validation ** 2).item(), 1e-12)
        )
        spatial = np.empty((len(pixel_traces), rank), dtype=np.float32)
        reconstructed = np.empty_like(pixel_traces)
        for start in range(0, len(pixel_traces), r.projection_chunk_pixels):
            stop = min(len(pixel_traces), start + r.projection_chunk_pixels)
            block = torch.as_tensor(
                np.ascontiguousarray(pixel_traces[start:stop]), device=device
            )
            encoded = model.encoder(block)
            spatial[start:stop] = encoded.cpu().numpy()
            reconstructed[start:stop] = model.decoder(encoded).cpu().numpy()
        if kind == "linear":
            temporal = model.decoder.weight.T.detach().cpu().numpy().astype(np.float32)
        else:
            zero = torch.zeros((1, rank), device=device)
            base = model.decoder(zero)
            traces = []
            for component in range(rank):
                probe = zero.clone()
                probe[0, component] = 1.0
                traces.append((model.decoder(probe) - base).squeeze(0).cpu().numpy())
            temporal = np.asarray(traces, dtype=np.float32)
    diagnostics = {
        "kind": kind, "rank": rank, "seed": seed, "epochs": a.epochs,
        "train_pixels": a.train_pixels, "validation_pixels": a.validation_pixels,
        "final_train_mse": losses[-1], "validation_mse": validation_mse,
        "validation_nmse": validation_nmse, "loss_curve": losses,
        "nonlinear_component_trace_contract": (
            "decoder(one_hot)-decoder(zero); diagnostic, not an additive decomposition"
            if kind == "nonlinear" else "decoder weights"
        ),
    }
    del model, optimizer
    if r.device == "cuda":
        torch.cuda.empty_cache()
    return spatial, temporal, reconstructed, diagnostics


def _write_display_tiff(path: Path, stack: np.ndarray, *, signed: bool) -> dict[str, Any]:
    sample = np.asarray(stack[:, ::4, ::4], dtype=np.float32)
    if signed:
        scale = max(float(np.percentile(np.abs(sample), 99.5)), 1e-6)
        encoded = np.clip(32768.0 + stack / scale * 32767.0, 0, 65535).astype(np.uint16)
        zero_code = 32768
    else:
        scale = max(float(np.percentile(sample, 99.5)), 1e-6)
        encoded = np.clip(stack / scale * 65535.0, 0, 65535).astype(np.uint16)
        zero_code = 0
    description = json.dumps({
        "display_only": True, "signed": signed, "scale_99p5": scale,
        "zero_code": zero_code, "fixed_across_frames": True,
    }, sort_keys=True)
    tifffile.imwrite(path, encoded, photometric="minisblack", compression="zlib", description=description)
    return {"path": str(path), "scale": scale, "signed": signed, "zero_code": zero_code}


def _component_gallery(
    path: Path,
    spatial: np.ndarray,
    temporal: np.ndarray,
    *,
    shape: tuple[int, int],
    quiet_frames: int,
    events: list[tuple[int, int]],
    count: int,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = _factor_metrics(spatial, temporal, quiet_frames, events)
    order = np.argsort(metrics["component_event_snr"])[::-1][:count]
    columns = min(4, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows * 2, columns, figsize=(3.5 * columns, 3.2 * rows * 2), dpi=130)
    axes = np.atleast_2d(axes)
    for position in range(rows * columns):
        map_axis = axes[(position // columns) * 2, position % columns]
        trace_axis = axes[(position // columns) * 2 + 1, position % columns]
        if position >= len(order):
            map_axis.axis("off"); trace_axis.axis("off"); continue
        component = int(order[position])
        image = spatial[:, component].reshape(shape)
        limit = max(float(np.percentile(np.abs(image), 99.5)), 1e-6)
        map_axis.imshow(image, cmap="coolwarm", vmin=-limit, vmax=limit)
        map_axis.set_title(f"component {component}; event SNR {metrics['component_event_snr'][component]:.2f}")
        map_axis.axis("off")
        trace_axis.plot(temporal[component], color="#0891b2", linewidth=1)
        for start, stop in events:
            trace_axis.axvspan(start, stop, color="#f59e0b", alpha=0.15)
        trace_axis.axvline(quiet_frames, color="#64748b", linestyle="--", linewidth=0.8)
        trace_axis.set_xlim(0, temporal.shape[1] - 1)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _write_summary_figures(root: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row.get("status") == "evaluated"]
    figure, axis = plt.subplots(figsize=(10, 6), dpi=140)
    for row in valid:
        axis.scatter(row["total_event_candidates"], row["mean_recall"], s=30)
    leaders = sorted(valid, key=lambda row: (row["mean_recall"], -row["total_event_candidates"]), reverse=True)[:12]
    for row in leaders:
        axis.annotate(row["lane"], (row["total_event_candidates"], row["mean_recall"]), fontsize=6)
    axis.set(
        xlabel="Quiet-calibrated event candidates",
        ylabel="Mean known-label recall",
        title="Neuron-ID recall and candidate burden",
    )
    figure.tight_layout()
    figure.savefig(root / "figures" / "recall_candidate_tradeoff.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6), dpi=140)
    ordered = sorted(valid, key=lambda row: row["fixed_budget_mean_recall"], reverse=True)[:25]
    axis.barh(
        [row["lane"] for row in ordered][::-1],
        [row["fixed_budget_mean_recall"] for row in ordered][::-1],
        color="#0891b2",
    )
    axis.axvline(RAW_EXPECTED, color="#f59e0b", linestyle="--", label="Raw Direct")
    axis.set(xlabel="Mean recall at 58 candidates per burst", title="Fixed-budget neuron-ID comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "figures" / "fixed_budget_recall.png")
    plt.close(figure)


def _optional_umap(
    root: Path, scores: np.ndarray, config: RepresentationBenchmarkConfig
) -> dict[str, Any]:
    if not config.umap.enabled_if_available:
        return {"status": "disabled"}
    try:
        import umap
    except Exception as exc:
        return {
            "status": "skipped_missing_optional_dependency", "error": repr(exc),
            "scientific_role": "visualization_only",
        }
    rng = np.random.default_rng(config.umap.seed)
    indices = np.sort(rng.choice(len(scores), size=min(config.umap.sample_pixels, len(scores)), replace=False))
    reducer = umap.UMAP(
        n_neighbors=config.umap.neighbors, min_dist=config.umap.min_dist,
        n_components=2, random_state=config.umap.seed,
    )
    embedding = reducer.fit_transform(scores[indices, :min(32, scores.shape[1])])
    np.savez_compressed(root / "embeddings" / "umap_pca_scores.npz", embedding=embedding, pixel_indices=indices)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 6), dpi=140)
    axis.scatter(embedding[:, 0], embedding[:, 1], s=2, alpha=0.35)
    axis.set(title="UMAP of sampled PCA spatial scores", xlabel="UMAP-1", ylabel="UMAP-2")
    figure.tight_layout()
    figure.savefig(root / "figures" / "umap_pca_scores.png")
    plt.close(figure)
    return {"status": "complete", "samples": len(indices), "scientific_role": "visualization_only"}


def run(
    config: RepresentationBenchmarkConfig, *, preflight_dir: str | Path
) -> dict[str, Any]:
    """Run the explicitly selected, collision-safe Spon benchmark."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config.resources.cpu_threads)
    reviewed = Path(preflight_dir).resolve()
    audit = _matching_preflight(config, reviewed)
    root = config.output_dir
    if root.exists():
        raise FileExistsError(f"Output root exists: {root}")
    for relative in ("factors", "metrics", "figures", "representative_tiffs", "embeddings"):
        (root / relative).mkdir(parents=True, exist_ok=False)
    shutil.copy2(reviewed / "preflight.json", root / "preflight.json")
    shutil.copy2(reviewed / "label_projection_overlay.png", root / "label_projection_overlay.png")
    _atomic_json(root / "config.resolved.json", config.to_dict())
    _atomic_json(root / "input_manifest.json", {
        "source_video_sha256": _sha256(config.source_video),
        "labels_sha256": _sha256(config.labels_tsv),
    })
    _atomic_json(root / "run_state.json", {"status": "running", "phase": "initialization"})
    progress = root / "progress.jsonl"
    started = time.monotonic()
    all_metrics: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    representative_tiffs: list[dict[str, Any]] = []
    try:
        source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        f = config.frames
        raw = np.asarray(source[f.review_start_ui - 1:f.review_end_ui], dtype=np.float32)
        frames, height, width = raw.shape
        pixels = height * width
        quiet_frames = f.quiet_end_ui - f.quiet_start_ui + 1
        labels = v1.load_labels(config.labels_tsv)
        intervals_dict = event_intervals(labels, f.review_start_ui)
        events = [intervals_dict[key] for key in sorted(intervals_dict)]
        baseline = np.median(raw[:quiet_frames], axis=0)
        low, high = np.percentile(raw[:quiet_frames, ::4, ::4], [1, 99.9])
        global_scale = max(float(high - low), 1.0)
        raw_direct = np.maximum((raw - baseline) / global_scale, 0).astype(np.float32)
        raw_metrics, raw_candidates, _ = _evaluate_stack("raw_direct", raw_direct, labels, config)
        all_metrics.append(raw_metrics); all_candidates.extend(raw_candidates)
        raw_valid = abs(raw_metrics["mean_recall"] - RAW_EXPECTED) < 1e-12
        if not raw_valid:
            raise RuntimeError(f"Raw Direct anchor failed: {raw_metrics['mean_recall']}")
        validation_rng = np.random.default_rng(20260728)
        validation_indices = np.sort(validation_rng.choice(pixels, size=min(8192, pixels), replace=False))
        ica_rng = np.random.default_rng(20260729)
        ica_indices = np.sort(ica_rng.choice(pixels, size=config.ica.fit_sample_pixels, replace=False))
        pca_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        representative_arrays: dict[str, np.ndarray] = {}

        for input_name in config.pca.inputs:
            _progress(progress, "pca", input=input_name, status="started")
            if input_name == "amplitude":
                pixel_traces = np.ascontiguousarray((raw / global_scale).reshape(frames, pixels).T)
            else:
                pixel_traces = np.ascontiguousarray(((raw - baseline) / global_scale).reshape(frames, pixels).T)
            scores, basis, singular, explained = _pca_full(
                pixel_traces, max(config.pca.ranks),
                device=config.resources.device,
                chunk_pixels=config.resources.projection_chunk_pixels,
            )
            pca_cache[input_name] = (scores, basis, singular, explained)
            _atomic_array(root / "factors" / f"pca_{input_name}_temporal_basis.npy", basis)
            _atomic_json(root / "metrics" / f"pca_{input_name}_spectrum.json", {
                "singular_values": singular.tolist(),
                "explained_energy_ratio": explained.tolist(),
                "cumulative_explained_energy_ratio": np.cumsum(explained).tolist(),
                "factorization": "uncentered_svd_after_explicit_preprocessing",
            })
            for rank in config.pca.ranks:
                lane = f"pca_{input_name}_rank{rank}_components"
                spatial = scores[:, :rank]
                temporal = basis[:rank]
                evidence, evidence_contract = _component_evidence(
                    spatial, temporal, quiet_frames=quiet_frames, events=events,
                    device=config.resources.device,
                    chunk_pixels=config.resources.projection_chunk_pixels,
                )
                stack = evidence.T.reshape(frames, height, width)
                metrics, candidates, _ = _evaluate_stack(lane, stack, labels, config)
                metrics.update({
                    "method": "pca", "input": input_name, "rank": rank,
                    "factor_metrics": _factor_metrics(spatial, temporal, quiet_frames, events),
                    "component_evidence_contract": evidence_contract,
                    "cumulative_explained_energy_ratio": float(np.sum(explained[:rank])),
                })
                all_metrics.append(metrics); all_candidates.extend(candidates)
                if rank == min(config.pca.ranks) and input_name == "amplitude":
                    representative_arrays[f"pca_amplitude_rank{rank}_component_evidence"] = stack.copy()
                if rank in config.evaluation.reconstruction_ranks:
                    reconstruction = _reconstruct(
                        spatial, temporal, device=config.resources.device,
                        chunk_pixels=config.resources.projection_chunk_pixels,
                    )
                    reconstruction_nmse = _nmse(pixel_traces, reconstruction, validation_indices)
                    reconstruction_lane = f"pca_{input_name}_rank{rank}_reconstruction"
                    reconstructed_stack = reconstruction.T.reshape(frames, height, width)
                    if input_name == "amplitude":
                        detection = np.maximum(reconstructed_stack - baseline / global_scale, 0)
                    else:
                        detection = np.maximum(reconstructed_stack, 0)
                    rec_metrics, rec_candidates, _ = _evaluate_stack(
                        reconstruction_lane, detection.astype(np.float32), labels, config
                    )
                    rec_metrics.update({
                        "method": "pca_reconstruction", "input": input_name,
                        "rank": rank, "heldout_pixel_nmse": reconstruction_nmse,
                    })
                    all_metrics.append(rec_metrics); all_candidates.extend(rec_candidates)
                    if rank == config.evaluation.representative_rank and input_name == "quiet_residual":
                        representative_arrays["pca_residual_reconstruction"] = reconstructed_stack.copy()
                    del reconstruction, reconstructed_stack, detection
                if rank == config.evaluation.representative_rank:
                    _component_gallery(
                        root / "figures" / f"pca_{input_name}_rank{rank}_components.png",
                        spatial, temporal, shape=(height, width), quiet_frames=quiet_frames,
                        events=events, count=config.evaluation.component_gallery_count,
                        title=f"PCA {input_name}, rank {rank}",
                    )
                del evidence, stack
            _progress(progress, "pca", input=input_name, status="complete")
            del pixel_traces

        ica_references: dict[tuple[str, int], np.ndarray] = {}
        for input_name in config.ica.inputs:
            scores, basis, _, _ = pca_cache[input_name]
            for rank in config.ica.ranks:
                for seed in config.ica.seeds:
                    lane = f"spatial_fastica_{input_name}_rank{rank}_seed{seed}"
                    _progress(progress, "ica", lane=lane, status="started")
                    sources, traces, diagnostics = _torch_fastica(
                        scores[:, :rank], basis[:rank], sample_indices=ica_indices,
                        seed=seed, max_iterations=config.ica.max_iterations,
                        tolerance=config.ica.tolerance, device=config.resources.device,
                        chunk_pixels=config.resources.projection_chunk_pixels,
                    )
                    evidence, contract = _component_evidence(
                        sources, traces, quiet_frames=quiet_frames, events=events,
                        device=config.resources.device,
                        chunk_pixels=config.resources.projection_chunk_pixels,
                    )
                    metrics, candidates, _ = _evaluate_stack(
                        lane, evidence.T.reshape(frames, height, width), labels, config
                    )
                    key = (input_name, rank)
                    if key not in ica_references:
                        ica_references[key] = sources[ica_indices].copy()
                        stability = {
                            "reference_seed": seed, "mean_absolute_correlation": 1.0,
                            "fraction_at_least_0p9": 1.0,
                        }
                    else:
                        stability = matched_component_stability(
                            ica_references[key], sources[ica_indices]
                        )
                        stability["reference_seed"] = config.ica.seeds[0]
                    metrics.update({
                        "method": "spatial_fastica", "input": input_name, "rank": rank,
                        "seed": seed, "fit": diagnostics, "stability": stability,
                        "factor_metrics": _factor_metrics(sources, traces, quiet_frames, events),
                        "component_evidence_contract": contract,
                    })
                    all_metrics.append(metrics); all_candidates.extend(candidates)
                    fit_rows.append({
                        "method": "spatial_fastica", "input": input_name, "rank": rank,
                        "seed": seed, "converged": diagnostics["converged"],
                        "iterations": diagnostics["iterations"], "final_delta": diagnostics["final_delta"],
                        "stability_mean_abs_corr": stability["mean_absolute_correlation"],
                        "mean_recall": metrics["mean_recall"],
                        "fixed_budget_mean_recall": metrics["fixed_budget_mean_recall"],
                        "event_candidates": metrics["total_event_candidates"],
                    })
                    if rank == config.evaluation.representative_rank and seed == config.ica.seeds[0]:
                        _atomic_array(root / "factors" / f"{lane}_spatial_sources.npy", sources)
                        _atomic_array(root / "factors" / f"{lane}_temporal_traces.npy", traces)
                        _component_gallery(
                            root / "figures" / f"{lane}_components.png", sources, traces,
                            shape=(height, width), quiet_frames=quiet_frames, events=events,
                            count=config.evaluation.component_gallery_count,
                            title=f"Spatial FastICA {input_name}, rank {rank}, seed {seed}",
                        )
                        if input_name == "quiet_residual":
                            representative_arrays["ica_residual_component_evidence"] = (
                                evidence.T.reshape(frames, height, width).copy()
                            )
                    del sources, traces, evidence
                    _progress(progress, "ica", lane=lane, status="complete")

        auto_references: dict[tuple[str, str, int], np.ndarray] = {}
        if config.autoencoder.enabled:
            for input_name in config.autoencoder.inputs:
                if input_name == "amplitude":
                    pixel_traces = np.ascontiguousarray((raw / global_scale).reshape(frames, pixels).T)
                else:
                    pixel_traces = np.ascontiguousarray(((raw - baseline) / global_scale).reshape(frames, pixels).T)
                for kind in config.autoencoder.kinds:
                    for rank in config.autoencoder.ranks:
                        for seed in config.autoencoder.seeds:
                            lane = f"{kind}_autoencoder_{input_name}_rank{rank}_seed{seed}"
                            _progress(progress, "autoencoder", lane=lane, status="started")
                            spatial, traces, reconstructed, diagnostics = _train_autoencoder(
                                pixel_traces, kind=kind, rank=rank, seed=seed, config=config
                            )
                            evidence, contract = _component_evidence(
                                spatial, traces, quiet_frames=quiet_frames, events=events,
                                device=config.resources.device,
                                chunk_pixels=config.resources.projection_chunk_pixels,
                            )
                            component_metrics, component_candidates, _ = _evaluate_stack(
                                lane + "_components",
                                evidence.T.reshape(frames, height, width), labels, config,
                            )
                            reconstruction_stack = reconstructed.T.reshape(frames, height, width)
                            if input_name == "amplitude":
                                detection = np.maximum(reconstruction_stack - baseline / global_scale, 0)
                            else:
                                detection = np.maximum(reconstruction_stack, 0)
                            reconstruction_metrics, reconstruction_candidates, _ = _evaluate_stack(
                                lane + "_reconstruction", detection.astype(np.float32), labels, config
                            )
                            key = (input_name, kind, rank)
                            stability_sample = spatial[ica_indices]
                            if key not in auto_references:
                                auto_references[key] = stability_sample.copy()
                                stability = {"reference_seed": seed, "mean_absolute_correlation": 1.0}
                            else:
                                stability = matched_component_stability(auto_references[key], stability_sample)
                                stability["reference_seed"] = config.autoencoder.seeds[0]
                            common = {
                                "input": input_name, "rank": rank, "seed": seed,
                                "autoencoder_kind": kind, "fit": diagnostics, "stability": stability,
                            }
                            component_metrics.update({
                                "method": f"{kind}_autoencoder_components", **common,
                                "factor_metrics": _factor_metrics(spatial, traces, quiet_frames, events),
                                "component_evidence_contract": contract,
                            })
                            reconstruction_metrics.update({
                                "method": f"{kind}_autoencoder_reconstruction", **common,
                                "heldout_pixel_nmse": diagnostics["validation_nmse"],
                            })
                            all_metrics.extend((component_metrics, reconstruction_metrics))
                            all_candidates.extend(component_candidates)
                            all_candidates.extend(reconstruction_candidates)
                            fit_rows.append({
                                "method": f"{kind}_autoencoder", "input": input_name,
                                "rank": rank, "seed": seed, "converged": True,
                                "iterations": config.autoencoder.epochs,
                                "final_delta": diagnostics["final_train_mse"],
                                "stability_mean_abs_corr": stability["mean_absolute_correlation"],
                                "mean_recall": component_metrics["mean_recall"],
                                "fixed_budget_mean_recall": component_metrics["fixed_budget_mean_recall"],
                                "event_candidates": component_metrics["total_event_candidates"],
                            })
                            if rank == config.evaluation.representative_rank and seed == config.autoencoder.seeds[0]:
                                _component_gallery(
                                    root / "figures" / f"{lane}_components.png", spatial, traces,
                                    shape=(height, width), quiet_frames=quiet_frames, events=events,
                                    count=config.evaluation.component_gallery_count,
                                    title=f"{kind.title()} autoencoder {input_name}, rank {rank}",
                                )
                                if input_name == "quiet_residual":
                                    representative_arrays[f"{kind}_autoencoder_residual_reconstruction"] = (
                                        reconstruction_stack.copy()
                                    )
                            del spatial, traces, reconstructed, evidence, reconstruction_stack, detection
                            _progress(progress, "autoencoder", lane=lane, status="complete")
                del pixel_traces

        umap_payload = _optional_umap(root, pca_cache["quiet_residual"][0], config)
        _atomic_json(root / "metrics" / "umap_status.json", umap_payload)
        if config.evaluation.write_representative_tiffs:
            for name, stack in representative_arrays.items():
                representative_tiffs.append(_write_display_tiff(
                    root / "representative_tiffs" / f"{name}.tif",
                    stack, signed="evidence" not in name,
                ))
        _write_tsv(root / "metrics" / "fit_summary.tsv", fit_rows)
        _write_tsv(root / "metrics" / "candidate_peaks.tsv", all_candidates)
        _atomic_json(root / "metrics" / "neuron_id_metrics.json", {
            "schema_version": 1, "raw_direct_anchor_valid": raw_valid,
            "lanes": all_metrics,
            "precision_contract": (
                "Sparse positives identify recall and candidate burden only. "
                "Unmatched candidates remain unknown, not false positives."
            ),
        })
        _write_summary_figures(root, all_metrics)
        leaders = sorted(
            [row for row in all_metrics if row.get("status") == "evaluated"],
            key=lambda row: (row["fixed_budget_mean_recall"], row["mean_recall"], -row["total_event_candidates"]),
            reverse=True,
        )
        best = leaders[0]
        summary = {
            "schema_version": 1, "experiment_id": config.experiment_id,
            "status": "complete", "raw_direct_anchor_valid": raw_valid,
            "combination_count": audit["combinations"],
            "evaluated_lanes": len(all_metrics),
            "best_fixed_budget_lane": best["lane"],
            "best_fixed_budget_mean_recall": best["fixed_budget_mean_recall"],
            "raw_direct_mean_recall": raw_metrics["mean_recall"],
            "raw_direct_fixed_budget_mean_recall": raw_metrics["fixed_budget_mean_recall"],
            "umap": umap_payload, "representative_tiffs": representative_tiffs,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(root / "experiment_summary.json", summary)
        index = [
            f"# {config.experiment_id}: important results", "",
            "## Start here", "",
            "- `report.md`: interpretation and decision.",
            "- `experiment_summary.json`: compact machine-readable outcome.",
            "- `metrics/neuron_id_metrics.json`: every neuron-ID lane and fold.",
            "- `metrics/fit_summary.tsv`: ICA/autoencoder convergence and stability.",
            "- `figures/fixed_budget_recall.png`: fair 58-candidate-per-burst comparison.",
            "- `figures/recall_candidate_tradeoff.png`: recall versus candidate burden.",
            "- `representative_tiffs/`: fixed-scale review videos.",
            "- `figures/*components.png`: component maps and temporal traces.", "",
            "## Scientific caution", "",
            "Unmatched candidates are unknown. These results do not identify ordinary precision.",
        ]
        (root / "RESULTS_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
        top_lines = [
            f"- `{row['lane']}`: fixed-budget mean recall `{row['fixed_budget_mean_recall']:.4f}`, "
            f"quiet-threshold mean recall `{row['mean_recall']:.4f}`, candidates `{row['total_event_candidates']}`."
            for row in leaders[:10]
        ]
        report = [
            f"# {config.experiment_id}", "",
            "## Outcome", "",
            f"Completed `{audit['combinations']['total_fits']}` fitted representations and "
            f"`{len(all_metrics)}` evaluated neuron-ID lanes.",
            f"Raw Direct reproduced at `{raw_metrics['mean_recall']:.9f}`.",
            f"Best fixed-budget lane: `{best['lane']}` at `{best['fixed_budget_mean_recall']:.4f}`.", "",
            "## Leading neuron-ID lanes", "", *top_lines, "",
            "## Interpretation", "",
            "PCA reconstruction and ICA reconstruction span the same subspace at a fixed rank; "
            "ICA is distinguished by component localization, temporal activity, seed stability, "
            "and component-evidence neuron-ID performance—not reconstruction NMSE alone.", "",
            "UMAP is visualization-only. Sparse positive labels do not identify precision, and "
            "unmatched candidates remain unknown.", "",
            "## Important artifacts", "",
            "See `RESULTS_INDEX.md`.", "",
        ]
        (root / "report.md").write_text("\n".join(report), encoding="utf-8")
        _atomic_json(root / "run_state.json", {
            "status": "complete", "phase": "complete",
            "elapsed_seconds": summary["elapsed_seconds"],
        })
        _progress(progress, "complete", status="complete")
        return summary
    except Exception as exc:
        _atomic_json(root / "run_state.json", {
            "status": "failed", "phase": "failed", "error": repr(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        _progress(progress, "failed", error=repr(exc))
        raise
