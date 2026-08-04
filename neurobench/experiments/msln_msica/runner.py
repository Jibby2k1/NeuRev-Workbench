"""Guarded, resumable MSLN/MS-ICA execution pipeline."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from neurobench.algorithms.multiscale_subspace import (
    bootstrap_summary,
    contiguous_block_bootstrap,
    fit_cross_context,
    fit_per_context_ica,
)
from neurobench.algorithms.quiet_calibration import (
    EmpiricalQuietTail,
    QuietRobustStandardizer,
    energy_mapping_bank,
    group_energy,
)

from .artifacts import atomic_json, sha256_file
from .config import MSLNMSICAConfig
from .context_bank import evaluate_context, ordered_contexts
from .diagnostics import render_diagnostics
from .evaluation import sparse_positive_evaluation, synthetic_fixture_audit
from .fitting import fit_context, pairs_at
from .inference import apply_innovation
from .preflight import matching_preflight
from .routing import product_interaction, route_evidence


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), delimiter="\t"
            )
            writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _is_tiny(config: MSLNMSICAConfig, movie: np.ndarray) -> bool:
    start, stop = config.source.review_interval_ui
    return stop - start + 1 <= 128 and movie.shape[1] <= 128 and movie.shape[2] <= 128


def _extended_diagnostics(root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = root / "diagnostics"

    context_paths = sorted((root / "features" / "msln").glob("*.npy"))
    fig, axis = plt.subplots(figsize=(7, 3))
    for path in context_paths[:4]:
        values = np.load(path, mmap_mode="r")
        axis.plot(values[:, values.shape[1] // 2, values.shape[2] // 2], label=path.stem)
    axis.set(title="Center-pixel signed standardized context traces", xlabel="review-relative frame", ylabel="signed z")
    axis.legend(fontsize=6); fig.tight_layout(); fig.savefig(destination / "context_traces.png", dpi=110); plt.close(fig)

    fit_paths = sorted((root / "fits" / "fold_all" / "per_context").glob("*/fit.json"))
    fit_rows = [json.loads(path.read_text()) for path in fit_paths]
    labels = [row["context_id"] for row in fit_rows]
    fig, axis = plt.subplots(figsize=(7, 3)); axis.bar(labels, [row["objective_value"] for row in fit_rows], label="CS-Parzen"); axis.scatter(labels, [row["baseline_objective_value"] for row in fit_rows], color="black", s=12, label="direct difference"); axis.set(title="Held-out objective by context", ylabel="CS-Parzen dependence objective"); axis.tick_params(axis="x", rotation=55, labelsize=7); axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(destination / "objective_by_context.png", dpi=110); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 3))
    for path in sorted((root / "fits" / "fold_all" / "per_context").glob("*/bootstrap.tsv")):
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        axis.plot([float(row["angle_degrees"]) for row in rows], marker=".", linewidth=.7, label=path.parent.name)
    axis.set(title="Contiguous-block bootstrap ICA angles", xlabel="replicate", ylabel="angle (degrees)"); axis.legend(fontsize=5, ncol=2); fig.tight_layout(); fig.savefig(destination / "angle_bootstrap.png", dpi=110); plt.close(fig)

    pca_path = root / "fits" / "cross_context" / "pca.json"
    loadings = np.asarray(json.loads(pca_path.read_text())["transform"]) if pca_path.is_file() else np.eye(len(context_paths))
    fig, axis = plt.subplots(figsize=(6, 4)); image = axis.imshow(loadings, aspect="auto", cmap="coolwarm"); fig.colorbar(image, ax=axis, label="loading"); axis.set(title="Cross-context PCA loadings", xlabel="context", ylabel="component"); fig.tight_layout(); fig.savefig(destination / "cross_context_loadings.png", dpi=110); plt.close(fig)

    sampled = np.load(root / "features" / "cross_context" / "sampled_innovations.npz")["innovations"]
    correlation = np.corrcoef(sampled, rowvar=False)
    fig, axis = plt.subplots(figsize=(5, 4)); image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm"); fig.colorbar(image, ax=axis, label="correlation"); axis.set_title("Sampled innovation correlations"); fig.tight_layout(); fig.savefig(destination / "component_correlations.png", dpi=110); plt.close(fig)

    energy_paths = sorted((root / "features" / "energy").glob("*_quiet_tail.npy"))
    sampled_energy = []
    for path in energy_paths:
        values = np.load(path, mmap_mode="r").reshape(-1)
        sampled_energy.append(np.asarray(values[::max(1, len(values) // 20000)], dtype=float))
    fig, axis = plt.subplots(figsize=(6, 3))
    for path, values in zip(energy_paths, sampled_energy):
        ordered = np.sort(values); survival = (len(ordered) - np.arange(len(ordered))) / len(ordered)
        axis.semilogy(ordered, survival, label=path.stem)
    axis.set(title="Empirical activity-evidence CCDF", xlabel="quiet-tail surprise", ylabel="survival"); axis.legend(fontsize=5, ncol=2); fig.tight_layout(); fig.savefig(destination / "quiet_ccdf.png", dpi=110); plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 3)); axis.boxplot(sampled_energy, tick_labels=[path.stem for path in energy_paths], showfliers=False); axis.set(title="Quiet-tail calibration by context", ylabel="surprise"); axis.tick_params(axis="x", rotation=55, labelsize=6); fig.tight_layout(); fig.savefig(destination / "quiet_calibration.png", dpi=110); plt.close(fig)

    mapping_paths = sorted((root / "features" / "energy").glob("*_raw_square.npy")) + sorted((root / "features" / "energy").glob("*_bounded_square.npy"))
    fig, axis = plt.subplots(figsize=(7, 3)); names, p99 = [], []
    for path in mapping_paths:
        values = np.load(path, mmap_mode="r").reshape(-1); sample = values[::max(1, len(values) // 20000)]
        names.append(path.stem); p99.append(float(np.percentile(sample, 99)))
    axis.bar(names, p99); axis.set(title="Energy mapping comparison", ylabel="sampled p99 energy"); axis.tick_params(axis="x", rotation=65, labelsize=5); fig.tight_layout(); fig.savefig(destination / "energy_mapping_comparison.png", dpi=110); plt.close(fig)

    sparse = json.loads((root / "evaluation" / "fixed_unsupervised" / "metrics.json").read_text())
    fig, axis = plt.subplots(figsize=(6, 3))
    lanes = sorted({row["lane"] for row in sparse["rows"]})
    for lane in lanes:
        budgets = sorted({row["budget"] for row in sparse["rows"] if row["lane"] == lane})
        recall = [np.mean([row["recall"] for row in sparse["rows"] if row["lane"] == lane and row["budget"] == budget]) for budget in budgets]
        axis.plot(budgets, recall, marker="o", label=lane)
    axis.set(title="Known-label recall by candidate budget", xlabel="candidate budget", ylabel="mean recall"); axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(destination / "budget_curves.png", dpi=110); plt.close(fig)

    candidate_path = root / "evaluation" / "native_proposals" / "candidates.tsv"
    with candidate_path.open(encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream, delimiter="\t"))
    known = [sum(row["lane"] == lane and row["matched_known_label"] == "True" for row in candidates) for lane in lanes]
    unknown = [sum(row["lane"] == lane and row["matched_known_label"] == "False" for row in candidates) for lane in lanes]
    fig, axis = plt.subplots(figsize=(6, 3)); axis.bar(lanes, known, label="known matches"); axis.bar(lanes, unknown, bottom=known, label="unknown candidates"); axis.set(title="Native candidate sources (unknown is not negative)", ylabel="candidates"); axis.tick_params(axis="x", rotation=20, labelsize=7); axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(destination / "candidate_overlap.png", dpi=110); plt.close(fig)

    synthetic = json.loads((root / "evaluation" / "synthetic" / "metrics.json").read_text())
    names = [row["fixture"] for row in synthetic["fixtures"]]
    fig, axis = plt.subplots(figsize=(7, 3)); axis.bar(names, [row["area_above_half_peak"] for row in synthetic["fixtures"]]); axis.set(title="Generated fixture morphology support", ylabel="truth pixels above half peak"); axis.tick_params(axis="x", rotation=55, labelsize=6); fig.tight_layout(); fig.savefig(destination / "preservation_by_morphology.png", dpi=110); plt.close(fig)

    resources = json.loads((root / "resource_plan.json").read_text())
    fig, axis = plt.subplots(figsize=(5, 3)); axis.bar(["estimated peak", "RAM cap", "selected output", "disk free"], [resources["peak_ram_bytes_estimate"] / 2**30, resources["ram_cap_bytes"] / 2**30, resources["selected_output_bytes_estimate"] / 2**30, resources["disk_free_bytes"] / 2**30]); axis.set(title="Preflight resource telemetry", ylabel="GiB"); axis.tick_params(axis="x", rotation=25, labelsize=7); fig.tight_layout(); fig.savefig(destination / "resource_telemetry.png", dpi=110); plt.close(fig)


def run(config: MSLNMSICAConfig, *, authorize_full_spon: bool = False, resume: bool = False, no_video: bool = False, fold: int | None = None, stage: str = "all") -> dict[str, Any]:
    preflight = matching_preflight(config)
    root = config.outputs.root_dir
    status_path = root / "status.json"
    prior = json.loads(status_path.read_text())
    if prior.get("status") == "complete":
        if resume:
            return summarize(root)
        raise FileExistsError("run is already complete; use summarize")
    if prior.get("status") == "partial" and not resume:
        raise RuntimeError("partial output exists; pass --resume to continue safely")
    movie_source = np.load(config.source.movie_path, mmap_mode="r", allow_pickle=False)
    if not _is_tiny(config, movie_source) and not authorize_full_spon:
        raise PermissionError("full real-data execution requires --authorize-full-spon after reviewing preflight")
    if fold is not None and fold not in config.fold_ids:
        raise ValueError("requested fold is not declared")
    if stage not in {"msln", "per-context-ica", "cross-context", "energy", "routing", "fusion", "all"}:
        raise ValueError("unknown stage")
    os.environ["OMP_NUM_THREADS"] = str(config.compute.cpu_threads)
    atomic_json(status_path, {"status": "running", "scientific_status": "not_established", "stage": stage})
    started = time.monotonic()
    try:
        start, stop = config.source.review_interval_ui
        raw = np.asarray(movie_source[start - 1:stop], dtype=np.float32)
        carrier = raw
        if config.preprocessing.input_domain == "raw_smoothed" and config.preprocessing.gaussian_sigma_px > 0:
            carrier = gaussian_filter(raw, sigma=(0, config.preprocessing.gaussian_sigma_px, config.preprocessing.gaussian_sigma_px)).astype(np.float32)
        quiet_start, quiet_stop = config.source.quiet_interval_ui
        quiet_mask = np.zeros(len(raw), dtype=bool)
        quiet_mask[quiet_start - start:quiet_stop - start + 1] = True
        definitions = ordered_contexts(config)
        context_rows: list[dict[str, Any]] = []
        sample_records: dict[str, np.ndarray] = {}
        context_paths: list[Path] = []
        innovation_paths: list[Path] = []
        standardized_paths: list[Path] = []
        energy_paths: list[Path] = []
        valid_masks: list[np.ndarray] = []
        fits = []
        for index, definition in enumerate(definitions):
            context_start = time.monotonic()
            result = evaluate_context(carrier, definition, quiet_mask=quiet_mask)
            context_path = root / "features" / "msln" / f"{definition.context_id}.npy"
            _atomic_npy(context_path, result.values)
            sidecar = dict(result.diagnostics)
            sidecar.update({"valid_frames": np.flatnonzero(result.valid_frames).tolist(), "input_domain": config.preprocessing.input_domain, "runtime_seconds": time.monotonic() - context_start, "sha256": sha256_file(context_path), "representation": "signed standardized evidence"})
            atomic_json(context_path.with_suffix(".json"), sidecar)
            context_rows.append({"context_index": index, "context_id": definition.context_id, "kind": definition.kind, "valid_frames": int(result.valid_frames.sum()), "scale_floor": result.scale_floor})
            context_paths.append(context_path)
            valid_masks.append(result.valid_frames.copy())
            fitted, screen_indices, confirmation_indices = fit_context(definition.context_id, result.values, result.valid_frames, config)
            fits.append(fitted)
            sample_records[f"{definition.context_id}_screen"] = screen_indices
            sample_records[f"{definition.context_id}_confirmation"] = confirmation_indices
            fit_dir = root / "fits" / "fold_all" / "per_context" / definition.context_id
            atomic_json(fit_dir / "fit.json", fitted.to_dict())
            fit_dir.mkdir(parents=True, exist_ok=True)
            _atomic_npz(fit_dir / "fit_arrays.npz", center=fitted.center, whitening=fitted.whitening, demixing=fitted.demixing)
            identical_screen = pairs_at(result.values, screen_indices)
            identical_confirmation = pairs_at(result.values, confirmation_indices)
            direct_fit = {
                "method": "direct_difference",
                "demixing": [[2 ** -0.5, 2 ** -0.5], [-2 ** -0.5, 2 ** -0.5]],
                "screen_sample_key": f"{definition.context_id}_screen",
                "confirmation_sample_key": f"{definition.context_id}_confirmation",
            }
            atomic_json(fit_dir / "direct_difference.json", direct_fit)
            if config.per_context_ica.run_fastica_ablation:
                fastica = fit_per_context_ica(
                    definition.context_id,
                    identical_screen,
                    identical_confirmation,
                    objective="fastica",
                    parzen_bandwidth=config.per_context_ica.parzen_bandwidth,
                    eigenvalue_floor_ratio=config.per_context_ica.eigenvalue_floor_ratio,
                    kernel_block_rows=config.per_context_ica.kernel_block_rows,
                    kernel_dtype=np.dtype(config.compute.kernel_dtype),
                )
                atomic_json(fit_dir / "fastica_ablation.json", fastica.to_dict())
            bootstrap_pairs = identical_screen[: min(len(identical_screen), 512)]
            bootstrap_rows = contiguous_block_bootstrap(
                definition.context_id,
                bootstrap_pairs,
                block_length=min(config.sampling.time_block_length_frames, len(bootstrap_pairs)),
                replicates=config.sampling.bootstrap_replicates,
                seed=config.sampling.seed + index,
                fitter_kwargs={
                    "parzen_bandwidth": config.per_context_ica.parzen_bandwidth,
                    "eigenvalue_floor_ratio": config.per_context_ica.eigenvalue_floor_ratio,
                    "coarse_step_degrees": config.per_context_ica.coarse_step_degrees,
                    "refine_half_width_degrees": config.per_context_ica.refine_half_width_degrees,
                    "refine_step_degrees": config.per_context_ica.refine_step_degrees,
                    "kernel_block_rows": min(config.per_context_ica.kernel_block_rows, 128),
                    "kernel_dtype": np.dtype(config.compute.kernel_dtype),
                },
            )
            _write_tsv(fit_dir / "bootstrap.tsv", bootstrap_rows)
            atomic_json(
                fit_dir / "bootstrap_summary.json",
                {**bootstrap_summary(bootstrap_rows), "sample_cap": 512, "resampling": "contiguous_blocks"},
            )
            persistence, innovation = apply_innovation(result.values, fitted, result.valid_frames)
            if config.outputs.save_all_latent_maps:
                _atomic_npy(root / "features" / "per_context_ica" / f"{definition.context_id}_persistence.npy", persistence)
            innovation_path = root / "features" / "per_context_ica" / f"{definition.context_id}_innovation.npy"
            _atomic_npy(innovation_path, innovation); innovation_paths.append(innovation_path)
            calibrator = QuietRobustStandardizer(mode="per_pixel", minimum_samples=min(3, int(quiet_mask.sum()))).fit(innovation, quiet_mask & result.valid_frames)
            standardized = calibrator.transform(innovation)
            standardized_path = root / "features" / "energy" / f"{definition.context_id}_standardized.npy"
            _atomic_npy(standardized_path, standardized); standardized_paths.append(standardized_path)
            mapping_bank = energy_mapping_bank(
                standardized,
                bounded_kappa=config.energy.bounded_kappa_z,
                huber_delta=1.5,
            )
            energy = mapping_bank["raw_square"]
            _atomic_npy(root / "features" / "energy" / f"{definition.context_id}_raw_square.npy", energy)
            _atomic_npy(root / "features" / "energy" / f"{definition.context_id}_bounded_square.npy", mapping_bank["bounded_square"])
            tail = EmpiricalQuietTail().fit(energy[quiet_mask & result.valid_frames])
            surprise = tail.surprise(energy, log_base=config.energy.tail_log_base)
            energy_path = root / "features" / "energy" / f"{definition.context_id}_quiet_tail.npy"
            _atomic_npy(energy_path, surprise); energy_paths.append(energy_path)
            atomic_json(energy_path.with_suffix(".json"), {"representation": "quiet-tail surprise", "standardizer": calibrator.to_dict(), "tail": tail.to_dict(), "source": definition.context_id})
            del result, persistence, innovation, standardized, energy, surprise
        _write_tsv(root / "context_manifest.tsv", context_rows)
        _atomic_npz(root / "sample_manifest.npz", **sample_records)
        context_ids = tuple(item.context_id for item in definitions)
        alias_lookup = {
            f"{name.removesuffix('_meanstd')}_innovation": index
            for index, name in enumerate(context_ids)
        }
        for index, name in enumerate(context_ids):
            if name.startswith("st_t"):
                parts = name.split("_")
                temporal = parts[1].removeprefix("t")
                spatial = parts[2].removeprefix("s")
                alias_lookup[f"st_{spatial}x{temporal}_innovation"] = index
        group_rows: dict[str, Any] = {}
        for group_id, member_names in config.cross_context.groups.items():
            try:
                indices = [alias_lookup[name] for name in member_names]
            except KeyError as exc:
                raise ValueError(f"group {group_id} references unknown innovation {exc.args[0]}") from exc
            coordinate_maps = [np.load(standardized_paths[index], mmap_mode="r") for index in indices]
            grouped = group_energy(np.stack(coordinate_maps), axis=0)
            group_valid = quiet_mask.copy()
            for index in indices:
                group_valid &= valid_masks[index]
            tail = EmpiricalQuietTail().fit(grouped[group_valid])
            group_surprise = tail.surprise(grouped, log_base=config.energy.tail_log_base)
            group_path = root / "features" / "energy" / f"{group_id}_quiet_tail_group_energy.npy"
            _atomic_npy(group_path, group_surprise)
            group_rows[group_id] = {"members": list(member_names), "context_indices": indices, "tail": tail.to_dict(), "representation": "quiet-tail group energy"}
        atomic_json(root / "features" / "energy" / "group_energy_manifest.json", group_rows)
        # Cross-context fits use a bounded, identical sampled innovation matrix.
        rng = np.random.default_rng(config.sampling.seed)
        total = int(np.prod(raw.shape))
        flat_indices = np.sort(rng.choice(total, size=min(total, config.sampling.cross_context_max_samples), replace=False))
        sampled = np.column_stack([np.load(path, mmap_mode="r").reshape(-1)[flat_indices] for path in innovation_paths])
        _atomic_npz(
            root / "features" / "cross_context" / "sampled_innovations.npz",
            flat_indices=flat_indices,
            innovations=sampled.astype(np.float32),
        )
        cross_rows = {}
        for mode in config.cross_context.modes:
            if mode not in {"identity", "pca", "fastica"}:
                continue
            cross = fit_cross_context(sampled, mode=mode, max_components=config.cross_context.max_components, max_samples=config.sampling.cross_context_max_samples, seed=config.sampling.seed)
            cross_rows[mode] = cross.to_dict()
            atomic_json(root / "fits" / "cross_context" / f"{mode}.json", cross.to_dict())
        # Fixed routing is chunked so only K small frame chunks coexist.
        evidence_path = root / "features" / "routing" / "activity_evidence.npy"
        dominant_path = root / "features" / "routing" / "dominant_context.npy"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_temporary = evidence_path.with_suffix(".partial.npy")
        dominant_temporary = dominant_path.with_suffix(".partial.npy")
        evidence = np.lib.format.open_memmap(evidence_temporary, mode="w+", dtype=np.float32, shape=raw.shape)
        dominant = np.lib.format.open_memmap(dominant_temporary, mode="w+", dtype=np.uint8, shape=raw.shape)
        energy_maps = [np.load(path, mmap_mode="r") for path in energy_paths]
        selected_mode = "compact_minus_broad" if "compact_minus_broad" in config.routing.modes and any(name.startswith("spatial_15_") for name in context_ids) else "max"
        scalar_modes = tuple(mode for mode in config.routing.modes if mode != "none")
        route_paths = {
            mode: root / "features" / "routing" / f"{mode}.npy"
            for mode in scalar_modes
        }
        route_temporaries = {
            mode: path.with_suffix(".partial.npy") for mode, path in route_paths.items()
        }
        route_maps = {
            mode: np.lib.format.open_memmap(
                route_temporaries[mode], mode="w+", dtype=np.float32, shape=raw.shape
            )
            for mode in scalar_modes
        }
        for frame_start in range(0, len(raw), config.compute.frame_chunk):
            frame_stop = min(frame_start + config.compute.frame_chunk, len(raw))
            stack = np.stack([item[frame_start:frame_stop] for item in energy_maps])
            selected = None
            for mode in scalar_modes:
                routed, mode_dominant = route_evidence(stack, context_ids, mode=mode, temperature=config.routing.softmax_temperature, complexity_penalty=config.routing.complexity_penalty, compact_minus_broad_weight=config.routing.compact_minus_broad_weight)
                route_maps[mode][frame_start:frame_stop] = routed
                if mode == selected_mode:
                    evidence[frame_start:frame_stop] = routed
                    selected = mode_dominant
            if selected is None:
                raise RuntimeError("selected routing mode was not materialized")
            dominant[frame_start:frame_stop] = selected
        for mode, mapped in route_maps.items():
            mapped.flush()
        del route_maps
        for mode, path in route_paths.items():
            route_temporaries[mode].replace(path)
        atomic_json(
            root / "features" / "routing" / "none.json",
            {"representation": "unrouted channel tuple", "energy_maps": [str(path.relative_to(root)) for path in energy_paths]},
        )
        evidence.flush(); dominant.flush()
        del evidence, dominant
        evidence_temporary.replace(evidence_path)
        dominant_temporary.replace(dominant_path)
        evidence = np.load(evidence_path, mmap_mode="r")
        dominant = np.load(dominant_path, mmap_mode="r")
        if config.fusion.evaluate_product_interaction:
            interaction = product_interaction(raw, evidence, beta=config.fusion.visualization_floor_beta, kappa=config.fusion.bounded_gate_kappa)
            _atomic_npy(root / "features" / "interactions" / "raw_times_activity_gate_display_only.npy", interaction)
        atomic_json(root / "features" / "raw_source_reference.json", {"path": str(config.source.movie_path), "review_interval_ui": list(config.source.review_interval_ui), "immutable": True, "authority": "raw amplitude"})
        synthetic = synthetic_fixture_audit(config, config.sampling.seed)
        atomic_json(root / "evaluation" / "synthetic" / "metrics.json", synthetic)
        sparse_metrics, candidate_rows = sparse_positive_evaluation(
            raw, evidence, config
        )
        atomic_json(
            root / "evaluation" / "fixed_unsupervised" / "metrics.json",
            sparse_metrics,
        )
        _write_tsv(
            root / "evaluation" / "native_proposals" / "candidates.tsv",
            candidate_rows,
        )
        atomic_json(
            root / "evaluation" / "identical_proposals" / "status.json",
            {
                "status": "not_materialized_in_tiny_validation",
                "required_before_real_data_claims": True,
            },
        )
        atomic_json(
            root / "evaluation" / "exploratory_crossfit" / "status.json",
            {
                "status": "not_run",
                "primary_track": False,
                "label_conditioned_fitting_in_primary_track": False,
            },
        )
        diagnostics = render_diagnostics(root, raw, context_ids, [np.load(path, mmap_mode="r") for path in context_paths], [np.load(path, mmap_mode="r") for path in innovation_paths], evidence, dominant, [fit.derivative_angle_distance_degrees for fit in fits])
        metrics = {"context_count": len(definitions), "fit_count": len(fits), "routing_mode": selected_mode, "finite": bool(np.isfinite(evidence).all()), "elapsed_seconds": time.monotonic() - started, "synthetic_fixture_count": synthetic["fixture_count"], "scientific_status": "implementation_validated_only"}
        _extended_diagnostics(root)
        stage_gate = {"A_numerical": "pass", "B_msln_utility": "tiny_fixture_only", "C_ica_incremental": "not_established", "D_energy_calibration": "implementation_pass", "E_routing": "implementation_pass", "F_morphology": "tiny_fixture_only", "G_resources": "pass", "advance_real_study": False}
        atomic_json(root / "metrics.json", metrics); atomic_json(root / "stage_gate.json", stage_gate)
        results = f"# MSLN/MS-ICA results\n\nStatus: implementation-validation run complete.\n\nThis artifact does **not** establish real-data efficacy. Raw amplitude remains the authority; `activity_evidence.npy` is auxiliary evidence and the product interaction is display-only.\n\n- Contexts: {len(definitions)}\n- Fixed routing: `{selected_mode}`\n- Synthetic fixtures: {synthetic['fixture_count']}\n- Full real-data advancement: no\n"
        (root / "RESULTS.md").write_text(results, encoding="utf-8")
        manifest = {"status": "complete", "preflight_fingerprint": preflight["preflight_fingerprint"], "execution_scope": {"requested_fold": fold, "requested_stage": stage, "no_video": no_video}, "artifacts": {"activity_evidence": str(evidence_path.relative_to(root)), "dominant_context": str(dominant_path.relative_to(root)), "diagnostics": diagnostics}, "cross_context_modes": list(cross_rows)}
        atomic_json(root / "run_manifest.json", manifest)
        (root / "run_manifest.partial.json").unlink(missing_ok=True)
        atomic_json(status_path, {"status": "complete", "scientific_status": "implementation_validated_only", "advance_real_study": False})
        return {**metrics, "output_root": str(root), "status": "complete"}
    except Exception as exc:
        atomic_json(status_path, {"status": "partial", "scientific_status": "not_established", "error_type": type(exc).__name__, "error": str(exc)})
        raise


def summarize(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    required = [root / "status.json", root / "metrics.json", root / "stage_gate.json", root / "run_manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete output root; missing: {missing}")
    return {"output_root": str(root), "status": json.loads(required[0].read_text()), "metrics": json.loads(required[1].read_text()), "stage_gate": json.loads(required[2].read_text())}
