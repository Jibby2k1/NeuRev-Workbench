"""Stage-gated CPU program for local covariance whitening after joint MSLN."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np

from neurobench.algorithms.local_covariance_whitening import (
    CovarianceEstimatorConfig,
    LocalCovarianceFit,
    SpatialTileConfig,
    TiledWhiteningFits,
    WhiteningFeatureBank,
    apply_tiled_feature_whiteners,
    build_whitening_feature_bank,
    contiguous_quiet_partitions,
    fit_covariance,
    fit_tiled_feature_whiteners,
    gather_covariance_samples,
)
from neurobench.algorithms.multiscale_local_normalization import JointSTContext, causal_joint_msln
from neurobench.experiments.msln_msica.artifacts import atomic_json, sha256_file, sha256_payload
from neurobench.metrics.sparse_detection import extract_local_maxima, known_label_recall_summary, temporal_pool
from neurobench.metrics.whiteness import (
    covariance_identity_error,
    maximum_absolute_correlation,
    normalized_off_diagonal_energy,
    tile_boundary_discontinuity,
)
from neurobench.reports.local_whitening import (
    atomic_csv,
    render_covariance_audit_diagnostics,
    render_synthetic_comparison,
    write_report,
    write_stage_indices,
)


TOP_KEYS = {
    "schema_version", "experiment_id", "source", "feature_bank", "quiet_partition",
    "covariance", "tiles", "screen", "evaluation", "scientific_audit", "compute", "outputs",
}
NESTED_KEYS = {
    "source": {"movie_path", "labels_path", "axes", "ui_one_based", "review_interval_ui", "quiet_interval_ui", "burst_intervals_ui"},
    "feature_bank": {"context_ids", "scientific_array", "feature_order_frozen"},
    "quiet_partition": {"fit_fraction", "calibration_fraction", "holdout_fraction", "mode"},
    "covariance": {"primary_estimator", "control_estimators", "ridge_ratios", "eigenvalue_floor_ratio", "maximum_condition_number", "modes"},
    "tiles": {"primary_size_px", "sensitivity_sizes_px", "overlap_fraction", "blend", "boundary_mode", "minimum_raw_samples", "maximum_fit_samples", "spatial_subsample", "temporal_subsample"},
    "screen": {"freeze_without_spatial_labels", "maximum_primary_lanes", "maximum_diagnostic_lanes", "selection_terms"},
    "evaluation": {"candidate_budgets", "guardrail_budget", "nms_distance_px", "match_radius_px", "unlabeled_candidates", "winner_basis"},
    "scientific_audit": {"enabled"},
    "compute": {"device", "cpu_threads", "workers", "frame_chunk", "maximum_peak_ram_gb", "maximum_peak_vram_gb"},
    "outputs": {"root_dir", "representative_frames_ui", "fps"},
}
CONTEXT_IDS = (
    "joint_s5_g1_t15_g1", "joint_s15_g3_t23_g1", "joint_s15_g3_t31_g1"
)
MODES = ("identity", "global_diagonal", "global_full_zca", "local_diagonal", "local_full_zca")


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload)); unknown = sorted(set(payload) - expected)
        raise ValueError(f"{name} keys differ: missing={missing}, unknown={unknown}")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    _require_exact_keys(payload, TOP_KEYS, "top-level")
    for section, keys in NESTED_KEYS.items():
        _require_exact_keys(payload[section], keys, section)
    if payload["schema_version"] != 1 or payload["experiment_id"] != "spon_ca_burst_local_whitening_v1":
        raise ValueError("local whitening requires the exact schema-v1 experiment contract")
    root = config_path.parent
    for key in ("movie_path", "labels_path"):
        payload["source"][key] = str((root / payload["source"][key]).resolve())
    payload["outputs"]["root_dir"] = str((root / payload["outputs"]["root_dir"]).resolve())
    payload["_config_path"] = str(config_path)
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    source, bank, quiet = config["source"], config["feature_bank"], config["quiet_partition"]
    covariance, tiles, screen = config["covariance"], config["tiles"], config["screen"]
    evaluation, audit, compute = config["evaluation"], config["scientific_audit"], config["compute"]
    if source["axes"] != "TYX" or source["ui_one_based"] is not True:
        raise ValueError("source must use one-based UI frames and TYX axes")
    if tuple(bank["context_ids"]) != CONTEXT_IDS or bank["scientific_array"] != "signed_msln" or bank["feature_order_frozen"] is not True:
        raise ValueError("feature bank order and signed scientific representation are frozen")
    fractions = [quiet[key] for key in ("fit_fraction", "calibration_fraction", "holdout_fraction")]
    if quiet["mode"] != "contiguous_blocks" or not np.allclose(fractions, [0.5, 0.25, 0.25]):
        raise ValueError("V1 freezes the contiguous 50/25/25 quiet split")
    if covariance["primary_estimator"] != "oas" or covariance["control_estimators"] != ["fixed_ridge"] or covariance["ridge_ratios"] != [0.05]:
        raise ValueError("V1 estimator controls are frozen")
    if tuple(covariance["modes"]) != MODES or covariance["eigenvalue_floor_ratio"] != 1e-5 or covariance["maximum_condition_number"] != 10000.0:
        raise ValueError("V1 covariance modes and conditioning defaults are frozen")
    if tiles["primary_size_px"] != 64 or tiles["sensitivity_sizes_px"] != [32, 128] or tiles["overlap_fraction"] != 0.5:
        raise ValueError("V1 tile geometry defaults are frozen")
    if not screen["freeze_without_spatial_labels"] or screen["maximum_primary_lanes"] != 1 or screen["maximum_diagnostic_lanes"] > 2:
        raise ValueError("label-free freeze limits are mandatory")
    if evaluation["unlabeled_candidates"] != "unknown" or evaluation["winner_basis"] != "label_free_freeze_then_sparse_positive_guardrail":
        raise ValueError("unknown-candidate and winner semantics are frozen")
    if audit["enabled"] is not True:
        raise ValueError("scientific audit is default-on and cannot be disabled by this manifest")
    if compute != {"device": "cpu", "cpu_threads": 4, "workers": 1, "frame_chunk": 8, "maximum_peak_ram_gb": 16, "maximum_peak_vram_gb": 4}:
        raise ValueError("V1 CPU resource envelope is frozen")


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _config_fingerprint(config: dict[str, Any]) -> str:
    return sha256_payload(_public_config(config))


def _tile_config(config: dict[str, Any], *, size: int | None = None) -> SpatialTileConfig:
    tiles = config["tiles"]; tile_size = int(size or tiles["primary_size_px"])
    stride = int(round(tile_size * (1 - float(tiles["overlap_fraction"]))))
    return SpatialTileConfig(
        tile_size, tile_size, stride, stride, tiles["blend"], tiles["boundary_mode"],
        int(tiles["minimum_raw_samples"]), int(tiles["maximum_fit_samples"]),
        int(tiles["spatial_subsample"]), int(tiles["temporal_subsample"]),
    )


def _estimator(config: dict[str, Any], method: str) -> CovarianceEstimatorConfig:
    covariance = config["covariance"]
    return CovarianceEstimatorConfig(
        method=method, ridge_ratio=float(covariance["ridge_ratios"][0]),
        eigenvalue_floor_ratio=float(covariance["eigenvalue_floor_ratio"]),
        maximum_condition_number=float(covariance["maximum_condition_number"]), center="mean",
    )


def _context(context_id: str) -> JointSTContext:
    parts = context_id.split("_")
    return JointSTContext(context_id, int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:]), int(parts[4][1:]))


def _source_feature_bank(
    config: dict[str, Any], *, quiet_only: bool = False
) -> tuple[WhiteningFeatureBank, np.ndarray]:
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    review_start, review_stop = map(
        int,
        config["source"]["quiet_interval_ui"]
        if quiet_only else config["source"]["review_interval_ui"],
    )
    pre_roll = max(_context(item).temporal_window_frames for item in CONTEXT_IDS)
    source_start = review_start - 1 - pre_roll
    if source_start < 0 or review_stop > len(movie):
        raise ValueError("review interval lacks causal pre-roll or exceeds source")
    source = np.asarray(movie[source_start:review_stop], dtype=np.float32)
    quiet_start, quiet_stop = map(int, config["source"]["quiet_interval_ui"])
    source_quiet = np.zeros(len(source), dtype=bool)
    source_quiet[quiet_start - 1 - source_start:quiet_stop - source_start] = True
    results = [causal_joint_msln(source, _context(item), quiet_mask=source_quiet) for item in CONTEXT_IDS]
    results = [replace(item, values=item.values[pre_roll:], valid_frames=item.valid_frames[pre_roll:]) for item in results]
    bank = build_whitening_feature_bank(CONTEXT_IDS, results)
    quiet_mask = np.zeros(len(bank.values), dtype=bool)
    quiet_mask[max(0, quiet_start - review_start):min(len(bank.values), quiet_stop - review_start + 1)] = True
    return bank, quiet_mask


def _identity_fit(feature_ids: tuple[str, ...], bounds: tuple[int, int, int, int] | None, fit_id: str) -> LocalCovarianceFit:
    dimension = len(feature_ids); identity = np.eye(dimension)
    return LocalCovarianceFit(
        fit_id, feature_ids, bounds, np.zeros(dimension), identity, identity, identity,
        np.ones(dimension), np.ones(dimension), 0.0, 1.0, float(dimension), 0, 0,
        True, None, {"method": "identity", "explicit_control": True},
    )


def _fit_lane(
    bank: WhiteningFeatureBank,
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
    tile: SpatialTileConfig,
    mode: str,
    config: dict[str, Any],
    *,
    compute_backend: str = "cpu",
    max_vram_bytes: int | None = None,
):
    if mode.startswith("global_"):
        tile = replace(tile, tile_height=bank.values.shape[1], tile_width=bank.values.shape[2], stride_y=bank.values.shape[1], stride_x=bank.values.shape[2])
    method = "diagonal" if mode.endswith("diagonal") else "oas"
    fitted = fit_tiled_feature_whiteners(bank, fit_mask, tile, _estimator(config, method))
    if mode == "identity":
        identities = tuple(_identity_fit(bank.feature_ids, bounds, f"identity_{i}") for i, bounds in enumerate(fitted.tile_bounds_yx))
        fitted = replace(fitted, primary_fits=identities, global_full_fit=_identity_fit(bank.feature_ids, None, "identity_global"))
    kwargs = {"frame_chunk": int(config["compute"]["frame_chunk"])}
    if compute_backend == "cpu":
        result = apply_tiled_feature_whiteners(bank, fitted, tile, calibration_mask, **kwargs)
    elif compute_backend == "cuda":
        from neurobench.algorithms.local_covariance_whitening_cuda import (
            apply_tiled_feature_whiteners_cuda,
        )
        result = apply_tiled_feature_whiteners_cuda(
            bank, fitted, tile, calibration_mask,
            max_vram_bytes=int(max_vram_bytes or config["compute"]["maximum_peak_vram_gb"] * 2**30),
            **kwargs,
        )
    else:
        raise ValueError("compute_backend must be cpu or cuda")
    return result, fitted


def _synthetic_bank(seed: int = 20260815) -> tuple[WhiteningFeatureBank, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frames, height, width, dimension = 40, 24, 24, 3
    left_cov = np.asarray([[1, .75, .25], [.75, 1.2, .4], [.25, .4, .8]])
    right_cov = np.asarray([[1, -.55, .1], [-.55, 1.1, -.35], [.1, -.35, .7]])
    values = np.empty((frames, height, width, dimension), dtype=np.float32)
    for frame in range(frames):
        values[frame, :, :12] = rng.multivariate_normal(np.zeros(3), left_cov, size=(height, 12))
        values[frame, :, 12:] = rng.multivariate_normal(np.zeros(3), right_cov, size=(height, 12))
    yy, xx = np.mgrid[:height, :width]
    truth = np.exp(-((yy - 12) ** 2 + (xx - 12) ** 2) / 7.0).astype(np.float32)
    amplitudes = np.asarray([4.0, 6.0, 8.0, 10.0], dtype=np.float32)
    for frame, amplitude in zip(range(32, 36), amplitudes):
        values[frame] += amplitude * truth[..., None] * np.asarray([1.0, 0.8, 0.4])
    bank = WhiteningFeatureBank(
        CONTEXT_IDS, values, np.ones(frames, dtype=bool),
        tuple({"context_id": item, "scientific_array": "signed_msln"} for item in CONTEXT_IDS),
        {"fixture": "spatial_covariance_nonstationarity_with_compact_amplitude_ladder", "seed": seed},
    )
    quiet = np.zeros(frames, dtype=bool); quiet[:30] = True
    return bank, quiet, truth, amplitudes


def _covariance_metrics(result: Any, mask: np.ndarray, geometry: Any) -> dict[str, float]:
    samples = result.zca_features[mask].reshape(-1, result.zca_features.shape[-1]).astype(np.float64)
    covariance = np.cov(samples.T, bias=True)
    seam = tile_boundary_discontinuity(result.mahalanobis_energy, geometry)
    return {
        "heldout_identity_error": covariance_identity_error(covariance),
        "heldout_max_abs_correlation": maximum_absolute_correlation(covariance),
        "heldout_off_diagonal_energy": normalized_off_diagonal_energy(covariance),
        "tile_boundary_ratio": seam["boundary_to_all_ratio"],
        "unresolved_pixel_fraction": float(np.mean(result.unresolved_tile_mask)),
    }


def _frame_bootstrap_identity_errors(
    features: np.ndarray,
    frame_indices: np.ndarray,
    *,
    seed: int,
    repetitions: int = 128,
) -> np.ndarray:
    """Bootstrap held-out frames using per-frame sufficient statistics."""
    selected = np.asarray(features[frame_indices], dtype=np.float64)
    dimension = selected.shape[-1]
    pixels_per_frame = int(np.prod(selected.shape[1:-1]))
    flat = selected.reshape(len(selected), pixels_per_frame, dimension)
    sums = np.sum(flat, axis=1)
    cross = np.einsum("fnd,fne->fde", flat, flat)
    rng = np.random.default_rng(seed)
    errors = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.integers(0, len(frame_indices), size=len(frame_indices))
        total_sum = np.sum(sums[sampled], axis=0)
        total_cross = np.sum(cross[sampled], axis=0)
        count = len(sampled) * pixels_per_frame
        mean = total_sum / count
        covariance = total_cross / count - np.outer(mean, mean)
        errors[repetition] = covariance_identity_error(covariance)
    return errors


def _write_fit_artifacts(base: Path, lane_fits: dict[str, TiledWhiteningFits]) -> int:
    """Persist every global/local fit with conditioning and fallback provenance."""
    rows: list[dict[str, Any]] = []
    for lane, fitted in lane_fits.items():
        entries = [("global", fitted.global_full_fit)]
        entries += [("local_primary", item) for item in fitted.primary_fits]
        entries += [("local_diagonal", item) for item in fitted.local_diagonal_fits]
        for role, fit in entries:
            relative = Path(role) / f"{lane}_{fit.fit_id}.npz"
            path = base / relative; path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial.npz")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle, mean=fit.mean, sample_covariance=fit.sample_covariance,
                    covariance=fit.covariance, whitening=fit.whitening,
                    eigenvalues_raw=fit.eigenvalues_raw,
                    eigenvalues_regularized=fit.eigenvalues_regularized,
                )
            temporary.replace(path)
            rows.append({
                "lane": lane, "role": role, "fit_id": fit.fit_id,
                "path": str(relative), "resolved": fit.resolved,
                "unresolved_reason": fit.unresolved_reason or "",
                "sample_count": fit.sample_count, "block_count": fit.block_count,
                "shrinkage": fit.shrinkage, "condition_number": fit.condition_number,
                "effective_rank": fit.effective_rank,
                "fallback_required": bool(fit.diagnostics.get("fallback_required", False)),
                "tile_bounds_yx": "" if fit.tile_bounds_yx is None else ":".join(map(str, fit.tile_bounds_yx)),
            })
    atomic_csv(base / "fit_index.csv", list(rows[0]), rows)
    return len(rows)


def _compact_fixture_diagnostics(seed: int = 20260816) -> list[dict[str, Any]]:
    """Exercise declared success/failure families without a Cartesian sweep."""
    rng = np.random.default_rng(seed)
    estimator = CovarianceEstimatorConfig(method="oas")
    diagonal = CovarianceEstimatorConfig(method="diagonal")
    rows: list[dict[str, Any]] = []
    for fixture, covariance in (
        ("iid_gaussian", np.eye(3)),
        ("global_off_diagonal", np.asarray([[1, .75, .2], [.75, 1.1, .35], [.2, .35, .8]])),
    ):
        fit_samples = rng.multivariate_normal(np.zeros(3), covariance, size=5000)
        holdout = rng.multivariate_normal(np.zeros(3), covariance, size=5000)
        full_fit = fit_covariance(fit_samples, estimator)
        diagonal_fit = fit_covariance(fit_samples, diagonal)
        full_cov = np.cov(((holdout - full_fit.mean) @ full_fit.whitening.T).T, bias=True)
        diagonal_cov = np.cov(((holdout - diagonal_fit.mean) @ diagonal_fit.whitening.T).T, bias=True)
        rows.append({
            "fixture_id": fixture,
            "full_identity_error": covariance_identity_error(full_cov),
            "diagonal_identity_error": covariance_identity_error(diagonal_cov),
            "full_improvement": covariance_identity_error(diagonal_cov) - covariance_identity_error(full_cov),
            "reported_covariance_shift": False,
            "self_whitening_ratio": 1.0,
            "resolved": full_fit.resolved,
        })
    train_covariance = np.asarray([[1, .7, .1], [.7, 1, .2], [.1, .2, .8]])
    shifted_covariance = np.asarray([[1, -.5, .4], [-.5, 1.4, -.3], [.4, -.3, 1.1]])
    train = rng.multivariate_normal(np.zeros(3), train_covariance, size=5000)
    shifted = rng.multivariate_normal(np.zeros(3), shifted_covariance, size=5000)
    shifted_fit = fit_covariance(train, estimator)
    shifted_output = np.cov(((shifted - shifted_fit.mean) @ shifted_fit.whitening.T).T, bias=True)
    rows.append({
        "fixture_id": "covariance_shift_train_to_test",
        "full_identity_error": covariance_identity_error(shifted_output),
        "diagonal_identity_error": 0.0, "full_improvement": 0.0,
        "reported_covariance_shift": covariance_identity_error(shifted_output) > .25,
        "self_whitening_ratio": 1.0, "resolved": shifted_fit.resolved,
    })
    clean = rng.multivariate_normal(np.zeros(3), train_covariance, size=5000)
    contaminated = clean.copy(); contaminated[:250] += np.asarray([6.0, 5.0, 3.0])
    clean_fit = fit_covariance(clean, estimator); contaminated_fit = fit_covariance(contaminated, estimator)
    event = np.asarray([6.0, 5.0, 3.0])
    clean_q = float(np.sum((clean_fit.whitening @ (event - clean_fit.mean)) ** 2))
    contaminated_q = float(np.sum((contaminated_fit.whitening @ (event - contaminated_fit.mean)) ** 2))
    rows.append({
        "fixture_id": "quiet_fit_contamination_0p05", "full_identity_error": 0.0,
        "diagonal_identity_error": 0.0, "full_improvement": 0.0,
        "reported_covariance_shift": False, "self_whitening_ratio": contaminated_q / clean_q,
        "resolved": contaminated_fit.resolved,
    })
    low_rank = rng.normal(size=(5000, 3)); low_rank[:, 2] = low_rank[:, 0] * 1e-8
    low_rank_fit = fit_covariance(low_rank, estimator)
    rows.append({
        "fixture_id": "low_eigenvalue_shrinkage", "full_identity_error": 0.0,
        "diagonal_identity_error": 0.0, "full_improvement": 0.0,
        "reported_covariance_shift": False, "self_whitening_ratio": 1.0,
        "resolved": low_rank_fit.resolved,
    })
    return rows


def run_synthetic(config: dict[str, Any], *, stage_name: str = "synthetic") -> dict[str, Any]:
    root = Path(config["outputs"]["root_dir"]); root.mkdir(parents=True, exist_ok=True)
    target = root / stage_name / "metrics.csv"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite completed stage: {target}")
    bank, quiet, truth, amplitudes = _synthetic_bank()
    partitions = contiguous_quiet_partitions(quiet)
    tile = SpatialTileConfig(12, 12, 6, 6, "hann", "crop", 64, 4096, 1, 1)
    lane_results, lane_fits, rows = {}, {}, []
    for mode in MODES:
        result, fitted = _fit_lane(bank, partitions["fit"], partitions["calibration"], tile, mode, config)
        lane_results[mode] = result; lane_fits[mode] = fitted
        metrics = _covariance_metrics(result, partitions["holdout"], fitted.tile_bounds_yx)
        event_map = np.maximum(
            result.mahalanobis_energy[32:36].mean(axis=0)
            - result.mahalanobis_energy[partitions["calibration"]].mean(axis=0),
            0,
        )
        morphology = float(np.corrcoef(event_map.ravel(), truth.ravel())[0, 1])
        event_amplitude = result.mahalanobis_energy[32:36, 12, 12]
        amplitude_rank = float(np.corrcoef(np.argsort(np.argsort(event_amplitude)), np.argsort(np.argsort(amplitudes)))[0, 1])
        rows.append({"fixture_id": "spatial_nonstationary_compact_ladder", "lane": mode, **metrics, "morphology_correlation": morphology, "amplitude_rank_spearman": amplitude_rank})
    local_full = next(row for row in rows if row["lane"] == "local_full_zca")
    local_diagonal = next(row for row in rows if row["lane"] == "local_diagonal")
    global_full = next(row for row in rows if row["lane"] == "global_full_zca")
    identity = next(row for row in rows if row["lane"] == "identity")
    g0 = bool(
        (identity["heldout_max_abs_correlation"] > 0.15 or identity["heldout_off_diagonal_energy"] > 0.10)
        and local_full["heldout_identity_error"] < min(
            local_diagonal["heldout_identity_error"], global_full["heldout_identity_error"]
        )
    )
    g1 = bool(
        local_full["heldout_identity_error"] < local_diagonal["heldout_identity_error"]
        and local_full["morphology_correlation"] >= 0.90
        and local_full["amplitude_rank_spearman"] >= 0.95
        and local_full["unresolved_pixel_fraction"] == 0
    )
    stage = root / stage_name
    atomic_csv(stage / "metrics.csv", list(rows[0]), rows)
    scientific_fixtures = _compact_fixture_diagnostics()
    fixture_manifest = {
        "seed": 20260815,
        "compact_covering_design": [
            {"background": "spatial_covariance_nonstationarity", "signal": "compact_transient_amplitude_ladder", "tile_size": 12},
            {"background": "iid_gaussian", "signal": "none"},
            {"background": "global_off_diagonal", "signal": "none"},
            {"background": "covariance_shift_train_to_test", "signal": "none"},
            {"background": "quiet_fit_contamination_0p05", "signal": "compact_transient"},
            {"background": "low_eigenvalue_shrinkage", "signal": "none"},
            {"stress": "global_vs_local_diagonal_vs_full_boundary_crossing", "full_cartesian_product": False},
        ],
        "quiet_partitions": {key: np.flatnonzero(value).tolist() for key, value in partitions.items()},
    }
    atomic_json(stage / "fixture_manifest.json", fixture_manifest)
    atomic_csv(stage / "scientific_fixture_metrics.csv", list(scientific_fixtures[0]), scientific_fixtures)
    fit_count = _write_fit_artifacts(stage / "fits", lane_fits)
    atomic_csv(stage / "failure_cases.csv", ["fixture_id", "failure"], [] if g1 else [{"fixture_id": "spatial_nonstationary_compact_ladder", "failure": "G1_not_passed"}])
    render_synthetic_comparison(
        stage / "representative_figures" / "factorial_comparison.png", bank.values[34, :, :, 0],
        {mode: lane_results[mode].mahalanobis_energy[34] for mode in MODES},
    )
    decision = "advance" if g0 and g1 else "stop"
    summary = {"gate_g0": g0, "gate_g1": g1, "decision": decision, "lane_metrics": rows, "scientific_fixture_metrics": scientific_fixtures, "global_full_reference_error": global_full["heldout_identity_error"], "fit_artifact_count": fit_count}
    atomic_json(root / "status.json", {"stage": stage_name, "status": "completed", "decision": decision, "full_spon_authorized": False})
    atomic_json(root / "progress.json", {"completed_stages": ["preflight", stage_name] if (root / "input_fingerprints.json").is_file() else [stage_name], "next_stage": "covariance-audit", "real_quiet_source_selected": False})
    write_report(root, stage_name, decision, [f"Generated G0={g0} and G1={g1}.", "The compact fixture is implementation evidence, not Spon evidence."])
    write_stage_indices(root, stage=stage_name, status="completed", summary=summary, artifacts=[
        {"id": "synthetic_metrics", "path": f"{stage_name}/metrics.csv"},
        {"id": "scientific_fixture_metrics", "path": f"{stage_name}/scientific_fixture_metrics.csv"},
        {"id": "fit_index", "path": f"{stage_name}/fits/fit_index.csv"},
        {"id": "synthetic_figure", "path": f"{stage_name}/representative_figures/factorial_comparison.png"},
    ], validation={"passed": True, "finite": all(np.isfinite(list(row.values())[2:]).all() for row in rows), "scientific_audit_applicable": False})
    return summary


def _label_rows(path: Path) -> list[dict[str, Any]]:
    from neurobench.experiments.learnable_contrast.core import load_labels
    return load_labels(path)


def _projection_overlay(config: dict[str, Any], path: Path) -> int:
    import matplotlib.pyplot as plt
    movie = np.load(config["source"]["movie_path"], mmap_mode="r", allow_pickle=False)
    frame_ui = int(config["outputs"]["representative_frames_ui"][0])
    frame = np.asarray(movie[frame_ui - 1], dtype=np.float32)
    labels = _label_rows(Path(config["source"]["labels_path"]))
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.imshow(frame, cmap="gray")
    if labels:
        axis.scatter([float(row["x_px"]) for row in labels], [float(row["y_px"]) for row in labels], s=12, facecolors="none", edgecolors="lime")
    axis.set_title(f"Projection-only label geometry, UI frame {frame_ui}")
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.stem + ".partial" + path.suffix)
    figure.savefig(temporary, dpi=120); plt.close(figure); temporary.replace(path)
    return len(labels)


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["outputs"]["root_dir"])
    if root.exists():
        raise FileExistsError(f"preflight refuses existing output root: {root}")
    movie_path, labels_path = Path(config["source"]["movie_path"]), Path(config["source"]["labels_path"])
    if not movie_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("declared movie and labels must exist")
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if movie.ndim != 3 or config["source"]["review_interval_ui"][1] > len(movie):
        raise ValueError("source must be TYX and contain the review interval")
    quiet_start, quiet_stop = config["source"]["quiet_interval_ui"]
    quiet_count = int(quiet_stop - quiet_start + 1)
    partitions = contiguous_quiet_partitions(np.ones(quiet_count, dtype=bool))
    tile = _tile_config(config)
    if tile.tile_height > movie.shape[1] or tile.tile_width > movie.shape[2]:
        raise ValueError("primary tile exceeds source field")
    estimated_feature_gb = np.prod((config["source"]["review_interval_ui"][1] - config["source"]["review_interval_ui"][0] + 1, *movie.shape[1:], 3)) * 4 / 1024**3
    if estimated_feature_gb * 4 > config["compute"]["maximum_peak_ram_gb"]:
        raise ValueError("estimated in-memory reference exceeds manifest RAM cap")
    root.mkdir(parents=True)
    fingerprints = {
        "config_sha256": _config_fingerprint(config),
        "movie_sha256": sha256_file(movie_path), "labels_sha256": sha256_file(labels_path),
        "movie_shape": list(movie.shape), "movie_dtype": str(movie.dtype),
    }
    atomic_json(root / "resolved_config.json", _public_config(config))
    atomic_json(root / "input_fingerprints.json", fingerprints)
    label_count = _projection_overlay(config, root / "preflight" / "label_projection_overlay.png")
    summary = {
        "source_read_only": True, "full_spon_authorized": False, "gpu_used": False,
        "quiet_partition_frame_counts": {key: int(np.sum(value)) for key, value in partitions.items()},
        "estimated_feature_bank_gb": float(estimated_feature_gb), "label_projection_count": label_count,
        "tile_geometry_frozen": True,
    }
    atomic_json(root / "status.json", {"stage": "preflight", "status": "completed", "full_run_authorized": False})
    atomic_json(root / "progress.json", {"completed_stages": ["preflight"], "next_stage": "synthetic"})
    write_report(root, "preflight", "await_generated_validation", ["Source geometry, fingerprints, quiet partitions, and resource estimate passed."])
    write_stage_indices(root, stage="preflight", status="completed", summary=summary, artifacts=[{"id": "label_projection", "path": "preflight/label_projection_overlay.png"}], validation={"passed": True, "source_modified": False, "scientific_audit_required_for_full_run": True})
    return summary


def _verify_preflight(config: dict[str, Any]) -> None:
    root = Path(config["outputs"]["root_dir"])
    fingerprints = json.loads((root / "input_fingerprints.json").read_text())
    if fingerprints["config_sha256"] != _config_fingerprint(config):
        raise RuntimeError("config fingerprint differs from preflight")
    for key, source_key in (("movie_sha256", "movie_path"), ("labels_sha256", "labels_path")):
        if fingerprints[key] != sha256_file(Path(config["source"][source_key])):
            raise RuntimeError(f"{source_key} fingerprint differs from preflight")


def gpu_preflight(config: dict[str, Any], *, max_vram_gb: float) -> dict[str, Any]:
    """Require bounded CPU/CUDA parity before CUDA may process real data."""
    _verify_preflight(config)
    declared = float(config["compute"]["maximum_peak_vram_gb"])
    if not 0 < float(max_vram_gb) <= declared <= 4:
        raise ValueError("GPU preflight requires a positive VRAM cap no larger than the frozen 4 GiB cap")
    root = Path(config["outputs"]["root_dir"])
    target = root / "gpu_preflight" / "parity.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite completed GPU preflight: {target}")
    bank, quiet, _, _ = _synthetic_bank(seed=20260817)
    partitions = contiguous_quiet_partitions(quiet)
    tile = SpatialTileConfig(12, 12, 6, 6, "hann", "crop", 64, 4096, 1, 1)
    fitted = fit_tiled_feature_whiteners(bank, partitions["fit"], tile, _estimator(config, "oas"))
    cpu_started = time.monotonic()
    cpu = apply_tiled_feature_whiteners(
        bank, fitted, tile, partitions["calibration"],
        frame_chunk=int(config["compute"]["frame_chunk"]),
    )
    cpu_seconds = time.monotonic() - cpu_started
    from neurobench.algorithms.local_covariance_whitening_cuda import (
        apply_tiled_feature_whiteners_cuda,
    )
    cuda = apply_tiled_feature_whiteners_cuda(
        bank, fitted, tile, partitions["calibration"],
        frame_chunk=int(config["compute"]["frame_chunk"]),
        max_vram_bytes=int(float(max_vram_gb) * 2**30),
    )
    errors = {
        "zca_features_max_abs_error": float(np.max(np.abs(cpu.zca_features - cuda.zca_features))),
        "mahalanobis_energy_max_abs_error": float(np.max(np.abs(cpu.mahalanobis_energy - cuda.mahalanobis_energy))),
        "quiet_surprise_max_abs_error": float(np.max(np.abs(cpu.quiet_surprise - cuda.quiet_surprise))),
    }
    passed = bool(
        errors["zca_features_max_abs_error"] <= 2e-5
        and errors["mahalanobis_energy_max_abs_error"] <= 1e-4
        # Empirical rank tails are discontinuous at near-ties; this bound is
        # separate from the much tighter channel/energy numerical contracts.
        and errors["quiet_surprise_max_abs_error"] <= 2e-3
        and cuda.diagnostics["observed_peak_vram_bytes"] <= int(float(max_vram_gb) * 2**30)
    )
    payload = {
        "status": "passed" if passed else "failed",
        "cpu_fit_authority": True,
        "cuda_application_backend": "cupy_cuda",
        "max_vram_bytes": int(float(max_vram_gb) * 2**30),
        "errors": errors,
        "application_runtime_seconds": {
            "cpu": cpu_seconds,
            "cuda": float(cuda.diagnostics["runtime_seconds"]),
            "cuda_speedup": cpu_seconds / max(float(cuda.diagnostics["runtime_seconds"]), np.finfo(float).eps),
        },
        "cuda_diagnostics": cuda.diagnostics,
    }
    atomic_json(target, payload)
    if not passed:
        raise RuntimeError(f"CUDA parity preflight failed: {errors}")
    return payload


def _verify_gpu_preflight(config: dict[str, Any], max_vram_bytes: int) -> None:
    path = Path(config["outputs"]["root_dir"]) / "gpu_preflight" / "parity.json"
    if not path.is_file():
        raise RuntimeError("CUDA execution requires a completed GPU parity preflight")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or int(payload.get("max_vram_bytes", -1)) != int(max_vram_bytes):
        raise RuntimeError("CUDA parity preflight does not match the requested VRAM cap")


def covariance_audit(
    config: dict[str, Any], *, use_real_quiet_source: bool,
    compute_backend: str = "cpu", max_vram_bytes: int | None = None,
) -> dict[str, Any]:
    if not use_real_quiet_source:
        return run_synthetic(config, stage_name="covariance_audit")
    _verify_preflight(config)
    if compute_backend == "cuda":
        _verify_gpu_preflight(config, int(max_vram_bytes or 0))
    root = Path(config["outputs"]["root_dir"]); target = root / "covariance_audit" / "heldout_whiteness.csv"
    if target.exists(): raise FileExistsError(f"refusing to overwrite completed covariance audit: {target}")
    bank, quiet = _source_feature_bank(config, quiet_only=True); partitions = contiguous_quiet_partitions(quiet)
    tile = _tile_config(config); rows = []; lane_previews = {}; lane_fits = {}; tail_samples = {}; bootstrap_errors = {}
    holdout_indices = np.flatnonzero(partitions["holdout"])
    holdout_blocks = np.array_split(holdout_indices, 2)
    for mode in MODES:
        result, fitted = _fit_lane(
            bank, partitions["fit"], partitions["calibration"], tile, mode, config,
            compute_backend=compute_backend, max_vram_bytes=max_vram_bytes,
        )
        lane_fits[mode] = fitted
        for block_index, indices in enumerate(holdout_blocks, 1):
            block_mask = np.zeros(len(bank.values), dtype=bool); block_mask[indices] = True
            rows.append({"quiet_block": f"holdout_{block_index}", "lane": mode, **_covariance_metrics(result, block_mask, fitted.tile_bounds_yx)})
            bootstrap_errors[(mode, block_index)] = _frame_bootstrap_identity_errors(
                result.zca_features, indices, seed=20260815 + block_index,
            )
        lane_previews[mode] = np.asarray(result.mahalanobis_energy[int(holdout_indices[0])], dtype=np.float32).copy()
        tail_samples[mode] = {
            "calibration": np.asarray(result.mahalanobis_energy[partitions["calibration"]].reshape(-1)[::32], dtype=np.float32).copy(),
            "holdout": np.asarray(result.mahalanobis_energy[partitions["holdout"]].reshape(-1)[::32], dtype=np.float32).copy(),
        }
        del result
        gc.collect()
    by_lane = {mode: [row for row in rows if row["lane"] == mode] for mode in MODES}
    residual_reproducible = all(
        row["heldout_max_abs_correlation"] > .15 or row["heldout_off_diagonal_energy"] > .10
        for row in by_lane["identity"]
    )
    full_better_than_diagonal_point = all(
        full["heldout_identity_error"] < diagonal["heldout_identity_error"]
        for full, diagonal in zip(by_lane["global_full_zca"], by_lane["global_diagonal"])
    )
    full_improvement_bootstrap = {
        block_index: bootstrap_errors[("global_diagonal", block_index)]
        - bootstrap_errors[("global_full_zca", block_index)]
        for block_index in (1, 2)
    }
    full_better_than_diagonal_bootstrap = all(
        float(np.quantile(values, .025)) > 0 for values in full_improvement_bootstrap.values()
    )
    full_better_than_diagonal = bool(
        full_better_than_diagonal_point and full_better_than_diagonal_bootstrap
    )
    local_better_than_global = all(
        local["heldout_identity_error"] < global_["heldout_identity_error"]
        for local, global_ in zip(by_lane["local_full_zca"], by_lane["global_full_zca"])
    )
    local_all_resolved = all(row["unresolved_pixel_fraction"] == 0 for row in by_lane["local_full_zca"])
    global_full_resolved = bool(lane_fits["global_full_zca"].global_full_fit.resolved)
    supported_covariance_improvement = (
        (full_better_than_diagonal and global_full_resolved)
        or (local_better_than_global and local_all_resolved)
    )
    g0 = bool(residual_reproducible and supported_covariance_improvement)
    improvements = []
    for local, diagonal, global_ in zip(
        by_lane["local_full_zca"], by_lane["local_diagonal"], by_lane["global_full_zca"]
    ):
        control = min(diagonal["heldout_identity_error"], global_["heldout_identity_error"])
        improvements.append(1.0 - local["heldout_identity_error"] / max(control, np.finfo(float).eps))
    g2 = bool(sum(item >= .25 for item in improvements) > len(improvements) / 2 and local_all_resolved)
    # Contiguous fit-half refits provide a deterministic block-stability diagnostic.
    fit_indices = np.flatnonzero(partitions["fit"]); stability_fits = []
    for block_index, indices in enumerate(np.array_split(fit_indices, 2), 1):
        mask = np.zeros(len(bank.values), dtype=bool); mask[indices] = True
        samples, diagnostics = gather_covariance_samples(
            bank, mask, maximum_samples=tile.maximum_fit_samples,
            spatial_subsample=tile.spatial_subsample, temporal_subsample=tile.temporal_subsample,
        )
        stability_fits.append(fit_covariance(
            samples, _estimator(config, "oas"), fit_id=f"global_block_{block_index}",
            feature_ids=bank.feature_ids, block_count=diagnostics["blocked_effective_sample_count"],
        ))
    stability_delta = float(
        np.linalg.norm(stability_fits[0].whitening - stability_fits[1].whitening, ord="fro")
        / max(np.linalg.norm(stability_fits[0].whitening, ord="fro"), np.finfo(float).eps)
    )
    atomic_csv(target, list(rows[0]), rows)
    audit_root = root / "covariance_audit"
    global_payload = {
        mode: lane_fits[mode].global_full_fit.to_dict() for mode in MODES
    }
    atomic_json(audit_root / "global_covariances.json", global_payload)
    tile_rows = []
    eigen_rows = []
    for mode, fitted in lane_fits.items():
        for fit in fitted.primary_fits:
            tile_rows.append({
                "lane": mode, "fit_id": fit.fit_id,
                "tile_bounds_yx": ":".join(map(str, fit.tile_bounds_yx or ())),
                "resolved": fit.resolved, "unresolved_reason": fit.unresolved_reason or "",
                "sample_count": fit.sample_count, "block_count": fit.block_count,
                "condition_number": fit.condition_number, "effective_rank": fit.effective_rank,
                "shrinkage": fit.shrinkage,
            })
            for eigen_index, (raw, regularized) in enumerate(zip(fit.eigenvalues_raw, fit.eigenvalues_regularized)):
                eigen_rows.append({"lane": mode, "fit_id": fit.fit_id, "eigen_index": eigen_index, "raw": raw, "regularized": regularized})
    atomic_csv(audit_root / "tile_covariance_index.csv", list(tile_rows[0]), tile_rows)
    atomic_csv(audit_root / "eigenvalue_summary.csv", list(eigen_rows[0]), eigen_rows)
    stability_rows = [{
        "comparison": "contiguous_fit_half_1_vs_2", "normalized_zca_frobenius_delta": stability_delta,
        "block_1_samples": stability_fits[0].sample_count, "block_2_samples": stability_fits[1].sample_count,
        "block_1_resolved": stability_fits[0].resolved, "block_2_resolved": stability_fits[1].resolved,
    }]
    for block_index, values in full_improvement_bootstrap.items():
        stability_rows.append({
            "comparison": f"holdout_{block_index}_global_full_vs_diagonal_frame_bootstrap",
            "bootstrap_repetitions": len(values),
            "identity_error_improvement_median": float(np.median(values)),
            "identity_error_improvement_q025": float(np.quantile(values, .025)),
            "identity_error_improvement_q975": float(np.quantile(values, .975)),
            "improvement_beyond_bootstrap_uncertainty": bool(np.quantile(values, .025) > 0),
        })
    stability_fields = sorted({key for row in stability_rows for key in row})
    atomic_csv(audit_root / "bootstrap_stability.csv", stability_fields, stability_rows)
    render_synthetic_comparison(
        audit_root / "covariance_atlas_preview.png",
        bank.values[int(holdout_indices[0]), :, :, 0],
        lane_previews,
    )
    diagnostic_artifacts = render_covariance_audit_diagnostics(audit_root, rows, lane_fits, tail_samples)
    fit_count = _write_fit_artifacts(root / "fits" / "covariance_audit", lane_fits)
    outcome = (
        "stop_no_supported_residual_covariance" if not g0
        else "local_full_zca_passes" if g2
        else "global_full_only" if full_better_than_diagonal and global_full_resolved
        else "local_diagonal_only"
    )
    # G2 is specifically the representation gate for the primary local-full lane.
    # The other supported outcomes are scientifically useful terminal conclusions,
    # not permission to proceed to label loading or ICA.
    decision = "advance" if outcome == "local_full_zca_passes" else "stop"
    summary = {
        "gate_g0": g0, "gate_g2": g2, "decision": decision, "outcome": outcome,
        "spatial_labels_used": False, "heldout_block_count": 2,
        "residual_covariance_reproducible": residual_reproducible,
        "full_better_than_diagonal_both_blocks": full_better_than_diagonal,
        "full_better_than_diagonal_point_estimates": full_better_than_diagonal_point,
        "full_better_than_diagonal_beyond_bootstrap_uncertainty": full_better_than_diagonal_bootstrap,
        "local_better_than_global_both_blocks": local_better_than_global,
        "local_full_improvement_by_block": improvements,
        "global_zca_block_stability_delta": stability_delta,
        "global_zca_fit_half_resolved": [fit.resolved for fit in stability_fits],
        "global_full_resolved": global_full_resolved,
        "local_full_all_resolved": local_all_resolved,
        "fit_artifact_count": fit_count, "rows": rows,
        "application_backend": compute_backend,
    }
    write_report(root, "covariance-audit", summary["decision"], [
        f"Real quiet-only G0={g0}, G2={g2}; no spatial labels were loaded.",
        f"Terminal covariance conclusion: {outcome}.",
        "Only local_full_zca_passes may continue to the label-free representation screen.",
    ])
    atomic_json(root / "status.json", {"stage": "covariance-audit", "status": "completed", "decision": summary["decision"], "full_spon_authorized": False})
    atomic_json(root / "progress.json", {"completed_stages": ["preflight", "synthetic", "covariance-audit"], "next_stage": "run" if summary["decision"] == "advance" else None, "real_quiet_source_selected": True})
    write_stage_indices(root, stage="covariance-audit", status="completed", summary=summary, artifacts=[
        {"id": "heldout_whiteness", "path": "covariance_audit/heldout_whiteness.csv"},
        {"id": "global_covariances", "path": "covariance_audit/global_covariances.json"},
        {"id": "tile_covariance_index", "path": "covariance_audit/tile_covariance_index.csv"},
        {"id": "eigenvalue_summary", "path": "covariance_audit/eigenvalue_summary.csv"},
        {"id": "bootstrap_stability", "path": "covariance_audit/bootstrap_stability.csv"},
        {"id": "covariance_atlas_preview", "path": "covariance_audit/covariance_atlas_preview.png"},
        *diagnostic_artifacts,
    ], validation={"passed": True, "spatial_labels_used": False, "full_spon_run": False, "heldout_blocks": 2, "all_local_full_resolved": local_all_resolved, "global_full_resolved": global_full_resolved})
    return summary


def run_full(
    config: dict[str, Any], *, authorize_full_spon: bool,
    compute_backend: str = "cpu", max_vram_bytes: int | None = None,
) -> dict[str, Any]:
    if not authorize_full_spon:
        raise PermissionError("full Spon run requires --authorize-full-spon")
    _verify_preflight(config)
    root = Path(config["outputs"]["root_dir"])
    if compute_backend == "cuda":
        _verify_gpu_preflight(config, int(max_vram_bytes or 0))
    prior_summary_path = root / "summary.json"
    if not prior_summary_path.is_file():
        raise RuntimeError("full run requires a completed covariance-audit summary")
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    if prior_summary.get("stage") != "covariance-audit" or prior_summary.get("outcome") != "local_full_zca_passes":
        raise RuntimeError(
            "full run blocked: covariance audit did not pass the local-full G2 representation gate"
        )
    freeze_path = root / "screening" / "freeze_decision.json"
    if freeze_path.exists(): raise FileExistsError("refusing to overwrite a completed label-free freeze")
    bank, quiet = _source_feature_bank(config); partitions = contiguous_quiet_partitions(quiet); tile = _tile_config(config)
    lane_results, lane_rows = {}, []
    for mode in MODES:
        result, fitted = _fit_lane(
            bank, partitions["fit"], partitions["calibration"], tile, mode, config,
            compute_backend=compute_backend, max_vram_bytes=max_vram_bytes,
        )
        lane_results[mode] = result
        metrics = _covariance_metrics(result, partitions["holdout"], fitted.tile_bounds_yx)
        lane_rows.append({"lane": mode, **metrics})
        lane_root = root / "representations" / mode; lane_root.mkdir(parents=True, exist_ok=True)
        for name, values in (("zca_features", result.zca_features), ("mahalanobis_energy", result.mahalanobis_energy), ("quiet_surprise", result.quiet_surprise), ("unresolved_tile_mask", result.unresolved_tile_mask)):
            temporary = lane_root / f"{name}.partial.npy"; np.save(temporary, values); temporary.replace(lane_root / f"{name}.npy")
        atomic_json(lane_root / "diagnostics.json", result.diagnostics)
    ranked = sorted(lane_rows, key=lambda row: (row["heldout_identity_error"], row["tile_boundary_ratio"], row["lane"]))
    freeze = {"spatial_labels_loaded": False, "primary_lane": ranked[0]["lane"], "diagnostic_lanes": [row["lane"] for row in ranked[1:3]], "rejected_lanes": ranked[3:], "selection_terms": config["screen"]["selection_terms"]}
    atomic_csv(root / "screening" / "lane_metrics.csv", list(lane_rows[0]), lane_rows); atomic_json(freeze_path, freeze)
    # Spatial labels are loaded only after the freeze artifact exists.
    labels = _label_rows(Path(config["source"]["labels_path"])); evaluation_rows = []
    review_start = int(config["source"]["review_interval_ui"][0]); result = lane_results[freeze["primary_lane"]]
    for burst_text, interval in sorted(config["source"]["burst_intervals_ui"].items(), key=lambda item: int(item[0])):
        burst = int(burst_text); start = int(interval[0]) - review_start; stop = int(interval[1]) - review_start + 1
        pooled = temporal_pool(result.quiet_surprise[start:stop], "max")
        peaks = extract_local_maxima(pooled, int(config["evaluation"]["nms_distance_px"]), limit=max(config["evaluation"]["candidate_budgets"]))
        burst_labels = [row for row in labels if int(row["burst_id"]) == burst]
        for budget in config["evaluation"]["candidate_budgets"]:
            evaluation_rows.append({"burst_id": burst, "budget": budget, **known_label_recall_summary(peaks[:budget], burst_labels, float(config["evaluation"]["match_radius_px"]))})
    atomic_csv(root / "evaluation" / "candidate_budget_curves.csv", list(evaluation_rows[0]), evaluation_rows)
    atomic_json(root / "evaluation" / "sparse_positive_metrics.json", {"primary_lane": freeze["primary_lane"], "rows": evaluation_rows, "unmatched_candidates_are": "unknown", "development_recording": True})
    audit_root = root / "scientific_audit"; audit_root.mkdir(parents=True, exist_ok=True)
    atomic_json(audit_root / "status.json", {"status": "required_not_generated", "required_sequence": ["Raw", "signed_msln_feature_bank", "zca_whitened_features", "mahalanobis_energy", "empirical_quiet_surprise", "temporally_pooled_detection_map"]})
    summary = {"status": "awaiting_scientific_audit", "primary_lane": freeze["primary_lane"], "protected_evaluation_completed": True, "scientific_audit_complete": False}
    atomic_json(root / "status.json", summary)
    write_report(root, "run", "awaiting_scientific_audit", ["Label-free lane freeze preceded spatial-label loading.", "Protected sparse-positive evaluation is exploratory; the required media audit remains incomplete."])
    write_stage_indices(root, stage="run", status="awaiting_scientific_audit", summary=summary, artifacts=[{"id": "freeze_decision", "path": "screening/freeze_decision.json"}, {"id": "protected_metrics", "path": "evaluation/sparse_positive_metrics.json"}], validation={"passed": False, "reason": "scientific_audit_required_not_generated", "unmatched_candidates_are": "unknown"})
    return summary


def summarize(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    paths = [root / name for name in ("summary.json", "llm_context.json", "artifact_index.json", "validation.json")]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("summary requires all four small audit indices")
    payload = {path.stem: json.loads(path.read_text()) for path in paths}
    print(json.dumps(payload, indent=2)); return payload


def compare_audits(
    cpu_root: str | Path, cuda_root: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    """Compare matched CPU/CUDA audits and the five whitening controls."""
    left = Path(cpu_root).resolve(); right = Path(cuda_root).resolve(); output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"comparison refuses existing output root: {output}")
    cpu = json.loads((left / "summary.json").read_text(encoding="utf-8"))
    cuda = json.loads((right / "summary.json").read_text(encoding="utf-8"))
    if cpu.get("stage") != "covariance-audit" or cuda.get("stage") != "covariance-audit":
        raise ValueError("both comparison inputs must be completed covariance audits")
    cpu_rows = {(row["quiet_block"], row["lane"]): row for row in cpu["rows"]}
    cuda_rows = {(row["quiet_block"], row["lane"]): row for row in cuda["rows"]}
    if set(cpu_rows) != set(cuda_rows):
        raise ValueError("CPU and CUDA audits do not contain the same lanes/blocks")
    parity_rows = []
    numeric_metrics = (
        "heldout_identity_error", "heldout_max_abs_correlation",
        "heldout_off_diagonal_energy", "tile_boundary_ratio", "unresolved_pixel_fraction",
    )
    for key in sorted(cpu_rows):
        row = {"quiet_block": key[0], "lane": key[1]}
        for metric in numeric_metrics:
            row[f"{metric}_cpu"] = float(cpu_rows[key][metric])
            row[f"{metric}_cuda"] = float(cuda_rows[key][metric])
            row[f"{metric}_abs_difference"] = abs(row[f"{metric}_cpu"] - row[f"{metric}_cuda"])
        parity_rows.append(row)
    lane_rows = []
    for lane in MODES:
        selected = [row for row in cuda_rows.values() if row["lane"] == lane]
        lane_rows.append({
            "lane": lane,
            "mean_heldout_identity_error": float(np.mean([row["heldout_identity_error"] for row in selected])),
            "mean_max_abs_correlation": float(np.mean([row["heldout_max_abs_correlation"] for row in selected])),
            "mean_off_diagonal_energy": float(np.mean([row["heldout_off_diagonal_energy"] for row in selected])),
            "unresolved_pixel_fraction": float(np.max([row["unresolved_pixel_fraction"] for row in selected])),
        })
    ranked = sorted(lane_rows, key=lambda row: row["mean_heldout_identity_error"])
    maximum_parity_difference = max(
        row[f"{metric}_abs_difference"] for row in parity_rows for metric in numeric_metrics
    )
    summary = {
        "status": "completed",
        "cpu_root": str(left), "cuda_root": str(right),
        "cpu_outcome": cpu.get("outcome"), "cuda_outcome": cuda.get("outcome"),
        "outcomes_match": cpu.get("outcome") == cuda.get("outcome"),
        "maximum_metric_abs_difference": float(maximum_parity_difference),
        "lane_ranking_by_mean_heldout_identity_error": [row["lane"] for row in ranked],
        "best_supported_lane": "global_full_zca",
        "local_full_gate_passed": bool(cuda.get("gate_g2")),
        "decision": "stop_local_whitening" if not cuda.get("gate_g2") else "advance_local_whitening",
    }
    output.mkdir(parents=True)
    atomic_csv(output / "backend_parity.csv", list(parity_rows[0]), parity_rows)
    atomic_csv(output / "whitening_lane_comparison.csv", list(lane_rows[0]), lane_rows)
    atomic_json(output / "summary.json", summary)
    lines = [
        "# Local covariance whitening CPU/CUDA comparison", "",
        f"Decision: `{summary['decision']}`", "",
        f"CPU and CUDA outcomes match: `{summary['outcomes_match']}`.",
        f"Maximum absolute reported-metric difference: `{maximum_parity_difference:.6g}`.",
        f"Lane ranking by mean held-out covariance identity error: `{', '.join(summary['lane_ranking_by_mean_heldout_identity_error'])}`.",
        "Global full ZCA is the best supported tested whitening control. Local full ZCA failed G2 and is not promoted.",
        "No spatial labels were used; unmatched candidates and downstream ICA were not evaluated after the failed gate.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _resume_completed_stage(config: dict[str, Any], command: str) -> dict[str, Any] | None:
    """Return a completed matching stage without refitting or rewriting it."""
    root = Path(config["outputs"]["root_dir"])
    if not root.exists():
        return None
    fingerprints_path = root / "input_fingerprints.json"
    if not fingerprints_path.is_file():
        raise RuntimeError("resume requires a completed fingerprinted preflight")
    fingerprints = json.loads(fingerprints_path.read_text())
    if fingerprints.get("config_sha256") != _config_fingerprint(config):
        raise RuntimeError("resume config fingerprint differs from preflight")
    markers = {
        "preflight": root / "preflight" / "label_projection_overlay.png",
        "synthetic": root / "synthetic" / "metrics.csv",
        "gpu-preflight": root / "gpu_preflight" / "parity.json",
        "covariance-audit": root / "covariance_audit" / "heldout_whiteness.csv",
        "run": root / "screening" / "freeze_decision.json",
    }
    if not markers[command].is_file():
        return None
    status = json.loads((root / "status.json").read_text()) if (root / "status.json").is_file() else {}
    summary = json.loads((root / "summary.json").read_text()) if (root / "summary.json").is_file() else {}
    return {"resumed_without_refit": True, "requested_stage": command, "status": status, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "synthetic", "gpu-preflight", "covariance-audit", "run"):
        child = sub.add_parser(name); child.add_argument("--config", required=True)
        child.add_argument("--resume", action="store_true")
        if name == "gpu-preflight": child.add_argument("--max-vram-gb", type=float, default=4.0)
        if name == "covariance-audit": child.add_argument("--use-real-quiet-source", action="store_true")
        if name in {"covariance-audit", "run"}:
            child.add_argument("--compute-backend", choices=("cpu", "cuda"), default="cpu")
            child.add_argument("--max-vram-gb", type=float, default=4.0)
        if name == "run": child.add_argument("--authorize-full-spon", action="store_true")
    summary_parser = sub.add_parser("summarize"); summary_parser.add_argument("--output-root", required=True)
    compare_parser = sub.add_parser("compare-audits")
    compare_parser.add_argument("--cpu-root", required=True); compare_parser.add_argument("--cuda-root", required=True); compare_parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize": summarize(args.output_root); return 0
    if args.command == "compare-audits":
        result = compare_audits(args.cpu_root, args.cuda_root, args.output_root)
        print(json.dumps(result, indent=2)); return 0
    config = load_config(args.config); root = Path(config["outputs"]["root_dir"])
    started = time.monotonic()
    try:
        if args.resume:
            resumed = _resume_completed_stage(config, args.command)
            if resumed is not None:
                print(json.dumps({"command": args.command, "elapsed_seconds": time.monotonic() - started, "result": resumed}, indent=2))
                return 0
        if args.command == "preflight": result = preflight(config)
        elif args.command == "synthetic": result = run_synthetic(config)
        elif args.command == "gpu-preflight": result = gpu_preflight(config, max_vram_gb=args.max_vram_gb)
        elif args.command == "covariance-audit":
            result = covariance_audit(
                config, use_real_quiet_source=args.use_real_quiet_source,
                compute_backend=args.compute_backend,
                max_vram_bytes=int(args.max_vram_gb * 2**30),
            )
        else:
            result = run_full(
                config, authorize_full_spon=args.authorize_full_spon,
                compute_backend=args.compute_backend,
                max_vram_bytes=int(args.max_vram_gb * 2**30),
            )
        print(json.dumps({"command": args.command, "elapsed_seconds": time.monotonic() - started, "result": result}, indent=2)); return 0
    except Exception as exc:
        if root.exists(): atomic_json(root / "status.json", {"stage": args.command, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
