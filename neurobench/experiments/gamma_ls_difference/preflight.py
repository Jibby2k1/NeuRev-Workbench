"""Read-only, collision-safe preflight for the GPU Gamma-LS experiment."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Iterable

import numpy as np

from neurobench.portable_paths import portable_path

from .config import GammaLSDifferenceConfig


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _run_probe(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout.strip()[:4000],
            "stderr": completed.stderr.strip()[:4000],
        }
    except Exception as error:  # pragma: no cover - host dependent
        return {"command": command, "probe_error": repr(error)}


def _git_state(repository: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    try:
        status_lines = [line for line in git("status", "--short").splitlines() if line]
        return {
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty_path_count": len(status_lines),
            "working_tree_clean": not status_lines,
            "release_boundary": (
                "implementation is local working-tree evidence and is not contained "
                "in HEAD while the listed files remain uncommitted"
            ),
        }
    except Exception as error:  # pragma: no cover - host dependent
        return {"inspection_error": repr(error), "working_tree_clean": False}


def _implementation_status(config: GammaLSDifferenceConfig) -> dict[str, Any]:
    relative_paths = [
        "examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json",
        "docs/research/SPON_CA_BURST_GAMMA_LS_DIFFERENCE_ABLATION_V1_PLAN.md",
        "docs/research/SPON_CA_BURST_GAMMA_LS_PAPER_V2_OUTLINE.md",
        "docs/workflows/spon_ca_burst_gamma_ls_difference_ablation.md",
        "neurobench/algorithms/gamma_local_standardization.py",
        "neurobench/algorithms/multilag_msica.py",
        "neurobench/metrics/sparse_detection.py",
        "neurobench/experiments/pairwise_separation/runner.py",
        "neurobench/experiments/pairwise_separation/sampling.py",
        "neurobench/experiments/pairwise_separation/fitting.py",
        "neurobench/experiments/msln_msica/multilag_program.py",
        "neurobench/experiments/gamma_ls_difference/__init__.py",
        "neurobench/experiments/gamma_ls_difference/config.py",
        "neurobench/experiments/gamma_ls_difference/cuda_runtime.py",
        "neurobench/experiments/gamma_ls_difference/preflight.py",
        "neurobench/experiments/gamma_ls_difference/__main__.py",
        "neurobench/experiments/gamma_ls_difference/representations.py",
        "neurobench/experiments/gamma_ls_difference/gpu_representations.py",
        "neurobench/experiments/gamma_ls_difference/evaluation.py",
        "neurobench/experiments/gamma_ls_difference/grid.py",
        "neurobench/experiments/gamma_ls_difference/screen.py",
        "neurobench/experiments/gamma_ls_difference/smoke.py",
        "neurobench/experiments/gamma_ls_difference/streaming_benchmark.py",
        "neurobench/experiments/gamma_ls_difference/protected.py",
        "neurobench/experiments/gamma_ls_difference/full_recording.py",
        "neurobench/experiments/gamma_ls_difference/scientific_audit.py",
        "neurobench/experiments/gamma_ls_difference/fixed_deployment_characterization.py",
        "tests/test_gamma_local_standardization.py",
        "tests/test_gamma_ls_difference_config.py",
        "tests/test_gamma_ls_difference_representations.py",
        "tests/test_gamma_ls_difference_gpu_representations.py",
        "tests/test_gamma_ls_difference_evaluation.py",
        "tests/test_gamma_ls_difference_grid.py",
        "tests/test_gamma_ls_difference_screen.py",
        "tests/test_gamma_ls_difference_smoke.py",
        "tests/test_gamma_ls_difference_streaming_benchmark.py",
        "tests/test_gamma_ls_difference_protected.py",
        "tests/test_gamma_ls_difference_full_recording.py",
        "tests/test_gamma_ls_fixed_deployment_characterization.py",
    ]
    files: dict[str, Any] = {}
    for relative in relative_paths:
        path = config.repository / relative
        files[f"repo://{relative}"] = (
            {"present": True, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            if path.is_file()
            else {"present": False}
        )
    return {
        "complete": all(row["present"] for row in files.values()),
        "files": files,
        "git": _git_state(config.repository),
    }


def _gpu_status() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requested_device": "cuda",
        "kernel_release": platform.release(),
        "device_nodes": sorted(str(path) for path in Path("/dev").glob("nvidia*")),
        "nvidia_smi": _run_probe(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        ),
    }
    try:
        import torch

        available = bool(torch.cuda.is_available())
        payload.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_build": str(torch.version.cuda),
                "torch_cuda_available": available,
                "device_count": int(torch.cuda.device_count()) if available else 0,
            }
        )
        if available:
            free, total = torch.cuda.mem_get_info(0)
            payload.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "free_vram_gib": free / 2**30,
                    "total_vram_gib": total / 2**30,
                }
            )
    except Exception as error:  # pragma: no cover - host dependent
        payload.update({"torch_cuda_available": False, "torch_error": repr(error)})
    payload["ready"] = bool(
        payload.get("torch_cuda_available")
        and payload["device_nodes"]
        and payload["nvidia_smi"].get("returncode") == 0
    )
    return payload


def verify_matching_preflight(
    config: GammaLSDifferenceConfig,
    artifact_dir: str | Path,
    *,
    require_gpu_ready: bool = True,
) -> dict[str, Any]:
    """Require an immutable preflight for the exact current config and code."""
    root = Path(artifact_dir).expanduser().resolve()
    payload = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    portable = json.loads((root / "config.portable.json").read_text(encoding="utf-8"))
    if portable != config.portable_dict():
        raise RuntimeError("preflight config does not match the requested config")
    if not payload.get("data_ready"):
        raise RuntimeError(f"preflight data gate is not ready: {payload.get('status')}")
    if require_gpu_ready and not payload.get("gpu_run_ready"):
        raise RuntimeError(f"preflight GPU gate is not ready: {payload.get('status')}")

    current_implementation = _implementation_status(config)
    frozen_files = payload.get("implementation", {}).get("files")
    if not current_implementation["complete"] or current_implementation["files"] != frozen_files:
        raise RuntimeError("implementation fingerprints changed after preflight")
    for key, path in config.source_paths.items():
        frozen = payload["source"][key]["sha256"]
        if _sha256(path) != frozen:
            raise RuntimeError(f"source fingerprint changed after preflight: {key}")
    return payload


def _available_ram_gib() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available / 2**30)
    except Exception:  # pragma: no cover - host dependent
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size / 2**30)
        except (ValueError, OSError, KeyError):
            return 0.0


def _active_experiment_processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        import psutil

        excluded = {os.getpid(), os.getppid()}
        for process in psutil.process_iter(("pid", "name", "cmdline")):
            if process.info["pid"] in excluded:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            name = (process.info.get("name") or "").lower()
            if name.startswith("python") and "gamma_ls_difference" in command.lower():
                rows.append(
                    {
                        "pid": int(process.info["pid"]),
                        "name": process.info.get("name") or "",
                        "command": command[:400],
                    }
                )
    except Exception as error:  # pragma: no cover - host dependent
        rows.append({"inspection_error": repr(error)})
    return rows


def _validate_coordinates(
    rows: Iterable[dict[str, str]], shape_yx: tuple[int, int], scope: str
) -> None:
    height, width = shape_yx
    for row in rows:
        x = float(row["x_px"])
        y = float(row["y_px"])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{scope} coordinate outside movie: {(x, y)}")


def _label_summary(
    path: Path,
    *,
    movie_shape: tuple[int, int, int],
    selector: str,
) -> dict[str, Any]:
    rows = _read_tsv(path)
    required = {
        "observation_id",
        "burst_id",
        "canonical_roi_id",
        "x_px",
        "y_px",
        selector,
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path.name} lacks required sparse-positive columns")
    selected = [row for row in rows if row[selector].strip().lower() == "true"]
    if len({row["observation_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate observation_id in {path.name}")
    _validate_coordinates(selected, movie_shape[1:], path.name)
    bursts = {
        burst: sum(row["burst_id"] == burst for row in selected)
        for burst in sorted({row["burst_id"] for row in selected})
    }
    return {
        "path": path,
        "rows_total": len(rows),
        "selection_field": selector,
        "rows_selected": len(selected),
        "canonical_identity_count": len(
            {row["canonical_roi_id"] for row in selected}
        ),
        "selected_by_burst": bursts,
    }


def _find_multilag_fit(surface: dict[str, Any], config_id: str) -> dict[str, Any]:
    matches = []
    for collection_name in ("calibration_rows", "expansion_rows"):
        for row in surface.get(collection_name, []):
            if row.get("config_id") == config_id:
                matches.append((collection_name, row))
    if len(matches) != 1:
        raise ValueError(f"expected one multilag config row, found {len(matches)}")
    collection, row = matches[0]
    fit = row.get("fit", {})
    whitening = np.asarray(fit.get("whitening"), dtype=np.float64)
    demixing = np.asarray(fit.get("demixing"), dtype=np.float64)
    if (
        row.get("formulation") != "delay_embedding"
        or row.get("objective_family") != "cs_parzen"
        or fit.get("lags") != [0, 1, 2, 4, 8, 16]
        or fit.get("residual_indices") != [2, 3, 4, 5]
        or fit.get("diagnostics", {}).get("explained_fraction") != 1.0
        or whitening.shape != (6, 6)
        or demixing.shape != (6, 6)
        or np.linalg.matrix_rank(whitening) != 6
        or not np.isfinite(whitening).all()
        or not np.isfinite(demixing).all()
    ):
        raise ValueError("multilag source does not match the frozen six-lag fit")
    return {
        "collection": collection,
        "config_id": config_id,
        "formulation": row["formulation"],
        "objective_family": row["objective_family"],
        "parameter": row.get("parameter"),
        "lags": fit["lags"],
        "residual_indices": fit["residual_indices"],
        "full_rank_whitening_explained_fraction": fit["diagnostics"][
            "explained_fraction"
        ],
        "whitening_rank": int(np.linalg.matrix_rank(whitening)),
        "converged": bool(fit.get("converged")),
    }


def _gamma_search_rows(config: GammaLSDifferenceConfig) -> list[dict[str, Any]]:
    grid = config.payload["gamma_ls_grid"]
    rows: list[dict[str, Any]] = []
    for half_width in grid["g1_half_width_px"]:
        for guard in grid["g1_guard_radius_px"]:
            rows.append(
                {
                    "stage": "G1",
                    "context_id": (
                        f"gamma_h{int(half_width)}_g{int(guard)}_n5_m0p75"
                    ),
                    "support_width_px": 2 * int(half_width) + 1,
                    "guard_radius_px": int(guard),
                    "shape_n": float(grid["g1_shape"]),
                    "mode_radius_px": float(half_width)
                    * float(grid["g1_mode_fraction_of_half_width"]),
                    "support": "disk",
                    "eligible_primary": True,
                }
            )
    rows.extend(
        [
            {
                "stage": "diagnostic",
                "context_id": "legacy_exact_n9_mode35_w23",
                "support_width_px": 23,
                "guard_radius_px": 0,
                "shape_n": 9.0,
                "mode_radius_px": 35.0,
                "support": "square",
                "eligible_primary": False,
            },
            {
                "stage": "diagnostic",
                "context_id": "square_box_outer11_guard3",
                "support_width_px": 23,
                "guard_radius_px": 3,
                "shape_n": "not_applicable",
                "mode_radius_px": "not_applicable",
                "support": "uniform_square_outer_minus_square_guard",
                "eligible_primary": False,
            },
        ]
    )
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _selected_coordinates(path: Path, selector: str) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_tsv(path)
    selected = [row for row in rows if row[selector].strip().lower() == "true"]
    x = np.asarray([float(row["x_px"]) for row in selected], dtype=np.float64)
    y = np.asarray([float(row["y_px"]) for row in selected], dtype=np.float64)
    return x, y


def _write_label_projection_overlay(
    path: Path,
    *,
    movie: np.ndarray,
    review_interval_ui: tuple[int, int],
    protected_path: Path,
    latest_path: Path,
) -> dict[str, Any]:
    """Write the required coordinate-orientation check on current source data."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start_ui, stop_ui = review_interval_ui
    projection = np.zeros(movie.shape[1:], dtype=movie.dtype)
    start_zero = start_ui - 1
    stop_zero = stop_ui
    for start in range(start_zero, stop_zero, 64):
        stop = min(start + 64, stop_zero)
        chunk_max = np.max(movie[start:stop], axis=0)
        np.maximum(projection, chunk_max, out=projection)
    finite = projection[np.isfinite(projection)]
    if finite.size == 0:
        raise ValueError("label projection source contains no finite pixels")
    low, high = np.percentile(finite.astype(np.float64), [1.0, 99.8])
    if not high > low:
        high = low + 1.0

    panels = (
        (protected_path, "include_inclusive", "Protected v1: 79 occurrences"),
        (latest_path, "include_confirmed", "Latest v7: 106 confirmed occurrences"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)
    for axis, (label_path, selector, title) in zip(axes, panels):
        x, y = _selected_coordinates(label_path, selector)
        axis.imshow(projection, cmap="gray", vmin=low, vmax=high, origin="upper")
        axis.scatter(
            x,
            y,
            s=19,
            facecolors="none",
            edgecolors="#00A65A",
            linewidths=0.8,
            label="expert sparse-positive coordinate",
        )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("x = column (px)")
        axis.set_ylabel("y = row (px)")
        axis.set_xlim(-0.5, movie.shape[2] - 0.5)
        axis.set_ylim(movie.shape[1] - 0.5, -0.5)
        axis.legend(loc="lower right", fontsize=7, framealpha=0.85)
    fig.suptitle(
        f"Label projection check on max image, UI frames {start_ui}-{stop_ui}",
        fontsize=11,
    )
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    fig.savefig(temporary, dpi=180, facecolor="white", format="png")
    plt.close(fig)
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "source_image": "max_projection_of_review_interval",
        "review_interval_ui": [start_ui, stop_ui],
        "coordinate_convention": "x_column_y_row",
        "expert_color": "green",
        "model_predictions_present": False,
    }


def _artifact_index(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "artifact_index.json":
            rows.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"schema_version": 1, "artifacts": rows}


def run_preflight(
    config: GammaLSDifferenceConfig, *, artifact_dir: str | Path
) -> dict[str, Any]:
    """Audit sources and runtime, writing evidence even when CUDA is blocked."""
    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"preflight artifact directory exists: {destination}")
    if config.output_root.exists():
        raise FileExistsError(f"experiment output root exists: {config.output_root}")
    missing = [str(path) for path in config.source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required sources are missing: {missing}")

    movie_path = config.source_paths["movie"]
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if tuple(movie.shape) != (2359, 340, 573) or movie.dtype != np.uint16:
        raise ValueError(f"unexpected movie contract: shape={movie.shape} dtype={movie.dtype}")

    protected = _label_summary(
        config.source_paths["protected_labels_v1"],
        movie_shape=tuple(movie.shape),
        selector="include_inclusive",
    )
    if (
        protected["rows_total"] != 79
        or protected["rows_selected"] != 79
        or protected["canonical_identity_count"] != 26
        or protected["selected_by_burst"] != {"1": 15, "2": 20, "3": 21, "4": 23}
    ):
        raise ValueError(f"protected v1 authority changed: {protected}")

    latest = _label_summary(
        config.source_paths["latest_labels_v7"],
        movie_shape=tuple(movie.shape),
        selector="include_confirmed",
    )
    if (
        latest["rows_total"] != 137
        or latest["rows_selected"] != 106
        or latest["canonical_identity_count"] != 44
        or latest["selected_by_burst"] != {"1": 24, "2": 23, "3": 26, "4": 33}
    ):
        raise ValueError(f"latest v7 authority changed: {latest}")

    two_frame_fit = json.loads(
        config.source_paths["two_frame_cs_parzen_fit"].read_text(encoding="utf-8")
    )
    two_frame_mean = np.asarray(two_frame_fit.get("mean"), dtype=np.float64)
    two_frame_whitening = np.asarray(
        two_frame_fit.get("whitening"), dtype=np.float64
    )
    two_frame_demixing = np.asarray(two_frame_fit.get("demixing"), dtype=np.float64)
    if (
        two_frame_fit.get("method_id") != "cs_parzen_ica"
        or not two_frame_fit.get("converged")
        or two_frame_fit.get("axes") != "TYX"
        or int(two_frame_fit.get("undefined_leading_frames", -1)) != 1
        or int(two_frame_fit.get("sample_seed", -1)) != 20260727
        or two_frame_fit.get("source_video_sha256") != _sha256(movie_path)
        or two_frame_fit.get("activity_component") != 1
        or two_frame_fit.get("activity_sign") != -1
        or float(two_frame_fit.get("diagnostics", {}).get("bandwidth", -1)) != 0.35
        or two_frame_fit.get("component_selection", {}).get("status") != "resolved"
        or two_frame_mean.shape != (2,)
        or two_frame_whitening.shape != (2, 2)
        or two_frame_demixing.shape != (2, 2)
        or np.linalg.matrix_rank(two_frame_whitening) != 2
        or np.linalg.matrix_rank(two_frame_demixing) != 2
        or not np.isfinite(two_frame_mean).all()
        or not np.isfinite(two_frame_whitening).all()
        or not np.isfinite(two_frame_demixing).all()
    ):
        raise ValueError("two-frame CS-Parzen source fit changed")

    surface = json.loads(
        config.source_paths["multilag_v5_surface"].read_text(encoding="utf-8")
    )
    multilag = _find_multilag_fit(
        surface, str(config.payload["sources"]["multilag_config_id"])
    )
    if not multilag["converged"]:
        raise ValueError("frozen multilag source fit is not converged")

    resource = config.payload["resources"]
    disk_probe = config.output_root.parent
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    free_disk_gib = shutil.disk_usage(disk_probe).free / 2**30
    available_ram_gib = _available_ram_gib()
    review_frames = (
        int(config.payload["frames"]["review_interval_ui"][1])
        - int(config.payload["frames"]["review_interval_ui"][0])
        + 1
    )
    review_float_gib = (
        review_frames * int(movie.shape[1]) * int(movie.shape[2]) * 4 / 2**30
    )
    estimated_peak_ram_gib = max(4.0, 8.0 * review_float_gib)
    implementation = _implementation_status(config)
    resource_ready = bool(
        free_disk_gib >= float(resource["minimum_free_disk_gib"])
        and available_ram_gib >= estimated_peak_ram_gib
        and estimated_peak_ram_gib <= float(resource["max_peak_ram_gib"])
    )
    gpu = _gpu_status()
    required_free_vram_gib = float(resource["max_peak_vram_gib"])
    gpu["required_free_vram_gib"] = required_free_vram_gib
    gpu["resource_ready"] = bool(
        gpu["ready"]
        and float(gpu.get("free_vram_gib", 0.0)) >= required_free_vram_gib
    )
    active = _active_experiment_processes()
    no_collision_or_duplicate = not active
    data_ready = resource_ready and no_collision_or_duplicate and implementation["complete"]
    ready = bool(data_ready and gpu["resource_ready"])
    status = "ready" if ready else (
        "blocked_gpu_runtime" if data_ready and not gpu["ready"] else "blocked_preflight"
    )

    source_payload: dict[str, Any] = {}
    for key, path in config.source_paths.items():
        source_payload[key] = {
            "path": portable_path(
                path,
                repository=config.repository,
                data=config.authority,
            ),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    for summary in (protected, latest):
        summary["path"] = portable_path(
            summary["path"],
            repository=config.repository,
            data=config.authority,
        )

    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "data_ready": data_ready,
        "gpu_run_ready": ready,
        "source": {
            "movie": {
                **source_payload.pop("movie"),
                "shape": list(movie.shape),
                "dtype": str(movie.dtype),
                "axes": "TYX",
                "frame_interval_ms": 20.0,
            },
            **source_payload,
        },
        "label_authorities": {
            "protected_v1": protected,
            "latest_v7_sensitivity": latest,
            "interpretation": (
                "sparse positives only; unmatched candidates are unknown. "
                "Preflight audits label eligibility. Context screening uses the "
                "declared temporal burst windows but not sparse-positive coordinates "
                "or identities; fitting and candidate construction must remain "
                "separate from those sparse-positive fields."
            ),
        },
        "representations": {
            "ordered_arms": list(config.payload["representations"]),
            "two_frame_source_fit": {
                "method_id": two_frame_fit["method_id"],
                "bandwidth": two_frame_fit["diagnostics"]["bandwidth"],
                "activity_component": two_frame_fit["activity_component"],
                "activity_sign": two_frame_fit["activity_sign"],
                "component_selection_uses_derivative_reference": True,
                "fit_scope": "whole_review_interval_transductive",
                "preprocessing_history": (
                    "review_slice_UI_1800_2359_with_EMA_initialized_at_UI_1800"
                ),
                "role": "implementation_parity_and_diagnostic_only",
            },
            "multilag_source_fit": {
                **multilag,
                "fit_scope": "whole_review_interval_transductive",
                "role": "implementation_parity_and_diagnostic_only",
                "residual_group_uses_analytic_postfit_axis_rules": True,
            },
            "multiscale_claim": (
                "The six-lag delay embedding is multi-lag inference; Gamma-LS "
                "is a context grid followed by one fold-local context. Version 1 "
                "does not run multi-context fusion and makes no multiscale-LS claim."
            ),
        },
        "implementation": implementation,
        "execution_capabilities": {
            "implemented": [
                "cpu_reference_and_parity_tests",
                "cuda_representation_and_gamma_ls_smoke",
                "burst_window_supervised_fold_local_gamma_context_screen",
                "sparse_positive_evaluation_primitives",
                "sustained_streaming_benchmark",
                "outer_fold_ica_refit_and_selection_executor",
                "protected_end_to_end_executor",
                "full_recording_candidate_executor",
            ],
            "not_yet_implemented": [],
        },
        "gamma_ls": {
            "g1_context_count": 9,
            "g2_max_context_count_per_fold": 18,
            "diagnostic_control_count": 2,
            "primary_support": "radial_gamma_disk_with_explicit_guard",
            "square_support_primary_eligible": False,
            "outer_fold_selection": "independent_per_heldout_burst",
            "positive_tail_quantile": float(
                config.payload["screen"]["positive_tail_quantile"]
            ),
            "burst_windows_used_for_context_selection": True,
            "sparse_positive_coordinates_used_for_context_selection": False,
            "sparse_positive_identities_used_for_context_selection": False,
        },
        "resources": {
            "available_ram_gib": available_ram_gib,
            "estimated_peak_ram_gib": estimated_peak_ram_gib,
            "ram_cap_gib": float(resource["max_peak_ram_gib"]),
            "free_disk_gib": free_disk_gib,
            "required_free_disk_gib": float(resource["minimum_free_disk_gib"]),
            "gpu": gpu,
            "active_same_experiment_processes": active,
        },
        "collisions": {
            "artifact_dir": False,
            "output_root": False,
            "active_same_experiment_process": bool(active),
        },
        "claim_boundary": (
            "Preflight readiness authorizes only implemented stages. It is not "
            "experiment completion, scientific promotion, streaming readiness, "
            "or evidence that ICA improves detection. Archived ICA fits are "
            "transductive parity anchors, not held-out learned-model estimates."
        ),
    }

    destination.mkdir(parents=True, exist_ok=False)
    overlay = _write_label_projection_overlay(
        destination / "label_projection_overlay.png",
        movie=movie,
        review_interval_ui=tuple(
            int(value) for value in config.payload["frames"]["review_interval_ui"]
        ),
        protected_path=config.source_paths["protected_labels_v1"],
        latest_path=config.source_paths["latest_labels_v7"],
    )
    payload["label_projection_overlay"] = overlay
    _atomic_json(destination / "preflight.json", payload)
    _atomic_json(destination / "config.portable.json", config.portable_dict())
    _write_tsv(destination / "gamma_contexts_g1_and_controls.tsv", _gamma_search_rows(config))
    _atomic_json(
        destination / "status.json",
        {
            "status": status,
            "data_ready": data_ready,
            "gpu_run_ready": ready,
            "next_stage": "gpu_smoke" if ready else "repair_gpu_runtime_then_repeat_preflight",
        },
    )
    _atomic_json(
        destination / "llm_context.json",
        {
            "data_source": "source movie plus sparse-positive v1/v7 authorities",
            "algorithm": "representation -> signed radial Gamma-LS -> CFAR -> strict NMS",
            "evaluation_grain": "occurrence recall and unknown candidate burden",
            "screening_boundary": (
                "burst-window-aware; sparse-positive-coordinate- and identity-free"
            ),
            "model_media_inputs": [
                "quiet/event score distributions",
                "kernel stencil",
                "candidate overlays",
                "matched-side-by-side representation panels",
            ],
            "expert_only_inputs": ["raw movie", "full-resolution event clips"],
        },
    )
    _atomic_json(
        destination / "validation.json",
        {
            "status": "passed_preflight_artifact_contract",
            "checks": {
                "source_and_label_contracts_valid": True,
                "label_projection_overlay_present": True,
                "x_is_column_y_is_row": True,
                "unmatched_candidates_treated_as_unknown": True,
                "gpu_ready": ready,
            },
        },
    )
    _atomic_json(destination / "artifact_index.json", _artifact_index(destination))
    return payload


__all__ = ["run_preflight", "verify_matching_preflight"]
