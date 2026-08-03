"""Guarded generated-only program for dependent multiscale W0--W3."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from neurobench.algorithms.dependent_multiscale import (
    ScaleViewSpec,
    build_scale_views,
    decompose_patch_baseline,
    default_dependency_graph,
    fit_local_pca,
    reconstruct_local_factorization,
)
from neurobench.metrics.multiscale_decomposition import attribution_metrics, closure_metrics
from neurobench.reports.dependent_multiscale import render_generated_report

from .dependent_multiscale_config import DependentMultiscaleConfig
from .dependent_multiscale_noise import fit_joint_quiet_noise_model, joint_cs_divergence
from .dependent_multiscale_synthetic import FIXTURE_IDS, make_fixture


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _specs(normalization_kind: str = "quiet_robust") -> tuple[ScaleViewSpec, ...]:
    return tuple(
        ScaleViewSpec(
            view_id=f"scale_{support}", support_px=support,
            operator_kind="quiet_normalized_local_support",
            normalization_kind=normalization_kind,
            parameters={"nested": True, "padding": "reflect"},
        )
        for support in (5, 7, 15)
    )


def generated_smoke(*, fixture_ids: tuple[str, ...] = FIXTURE_IDS, seed: int = 7) -> dict[str, Any]:
    """Run the small exact-truth W3 structural baseline without filesystem I/O."""
    rows = []
    maximum_closure = 0.0
    for fixture_id in fixture_ids:
        fixture = make_fixture(fixture_id, seed=seed)
        views = build_scale_views(fixture.observation, _specs(), quiet_count=8)
        factor = fit_local_pca(
            views["scale_7"], patch_id=fixture_id, view_id="scale_7",
            origin_yx=(0, 0), rank=4,
        )
        restored = reconstruct_local_factorization(factor)
        decomposition = decompose_patch_baseline(
            fixture.observation, views, patch_id=fixture_id
        )
        estimate = {
            "background": decomposition.background,
            "structured_signal": decomposition.structured_signal,
            "structured_artifact": decomposition.structured_artifact,
            "noise_candidate": decomposition.noise_candidate,
        }
        truth = {
            "background": fixture.background,
            "structured_signal": fixture.structured_signal,
            "structured_artifact": fixture.structured_artifact,
            "noise_candidate": fixture.noise,
        }
        closure = closure_metrics(fixture.observation, estimate)
        maximum_closure = max(maximum_closure, closure["normalized_maximum"])
        quiet_views = {key: value[:8] for key, value in views.items()}
        noise_model = fit_joint_quiet_noise_model(quiet_views)
        aligned = np.column_stack([views[key][8:].reshape(-1) for key in noise_model.view_ids])
        rows.append({
            "fixture_id": fixture_id,
            "seed": seed,
            "closure": closure,
            "attribution": attribution_metrics(truth, estimate),
            "local_pca_reconstruction_nmse": factor.reconstruction_nmse,
            "local_pca_roundtrip_nmse": float(
                np.sum((views["scale_7"] - restored) ** 2)
                / max(float(np.sum(views["scale_7"] ** 2)), np.finfo(float).eps)
            ),
            "joint_quiet_cs_divergence": joint_cs_divergence(aligned, noise_model),
            "residual_name": "noise_candidate",
        })
    return {
        "schema_version": 1,
        "status": "completed_generated_only",
        "fixture_count": len(rows),
        "fixtures": rows,
        "maximum_normalized_closure": maximum_closure,
        "dependency_graph": _jsonable(default_dependency_graph()),
        "gates": {
            "C1_numerical_reconstruction": "pass" if maximum_closure <= 1e-4 else "fail",
            "C2_generated_attribution": "not_run_requires_W4_ITL",
            "C4_residual_qualification": "not_qualified",
        },
        "scientific_carrier_status": "retained_external_authority",
        "labels_used": False,
    }


def synthetic(output_dir: str | Path) -> dict[str, Any]:
    """Write collision-safe generated W3 artifacts and a baseline report."""
    target = Path(output_dir).resolve()
    partial = Path(str(target) + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError("generated output or partial output already exists")
    partial.mkdir(parents=True)
    progress = partial / "progress.jsonl"
    started = time.time()
    try:
        progress.write_text(json.dumps({"event": "started", "time": started}) + "\n", encoding="utf-8")
        metrics = generated_smoke()
        _atomic_json(partial / "metrics.json", metrics)
        _atomic_json(partial / "dependency_graph.json", metrics["dependency_graph"])
        _atomic_json(partial / "run_state.json", {
            "status": "completed_generated_only",
            "elapsed_seconds": time.time() - started,
            "fixture_count": metrics["fixture_count"],
        })
        (partial / "REPORT.md").write_text(render_generated_report(metrics), encoding="utf-8")
        (partial / "RESULTS_INDEX.md").write_text(
            "# Results index\n\nStart with `REPORT.md`, then `metrics.json` and `dependency_graph.json`.\n",
            encoding="utf-8",
        )
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "completed", "time": time.time()}) + "\n")
        partial.replace(target)
        return metrics
    except Exception:
        # Preserve the partial directory and heartbeat for diagnosis/resume.
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "failed", "time": time.time()}) + "\n")
        raise


def preflight(config: DependentMultiscaleConfig) -> dict[str, Any]:
    """Write a read-only-source audit to a new collision-safe preflight root."""
    if config.preflight_dir.exists():
        raise FileExistsError("preflight directory already exists")
    observation_path = Path(config.input["observation_npy"])
    display_path = Path(config.input["display_observation_npy"])
    carrier_path = Path(config.input["scientific_carrier_npy"])
    labels_path = Path(config.input["labels_tsv"])
    provider = config.input["provider_local_pca_metadata"]
    required = [observation_path, display_path, carrier_path, labels_path]
    if provider is not None:
        required.append(Path(provider))
    missing = [str(path) for path in required if not path.is_file()]
    shape = None
    review = None
    finite_probe = False
    frame_alignment = False
    if observation_path.is_file() and carrier_path.is_file():
        observation = np.load(observation_path, mmap_mode="r", allow_pickle=False)
        carrier = np.load(carrier_path, mmap_mode="r", allow_pickle=False)
        start_ui = int(config.frames["review_start_ui"])
        stop_ui = int(config.frames["review_end_ui"])
        expected = stop_ui - start_ui + 1
        if observation.ndim == 3 and len(observation) >= stop_ui:
            review = observation[start_ui - 1:stop_ui]
        elif observation.ndim == 3 and len(observation) == expected:
            review = observation
        shape = None if review is None else list(review.shape)
        frame_alignment = bool(
            review is not None and carrier.ndim == 3
            and review.shape == carrier.shape and len(review) == expected
        )
        finite_probe = bool(
            review is not None
            and np.isfinite(review[::max(1, len(review)//8), ::16, ::16]).all()
        )
    estimated_output = float(5 * np.prod(shape or [1]) * 4 / 2**20)
    estimated_ram = float(24 * np.prod(shape or [1]) * 4 / 2**20)
    probe = config.output_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**20
    gates = {
        "inputs_exist": not missing,
        "frame_and_coordinate_alignment": frame_alignment,
        "finite_observation_probe": finite_probe,
        "labels_evaluation_only": config.input["labels_role"] == "evaluation_only",
        "normalization_declared": bool(config.input["input_normalization_state"]),
        "output_absent": not config.output_dir.exists(),
        "partial_output_absent": not Path(str(config.output_dir) + ".partial").exists(),
        "ram_cap_sufficient": estimated_ram <= int(config.resources["max_ram_mib"]),
        "output_cap_sufficient": estimated_output <= int(config.resources["max_output_mib"]),
        "disk_headroom_sufficient": free_disk >= estimated_output + 256,
        "cpu_only": config.resources["device"] == "cpu",
    }
    payload = {
        "schema_version": 1,
        "kind": "dependent_multiscale_read_only_source_preflight",
        "experiment_id": config.experiment_id,
        "ready": all(gates.values()),
        "gates": gates,
        "missing": missing,
        "observation_shape": shape,
        "resources": {
            "estimated_peak_ram_mib": estimated_ram,
            "estimated_output_mib": estimated_output,
            "free_disk_mib": free_disk,
            **config.resources,
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in required if path.is_file()
        ],
        "authorized_execution": "preflight_only; no real-data run authorized",
    }
    config.preflight_dir.mkdir(parents=True)
    _atomic_json(config.preflight_dir / "preflight.json", payload)
    _atomic_json(config.preflight_dir / "config.resolved.json", config.to_dict())
    _atomic_json(config.preflight_dir / "dependency_graph.json", default_dependency_graph())
    if payload["ready"] and bool(config.evaluation["write_label_projection"]):
        from .dependent_multiscale_artifacts import write_label_projection
        label_count = write_label_projection(
            review[:int(config.frames["quiet_count"])], labels_path,
            config.preflight_dir / "label_projection_overlay.png",
        )
        payload["label_projection_count"] = label_count
        _atomic_json(config.preflight_dir / "preflight.json", payload)
    if not payload["ready"]:
        raise RuntimeError(f"dependent multiscale preflight failed: {payload}")
    return payload


def run(config: DependentMultiscaleConfig) -> dict[str, Any]:
    """Run the reviewed diagnostic-only real-data failure-analysis lane."""
    preflight_path = config.preflight_dir / "preflight.json"
    resolved_path = config.preflight_dir / "config.resolved.json"
    if not preflight_path.is_file() or not resolved_path.is_file():
        raise RuntimeError("run requires an exact matching ready preflight")
    audit = json.loads(preflight_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not audit.get("ready") or resolved != config.to_dict():
        raise RuntimeError("run requires an exact matching ready preflight")
    from .dependent_multiscale_real import run_real_diagnostic
    return run_real_diagnostic(config, preflight=audit)


def report(run_dir: str | Path) -> dict[str, Any]:
    """Read completed metrics without modifying the run directory."""
    root = Path(run_dir).resolve()
    metrics_path = root / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("run directory has no metrics.json")
    return json.loads(metrics_path.read_text(encoding="utf-8"))
