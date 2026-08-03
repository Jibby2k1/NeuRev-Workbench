"""Bounded full-frame real-data diagnostics after a failed W5 gate.

This module intentionally does not claim the W7 scientific patchwise run. It
applies linear full-frame proxies to real data for failure analysis and writes
artifacts marked ``diagnostic_only_do_not_advance``.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import resource
import time
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import uniform_filter1d

from neurobench.algorithms.dependent_multiscale import PatchDecomposition, ScaleViewSpec, build_scale_views
from neurobench.metrics.multiscale_decomposition import closure_metrics

from .dependent_multiscale_config import DependentMultiscaleConfig
from .dependent_multiscale_evaluation import evaluate_generated_matrix
from .dependent_multiscale_figures import write_decomposition_video
from .dependent_multiscale_information import build_frame_nuisance, refine_group_dependence


def _atomic_json(path: Path, payload: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    probe = np.load(partial, mmap_mode="r", allow_pickle=False)
    if probe.shape != values.shape or probe.dtype != values.dtype:
        raise RuntimeError(f"atomic NPY verification failed for {path.name}")
    partial.replace(path)


def _review(values: np.ndarray, config: DependentMultiscaleConfig) -> np.ndarray:
    expected = int(config.frames["review_end_ui"]) - int(config.frames["review_start_ui"]) + 1
    if values.ndim != 3:
        raise ValueError("real input must have shape [T,Y,X]")
    if len(values) == expected:
        return values
    stop = int(config.frames["review_end_ui"])
    start = int(config.frames["review_start_ui"]) - 1
    if len(values) < stop:
        raise ValueError("real input does not cover the configured UI interval")
    return values[start:stop]


def _specs(config: DependentMultiscaleConfig) -> tuple[ScaleViewSpec, ...]:
    return tuple(
        ScaleViewSpec(
            view_id=f"scale_{support}", support_px=int(support),
            operator_kind=str(config.views["operator_kind"]),
            normalization_kind=str(config.views["normalization_kind"]),
            parameters={"nested": True, "padding": "reflect"},
        )
        for support in config.views["supports_px"]
    )


def _linear_proxy(observation: np.ndarray, views: Mapping[str, np.ndarray]) -> PatchDecomposition:
    """Return a scalable real-data failure-analysis proxy with exact closure."""
    broad = np.asarray(views["scale_15"], dtype=np.float32)
    background = uniform_filter1d(broad, size=31, axis=0, mode="nearest")
    signal = np.add(views["scale_5"], views["scale_7"], dtype=np.float32)
    signal *= 0.5
    signal -= background
    artifact = np.subtract(views["scale_5"], views["scale_7"], dtype=np.float32)
    artifact *= 0.5
    noise = np.asarray(observation, dtype=np.float32).copy()
    noise -= background
    noise -= signal
    noise -= artifact
    closure = np.zeros_like(noise)
    return PatchDecomposition(
        patch_id="full_frame_diagnostic_proxy",
        background=background,
        structured_signal=signal,
        structured_artifact=artifact,
        noise_candidate=noise,
        closure_residual=closure,
        posterior_uncertainty=None,
        diagnostics={
            "method_id": "full_frame_linear_proxy",
            "scientific_status": "diagnostic_only_do_not_advance",
            "patchwise_W7_claim": False,
            "noise_status": "noise_candidate",
        },
    )


def _channels(decomposition: PatchDecomposition) -> dict[str, np.ndarray]:
    return {
        "background": decomposition.background,
        "structured_signal": decomposition.structured_signal,
        "structured_artifact": decomposition.structured_artifact,
        "noise_candidate": decomposition.noise_candidate,
        "closure_residual": decomposition.closure_residual,
    }


def _write_report(path: Path, metrics: Mapping[str, Any]) -> None:
    summary = metrics["generated_gate"]["summary"]
    lines = [
        "# Dependent multiscale real-data diagnostic", "",
        "## Decision", "",
        "**do_not_advance**. This user-selected real-data application is a visual failure-analysis lane after generated C2/C3 failed. It is not the W7 scientific patchwise run and does not replace the accepted carrier.", "",
        "## Generated gate carried into this run", "",
        f"- Signal-leakage relative change: `{summary['relative_signal_leakage_improvement']:.6f}`.",
        f"- Median peak-amplitude ratio: `{summary['median_peak_amplitude_ratio']:.6f}`.",
        f"- Median temporal-area ratio: `{summary['median_temporal_area_ratio']:.6f}`.", "",
        "## Real-data diagnostic", "",
        f"- Maximum normalized closure: `{metrics['closure']['normalized_maximum']:.3e}`.",
        f"- Frames: `{metrics['frames']}`; geometry: `{metrics['shape_yx']}`.",
        "- Residual name: `noise_candidate` (not qualified measurement noise).",
        "- Scientific trace: unchanged accepted carrier.", "",
        "## Visuals", "",
        "Open `visuals/full_interval_decomposition_diagnostic.mp4`. All panels use fixed per-channel scaling across the complete review interval. Cyan rings are sparse-positive evaluation labels; unmatched pixels remain unknown.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_real_diagnostic(
    config: DependentMultiscaleConfig,
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a collision-safe CPU real-data diagnostic and fixed-scale video."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(config.resources["max_threads"]))
    target = config.output_dir
    partial = Path(str(target) + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError("completed or partial real diagnostic output exists")
    partial.mkdir(parents=True)
    (partial / "reconstruction").mkdir()
    (partial / "views").mkdir()
    (partial / "metrics").mkdir()
    (partial / "visuals").mkdir()
    progress = partial / "progress.jsonl"

    def heartbeat(event: str, **payload: Any) -> None:
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), "event": event, **payload}, sort_keys=True) + "\n")

    started = time.time()
    heartbeat("started")
    try:
        observation_source = np.load(config.input["observation_npy"], mmap_mode="r", allow_pickle=False)
        carrier_source = np.load(config.input["scientific_carrier_npy"], mmap_mode="r", allow_pickle=False)
        display_source = np.load(config.input["display_observation_npy"], mmap_mode="r", allow_pickle=False)
        observation = np.asarray(_review(observation_source, config), dtype=np.float32)
        if Path(config.input["observation_npy"]).resolve() == Path(
            config.input["scientific_carrier_npy"]
        ).resolve():
            carrier = observation
        else:
            carrier = np.asarray(_review(carrier_source, config), dtype=np.float32)
        display = _review(display_source, config)
        if observation.shape != carrier.shape or display.shape != carrier.shape:
            raise ValueError("observation, carrier, and display observation are not aligned")
        heartbeat("inputs_loaded", shape=list(observation.shape))
        views = build_scale_views(
            observation, _specs(config), quiet_count=int(config.frames["quiet_count"])
        )
        heartbeat("scale_views_completed")
        baseline = _linear_proxy(observation, views)
        nuisance, nuisance_names = build_frame_nuisance(observation)
        refined = refine_group_dependence(
            baseline, observation=observation, nuisance=nuisance,
            authority=0.75, maximum_information_samples=256, in_place=True,
        )
        decomposition = refined.decomposition
        channels = _channels(decomposition)
        closure = closure_metrics(
            observation,
            {key: channels[key] for key in (
                "background", "structured_signal", "structured_artifact", "noise_candidate"
            )},
        )
        heartbeat("decomposition_completed", closure_max=closure["normalized_maximum"])
        if bool(config.artifacts["write_dense_channels"]):
            for name, values in channels.items():
                _atomic_npy(partial / "reconstruction" / f"{name}.npy", np.asarray(values, dtype=np.float32))
                heartbeat("dense_channel_written", channel=name)
        with Path(config.input["labels_tsv"]).open("r", encoding="utf-8", newline="") as handle:
            label_rows = list(csv.DictReader(handle, delimiter="\t"))
        labels_xy = tuple({(float(row["x_px"]), float(row["y_px"])) for row in label_rows})
        video = write_decomposition_video(
            display_observation=display,
            scientific_carrier=carrier,
            channels=channels,
            views=views,
            labels_xy=labels_xy,
            review_start_ui=int(config.frames["review_start_ui"]),
            destination=partial / "visuals" / "full_interval_decomposition_diagnostic.mp4",
            fps=10.0,
        )
        video["path"] = "visuals/full_interval_decomposition_diagnostic.mp4"
        heartbeat("diagnostic_video_completed", bytes=video["bytes"])
        generated_gate = evaluate_generated_matrix()
        max_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        resource_cap_pass = max_rss_mib <= float(config.resources["max_ram_mib"])
        status = (
            "diagnostic_only_do_not_advance"
            if resource_cap_pass
            else "diagnostic_only_resource_cap_exceeded"
        )
        metrics = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "status": status,
            "scientific_carrier_status": "retained_external_authority",
            "real_execution_kind": "full_frame_linear_failure_analysis_proxy",
            "patchwise_W7_scientific_run_completed": False,
            "frames": int(len(observation)),
            "shape_yx": list(observation.shape[1:]),
            "closure": closure,
            "information_objective": refined.objective_terms,
            "nuisance_variables": list(nuisance_names),
            "residual_status": "noise_candidate",
            "generated_gate": generated_gate,
            "video": video,
            "preflight": dict(preflight),
            "elapsed_seconds": time.time() - started,
            "max_rss_mib": max_rss_mib,
            "resource_gate": {
                "max_ram_mib": float(config.resources["max_ram_mib"]),
                "peak_ram_mib": max_rss_mib,
                "pass": resource_cap_pass,
            },
        }
        _atomic_json(partial / "metrics.json", metrics)
        _atomic_json(partial / "metrics" / "dependence_diagnostics.json", {
            "objective_terms": refined.objective_terms,
            "diagnostics": refined.diagnostics,
        })
        _atomic_json(partial / "reconstruction" / "closure_summary.json", closure)
        _atomic_json(partial / "run_state.json", {
            "status": metrics["status"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "max_rss_mib": metrics["max_rss_mib"],
        })
        _write_report(partial / "REPORT.md", metrics)
        (partial / "RESULTS_INDEX.md").write_text(
            "# Results index\n\nStart with `REPORT.md`, then `visuals/full_interval_decomposition_diagnostic.mp4`, `metrics.json`, and `reconstruction/closure_summary.json`.\n",
            encoding="utf-8",
        )
        heartbeat("completed", status=metrics["status"])
        partial.replace(target)
        return metrics
    except Exception:
        heartbeat("failed")
        raise
