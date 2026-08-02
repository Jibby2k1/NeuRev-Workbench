"""Read-only resource audit for the preregistered generated screen."""
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from .config import InformationSeparationConfig
from .design import staged_fit_counts


def audit_generated_screen(
    config: InformationSeparationConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Estimate bounded screen resources without writing or authorizing a run."""
    output = output_dir.resolve()
    partial = Path(str(output) + ".partial")
    counts = staged_fit_counts(config)
    frames = int(config.generated["frame_count"])
    pixels = int(config.generated["height_px"]) * int(config.generated["width_px"])
    maximum_rank = max(
        int(value)
        for method_id in (
            "pca_reference", "multilag_sobi", "kernel_hsic_pairwise_rotation",
            "knn_mi_pairwise_rotation",
        )
        for value in config.methods[method_id]["ranks"]
    )
    largest_kernel_samples = max(
        int(config.methods["kernel_hsic_pairwise_rotation"]["max_fit_samples"]),
        int(config.methods["knn_mi_pairwise_rotation"]["max_fit_samples"]),
    )
    # Conservative simultaneous-array estimate, not a measured peak.
    movie_mib = frames * pixels * 8 / 2**20
    covariance_mib = pixels * pixels * 8 / 2**20
    kernels_mib = 4 * largest_kernel_samples**2 * 8 / 2**20
    estimate_mib = 512 + 4 * movie_mib + 2 * covariance_mib + kernels_mib + maximum_rank * frames * 8 / 2**20
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    free_disk_mib = shutil.disk_usage(probe).free / 2**20
    estimated_output_mib = counts["screen_fit_count"] * 0.15 + 32
    gates = {
        "output_absent": not output.exists(),
        "partial_output_absent": not partial.exists(),
        "estimated_ram_within_manifest_cap": estimate_mib <= int(config.resources["max_ram_mib"]),
        "estimated_output_within_manifest_cap": estimated_output_mib <= int(config.resources["max_output_mib"]),
        "disk_headroom_sufficient": free_disk_mib >= int(config.resources["min_free_disk_mib"]),
        "cpu_threads_bounded": 1 <= int(config.resources["cpu_threads"]) <= 8,
        "gpu_not_requested": True,
    }
    return {
        "schema_version": 1,
        "kind": "information_source_separation_generated_screen_read_only_preflight",
        "experiment_id": config.experiment_id,
        "ready_for_explicit_user_selection": bool(all(gates.values())),
        "run_authorized": False,
        "gates": gates,
        "output_dir": str(output),
        "counts": counts,
        "resources": {
            "estimated_peak_ram_mib": estimate_mib,
            "estimated_output_mib": estimated_output_mib,
            "free_disk_mib": free_disk_mib,
            "maximum_rank": maximum_rank,
            "largest_kernel_fit_samples": largest_kernel_samples,
            **config.resources,
        },
        "interpretation": (
            "A passing audit makes the bounded screen selectable; it does not launch "
            "or authorize 672 generated fits. The confirmation stage remains separate."
        ),
    }
