"""Read-only preflight for the information source-separation benchmark."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .cnmf_adapter import audit_caiman_backend
from .config import InformationSeparationConfig


def method_configuration_counts(config: InformationSeparationConfig) -> dict[str, int]:
    methods = config.methods
    counts = {
        "pca_reference": (
            len(methods["pca_reference"]["ranks"])
            if methods["pca_reference"]["enabled"] else 0
        ),
        "multilag_sobi": (
            len(methods["multilag_sobi"]["ranks"])
            * len(methods["multilag_sobi"]["lag_sets"])
            * len(methods["multilag_sobi"]["covariance_shrinkages"])
            if methods["multilag_sobi"]["enabled"] else 0
        ),
        "kernel_hsic_pairwise_rotation": (
            len(methods["kernel_hsic_pairwise_rotation"]["ranks"])
            * len(methods["kernel_hsic_pairwise_rotation"]["bandwidth_scales"])
            if methods["kernel_hsic_pairwise_rotation"]["enabled"] else 0
        ),
        "knn_mi_pairwise_rotation": (
            len(methods["knn_mi_pairwise_rotation"]["ranks"])
            * len(methods["knn_mi_pairwise_rotation"]["neighbors"])
            if methods["knn_mi_pairwise_rotation"]["enabled"] else 0
        ),
        "caiman_cnmf_reference_adapter": 0,
        "group_energy_isa": 0,
        "spatial_noisy_parzen_infomax": 0,
    }
    return counts


def audit(config: InformationSeparationConfig, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Inspect inputs, collisions, dependencies, counts, RAM, and disk without writes."""
    target = (output_dir or config.output_dir).resolve()
    partial = Path(str(target) + ".partial")
    source_exists = config.source_video.is_file()
    source_valid = False
    source_shape = source_dtype = None
    if source_exists:
        source = np.load(config.source_video, mmap_mode="r", allow_pickle=False)
        source_shape = list(source.shape)
        source_dtype = str(source.dtype)
        source_valid = bool(source.ndim == 3 and source.shape[0] >= 2359)
    counts = method_configuration_counts(config)
    configurations = int(sum(counts.values()))
    fixture_count = config.generated_fixture_count()
    full_fit_count = configurations * fixture_count
    smoke_methods = sum(counts[key] > 0 for key in (
        "pca_reference", "multilag_sobi", "kernel_hsic_pairwise_rotation",
        "knn_mi_pairwise_rotation",
    ))
    smoke_fit_count = 2 * smoke_methods
    free_disk_mib = shutil.disk_usage(target.parent if target.parent.exists() else Path.cwd()).free / 2**20
    cnmf = audit_caiman_backend(
        str(config.methods["caiman_cnmf_reference_adapter"]["expected_version"])
    )
    gates = {
        "source_exists": source_exists,
        "source_geometry_valid": source_valid,
        "output_absent": not target.exists(),
        "partial_output_absent": not partial.exists(),
        "disk_headroom_sufficient": free_disk_mib >= int(config.resources["min_free_disk_mib"]),
        "full_matrix_not_authorized": True,
        "gpu_not_authorized": True,
    }
    return {
        "schema_version": 1,
        "kind": "information_source_separation_read_only_preflight",
        "experiment_id": config.experiment_id,
        "ready_for_tiny_cpu_smoke": bool(all(gates.values())),
        "ready_for_full_generated_matrix": False,
        "gates": gates,
        "output_dir": str(target),
        "source_video": str(config.source_video),
        "source_shape": source_shape,
        "source_dtype": source_dtype,
        "generated_fixture_count": fixture_count,
        "method_configuration_counts": counts,
        "method_configuration_count": configurations,
        "full_cartesian_fit_count": full_fit_count,
        "tiny_smoke_fit_count": smoke_fit_count,
        "cnmf_backend": cnmf,
        "resources": {
            **config.resources,
            "free_disk_mib": free_disk_mib,
        },
        "interpretation": (
            "This audit authorizes only a new-root tiny CPU smoke when ready. "
            "The full Cartesian generated matrix requires a staged screen/confirm "
            "manifest and separate explicit selection."
        ),
    }
