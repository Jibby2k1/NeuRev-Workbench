"""Preregistered center -> whiten -> ICA architecture screen.

The first-pass helpers in this module are deliberately label-free.  They keep
the three feature coordinates and sampled rows fixed while changing centering,
whitening, and ICA.  Heavy diagnostic media belongs to the frozen-finalist
stage, not this screen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from neurobench.experiments.msln_msica.artifacts import atomic_json, sha256_file, sha256_payload


CenterFamily = Literal["joint_residual", "joint_msln"]
WhiteningMode = Literal[
    "none",
    "ica_native_global",
    "global_diagonal_quiet",
    "global_full_oas_quiet",
    "global_full_fixed_ridge_0p05_quiet",
    "local_diagonal_quiet_t64",
    "local_full_oas_quiet_t64",
]

CONTEXT_BANKS: dict[str, tuple[str, str, str]] = {
    "reference_diverse": (
        "joint_s5_g1_t15_g1", "joint_s15_g3_t23_g1", "joint_s15_g3_t31_g1",
    ),
    "compact_temporal": (
        "joint_s5_g1_t9_g1", "joint_s5_g1_t15_g1", "joint_s5_g1_t31_g1",
    ),
    "broad_temporal": (
        "joint_s15_g3_t9_g1", "joint_s15_g3_t23_g1", "joint_s15_g3_t31_g1",
    ),
    "spatial_diverse_t15": (
        "joint_s5_g1_t15_g1", "joint_s9_g3_t15_g1", "joint_s15_g3_t15_g1",
    ),
}
CENTER_FAMILIES: tuple[CenterFamily, ...] = ("joint_residual", "joint_msln")
WHITENING_MODES: tuple[WhiteningMode, ...] = (
    "none", "ica_native_global", "global_diagonal_quiet",
    "global_full_oas_quiet", "global_full_fixed_ridge_0p05_quiet",
    "local_diagonal_quiet_t64", "local_full_oas_quiet_t64",
)
SCREEN_AUDIT_OPTOUT = (
    "User requested a metrics-only first-pass architecture screen on 2026-08-15; "
    "full diagnostics are mandatory for the frozen finalists."
)
SCHEMA_VERSION = 1
EXPERIMENT_ID = "spon_ca_burst_center_whiten_ica_v1"


@dataclass(frozen=True)
class ICAChoice:
    fun: Literal["identity", "logcosh", "exp", "cube"]
    seed: int
    max_iter: int
    tolerance: float

    @property
    def choice_id(self) -> str:
        tol = f"{self.tolerance:.0e}".replace("-", "m")
        return f"{self.fun}_s{self.seed}_i{self.max_iter}_t{tol}"


ICA_CHOICES: tuple[ICAChoice, ...] = (
    ICAChoice("identity", 7, 0, 0.0),
    ICAChoice("logcosh", 7, 300, 1e-5),
    ICAChoice("logcosh", 13, 300, 1e-5),
    ICAChoice("logcosh", 19, 300, 1e-5),
    ICAChoice("exp", 7, 300, 1e-5),
    ICAChoice("cube", 7, 300, 1e-5),
    ICAChoice("logcosh", 7, 800, 1e-4),
    ICAChoice("logcosh", 7, 800, 1e-5),
)


@dataclass(frozen=True)
class ArchitectureLane:
    center_family: CenterFamily
    context_bank: str
    whitening_mode: WhiteningMode
    ica: ICAChoice

    @property
    def lane_id(self) -> str:
        return "__".join((
            self.center_family, self.context_bank, self.whitening_mode,
            self.ica.choice_id,
        ))

    def to_dict(self) -> dict[str, object]:
        return {
            "lane_id": self.lane_id,
            "center_family": self.center_family,
            "context_bank": self.context_bank,
            "context_ids": list(CONTEXT_BANKS[self.context_bank]),
            "whitening_mode": self.whitening_mode,
            "ica": asdict(self.ica),
        }


def stage_a_lanes() -> tuple[tuple[CenterFamily, str], ...]:
    return tuple(product(CENTER_FAMILIES, CONTEXT_BANKS))


def stage_b_lanes(
    retained_centers: Sequence[tuple[CenterFamily, str]],
) -> tuple[tuple[CenterFamily, str, WhiteningMode], ...]:
    _validate_retained_centers(retained_centers)
    return tuple((*center, whitening) for center in retained_centers for whitening in WHITENING_MODES)


def stage_c_lanes(
    retained_center_whitening: Sequence[tuple[CenterFamily, str, WhiteningMode]],
) -> tuple[ArchitectureLane, ...]:
    if len(retained_center_whitening) > 6:
        raise ValueError("Stage B may retain at most six center/whitening combinations")
    lanes: list[ArchitectureLane] = []
    for center, bank, whitening in retained_center_whitening:
        _validate_center_bank(center, bank)
        if whitening not in WHITENING_MODES:
            raise ValueError(f"unknown whitening mode: {whitening}")
        lanes.extend(ArchitectureLane(center, bank, whitening, choice) for choice in ICA_CHOICES)
    return tuple(lanes)


def _validate_center_bank(center: str, bank: str) -> None:
    if center not in CENTER_FAMILIES or bank not in CONTEXT_BANKS:
        raise ValueError(f"unknown center/context bank: {center}/{bank}")


def _validate_retained_centers(values: Sequence[tuple[CenterFamily, str]]) -> None:
    if len(values) > 4:
        raise ValueError("Stage A may retain at most four center banks")
    for center, bank in values:
        _validate_center_bank(center, bank)


@dataclass(frozen=True)
class ICARotationFit:
    rotation: np.ndarray
    mean: np.ndarray
    converged: bool
    iterations: int
    internal_whitening: bool
    choice: ICAChoice

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation)
        mean = np.asarray(self.mean)
        if (
            rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]
            or mean.shape != (rotation.shape[1],)
            or not np.isfinite(rotation).all() or not np.isfinite(mean).all()
        ):
            raise ValueError("invalid ICA rotation fit")


def fit_ica_rotation(
    samples: np.ndarray,
    choice: ICAChoice,
    *,
    native_global_whitening: bool,
) -> ICARotationFit:
    """Fit a square ICA transform without accidental double whitening."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= values.shape[1] or not np.isfinite(values).all():
        raise ValueError("ICA samples must be a finite overdetermined N,D matrix")
    dimension = values.shape[1]
    if choice.fun == "identity":
        if native_global_whitening:
            mean = np.mean(values, axis=0)
            covariance = np.cov((values - mean).T, bias=True)
            eigenvalues, vectors = np.linalg.eigh(covariance)
            floor = max(float(np.max(eigenvalues)) * 1e-7, np.finfo(np.float64).eps)
            rotation = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, floor))) @ vectors.T
            return ICARotationFit(rotation, mean, True, 0, True, choice)
        return ICARotationFit(
            np.eye(dimension), np.zeros(dimension), True, 0,
            False, choice,
        )
    try:
        from sklearn.decomposition import FastICA
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for the ICA screen") from exc
    # Externally whitened coordinates must not be whitened a second time.
    whiten: str | bool = "unit-variance" if native_global_whitening else False
    estimator_kwargs = {
        "algorithm": "parallel", "whiten": whiten, "fun": choice.fun,
        "random_state": choice.seed, "max_iter": choice.max_iter,
        "tol": choice.tolerance,
    }
    if native_global_whitening:
        estimator_kwargs["n_components"] = dimension
    estimator = FastICA(**estimator_kwargs)
    estimator.fit(values)
    if native_global_whitening:
        # sklearn.transform(X) == (X - mean_) @ components_.T
        rotation = np.asarray(estimator.components_, dtype=np.float64)
        mean = np.asarray(estimator.mean_, dtype=np.float64)
    else:
        # With whiten=False sklearn does not center; preserve that exact contract.
        rotation = np.asarray(estimator.components_, dtype=np.float64)
        mean = np.zeros(dimension, dtype=np.float64)
    iterations = int(estimator.n_iter_)
    return ICARotationFit(
        rotation, mean, iterations < choice.max_iter, iterations,
        native_global_whitening, choice,
    )


def apply_ica_rotation(values: np.ndarray, fit: ICARotationFit) -> np.ndarray:
    source = np.asarray(values)
    if source.shape[-1] != len(fit.mean) or not np.isfinite(source).all():
        raise ValueError("ICA application values must be finite and match fitted coordinates")
    transformed = np.einsum(
        "...d,ed->...e", source.astype(np.float64) - fit.mean,
        fit.rotation, optimize=True,
    )
    return transformed.astype(np.float32)


def deterministic_sample_rows(
    values: np.ndarray,
    valid_frames: np.ndarray,
    *,
    maximum_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose fixed natural-prevalence rows; return data and flat sample IDs."""
    source = np.asarray(values)
    valid = np.asarray(valid_frames, dtype=bool)
    if source.ndim != 4 or valid.shape != (len(source),) or maximum_samples < 1:
        raise ValueError("expected T,Y,X,D values, aligned frames, and a positive budget")
    frame_ids = np.flatnonzero(valid)
    population = len(frame_ids) * source.shape[1] * source.shape[2]
    if population <= source.shape[-1]:
        raise ValueError("too few valid samples")
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(population, min(maximum_samples, population), replace=False))
    frame_position, pixel = np.divmod(ids, source.shape[1] * source.shape[2])
    y, x = np.divmod(pixel, source.shape[2])
    rows = source[frame_ids[frame_position], y, x]
    return np.asarray(rows, dtype=np.float64), ids.astype(np.int64)


def component_ambiguity(rotation: np.ndarray) -> float:
    """Return largest off-diagonal absolute row cosine (zero is separated)."""
    matrix = np.asarray(rotation, dtype=np.float64)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    cosine = np.abs(normalized @ normalized.T)
    np.fill_diagonal(cosine, 0)
    return float(np.max(cosine))


def load_screen_config(path: str | Path) -> dict[str, object]:
    """Load the small, strict screen contract without any label path."""
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"schema_version", "experiment_id", "source", "screen", "compute", "outputs"}
    if set(payload) != required:
        raise ValueError("screen config top-level keys differ from the frozen schema")
    if payload["schema_version"] != SCHEMA_VERSION or payload["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("wrong center-whiten-ICA experiment schema")
    expected = {
        "source": {"movie_path", "axes", "ui_one_based", "review_interval_ui", "quiet_interval_ui", "burst_intervals_ui"},
        "screen": {"maximum_stage_a", "maximum_stage_b", "maximum_finalists", "maximum_ica_samples", "sample_seed", "save_diagnostics", "audit_opt_out_reason"},
        "compute": {"device", "cpu_threads", "workers", "frame_chunk", "maximum_peak_ram_gb", "maximum_peak_vram_gb"},
        "outputs": {"root_dir"},
    }
    for section, keys in expected.items():
        if set(payload[section]) != keys:
            raise ValueError(f"{section} keys differ from the frozen schema")
    source, screen, compute = payload["source"], payload["screen"], payload["compute"]
    if source["axes"] != "TYX" or source["ui_one_based"] is not True:
        raise ValueError("source axes must be TYX with one-based UI intervals")
    if [screen["maximum_stage_a"], screen["maximum_stage_b"], screen["maximum_finalists"]] != [4, 6, 4]:
        raise ValueError("stage retention caps must remain 4/6/4")
    if screen["save_diagnostics"] is not False or screen["audit_opt_out_reason"] != SCREEN_AUDIT_OPTOUT:
        raise ValueError("first screen must retain its exact metrics-only audit declaration")
    if compute != {
        "device": "cuda", "cpu_threads": 4, "workers": 1, "frame_chunk": 4,
        "maximum_peak_ram_gb": 12, "maximum_peak_vram_gb": 4,
    }:
        raise ValueError("local safety envelope must remain CUDA/4 threads/1 worker/12 GiB RAM/4 GiB VRAM")
    base = config_path.parent
    source["movie_path"] = str((base / source["movie_path"]).resolve())
    payload["outputs"]["root_dir"] = str((base / payload["outputs"]["root_dir"]).resolve())
    payload["_config_path"] = str(config_path)
    return payload


def preflight_screen(config_path: str | Path) -> dict[str, object]:
    """Write collision-safe, label-free sizing and provenance artifacts."""
    config = load_screen_config(config_path)
    movie_path = Path(config["source"]["movie_path"])
    movie = np.load(movie_path, mmap_mode="r", allow_pickle=False)
    if movie.ndim != 3 or not np.issubdtype(movie.dtype, np.number):
        raise ValueError("movie must be a numeric TYX array")
    review_start, review_stop = map(int, config["source"]["review_interval_ui"])
    quiet_start, quiet_stop = map(int, config["source"]["quiet_interval_ui"])
    if not (1 <= review_start <= quiet_start <= quiet_stop <= review_stop <= len(movie)):
        raise ValueError("review/quiet intervals are invalid")
    root = Path(config["outputs"]["root_dir"])
    allowed_refresh_files = {"config.resolved.json", "preflight.json", "status.json"}
    if root.exists() and any(root.iterdir()):
        present = {item.name for item in root.iterdir()}
        if not present <= allowed_refresh_files:
            raise FileExistsError("refusing to reuse an architecture-screen root with stage artifacts")
    root.mkdir(parents=True, exist_ok=True)
    frames = review_stop - review_start + 1
    bank_bytes = frames * movie.shape[1] * movie.shape[2] * 3 * 4
    estimated_peak_ram = int(bank_bytes * 2.25 + frames * movie.shape[1] * movie.shape[2] * 4)
    ram_cap = int(config["compute"]["maximum_peak_ram_gb"] * 2**30)
    if estimated_peak_ram > ram_cap:
        raise RuntimeError("estimated one-bank peak exceeds the frozen host RAM cap")
    import shutil
    import psutil
    available_ram = int(psutil.virtual_memory().available)
    if estimated_peak_ram > available_ram:
        raise MemoryError("estimated one-bank peak exceeds currently available host RAM")
    disk = shutil.disk_usage(root)
    required_disk = int(bank_bytes * 0.1 + 64 * 2**20)
    if disk.free < required_disk:
        raise MemoryError("insufficient free disk for metrics and atomic checkpoints")
    try:
        import cupy as cp
        from neurobench.algorithms.local_covariance_whitening import SpatialTileConfig
        from neurobench.algorithms.local_covariance_whitening_cuda import estimate_cuda_apply_peak_bytes
        free_vram, total_vram = map(int, cp.cuda.runtime.memGetInfo())
        estimated_peak_vram = estimate_cuda_apply_peak_bytes(
            (frames, int(movie.shape[1]), int(movie.shape[2]), 3),
            SpatialTileConfig(64, 64, 32, 32),
            frame_chunk=int(config["compute"]["frame_chunk"]),
        )
        vram_cap = int(config["compute"]["maximum_peak_vram_gb"] * 2**30)
        if estimated_peak_vram > min(free_vram, vram_cap):
            raise MemoryError("estimated CUDA application peak exceeds free VRAM or the frozen cap")
        rng = np.random.default_rng(20260815)
        fixture = rng.normal(size=(257, 3)).astype(np.float32)
        transform = rng.normal(size=(3, 3)).astype(np.float32)
        cpu = fixture @ transform.T
        gpu = cp.asnumpy(cp.asarray(fixture) @ cp.asarray(transform).T)
        parity_error = float(np.max(np.abs(cpu - gpu)))
        cp.get_default_memory_pool().free_all_blocks()
        if parity_error > 2e-5:
            raise RuntimeError("CPU/CUDA matrix-application parity failed")
        gpu_payload: dict[str, object] = {
            "available": True, "free_bytes": free_vram, "total_bytes": total_vram,
            "estimated_peak_bytes": estimated_peak_vram,
            "configured_cap_bytes": vram_cap, "parity_max_abs_error": parity_error,
        }
    except ImportError as exc:
        raise RuntimeError("CUDA screen requires an importable CuPy environment") from exc
    payload = {
        "status": "preflight_ready",
        "labels_loaded": False,
        "source_shape": list(movie.shape),
        "source_dtype": str(movie.dtype),
        "source_sha256": sha256_file(movie_path),
        "review_frames": frames,
        "one_bank_bytes": int(bank_bytes),
        "estimated_peak_ram_bytes": estimated_peak_ram,
        "available_ram_bytes": available_ram,
        "disk_free_bytes": int(disk.free),
        "minimum_required_disk_bytes": required_disk,
        "active_python_processes": sum(
            "python" in (process.info.get("name") or "").lower()
            for process in psutil.process_iter(["name"])
        ),
        "gpu": gpu_payload,
        "stage_a_lane_count": len(stage_a_lanes()),
        "maximum_stage_b_lane_count": 4 * len(WHITENING_MODES),
        "maximum_stage_c_lane_count": 6 * len(ICA_CHOICES),
        "maximum_finalists": 4,
        "screen_outputs": "JSON/CSV/checkpoints only",
        "diagnostics_saved": False,
        "audit_opt_out_reason": SCREEN_AUDIT_OPTOUT,
        "execution_site": "local_cuda",
        "bala_allowed": False,
    }
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    payload["config_sha256"] = sha256_payload(public)
    atomic_json(root / "config.resolved.json", public)
    atomic_json(root / "preflight.json", payload)
    atomic_json(root / "status.json", {"status": "preflight_ready", "scientific_status": "not_run"})
    return payload
