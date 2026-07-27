"""Resource-bounded multi-hypothesis CFAR screen for Spon Ca Burst.

The screen deliberately separates morphology (center or membrane), spatial
scale, reference estimator, and temporal pooling.  It calibrates every fixed
expert and each predeclared fusion only on quiet full-field pseudo-events.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import core as v1


@dataclass(frozen=True)
class ExpertSpec:
    morphology: str
    radius_px: int
    reference: str
    temporal: str

    @property
    def expert_id(self) -> str:
        return f"{self.morphology}_r{self.radius_px}_{self.reference}_{self.temporal}"


def expert_matrix(
    radii_px: Iterable[int] = (4, 6, 8),
) -> list[ExpertSpec]:
    """Return the predeclared 2 x 3 x 2 x 2 deterministic screen."""
    return [
        ExpertSpec(morphology, int(radius), reference, temporal)
        for morphology in ("center", "membrane")
        for radius in radii_px
        for reference in ("classic", "sector_censored")
        for temporal in ("lme", "causal_coherence")
    ]


@dataclass(frozen=True)
class MultiCFARConfig:
    experiment_id: str
    source_video: Path
    source_workbook: Path
    labels_tsv: Path
    label_summary: Path
    design_document: Path
    output_dir: Path
    quiet_start_ui: int
    quiet_end_ui: int
    radii_px: tuple[int, ...]
    support_px: int
    tolerance_px: int
    nms_distance_px: int
    false_alarms_per_field: int
    temporal_tau: float
    coherence_weight: float
    censored_sectors_kept: int
    cpu_threads: int
    frame_batch: int
    expert_batch: int
    max_ram_mib: int
    max_gpu_memory_mib: int
    min_free_disk_mib: int
    max_output_mib: int
    fusion_epochs: int
    fusion_learning_rate: float
    fusion_context_bound: float
    hard_negatives_per_window: int
    fusion_prior_temperature: float
    fusion_top_k: int

    @classmethod
    def load(cls, path: str | Path) -> "MultiCFARConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        root = source.parent
        resolve = lambda key: (root / raw[key]).resolve()
        frames = raw["frames"]
        detector = raw["detector"]
        resources = raw["resources"]
        training = raw.get("fusion_training", {})
        config = cls(
            experiment_id=str(raw["experiment_id"]),
            source_video=resolve("source_video"),
            source_workbook=resolve("source_workbook"),
            labels_tsv=resolve("labels_tsv"),
            label_summary=resolve("label_summary"),
            design_document=resolve("design_document"),
            output_dir=resolve("output_dir"),
            quiet_start_ui=int(frames["quiet_start_ui"]),
            quiet_end_ui=int(frames["quiet_end_ui"]),
            radii_px=tuple(int(x) for x in detector.get("radii_px", (4, 6, 8))),
            support_px=int(detector.get("support_px", 35)),
            tolerance_px=int(detector.get("tolerance_px", 4)),
            nms_distance_px=int(detector.get("nms_distance_px", 6)),
            false_alarms_per_field=int(detector.get("false_alarms_per_field", 5)),
            temporal_tau=float(detector.get("temporal_tau", 0.25)),
            coherence_weight=float(detector.get("coherence_weight", 0.25)),
            censored_sectors_kept=int(detector.get("censored_sectors_kept", 2)),
            cpu_threads=int(resources.get("cpu_threads", 6)),
            frame_batch=int(resources.get("frame_batch", 4)),
            expert_batch=int(resources.get("expert_batch", 4)),
            max_ram_mib=int(resources["max_ram_mib"]),
            max_gpu_memory_mib=int(resources["max_gpu_memory_mib"]),
            min_free_disk_mib=int(resources.get("min_free_disk_mib", 4096)),
            max_output_mib=int(resources.get("max_output_mib", 1024)),
            fusion_epochs=int(training.get("epochs", 300)),
            fusion_learning_rate=float(training.get("learning_rate", 1e-3)),
            fusion_context_bound=float(training.get("context_bound", 0.10)),
            hard_negatives_per_window=int(training.get("hard_negatives_per_window", 100)),
            fusion_prior_temperature=float(training.get("prior_temperature", 0.02)),
            fusion_top_k=int(training.get("top_k", 2)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.support_px < 21 or self.support_px % 2 != 1:
            raise ValueError("detector.support_px must be an odd integer >= 21")
        if not self.radii_px or min(self.radii_px) < 2:
            raise ValueError("detector.radii_px must contain radii >= 2")
        if 2 * max(self.radii_px) >= self.support_px:
            raise ValueError("support_px must exceed twice the largest radius")
        if not 1 <= self.censored_sectors_kept <= 4:
            raise ValueError("censored_sectors_kept must be in [1, 4]")
        if not 1 <= self.cpu_threads <= 24:
            raise ValueError("resources.cpu_threads must be in [1, 24]")
        if not 1 <= self.frame_batch <= 16 or not 1 <= self.expert_batch <= 8:
            raise ValueError("frame_batch must be <=16 and expert_batch must be <=8")
        if self.quiet_end_ui < self.quiet_start_ui:
            raise ValueError("quiet frame interval is empty")
        if not 1 <= self.fusion_epochs <= 2000:
            raise ValueError("fusion_training.epochs must be in [1, 2000]")
        if not 0 < self.fusion_learning_rate <= 0.01:
            raise ValueError("fusion_training.learning_rate must be in (0, 0.01]")
        if not 0 <= self.fusion_context_bound <= 0.25:
            raise ValueError("fusion_training.context_bound must be in [0, 0.25]")
        if not 10 <= self.hard_negatives_per_window <= 1000:
            raise ValueError("hard_negatives_per_window must be in [10, 1000]")
        if not 0 < self.fusion_prior_temperature <= 0.1:
            raise ValueError("fusion_training.prior_temperature must be in (0, 0.1]")
        if not 1 <= self.fusion_top_k <= 4:
            raise ValueError("fusion_training.top_k must be in [1, 4]")


def _normalize(mask: np.ndarray) -> np.ndarray:
    result = mask.astype(np.float32)
    total = float(result.sum())
    if total <= 0:
        raise ValueError("A CFAR kernel region is empty")
    return result / total


def build_kernel_bank(
    specs: list[ExpertSpec], support_px: int
) -> dict[str, np.ndarray]:
    """Build fixed test, full-reference, sector, and core kernels."""
    half = support_px // 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    distance = np.sqrt(xx * xx + yy * yy)
    angle = (np.arctan2(yy, xx) + 2 * np.pi) % (2 * np.pi)
    tests, references, sectors, cores = [], [], [], []
    for spec in specs:
        radius = float(spec.radius_px)
        if spec.morphology == "center":
            test = np.exp(-0.5 * (distance / max(radius / 2.0, 1.0)) ** 2)
            test[distance > radius] = 0
            core = distance <= max(1.0, radius * 0.4)
        elif spec.morphology == "membrane":
            test = (distance >= radius * 0.52) & (distance <= radius)
            core = distance <= max(1.0, radius * 0.38)
        else:
            raise ValueError(f"Unknown morphology: {spec.morphology}")
        outer = (distance >= radius * 1.12) & (distance <= radius * 1.95)
        outer_sectors = []
        for index in range(4):
            lo, hi = index * np.pi / 2, (index + 1) * np.pi / 2
            outer_sectors.append(_normalize(outer & (angle >= lo) & (angle < hi)))
        core_kernel = _normalize(core)
        outer_kernel = _normalize(outer)
        reference = outer_kernel if spec.morphology == "center" else 0.5 * (core_kernel + outer_kernel)
        tests.append(_normalize(test))
        references.append(reference.astype(np.float32))
        sectors.append(np.stack(outer_sectors))
        cores.append(core_kernel)
    return {
        "test": np.stack(tests),
        "reference": np.stack(references),
        "sectors": np.stack(sectors),
        "core": np.stack(cores),
    }


def _legacy_config(config: MultiCFARConfig) -> v1.Config:
    return v1.Config(
        experiment_id=config.experiment_id,
        source_video=config.source_video,
        source_workbook=config.source_workbook,
        labels_tsv=config.labels_tsv,
        label_summary=config.label_summary,
        design_document=config.design_document,
        output_dir=config.output_dir,
        quiet_start_ui=config.quiet_start_ui,
        quiet_end_ui=config.quiet_end_ui,
        scored_start_ui=1900,
        scored_end_ui=2359,
        support_px=config.support_px,
        tolerance_px=config.tolerance_px,
        nms_distance_px=config.nms_distance_px,
        epochs=1,
        masked_seeds=(0,),
        final_seeds=(0,),
        device="cuda",
        cpu_threads=config.cpu_threads,
        frame_batch=config.frame_batch,
        max_ram_mib=config.max_ram_mib,
        max_gpu_memory_mib=config.max_gpu_memory_mib,
        min_free_disk_mib=config.min_free_disk_mib,
        max_output_mib=config.max_output_mib,
    )


def preflight(
    config: MultiCFARConfig,
    artifact_dir: Path | None = None,
    *,
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    """Validate inputs, coordinates, resources, output collision, and matrix."""
    import shutil
    import torch

    files = (
        config.source_video,
        config.source_workbook,
        config.labels_tsv,
        config.label_summary,
        config.design_document,
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    if config.output_dir.exists() and not allow_existing_output:
        raise FileExistsError(f"Output exists: {config.output_dir}")
    array = np.load(config.source_video, mmap_mode="r")
    labels = v1.load_labels(config.labels_tsv)
    if len(array.shape) != 3:
        raise ValueError(f"Expected a T,Y,X array, got {array.shape}")
    if not labels:
        raise ValueError("No normalized labels were loaded")
    for row in labels:
        if not (0 <= row["x_px"] < array.shape[2] and 0 <= row["y_px"] < array.shape[1]):
            raise ValueError(f"Label outside video: {row}")
    free_gpu, total_gpu = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    disk_free = shutil.disk_usage(config.output_dir.parent).free // 2**20
    ram_free = v1.available_ram_mib()
    ready = (
        torch.cuda.is_available()
        and free_gpu // 2**20 >= config.max_gpu_memory_mib
        and ram_free >= config.max_ram_mib
        and disk_free >= config.min_free_disk_mib
    )
    specs = expert_matrix(config.radii_px)
    bank = build_kernel_bank(specs, config.support_px)
    payload = {
        "schema_version": 1,
        "generated_at": v1.utc_now(),
        "ready": bool(ready),
        "experiment_id": config.experiment_id,
        "video_shape": list(array.shape),
        "label_rows": len(labels),
        "unique_rois": len({row["roi_identity"] for row in labels}),
        "burst_ids": sorted({row["burst_id"] for row in labels}),
        "morphology_labels_available": False,
        "coordinate_contract": "UI frames one-based inclusive; arrays zero-based half-open; x=column,y=row.",
        "unknown_pixel_contract": "Unlabeled event pixels are unknown, never training negatives.",
        "matrix": [asdict(spec) | {"expert_id": spec.expert_id} for spec in specs],
        "planned_fixed_experts": len(specs),
        "planned_quiet_calibrations": len(specs) + 2,
        "planned_predeclared_fusions": ["max_margin", "logmeanexp_margin"],
        "planned_crossfit_fixed_selections": 4,
        "planned_bounded_gate_fits": 4,
        "kernel_checks": {
            "test_sum_min": float(bank["test"].sum((1, 2)).min()),
            "reference_sum_min": float(bank["reference"].sum((1, 2)).min()),
            "sector_sum_min": float(bank["sectors"].sum((2, 3)).min()),
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": v1.sha256(path)}
            for path in files
        ],
        "resources": {
            "cpu_threads": config.cpu_threads,
            "frame_batch": config.frame_batch,
            "expert_batch": config.expert_batch,
            "ram_available_mib": ram_free,
            "ram_cap_mib": config.max_ram_mib,
            "gpu_free_mib": free_gpu // 2**20,
            "gpu_total_mib": total_gpu // 2**20,
            "gpu_cap_mib": config.max_gpu_memory_mib,
            "disk_free_mib": disk_free,
        },
        "checkpoint": {
            "name": "C0_preflight",
            "advance": bool(ready),
            "limitation": "Overall sparse-label recall is measurable; per-morphology performance requires new type labels.",
        },
    }
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        overlay = artifact_dir / "projection_overlay.png"
        v1._write_overlay(array, labels, overlay)
        payload["projection_overlay"] = str(overlay)
        v1.atomic_json(artifact_dir / "preflight.json", payload)
    if not ready:
        raise RuntimeError("Multi-hypothesis CFAR preflight failed its resource gate")
    return payload


def _score_maps(
    frames: np.ndarray,
    specs: list[ExpertSpec],
    bank: dict[str, np.ndarray],
    config: MultiCFARConfig,
) -> np.ndarray:
    """Pool framewise expert maps while keeping CUDA allocations bounded."""
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda")
    tau = config.temporal_tau
    results: list[np.ndarray] = []
    padding = config.support_px // 2
    for first in range(0, len(specs), config.expert_batch):
        block = specs[first : first + config.expert_batch]
        stop = first + len(block)
        kt = torch.from_numpy(bank["test"][first:stop, None]).to(device)
        kr = torch.from_numpy(bank["reference"][first:stop, None]).to(device)
        ks = torch.from_numpy(bank["sectors"][first:stop].reshape(-1, 1, config.support_px, config.support_px)).to(device)
        kc = torch.from_numpy(bank["core"][first:stop, None]).to(device)
        q = ((kt - kr) ** 2).sum((1, 2, 3)).clamp_min(1e-6)
        b = (1 - kr.square().sum((1, 2, 3))).clamp_min(1e-4)
        accumulator = None
        previous = None
        count = 0
        for start in range(0, len(frames), config.frame_batch):
            chunk = torch.from_numpy(np.ascontiguousarray(frames[start : start + config.frame_batch, None])).to(device)
            padded = F.pad(chunk, (padding,) * 4, mode="reflect")
            padded2 = padded.square()
            test = F.conv2d(padded, kt)
            ref = F.conv2d(padded, kr)
            second = F.conv2d(padded2, kr)
            sector_mean = F.conv2d(padded, ks).reshape(len(chunk), len(block), 4, *test.shape[-2:])
            sector_second = F.conv2d(padded2, ks).reshape_as(sector_mean)
            keep = min(config.censored_sectors_kept, 4)
            selected = sector_mean.topk(keep, dim=2, largest=False).indices
            censored_mean = sector_mean.gather(2, selected).mean(2)
            censored_second = sector_second.gather(2, selected).mean(2)
            core_mean = F.conv2d(padded, kc)
            core_second = F.conv2d(padded2, kc)
            for local, spec in enumerate(block):
                if spec.reference == "sector_censored":
                    mu = censored_mean[:, local]
                    sec = censored_second[:, local]
                    if spec.morphology == "membrane":
                        mu = 0.5 * (mu + core_mean[:, local])
                        sec = 0.5 * (sec + core_second[:, local])
                else:
                    mu, sec = ref[:, local], second[:, local]
                variance = (sec - mu.square()).clamp_min(0)
                z = (test[:, local] - mu) / torch.sqrt(q[local] * variance / b[local] + 1e-6)
                score = torch.log1p(torch.relu(z).square())
                if previous is None:
                    previous = [None] * len(block)
                for frame_score in score:
                    if spec.temporal == "causal_coherence" and previous[local] is not None:
                        frame_score = frame_score + config.coherence_weight * torch.sqrt(
                            frame_score.clamp_min(0) * previous[local].clamp_min(0) + 1e-8
                        )
                    value = frame_score / tau
                    if accumulator is None:
                        accumulator = [None] * len(block)
                    accumulator[local] = value if accumulator[local] is None else torch.logaddexp(accumulator[local], value)
                    count += 1 if local == 0 else 0
                    previous[local] = frame_score
            del chunk, padded, padded2, test, ref, second, sector_mean, sector_second
        if not accumulator or count != len(frames):
            raise RuntimeError("Temporal pooling did not consume every frame")
        results.extend([(tau * (item - math.log(count))).cpu().numpy() for item in accumulator])
        del kt, kr, ks, kc, accumulator, previous
        torch.cuda.empty_cache()
    return np.stack(results)


def _quiet_windows(quiet: np.ndarray) -> list[np.ndarray]:
    durations = (24, 24, 28, 47)
    starts = (0, 24, 48, 53)
    if max(start + duration for start, duration in zip(starts, durations)) > len(quiet):
        raise ValueError("Quiet interval is too short for predeclared pseudo-events")
    return [quiet[start : start + duration] for start, duration in zip(starts, durations)]


def _thresholds(quiet_maps: list[np.ndarray], config: MultiCFARConfig) -> tuple[np.ndarray, np.ndarray]:
    experts = quiet_maps[0].shape[0]
    thresholds, scales = [], []
    for index in range(experts):
        values = []
        for maps in quiet_maps:
            values.extend(value for value, _, _ in v1._peaks(maps[index], config.nms_distance_px, limit=2000))
        ranked = np.sort(np.asarray(values, dtype=np.float64))[::-1]
        if len(ranked) < config.false_alarms_per_field:
            raise RuntimeError("Too few quiet peaks for calibration")
        thresholds.append(float(np.nextafter(ranked[config.false_alarms_per_field - 1], np.inf)))
        median = float(np.median(ranked))
        mad = float(np.median(np.abs(ranked - median)) * 1.4826)
        scales.append(max(mad, float(np.std(ranked)), 1e-3))
    return np.asarray(thresholds, np.float32), np.asarray(scales, np.float32)


def _margin_maps(maps: np.ndarray, thresholds: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (maps - thresholds[:, None, None]) / scales[:, None, None]


def _fuse(margins: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max_margin":
        return margins.max(0)
    if mode == "logmeanexp_margin":
        from scipy.special import logsumexp

        return logsumexp(margins, axis=0) - math.log(len(margins))
    raise ValueError(mode)


def _calibrate_scalar(maps: list[np.ndarray], config: MultiCFARConfig) -> float:
    values = []
    for score in maps:
        values.extend(value for value, _, _ in v1._peaks(score, config.nms_distance_px, limit=2000))
    ranked = sorted(values, reverse=True)
    return float(np.nextafter(ranked[config.false_alarms_per_field - 1], np.inf))


def _evaluate_map(score: np.ndarray, threshold: float, rows: list[dict[str, Any]], config: MultiCFARConfig) -> dict[str, Any]:
    peaks = [peak for peak in v1._peaks(score, config.nms_distance_px, limit=2000) if peak[0] >= threshold]
    matches = v1._match(peaks, rows, config.tolerance_px)
    return {
        "label_count": len(rows),
        "matched": len(matches),
        "recall": len(matches) / len(rows) if rows else 0.0,
        "candidate_count": len(peaks),
        "threshold": float(threshold),
    }


def _crossfit_select(experts: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a fixed expert on three bursts and report only the fourth."""
    folds = []
    for heldout in (1, 2, 3, 4):
        def rank(item: dict[str, Any]) -> tuple[float, int, str]:
            train = [fold for fold in item["folds"] if fold["burst_id"] != heldout]
            return (
                float(np.mean([fold["recall"] for fold in train])),
                -sum(fold["candidate_count"] for fold in train),
                item["expert_id"],
            )

        selected = max(experts, key=rank)
        test = next(fold for fold in selected["folds"] if fold["burst_id"] == heldout)
        folds.append({
            "burst_id": heldout,
            "selected_expert": selected["expert_id"],
            "training_mean_recall": rank(selected)[0],
            **{key: value for key, value in test.items() if key != "burst_id"},
        })
    return {
        "method": "crossfit_train_burst_expert_selection",
        "mean_recall": float(np.mean([fold["recall"] for fold in folds])),
        "pooled_matched": int(sum(fold["matched"] for fold in folds)),
        "pooled_labels": int(sum(fold["label_count"] for fold in folds)),
        "folds": folds,
    }


def _label_features(
    margins: np.ndarray,
    rows: list[dict[str, Any]],
    tolerance_px: int,
) -> np.ndarray:
    """Extract each expert's best score inside the registered label disk."""
    features = []
    height, width = margins.shape[-2:]
    for row in rows:
        x, y = int(round(row["x_px"])), int(round(row["y_px"]))
        x0, x1 = max(0, x - tolerance_px), min(width, x + tolerance_px + 1)
        y0, y1 = max(0, y - tolerance_px), min(height, y + tolerance_px + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= tolerance_px**2
        patch = margins[:, y0:y1, x0:x1]
        features.append(patch[:, disk].max(1))
    return np.asarray(features, dtype=np.float32)


def _quiet_hard_negatives(
    quiet_margins: list[np.ndarray],
    config: MultiCFARConfig,
) -> np.ndarray:
    """Mine label-free full-field quiet candidates from the union of experts."""
    features = []
    for margins in quiet_margins:
        union = margins.max(0)
        peaks = v1._peaks(
            union,
            config.nms_distance_px,
            limit=config.hard_negatives_per_window,
        )
        features.extend(margins[:, y, x] for _, x, y in peaks)
    return np.asarray(features, dtype=np.float32)


def _fit_bounded_gate(
    positives: np.ndarray,
    negatives: np.ndarray,
    prior_quality: np.ndarray,
    config: MultiCFARConfig,
    seed: int,
) -> dict[str, Any]:
    """Tune a small contextual residual around train-fold expert priors."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    device = torch.device("cuda")
    pos = torch.from_numpy(positives).to(device)
    neg = torch.from_numpy(negatives).to(device)
    quality = np.asarray(prior_quality, dtype=np.float32)
    active = np.argsort(quality)[-config.fusion_top_k :]
    prior_values = np.full_like(quality, -20.0)
    prior_values[active] = quality[active] / config.fusion_prior_temperature
    prior_values -= prior_values.max()
    prior = torch.from_numpy(prior_values).to(device)
    raw_context = torch.zeros(pos.shape[1], device=device, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [raw_context], lr=config.fusion_learning_rate, weight_decay=1e-4
    )

    def score(features):
        clipped = features.clamp(-4, 4)
        logits = prior[None] + config.fusion_context_bound * torch.tanh(raw_context)[None] * clipped
        weights = torch.softmax(logits, dim=1)
        return (weights * features).sum(1)

    history = []
    for epoch in range(1, config.fusion_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        positive_score = score(pos)
        negative_score = score(neg)
        hard_count = min(len(negative_score), max(16, len(positive_score) * 2))
        hard = torch.topk(negative_score, hard_count).values
        rank_loss = F.softplus(1.0 - positive_score[:, None] + hard[None]).mean()
        drift = raw_context.square().mean()
        loss = rank_loss + 0.02 * drift
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_context], 1.0)
        optimizer.step()
        if epoch in (1, config.fusion_epochs):
            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "rank_loss": float(rank_loss.item()),
                "positive_mean": float(positive_score.mean().item()),
                "hard_negative_mean": float(hard.mean().item()),
            })
    return {
        "prior_logits": prior.detach().cpu().numpy(),
        "context": torch.tanh(raw_context).detach().cpu().numpy(),
        "history": history,
    }


def _apply_bounded_gate(
    margins: np.ndarray,
    fitted: dict[str, Any],
    config: MultiCFARConfig,
) -> np.ndarray:
    prior = np.asarray(fitted["prior_logits"], dtype=np.float32)[:, None, None]
    context = np.asarray(fitted["context"], dtype=np.float32)[:, None, None]
    clipped = np.clip(margins, -4, 4)
    logits = prior + config.fusion_context_bound * context * clipped
    logits -= logits.max(0, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(0, keepdims=True)
    return np.sum(weights * margins, axis=0, dtype=np.float32)


def _run_crossfit_gate(
    quiet_maps: list[np.ndarray],
    event_maps: dict[int, np.ndarray],
    thresholds: np.ndarray,
    scales: np.ndarray,
    labels: list[dict[str, Any]],
    experts: list[dict[str, Any]],
    config: MultiCFARConfig,
    heartbeat,
) -> dict[str, Any]:
    quiet_margins = [_margin_maps(maps, thresholds, scales) for maps in quiet_maps]
    negatives = _quiet_hard_negatives(quiet_margins, config)
    folds = []
    for heldout in (1, 2, 3, 4):
        train_bursts = [burst for burst in (1, 2, 3, 4) if burst != heldout]
        positives = np.concatenate([
            _label_features(
                _margin_maps(event_maps[burst], thresholds, scales),
                [row for row in labels if row["burst_id"] == burst],
                config.tolerance_px,
            )
            for burst in train_bursts
        ])
        prior_quality = np.asarray([
            np.mean([
                fold["recall"] for fold in expert["folds"]
                if fold["burst_id"] in train_bursts
            ])
            for expert in experts
        ], dtype=np.float32)
        fitted = _fit_bounded_gate(positives, negatives, prior_quality, config, 4100 + heldout)
        quiet_scores = [_apply_bounded_gate(margins, fitted, config) for margins in quiet_margins]
        threshold = _calibrate_scalar(quiet_scores, config)
        event_margin = _margin_maps(event_maps[heldout], thresholds, scales)
        score = _apply_bounded_gate(event_margin, fitted, config)
        rows = [row for row in labels if row["burst_id"] == heldout]
        metrics = _evaluate_map(score, threshold, rows, config)
        prior_weights = np.exp(fitted["prior_logits"] - np.max(fitted["prior_logits"]))
        prior_weights /= prior_weights.sum()
        folds.append({
            "burst_id": heldout,
            **metrics,
            "training_positive_count": len(positives),
            "quiet_hard_negative_count": len(negatives),
            "largest_prior_expert": experts[int(np.argmax(prior_weights))]["expert_id"],
            "largest_prior_weight": float(prior_weights.max()),
            "max_context_residual": float(np.max(np.abs(fitted["context"]))),
            "history": fitted["history"],
        })
        heartbeat({"stage": "C2_crossfit_gate", "completed_fold": heldout, "total": 4})
    return {
        "method": "crossfit_bounded_context_gate",
        "fit_count": 4,
        "mean_recall": float(np.mean([fold["recall"] for fold in folds])),
        "pooled_matched": int(sum(fold["matched"] for fold in folds)),
        "pooled_labels": int(sum(fold["label_count"] for fold in folds)),
        "folds": folds,
    }


def run(config: MultiCFARConfig) -> dict[str, Any]:
    """Execute the deterministic checkpoint; later stages consume its manifest."""
    import torch

    preflight(config)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    resolved = asdict(config)
    for key, value in tuple(resolved.items()):
        if isinstance(value, Path):
            resolved[key] = str(value)
    v1.atomic_json(config.output_dir / "config.resolved.json", resolved)
    labels = v1.load_labels(config.labels_tsv)
    quiet, bursts, baseline = v1._prepare_arrays(_legacy_config(config), labels)
    preparation = {
        "mode": "quiet-median positive residual",
        "quiet_frames": len(quiet),
        "baseline_min": float(baseline.min()),
        "baseline_median": float(np.median(baseline)),
        "baseline_max": float(baseline.max()),
        "quiet_only_normalization": True,
    }
    specs = expert_matrix(config.radii_px)
    bank = build_kernel_bank(specs, config.support_px)
    total_gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, config.max_gpu_memory_mib * 2**20 / total_gpu_bytes), device=0
    )
    torch.cuda.reset_peak_memory_stats()
    quiet_maps, event_maps = [], {}
    progress_path = config.output_dir / "progress.jsonl"

    def heartbeat(payload: dict[str, Any]) -> None:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": v1.utc_now()} | payload, sort_keys=True) + "\n")

    heartbeat({"stage": "C1_deterministic", "status": "started", "experts": len(specs)})
    for index, frames in enumerate(_quiet_windows(quiet), 1):
        quiet_maps.append(_score_maps(frames, specs, bank, config))
        heartbeat({"stage": "C1_deterministic", "kind": "quiet", "completed": index, "total": 4})
    for burst_id in sorted(bursts):
        event_maps[burst_id] = _score_maps(bursts[burst_id], specs, bank, config)
        heartbeat({"stage": "C1_deterministic", "kind": "event", "completed_burst": burst_id, "total": 4})
    thresholds, scales = _thresholds(quiet_maps, config)
    rows_out = []
    expert_summaries = []
    for expert_index, spec in enumerate(specs):
        folds = []
        for burst_id in sorted(event_maps):
            rows = [row for row in labels if row["burst_id"] == burst_id]
            metrics = _evaluate_map(event_maps[burst_id][expert_index], thresholds[expert_index], rows, config)
            folds.append({"burst_id": burst_id} | metrics)
            rows_out.append({"method": spec.expert_id, "burst_id": burst_id} | metrics)
        expert_summaries.append({
            "expert_id": spec.expert_id,
            "factors": asdict(spec),
            "mean_recall": float(np.mean([fold["recall"] for fold in folds])),
            "pooled_matched": int(sum(fold["matched"] for fold in folds)),
            "pooled_labels": int(sum(fold["label_count"] for fold in folds)),
            "folds": folds,
        })
    fusion_summaries = []
    for mode in ("max_margin", "logmeanexp_margin"):
        quiet_fused = [_fuse(_margin_maps(maps, thresholds, scales), mode) for maps in quiet_maps]
        fusion_threshold = _calibrate_scalar(quiet_fused, config)
        folds = []
        for burst_id in sorted(event_maps):
            score = _fuse(_margin_maps(event_maps[burst_id], thresholds, scales), mode)
            rows = [row for row in labels if row["burst_id"] == burst_id]
            metrics = _evaluate_map(score, fusion_threshold, rows, config)
            folds.append({"burst_id": burst_id} | metrics)
            rows_out.append({"method": mode, "burst_id": burst_id} | metrics)
        fusion_summaries.append({
            "method": mode,
            "mean_recall": float(np.mean([fold["recall"] for fold in folds])),
            "pooled_matched": int(sum(fold["matched"] for fold in folds)),
            "pooled_labels": int(sum(fold["label_count"] for fold in folds)),
            "threshold": fusion_threshold,
            "folds": folds,
        })
    best = max(expert_summaries, key=lambda item: (item["mean_recall"], -sum(f["candidate_count"] for f in item["folds"])))
    predeclared_best = max(fusion_summaries, key=lambda item: item["mean_recall"])
    crossfit_fixed = _crossfit_select(expert_summaries)
    fixed_baseline = 0.132763975
    baseline_folds = (1 / 15, 2 / 20, 4 / 21, 4 / 23)
    crossfit_wins = sum(
        fold["recall"] > baseline
        for fold, baseline in zip(crossfit_fixed["folds"], baseline_folds)
    )
    checkpoint = {
        "name": "C1_deterministic",
        "fixed_guarded_cfar_mean_recall": fixed_baseline,
        "best_single_diagnostic": best["expert_id"],
        "best_single_mean_recall": best["mean_recall"],
        "best_predeclared_fusion": predeclared_best["method"],
        "best_predeclared_fusion_mean_recall": predeclared_best["mean_recall"],
        "crossfit_fixed_selection_mean_recall": crossfit_fixed["mean_recall"],
        "crossfit_fixed_selection_burst_wins": crossfit_wins,
        "advance_to_hard_negative_fusion": bool(
            crossfit_fixed["mean_recall"] >= fixed_baseline + 0.03
            and crossfit_wins >= 2
        ),
        "revision_from_v4": "Use leakage-safe train-burst expert selection rather than an all-expert fusion gate; v4 showed that indiscriminate inclusion diluted strong center experts.",
        "stop_reason_if_false": "Cross-fitted expert selection did not improve fixed guarded CFAR by >=0.03 on mean recall and >=2 bursts.",
    }
    learned_fusion = None
    checkpoint_c2 = {
        "name": "C2_crossfit_gate",
        "status": "not_run_C1_gate",
        "advance_to_kernel_residuals": False,
    }
    if checkpoint["advance_to_hard_negative_fusion"]:
        heartbeat({"stage": "C2_crossfit_gate", "status": "started", "fits": 4})
        learned_fusion = _run_crossfit_gate(
            quiet_maps,
            event_maps,
            thresholds,
            scales,
            labels,
            expert_summaries,
            config,
            heartbeat,
        )
        losses = sum(
            learned["recall"] < fixed["recall"]
            for learned, fixed in zip(learned_fusion["folds"], crossfit_fixed["folds"])
        )
        advance_c3 = bool(
            learned_fusion["mean_recall"] >= crossfit_fixed["mean_recall"] + 0.02
            and losses <= 1
        )
        checkpoint_c2 = {
            "name": "C2_crossfit_gate",
            "status": "complete",
            "crossfit_fixed_mean_recall": crossfit_fixed["mean_recall"],
            "learned_gate_mean_recall": learned_fusion["mean_recall"],
            "burst_losses": losses,
            "advance_to_kernel_residuals": advance_c3,
            "stop_reason_if_false": "Bounded gate did not improve cross-fitted fixed selection by >=0.02 without losing more than one burst.",
        }
        for fold in learned_fusion["folds"]:
            rows_out.append({
                "method": learned_fusion["method"],
                **{key: fold[key] for key in ("burst_id", "label_count", "matched", "recall", "candidate_count", "threshold")},
            })
    with (config.output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["method", "burst_id", "label_count", "matched", "recall", "candidate_count", "threshold"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    payload = {
        "schema_version": 1,
        "generated_at": v1.utc_now(),
        "experiment_id": config.experiment_id,
        "status": "checkpoint_complete",
        "scientific_scope": "Sparse-label recall and calibrated candidate burden; unlabeled event pixels remain unknown.",
        "preparation": preparation,
        "expert_count": len(specs),
        "fixed_experts": expert_summaries,
        "predeclared_fusions": fusion_summaries,
        "crossfit_fixed_selection": crossfit_fixed,
        "learned_fusion": learned_fusion,
        "checkpoint": checkpoint,
        "checkpoint_c2": checkpoint_c2,
        "checkpoint_c3": {
            "name": "C3_bounded_kernel_residuals",
            "status": "pending" if checkpoint_c2["advance_to_kernel_residuals"] else "not_run_C2_gate",
        },
        "resources": {
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() // 2**20,
            "rss_mib": v1.rss_mib(),
        },
    }
    v1.atomic_json(config.output_dir / "results.json", payload)
    heartbeat({
        "stage": "experiment",
        "status": "checkpoint_complete",
        "advance_to_kernel_residuals": checkpoint_c2["advance_to_kernel_residuals"],
    })
    return payload
