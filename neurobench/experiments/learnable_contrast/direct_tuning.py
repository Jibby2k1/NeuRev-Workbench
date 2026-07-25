"""Conservatively tune a detector initialized from the raw-direct baseline."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import core as v1


VARIANTS = ("temporal_amplitude", "spatial", "guarded_auxiliary")


@dataclass(frozen=True)
class DirectTuningConfig:
    experiment_id: str
    source_video: Path
    source_workbook: Path
    labels_tsv: Path
    label_summary: Path
    design_document: Path
    output_dir: Path
    quiet_start_ui: int
    quiet_end_ui: int
    epochs: int
    min_epochs: int
    patience: int
    learning_rates: tuple[float, ...]
    screen_seed: int
    confirmation_seeds: tuple[int, ...]
    confirmation_jitter_std: float
    trust_region_weight: float
    auxiliary_init_weight: float
    frame_batch: int
    cpu_threads: int
    max_ram_mib: int
    max_gpu_memory_mib: int
    min_free_disk_mib: int
    support_px: int = 21
    spatial_support_px: int = 5
    tolerance_px: int = 4
    nms_distance_px: int = 6

    @classmethod
    def load(cls, path: str | Path) -> "DirectTuningConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        root = source.parent
        p = lambda name: (root / raw[name]).resolve()
        tr, res, frames = raw["training"], raw["resources"], raw["frames"]
        config = cls(
            experiment_id=str(raw["experiment_id"]), source_video=p("source_video"),
            source_workbook=p("source_workbook"), labels_tsv=p("labels_tsv"),
            label_summary=p("label_summary"), design_document=p("design_document"),
            output_dir=p("output_dir"), quiet_start_ui=int(frames["quiet_start_ui"]),
            quiet_end_ui=int(frames["quiet_end_ui"]), epochs=int(tr["epochs"]),
            min_epochs=int(tr["min_epochs"]), patience=int(tr["patience"]),
            learning_rates=tuple(float(x) for x in tr["learning_rates"]),
            screen_seed=int(tr["screen_seed"]),
            confirmation_seeds=tuple(int(x) for x in tr["confirmation_seeds"]),
            confirmation_jitter_std=float(tr["confirmation_jitter_std"]),
            trust_region_weight=float(tr["trust_region_weight"]),
            auxiliary_init_weight=float(tr["auxiliary_init_weight"]),
            frame_batch=int(res["frame_batch"]), cpu_threads=int(res["cpu_threads"]),
            max_ram_mib=int(res["max_ram_mib"]),
            max_gpu_memory_mib=int(res["max_gpu_memory_mib"]),
            min_free_disk_mib=int(res["min_free_disk_mib"]),
            support_px=int(raw.get("support_px", 21)),
            spatial_support_px=int(raw.get("spatial_support_px", 5)),
            tolerance_px=int(raw.get("tolerance_px", 4)),
            nms_distance_px=int(raw.get("nms_distance_px", 6)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.output_dir.exists():
            raise FileExistsError(f"Output exists: {self.output_dir}")
        if self.learning_rates != (3e-5, 1e-4, 3e-4):
            raise ValueError("The guarded v3 screen requires learning rates 3e-5, 1e-4, 3e-4")
        if len(self.confirmation_seeds) != 3:
            raise ValueError("Exactly three confirmation seeds are required")
        if self.epochs < self.min_epochs or self.min_epochs < 1 or self.patience < 1:
            raise ValueError("Invalid epoch or patience settings")
        if self.support_px < 3 or self.support_px % 2 != 1:
            raise ValueError("support_px must be odd and >=3")
        if self.spatial_support_px < 3 or self.spatial_support_px % 2 != 1:
            raise ValueError("spatial_support_px must be odd and >=3")
        if not 1 <= self.frame_batch <= 128 or not 1 <= self.cpu_threads <= 24:
            raise ValueError("Invalid resource bounds")
        if not 0 < self.auxiliary_init_weight <= 0.01:
            raise ValueError("auxiliary_init_weight must be in (0, 0.01]")


def screen_matrix(learning_rates: tuple[float, ...] = (3e-5, 1e-4, 3e-4)) -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANTS:
        for lr in learning_rates:
            token = f"{lr:.0e}".replace("-", "m")
            rows.append({"variant": variant, "learning_rate": lr,
                         "combination_id": f"{variant}__lr{token}"})
    return rows


def _legacy_config(c: DirectTuningConfig) -> v1.Config:
    return v1.Config(
        experiment_id=c.experiment_id, source_video=c.source_video,
        source_workbook=c.source_workbook, labels_tsv=c.labels_tsv,
        label_summary=c.label_summary, design_document=c.design_document,
        output_dir=c.output_dir, quiet_start_ui=c.quiet_start_ui,
        quiet_end_ui=c.quiet_end_ui, scored_start_ui=1900, scored_end_ui=2359,
        support_px=c.support_px, tolerance_px=c.tolerance_px,
        nms_distance_px=c.nms_distance_px, epochs=c.epochs,
        masked_seeds=tuple(range(10)), final_seeds=tuple(range(5)), device="cuda",
        cpu_threads=c.cpu_threads, frame_batch=c.frame_batch,
        max_ram_mib=c.max_ram_mib, max_gpu_memory_mib=c.max_gpu_memory_mib,
        min_free_disk_mib=c.min_free_disk_mib, max_output_mib=2048,
    )


def preflight(c: DirectTuningConfig, artifact_dir: Path | None = None) -> dict[str, Any]:
    import torch

    files = (c.source_video, c.source_workbook, c.labels_tsv, c.label_summary, c.design_document)
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    video = np.load(c.source_video, mmap_mode="r", allow_pickle=False)
    labels = v1.load_labels(c.labels_tsv)
    free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    disk_probe = c.output_dir.parent
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    disk_free = shutil.disk_usage(disk_probe).free // 2**20
    ready = bool(torch.cuda.is_available() and v1.available_ram_mib() >= c.max_ram_mib
                 and free // 2**20 >= c.max_gpu_memory_mib and disk_free >= c.min_free_disk_mib)
    payload = {
        "schema_version": 1, "generated_at": v1.utc_now(), "ready": ready,
        "experiment_id": c.experiment_id, "video_shape": list(video.shape),
        "label_rows": len(labels), "unique_rois": len({r["roi_identity"] for r in labels}),
        "screen_combinations": len(screen_matrix(c.learning_rates)),
        "screen_outer_folds": 4, "planned_screen_fits": 36,
        "conditional_confirmation_fits": 12, "maximum_learned_fits": 48,
        "frozen_baseline_evaluations": 4,
        "inputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": v1.sha256(p)} for p in files],
        "resources": {"cpu_threads": c.cpu_threads, "ram_available_mib": v1.available_ram_mib(),
                      "ram_cap_mib": c.max_ram_mib, "gpu_free_mib": free // 2**20,
                      "gpu_total_mib": total // 2**20, "gpu_cap_mib": c.max_gpu_memory_mib,
                      "disk_free_mib": disk_free, "frame_batch": c.frame_batch},
        "leakage_contract": "Training uses three bursts; the held-out burst is evaluation-only. Negatives come only from quiet frames.",
        "initialization_contract": "Temporal/amplitude and spatial variants exactly reproduce raw-direct scoring at zero drift; guarded auxiliaries start at low weight.",
    }
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        overlay = artifact_dir / "label_projection_overlay.png"
        v1._write_overlay(video, labels, overlay)
        payload["label_projection_overlay"] = str(overlay.resolve())
        v1.atomic_json(artifact_dir / "preflight.json", payload)
    if not ready:
        raise RuntimeError("Direct-tuning preflight failed")
    return payload


def _model_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LearnableDirect(nn.Module):
        def __init__(self, variant: str, support: int, spatial_support: int,
                     auxiliary_init_weight: float, jitter_std: float = 0.0):
            super().__init__()
            if variant not in VARIANTS:
                raise ValueError(variant)
            self.variant = variant
            self.support = support
            self.spatial_support = spatial_support
            self.raw_gain = nn.Parameter(torch.zeros(()))
            self.raw_gamma = nn.Parameter(torch.zeros(()))
            self.raw_tau = nn.Parameter(torch.zeros(()))
            if variant in ("spatial", "guarded_auxiliary"):
                self.raw_spatial = nn.Parameter(torch.zeros(spatial_support, spatial_support))
            if variant == "guarded_auxiliary":
                r = support // 2
                yy, xx = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
                radius = (xx.float().square() + yy.float().square()).sqrt()
                target = torch.exp(-0.5 * (radius / 2.0).square())
                ring = ((radius >= 4) & (radius <= 10)).float() + 0.02
                self.register_buffer("target0", target / target.sum())
                self.register_buffer("ring0", ring / ring.sum())
                self.raw_target_delta = nn.Parameter(torch.zeros_like(target))
                self.raw_ring_delta = nn.Parameter(torch.zeros_like(ring))
                maximum = 0.05
                raw_beta = math.log(auxiliary_init_weight / (maximum - auxiliary_init_weight))
                self.raw_beta_contrast = nn.Parameter(torch.tensor(raw_beta))
                self.raw_beta_coherence = nn.Parameter(torch.tensor(raw_beta))
            if jitter_std:
                with torch.no_grad():
                    for parameter in self.parameters():
                        parameter.add_(torch.randn_like(parameter) * jitter_std)
            self._initial = {name: parameter.detach().clone() for name, parameter in self.named_parameters()}

        def amplitude_parameters(self):
            gain = torch.exp(0.25 * torch.tanh(self.raw_gain))
            gamma = torch.exp(0.25 * torch.tanh(self.raw_gamma))
            tau = 0.25 * torch.exp(0.5 * torch.tanh(self.raw_tau))
            return gain, gamma, tau

        def spatial_kernel(self):
            r = torch.tanh(self.raw_spatial)
            r = r - r.mean()
            kernel = 0.01 * r
            c = self.spatial_support // 2
            kernel = kernel.clone()
            kernel[c, c] = kernel[c, c] + 1.0
            return kernel

        def guarded_kernels(self):
            target = self.target0 * torch.exp(0.10 * torch.tanh(self.raw_target_delta))
            ring = self.ring0 * torch.exp(0.10 * torch.tanh(self.raw_ring_delta))
            return target / target.sum(), ring / ring.sum()

        def auxiliary_weights(self):
            return 0.05 * torch.sigmoid(self.raw_beta_contrast), 0.05 * torch.sigmoid(self.raw_beta_coherence)

        def frame_scores(self, x):
            gain, gamma, _ = self.amplitude_parameters()
            amplitude = gain * torch.expm1(gamma * torch.log1p(x.clamp_min(0)))
            b, t, _, h, w = amplitude.shape
            flat = amplitude.reshape(b * t, 1, h, w)
            if self.variant in ("spatial", "guarded_auxiliary"):
                k = self.spatial_kernel()
                p = self.spatial_support // 2
                flat = F.conv2d(F.pad(flat, (p, p, p, p), mode="reflect"), k[None, None]).clamp_min(0)
            amplitude = flat.reshape(b, t, h, w)
            score = amplitude
            if self.variant == "guarded_auxiliary":
                kt, kr = self.guarded_kernels()
                p = self.support // 2
                padded = F.pad(flat, (p, p, p, p), mode="reflect")
                target = F.conv2d(padded, kt[None, None])
                mean = F.conv2d(padded, kr[None, None])
                variance = (F.conv2d(padded.square(), kr[None, None]) - mean.square()).clamp_min(0)
                correction = (1 - kr.square().sum()).clamp_min(1e-4)
                noise = ((kt - kr).square().sum()).clamp_min(1e-6)
                contrast = F.relu((target - mean) / torch.sqrt(noise * variance / correction + 1e-6)).square()
                contrast = torch.log1p(contrast).reshape(b, t, h, w)
                previous = torch.cat((amplitude[:, :1], amplitude[:, :-1]), dim=1)
                following = torch.cat((amplitude[:, 1:], amplitude[:, -1:]), dim=1)
                coherence = torch.sqrt((amplitude * ((previous + amplitude + following) / 3)).clamp_min(0) + 1e-8)
                beta_c, beta_h = self.auxiliary_weights()
                score = amplitude + beta_c * contrast + beta_h * coherence
            return score

        def pool(self, x, mask=None, tolerance: int | None = None):
            score = self.frame_scores(x)
            _, _, tau = self.amplitude_parameters()
            if mask is None:
                count = score.shape[1]
                return tau * (torch.logsumexp(score / tau, dim=1) - math.log(count))
            b, t, h, w = score.shape
            yy, xx = torch.meshgrid(torch.arange(h, device=score.device), torch.arange(w, device=score.device), indexing="ij")
            disk = (yy - h // 2).square() + (xx - w // 2).square() <= int(tolerance) ** 2
            valid = mask[:, :, None, None] & disk[None, None]
            count = valid.sum((1, 2, 3)).clamp_min(1)
            z = (score / tau).masked_fill(~valid, -torch.inf)
            return tau * (torch.logsumexp(z.reshape(b, -1), dim=1) - torch.log(count.float()))

        def trust_penalty(self):
            terms = [(parameter - self._initial[name]).square().mean() for name, parameter in self.named_parameters()]
            return torch.stack(terms).sum()

        def diagnostics(self) -> dict[str, float]:
            gain, gamma, tau = self.amplitude_parameters()
            result = {"gain": float(gain.detach()), "gamma": float(gamma.detach()),
                      "temperature": float(tau.detach()),
                      "trust_distance": float(self.trust_penalty().detach())}
            if self.variant in ("spatial", "guarded_auxiliary"):
                kernel = self.spatial_kernel().detach()
                identity = torch.zeros_like(kernel)
                identity[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1
                result["spatial_l1_drift"] = float((kernel - identity).abs().sum())
            if self.variant == "guarded_auxiliary":
                bc, bh = self.auxiliary_weights()
                kt, kr = self.guarded_kernels()
                result.update(beta_contrast=float(bc.detach()), beta_coherence=float(bh.detach()),
                              target_l1_drift=float((kt - self.target0).abs().sum().detach()),
                              ring_l1_drift=float((kr - self.ring0).abs().sum().detach()))
            return result

    return LearnableDirect


def _parameter_groups(model, learning_rate: float):
    amplitude = [model.raw_gain, model.raw_gamma, model.raw_tau]
    groups = [{"params": amplitude, "lr": learning_rate, "name": "temporal_amplitude"}]
    if model.variant in ("spatial", "guarded_auxiliary"):
        spatial = [model.raw_spatial]
        if model.variant == "guarded_auxiliary":
            spatial += [model.raw_target_delta, model.raw_ring_delta]
        groups.append({"params": spatial, "lr": learning_rate * 0.3, "name": "spatial_kernels"})
    if model.variant == "guarded_auxiliary":
        groups.append({"params": [model.raw_beta_contrast, model.raw_beta_coherence],
                       "lr": learning_rate, "name": "auxiliary_weights"})
    return groups


def _fit(c: DirectTuningConfig, bags, indices: list[int], variant: str,
         learning_rate: float, seed: int, jitter_std: float = 0.0):
    import torch
    import torch.nn.functional as F

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    Model = _model_class()
    model = Model(variant, c.support_px, c.spatial_support_px,
                  c.auxiliary_init_weight, jitter_std=jitter_std).cuda()
    model._initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    positive, negative, mask = [torch.from_numpy(x).cuda() for x in bags]
    ix = torch.tensor(indices, device="cuda")
    optimizer = torch.optim.AdamW(_parameter_groups(model, learning_rate), weight_decay=0)
    with torch.no_grad():
        initial_delta = model.pool(positive[ix], mask[ix], c.tolerance_px) - model.pool(negative[ix], mask[ix], c.tolerance_px)
        score_scale = float(initial_delta.std().clamp_min(0.05).item())
    best = (float("inf"), 0, None)
    stale = 0
    history = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for epoch in range(1, c.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        ps = model.pool(positive[ix], mask[ix], c.tolerance_px)
        qs = model.pool(negative[ix], mask[ix], c.tolerance_px)
        rank = F.softplus(1.0 - (ps - qs) / score_scale).mean()
        trust = model.trust_penalty()
        loss = rank + c.trust_region_weight * trust
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == c.epochs:
            value = float(loss.detach())
            history.append({"epoch": epoch, "rank_loss": float(rank.detach()),
                            "trust_penalty": float(trust.detach()), "total_loss": value,
                            "positive_score": float(ps.mean().detach()), "quiet_score": float(qs.mean().detach())})
            if value < best[0] - 1e-6:
                best = (value, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                stale = 0
            else:
                stale += 10
            if epoch >= c.min_epochs and stale >= c.patience:
                break
    model.load_state_dict(best[2])
    resource = {"elapsed_seconds": time.monotonic() - started,
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "rss_mib": v1.rss_mib()}
    del positive, negative, mask
    torch.cuda.empty_cache()
    return model, history, {"best_epoch": best[1], "score_scale": score_scale,
                            "parameters": model.diagnostics(), "resources": resource}


def _score_map(model, frames: np.ndarray, frame_batch: int) -> tuple[np.ndarray, int]:
    import torch

    if len(frames) > frame_batch:
        raise ValueError("frame_batch must cover a full pseudo-burst for temporal coherence")
    model.eval()
    try:
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frames[None, :, None])).cuda()
            result = model.pool(x).squeeze(0).float().cpu().numpy()
        return result, frame_batch
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError("Temporal sequence cannot be split without changing the score") from exc


def _direct_map(frames: np.ndarray) -> np.ndarray:
    from scipy.special import logsumexp
    return (0.25 * (logsumexp(frames / 0.25, axis=0) - math.log(len(frames)))).astype(np.float32)


def _quiet_threshold(score_function, quiet: np.ndarray, c: DirectTuningConfig):
    peaks = []
    for duration, start in zip((24, 24, 28, 47), (0, 24, 48, 53)):
        score, _ = score_function(quiet[start:start + duration])
        peaks.extend(v1._peaks(score, c.nms_distance_px, limit=2000))
    ranked = sorted((x[0] for x in peaks), reverse=True)
    return float(np.nextafter(ranked[4], np.inf))


def _evaluate_model(model, quiet, bursts, labels, heldout, c):
    threshold = _quiet_threshold(lambda x: _score_map(model, x, c.frame_batch), quiet, c)
    score, resolved = _score_map(model, bursts[heldout], c.frame_batch)
    peaks = v1._peaks(score, c.nms_distance_px, threshold, limit=500)
    rows = [r for r in labels if r["burst_id"] == heldout]
    matches = v1._match(peaks, rows, c.nms_distance_px)
    return {"heldout_burst": heldout, "recall": len(matches) / len(rows),
            "matched": len(matches), "labels": len(rows), "event_peaks": len(peaks),
            "threshold": threshold, "resolved_batch": resolved}


def _direct_baseline(quiet, bursts, labels, c):
    threshold = _quiet_threshold(lambda x: (_direct_map(x), c.frame_batch), quiet, c)
    folds = []
    for heldout in range(1, 5):
        score = _direct_map(bursts[heldout])
        peaks = v1._peaks(score, c.nms_distance_px, threshold, limit=500)
        rows = [r for r in labels if r["burst_id"] == heldout]
        matches = v1._match(peaks, rows, c.nms_distance_px)
        folds.append({"heldout_burst": heldout, "recall": len(matches) / len(rows),
                      "matched": len(matches), "labels": len(rows),
                      "event_peaks": len(peaks), "threshold": threshold})
    return {"name": "frozen_raw_direct", "mean_recall": float(np.mean([x["recall"] for x in folds])),
            "outer_folds": folds}


def _summaries(results, matrix, baseline):
    summaries = []
    for combo in matrix:
        rows = [r for r in results if r["combination_id"] == combo["combination_id"] and r["stage"] == "screen"]
        fold = {b: float(np.mean([r["recall"] for r in rows if r["heldout_burst"] == b])) for b in range(1, 5)}
        wins = sum(fold[b] > baseline["outer_folds"][b - 1]["recall"] for b in range(1, 5))
        summaries.append({**combo, "mean_recall": float(np.mean(list(fold.values()))),
                          "fold_mean_recall": fold, "fold_wins_vs_direct": wins,
                          "direct_baseline_mean_recall": baseline["mean_recall"], "fit_count": len(rows)})
    return summaries


def run(c: DirectTuningConfig) -> dict[str, Any]:
    import torch

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(c.cpu_threads)
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(min(0.95, c.max_gpu_memory_mib /
        (torch.cuda.get_device_properties(0).total_memory / 2**20)))
    ready = preflight(c)
    c.output_dir.mkdir(parents=True, exist_ok=False)
    labels = v1.load_labels(c.labels_tsv)
    video = np.load(c.source_video, mmap_mode="r", allow_pickle=False)
    overlay = c.output_dir / "label_projection_overlay.png"
    v1._write_overlay(video, labels, overlay)
    ready["label_projection_overlay"] = str(overlay.resolve())
    v1.atomic_json(c.output_dir / "preflight.json", ready)
    v1.atomic_json(c.output_dir / "resolved_config.json", {
        k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
        for k, v in c.__dict__.items()})
    state = {"status": "running", "started_at": v1.utc_now(), "pid": os.getpid()}
    v1.atomic_json(c.output_dir / "run_state.json", state)
    progress_path = c.output_dir / "progress.jsonl"
    def log(stage, **payload):
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": v1.utc_now(), "stage": stage, **payload}) + "\n")

    quiet, bursts, _ = v1._prepare_arrays(_legacy_config(c), labels)
    log("preprocessing_complete", mode="raw quiet-median positive residual", rss_mib=v1.rss_mib())
    baseline = _direct_baseline(quiet, bursts, labels, c)
    radius = max(c.support_px // 2, c.spatial_support_px // 2) + c.tolerance_px
    bags = v1._bag_tensors(quiet, bursts, labels, radius)
    matrix = screen_matrix(c.learning_rates)
    results = []
    for combo in matrix:
        for heldout in range(1, 5):
            indices = [i for i, row in enumerate(labels) if row["burst_id"] != heldout]
            model, history, training = _fit(c, bags, indices, combo["variant"],
                                            combo["learning_rate"], c.screen_seed + heldout * 100)
            evaluation = _evaluate_model(model, quiet, bursts, labels, heldout, c)
            row = {**combo, **evaluation, "stage": "screen", "seed": c.screen_seed,
                   "history": history, "training": training}
            results.append(row)
            log("fit_complete", **{k: v for k, v in row.items() if k not in ("stage", "history", "training")},
                best_epoch=training["best_epoch"], parameters=training["parameters"], resources=training["resources"])
            del model; torch.cuda.empty_cache()
    summaries = _summaries(results, matrix, baseline)
    winner = max(summaries, key=lambda row: (row["mean_recall"], row["fold_wins_vs_direct"], -row["learning_rate"]))
    tied = [row["combination_id"] for row in summaries
            if row["mean_recall"] == winner["mean_recall"]
            and row["fold_wins_vs_direct"] == winner["fold_wins_vs_direct"]]
    selection = {"status": "tie" if len(tied) > 1 else "unique", "tie_count": len(tied), "combination_ids": tied}
    screen_pass = winner["mean_recall"] > baseline["mean_recall"] and winner["fold_wins_vs_direct"] >= 3
    confirmation = []
    if screen_pass:
        for heldout in range(1, 5):
            indices = [i for i, row in enumerate(labels) if row["burst_id"] != heldout]
            for seed in c.confirmation_seeds:
                model, history, training = _fit(c, bags, indices, winner["variant"], winner["learning_rate"],
                                                seed + heldout * 100, c.confirmation_jitter_std)
                evaluation = _evaluate_model(model, quiet, bursts, labels, heldout, c)
                row = {"combination_id": winner["combination_id"], "variant": winner["variant"],
                       "learning_rate": winner["learning_rate"], **evaluation, "stage": "confirmation",
                       "seed": seed, "history": history, "training": training}
                results.append(row); confirmation.append(row)
                log("confirmation_fit_complete", **{k: v for k, v in row.items() if k not in ("stage", "history", "training")})
                del model; torch.cuda.empty_cache()
    if confirmation:
        fold = {b: float(np.mean([r["recall"] for r in confirmation if r["heldout_burst"] == b])) for b in range(1, 5)}
        confirmation_summary = {"mean_recall": float(np.mean(list(fold.values()))), "fold_mean_recall": fold,
                                "fold_wins_vs_direct": sum(fold[b] > baseline["outer_folds"][b - 1]["recall"] for b in range(1, 5)),
                                "fit_count": len(confirmation)}
        advance = confirmation_summary["mean_recall"] > baseline["mean_recall"] and confirmation_summary["fold_wins_vs_direct"] >= 3
    else:
        confirmation_summary = {"status": "not_run", "reason": "screen gate did not pass", "fit_count": 0}
        advance = False
    gate = {"decision": "advance_to_masked_validation" if advance else "do_not_advance",
            "screen_pass": screen_pass,
            "reason": "requires recall above frozen raw-direct and wins in at least 3/4 held-out bursts; confirmation is required after a screen pass"}
    resources = {"maximum_peak_gpu_allocated_mib": max(r["training"]["resources"]["peak_gpu_allocated_mib"] for r in results),
                 "maximum_peak_gpu_reserved_mib": max(r["training"]["resources"]["peak_gpu_reserved_mib"] for r in results),
                 "maximum_rss_mib": max(r["training"]["resources"]["rss_mib"] for r in results),
                 "learned_fit_count": len(results)}
    metrics = {"schema_version": 1, "completed_at": v1.utc_now(), "frozen_direct_baseline": baseline,
               "screen_matrix": matrix, "combination_summaries": summaries, "winner": winner, "selection": selection,
               "screen_fit_count": sum(r["stage"] == "screen" for r in results),
               "confirmation_summary": confirmation_summary, "learned_fit_count": len(results),
               "fit_results": results, "gate": gate,
               "masked_stage": {"status": "not_run", "reason": "requires confirmed outer-fold gate"},
               "final_stage": {"status": "not_run", "reason": "requires outer-fold and masked gates"},
               "resource_observations": resources}
    v1.atomic_json(c.output_dir / "metrics.json", metrics)
    v1.atomic_json(c.output_dir / "resource_observations.json", resources)
    _write_tables(c.output_dir, results, summaries)
    _write_report(c.output_dir / "report.md", c, metrics)
    summary = {"schema_version": 1, "experiment_id": c.experiment_id, "status": "completed",
               "frozen_direct_mean_recall": baseline["mean_recall"], "winner": winner, "selection": selection,
               "gate": gate, "learned_fit_count": len(results), "report": "report.md", "metrics": "metrics.json"}
    v1.atomic_json(c.output_dir / "experiment_summary.json", summary)
    state = {"status": "completed", "completed_at": v1.utc_now(), "pid": os.getpid(),
             "gate": gate["decision"], "learned_fit_count": len(results)}
    v1.atomic_json(c.output_dir / "run_state.json", state); log("completed", **state)
    return summary


def _write_tables(out: Path, results: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    with (out / "combination_summary.tsv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["combination_id", "variant", "learning_rate", "mean_recall", "fold_wins_vs_direct",
                  "direct_baseline_mean_recall", "fit_count"]
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    with (out / "fit_results.tsv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["stage", "combination_id", "variant", "learning_rate", "heldout_burst", "seed",
                  "recall", "matched", "labels", "event_peaks", "threshold", "resolved_batch"]
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)


def _write_report(path: Path, c: DirectTuningConfig, metrics: dict[str, Any]) -> None:
    winner = metrics["winner"]
    lines = [f"# {c.experiment_id}", "", f"Status: completed. Gate: `{metrics['gate']['decision']}`.", "",
             "## Design", "", "The frozen raw-direct detector is the reference. Cumulative variants tune temporal/amplitude calibration, then spatial filtering, then low-weight guarded contrast and temporal coherence. NMS and quiet calibration remain fixed.", "",
             f"The screen evaluated 9 combinations across four held-out bursts ({metrics['screen_fit_count']} learned fits). Confirmation was conditional on passing the screen gate.", "",
             "## Result", "", f"- Frozen raw-direct mean recall: `{metrics['frozen_direct_baseline']['mean_recall']:.4f}`",
             f"- Best combination: `{winner['combination_id']}`", f"- Best mean recall: `{winner['mean_recall']:.4f}`",
             f"- Fold wins versus direct: `{winner['fold_wins_vs_direct']}/4`", f"- Confirmation: `{metrics['confirmation_summary'].get('status', 'completed')}`", "",
             "Detailed results are in `combination_summary.tsv` and `fit_results.tsv`; full histories and bounded parameter drift are in `metrics.json`.", "",
             "Unlabeled event pixels were never used as negatives. Masked and final stages remain gated."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
