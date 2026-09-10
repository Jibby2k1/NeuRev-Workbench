"""Standalone framewise exact-truth benchmark for the Gamma-LS paper.

The benchmark has a mandatory two-step lifecycle: freeze the source-controlled
plan/config, then execute that immutable plan.  Threshold and scale-floor
calibration use source-free movies from disjoint development seeds.  Evaluation
uses untouched seeds and includes three mechanism-family holdouts.  No temporal
pooling or source-count-derived proposal budget is available in this module.

All precision-like claims are confined to the exhaustive synthetic truth.  The
simulator is a mechanistic stress fixture, not biological validation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from neurobench.algorithms.gamma_local_standardization import (
    GammaReferenceSpec,
    gamma_local_standardization,
)
from neurobench.experiments.gamma_ls_difference.gpu_representations import (
    causal_preprocess_common_input,
    energy_normalized_difference_representation,
    raw_representation,
    signed_difference_representation,
)
from neurobench.metrics.sparse_detection import extract_separated_local_maxima

from .support_sufficiency import (
    _maintained_positive_box_score,
    _square_annulus_moments,
)


FAMILIES = (
    "baseline",
    "dense_neuropil",
    "bleaching",
    "source_specific_nonrigid_motion",
    "empirical_noise",
    "compound_shift",
)
DEVELOPMENT_FAMILIES = FAMILIES[:3]
FAMILY_HOLDOUTS = FAMILIES[3:]
REPRESENTATIONS = (
    "raw",
    "difference_signed",
    "difference_energy_normalized",
)
OPERATORS = (
    "radial_gamma_h15_g7_n9_m0p5",
    "signed_square_annulus_h11_g3",
    "maintained_positive_box_cfar_h11_g3",
)
BURDENS = (0.25, 0.5, 1.0, 2.0, 5.0)
SOURCE_COUNTS = (2, 4, 8)
MATCHING_RULE = "score_ranked_greedy_nearest_remaining_truth"
PROTOCOL_VERSION = 1


class ExactTruthBenchmarkError(ValueError):
    """Raised when a frozen exact-truth contract is missing or ambiguous."""


def _strict_keys(mapping: Mapping[str, Any], expected: set[str], scope: str) -> None:
    actual = set(mapping)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ExactTruthBenchmarkError(
            f"invalid {scope} fields: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    if array.nbytes:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ExactTruthBenchmarkError(f"cannot write empty table {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ExactTruthBenchmarkError(f"inconsistent fields in {path.name}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class AtomicTsvSink:
    """Bounded-memory TSV writer that publishes only a fully flushed table."""

    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        self.path = path
        self.fields = tuple(fields)
        self.temporary = path.with_name(path.name + ".partial")
        self.stream: Any | None = None
        self.writer: csv.DictWriter | None = None
        self.row_count = 0

    def __enter__(self) -> "AtomicTsvSink":
        self.stream = self.temporary.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.stream, fieldnames=list(self.fields), delimiter="\t"
        )
        self.writer.writeheader()
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        if self.writer is None or list(row) != list(self.fields):
            raise ExactTruthBenchmarkError(
                f"row schema changed while writing {self.path.name}"
            )
        self.writer.writerow(row)
        self.row_count += 1

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> None:
        if self.stream is None:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        if error_type is None:
            self.temporary.replace(self.path)


def _artifact_index(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.name != "artifact_index.json"
            and not path.name.endswith(".partial")
        ):
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"schema_version": 1, "artifacts": artifacts}


def _verify_artifact_index(root: Path) -> dict[str, Any]:
    index_path = root / "artifact_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("artifacts"), list):
        raise ExactTruthBenchmarkError("frozen-plan artifact index is invalid")
    seen: set[str] = set()
    for row in payload["artifacts"]:
        relative = str(row.get("path", ""))
        if not relative or relative in seen:
            raise ExactTruthBenchmarkError("frozen-plan artifact index has duplicate paths")
        seen.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ExactTruthBenchmarkError("frozen-plan artifact escapes its root") from error
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or _sha256(path) != row["sha256"]
        ):
            raise ExactTruthBenchmarkError(f"frozen-plan artifact changed: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_index.json"
    }
    if actual != seen:
        raise ExactTruthBenchmarkError("frozen-plan artifact inventory changed")
    return {
        "artifact_index_sha256": _sha256(index_path),
        "artifact_count": len(seen),
    }


@dataclass(frozen=True)
class ExactTruthConfig:
    """Strict source-controlled benchmark configuration."""

    path: Path
    repository: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ExactTruthConfig":
        resolved = Path(path).expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ExactTruthBenchmarkError("config must be a JSON object")
        result = cls(
            path=resolved,
            repository=Path(__file__).resolve().parents[3],
            payload=payload,
        )
        result.validate()
        return result

    def validate(self) -> None:
        payload = self.payload
        _strict_keys(
            payload,
            {
                "schema_version",
                "experiment_id",
                "simulator",
                "splits",
                "representations",
                "operators",
                "calibration",
                "evaluation",
                "empirical_noise_profile",
                "resources",
                "scientific_audit",
                "outputs",
            },
            "top-level",
        )
        if payload["schema_version"] != 1 or payload["experiment_id"] != "gamma_ls_exact_truth_framewise_v1":
            raise ExactTruthBenchmarkError("unexpected schema version or experiment id")
        simulator = payload["simulator"]
        _strict_keys(
            simulator,
            {
                "frames",
                "height_px",
                "width_px",
                "frame_interval_ms",
                "source_counts",
                "active_trace_threshold",
                "source_margin_px",
                "minimum_anchor_separation_px",
                "families",
            },
            "simulator",
        )
        if (
            int(simulator["frames"]) != 96
            or int(simulator["height_px"]) != 64
            or int(simulator["width_px"]) != 64
            or float(simulator["frame_interval_ms"]) != 20.0
            or tuple(simulator["source_counts"]) != SOURCE_COUNTS
            or tuple(simulator["families"]) != FAMILIES
            or float(simulator["active_trace_threshold"]) != 0.12
            or float(simulator["source_margin_px"]) != 10.0
            or float(simulator["minimum_anchor_separation_px"]) != 8.0
        ):
            raise ExactTruthBenchmarkError("simulator design changed")

        splits = payload["splits"]
        _strict_keys(
            splits,
            {"scale_floor_fit", "threshold_calibration", "evaluation", "latency"},
            "splits",
        )
        for name in ("scale_floor_fit", "threshold_calibration"):
            row = splits[name]
            _strict_keys(row, {"families", "seeds", "source_count"}, f"splits.{name}")
            if tuple(row["families"]) != DEVELOPMENT_FAMILIES or int(row["source_count"]) != 0:
                raise ExactTruthBenchmarkError(
                    f"{name} must use zero-source development movies only"
                )
        evaluation = splits["evaluation"]
        _strict_keys(evaluation, {"families", "seeds", "source_counts"}, "splits.evaluation")
        if tuple(evaluation["families"]) != FAMILIES or tuple(evaluation["source_counts"]) != SOURCE_COUNTS:
            raise ExactTruthBenchmarkError("evaluation family/source-count design changed")
        latency = splits["latency"]
        _strict_keys(latency, {"family", "seed", "source_count"}, "splits.latency")
        if latency != {"family": "baseline", "seed": 9001, "source_count": 0}:
            raise ExactTruthBenchmarkError("latency fixture changed")
        seed_sets = {
            "scale_floor_fit": set(map(int, splits["scale_floor_fit"]["seeds"])),
            "threshold_calibration": set(
                map(int, splits["threshold_calibration"]["seeds"])
            ),
            "evaluation": set(map(int, evaluation["seeds"])),
            "latency": {int(latency["seed"])},
        }
        if any(not values for values in seed_sets.values()):
            raise ExactTruthBenchmarkError("every split requires at least one seed")
        if any(
            len(values) != len(splits[name]["seeds"])
            for name, values in seed_sets.items()
            if name != "latency"
        ):
            raise ExactTruthBenchmarkError("seeds must be unique within every split")
        for first, first_values in seed_sets.items():
            for second, second_values in seed_sets.items():
                if first < second and first_values & second_values:
                    raise ExactTruthBenchmarkError(
                        f"split seeds overlap: {first} and {second}"
                    )

        representations = payload["representations"]
        _strict_keys(
            representations,
            {
                "ordered",
                "common_preprocessing",
                "aligned_current_frame_starts_at_zero_based_index",
                "energy_epsilon",
            },
            "representations",
        )
        if (
            tuple(representations["ordered"]) != REPRESENTATIONS
            or representations["common_preprocessing"]
            != "causal_gaussian_sigma1_then_ema_alpha0p4"
            or int(representations["aligned_current_frame_starts_at_zero_based_index"])
            != 1
            or float(representations["energy_epsilon"]) != 1e-8
        ):
            raise ExactTruthBenchmarkError("representation contract changed")

        operators = payload["operators"]
        _strict_keys(
            operators,
            {
                "ordered",
                "radial_gamma",
                "signed_square_annulus",
                "maintained_positive_box_cfar",
                "epsilon",
                "scale_floor_percentile",
            },
            "operators",
        )
        if tuple(operators["ordered"]) != OPERATORS:
            raise ExactTruthBenchmarkError("operator order changed")
        if operators["radial_gamma"] != {
            "half_width_px": 15,
            "guard_radius_px": 7,
            "shape_n": 9.0,
            "mode_fraction_of_half_width": 0.5,
            "support_geometry": "disk",
            "boundary_mode": "valid_renormalized_zero",
        }:
            raise ExactTruthBenchmarkError("frozen radial deployment context changed")
        if operators["signed_square_annulus"] != {
            "outer_half_width_px": 11,
            "guard_half_width_px": 3,
            "boundary_mode": "valid_reference_renormalized_zero",
        }:
            raise ExactTruthBenchmarkError("signed square-annulus control changed")
        if operators["maintained_positive_box_cfar"] != {
            "outer_half_width_px": 11,
            "guard_half_width_px": 3,
            "boundary_mode": "replicate",
            "input_clipping": "positive",
        }:
            raise ExactTruthBenchmarkError("maintained positive-box control changed")
        if float(operators["epsilon"]) != 1e-6 or float(
            operators["scale_floor_percentile"]
        ) != 10.0:
            raise ExactTruthBenchmarkError("operator numerical contract changed")

        calibration = payload["calibration"]
        _strict_keys(
            calibration,
            {
                "target_nms_proposals_per_frame",
                "primary_target_nms_proposals_per_frame",
                "threshold_rule",
                "source_count_used",
                "source_count_derived_threshold_or_budget",
                "probability_of_false_alarm_claimed",
            },
            "calibration",
        )
        if (
            tuple(float(value) for value in calibration["target_nms_proposals_per_frame"])
            != BURDENS
            or float(calibration["primary_target_nms_proposals_per_frame"]) != 1.0
            or calibration["threshold_rule"]
            != "largest_empirical_null_nms_burden_not_exceeding_target"
            or int(calibration["source_count_used"]) != 0
            or calibration["source_count_derived_threshold_or_budget"] is not False
            or calibration["probability_of_false_alarm_claimed"] is not False
        ):
            raise ExactTruthBenchmarkError("calibration contract changed")

        evaluation_contract = payload["evaluation"]
        _strict_keys(
            evaluation_contract,
            {
                "grain",
                "temporal_aggregation",
                "temporal_max_pooling",
                "time_collapsed_site_score",
                "nms_distance_px",
                "match_radius_px",
                "one_to_one_matching",
                "candidate_budget",
                "safety_peak_limit_per_frame",
                "metrics",
            },
            "evaluation",
        )
        if (
            evaluation_contract["grain"] != "source_frame"
            or evaluation_contract["temporal_aggregation"] != "none_framewise"
            or evaluation_contract["temporal_max_pooling"] is not False
            or evaluation_contract["time_collapsed_site_score"] is not False
            or int(evaluation_contract["nms_distance_px"]) != 6
            or float(evaluation_contract["match_radius_px"]) != 6.0
            or evaluation_contract["one_to_one_matching"] != MATCHING_RULE
            or evaluation_contract["candidate_budget"] != "none_threshold_only"
            or int(evaluation_contract["safety_peak_limit_per_frame"]) != 10_000
            or tuple(evaluation_contract["metrics"])
            != ("precision", "recall", "f1", "localization_px", "duplicates")
        ):
            raise ExactTruthBenchmarkError("framewise evaluation contract changed")

        profile = payload["empirical_noise_profile"]
        _strict_keys(
            profile,
            {
                "recording_id",
                "source_sha256",
                "source_shape_tyx",
                "sampled_frame_count",
                "sample_crop_yxhw",
                "median_intensity",
                "frame_difference_noise_mad",
                "robust_noise_to_median",
                "simulator_read_sigma",
                "label_use",
                "role",
            },
            "empirical_noise_profile",
        )
        if (
            profile["recording_id"] != "15 right crop512"
            or profile["source_sha256"]
            != "ede3f57dd65d5fb2a13c389da3ea0ef4cd1f9fdd83c19f4b3b9ffa7009a13603"
            or profile["source_shape_tyx"] != [1608, 512, 512]
            or int(profile["sampled_frame_count"]) != 48
            or profile["sample_crop_yxhw"] != [128, 128, 256, 256]
            or float(profile["median_intensity"]) != 716.0
            or float(profile["frame_difference_noise_mad"])
            != 57.659607740394165
            or float(profile["robust_noise_to_median"])
            != 0.08053017840837173
            or float(profile["simulator_read_sigma"]) != 0.40265089204185867
            or profile["label_use"] != "none"
            or profile["role"]
            != "frozen_label_free_noise_scale_only_not_biological_truth"
        ):
            raise ExactTruthBenchmarkError("empirical-noise profile changed")

        resources = payload["resources"]
        _strict_keys(
            resources,
            {
                "device",
                "cpu_threads",
                "chunk_frames",
                "latency_warmup_iterations",
                "latency_timed_iterations",
                "minimum_free_disk_gib",
            },
            "resources",
        )
        if (
            resources["device"] != "cpu"
            or int(resources["cpu_threads"]) != 1
            or int(resources["chunk_frames"]) != 32
            or int(resources["latency_warmup_iterations"]) != 10
            or int(resources["latency_timed_iterations"]) != 50
            or float(resources["minimum_free_disk_gib"]) != 2.0
        ):
            raise ExactTruthBenchmarkError("resource contract changed")
        audit = payload["scientific_audit"]
        if audit != {
            "enabled": True,
            "status_after_metric_runner": "pending_full_media_renderer_and_inventory_validation",
        }:
            raise ExactTruthBenchmarkError("scientific-audit boundary changed")
        _strict_keys(payload["outputs"], {"plan_root", "result_root"}, "outputs")

    def portable(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload))


def pipeline_ids() -> tuple[str, ...]:
    return tuple(
        f"{representation}__{operator}"
        for representation in REPRESENTATIONS
        for operator in OPERATORS
    )


@dataclass(frozen=True)
class MovieSpec:
    split: str
    family: str
    seed: int
    source_count: int
    family_holdout: bool

    @property
    def movie_id(self) -> str:
        return (
            f"etv1__{self.split}__{self.family}__seed{self.seed:05d}"
            f"__sources{self.source_count:02d}"
        )

    @property
    def rng_seed(self) -> int:
        digest = hashlib.sha256(self.movie_id.encode("ascii")).digest()
        return int.from_bytes(digest[:8], "little", signed=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "split": self.split,
            "family": self.family,
            "seed": self.seed,
            "source_count": self.source_count,
            "family_holdout": self.family_holdout,
            "rng_seed_uint64": self.rng_seed,
        }


def movie_specs(config: ExactTruthConfig) -> tuple[MovieSpec, ...]:
    splits = config.payload["splits"]
    rows: list[MovieSpec] = []
    for split in ("scale_floor_fit", "threshold_calibration"):
        for family in splits[split]["families"]:
            for seed in splits[split]["seeds"]:
                rows.append(MovieSpec(split, family, int(seed), 0, False))
    for family in splits["evaluation"]["families"]:
        for seed in splits["evaluation"]["seeds"]:
            for count in splits["evaluation"]["source_counts"]:
                rows.append(
                    MovieSpec(
                        "evaluation",
                        family,
                        int(seed),
                        int(count),
                        family in FAMILY_HOLDOUTS,
                    )
                )
    latency = splits["latency"]
    rows.append(
        MovieSpec(
            "latency",
            str(latency["family"]),
            int(latency["seed"]),
            int(latency["source_count"]),
            False,
        )
    )
    identifiers = [row.movie_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ExactTruthBenchmarkError("split manifest has duplicate movie identities")
    expected = 3 * 3 + 3 * 4 + 6 * 6 * 3 + 1
    if len(rows) != expected:
        raise ExactTruthBenchmarkError(f"split manifest has {len(rows)} movies, expected {expected}")
    return tuple(rows)


@dataclass(frozen=True)
class ExactTruthMovie:
    """One simulator movie and exhaustive, frame-aligned source truth."""

    spec: MovieSpec
    values: np.ndarray
    source_ids: tuple[str, ...]
    centers_yx: np.ndarray
    traces: np.ndarray
    events: np.ndarray
    active: np.ndarray
    active_trace_threshold: float

    def __post_init__(self) -> None:
        frames, height, width = self.values.shape
        sources = len(self.source_ids)
        if self.values.dtype != np.float32 or not np.isfinite(self.values).all():
            raise ExactTruthBenchmarkError("simulated movie must be finite float32 TYX")
        if len(set(self.source_ids)) != sources or sources != self.spec.source_count:
            raise ExactTruthBenchmarkError("simulator source identities are not exact and unique")
        if self.centers_yx.shape != (frames, sources, 2):
            raise ExactTruthBenchmarkError("one y/x center is required per source per frame")
        if self.traces.shape != (frames, sources) or self.events.shape != (
            frames,
            sources,
        ):
            raise ExactTruthBenchmarkError("trace/event truth shape changed")
        if (
            self.centers_yx.dtype != np.float32
            or self.traces.dtype != np.float32
            or self.events.dtype != np.bool_
            or not np.isfinite(self.centers_yx).all()
            or not np.isfinite(self.traces).all()
            or np.any(self.traces < 0)
        ):
            raise ExactTruthBenchmarkError("dynamic center/trace/event truth is invalid")
        if self.active.shape != (frames, sources) or self.active.dtype != np.bool_:
            raise ExactTruthBenchmarkError("active-source truth shape/dtype changed")
        if not np.array_equal(self.active, self.traces >= self.active_trace_threshold):
            raise ExactTruthBenchmarkError("active-source truth disagrees with frozen threshold")
        if sources:
            y = self.centers_yx[:, :, 0]
            x = self.centers_yx[:, :, 1]
            if not (
                np.isfinite(self.centers_yx).all()
                and np.all((y >= 0) & (y < height))
                and np.all((x >= 0) & (x < width))
            ):
                raise ExactTruthBenchmarkError("time-varying source center left the movie")

    @property
    def truth_sha256(self) -> str:
        payload = {
            "movie_id": self.spec.movie_id,
            "source_ids": list(self.source_ids),
            "active_trace_threshold": self.active_trace_threshold,
            "centers_yx_sha256": _array_sha256(self.centers_yx),
            "traces_sha256": _array_sha256(self.traces),
            "events_sha256": _array_sha256(self.events),
            "active_sha256": _array_sha256(self.active),
        }
        return _canonical_sha256(payload)

    def active_labels(self, source_frame_zero_based: int) -> list[dict[str, Any]]:
        frame = int(source_frame_zero_based)
        if not 0 <= frame < len(self.values):
            raise ExactTruthBenchmarkError("truth frame is outside the movie")
        labels = []
        for index, source_id in enumerate(self.source_ids):
            if bool(self.active[frame, index]):
                labels.append(
                    {
                        "source_id": source_id,
                        "x_px": float(self.centers_yx[frame, index, 1]),
                        "y_px": float(self.centers_yx[frame, index, 0]),
                    }
                )
        return labels


def _footprint(
    yy: np.ndarray,
    xx: np.ndarray,
    center_y: float,
    center_x: float,
    morphology: str,
) -> np.ndarray:
    if morphology == "ellipse":
        return np.exp(
            -0.5
            * (
                ((xx - center_x) / 2.4) ** 2
                + ((yy - center_y) / 1.45) ** 2
            )
        )
    broad = np.exp(
        -0.5
        * (((xx - center_x) / 2.5) ** 2 + ((yy - center_y) / 2.1) ** 2)
    )
    cut = np.exp(
        -0.5
        * (
            ((xx - (center_x + 1.25)) / 1.9) ** 2
            + ((yy - (center_y - 0.3)) / 1.6) ** 2
        )
    )
    return np.clip(broad - 0.78 * cut, 0.0, None)


def _anchors(
    rng: np.random.Generator,
    *,
    count: int,
    height: int,
    width: int,
    margin: float,
    minimum_separation: float,
) -> np.ndarray:
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    rows = int(math.ceil(math.sqrt(count)))
    columns = int(math.ceil(count / rows))
    y_grid = np.linspace(margin, height - 1 - margin, rows)
    x_grid = np.linspace(margin, width - 1 - margin, columns)
    anchors = []
    for y in y_grid:
        for x in x_grid:
            if len(anchors) == count:
                break
            candidate = np.asarray(
                [y + rng.uniform(-1.0, 1.0), x + rng.uniform(-1.0, 1.0)]
            )
            if any(np.linalg.norm(candidate - previous) < minimum_separation for previous in anchors):
                raise ExactTruthBenchmarkError("anchor grid violated minimum separation")
            anchors.append(candidate)
    return np.asarray(anchors, dtype=np.float32)


def simulate_exact_truth_movie(
    config: ExactTruthConfig,
    spec: MovieSpec,
) -> ExactTruthMovie:
    """Generate a deterministic six-family movie with exhaustive dynamic truth."""

    planned = {row.movie_id: row for row in movie_specs(config)}.get(spec.movie_id)
    if planned != spec:
        raise ExactTruthBenchmarkError("movie spec left the frozen split/family/source grid")
    simulator = config.payload["simulator"]
    frames = int(simulator["frames"])
    height = int(simulator["height_px"])
    width = int(simulator["width_px"])
    threshold = float(simulator["active_trace_threshold"])
    rng = np.random.default_rng(spec.rng_seed)
    yy, xx = np.mgrid[:height, :width]
    anchors = _anchors(
        rng,
        count=spec.source_count,
        height=height,
        width=width,
        margin=float(simulator["source_margin_px"]),
        minimum_separation=float(simulator["minimum_anchor_separation_px"]),
    )
    source_ids = tuple(
        f"{spec.movie_id}__source{index:03d}" for index in range(spec.source_count)
    )

    event_truth = np.zeros((frames, spec.source_count), dtype=np.bool_)
    traces = np.zeros((frames, spec.source_count), dtype=np.float32)
    kernel_time = np.arange(28, dtype=np.float64)
    kernel = np.exp(-kernel_time / 7.0) * (1.0 - np.exp(-kernel_time / 1.4))
    kernel /= float(kernel.max())
    shared = rng.normal(size=frames)
    for index in range(spec.source_count):
        innovation = 0.2 * shared + math.sqrt(1.0 - 0.2**2) * rng.normal(size=frames)
        events = innovation > 1.45
        if np.count_nonzero(events) < 2:
            available = np.arange(4, frames - 8)
            forced = rng.choice(available, size=2, replace=False)
            events[forced] = True
        event_truth[:, index] = events
        amplitude = rng.uniform(0.75, 1.45)
        traces[:, index] = (
            np.convolve(events.astype(np.float64), kernel, mode="full")[:frames]
            * amplitude
        ).astype(np.float32)
    active = traces >= threshold

    centers = np.empty((frames, spec.source_count, 2), dtype=np.float32)
    motion_family = spec.family in {
        "source_specific_nonrigid_motion",
        "compound_shift",
    }
    for frame in range(frames):
        shared_y = 0.22 * math.sin(2.0 * math.pi * frame / 43.0)
        shared_x = 0.22 * math.cos(2.0 * math.pi * frame / 47.0)
        for index in range(spec.source_count):
            if motion_family:
                dy = (
                    shared_y
                    + 1.55
                    * math.sin(2.0 * math.pi * frame / (29.0 + 2 * index) + 0.7 * index)
                )
                dx = (
                    shared_x
                    + 1.25
                    * math.cos(2.0 * math.pi * frame / (31.0 + 3 * index) + 0.4 * index)
                )
            else:
                dy, dx = shared_y, shared_x
            centers[frame, index] = anchors[index] + (dy, dx)

    dense_family = spec.family in {"dense_neuropil", "compound_shift"}
    bleach_family = spec.family in {"bleaching", "compound_shift"}
    empirical_family = spec.family in {"empirical_noise", "compound_shift"}
    background = 2.0 + gaussian_filter(rng.normal(size=(height, width)), 7.0) * 0.7
    neuropil = np.zeros((frames, height, width), dtype=np.float32)
    if dense_family:
        neuropil_kernel = np.exp(-np.arange(36, dtype=np.float64) / 12.0)
        for _ in range(8):
            center_y = rng.uniform(0, height)
            center_x = rng.uniform(0, width)
            sigma = rng.uniform(5.0, 11.0)
            field = np.exp(
                -((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2.0 * sigma**2)
            )
            events = rng.normal(size=frames) > 1.5
            trace = np.convolve(events.astype(float), neuropil_kernel, mode="full")[:frames]
            neuropil += (0.22 * trace[:, None, None] * field).astype(np.float32)

    video = np.empty((frames, height, width), dtype=np.float32)
    colored_state = np.zeros((height, width), dtype=np.float64)
    read_sigma = (
        float(config.payload["empirical_noise_profile"]["simulator_read_sigma"])
        if empirical_family
        else 0.14
    )
    for frame in range(frames):
        signal = np.zeros((height, width), dtype=np.float64)
        for index, source_id in enumerate(source_ids):
            del source_id  # identity is carried in the exact truth, not pixel generation
            morphology = "crescent" if index % 3 == 0 else "ellipse"
            signal += float(traces[frame, index]) * _footprint(
                yy,
                xx,
                float(centers[frame, index, 0]),
                float(centers[frame, index, 1]),
                morphology,
            )
        bleach = math.exp(-frame / 115.0) if bleach_family else 1.0
        latent = bleach * (background + signal + neuropil[frame])
        shot = rng.normal(
            scale=0.11 * np.sqrt(np.maximum(latent, 0.05)), size=latent.shape
        )
        if empirical_family:
            innovation = gaussian_filter(rng.normal(size=latent.shape), 0.8)
            innovation /= max(float(np.std(innovation)), 1e-12)
            colored_state = 0.68 * colored_state + math.sqrt(1.0 - 0.68**2) * innovation
            read = read_sigma * colored_state
        else:
            read = rng.normal(scale=read_sigma, size=latent.shape)
        video[frame] = (latent + shot + read).astype(np.float32)
    return ExactTruthMovie(
        spec=spec,
        values=video,
        source_ids=source_ids,
        centers_yx=centers,
        traces=traces,
        events=event_truth,
        active=active,
        active_trace_threshold=threshold,
    )


def _radial_spec(*, scale_floor: float) -> GammaReferenceSpec:
    return GammaReferenceSpec.from_mode(
        "exact_truth_gamma_h15_g7_n9_m0p5",
        support_width_px=31,
        shape_n=9.0,
        mode_radius_px=7.5,
        guard_radius_px=7.0,
        support_geometry="disk",
        boundary_mode="valid_renormalized_zero",
        epsilon=1e-6,
        scale_floor=float(scale_floor),
    )


def aligned_representations(
    movie: np.ndarray,
    *,
    energy_epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Apply the frozen common causal input and align all arms to current t>=1."""

    import torch

    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ExactTruthBenchmarkError("movie must be finite float32 TYX with >=2 frames")
    source = torch.from_numpy(np.ascontiguousarray(values))
    with torch.no_grad():
        common = causal_preprocess_common_input(source).values
        result = {
            "raw": raw_representation(common).values[1:],
            "difference_signed": signed_difference_representation(common).values,
            "difference_energy_normalized": energy_normalized_difference_representation(
                common, epsilon=energy_epsilon
            ).values,
        }
    expected = values.shape[0] - 1
    if set(result) != set(REPRESENTATIONS) or any(
        tensor.shape != (expected, *values.shape[1:]) for tensor in result.values()
    ):
        raise ExactTruthBenchmarkError("representation frame alignment changed")
    return result


def _local_std(
    representation: Any,
    *,
    operator: str,
    chunk_frames: int,
) -> Any | None:
    if operator == OPERATORS[0]:
        result = gamma_local_standardization(
            representation,
            _radial_spec(scale_floor=0.0),
            chunk_frames=chunk_frames,
            return_statistics=True,
        )
        if result.local_std is None:
            raise AssertionError("radial local standard deviation is required")
        return result.local_std
    if operator == OPERATORS[1]:
        _, std = _square_annulus_moments(
            representation,
            outer=11,
            guard=3,
            chunk_frames=chunk_frames,
        )
        return std
    if operator == OPERATORS[2]:
        return None
    raise ExactTruthBenchmarkError(f"unknown operator: {operator}")


def score_representation(
    representation: Any,
    *,
    operator: str,
    scale_floor: float | None,
    chunk_frames: int,
) -> Any:
    """Return framewise spatial scores; never pool over time."""

    import torch

    if operator == OPERATORS[0]:
        if scale_floor is None or not math.isfinite(scale_floor) or scale_floor <= 0:
            raise ExactTruthBenchmarkError("radial score requires a positive frozen floor")
        return gamma_local_standardization(
            representation,
            _radial_spec(scale_floor=scale_floor),
            chunk_frames=chunk_frames,
            return_statistics=False,
        ).values
    if operator == OPERATORS[1]:
        if scale_floor is None or not math.isfinite(scale_floor) or scale_floor <= 0:
            raise ExactTruthBenchmarkError("square score requires a positive frozen floor")
        mean, std = _square_annulus_moments(
            representation,
            outer=11,
            guard=3,
            chunk_frames=chunk_frames,
        )
        return (representation - mean) / (
            torch.maximum(std, torch.as_tensor(scale_floor, dtype=std.dtype)) + 1e-6
        )
    if operator == OPERATORS[2]:
        if scale_floor is not None:
            raise ExactTruthBenchmarkError("positive-box control has no fitted scale floor")
        return _maintained_positive_box_score(
            representation,
            outer=11,
            guard=3,
            chunk_frames=chunk_frames,
        )
    raise ExactTruthBenchmarkError(f"unknown operator: {operator}")


def fit_scale_floors(
    config: ExactTruthConfig,
    specs: Sequence[MovieSpec],
) -> tuple[dict[str, float | None], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit radial/square floors using only zero-source floor-fit movies."""

    percentile = float(config.payload["operators"]["scale_floor_percentile"])
    chunk_frames = int(config.payload["resources"]["chunk_frames"])
    samples: dict[str, list[np.ndarray]] = {
        f"{representation}__{operator}": []
        for representation in REPRESENTATIONS
        for operator in OPERATORS[:2]
    }
    manifests = []
    for spec in specs:
        if spec.split != "scale_floor_fit" or spec.source_count != 0:
            raise ExactTruthBenchmarkError("scale floor received a non-null or wrong-split movie")
        movie = simulate_exact_truth_movie(config, spec)
        if movie.source_ids or movie.active.size:
            raise ExactTruthBenchmarkError("scale-floor movie unexpectedly contains truth sources")
        representations = aligned_representations(
            movie.values,
            energy_epsilon=float(config.payload["representations"]["energy_epsilon"]),
        )
        for representation_name, representation in representations.items():
            for operator in OPERATORS[:2]:
                std = _local_std(
                    representation, operator=operator, chunk_frames=chunk_frames
                )
                host = std.detach().cpu().numpy().astype(np.float32, copy=False)
                positive = host[host > 0]
                if positive.size == 0:
                    raise ExactTruthBenchmarkError("null local std contains no positive values")
                samples[f"{representation_name}__{operator}"].append(positive.copy())
        manifests.append(movie_manifest_row(movie))
    floors: dict[str, float | None] = {}
    rows = []
    for pipeline in pipeline_ids():
        operator = pipeline.split("__", 1)[1]
        if operator == OPERATORS[2]:
            floors[pipeline] = None
            continue
        pooled = np.concatenate(samples[pipeline]).astype(np.float64, copy=False)
        floor = float(np.percentile(pooled, percentile, method="linear"))
        if not math.isfinite(floor) or floor <= 0:
            raise ExactTruthBenchmarkError("fitted scale floor is not finite and positive")
        floors[pipeline] = floor
        rows.append(
            {
                "pipeline_id": pipeline,
                "representation": pipeline.split("__", 1)[0],
                "operator": operator,
                "scale_floor_percentile": percentile,
                "scale_floor": floor,
                "positive_local_std_samples": int(pooled.size),
                "fit_split": "scale_floor_fit",
                "fit_source_count": 0,
                "evaluation_truth_used": False,
            }
        )
    if len(rows) != len(REPRESENTATIONS) * 2:
        raise AssertionError("scale-floor row count changed")
    return floors, rows, manifests


def score_movie(
    config: ExactTruthConfig,
    movie: ExactTruthMovie,
    floors: Mapping[str, float | None],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Compute all nine full framewise score tensors and separate timing rows."""

    chunk_frames = int(config.payload["resources"]["chunk_frames"])
    started = time.perf_counter()
    representations = aligned_representations(
        movie.values,
        energy_epsilon=float(config.payload["representations"]["energy_epsilon"]),
    )
    representation_seconds = time.perf_counter() - started
    outputs: dict[str, np.ndarray] = {}
    timing = []
    for representation_name in REPRESENTATIONS:
        representation = representations[representation_name]
        for operator in OPERATORS:
            pipeline = f"{representation_name}__{operator}"
            before = time.perf_counter()
            score = score_representation(
                representation,
                operator=operator,
                scale_floor=floors[pipeline],
                chunk_frames=chunk_frames,
            )
            elapsed = time.perf_counter() - before
            host = score.detach().cpu().numpy().astype(np.float32, copy=False)
            if host.shape != (len(movie.values) - 1, *movie.values.shape[1:]) or not np.isfinite(host).all():
                raise ExactTruthBenchmarkError("pipeline score shape/finite contract changed")
            outputs[pipeline] = host.copy()
            timing.append(
                {
                    "movie_id": movie.spec.movie_id,
                    "split": movie.spec.split,
                    "pipeline_id": pipeline,
                    "representation": representation_name,
                    "operator": operator,
                    "score_seconds": elapsed,
                    "score_ms_per_output_frame": 1000.0 * elapsed / len(host),
                    "common_and_representation_seconds_shared": representation_seconds,
                    "quality_metric_selection_used_timing": False,
                }
            )
    return outputs, timing


def frame_nms_candidates(
    score: np.ndarray,
    *,
    nms_distance_px: int,
) -> list[list[tuple[float, int, int]]]:
    """Extract the complete deterministic spatial-NMS universe per frame."""

    values = np.asarray(score, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ExactTruthBenchmarkError("framewise score must be finite TYX")
    result = []
    for frame in values:
        result.append(
            extract_separated_local_maxima(
                frame,
                int(nms_distance_px),
                threshold=-np.inf,
                limit=10_000,
            )
        )
    return result


def frame_peak_scores(score: np.ndarray, *, nms_distance_px: int) -> list[np.ndarray]:
    """Extract deterministic per-frame peak score vectors without thresholding."""

    return [
        np.asarray([peak[0] for peak in peaks], dtype=np.float64)
        for peaks in frame_nms_candidates(score, nms_distance_px=nms_distance_px)
    ]


def calibrate_empirical_thresholds(
    null_peak_scores: Mapping[str, Sequence[np.ndarray]],
    *,
    target_burdens: Sequence[float] = BURDENS,
) -> tuple[dict[tuple[str, float], float], list[dict[str, Any]]]:
    """Calibrate fixed proposal burdens using only already-extracted null peaks.

    The function has no source-count or evaluation-budget input.  Thresholds
    are keyed only by pipeline and target null proposals per source frame.
    """

    targets = tuple(float(value) for value in target_burdens)
    if targets != BURDENS:
        raise ExactTruthBenchmarkError(f"target burdens must remain {BURDENS}")
    if set(null_peak_scores) != set(pipeline_ids()):
        raise ExactTruthBenchmarkError(
            "null calibration must contain exactly the frozen nine pipelines"
        )
    thresholds: dict[tuple[str, float], float] = {}
    rows = []
    for pipeline in sorted(null_peak_scores):
        frames = list(null_peak_scores[pipeline])
        if not frames:
            raise ExactTruthBenchmarkError(f"pipeline {pipeline} has no null frames")
        pooled = np.sort(
            np.concatenate([np.asarray(row, dtype=np.float64) for row in frames])
        )[::-1]
        if pooled.size == 0 or not np.isfinite(pooled).all():
            raise ExactTruthBenchmarkError("null calibration has no finite NMS peaks")
        for target in targets:
            allowed = int(math.floor(target * len(frames)))
            if allowed <= 0:
                threshold = float(pooled[0])
            elif allowed >= len(pooled):
                threshold = float(np.nextafter(pooled[-1], -np.inf))
            else:
                threshold = float(pooled[allowed])
            counts = np.asarray(
                [np.count_nonzero(np.asarray(row) > threshold) for row in frames]
            )
            achieved = float(counts.mean())
            if achieved > target + 1e-12:
                raise ExactTruthBenchmarkError("empirical calibration exceeded target burden")
            thresholds[(pipeline, target)] = threshold
            representation, operator = pipeline.split("__", 1)
            rows.append(
                {
                    "pipeline_id": pipeline,
                    "representation": representation,
                    "operator": operator,
                    "target_nms_proposals_per_frame": target,
                    "threshold": threshold,
                    "calibration_frame_count": len(frames),
                    "calibration_total_peak_count": int(pooled.size),
                    "allowed_total_proposals": allowed,
                    "achieved_null_nms_proposals_per_frame": achieved,
                    "maximum_null_frame_proposals": int(counts.max()),
                    "calibration_source_count": 0,
                    "threshold_depends_on_source_count": False,
                    "candidate_budget_used": False,
                    "probability_of_false_alarm_claimed": False,
                }
            )
    return thresholds, rows


def framewise_truth_metrics(
    peaks: Sequence[tuple[float, int, int]],
    truth: Sequence[Mapping[str, Any]],
    *,
    match_radius_px: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score-ranked one-to-one matching against exhaustive simulator truth."""

    if not math.isfinite(match_radius_px) or match_radius_px <= 0:
        raise ExactTruthBenchmarkError("match radius must be finite and positive")
    if any(peaks[index][0] < peaks[index + 1][0] for index in range(len(peaks) - 1)):
        raise ExactTruthBenchmarkError("candidate peaks must be score ranked")
    source_ids = [str(row["source_id"]) for row in truth]
    if len(source_ids) != len(set(source_ids)):
        raise ExactTruthBenchmarkError("active truth source identities must be unique")
    remaining = set(range(len(truth)))
    assignments: dict[int, tuple[int, float]] = {}
    for candidate_index, (_, x_px, y_px) in enumerate(peaks):
        choices = [
            (
                math.hypot(
                    float(x_px) - float(truth[index]["x_px"]),
                    float(y_px) - float(truth[index]["y_px"]),
                ),
                index,
            )
            for index in remaining
        ]
        if not choices:
            break
        distance, truth_index = min(choices)
        if distance <= match_radius_px:
            remaining.remove(truth_index)
            assignments[candidate_index] = (truth_index, float(distance))
    candidate_rows = []
    duplicates = 0
    localization = []
    for index, (score, x_px, y_px) in enumerate(peaks):
        assignment = assignments.get(index)
        nearest = min(
            (
                math.hypot(
                    float(x_px) - float(row["x_px"]),
                    float(y_px) - float(row["y_px"]),
                )
                for row in truth
            ),
            default=float("inf"),
        )
        if assignment is not None:
            truth_index, distance = assignment
            classification = "true_positive"
            matched_source_id = str(truth[truth_index]["source_id"])
            localization.append(distance)
        elif nearest <= match_radius_px:
            classification = "duplicate_false_positive"
            matched_source_id = ""
            duplicates += 1
        else:
            classification = "background_false_positive"
            matched_source_id = ""
        candidate_rows.append(
            {
                "candidate_rank": index + 1,
                "score": float(score),
                "x_px": int(x_px),
                "y_px": int(y_px),
                "classification": classification,
                "matched_source_id": matched_source_id,
                "match_distance_px": "" if assignment is None else assignment[1],
                "nearest_active_truth_distance_px": "" if not truth else nearest,
            }
        )
    tp = len(assignments)
    proposals = len(peaks)
    truth_count = len(truth)
    fp = proposals - tp
    fn = truth_count - tp
    denominator = 2 * tp + fp + fn
    metric = {
        "active_truth_count": truth_count,
        "proposal_count": proposals,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicate_false_positive_count": duplicates,
        "background_false_positive_count": fp - duplicates,
        "precision": "" if proposals == 0 else tp / proposals,
        "recall": "" if truth_count == 0 else tp / truth_count,
        "f1": "" if denominator == 0 else 2 * tp / denominator,
        "localization_distance_sum_px": float(sum(localization)),
        "localized_match_count": len(localization),
        "mean_localization_px": "" if not localization else float(np.mean(localization)),
    }
    if (
        tp + fp != proposals
        or tp + fn != truth_count
        or duplicates + metric["background_false_positive_count"] != fp
        or len(localization) != tp
    ):
        raise AssertionError("framewise metric accounting failed")
    return metric, candidate_rows


def movie_manifest_row(movie: ExactTruthMovie) -> dict[str, Any]:
    return {
        **movie.spec.as_dict(),
        "frames": len(movie.values),
        "height_px": movie.values.shape[1],
        "width_px": movie.values.shape[2],
        "movie_sha256": _array_sha256(movie.values),
        "truth_sha256": movie.truth_sha256,
        "source_identity_count": len(movie.source_ids),
        "active_source_frame_count": int(np.count_nonzero(movie.active)),
        "truth_is_exhaustive_within_simulator": True,
    }


def aggregate_frame_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Reconcile micro metrics from exact integer frame accounting."""

    groups: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in rows:
        tp = int(row["tp"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        proposals = int(row["proposal_count"])
        truth_count = int(row["active_truth_count"])
        duplicates = int(row["duplicate_false_positive_count"])
        background = int(row["background_false_positive_count"])
        localized = int(row["localized_match_count"])
        if (
            tp + fp != proposals
            or tp + fn != truth_count
            or duplicates + background != fp
            or localized != tp
        ):
            raise ExactTruthBenchmarkError("frame metric row does not reconcile")
        key = tuple(row[field] for field in group_fields)
        aggregate = groups.setdefault(
            key,
            {
                "frame_count": 0,
                "active_truth_count": 0,
                "proposal_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "duplicate_false_positive_count": 0,
                "background_false_positive_count": 0,
                "localization_distance_sum_px": 0.0,
                "localized_match_count": 0,
            },
        )
        for field in (
            "frame_count",
            "active_truth_count",
            "proposal_count",
            "tp",
            "fp",
            "fn",
            "duplicate_false_positive_count",
            "background_false_positive_count",
            "localized_match_count",
        ):
            source_field = field if field != "frame_count" else None
            aggregate[field] += 1 if source_field is None else int(row[source_field])
        aggregate["localization_distance_sum_px"] += float(
            row["localization_distance_sum_px"]
        )
    output = []
    for key, counts in sorted(groups.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else ""
        recall = tp / (tp + fn) if tp + fn else ""
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else ""
        localization = (
            counts["localization_distance_sum_px"] / counts["localized_match_count"]
            if counts["localized_match_count"]
            else ""
        )
        output.append(
            {
                **dict(zip(group_fields, key)),
                **{
                    field: int(value)
                    for field, value in counts.items()
                    if field != "localization_distance_sum_px"
                },
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_localization_px": localization,
                "duplicates_per_frame": counts["duplicate_false_positive_count"]
                / counts["frame_count"],
                "precision_scope": "exhaustive_synthetic_truth_only",
            }
        )
    return output


def _read_frame_metric_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            yield row


def _implementation_hashes(config: ExactTruthConfig) -> dict[str, str]:
    relative_paths = (
        "neurobench/experiments/gamma_ls_difference/exact_truth_benchmark.py",
        "neurobench/algorithms/gamma_local_standardization.py",
        "neurobench/experiments/gamma_ls_difference/gpu_representations.py",
        "neurobench/experiments/gamma_ls_difference/support_sufficiency.py",
        "neurobench/metrics/sparse_detection.py",
        "docs/workflows/gamma_ls_exact_truth_benchmark_v1.md",
    )
    result = {}
    for relative in relative_paths:
        path = config.repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"planned implementation file is missing: {path}")
        result[relative] = _sha256(path)
    return result


def build_frozen_plan(config: ExactTruthConfig) -> dict[str, Any]:
    specs = movie_specs(config)
    split_counts = {
        split: sum(row.split == split for row in specs)
        for split in ("scale_floor_fit", "threshold_calibration", "evaluation", "latency")
    }
    return {
        "schema_version": PROTOCOL_VERSION,
        "experiment_id": config.payload["experiment_id"],
        "status": "frozen_before_result_execution",
        "config_sha256": _canonical_sha256(config.portable()),
        "implementation_hashes": _implementation_hashes(config),
        "split_movie_counts": split_counts,
        "movie_count": len(specs),
        "pipeline_ids": list(pipeline_ids()),
        "fixed_radial_context": {
            "half_width_px": 15,
            "guard_radius_px": 7,
            "shape_n": 9.0,
            "mode_fraction_of_half_width": 0.5,
            "mode_radius_px": 7.5,
            "support_width_px": 31,
        },
        "calibration": {
            "scale_floor_source_count": 0,
            "threshold_source_count": 0,
            "target_nms_proposals_per_frame": list(BURDENS),
            "threshold_or_budget_depends_on_evaluation_source_count": False,
            "evaluation_candidate_budget": None,
        },
        "evaluation": {
            "grain": "source_frame",
            "scored_source_frames_per_movie": int(config.payload["simulator"]["frames"]) - 1,
            "temporal_pooling": False,
            "time_collapsed_site_scores": False,
            "deterministic_nms": True,
            "one_to_one_framewise_matching": True,
            "family_holdouts": list(FAMILY_HOLDOUTS),
            "precision_scope": "exhaustive_synthetic_truth_only",
        },
        "latency": {
            "separate_fixture": True,
            "device": "cpu",
            "quality_metric_selection_uses_latency": False,
        },
        "claim_boundary": (
            "Mechanistic exact target-source truth only; generated neuropil and noise "
            "are defined nuisance processes. This is not biological validation, "
            "independent-animal generalization, or evidence of in-vivo precision."
        ),
        "scientific_audit_status_after_metric_runner": "pending_full_media_renderer_and_inventory_validation",
    }


def freeze_plan(
    config: ExactTruthConfig,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"exact-truth plan output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"exact-truth plan parent is missing: {destination.parent}")
    work = destination.parent / f".{destination.name}.freeze-work"
    if work.exists():
        raise FileExistsError(f"exact-truth plan work directory exists: {work}")
    plan = build_frozen_plan(config)
    work.mkdir()
    try:
        _atomic_json(work / "config.frozen.json", config.portable())
        _atomic_json(work / "plan.json", plan)
        _write_tsv(work / "split_manifest.tsv", [row.as_dict() for row in movie_specs(config)])
        validation = {
            "status": "passed_frozen_plan_contract",
            "checks": {
                "plan_precedes_result_execution": True,
                "scale_floor_and_threshold_splits_are_source_free": True,
                "split_seeds_are_disjoint": True,
                "evaluation_seeds_untouched": True,
                "three_family_holdouts": len(FAMILY_HOLDOUTS) == 3,
                "nine_pipeline_grid": len(pipeline_ids()) == 9,
                "no_temporal_pooling": True,
                "no_source_count_derived_threshold_or_budget": True,
                "cpu_only_runner": True,
            },
        }
        _atomic_json(work / "validation.json", validation)
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        if destination.exists():
            raise FileExistsError(f"exact-truth plan appeared during freeze: {destination}")
        work.replace(destination)
        return plan
    except Exception:
        if work.exists():
            _atomic_json(
                work / "status.json",
                {"status": "failed_plan_freeze_not_executable"},
            )
        raise


def verify_frozen_plan(
    config: ExactTruthConfig,
    plan_dir: str | Path,
) -> dict[str, Any]:
    root = Path(plan_dir).expanduser().resolve()
    indexed = _verify_artifact_index(root)
    frozen_config = json.loads((root / "config.frozen.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    if frozen_config != config.portable():
        raise ExactTruthBenchmarkError("frozen plan config differs from requested config")
    if plan != build_frozen_plan(config):
        raise ExactTruthBenchmarkError("frozen plan or implementation hashes changed")
    if validation.get("status") != "passed_frozen_plan_contract" or not all(
        validation.get("checks", {}).values()
    ):
        raise ExactTruthBenchmarkError("frozen plan validation did not pass")
    return {
        "root": str(root),
        "plan_sha256": _sha256(root / "plan.json"),
        "config_frozen_sha256": _sha256(root / "config.frozen.json"),
        **indexed,
    }


def _candidate_id(
    movie_id: str,
    pipeline: str,
    target: float,
    source_frame: int,
    rank: int,
    x_px: int,
    y_px: int,
) -> str:
    payload = (
        f"{movie_id}|{pipeline}|{target:.12g}|{source_frame}|{rank}|{x_px}|{y_px}"
    )
    return "etcand_" + hashlib.sha256(payload.encode("ascii")).hexdigest()[:24]


def _latency_rows(
    config: ExactTruthConfig,
    movie: ExactTruthMovie,
    floors: Mapping[str, float | None],
) -> list[dict[str, Any]]:
    import torch

    resources = config.payload["resources"]
    warmups = int(resources["latency_warmup_iterations"])
    iterations = int(resources["latency_timed_iterations"])
    representations = aligned_representations(
        movie.values,
        energy_epsilon=float(config.payload["representations"]["energy_epsilon"]),
    )
    rows = []
    for representation_name in REPRESENTATIONS:
        representation = representations[representation_name]
        for operator in OPERATORS:
            pipeline = f"{representation_name}__{operator}"

            def operation() -> Any:
                return score_representation(
                    representation,
                    operator=operator,
                    scale_floor=floors[pipeline],
                    chunk_frames=int(resources["chunk_frames"]),
                )

            with torch.no_grad():
                for _ in range(warmups):
                    result = operation()
                    del result
                samples = []
                for _ in range(iterations):
                    started = time.perf_counter()
                    result = operation()
                    samples.append((time.perf_counter() - started) * 1000.0)
                    del result
            values = np.asarray(samples, dtype=np.float64)
            rows.append(
                {
                    "movie_id": movie.spec.movie_id,
                    "pipeline_id": pipeline,
                    "representation": representation_name,
                    "operator": operator,
                    "device": "cpu",
                    "cpu_threads": int(resources["cpu_threads"]),
                    "output_frames_per_iteration": len(representation),
                    "warmup_iterations": warmups,
                    "timed_iterations": iterations,
                    "execution_mode": "batch_95_output_frames_chunked_by_32",
                    "p50_ms_per_movie": float(np.percentile(values, 50)),
                    "p95_ms_per_movie": float(np.percentile(values, 95)),
                    "mean_ms_per_movie": float(values.mean()),
                    "p50_ms_per_output_frame": float(
                        np.percentile(values, 50) / len(representation)
                    ),
                    "scope": "batched_spatial_operator_throughput_only_preprocessing_and_nms_excluded",
                    "quality_metric_selection_used_latency": False,
                    "gpu_or_1khz_claim": False,
                }
            )
    return rows


TRUTH_FIELDS = (
    "movie_id",
    "family",
    "family_holdout",
    "seed",
    "source_count",
    "source_id",
    "source_frame_zero_based",
    "source_frame_ui_one_based",
    "center_x_px",
    "center_y_px",
    "latent_trace",
    "event_impulse",
    "active",
    "active_trace_threshold",
)
FRAME_FIELDS = (
    "movie_id",
    "family",
    "family_holdout",
    "seed",
    "source_count",
    "pipeline_id",
    "representation",
    "operator",
    "target_nms_proposals_per_frame",
    "threshold",
    "source_frame_zero_based",
    "source_frame_ui_one_based",
    "active_truth_count",
    "proposal_count",
    "tp",
    "fp",
    "fn",
    "duplicate_false_positive_count",
    "background_false_positive_count",
    "precision",
    "recall",
    "f1",
    "localization_distance_sum_px",
    "localized_match_count",
    "mean_localization_px",
    "metric_scope",
)
CANDIDATE_FIELDS = (
    "candidate_id",
    "movie_id",
    "family",
    "family_holdout",
    "seed",
    "source_count",
    "pipeline_id",
    "target_nms_proposals_per_frame",
    "threshold",
    "source_frame_zero_based",
    "source_frame_ui_one_based",
    "candidate_rank",
    "score",
    "x_px",
    "y_px",
    "classification",
    "matched_source_id",
    "match_distance_px",
    "nearest_active_truth_distance_px",
    "truth_scope",
)


def run_benchmark(
    config: ExactTruthConfig,
    *,
    plan_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute the frozen CPU exact-truth plan and publish metric artifacts."""

    if device != "cpu" or config.payload["resources"]["device"] != "cpu":
        raise ExactTruthBenchmarkError("version 1 exact-truth runner is CPU-only")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"exact-truth result output exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"exact-truth result parent is missing: {destination.parent}")
    work = destination.parent / f".{destination.name}.exact-truth-work"
    if work.exists():
        raise FileExistsError(f"exact-truth result work directory exists: {work}")
    frozen_plan = verify_frozen_plan(config, plan_dir)
    minimum_disk = int(float(config.payload["resources"]["minimum_free_disk_gib"]) * 2**30)
    free_disk = shutil.disk_usage(destination.parent).free
    if free_disk < minimum_disk:
        raise ExactTruthBenchmarkError(
            f"free disk {free_disk} is below frozen minimum {minimum_disk} bytes"
        )
    import torch

    torch.set_num_threads(int(config.payload["resources"]["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    import platform as platform_module
    import scipy
    import sys

    specs = movie_specs(config)
    work.mkdir()
    runtime_environment = {
        "python": sys.version,
        "platform": platform_module.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_cpu_threads": torch.get_num_threads(),
        "cuda_used": False,
    }
    try:
        _atomic_json(work / "runtime_environment.json", runtime_environment)
        _atomic_json(
            work / "run_contract.json",
            {
                "schema_version": 1,
                "experiment_id": config.payload["experiment_id"],
                "run_type": "framewise_exact_truth_gamma_ls_benchmark",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "frozen_plan": frozen_plan,
                "config_sha256": _canonical_sha256(config.portable()),
                "device": "cpu",
                "cpu_threads": int(config.payload["resources"]["cpu_threads"]),
                "runtime_environment": runtime_environment,
                "precision_scope": "exhaustive_synthetic_truth_only",
                "biological_validation_claimed": False,
            },
        )
    except Exception as error:
        _atomic_json(
            work / "status.json",
            {
                "status": "failed_exact_truth_initialization_not_promotable",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
                "resume_authorized": False,
            },
        )
        raise

    def heartbeat(stage: str, **details: Any) -> None:
        _atomic_json(
            work / "heartbeat.json",
            {
                "status": "running",
                "stage": stage,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **details,
            },
        )

    try:
        floor_specs = [row for row in specs if row.split == "scale_floor_fit"]
        heartbeat("scale_floor_fit", completed=0, total=len(floor_specs))
        floors, floor_rows, movie_manifest = fit_scale_floors(config, floor_specs)
        _write_tsv(work / "scale_floors.tsv", floor_rows)

        calibration_specs = [row for row in specs if row.split == "threshold_calibration"]
        null_peaks: dict[str, list[np.ndarray]] = {
            pipeline: [] for pipeline in pipeline_ids()
        }
        score_manifest_rows = []
        stage_timing_rows = []
        for index, spec in enumerate(calibration_specs, start=1):
            heartbeat("threshold_calibration", completed=index - 1, total=len(calibration_specs))
            movie = simulate_exact_truth_movie(config, spec)
            if movie.source_ids or movie.active.size:
                raise ExactTruthBenchmarkError("threshold calibration movie contains sources")
            movie_manifest.append(movie_manifest_row(movie))
            scores, timing = score_movie(config, movie, floors)
            stage_timing_rows.extend(timing)
            for pipeline, score in scores.items():
                null_peaks[pipeline].extend(
                    frame_peak_scores(
                        score,
                        nms_distance_px=int(config.payload["evaluation"]["nms_distance_px"]),
                    )
                )
                score_manifest_rows.append(
                    {
                        "movie_id": spec.movie_id,
                        "split": spec.split,
                        "pipeline_id": pipeline,
                        "score_shape_tyx": json.dumps(list(score.shape)),
                        "score_sha256": _array_sha256(score),
                        "temporal_pooling": False,
                    }
                )
        thresholds, threshold_rows = calibrate_empirical_thresholds(null_peaks)
        _write_tsv(work / "empirical_thresholds.tsv", threshold_rows)
        del null_peaks

        evaluation_specs = [row for row in specs if row.split == "evaluation"]
        truth_rows_written = 0
        expected_truth_rows = sum(
            int(config.payload["simulator"]["frames"]) * row.source_count
            for row in evaluation_specs
        )
        with AtomicTsvSink(work / "source_truth.tsv", TRUTH_FIELDS) as truth_sink, AtomicTsvSink(
            work / "frame_metrics.tsv", FRAME_FIELDS
        ) as frame_sink, AtomicTsvSink(
            work / "candidates.tsv", CANDIDATE_FIELDS
        ) as candidate_sink:
            for movie_index, spec in enumerate(evaluation_specs, start=1):
                heartbeat(
                    "untouched_evaluation",
                    completed_movies=movie_index - 1,
                    total_movies=len(evaluation_specs),
                )
                movie = simulate_exact_truth_movie(config, spec)
                movie_manifest.append(movie_manifest_row(movie))
                for frame in range(len(movie.values)):
                    for source_index, source_id in enumerate(movie.source_ids):
                        truth_sink.write(
                            {
                                "movie_id": spec.movie_id,
                                "family": spec.family,
                                "family_holdout": spec.family_holdout,
                                "seed": spec.seed,
                                "source_count": spec.source_count,
                                "source_id": source_id,
                                "source_frame_zero_based": frame,
                                "source_frame_ui_one_based": frame + 1,
                                "center_x_px": float(movie.centers_yx[frame, source_index, 1]),
                                "center_y_px": float(movie.centers_yx[frame, source_index, 0]),
                                "latent_trace": float(movie.traces[frame, source_index]),
                                "event_impulse": bool(movie.events[frame, source_index]),
                                "active": bool(movie.active[frame, source_index]),
                                "active_trace_threshold": movie.active_trace_threshold,
                            }
                        )
                        truth_rows_written += 1
                scores, timing = score_movie(config, movie, floors)
                stage_timing_rows.extend(timing)
                for pipeline, score in scores.items():
                    representation, operator = pipeline.split("__", 1)
                    score_manifest_rows.append(
                        {
                            "movie_id": spec.movie_id,
                            "split": spec.split,
                            "pipeline_id": pipeline,
                            "score_shape_tyx": json.dumps(list(score.shape)),
                            "score_sha256": _array_sha256(score),
                            "temporal_pooling": False,
                        }
                    )
                    frame_candidates = frame_nms_candidates(
                        score,
                        nms_distance_px=int(
                            config.payload["evaluation"]["nms_distance_px"]
                        ),
                    )
                    for score_frame_index, all_peaks in enumerate(frame_candidates):
                        source_frame = score_frame_index + 1
                        truth = movie.active_labels(source_frame)
                        for target in BURDENS:
                            threshold = thresholds[(pipeline, target)]
                            # Candidate construction is shared across burdens. Because
                            # NMS is score-ranked, filtering this complete universe is
                            # exactly equivalent to applying the strict threshold first.
                            peaks = [
                                peak for peak in all_peaks if peak[0] > threshold
                            ]
                            metric, candidate_rows = framewise_truth_metrics(
                                peaks,
                                truth,
                                match_radius_px=float(
                                    config.payload["evaluation"]["match_radius_px"]
                                ),
                            )
                            frame_sink.write(
                                {
                                    "movie_id": spec.movie_id,
                                    "family": spec.family,
                                    "family_holdout": spec.family_holdout,
                                    "seed": spec.seed,
                                    "source_count": spec.source_count,
                                    "pipeline_id": pipeline,
                                    "representation": representation,
                                    "operator": operator,
                                    "target_nms_proposals_per_frame": target,
                                    "threshold": threshold,
                                    "source_frame_zero_based": source_frame,
                                    "source_frame_ui_one_based": source_frame + 1,
                                    **metric,
                                    "metric_scope": "exhaustive_synthetic_truth_only",
                                }
                            )
                            for candidate in candidate_rows:
                                candidate_sink.write(
                                    {
                                        "candidate_id": _candidate_id(
                                            spec.movie_id,
                                            pipeline,
                                            target,
                                            source_frame,
                                            int(candidate["candidate_rank"]),
                                            int(candidate["x_px"]),
                                            int(candidate["y_px"]),
                                        ),
                                        "movie_id": spec.movie_id,
                                        "family": spec.family,
                                        "family_holdout": spec.family_holdout,
                                        "seed": spec.seed,
                                        "source_count": spec.source_count,
                                        "pipeline_id": pipeline,
                                        "target_nms_proposals_per_frame": target,
                                        "threshold": threshold,
                                        "source_frame_zero_based": source_frame,
                                        "source_frame_ui_one_based": source_frame + 1,
                                        **candidate,
                                        "truth_scope": "exhaustive_synthetic_truth_only",
                                    }
                                )
        if truth_rows_written != expected_truth_rows:
            raise AssertionError("source-truth row count changed")

        family_group_fields = (
            "family",
            "family_holdout",
            "source_count",
            "pipeline_id",
            "representation",
            "operator",
            "target_nms_proposals_per_frame",
        )
        seed_summary = aggregate_frame_metrics(
            _read_frame_metric_rows(work / "frame_metrics.tsv"),
            group_fields=(
                "family",
                "family_holdout",
                "seed",
                "source_count",
                "pipeline_id",
                "representation",
                "operator",
                "target_nms_proposals_per_frame",
            ),
        )
        family_summary = aggregate_frame_metrics(
            _read_frame_metric_rows(work / "frame_metrics.tsv"),
            group_fields=family_group_fields,
        )
        overall_group_fields = (
            "pipeline_id",
            "representation",
            "operator",
            "target_nms_proposals_per_frame",
        )
        overall_summary = aggregate_frame_metrics(
            _read_frame_metric_rows(work / "frame_metrics.tsv"),
            group_fields=overall_group_fields,
        )
        _write_tsv(work / "seed_summary.tsv", seed_summary)
        _write_tsv(work / "family_summary.tsv", family_summary)
        _write_tsv(work / "overall_summary.tsv", overall_summary)
        _write_tsv(work / "movie_manifest.tsv", movie_manifest)
        _write_tsv(work / "score_manifest.tsv", score_manifest_rows)
        _write_tsv(work / "stage_runtime_diagnostics.tsv", stage_timing_rows)

        latency_spec = next(row for row in specs if row.split == "latency")
        latency_movie = simulate_exact_truth_movie(config, latency_spec)
        movie_manifest.append(movie_manifest_row(latency_movie))
        latency_rows = _latency_rows(config, latency_movie, floors)
        _write_tsv(work / "latency.tsv", latency_rows)

        # Re-write the manifest after adding the isolated latency movie.
        _write_tsv(work / "movie_manifest.tsv", movie_manifest)
        _write_tsv(work / "split_manifest.tsv", [row.as_dict() for row in specs])
        audit_hooks = {
            "schema_version": 1,
            "status": "pending_full_media_renderer_and_inventory_validation",
            "exact_truth_section": "source_truth.tsv",
            "model_candidate_section": "candidates.tsv",
            "comparison_tables": [
                "seed_summary.tsv",
                "family_summary.tsv",
                "overall_summary.tsv",
            ],
            "required_representative_family_movies": list(FAMILIES),
            "required_stage_sequence": [
                "simulated raw frame",
                "causal Gaussian-plus-EMA common input",
                "aligned fixed representation",
                "framewise spatial operator score",
                "empirical threshold",
                "deterministic NMS candidates",
            ],
            "truth_marker_role": "exact simulator source center",
            "model_marker_role": "thresholded framewise candidate",
            "temporal_pooling": False,
            "scientific_audit_complete": False,
        }
        _atomic_json(work / "scientific_audit_hooks.json", audit_hooks)

        expected_frame_rows = (
            len(evaluation_specs)
            * (int(config.payload["simulator"]["frames"]) - 1)
            * len(pipeline_ids())
            * len(BURDENS)
        )
        actual_frame_rows = sum(1 for _ in _read_frame_metric_rows(work / "frame_metrics.tsv"))
        if actual_frame_rows != expected_frame_rows:
            raise AssertionError(
                f"frame metric count {actual_frame_rows} != {expected_frame_rows}"
            )
        primary = [
            row
            for row in overall_summary
            if math.isclose(float(row["target_nms_proposals_per_frame"]), 1.0)
        ]
        if len(primary) != len(pipeline_ids()) or any(row["f1"] == "" for row in primary):
            raise ExactTruthBenchmarkError("primary synthetic summary is incomplete")
        descriptive_top = max(primary, key=lambda row: float(row["f1"]))
        claim_boundary = {
            "truth": "exhaustive_and_exact_for_defined_discrete_target_sources_within_generated_movies",
            "simulated_neuropil_bleaching_and_noise_role": "defined_nuisance_processes_not_target_sources",
            "precision_scope": "synthetic_truth_only",
            "biological_precision_claimed": False,
            "biological_validation_claimed": False,
            "independent_animal_generalization_claimed": False,
            "calibration_movies_are_source_free": True,
            "calibration_and_evaluation_seeds_disjoint": True,
            "family_holdouts": list(FAMILY_HOLDOUTS),
            "temporal_pooling_used": False,
            "time_collapsed_site_scores_used": False,
            "source_count_derived_threshold_or_budget_used": False,
            "latency_used_for_quality_selection": False,
            "evaluation_used_for_model_or_operator_selection": False,
            "scientific_audit_complete": False,
            "paper_promotion_ready": False,
        }
        summary = {
            "schema_version": 1,
            "experiment_id": config.payload["experiment_id"],
            "status": "complete_exact_truth_metrics_scientific_audit_pending",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_plan": frozen_plan,
            "design": {
                "families": list(FAMILIES),
                "family_holdouts": list(FAMILY_HOLDOUTS),
                "source_counts": list(SOURCE_COUNTS),
                "evaluation_movies": len(evaluation_specs),
                "scored_frames_per_movie": int(config.payload["simulator"]["frames"])
                - 1,
                "pipelines": list(pipeline_ids()),
                "target_proposal_burdens_per_frame": list(BURDENS),
            },
            "primary_target_nms_proposals_per_frame": 1.0,
            "descriptive_highest_aggregate_f1_at_primary_burden": {
                **descriptive_top,
                "used_for_model_or_operator_selection": False,
                "independently_confirmed_winner": False,
            },
            "thresholds": threshold_rows,
            "scale_floors": floor_rows,
            "claim_boundary": claim_boundary,
        }
        _atomic_json(work / "claim_boundary.json", claim_boundary)
        _atomic_json(work / "summary.json", summary)
        checks = {
            "frozen_plan_verified_before_output_mutation": True,
            "exact_truth_rows_complete": truth_rows_written == expected_truth_rows,
            "frame_metric_rows_exact": actual_frame_rows == expected_frame_rows,
            "calibration_source_count_zero": all(
                int(row["calibration_source_count"]) == 0 for row in threshold_rows
            ),
            "thresholds_not_keyed_by_source_count": len(thresholds)
            == len(pipeline_ids()) * len(BURDENS),
            "evaluation_seed_split_isolated": True,
            "all_six_families_evaluated": {row["family"] for row in family_summary}
            == set(FAMILIES),
            "three_family_holdouts_evaluated": {
                row["family"] for row in family_summary if str(row["family_holdout"]).lower() == "true"
            }
            == set(FAMILY_HOLDOUTS),
            "no_temporal_pooling": True,
            "no_time_collapsed_site_scores": True,
            "latency_separate_from_quality": all(
                row["quality_metric_selection_used_latency"] is False
                for row in latency_rows
            ),
            "precision_limited_to_synthetic_truth": True,
            "scientific_audit_pending": True,
        }
        _atomic_json(
            work / "validation.json",
            {
                "status": "passed_exact_truth_metric_artifact_contract",
                "checks": checks,
                "all_checks_pass": all(checks.values()),
                "row_counts": {
                    "source_truth": truth_rows_written,
                    "frame_metrics": actual_frame_rows,
                    "thresholds": len(threshold_rows),
                    "seed_summary": len(seed_summary),
                    "family_summary": len(family_summary),
                    "overall_summary": len(overall_summary),
                },
            },
        )
        _atomic_json(
            work / "llm_context.json",
            {
                "entrypoint": "summary.json",
                "question": "How do frozen radial Gamma-LS and two simpler controls behave under exact framewise synthetic truth and mechanism-family shift?",
                "exact_truth": "source_truth.tsv",
                "frame_metrics": "frame_metrics.tsv",
                "candidate_ledger": "candidates.tsv",
                "seed_results": "seed_summary.tsv",
                "family_results": "family_summary.tsv",
                "latency": "latency.tsv",
                "calibration": "empirical_thresholds.tsv",
                "audit_hooks": "scientific_audit_hooks.json",
                "coordinate_convention": "x=column, y=row",
                "frame_convention": "zero-based source frame plus one-based UI frame",
                "limitations": [
                    "synthetic truth is exhaustive only inside generated movies",
                    "the six mechanisms do not establish biological realism",
                    "CPU operator latency is not a GPU or 1-kHz streaming claim",
                    "full scientific-audit media remain pending",
                ],
            },
        )
        _atomic_text(
            work / "REPORT.md",
            "# Framewise exact-truth Gamma-LS benchmark\n\n"
            f"The highest descriptive aggregate synthetic F1 at one calibrated null "
            f"proposal per source frame was `{descriptive_top['pipeline_id']}` "
            f"({float(descriptive_top['f1']):.4f}). This post-evaluation ordering is not "
            "model selection or an independently confirmed winner. See "
            "`family_summary.tsv` before interpreting "
            "the aggregate: source-specific nonrigid motion, empirical noise, and "
            "compound shift are mechanism-family holdouts.\n\n"
            "Thresholds were fit only on disjoint zero-source development movies and "
            "do not depend on evaluation source count. Every score and match is "
            "framewise; temporal max pooling and time-collapsed site scores are absent. "
            "Precision, localization, and duplicate counts are valid only against the "
            "exhaustive simulator truth. This is not biological validation. CPU latency "
            "is reported separately and full scientific-audit media remain pending.\n",
        )
        _atomic_json(
            work / "heartbeat.json",
            {
                "status": "complete_exact_truth_metrics_scientific_audit_pending",
                "stage": "metric_artifacts_complete",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        _atomic_json(
            work / "status.json",
            {
                "status": "complete_exact_truth_metrics_scientific_audit_pending",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        _atomic_json(work / "artifact_index.json", _artifact_index(work))
        if destination.exists():
            raise FileExistsError(f"exact-truth output appeared during run: {destination}")
        work.replace(destination)
        return summary
    except Exception as error:
        if work.exists():
            _atomic_json(
                work / "status.json",
                {
                    "status": "failed_exact_truth_run_not_promotable",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                    "resume_authorized": False,
                },
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or run the standalone framewise Gamma-LS exact-truth benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze the plan before result execution")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--artifact-dir", required=True)
    run = subparsers.add_parser("run", help="execute an already-frozen plan")
    run.add_argument("--config", required=True)
    run.add_argument("--plan-dir", required=True)
    run.add_argument("--artifact-dir", required=True)
    run.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ExactTruthConfig.load(args.config)
    if args.command == "freeze":
        payload = freeze_plan(config, output_dir=args.artifact_dir)
    else:
        payload = run_benchmark(
            config,
            plan_dir=args.plan_dir,
            output_dir=args.artifact_dir,
            device=args.device,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through exact commands
    raise SystemExit(main())


__all__ = [
    "BURDENS",
    "DEVELOPMENT_FAMILIES",
    "ExactTruthBenchmarkError",
    "ExactTruthConfig",
    "ExactTruthMovie",
    "FAMILIES",
    "FAMILY_HOLDOUTS",
    "MovieSpec",
    "OPERATORS",
    "REPRESENTATIONS",
    "aggregate_frame_metrics",
    "aligned_representations",
    "build_frozen_plan",
    "calibrate_empirical_thresholds",
    "fit_scale_floors",
    "frame_nms_candidates",
    "frame_peak_scores",
    "framewise_truth_metrics",
    "freeze_plan",
    "main",
    "movie_specs",
    "pipeline_ids",
    "run_benchmark",
    "score_movie",
    "score_representation",
    "simulate_exact_truth_movie",
    "verify_frozen_plan",
]
