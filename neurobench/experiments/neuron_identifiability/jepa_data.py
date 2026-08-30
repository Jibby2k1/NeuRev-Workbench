"""Leakage-resistant, copy-free data contracts for compact video JEPA pilots.

This module deliberately owns only the recording boundary.  It does not know
about annotations, ROIs, bursts, model architecture, or training losses.  Raw
TIFFs are opened read-only with :func:`tifffile.memmap`; manifests retain their
frozen hashes and portable URIs rather than workstation paths.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile

from neurobench.portable_paths import portable_path

from .contracts import stable_hash
from .discovery import sha256_file


EXPECTED_060126_RECORDING_COUNT = 11
EXPECTED_060126_TOTAL_FRAMES = 17_664
EXPECTED_060126_FRAME_SHAPE = (764, 1046)
EXPECTED_060126_DTYPE = "uint16"
EXPECTED_060126_BEHAVIOR_COUNTS = {"rest": 4, "left": 4, "right": 3}

# Frozen before representation learning.  The highest numbered recording for
# each behavior is a transparent acquisition-order extrapolation set.
FIXED_060126_VALIDATION_NUMBERS = {"rest": 10, "left": 12, "right": 15}
FIXED_060126_SPLIT_ID = "060126_behavior_stratified_last_trial_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORDING_NUMBER = re.compile(r"^\s*(\d+)\b")
_SPON_MARKERS = ("spon ca burst", "spon_ca_burst", "hindbrain to tail")


def _canonical_behavior(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"rest", "resting"}:
        return "rest"
    if normalized in {"left", "right"}:
        return normalized
    raise ValueError(f"unsupported 060126 behavior: {value!r}")


def _infer_behavior(name: str) -> str:
    lowered = name.casefold()
    matches = [token for token in ("resting", "rest", "left", "right") if token in lowered]
    canonical = {_canonical_behavior(token) for token in matches}
    if len(canonical) != 1:
        raise ValueError(f"cannot infer one behavior from recording name: {name!r}")
    return canonical.pop()


def _infer_recording_number(name: str) -> int:
    match = _RECORDING_NUMBER.match(Path(name).name)
    if match is None:
        raise ValueError(f"recording filename must begin with its trial number: {name!r}")
    return int(match.group(1))


def _valid_sha256(value: Any) -> str:
    normalized = str(value).strip().casefold()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError("recording descriptor requires a lowercase 64-character SHA-256")
    return normalized


def _portable_uri_from_mapping(
    row: Mapping[str, Any],
    *,
    repository_root: Path,
    data_root: Path,
) -> tuple[str, Path]:
    raw_uri = row.get("portable_path", row.get("uri"))
    raw_path = row.get("path")
    uri_was_explicit = raw_uri is not None

    if raw_uri is None and isinstance(raw_path, str) and "://" in raw_path:
        raw_uri, raw_path = raw_path, None
        uri_was_explicit = True

    resolved: Path | None = None
    if raw_path is not None:
        candidate = Path(str(raw_path)).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (repository_root / candidate).resolve()

    if raw_uri is None:
        if resolved is None:
            raise ValueError("recording descriptor requires portable_path/uri or path")
        raw_uri = portable_path(resolved, repository=repository_root, data=data_root)

    uri = str(raw_uri)
    if uri.startswith("repo://"):
        uri_path = (repository_root / uri.removeprefix("repo://")).resolve()
    elif uri.startswith("data://"):
        uri_path = (data_root / uri.removeprefix("data://")).resolve()
    elif uri.startswith("external://"):
        if resolved is None:
            raise ValueError("external:// descriptors require a runtime path")
        uri_path = resolved
    else:
        raise ValueError(f"unsupported recording URI scheme: {uri!r}")

    if uri_was_explicit and resolved is not None and resolved != uri_path and not uri.startswith("external://"):
        # A legacy absolute path can coexist with a portable URI only when both
        # point at the same configured authority.  Fail closed on ambiguity.
        raise ValueError(f"portable URI and runtime path disagree for {uri!r}")
    return uri, (resolved if resolved is not None else uri_path)


@dataclass(frozen=True)
class RecordingDescriptor:
    """One hash-frozen raw recording plus its runtime-only resolved path."""

    recording_id: str
    dataset_id: str
    recording_number: int
    behavior: str
    uri: str
    resolved_path: Path
    sha256: str
    shape: tuple[int, int, int]
    dtype: str
    pixel_size_um: float | None = None
    frame_interval_s: float | None = None

    def __post_init__(self) -> None:
        if not self.recording_id or not self.dataset_id:
            raise ValueError("recording_id and dataset_id are required")
        if self.recording_number < 1:
            raise ValueError("recording_number must be positive")
        object.__setattr__(self, "behavior", _canonical_behavior(self.behavior))
        object.__setattr__(self, "sha256", _valid_sha256(self.sha256))
        try:
            object.__setattr__(self, "dtype", np.dtype(self.dtype).name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid recording dtype: {self.dtype!r}") from exc
        if len(self.shape) != 3 or any(int(value) < 1 for value in self.shape):
            raise ValueError("recording shape must be positive TYX")
        if not self.uri.startswith(("repo://", "data://", "external://")):
            raise ValueError("recording URI must be portable")
        if self.pixel_size_um is not None and self.pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive when resolved")
        if self.frame_interval_s is not None and self.frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be positive when resolved")

    @property
    def frame_count(self) -> int:
        return int(self.shape[0])

    @property
    def frame_shape(self) -> tuple[int, int]:
        return int(self.shape[1]), int(self.shape[2])

    def to_manifest(self) -> dict[str, Any]:
        """Serialize provenance without leaking a workstation path."""
        return {
            "recording_id": self.recording_id,
            "dataset_id": self.dataset_id,
            "recording_number": self.recording_number,
            "behavior": self.behavior,
            "uri": self.uri,
            "sha256": self.sha256,
            "shape_tyx": list(self.shape),
            "dtype": self.dtype,
            "pixel_size_um": self.pixel_size_um,
            "frame_interval_s": self.frame_interval_s,
        }


@dataclass(frozen=True)
class RecordingInventory:
    dataset_id: str
    recordings: tuple[RecordingDescriptor, ...]
    descriptor_uri: str
    descriptor_sha256: str
    independence: str | None = None

    def __post_init__(self) -> None:
        if not self.recordings:
            raise ValueError("recording inventory cannot be empty")
        ids = [item.recording_id for item in self.recordings]
        uris = [item.uri for item in self.recordings]
        hashes = [item.sha256 for item in self.recordings]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate recording_id")
        if len(uris) != len(set(uris)):
            raise ValueError("duplicate recording URI")
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate recording SHA-256")
        if any(item.dataset_id != self.dataset_id for item in self.recordings):
            raise ValueError("recording dataset_id disagrees with inventory")
        object.__setattr__(self, "descriptor_sha256", _valid_sha256(self.descriptor_sha256))
        if not self.descriptor_uri.startswith(("repo://", "data://", "external://")):
            raise ValueError("descriptor URI must be portable")

    def by_id(self) -> dict[str, RecordingDescriptor]:
        return {item.recording_id: item for item in self.recordings}


@dataclass(frozen=True)
class RecordingSplit:
    protocol_id: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.train_ids or not self.validation_ids:
            raise ValueError("both train and validation recordings are required")
        if set(self.train_ids) & set(self.validation_ids):
            raise ValueError("recording split overlap")
        if len(self.train_ids) != len(set(self.train_ids)) or len(self.validation_ids) != len(set(self.validation_ids)):
            raise ValueError("recording split contains duplicate IDs")

    def to_manifest(self, recordings: Sequence[RecordingDescriptor]) -> dict[str, Any]:
        lookup = {item.recording_id: item for item in recordings}

        def behavior_counts(ids: Sequence[str]) -> dict[str, int]:
            return dict(sorted(Counter(lookup[item].behavior for item in ids).items()))

        payload = {
            "protocol_id": self.protocol_id,
            "unit": "recording",
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "train_behavior_counts": behavior_counts(self.train_ids),
            "validation_behavior_counts": behavior_counts(self.validation_ids),
        }
        payload["split_sha256"] = stable_hash(payload)
        return payload


@dataclass(frozen=True)
class TemporalClipContract:
    """A contiguous TYX clip with source frames protected on both sides."""

    clip_frames: int = 32
    guard_frames: int = 16
    frame_step: int = 1

    def __post_init__(self) -> None:
        if self.clip_frames < 2:
            raise ValueError("clip_frames must be at least two")
        if self.guard_frames < 0:
            raise ValueError("guard_frames must be nonnegative")
        if self.frame_step != 1:
            raise ValueError("JEPA source clips must use contiguous frames (frame_step=1)")

    def valid_start_bounds(self, frame_count: int) -> tuple[int, int]:
        """Return inclusive lower and exclusive upper bounds for clip starts."""
        lower = self.guard_frames
        upper = frame_count - self.guard_frames - self.clip_frames + 1
        if upper <= lower:
            raise ValueError("recording is too short for clip plus temporal guards")
        return lower, upper

    def validate(self, start: int, stop: int, frame_count: int) -> None:
        if stop - start != self.clip_frames:
            raise ValueError("temporal clip must have exactly clip_frames contiguous frames")
        lower, upper = self.valid_start_bounds(frame_count)
        if start < lower or start >= upper:
            raise ValueError("temporal clip violates protected guard frames")

    def validate_indices(self, indices: Sequence[int], frame_count: int) -> None:
        values = np.asarray(indices, dtype=np.int64)
        if len(values) != self.clip_frames or np.any(np.diff(values) != 1):
            raise ValueError("temporal frame indices must be one contiguous block")
        self.validate(int(values[0]), int(values[-1]) + 1, frame_count)


@dataclass(frozen=True)
class ClipRequest:
    recording_id: str
    split: str
    start_frame_zero: int
    stop_frame_zero_exclusive: int
    y_zero: int
    x_zero: int
    height: int
    width: int
    sampling_stratum: str
    temporal_mad: float

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation"}:
            raise ValueError("clip split must be train or validation")
        if self.sampling_stratum not in {"uniform", "high_mad", "low_mad"}:
            raise ValueError("unknown crop sampling stratum")
        if self.stop_frame_zero_exclusive <= self.start_frame_zero:
            raise ValueError("invalid clip interval")
        if min(self.x_zero, self.y_zero) < 0 or min(self.height, self.width) < 1:
            raise ValueError("invalid spatial crop")
        if not np.isfinite(self.temporal_mad) or self.temporal_mad < 0:
            raise ValueError("temporal_mad must be finite and nonnegative")

    def key(self) -> tuple[str, int, int, int, int, int, int]:
        return (
            self.recording_id,
            self.start_frame_zero,
            self.stop_frame_zero_exclusive,
            self.y_zero,
            self.x_zero,
            self.height,
            self.width,
        )

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustNormalization:
    center: float
    scale: float
    unscaled_mad: float
    scale_was_floored: bool
    fit_recording_ids: tuple[str, ...]
    uniform_frame_indices: tuple[tuple[str, tuple[int, ...]], ...]
    spatial_stride: int
    estimator: str = "global_training_median_and_normal_consistent_mad"

    def __post_init__(self) -> None:
        if not np.isfinite(self.center) or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("normalization center/scale must be finite with positive scale")
        if self.spatial_stride < 1:
            raise ValueError("spatial_stride must be positive")

    def apply(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - np.float32(self.center)) / np.float32(self.scale)).astype(
            np.float32,
            copy=False,
        )

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "center": self.center,
            "scale": self.scale,
            "unscaled_mad": self.unscaled_mad,
            "scale_was_floored": self.scale_was_floored,
            "fit_recording_ids": list(self.fit_recording_ids),
            "uniform_frame_indices": {key: list(values) for key, values in self.uniform_frame_indices},
            "spatial_stride": self.spatial_stride,
            "estimator": self.estimator,
            "fit_scope": "training_recordings_only",
        }
        payload["normalization_sha256"] = stable_hash(payload)
        return payload


def load_recording_inventory(
    descriptor_path: Path,
    *,
    repository_root: Path,
    data_root: Path | None = None,
    dataset_id: str = "060126",
    verify_live: bool = False,
    verify_hashes: bool = False,
) -> RecordingInventory:
    """Read legacy or portable descriptors without serializing absolute paths."""
    descriptor_path = descriptor_path.expanduser().resolve()
    repository_root = repository_root.expanduser().resolve()
    data_root = (data_root or repository_root).expanduser().resolve()
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    rows = payload.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("descriptor must contain a nonempty recordings list")

    recordings: list[RecordingDescriptor] = []
    payload_dataset_id = str(payload.get("dataset_id", dataset_id))
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each recording descriptor must be an object")
        uri, resolved_path = _portable_uri_from_mapping(
            row,
            repository_root=repository_root,
            data_root=data_root,
        )
        name = resolved_path.name or Path(uri).name
        behavior_value = row["behavior"] if "behavior" in row else _infer_behavior(name)
        number_value = row["recording_number"] if "recording_number" in row else _infer_recording_number(name)
        behavior = _canonical_behavior(str(behavior_value))
        number = int(number_value)
        recording_id = str(row.get("recording_id", f"{payload_dataset_id}_{number:02d}_{behavior}"))
        raw_shape = row.get("shape_tyx", row.get("shape"))
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
            raise ValueError(f"recording {recording_id} requires TYX shape")
        recordings.append(
            RecordingDescriptor(
                recording_id=recording_id,
                dataset_id=str(row.get("dataset_id", payload_dataset_id)),
                recording_number=number,
                behavior=behavior,
                uri=uri,
                resolved_path=resolved_path,
                sha256=_valid_sha256(row.get("sha256")),
                shape=tuple(int(value) for value in raw_shape),
                dtype=str(row.get("dtype", "")),
                pixel_size_um=(None if row.get("pixel_size_um") is None else float(row["pixel_size_um"])),
                frame_interval_s=(
                    None if row.get("frame_interval_s") is None else float(row["frame_interval_s"])
                ),
            )
        )

    inventory = RecordingInventory(
        dataset_id=payload_dataset_id,
        recordings=tuple(sorted(recordings, key=lambda item: item.recording_number)),
        descriptor_uri=portable_path(descriptor_path, repository=repository_root, data=data_root),
        descriptor_sha256=sha256_file(descriptor_path),
        independence=(None if payload.get("independence") is None else str(payload["independence"])),
    )
    assert_spon_excluded(inventory.recordings)
    if verify_live:
        for item in inventory.recordings:
            open_recording_memmap(item)
            if verify_hashes and sha256_file(item.resolved_path) != item.sha256:
                raise ValueError(f"recording hash mismatch: {item.recording_id}")
    elif verify_hashes:
        raise ValueError("verify_hashes requires verify_live=True")
    return inventory


def assert_spon_excluded(recordings: Iterable[RecordingDescriptor]) -> None:
    """Fail closed if the annotated Spon source enters unlabeled pretraining."""
    for item in recordings:
        source = " ".join((item.dataset_id, item.recording_id, item.uri, item.resolved_path.name)).casefold()
        if item.dataset_id.casefold() != "060126" or any(marker in source for marker in _SPON_MARKERS):
            raise ValueError(f"Spon source is forbidden from JEPA pretraining: {item.recording_id}")


def _metadata_resolution(recordings: Sequence[RecordingDescriptor], field: str, unit: str) -> dict[str, Any]:
    values = [getattr(item, field) for item in recordings]
    resolved = [float(value) for value in values if value is not None]
    if not resolved:
        status = "unresolved"
    elif len(resolved) == len(values):
        status = "resolved"
    else:
        status = "partially_resolved"
    return {
        "status": status,
        "unit": unit,
        "resolved_values": sorted(set(resolved)),
        "recordings_missing": [item.recording_id for item, value in zip(recordings, values) if value is None],
        "inference_policy": "do_not_infer_from_filename_or_other_recording",
    }


def validate_060126_inventory(
    inventory: RecordingInventory,
    *,
    verify_live: bool = False,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Validate the frozen 11-recording corpus and optionally its live TIFFs."""
    recordings = inventory.recordings
    assert_spon_excluded(recordings)
    checks = {
        "dataset_id": inventory.dataset_id == "060126",
        "recording_count": len(recordings) == EXPECTED_060126_RECORDING_COUNT,
        "total_frames": sum(item.frame_count for item in recordings) == EXPECTED_060126_TOTAL_FRAMES,
        "frame_shape": all(item.frame_shape == EXPECTED_060126_FRAME_SHAPE for item in recordings),
        "dtype": all(item.dtype == EXPECTED_060126_DTYPE for item in recordings),
        "behavior_counts": Counter(item.behavior for item in recordings) == EXPECTED_060126_BEHAVIOR_COUNTS,
    }
    live_checks: dict[str, Any] = {"performed": verify_live, "hashes_verified": verify_hashes}
    if verify_hashes and not verify_live:
        raise ValueError("verify_hashes requires verify_live=True")
    if verify_live:
        rows = []
        for item in recordings:
            movie = open_recording_memmap(item)
            hash_matches = None
            if verify_hashes:
                hash_matches = sha256_file(item.resolved_path) == item.sha256
                if not hash_matches:
                    raise ValueError(f"recording hash mismatch: {item.recording_id}")
            rows.append(
                {
                    "recording_id": item.recording_id,
                    "shape_matches": tuple(movie.shape) == item.shape,
                    "dtype_matches": movie.dtype.name == item.dtype,
                    "hash_matches": hash_matches,
                }
            )
        live_checks["recordings"] = rows
        checks["live_descriptors"] = all(row["shape_matches"] and row["dtype_matches"] for row in rows)
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise ValueError(f"invalid frozen 060126 inventory: {', '.join(failed)}")
    result = {
        "status": "passed",
        "checks": checks,
        "observed": {
            "recording_count": len(recordings),
            "total_frames": sum(item.frame_count for item in recordings),
            "frame_shape_yx": list(EXPECTED_060126_FRAME_SHAPE),
            "dtype": EXPECTED_060126_DTYPE,
            "behavior_counts": dict(sorted(Counter(item.behavior for item in recordings).items())),
        },
        "physical_metadata": {
            "pixel_size_um": _metadata_resolution(recordings, "pixel_size_um", "um_per_pixel"),
            "frame_interval_s": _metadata_resolution(recordings, "frame_interval_s", "seconds_per_frame"),
        },
        "live_validation": live_checks,
    }
    result["validation_sha256"] = stable_hash(result)
    return result


def fixed_behavior_stratified_split(recordings: Sequence[RecordingDescriptor]) -> RecordingSplit:
    """Return the frozen 8/3 recording split, one validation trial per behavior."""
    assert_spon_excluded(recordings)
    selected: dict[str, str] = {}
    for item in recordings:
        if item.recording_number == FIXED_060126_VALIDATION_NUMBERS.get(item.behavior):
            if item.behavior in selected:
                raise ValueError(f"multiple frozen validation recordings for {item.behavior}")
            selected[item.behavior] = item.recording_id
    if set(selected) != set(FIXED_060126_VALIDATION_NUMBERS):
        raise ValueError("frozen validation recording is missing for one or more behaviors")
    validation = tuple(sorted(selected.values()))
    train = tuple(sorted(item.recording_id for item in recordings if item.recording_id not in validation))
    split = RecordingSplit(FIXED_060126_SPLIT_ID, train, validation)
    if len(train) != 8 or len(validation) != 3:
        raise ValueError("frozen 060126 split must contain exactly 8 train and 3 validation recordings")
    manifest = split.to_manifest(recordings)
    if manifest["train_behavior_counts"] != {"left": 3, "rest": 3, "right": 2}:
        raise ValueError("training split is not behavior-stratified")
    if manifest["validation_behavior_counts"] != {"left": 1, "rest": 1, "right": 1}:
        raise ValueError("validation split is not behavior-stratified")
    return split


def open_recording_memmap(descriptor: RecordingDescriptor) -> np.memmap:
    """Open a TIFF read-only, refusing an in-memory or copied fallback."""
    if not descriptor.resolved_path.is_file():
        raise FileNotFoundError(descriptor.resolved_path)
    try:
        movie = tifffile.memmap(descriptor.resolved_path, mode="r")
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"recording is not TIFF-memmap compatible; raw-video copies are forbidden: {descriptor.recording_id}"
        ) from exc
    if tuple(movie.shape) != descriptor.shape:
        raise ValueError(f"live shape mismatch for {descriptor.recording_id}: {tuple(movie.shape)}")
    if movie.dtype.name != descriptor.dtype:
        raise ValueError(f"live dtype mismatch for {descriptor.recording_id}: {movie.dtype}")
    if movie.flags.writeable:
        raise ValueError("JEPA raw recording memmap must be read-only")
    return movie


def read_sequential_clip(
    descriptor: RecordingDescriptor,
    request: ClipRequest,
    contract: TemporalClipContract,
    *,
    normalization: RobustNormalization | None = None,
) -> np.ndarray:
    """Read one contiguous TYX crop from a read-only TIFF memmap."""
    if descriptor.recording_id != request.recording_id:
        raise ValueError("clip request recording_id mismatch")
    contract.validate(request.start_frame_zero, request.stop_frame_zero_exclusive, descriptor.frame_count)
    if request.y_zero + request.height > descriptor.frame_shape[0]:
        raise ValueError("clip exceeds recording height")
    if request.x_zero + request.width > descriptor.frame_shape[1]:
        raise ValueError("clip exceeds recording width")
    movie = open_recording_memmap(descriptor)
    clip = movie[
        request.start_frame_zero : request.stop_frame_zero_exclusive,
        request.y_zero : request.y_zero + request.height,
        request.x_zero : request.x_zero + request.width,
    ]
    expected = (contract.clip_frames, request.height, request.width)
    if tuple(clip.shape) != expected:
        raise RuntimeError(f"sequential clip shape mismatch: expected {expected}, observed {tuple(clip.shape)}")
    return clip if normalization is None else normalization.apply(clip)


def deterministic_uniform_frame_indices(frame_count: int, count: int) -> tuple[int, ...]:
    if frame_count < 1 or count < 1:
        raise ValueError("frame_count and count must be positive")
    count = min(frame_count, count)
    # Integer arithmetic avoids platform-dependent floating point rounding.
    if count == 1:
        return (frame_count // 2,)
    values = [(index * (frame_count - 1)) // (count - 1) for index in range(count)]
    return tuple(dict.fromkeys(values))


def fit_training_robust_normalization(
    recordings: Sequence[RecordingDescriptor],
    split: RecordingSplit,
    *,
    frames_per_recording: int = 8,
    spatial_stride: int = 8,
    minimum_scale: float = 1e-6,
) -> RobustNormalization:
    """Fit a global median/MAD using uniform frames from training recordings only."""
    if frames_per_recording < 1 or spatial_stride < 1 or minimum_scale <= 0:
        raise ValueError("invalid normalization sampling parameters")
    lookup = {item.recording_id: item for item in recordings}
    unknown = (set(split.train_ids) | set(split.validation_ids)) - set(lookup)
    if unknown:
        raise ValueError(f"split references unknown recording IDs: {sorted(unknown)}")
    if set(split.train_ids) & set(split.validation_ids):
        raise ValueError("normalization split overlap")

    samples: list[np.ndarray] = []
    used: list[tuple[str, tuple[int, ...]]] = []
    for recording_id in sorted(split.train_ids):
        descriptor = lookup[recording_id]
        movie = open_recording_memmap(descriptor)
        indices = deterministic_uniform_frame_indices(descriptor.frame_count, frames_per_recording)
        used.append((recording_id, indices))
        for frame_index in indices:
            samples.append(np.asarray(movie[frame_index, ::spatial_stride, ::spatial_stride], dtype=np.float32).reshape(-1))
    if not samples:
        raise ValueError("normalization requires training samples")
    pooled = np.concatenate(samples)
    center = float(np.median(pooled))
    unscaled_mad = float(np.median(np.abs(pooled - center)))
    raw_scale = 1.4826 * unscaled_mad
    scale = max(raw_scale, minimum_scale)
    return RobustNormalization(
        center=center,
        scale=scale,
        unscaled_mad=unscaled_mad,
        scale_was_floored=raw_scale < minimum_scale,
        fit_recording_ids=tuple(sorted(split.train_ids)),
        uniform_frame_indices=tuple(used),
        spatial_stride=spatial_stride,
    )


def _candidate_request(
    descriptor: RecordingDescriptor,
    *,
    rng: np.random.Generator,
    contract: TemporalClipContract,
    crop_shape: tuple[int, int],
    stratum: str,
    split_name: str = "train",
) -> ClipRequest:
    crop_height, crop_width = crop_shape
    if crop_height > descriptor.frame_shape[0] or crop_width > descriptor.frame_shape[1]:
        raise ValueError("crop shape exceeds recording frame")
    lower, upper = contract.valid_start_bounds(descriptor.frame_count)
    start = int(rng.integers(lower, upper))
    y_zero = int(rng.integers(0, descriptor.frame_shape[0] - crop_height + 1))
    x_zero = int(rng.integers(0, descriptor.frame_shape[1] - crop_width + 1))
    return ClipRequest(
        recording_id=descriptor.recording_id,
        split=split_name,
        start_frame_zero=start,
        stop_frame_zero_exclusive=start + contract.clip_frames,
        y_zero=y_zero,
        x_zero=x_zero,
        height=crop_height,
        width=crop_width,
        sampling_stratum=stratum,
        temporal_mad=0.0,
    )


def _score_temporal_mad(
    movie: np.memmap,
    request: ClipRequest,
    *,
    temporal_stride: int,
    spatial_stride: int,
) -> float:
    sample = np.asarray(
        movie[
            request.start_frame_zero : request.stop_frame_zero_exclusive : temporal_stride,
            request.y_zero : request.y_zero + request.height : spatial_stride,
            request.x_zero : request.x_zero + request.width : spatial_stride,
        ],
        dtype=np.float32,
    )
    per_pixel_center = np.median(sample, axis=0, keepdims=True)
    return float(1.4826 * np.median(np.abs(sample - per_pixel_center)))


def _with_score(request: ClipRequest, score: float, stratum: str | None = None) -> ClipRequest:
    values = request.to_manifest()
    values["temporal_mad"] = score
    if stratum is not None:
        values["sampling_stratum"] = stratum
    return ClipRequest(**values)


def sample_training_crop_plan(
    recordings: Sequence[RecordingDescriptor],
    split: RecordingSplit,
    *,
    clip_count: int,
    seed: int,
    contract: TemporalClipContract = TemporalClipContract(),
    crop_shape: tuple[int, int] = (64, 64),
    candidate_multiplier: int = 8,
    score_temporal_stride: int = 2,
    score_spatial_stride: int = 4,
) -> tuple[ClipRequest, ...]:
    """Sample 50/25/25 uniform/high-MAD/low-MAD training crops label-free."""
    if clip_count < 4 or clip_count % 4:
        raise ValueError("clip_count must be a positive multiple of four for an exact 50/25/25 plan")
    if candidate_multiplier < 2 or min(score_temporal_stride, score_spatial_stride) < 1:
        raise ValueError("invalid candidate scoring parameters")
    if len(crop_shape) != 2 or min(crop_shape) < 1:
        raise ValueError("crop_shape must contain two positive dimensions")

    lookup = {item.recording_id: item for item in recordings}
    missing = set(split.train_ids) - set(lookup)
    if missing:
        raise ValueError(f"training split references missing recordings: {sorted(missing)}")
    train = [lookup[item] for item in sorted(split.train_ids)]
    assert_spon_excluded(train)
    rng = np.random.default_rng(seed)
    movies = {item.recording_id: open_recording_memmap(item) for item in train}
    uniform_count = clip_count // 2
    tail_count = clip_count // 4

    used_keys: set[tuple[str, int, int, int, int, int, int]] = set()
    uniform: list[ClipRequest] = []
    attempt = 0
    while len(uniform) < uniform_count:
        descriptor = train[len(uniform) % len(train)]
        attempt += 1
        candidate = _candidate_request(
            descriptor,
            rng=rng,
            contract=contract,
            crop_shape=crop_shape,
            stratum="uniform",
        )
        if candidate.key() in used_keys:
            continue
        used_keys.add(candidate.key())
        score = _score_temporal_mad(
            movies[candidate.recording_id],
            candidate,
            temporal_stride=score_temporal_stride,
            spatial_stride=score_spatial_stride,
        )
        uniform.append(_with_score(candidate, score))

    bank_size = max(clip_count * candidate_multiplier, 4 * len(train))
    bank: list[ClipRequest] = []
    attempts = 0
    maximum_attempts = bank_size * 100
    while len(bank) < bank_size and attempts < maximum_attempts:
        descriptor = train[len(bank) % len(train)]
        attempts += 1
        candidate = _candidate_request(
            descriptor,
            rng=rng,
            contract=contract,
            crop_shape=crop_shape,
            stratum="uniform",
        )
        if candidate.key() in used_keys:
            continue
        used_keys.add(candidate.key())
        score = _score_temporal_mad(
            movies[candidate.recording_id],
            candidate,
            temporal_stride=score_temporal_stride,
            spatial_stride=score_spatial_stride,
        )
        bank.append(_with_score(candidate, score))
    if len(bank) < 2 * tail_count:
        raise ValueError("insufficient unique crops for the requested MAD strata")

    ranked = sorted(
        bank,
        key=lambda item: (item.temporal_mad, item.recording_id, item.start_frame_zero, item.y_zero, item.x_zero),
    )
    low = [_with_score(item, item.temporal_mad, "low_mad") for item in ranked[:tail_count]]
    high = [_with_score(item, item.temporal_mad, "high_mad") for item in ranked[-tail_count:]]
    requests = tuple(uniform + high + low)
    if Counter(item.sampling_stratum for item in requests) != {
        "uniform": uniform_count,
        "high_mad": tail_count,
        "low_mad": tail_count,
    }:
        raise RuntimeError("crop-stratum contract failed")
    return requests


def sample_validation_crop_plan(
    recordings: Sequence[RecordingDescriptor],
    split: RecordingSplit,
    *,
    clip_count: int,
    seed: int,
    contract: TemporalClipContract = TemporalClipContract(),
    crop_shape: tuple[int, int] = (64, 64),
) -> tuple[ClipRequest, ...]:
    """Sample fixed uniform coverage from validation recordings only.

    No validation-derived MAD threshold, quantile, label, ROI, or burst enters
    this plan.  Recording assignment is round-robin and coordinates are drawn
    from a frozen RNG seed, so the same request count has a stable identity.
    """
    if clip_count < 1:
        raise ValueError("clip_count must be positive")
    if len(crop_shape) != 2 or min(crop_shape) < 1:
        raise ValueError("crop_shape must contain two positive dimensions")
    lookup = {item.recording_id: item for item in recordings}
    missing = set(split.validation_ids) - set(lookup)
    if missing:
        raise ValueError(f"validation split references missing recordings: {sorted(missing)}")
    validation = [lookup[item] for item in sorted(split.validation_ids)]
    assert_spon_excluded(validation)
    rng = np.random.default_rng(seed)
    requests: list[ClipRequest] = []
    used: set[tuple[str, int, int, int, int, int, int]] = set()
    attempts = 0
    maximum_attempts = clip_count * 100
    while len(requests) < clip_count and attempts < maximum_attempts:
        descriptor = validation[len(requests) % len(validation)]
        attempts += 1
        candidate = _candidate_request(
            descriptor,
            rng=rng,
            contract=contract,
            crop_shape=crop_shape,
            stratum="uniform",
            split_name="validation",
        )
        if candidate.key() in used:
            continue
        used.add(candidate.key())
        requests.append(candidate)
    if len(requests) != clip_count:
        raise ValueError("insufficient unique validation crops")
    return tuple(requests)


def build_jepa_data_manifest(
    inventory: RecordingInventory,
    split: RecordingSplit,
    *,
    temporal_contract: TemporalClipContract,
    normalization: RobustNormalization | None = None,
    crop_plan: Sequence[ClipRequest] | None = None,
    validation_crop_plan: Sequence[ClipRequest] | None = None,
    validate_frozen_060126: bool = True,
) -> dict[str, Any]:
    """Build a deterministic, portable manifest with no raw-video copies."""
    assert_spon_excluded(inventory.recordings)
    inventory_validation = (
        validate_060126_inventory(inventory) if validate_frozen_060126 else {"status": "not_requested"}
    )
    split_manifest = split.to_manifest(inventory.recordings)
    if set(split.train_ids) | set(split.validation_ids) != {item.recording_id for item in inventory.recordings}:
        raise ValueError("recording split must cover the complete inventory exactly once")
    if normalization is not None and tuple(sorted(normalization.fit_recording_ids)) != tuple(sorted(split.train_ids)):
        raise ValueError("normalization was not fitted on exactly the training recordings")

    sampling: dict[str, Any] = {
        "label_access": "forbidden_by_data_api",
        "roi_access": "forbidden_by_data_api",
        "burst_access": "forbidden_by_data_api",
        "requested_ratio": {"uniform": 0.5, "high_mad": 0.25, "low_mad": 0.25},
    }
    if crop_plan is not None:
        requests = [item.to_manifest() for item in crop_plan]
        if any(item.split != "train" or item.recording_id not in split.train_ids for item in crop_plan):
            raise ValueError("pretraining crop plan may contain training recordings only")
        counts = Counter(item.sampling_stratum for item in crop_plan)
        if counts and not (counts["uniform"] == 2 * counts["high_mad"] == 2 * counts["low_mad"]):
            raise ValueError("crop plan violates the 50/25/25 sampling contract")
        sampling.update(
            {
                "observed_counts": dict(sorted(counts.items())),
                "requests": requests,
                "crop_plan_sha256": stable_hash(requests),
            }
        )
    if validation_crop_plan is not None:
        validation_requests = [item.to_manifest() for item in validation_crop_plan]
        if any(
            item.split != "validation"
            or item.recording_id not in split.validation_ids
            or item.sampling_stratum != "uniform"
            for item in validation_crop_plan
        ):
            raise ValueError("validation crop plan must be uniform and validation-recording only")
        sampling["validation_uniform"] = {
            "selection": "fixed_uniform_coverage_only",
            "validation_statistics_fit": False,
            "checkpoint_selection_role": "fixed_evaluation_view_only",
            "observed_count": len(validation_crop_plan),
            "requests": validation_requests,
            "crop_plan_sha256": stable_hash(validation_requests),
        }

    recordings_manifest = [item.to_manifest() for item in inventory.recordings]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": inventory.dataset_id,
        "axes": "TYX",
        "descriptor": {
            "uri": inventory.descriptor_uri,
            "sha256": inventory.descriptor_sha256,
            "independence": inventory.independence,
        },
        "recordings": recordings_manifest,
        "recording_inventory_sha256": stable_hash(recordings_manifest),
        "inventory_validation": inventory_validation,
        "physical_metadata": {
            "pixel_size_um": _metadata_resolution(inventory.recordings, "pixel_size_um", "um_per_pixel"),
            "frame_interval_s": _metadata_resolution(
                inventory.recordings,
                "frame_interval_s",
                "seconds_per_frame",
            ),
        },
        "split": split_manifest,
        "source_access": {
            "reader": "tifffile.memmap",
            "mode": "read_only_sequential_clips",
            "raw_video_copied": False,
            "fallback_copy_allowed": False,
        },
        "temporal_contract": asdict(temporal_contract),
        "normalization": None if normalization is None else normalization.to_manifest(),
        "sampling": sampling,
        "spon_pretraining_excluded": True,
    }
    payload["manifest_sha256"] = stable_hash(payload)
    return payload
