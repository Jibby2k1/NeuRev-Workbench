"""Validated JSON configuration for the soma-excitation transfer experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MAX_CHUNK_FRAMES = 128
MAX_CPU_THREADS = 8
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ConfigValidationError(ValueError):
    """An unsafe or internally invalid experiment manifest."""


@dataclass(frozen=True)
class ResourceLimits:
    """Hard limits chosen to keep the workstation responsive."""
    device: str = "cpu"
    worker_count: int = 1
    chunk_frames: int = 32
    cpu_threads: int = 2
    max_ram_mib: int = 2048
    max_output_mib: int = 512

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResourceLimits":
        return _from_typed_dict(
            cls, payload, "resources",
            ints=("worker_count", "chunk_frames", "cpu_threads", "max_ram_mib", "max_output_mib"),
            strings=("device",),
        )

    def validate(self) -> None:
        if self.device.strip().lower() != "cpu":
            raise ConfigValidationError("resources.device must be 'cpu'; this experiment intentionally disables GPU use.")
        if self.worker_count != 1:
            raise ConfigValidationError("resources.worker_count must be 1 to prevent concurrent video/model memory spikes.")
        if not 1 <= self.chunk_frames <= MAX_CHUNK_FRAMES:
            raise ConfigValidationError(
                f"resources.chunk_frames must be between 1 and {MAX_CHUNK_FRAMES}; got {self.chunk_frames}."
            )
        if not 1 <= self.cpu_threads <= MAX_CPU_THREADS:
            raise ConfigValidationError(
                f"resources.cpu_threads must be between 1 and {MAX_CPU_THREADS}; got {self.cpu_threads}."
            )
        if self.max_ram_mib <= 0:
            raise ConfigValidationError("resources.max_ram_mib must be positive.")
        if self.max_output_mib <= 0:
            raise ConfigValidationError("resources.max_output_mib must be positive.")


@dataclass(frozen=True)
class CFARConfig:
    """CFAR guard/training geometry and threshold controls."""

    small_radius_px: int = 2
    large_radius_px: int = 11
    pfa: float = 0.001
    epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CFARConfig":
        return _from_typed_dict(
            cls, payload, "cfar", ints=("small_radius_px", "large_radius_px"),
            floats=("pfa", "epsilon"),
            aliases={"guard_px": "small_radius_px", "training_radius_px": "large_radius_px"},
        )

    def validate(self) -> None:
        if self.small_radius_px < 0:
            raise ConfigValidationError("cfar.small_radius_px must be non-negative.")
        if self.large_radius_px <= self.small_radius_px:
            raise ConfigValidationError("cfar.large_radius_px must be larger than cfar.small_radius_px.")
        if not 0.0 < self.pfa < 1.0:
            raise ConfigValidationError("cfar.pfa must be strictly between 0 and 1.")
        if self.epsilon <= 0.0:
            raise ConfigValidationError("cfar.epsilon must be positive.")


@dataclass(frozen=True)
class DarkZoneConfig:
    """JSON-facing parameters for dark soma cores and excitation annuli."""

    inner_sigma: float = 1.0
    outer_sigma: float = 5.0
    min_contrast_z: float = 3.0
    min_distance_px: float = 6.0
    border_px: int = 10
    max_zones: int = 300
    core_radius_px: float = 4.0
    ring_inner_radius_px: float = 4.0
    ring_outer_radius_px: float = 10.0
    saturation_threshold: float | None = None
    saturation_flag_fraction: float = 0.1
    saturation_penalty: float = 0.0
    robust_scale_epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "DarkZoneConfig":
        return _from_typed_dict(
            cls, payload, "dark_zones", ints=("border_px", "max_zones"),
            floats=("inner_sigma", "outer_sigma", "min_contrast_z", "min_distance_px", "core_radius_px",
                    "ring_inner_radius_px", "ring_outer_radius_px", "saturation_threshold",
                    "saturation_flag_fraction", "saturation_penalty", "robust_scale_epsilon"),
            aliases={"z_threshold": "min_contrast_z", "min_distance": "min_distance_px", "border": "border_px",
                     "core_radius": "core_radius_px", "ring_inner_radius": "ring_inner_radius_px",
                     "ring_outer_radius": "ring_outer_radius_px"},
        )

    def validate(self) -> None:
        self.to_zone_config()

    def as_zone_kwargs(self) -> dict[str, Any]:
        """Return the exact keyword contract consumed by ``DarkSomaZoneConfig``."""
        return {
            "inner_sigma": self.inner_sigma,
            "outer_sigma": self.outer_sigma,
            "z_threshold": self.min_contrast_z,
            "min_distance": self.min_distance_px,
            "border": self.border_px,
            "max_zones": self.max_zones,
            "core_radius": self.core_radius_px,
            "ring_inner_radius": self.ring_inner_radius_px,
            "ring_outer_radius": self.ring_outer_radius_px,
            "saturation_threshold": self.saturation_threshold,
            "saturation_flag_fraction": self.saturation_flag_fraction,
            "saturation_penalty": self.saturation_penalty,
            "robust_scale_epsilon": self.robust_scale_epsilon,
        }

    def to_zone_config(self):
        """Build and validate the numerical zone module's native config."""
        from .zones import DarkSomaZoneConfig
        config = DarkSomaZoneConfig(**self.as_zone_kwargs())
        try:
            config.validate()
        except ValueError as exc:
            raise ConfigValidationError(f"Invalid dark_zones configuration: {exc}") from exc
        return config


@dataclass(frozen=True)
class DynamicsCheckpoint:
    """One existing dynamics checkpoint evaluated sequentially."""
    path: str
    model_id: str = ""
    horizon_frames: int | None = None

    @classmethod
    def from_value(cls, value: str | Path | Mapping[str, Any], *, base_dir: Path) -> "DynamicsCheckpoint":
        if isinstance(value, Mapping):
            unknown = set(value) - {"path", "model_id", "horizon_frames"}
            if unknown:
                raise ConfigValidationError(
                    f"Unknown dynamics checkpoint fields: {', '.join(sorted(str(item) for item in unknown))}."
                )
            raw_path = value.get("path")
            model_id = str(value.get("model_id") or "")
            horizon = value.get("horizon_frames")
        else:
            raw_path = value
            model_id = ""
            horizon = None
        path = _resolve_path(raw_path, base_dir=base_dir, field_name="dynamics_checkpoints.path")
        if model_id and not _ID_PATTERN.fullmatch(model_id):
            raise ConfigValidationError(
                "dynamics_checkpoints.model_id may contain only letters, numbers, periods, underscores, and hyphens."
            )
        horizon = None if horizon is None else _as_int(horizon, "dynamics_checkpoints.horizon_frames")
        if horizon is not None and horizon <= 0:
            raise ConfigValidationError("dynamics_checkpoints.horizon_frames must be positive when provided.")
        return cls(path=str(path), model_id=model_id, horizon_frames=horizon)

    def to_dict(self) -> dict[str, Any]:
        payload = {"path": self.path}
        if self.model_id:
            payload["model_id"] = self.model_id
        if self.horizon_frames is not None:
            payload["horizon_frames"] = self.horizon_frames
        return payload


@dataclass(frozen=True)
class SomaExcitationConfig:
    """JSON config: UI frames are one-based; array bounds are zero-based/half-open."""

    source_video: str
    output_dir: str
    experiment_id: str = "soma_excitation_transfer_v1"
    schema_version: int = SCHEMA_VERSION
    onset_frame_ui: int = 1900
    control_preroll_frames: int = 100
    end_frame_ui: int | None = None
    frame_rate_hz: float = 50.0
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    cfar: CFARConfig = field(default_factory=CFARConfig)
    dark_zones: DarkZoneConfig = field(default_factory=DarkZoneConfig)
    dynamics_checkpoints: tuple[DynamicsCheckpoint, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, base_dir: str | Path | None = None) -> "SomaExcitationConfig":
        values = dict(payload)
        aliases = {
            "source": "source_video",
            "source_path": "source_video",
            "video_path": "source_video",
            "output_root": "output_dir",
            "model_checkpoints": "dynamics_checkpoints",
            "dynamics_model_checkpoints": "dynamics_checkpoints",
        }
        for alias, canonical in aliases.items():
            if alias in values:
                if canonical in values:
                    raise ConfigValidationError(f"Specify only one of '{canonical}' and its alias '{alias}'.")
                values[canonical] = values.pop(alias)
        _reject_unknown(values, cls, "config")
        root = Path(base_dir).expanduser() if base_dir is not None else Path.cwd()
        source = _resolve_path(values.get("source_video"), base_dir=root, field_name="source_video")
        output = _resolve_path(values.get("output_dir"), base_dir=root, field_name="output_dir")
        checkpoints_raw = values.get("dynamics_checkpoints") or []
        if isinstance(checkpoints_raw, (str, bytes, Path)) or not isinstance(checkpoints_raw, Sequence):
            raise ConfigValidationError("dynamics_checkpoints must be a JSON list.")
        checkpoints = tuple(DynamicsCheckpoint.from_value(item, base_dir=root) for item in checkpoints_raw)
        result = cls(
            schema_version=_as_int(values.get("schema_version", SCHEMA_VERSION), "schema_version"),
            experiment_id=str(values.get("experiment_id", "soma_excitation_transfer_v1")),
            source_video=str(source),
            output_dir=str(output),
            onset_frame_ui=_as_int(values.get("onset_frame_ui", 1900), "onset_frame_ui"),
            control_preroll_frames=_as_int(values.get("control_preroll_frames", 100), "control_preroll_frames"),
            end_frame_ui=None if values.get("end_frame_ui") is None else _as_int(values["end_frame_ui"], "end_frame_ui"),
            frame_rate_hz=_as_float(values.get("frame_rate_hz", 50.0), "frame_rate_hz"),
            resources=ResourceLimits.from_dict(_mapping_or_none(values.get("resources"), "resources")),
            cfar=CFARConfig.from_dict(_mapping_or_none(values.get("cfar"), "cfar")),
            dark_zones=DarkZoneConfig.from_dict(_mapping_or_none(values.get("dark_zones"), "dark_zones")),
            dynamics_checkpoints=checkpoints,
        )
        result.validate()
        return result

    @classmethod
    def load_json(cls, path: str | Path) -> "SomaExcitationConfig":
        source = Path(path).expanduser()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(f"Invalid JSON config {source}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ConfigValidationError("The soma-excitation JSON config must contain one object.")
        return cls.from_dict(payload, base_dir=source.parent)

    from_json = load_json

    @property
    def onset_frame_zero(self) -> int:
        return self.onset_frame_ui - 1

    @property
    def onset_frame_zero_based(self) -> int:
        return self.onset_frame_zero

    @property
    def control_start_frame_zero(self) -> int:
        return self.onset_frame_zero - self.control_preroll_frames

    @property
    def dynamics_model_checkpoints(self) -> tuple[DynamicsCheckpoint, ...]:
        return self.dynamics_checkpoints

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigValidationError(
                f"Unsupported soma-excitation schema_version {self.schema_version}; expected {SCHEMA_VERSION}."
            )
        if not _ID_PATTERN.fullmatch(self.experiment_id):
            raise ConfigValidationError(
                "experiment_id must start with a letter or number and contain only letters, numbers, periods, underscores, and hyphens."
            )
        if not self.source_video.strip():
            raise ConfigValidationError("source_video is required.")
        if not self.output_dir.strip():
            raise ConfigValidationError("output_dir is required.")
        if self.onset_frame_ui < 1:
            raise ConfigValidationError("onset_frame_ui must be a positive one-based frame number.")
        if self.control_preroll_frames < 1:
            raise ConfigValidationError("control_preroll_frames must be positive.")
        if self.control_start_frame_zero < 0:
            raise ConfigValidationError(
                "onset_frame_ui does not leave enough source frames for the requested control_preroll_frames."
            )
        if self.end_frame_ui is not None and self.end_frame_ui < self.onset_frame_ui:
            raise ConfigValidationError("end_frame_ui must be at or after onset_frame_ui.")
        if self.frame_rate_hz <= 0.0:
            raise ConfigValidationError("frame_rate_hz must be positive.")
        self.resources.validate()
        self.cfar.validate()
        self.dark_zones.validate()
        checkpoint_paths = [item.path for item in self.dynamics_checkpoints]
        if len(set(checkpoint_paths)) != len(checkpoint_paths):
            raise ConfigValidationError("dynamics_checkpoints contains duplicate paths.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dynamics_checkpoints"] = [item.to_dict() for item in self.dynamics_checkpoints]
        return payload

    def write_json(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


SomaExcitationExperimentConfig = SomaExcitationConfig
ExperimentConfig = SomaExcitationConfig


def load_soma_excitation_config(path: str | Path) -> SomaExcitationConfig:
    """Load and validate a soma-excitation JSON manifest."""
    return SomaExcitationConfig.load_json(path)


def _mapping_or_none(value: Any, field_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{field_name} must be a JSON object.")
    return value


def _resolve_path(value: Any, *, base_dir: Path, field_name: str) -> Path:
    if value is None or not str(value).strip():
        raise ConfigValidationError(f"{field_name} is required.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigValidationError(f"{field_name} must be an integer, not a boolean.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{field_name} must be an integer; got {value!r}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ConfigValidationError(f"{field_name} must be an integer; got {value!r}.")
    return result


def _as_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigValidationError(f"{field_name} must be numeric, not a boolean.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{field_name} must be numeric; got {value!r}.") from exc


def _from_typed_dict(
    cls: type[Any], payload: Mapping[str, Any] | None, section: str, *,
    ints: Sequence[str] = (), floats: Sequence[str] = (), strings: Sequence[str] = (),
    aliases: Mapping[str, str] | None = None,
):
    values = dict(payload or {})
    for alias, canonical in (aliases or {}).items():
        if alias not in values:
            continue
        if canonical in values:
            raise ConfigValidationError(f"Specify only one of '{section}.{canonical}' and '{section}.{alias}'.")
        values[canonical] = values.pop(alias)
    _reject_unknown(values, cls, section)
    for name in set(ints) & values.keys():
        values[name] = _as_int(values[name], f"{section}.{name}")
    for name in set(floats) & values.keys():
        if values[name] is not None:
            values[name] = _as_float(values[name], f"{section}.{name}")
    for name in set(strings) & values.keys():
        values[name] = str(values[name])
    result = cls(**values)
    result.validate()
    return result


def _reject_unknown(values: Mapping[str, Any], cls: type[Any], section: str) -> None:
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(values) - known
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise ConfigValidationError(f"Unknown {section} field(s): {names}.")
