"""Memory-safe transfer experiments for dark-soma excitation zones."""

from .config import (
    CFARConfig,
    ConfigValidationError,
    DarkZoneConfig,
    DynamicsCheckpoint,
    ExperimentConfig,
    ResourceLimits,
    SomaExcitationConfig,
    SomaExcitationExperimentConfig,
    load_soma_excitation_config,
)
from .preflight import (
    PreflightError,
    ResourceBudgetError,
    available_ram_bytes,
    build_soma_excitation_preflight,
    preflight_soma_excitation,
    run_preflight,
)
from .zones import DarkSomaZone, DarkSomaZoneConfig, DarkSomaZones, detect_dark_soma_zones

__all__ = [
    "CFARConfig",
    "ConfigValidationError",
    "DarkZoneConfig",
    "DarkSomaZone",
    "DarkSomaZoneConfig",
    "DarkSomaZones",
    "DynamicsCheckpoint",
    "ExperimentConfig",
    "PreflightError",
    "ResourceBudgetError",
    "ResourceLimits",
    "SomaExcitationConfig",
    "SomaExcitationExperimentConfig",
    "available_ram_bytes",
    "build_soma_excitation_preflight",
    "detect_dark_soma_zones",
    "load_soma_excitation_config",
    "preflight_soma_excitation",
    "run_preflight",
]
