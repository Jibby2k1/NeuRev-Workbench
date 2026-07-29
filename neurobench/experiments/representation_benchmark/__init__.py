"""PCA, spatial ICA, autoencoder, and embedding benchmark."""

from .config import RepresentationBenchmarkConfig
from .preflight import preflight

__all__ = ["RepresentationBenchmarkConfig", "preflight"]
