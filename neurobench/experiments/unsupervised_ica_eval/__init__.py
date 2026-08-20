"""Label-free two-frame ICA identification and external trace utilities."""

from .core import (
    align_components,
    analytic_baselines,
    distribution_summary,
    fit_two_frame_ica,
    representation_fingerprint,
    stability_summary,
)
from .traces import candidate_signature_summary, paired_trace_metrics

__all__ = [
    "align_components", "analytic_baselines", "distribution_summary",
    "fit_two_frame_ica", "representation_fingerprint", "stability_summary",
    "candidate_signature_summary", "paired_trace_metrics",
]
