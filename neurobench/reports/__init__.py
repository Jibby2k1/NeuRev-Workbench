"""Report rendering helpers with lazy imports for optional dependencies."""

__all__ = [
    "build_metrics_report_from_pipeline_run",
    "build_metrics_report_from_pipeline_runs",
    "build_run_comparison_report",
    "build_sweep_evidence_report",
    "render_metrics_report_markdown",
    "render_run_comparison_markdown",
    "render_sweep_evidence_markdown",
    "write_metrics_report_markdown",
    "write_run_comparison_markdown",
    "write_sweep_evidence_report",
]


def __getattr__(name: str):
    if name in {"build_metrics_report_from_pipeline_run", "build_metrics_report_from_pipeline_runs"}:
        from neurobench.reports import builder as module
    elif name in {"build_run_comparison_report", "render_run_comparison_markdown", "write_run_comparison_markdown"}:
        from neurobench.reports import comparison as module
    elif name in {"render_metrics_report_markdown", "write_metrics_report_markdown"}:
        from neurobench.reports import render as module
    elif name in {"build_sweep_evidence_report", "render_sweep_evidence_markdown", "write_sweep_evidence_report"}:
        from neurobench.reports import sweep_evidence as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
