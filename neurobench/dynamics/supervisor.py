"""Health reports and recovery helpers for long dynamics sweeps."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


def build_sweep_health_report(
    *,
    sweep_dir: str | Path,
    out_path: str | Path | None = None,
    include_archives: bool = True,
    stale_minutes: float = 60.0,
) -> dict[str, Any]:
    """Analyze sweep progress and write a Markdown health report."""
    sweep = Path(sweep_dir)
    manifest = _load_json_if_exists(sweep / "sweep_manifest.json")
    current_rows = _load_progress(sweep / "sweep_progress.jsonl")
    active_status = _load_json_if_exists(sweep / "sweep_active.json")
    archive_paths = _archive_progress_paths(sweep) if include_archives else []
    archive_rows = {path.name: _load_progress(path) for path in archive_paths}
    spec_index = _manifest_spec_index(manifest)
    summary = _summarize_progress(
        sweep_dir=sweep,
        manifest=manifest,
        current_rows=current_rows,
        archive_rows=archive_rows,
        spec_index=spec_index,
        stale_minutes=stale_minutes,
        active_status=active_status,
    )
    report_path = Path(out_path) if out_path is not None else sweep / "sweep_health_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_sweep_health_markdown(summary), encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


def create_resume_script(
    *,
    sweep_dir: str | Path,
    batch_size: int,
    script_path: str | Path,
    log_path: str | Path | None = None,
    python_executable: str = ".venv-neurobench/bin/python",
    device: str | None = None,
    time_limit_hours: float | None = None,
) -> Path:
    """Create a detached-safe resume script without modifying sweep artifacts."""
    sweep = Path(sweep_dir)
    manifest = _load_json_if_exists(sweep / "sweep_manifest.json")
    profile = str(manifest.get("profile") or "overnight")
    epochs = int(manifest.get("epochs") or 50)
    seeds = ",".join(str(seed) for seed in manifest.get("seeds", [7, 13]))
    device_arg = str(device or manifest.get("device") or "cuda")
    limit = time_limit_hours if time_limit_hours is not None else manifest.get("time_limit_hours")
    script = Path(script_path)
    log = Path(log_path) if log_path is not None else script.with_suffix(".log")
    script.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        "-m",
        "neurobench.dynamics.overnight_sweep",
        "--profile",
        profile,
        "--out-dir",
        str(sweep),
        "--device",
        device_arg,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(int(batch_size)),
        "--seeds",
        seeds,
    ]
    if limit is not None:
        command.extend(["--time-limit-hours", str(limit)])
    body = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "export PYTHONUNBUFFERED=1",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "",
            f"LOG={json.dumps(str(log))}",
            f"setsid {' '.join(_shell_quote(part) for part in command)} > \"$LOG\" 2>&1 < /dev/null &",
            "echo $!",
            "",
        ]
    )
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def build_sweep_live_status(
    *,
    sweep_dir: str | Path,
    out_path: str | Path | None = None,
    pid: int | None = None,
    include_gpu: bool = True,
) -> dict[str, Any]:
    """Write a compact live status report for a currently running sweep."""
    sweep = Path(sweep_dir)
    manifest = _load_json_if_exists(sweep / "sweep_manifest.json")
    current_rows = _load_progress(sweep / "sweep_progress.jsonl")
    active_status = _load_json_if_exists(sweep / "sweep_active.json")
    current_index = int(current_rows[-1].get("index") if current_rows else 0)
    experiment_count = int(manifest.get("experiment_count") or (current_rows[-1].get("experiment_count") if current_rows else 0) or 0)
    inferred = _infer_next_spec(manifest, current_index)
    active_exp_id = str(active_status.get("experiment_id") or inferred.get("experiment_id") or "")
    active_artifacts = _active_artifact_status(sweep, active_exp_id)
    process = _process_status(sweep, pid=pid)
    gpu = _gpu_status(process.get("pid"), include_gpu=include_gpu)
    live_state = _classify_live_state(
        active_status=active_status,
        process=process,
        gpu=gpu,
        active_artifacts=active_artifacts,
        current_rows=current_rows,
    )
    status = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep_dir": str(sweep),
        "profile": manifest.get("profile", ""),
        "experiment_count": experiment_count,
        "progress_index": current_index,
        "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in current_rows)),
        "active_status": dict(active_status or {}),
        "inferred_next_spec": inferred if not active_status else {},
        "process": process,
        "gpu": gpu,
        "active_artifacts": active_artifacts,
        "live_state": live_state,
        "recent_records": [dict(row) for row in current_rows[-5:]],
    }
    report_path = Path(out_path) if out_path is not None else sweep / "sweep_live_status.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_sweep_live_status_markdown(status), encoding="utf-8")
    status["report_path"] = str(report_path)
    return status


def render_sweep_live_status_markdown(status: Mapping[str, Any]) -> str:
    """Render a compact live sweep status report."""
    active = status.get("active_status") or status.get("inferred_next_spec") or {}
    process = status.get("process") or {}
    gpu = status.get("gpu") or {}
    artifacts = status.get("active_artifacts") or {}
    lines = [
        "# Sweep Live Status",
        "",
        f"Generated: `{status.get('generated_at')}`",
        f"Sweep directory: `{status.get('sweep_dir')}`",
        f"Profile: `{status.get('profile', '')}`",
        f"Progress: `{status.get('progress_index')}` / `{status.get('experiment_count')}`",
        f"Live state: `{status.get('live_state')}`",
        "",
        "## Active Spec",
        "",
        _table(
            ["Field", "Value"],
            [
                ["status", active.get("status")],
                ["index", active.get("index")],
                ["experiment", active.get("experiment_id")],
                ["dataset", active.get("dataset_key")],
                ["kind", active.get("kind")],
            ],
        ),
        "",
        "## Runtime",
        "",
        _table(
            ["Field", "Value"],
            [
                ["pid", process.get("pid")],
                ["stat", process.get("stat")],
                ["elapsed", process.get("elapsed")],
                ["cpu_percent", process.get("cpu_percent")],
                ["mem_percent", process.get("mem_percent")],
                ["gpu_utilization_percent", gpu.get("utilization_gpu_percent")],
                ["gpu_memory_used_mib", gpu.get("process_used_memory_mib")],
            ],
        ),
        "",
        "## Active Artifacts",
        "",
        _table(
            ["Field", "Value"],
            [
                ["experiment_dir", artifacts.get("experiment_dir")],
                ["exists", artifacts.get("exists")],
                ["metrics_present", artifacts.get("metrics_present")],
                ["file_count", artifacts.get("file_count")],
            ],
        ),
        "",
        "## Recent Records",
        "",
        _table(
            ["Index", "Status", "Experiment"],
            [[row.get("index"), row.get("status"), row.get("experiment_id")] for row in status.get("recent_records", [])],
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_sweep_health_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sweep Health Report",
        "",
        f"Generated: `{summary.get('generated_at')}`",
        f"Sweep directory: `{summary.get('sweep_dir')}`",
        f"Profile: `{summary.get('profile', '')}`",
        f"Progress: `{summary.get('current_index')}` / `{summary.get('experiment_count')}`",
        f"Completion: `{summary.get('completion_fraction', 0.0):.1%}`",
        "",
        "## Current Progress",
        "",
        _table(
            ["Status", "Count"],
            [[key, value] for key, value in sorted(summary.get("status_counts", {}).items())],
        ),
    ]
    active_status = summary.get("active_status") or {}
    if active_status:
        lines.extend(
            [
                "",
                "## Active Spec",
                "",
                _table(
                    ["Field", "Value"],
                    [
                        ["status", active_status.get("status")],
                        ["index", active_status.get("index")],
                        ["experiment", active_status.get("experiment_id")],
                        ["dataset", active_status.get("dataset_key")],
                        ["updated_at", active_status.get("updated_at")],
                    ],
                ),
            ]
        )
    inferred = summary.get("inferred_next_spec") or {}
    if inferred and not active_status:
        lines.extend(
            [
                "",
                "## Inferred Next Spec",
                "",
                "No `sweep_active.json` was found. This is inferred from the latest progress row and manifest order.",
                "",
                _table(
                    ["Field", "Value"],
                    [
                        ["index", inferred.get("index")],
                        ["experiment", inferred.get("experiment_id")],
                        ["kind", inferred.get("kind")],
                        ["dataset", inferred.get("dataset_key")],
                        ["seed", inferred.get("seed")],
                        ["summary", inferred.get("hyperparameter_summary")],
                    ],
                ),
            ]
        )
    lines.extend([
        "",
        "## Health",
        "",
    ])
    for item in summary.get("health_flags", []):
        lines.append(f"- {item}")
    if not summary.get("health_flags"):
        lines.append("- No active health flags detected.")
    lines.extend(["", "## Recommendations", ""])
    for item in summary.get("recommendations", []):
        lines.append(f"- {item}")
    if not summary.get("recommendations"):
        lines.append("- No recovery action recommended from current evidence.")
    lines.extend(["", "## Current Failures", ""])
    lines.append(_failure_section(summary.get("current_failure_summary", {})))
    if summary.get("archive_failure_summary"):
        lines.extend(["", "## Archived Failure Evidence", ""])
        for archive_name, archive_summary in summary.get("archive_failure_summary", {}).items():
            lines.append(f"### `{archive_name}`")
            lines.append("")
            lines.append(_failure_section(archive_summary))
            lines.append("")
    recent = summary.get("recent_records", [])
    lines.extend(["", "## Recent Records", ""])
    lines.append(_table(["Index", "Status", "Experiment"], [[r.get("index"), r.get("status"), r.get("experiment_id")] for r in recent]))
    return "\n".join(lines).rstrip() + "\n"


def _summarize_progress(
    *,
    sweep_dir: Path,
    manifest: Mapping[str, Any],
    current_rows: Sequence[Mapping[str, Any]],
    archive_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    spec_index: Mapping[str, Mapping[str, Any]],
    stale_minutes: float,
    active_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    experiment_count = int(manifest.get("experiment_count") or (current_rows[-1].get("experiment_count") if current_rows else 0) or 0)
    current_index = int(current_rows[-1].get("index") if current_rows else 0)
    status_counts = Counter(str(row.get("status", "unknown")) for row in current_rows)
    current_failure_summary = _failure_summary(current_rows, spec_index)
    archive_failure_summary = {name: _failure_summary(rows, spec_index) for name, rows in archive_rows.items()}
    progress_path = sweep_dir / "sweep_progress.jsonl"
    progress_age_min = None
    if progress_path.exists():
        progress_age_min = (datetime.now(timezone.utc).timestamp() - progress_path.stat().st_mtime) / 60.0
    health_flags: list[str] = []
    if current_failure_summary.get("failure_count"):
        health_flags.append(f"Current progress contains {current_failure_summary['failure_count']} failures.")
    if current_failure_summary.get("failure_classes", {}).get("cuda_oom"):
        health_flags.append("Current progress contains CUDA OOM failures.")
    active_state = str((active_status or {}).get("status") or "")
    is_stopped_active_marker = active_state in {"completed", "failed", "stopped"}
    if progress_age_min is not None and progress_age_min > float(stale_minutes) and not is_stopped_active_marker:
        health_flags.append(f"Progress file is stale: last update was {progress_age_min:.1f} minutes ago.")
    if is_stopped_active_marker:
        health_flags.append(f"Sweep active marker is `{active_state}`; progress age reflects a stopped run.")
    if experiment_count and current_index >= experiment_count and status_counts.get("failed", 0) == 0:
        health_flags.append("Sweep progress reached the expected experiment count with no current failures.")
    recommendations = _recommendations(
        manifest,
        current_failure_summary,
        archive_failure_summary,
        progress_age_min,
        stale_minutes,
        active_status=active_status,
    )
    inferred_next_spec = _infer_next_spec(manifest, current_index)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep_dir": str(sweep_dir),
        "profile": manifest.get("profile", ""),
        "experiment_count": experiment_count,
        "current_index": current_index,
        "completion_fraction": (float(current_index) / float(experiment_count)) if experiment_count else 0.0,
        "status_counts": dict(status_counts),
        "progress_age_minutes": progress_age_min,
        "active_status": dict(active_status or {}),
        "inferred_next_spec": inferred_next_spec,
        "current_failure_summary": current_failure_summary,
        "archive_failure_summary": archive_failure_summary,
        "health_flags": health_flags,
        "recommendations": recommendations,
        "recent_records": [dict(row) for row in current_rows[-10:]],
    }


def _infer_next_spec(manifest: Mapping[str, Any], current_index: int) -> dict[str, Any]:
    experiments = manifest.get("experiments", []) or []
    if not isinstance(experiments, Sequence) or isinstance(experiments, (str, bytes)):
        return {}
    next_index = int(current_index) + 1
    if next_index < 1 or next_index > len(experiments):
        return {}
    spec = experiments[next_index - 1]
    if not isinstance(spec, Mapping):
        return {}
    params = spec.get("params", {})
    if not isinstance(params, Mapping):
        params = {}
    return {
        "index": next_index,
        "experiment_count": len(experiments),
        "experiment_id": spec.get("experiment_id"),
        "kind": spec.get("kind"),
        "dataset_key": spec.get("dataset_key"),
        "seed": spec.get("seed"),
        "hyperparameter_summary": params.get("hyperparameter_summary"),
    }


def _failure_summary(rows: Sequence[Mapping[str, Any]], spec_index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if row.get("status") == "failed"]
    by_class: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()
    by_architecture: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    first = None
    last = None
    for row in failures:
        exp_id = str(row.get("experiment_id", ""))
        spec = spec_index.get(exp_id, {})
        params = spec.get("params", {}) if isinstance(spec.get("params"), Mapping) else {}
        cls = classify_failure(row.get("error", ""))
        by_class[cls] += 1
        by_kind[str(row.get("kind") or spec.get("kind") or "unknown")] += 1
        by_dataset[str(row.get("dataset_key") or spec.get("dataset_key") or "unknown")] += 1
        architecture = str(params.get("architecture") or row.get("kind") or spec.get("kind") or "unknown")
        by_architecture[architecture] += 1
        examples.setdefault(cls, _failure_example(row))
        if first is None or int(row.get("index") or 10**12) < int(first.get("index") or 10**12):
            first = row
        if last is None or int(row.get("index") or -1) > int(last.get("index") or -1):
            last = row
    trailing_failure_count = 0
    for row in reversed(rows):
        if row.get("status") == "failed":
            trailing_failure_count += 1
        else:
            break
    successful_records_after_last_failure = 0
    last_failure_position = None
    for position, row in enumerate(rows):
        if row.get("status") == "failed":
            last_failure_position = position
    if last_failure_position is not None:
        successful_records_after_last_failure = sum(
            1
            for row in rows[last_failure_position + 1 :]
            if row.get("status") in {"completed", "skipped"}
        )
    return {
        "failure_count": len(failures),
        "failure_classes": dict(by_class),
        "by_kind": dict(by_kind),
        "by_dataset": dict(by_dataset),
        "by_architecture": dict(by_architecture),
        "first_failure": _failure_example(first) if first else None,
        "last_failure": _failure_example(last) if last else None,
        "trailing_failure_count": trailing_failure_count,
        "successful_records_after_last_failure": successful_records_after_last_failure,
        "examples": examples,
    }


def classify_failure(error: Any) -> str:
    text = str(error).lower()
    if "outofmemoryerror" in text or "cuda out of memory" in text or "out of memory" in text:
        return "cuda_oom"
    if "filenotfounderror" in text or "no such file" in text or "missing" in text:
        return "missing_artifact"
    if "nan" in text or "inf" in text or "overflow" in text:
        return "numeric_instability"
    if "shape" in text or "size mismatch" in text or "dimension" in text:
        return "shape_mismatch"
    return "other"


def _recommendations(
    manifest: Mapping[str, Any],
    current_failure_summary: Mapping[str, Any],
    archive_failure_summary: Mapping[str, Mapping[str, Any]],
    progress_age_min: float | None,
    stale_minutes: float,
    active_status: Mapping[str, Any] | None = None,
) -> list[str]:
    recommendations: list[str] = []
    batch_size = int(manifest.get("batch_size") or 0)
    current_ooms = int(current_failure_summary.get("failure_classes", {}).get("cuda_oom", 0))
    archived_ooms = sum(int(summary.get("failure_classes", {}).get("cuda_oom", 0)) for summary in archive_failure_summary.values())
    if current_ooms:
        safer = max(1, batch_size // 2) if batch_size else 1
        successful_after_last = int(current_failure_summary.get("successful_records_after_last_failure") or 0)
        trailing_failures = int(current_failure_summary.get("trailing_failure_count") or 0)
        if successful_after_last:
            recommendations.append(
                f"CUDA OOMs are present, but {successful_after_last} later progress-file records completed or skipped; keep batch_size={batch_size} for now and monitor whether the failure cluster expands."
            )
            recommendations.append(f"If the latest records start failing again, archive progress and resume with batch_size={safer}.")
        elif trailing_failures:
            recommendations.append(f"The latest {trailing_failures} record(s) failed with CUDA OOM; archive progress and resume with batch_size={safer}.")
        else:
            recommendations.append(f"CUDA OOMs are present in current progress; archive progress and resume with batch_size={safer}.")
    elif archived_ooms and batch_size:
        recommendations.append(f"Archived OOMs found, but current batch_size={batch_size} has no OOMs so far; keep current batch size unless new failures appear.")
    active_state = str((active_status or {}).get("status") or "")
    is_stopped_active_marker = active_state in {"completed", "failed", "stopped"}
    if progress_age_min is not None and progress_age_min > float(stale_minutes) and not is_stopped_active_marker:
        recommendations.append("Progress appears stale; check the process list and GPU utilization before taking recovery action.")
    if not recommendations:
        recommendations.append("Continue monitoring; no current failure pattern requires intervention.")
    return recommendations


def _failure_section(summary: Mapping[str, Any]) -> str:
    if not summary.get("failure_count"):
        return "No failures recorded."
    lines = [
        f"Failure count: `{summary.get('failure_count')}`",
        "",
        _table(["Class", "Count"], [[k, v] for k, v in sorted(summary.get("failure_classes", {}).items())]),
        "",
        "By model kind:",
        "",
        _table(["Kind", "Count"], [[k, v] for k, v in sorted(summary.get("by_kind", {}).items())]),
        "",
        "By dataset:",
        "",
        _table(["Dataset", "Count"], [[k, v] for k, v in sorted(summary.get("by_dataset", {}).items())]),
    ]
    first = summary.get("first_failure")
    if first:
        lines.extend(
            [
                "",
                "First failure:",
                "",
                f"- Index: `{first.get('index')}`",
                f"- Experiment: `{first.get('experiment_id')}`",
                f"- Error class: `{classify_failure(first.get('error', ''))}`",
                f"- Error: `{str(first.get('error', ''))[:500]}`",
            ]
        )
    return "\n".join(lines)


def _table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_None._"
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _manifest_spec_index(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index = {}
    for spec in manifest.get("experiments", []) or []:
        if isinstance(spec, Mapping):
            index[str(spec.get("experiment_id"))] = spec
    return index


def _load_progress(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _archive_progress_paths(sweep_dir: Path) -> list[Path]:
    return sorted(path for path in sweep_dir.glob("sweep_progress_*.jsonl") if path.name != "sweep_progress.jsonl")


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _failure_example(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "index": row.get("index"),
        "experiment_id": row.get("experiment_id"),
        "kind": row.get("kind"),
        "dataset_key": row.get("dataset_key"),
        "seed": row.get("seed"),
        "error": row.get("error"),
    }


def _active_artifact_status(sweep: Path, experiment_id: str) -> dict[str, Any]:
    if not experiment_id:
        return {"experiment_dir": None, "exists": False, "metrics_present": False, "file_count": 0, "files": []}
    exp_dir = sweep / experiment_id
    files = sorted(path for path in exp_dir.rglob("*") if path.is_file()) if exp_dir.exists() else []
    rel_files = [str(path.relative_to(exp_dir)) for path in files[:20]]
    metrics_present = any(path.name.endswith("metrics.json") for path in files)
    return {
        "experiment_dir": str(exp_dir),
        "exists": exp_dir.exists(),
        "metrics_present": metrics_present,
        "file_count": len(files),
        "files": rel_files,
    }


def _process_status(sweep: Path, *, pid: int | None = None) -> dict[str, Any]:
    if pid is not None:
        result = _run_command(["ps", "-p", str(int(pid)), "-o", "pid=,ppid=,stat=,etime=,%cpu=,%mem=,rss=,vsz=,cmd="])
        rows = _parse_ps_rows(result.get("stdout", ""))
        return rows[0] if rows else {"pid": int(pid), "found": False, "error": result.get("error")}
    result = _run_command(["ps", "-eo", "pid=,ppid=,stat=,etime=,%cpu=,%mem=,rss=,vsz=,cmd="])
    rows = _parse_ps_rows(result.get("stdout", ""))
    sweep_text = str(sweep)
    for row in rows:
        cmd = str(row.get("cmd", ""))
        if "neurobench.dynamics.overnight_sweep" in cmd and sweep_text in cmd:
            return row
    for row in rows:
        cmd = str(row.get("cmd", ""))
        if "neurobench.dynamics.overnight_sweep" in cmd and sweep.name in cmd:
            return row
    return {"found": False, "error": result.get("error")}


def _gpu_status(pid: Any, *, include_gpu: bool) -> dict[str, Any]:
    if not include_gpu:
        return {"checked": False}
    gpu = _run_command(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv,noheader,nounits"])
    apps = _run_command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    status: dict[str, Any] = {"checked": True}
    first_gpu_line = next((line for line in gpu.get("stdout", "").splitlines() if line.strip()), "")
    if first_gpu_line:
        parts = [part.strip() for part in first_gpu_line.split(",")]
        if len(parts) >= 5:
            status.update(
                {
                    "timestamp": parts[0],
                    "utilization_gpu_percent": _to_float(parts[1]),
                    "utilization_memory_percent": _to_float(parts[2]),
                    "memory_used_mib": _to_float(parts[3]),
                    "memory_total_mib": _to_float(parts[4]),
                }
            )
    target_pid = int(pid) if pid not in (None, "", False) else None
    process_rows = []
    for line in apps.get("stdout", "").splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            row = {"pid": _to_int(parts[0]), "process_name": parts[1], "used_memory_mib": _to_float(parts[2])}
            process_rows.append(row)
            if target_pid is not None and row["pid"] == target_pid:
                status["process_used_memory_mib"] = row["used_memory_mib"]
                status["process_name"] = row["process_name"]
    status["processes"] = process_rows
    if gpu.get("error"):
        status["gpu_error"] = gpu.get("error")
    if apps.get("error"):
        status["apps_error"] = apps.get("error")
    return status


def _classify_live_state(
    *,
    active_status: Mapping[str, Any],
    process: Mapping[str, Any],
    gpu: Mapping[str, Any],
    active_artifacts: Mapping[str, Any],
    current_rows: Sequence[Mapping[str, Any]],
) -> str:
    if current_rows and current_rows[-1].get("status") == "failed":
        return "latest_record_failed"
    if active_status.get("status") == "completed":
        return "active_status_completed"
    if active_artifacts.get("metrics_present"):
        return "metrics_written"
    if active_status.get("status") == "running" and process.get("found") is not False:
        gpu_util = float(gpu.get("utilization_gpu_percent") or 0.0)
        cpu_util = float(process.get("cpu_percent") or 0.0)
        process_gpu_mem = float(gpu.get("process_used_memory_mib") or 0.0)
        if gpu_util >= 10.0:
            return "active_training"
        if cpu_util >= 10.0 and process_gpu_mem > 0.0:
            return "active_cpu_phase"
        if cpu_util >= 10.0:
            return "active_cpu_only"
        return "running_but_idle"
    if active_status.get("status") == "running":
        return "active_without_process"
    return "no_active_status"


def _parse_ps_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(None, 8)
        if len(parts) < 9:
            continue
        rows.append(
            {
                "pid": _to_int(parts[0]),
                "ppid": _to_int(parts[1]),
                "stat": parts[2],
                "elapsed": parts[3],
                "cpu_percent": _to_float(parts[4]),
                "mem_percent": _to_float(parts[5]),
                "rss_kib": _to_int(parts[6]),
                "vsz_kib": _to_int(parts[7]),
                "cmd": parts[8],
                "found": True,
            }
        )
    return rows


def _run_command(command: Sequence[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except (FileNotFoundError, OSError) as exc:
        return {"stdout": "", "stderr": "", "returncode": None, "error": str(exc)}
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode, "error": result.stderr.strip() if result.returncode else None}


def _to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _shell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"
