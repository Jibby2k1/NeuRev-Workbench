#!/usr/bin/env python3
"""Run a long Neurobench validation soak and write interpretable reports."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / ".venv-neurobench" / "bin" / "python"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    purpose: str
    args: tuple[str, ...]


COMMAND_MATRIX = (
    CommandSpec(
        "collect_tests",
        "Confirm the test inventory and expose accidental collection/import failures.",
        ("-m", "pytest", "--collect-only", "-q"),
    ),
    CommandSpec(
        "full_pytest",
        "Exercise the full CPU-safe repository contract: models, schemas, CLI, workbench, metrics, reports, pipelines, and docs.",
        ("-m", "pytest", "-q"),
    ),
    CommandSpec(
        "pipeline_and_device",
        "Stress executable pipeline paths, stage registry metadata, sweeps, local runners, and CPU/GPU device fallback behavior.",
        (
            "-m",
            "pytest",
            "-q",
            "tests/test_pipeline_executor.py",
            "tests/test_pipeline_sweep_execution.py",
            "tests/test_pipeline_sweeps.py",
            "tests/test_device_abstraction.py",
            "tests/test_gpu_smoke.py",
        ),
    ),
    CommandSpec(
        "workbench_process_lab",
        "Stress the annotation dashboard contract, server safety, asset packaging, Process Lab intermediate export, and CLI attachment paths.",
        (
            "-m",
            "pytest",
            "-q",
            "tests/test_workbench_builder.py",
            "tests/test_workbench_server.py",
            "tests/test_workbench_structure.py",
            "tests/test_workbench_assets.py",
            "tests/test_intermediate_export.py",
            "tests/test_attach_pipeline_intermediates.py",
            "tests/test_cli_main.py",
        ),
    ),
    CommandSpec(
        "science_metrics_exports",
        "Stress object/event metrics, run comparison, annotation exports, inverse-dynamics export, reports, and behavior alignment diagnostics.",
        (
            "-m",
            "pytest",
            "-q",
            "tests/test_object_metrics.py",
            "tests/test_event_metrics.py",
            "tests/test_run_comparison_metrics.py",
            "tests/test_run_comparison_report.py",
            "tests/test_annotation_exports.py",
            "tests/test_inverse_dynamics_export.py",
            "tests/test_behavior_alignment.py",
            "tests/test_metrics_report_builder.py",
            "tests/test_metrics_report_render.py",
        ),
    ),
    CommandSpec(
        "docs_api_setup",
        "Confirm documentation, generated API reference, portable environments, schemas, and workflow examples remain synchronized.",
        (
            "-m",
            "pytest",
            "-q",
            "tests/test_api_reference_generation.py",
            "tests/test_docs_workflows.py",
            "tests/test_developer_docs.py",
            "tests/test_environment_setup.py",
            "tests/test_schema_validation.py",
        ),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_text(command: list[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


MappingLike = dict[str, Any]


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("NEUROBENCH_BROWSER_SMOKE", "0")
    return env


def run_command(
    *,
    python: Path,
    spec: CommandSpec,
    run_dir: Path,
    cycle_index: int,
    command_index: int,
) -> dict[str, Any]:
    start = time.time()
    started_at = utc_now()
    cmd = [str(python), *spec.args]
    log_name = f"{cycle_index:04d}_{command_index:02d}_{spec.name}.log"
    log_path = run_dir / "logs" / log_name
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=command_env(), check=False)
    duration = time.time() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"$ {' '.join(cmd)}",
                f"started_at_utc: {started_at}",
                f"finished_at_utc: {utc_now()}",
                f"duration_seconds: {duration:.3f}",
                f"returncode: {result.returncode}",
                "",
                "## stdout",
                result.stdout,
                "",
                "## stderr",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return {
        "cycle": cycle_index,
        "command_index": command_index,
        "name": spec.name,
        "purpose": spec.purpose,
        "command": cmd,
        "returncode": int(result.returncode),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "duration_seconds": round(duration, 3),
        "log": str(log_path.relative_to(run_dir)),
    }


def summarize(records: list[dict[str, Any]], *, started_at: str, finished_at: str, requested_duration_seconds: float) -> dict[str, Any]:
    by_command: dict[str, dict[str, Any]] = {}
    for record in records:
        item = by_command.setdefault(
            record["name"],
            {"runs": 0, "passes": 0, "failures": 0, "total_seconds": 0.0, "first_failure_log": None},
        )
        item["runs"] += 1
        item["total_seconds"] += float(record["duration_seconds"])
        if record["returncode"] == 0:
            item["passes"] += 1
        else:
            item["failures"] += 1
            item["first_failure_log"] = item["first_failure_log"] or record["log"]
    for item in by_command.values():
        item["average_seconds"] = round(item["total_seconds"] / max(1, item["runs"]), 3)
        item["total_seconds"] = round(item["total_seconds"], 3)
    failures = [record for record in records if record["returncode"] != 0]
    return {
        "schema_version": 1,
        "kind": "neurobench_validation_soak_summary",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "requested_duration_seconds": int(requested_duration_seconds),
        "actual_duration_seconds": round(sum(float(record["duration_seconds"]) for record in records), 3),
        "total_command_runs": len(records),
        "passed_command_runs": len(records) - len(failures),
        "failed_command_runs": len(failures),
        "cycles_completed": max((int(record["cycle"]) for record in records), default=0),
        "commands": by_command,
        "failures": failures,
    }


def render_markdown(summary: dict[str, Any], run_dir: Path, metadata: dict[str, Any]) -> str:
    status = "PASS" if summary["failed_command_runs"] == 0 else "FAIL"
    lines = [
        f"# Neurobench Overnight Validation: {status}",
        "",
        "## Purpose",
        "",
        "This run repeatedly exercised the Neurobench codebase before starting new-data experiments. "
        "The goal was to catch regressions in the scientific pipeline, workbench, annotation/export contracts, documentation, and CLI workflows.",
        "",
        "## Run Summary",
        "",
        f"- Started UTC: `{summary['started_at_utc']}`",
        f"- Finished UTC: `{summary['finished_at_utc']}`",
        f"- Requested duration: `{summary['requested_duration_seconds']} seconds`",
        f"- Command runs: `{summary['total_command_runs']}`",
        f"- Passed command runs: `{summary['passed_command_runs']}`",
        f"- Failed command runs: `{summary['failed_command_runs']}`",
        f"- Completed cycles: `{summary['cycles_completed']}`",
        f"- Result directory: `{run_dir}`",
        "",
        "## What Was Tested",
        "",
    ]
    for spec in COMMAND_MATRIX:
        stats = summary["commands"].get(spec.name, {"runs": 0, "passes": 0, "failures": 0, "average_seconds": 0})
        lines.append(
            f"- `{spec.name}`: {spec.purpose} "
            f"Runs `{stats['runs']}`, passes `{stats['passes']}`, failures `{stats['failures']}`, avg seconds `{stats['average_seconds']}`."
        )
    lines.extend(["", "## Reproducibility Context", ""])
    for key in ["git_head", "git_status_short", "python", "platform", "test_inventory"]:
        value = metadata.get(key, "")
        if key == "git_status_short":
            lines.append("- Git status at start:")
            lines.append("")
            lines.append("```text")
            lines.append(str(value).strip() or "clean")
            lines.append("```")
        elif key == "test_inventory":
            lines.append(f"- Test inventory at start: `{str(value).splitlines()[-1] if value else 'unknown'}`")
        else:
            lines.append(f"- {key.replace('_', ' ').title()}: `{value}`")
    if summary["failures"]:
        lines.extend(["", "## Failures To Inspect", ""])
        for failure in summary["failures"][:20]:
            lines.append(f"- Cycle `{failure['cycle']}`, command `{failure['name']}`, log `{failure['log']}`")
        if len(summary["failures"]) > 20:
            lines.append(f"- Additional failures omitted from this brief: `{len(summary['failures']) - 20}`")
    else:
        lines.extend(["", "## Interpretation", "", "No command failures were observed during the validation soak."])
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.json`: machine-readable aggregate result.",
            "- `events.jsonl`: one record per command execution.",
            "- `logs/`: stdout/stderr for every command run.",
            "- `metadata.json`: git, platform, and test-inventory context.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-hours", type=float, default=9.0, help="Approximate wall-clock duration to keep cycling.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to Outputs/ValidationRuns/<timestamp>.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python executable to use for pytest commands.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first command failure.")
    parser.add_argument("--max-cycles", type=int, default=None, help="Optional cycle cap for smoke testing the runner itself.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started_at = utc_now()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir or ROOT / "Outputs" / "ValidationRuns" / f"validation_soak_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    duration_seconds = max(0.0, float(args.duration_hours) * 3600.0)
    deadline = time.time() + duration_seconds
    metadata = {
        "started_at_utc": started_at,
        "requested_duration_hours": args.duration_hours,
        "python": str(args.python),
        "platform": platform.platform(),
        "git_head": run_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_text(["git", "status", "--short"]),
        "test_inventory": run_text([str(args.python), "-m", "pytest", "--collect-only", "-q"]),
        "command_matrix": [spec.__dict__ for spec in COMMAND_MATRIX],
    }
    write_json(run_dir / "metadata.json", metadata)

    records: list[dict[str, Any]] = []
    cycle = 0
    while True:
        if args.max_cycles is not None and cycle >= args.max_cycles:
            break
        if records and time.time() >= deadline:
            break
        cycle += 1
        for index, spec in enumerate(COMMAND_MATRIX, start=1):
            record = run_command(python=args.python, spec=spec, run_dir=run_dir, cycle_index=cycle, command_index=index)
            records.append(record)
            append_jsonl(run_dir / "events.jsonl", record)
            summary = summarize(records, started_at=started_at, finished_at=utc_now(), requested_duration_seconds=duration_seconds)
            write_json(run_dir / "summary.json", summary)
            (run_dir / "experiment_brief.md").write_text(render_markdown(summary, run_dir, metadata), encoding="utf-8")
            if record["returncode"] != 0 and args.stop_on_failure:
                return 1
            if time.time() >= deadline:
                break

    summary = summarize(records, started_at=started_at, finished_at=utc_now(), requested_duration_seconds=duration_seconds)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "experiment_brief.md").write_text(render_markdown(summary, run_dir, metadata), encoding="utf-8")
    print(f"Validation soak complete: {run_dir}")
    print(f"command runs: {summary['total_command_runs']}")
    print(f"failures: {summary['failed_command_runs']}")
    return 0 if summary["failed_command_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
