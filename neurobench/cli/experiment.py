"""Thin CLI for focused, manifest-driven experiment workflows."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_DEFAULT_CPU_THREADS = 2
_MAX_CPU_THREADS = 8


def add_experiment_subcommands(subparsers) -> None:
    """Register experiment workflows without importing heavy runtime modules."""
    root = subparsers.add_parser(
        "experiment",
        help="Preflight and run focused resource-bounded experiments.",
        description="Focused resource-bounded experiment workflows.",
    )
    workflows = root.add_subparsers(dest="experiment_workflow", required=True)
    soma = workflows.add_parser(
        "soma-excitation",
        help="Evaluate dark-soma zones and frozen model transfer.",
    )
    actions = soma.add_subparsers(dest="experiment_action", required=True)

    preflight = actions.add_parser("preflight", help="Validate inputs and resource caps.")
    preflight.add_argument("--config", required=True, help="Soma-excitation JSON manifest.")
    preflight.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Permit preflight inspection of an existing output directory; never overwrites it.",
    )
    preflight.add_argument("--output-json", help="Optional path for the preflight JSON.")
    preflight.set_defaults(func=_run_soma_preflight)

    run = actions.add_parser("run", help="Run the bounded CPU experiment.")
    run.add_argument("--config", required=True, help="Soma-excitation JSON manifest.")
    run.set_defaults(func=_run_soma_experiment)

    learnable = workflows.add_parser("learnable-contrast", help="Train and evaluate guarded weakly-supervised contrast kernels.")
    learnable_actions = learnable.add_subparsers(dest="experiment_action", required=True)
    lc_preflight = learnable_actions.add_parser("preflight", help="Validate labels, source data, coordinates, and CUDA resources.")
    lc_preflight.add_argument("--config", required=True)
    lc_preflight.add_argument("--artifact-dir", type=Path, required=True)
    lc_preflight.set_defaults(func=_run_learnable_contrast_preflight)
    lc_diag = learnable_actions.add_parser("diagnostic", help="Run the 2x2x2 spatiotemporal factorial diagnostic.")
    lc_diag.add_argument("--config", required=True)
    lc_diag.set_defaults(func=_run_learnable_contrast_diagnostic)
    lc_direct = learnable_actions.add_parser("direct-tuning", help="Tune a detector initialized from the raw-direct baseline.")
    lc_direct.add_argument("--config", required=True)
    lc_direct.set_defaults(func=_run_learnable_direct_tuning)
    lc_run = learnable_actions.add_parser("run", help="Run the guarded CUDA learnable-contrast experiment.")
    lc_run.add_argument("--config", required=True)
    lc_run.set_defaults(func=_run_learnable_contrast_experiment)


def _run_soma_preflight(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.soma_excitation import (
        SomaExcitationConfig,
        build_soma_excitation_preflight,
    )

    config = SomaExcitationConfig.load_json(args.config)
    payload = build_soma_excitation_preflight(
        config,
        allow_existing_output=bool(args.allow_existing_output),
    )
    if args.output_json:
        _atomic_json(Path(args.output_json), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_soma_experiment(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.soma_excitation.runner import (
        run_soma_excitation_experiment,
    )

    payload = run_soma_excitation_experiment(args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _configure_resource_environment_from_manifest(config_path: str | Path) -> int:
    """Set process limits from JSON before importing a scientific dependency."""
    source = Path(config_path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The soma-excitation JSON config must contain one object.")
    resources = payload.get("resources")
    if resources is None:
        resources = {}
    if not isinstance(resources, dict):
        raise ValueError("resources must be a JSON object.")
    raw_threads = resources.get("cpu_threads", _DEFAULT_CPU_THREADS)
    if isinstance(raw_threads, bool):
        raise ValueError("resources.cpu_threads must be an integer, not a boolean.")
    try:
        cpu_threads = int(raw_threads)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"resources.cpu_threads must be an integer; got {raw_threads!r}."
        ) from exc
    if isinstance(raw_threads, float) and not raw_threads.is_integer():
        raise ValueError(
            f"resources.cpu_threads must be an integer; got {raw_threads!r}."
        )
    if not 1 <= cpu_threads <= _MAX_CPU_THREADS:
        raise ValueError(
            "resources.cpu_threads must be between "
            f"1 and {_MAX_CPU_THREADS}; got {cpu_threads}."
        )
    value = str(cpu_threads)
    for name in _THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return cpu_threads


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _configure_cuda_resource_environment(config_path: str | Path) -> int:
    """Set bounded CPU library threads without hiding the requested CUDA device."""
    payload = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    resources = payload.get("resources", {})
    raw = resources.get("cpu_threads", 4)
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 24:
        raise ValueError("resources.cpu_threads must be an integer between 1 and 24")
    for name in _THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = str(raw)
    return raw


def _run_learnable_contrast_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast import Config, preflight
    payload = preflight(Config.load(args.config), artifact_dir=args.artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_learnable_contrast_experiment(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast import Config, run
    payload = run(Config.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_learnable_contrast_diagnostic(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast.diagnostic import DiagnosticConfig, run
    payload = run(DiagnosticConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_learnable_direct_tuning(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast.direct_tuning import DirectTuningConfig, run
    payload = run(DirectTuningConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
