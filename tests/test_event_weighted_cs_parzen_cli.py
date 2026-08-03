import os
import sys

from neurobench.cli.experiment import _configure_cuda_resource_environment
from neurobench.cli.main import build_parser


def test_event_weighted_parser_is_lazy_and_exposes_explicit_run_guards():
    before = set(sys.modules)
    args = build_parser(active_command="experiment").parse_args(
        [
            "experiment",
            "event-weighted-cs-parzen",
            "run",
            "--config",
            "config.yaml",
            "--preflight-dir",
            "preflight",
            "--authorize-full-spon",
            "--resume",
        ]
    )
    assert args.experiment_workflow == "event-weighted-cs-parzen"
    assert args.authorize_full_spon is True
    assert args.resume is True
    imported = set(sys.modules) - before
    assert "neurobench.experiments.event_weighted_cs_parzen.runner" not in imported


def test_event_weighted_yaml_sets_bounded_threads_before_scientific_imports(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("compute:\n  max_worker_processes: 3\n", encoding="utf-8")
    assert _configure_cuda_resource_environment(path) == 3
    assert os.environ["OMP_NUM_THREADS"] == "3"
