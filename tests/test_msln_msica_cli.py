import os
import sys

from neurobench.cli.experiment import _configure_cuda_resource_environment
from neurobench.cli.main import build_parser


def test_msln_msica_cli_is_lazy_and_exposes_guarded_actions() -> None:
    before = set(sys.modules)
    parser = build_parser(active_command="experiment")
    run = parser.parse_args([
        "experiment", "msln-msica", "run", "--config", "config.json",
        "--authorize-full-spon", "--resume", "--no-video", "--stage", "all",
    ])
    assert run.experiment_workflow == "msln-msica"
    assert run.authorize_full_spon and run.resume and run.no_video
    assert "neurobench.experiments.msln_msica.runner" not in set(sys.modules) - before
    summary = parser.parse_args([
        "experiment", "msln-msica", "summarize", "--output-root", "output"
    ])
    assert summary.func.__name__ == "_summarize_msln_msica"


def test_msln_manifest_sets_declared_cpu_threads(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"compute":{"cpu_threads":3}}', encoding="utf-8")
    assert _configure_cuda_resource_environment(path) == 3
    assert os.environ["OMP_NUM_THREADS"] == "3"
