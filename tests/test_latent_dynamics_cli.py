from __future__ import annotations

import sys
import json

from neurobench.cli.main import build_parser
from neurobench.cli.experiment import _configure_cuda_resource_environment


def test_latent_cli_registration_is_lazy():
    sys.modules.pop("neurobench.experiments.latent_dynamics.runner", None)
    parser = build_parser(active_command="experiment")
    args = parser.parse_args(["experiment", "latent-dynamics", "synthetic", "--output-dir", "x"])
    assert args.experiment_action == "synthetic"
    assert "neurobench.experiments.latent_dynamics.runner" not in sys.modules


def test_latent_preflight_cli_requires_explicit_artifact_dir():
    parser = build_parser(active_command="experiment")
    try:
        parser.parse_args(["experiment", "latent-dynamics", "preflight", "--config", "x.json"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("artifact-dir must be required")


def test_manifest_resource_limit_sets_blas_threads(tmp_path, monkeypatch):
    manifest = tmp_path / "config.json"
    manifest.write_text(json.dumps({"resources": {"cpu_threads": 3}}))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert _configure_cuda_resource_environment(manifest) == 3
    assert __import__("os").environ["OMP_NUM_THREADS"] == "3"
