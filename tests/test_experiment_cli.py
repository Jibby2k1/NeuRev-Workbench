from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import textwrap

import numpy as np

from neurobench.cli.main import build_parser, main


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = ROOT / ".venv-neurobench" / "bin" / "python"
RESOURCE_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def test_parser_registers_nested_soma_excitation_workflow():
    args = build_parser(active_command="experiment").parse_args(
        [
            "experiment",
            "soma-excitation",
            "preflight",
            "--config",
            "experiment.json",
        ]
    )

    assert args.experiment_workflow == "soma-excitation"
    assert args.experiment_action == "preflight"
    assert callable(args.func)


def test_preflight_cli_prints_json_and_can_write_explicit_copy(tmp_path, capsys):
    source = tmp_path / "source.npy"
    np.save(source, np.zeros((20, 8, 9), dtype=np.uint16))
    config = tmp_path / "experiment.json"
    config.write_text(
        json.dumps(
            {
                "source_video": source.name,
                "output_dir": "result",
                "onset_frame_ui": 11,
                "control_preroll_frames": 10,
                "resources": {
                    "device": "cpu",
                    "worker_count": 1,
                    "chunk_frames": 2,
                    "cpu_threads": 1,
                    "max_ram_mib": 1024,
                    "max_output_mib": 64,
                },
            }
        )
    )
    output = tmp_path / "preflight.json"

    code = main(
        [
            "experiment",
            "soma-excitation",
            "preflight",
            "--config",
            str(config),
            "--output-json",
            str(output),
        ]
    )

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed["ready"] is True
    assert written["frame_bounds"]["score_start_frame_zero"] == 10
    assert not (tmp_path / "result").exists()


def test_resource_bootstrap_defaults_to_two_threads(tmp_path, monkeypatch):
    from neurobench.cli.experiment import (
        _configure_resource_environment_from_manifest,
    )

    config = tmp_path / "experiment.json"
    config.write_text("{}", encoding="utf-8")
    for name in RESOURCE_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert _configure_resource_environment_from_manifest(config) == 2
    assert {
        name: os.environ[name] for name in RESOURCE_ENVIRONMENT_VARIABLES
    } == {name: "2" for name in RESOURCE_ENVIRONMENT_VARIABLES}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_resource_bootstrap_rejects_unsafe_thread_counts(tmp_path):
    from neurobench.cli.experiment import (
        _configure_resource_environment_from_manifest,
    )

    config = tmp_path / "experiment.json"
    for value in (0, 9, True, 1.5):
        config.write_text(
            json.dumps({"resources": {"cpu_threads": value}}),
            encoding="utf-8",
        )
        try:
            _configure_resource_environment_from_manifest(config)
        except ValueError:
            continue
        raise AssertionError(f"unsafe cpu_threads value was accepted: {value!r}")


def test_explicit_experiment_preflight_limits_resources_before_scientific_imports(
    tmp_path,
):
    source = tmp_path / "source.npy"
    np.save(source, np.zeros((20, 8, 9), dtype=np.uint16))
    config = tmp_path / "experiment.json"
    config.write_text(
        json.dumps(
            {
                "source_video": source.name,
                "output_dir": "result",
                "onset_frame_ui": 11,
                "control_preroll_frames": 10,
                "resources": {
                    "device": "cpu",
                    "worker_count": 1,
                    "chunk_frames": 2,
                    "cpu_threads": 3,
                    "max_ram_mib": 1024,
                    "max_output_mib": 64,
                },
            }
        ),
        encoding="utf-8",
    )
    script = textwrap.dedent(
        """
        import contextlib
        import io
        import json
        import os
        import sys

        CONFIG_PATH = __CONFIG_PATH__
        RESOURCE_ENV = (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        SCIENTIFIC_ROOTS = {"numpy", "scipy", "torch"}

        from neurobench.cli.main import main

        before = sorted(
            name for name in (
                "neurobench.cli.dynamics",
                "numpy",
                "scipy",
                "torch",
            )
            if name in sys.modules
        )
        if before:
            raise RuntimeError(f"heavy modules loaded before experiment dispatch: {before}")

        class EnvironmentGuard:
            def __init__(self):
                self.seen = set()

            def find_spec(self, fullname, path=None, target=None):
                root = fullname.partition(".")[0]
                if root in SCIENTIFIC_ROOTS:
                    expected = {name: "3" for name in RESOURCE_ENV}
                    actual = {name: os.environ.get(name) for name in RESOURCE_ENV}
                    if actual != expected:
                        raise RuntimeError(
                            f"scientific import preceded resource limits: {actual}"
                        )
                    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                        raise RuntimeError("CUDA was not hidden before scientific import")
                    self.seen.add(root)
                return None

        guard = EnvironmentGuard()
        sys.meta_path.insert(0, guard)
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(
                [
                    "experiment",
                    "soma-excitation",
                    "preflight",
                    "--config",
                    CONFIG_PATH,
                ]
            )
        print(
            json.dumps(
                {
                    "code": code,
                    "before": before,
                    "dynamics_loaded": "neurobench.cli.dynamics" in sys.modules,
                    "torch_loaded": "torch" in sys.modules,
                    "scipy_loaded": "scipy" in sys.modules,
                    "seen": sorted(guard.seen),
                    "environment": {
                        name: os.environ.get(name) for name in RESOURCE_ENV
                    },
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ),
                },
                sort_keys=True,
            )
        )
        """
    ).replace("__CONFIG_PATH__", repr(str(config)))

    environment = os.environ.copy()
    for name in (*RESOURCE_ENVIRONMENT_VARIABLES, "CUDA_VISIBLE_DEVICES"):
        environment.pop(name, None)
    result = subprocess.run(
        [str(PROJECT_PYTHON), "-c", script],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["code"] == 0
    assert payload["before"] == []
    assert payload["dynamics_loaded"] is False
    assert payload["torch_loaded"] is False
    assert payload["scipy_loaded"] is True
    assert {"numpy", "scipy"} <= set(payload["seen"])
    assert payload["environment"] == {
        name: "3" for name in RESOURCE_ENVIRONMENT_VARIABLES
    }
    assert payload["cuda_visible_devices"] == ""
