from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import patch


def _review_payload() -> dict:
    return {
        "dataset": {"dataset_id": "demo"},
        "video": {
            "name": "demo.npy",
            "width": 2,
            "height": 2,
            "frames": 1,
            "framePattern": "frames/frame_%03d.png",
        },
        "rois": [],
        "discovery": {"suggestions": []},
    }


def test_serve_parser_exposes_explicit_current_and_installed_asset_modes():
    from neurobench.cli.main import build_parser

    parser = build_parser(active_command="workbench")
    current = parser.parse_args(["workbench", "serve", "--app-dir", "demo"])
    installed = parser.parse_args(
        ["workbench", "serve", "--app-dir", "demo", "--asset-mode", "installed"]
    )

    assert current.asset_mode == "current"
    assert installed.asset_mode == "installed"


def test_status_reports_tampered_css_even_when_html_marker_is_current(tmp_path: Path):
    from neurobench.cli.workbench import workbench_status_command
    from neurobench.workbench.builder import build_workbench

    review_path = tmp_path / "review_data.json"
    review_path.write_text(json.dumps(_review_payload()), encoding="utf-8")
    paths = build_workbench(
        app_dir=tmp_path / "app",
        review_data_path=review_path,
        dataset_id="demo",
    )
    paths["css"].write_bytes(paths["css"].read_bytes() + b"/* tampered */\n")
    args = argparse.Namespace(
        app_dir=tmp_path / "app",
        dataset_id=None,
        catalog_root=tmp_path,
        json=True,
    )
    output = io.StringIO()

    with redirect_stdout(output):
        result = workbench_status_command(args)
    payload = json.loads(output.getvalue())

    assert result == 0
    assert payload["assets"]["marker_current"] is True
    assert payload["assets"]["css_current"] is False
    assert payload["assets"]["js_current"] is True
    assert payload["assets"]["current"] is False


class _OneShotServer:
    server_address = ("127.0.0.1", 4321)

    def __init__(self) -> None:
        self.served = False
        self.closed = False

    def serve_forever(self) -> None:
        self.served = True

    def server_close(self) -> None:
        self.closed = True


def test_serve_cli_passes_current_asset_mode_without_building_or_writing(tmp_path: Path):
    from neurobench.cli.workbench import workbench_serve_command

    fake = _OneShotServer()
    args = argparse.Namespace(
        app_dir=tmp_path,
        dataset_id=None,
        catalog_root=tmp_path,
        host="127.0.0.1",
        port=0,
        asset_mode="current",
    )
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    output = io.StringIO()

    with patch("neurobench.workbench.server.create_workbench_server", return_value=(fake, tmp_path)), redirect_stdout(output):
        result = workbench_serve_command(args)
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    assert result == 0
    assert fake.served and fake.closed
    assert before == after
    assert "(current assets)" in output.getvalue()


def test_installed_serve_mode_warns_when_asset_bytes_are_stale(tmp_path: Path):
    from neurobench.cli.workbench import workbench_serve_command

    fake = _OneShotServer()
    args = argparse.Namespace(
        app_dir=tmp_path,
        dataset_id=None,
        catalog_root=tmp_path,
        host="127.0.0.1",
        port=0,
        asset_mode="installed",
    )
    stderr = io.StringIO()

    with patch("neurobench.workbench.server.create_workbench_server", return_value=(fake, tmp_path)), redirect_stderr(stderr):
        result = workbench_serve_command(args)

    assert result == 0
    assert "WARNING: serving stale or tampered installed workbench assets" in stderr.getvalue()
