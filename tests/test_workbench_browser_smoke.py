from __future__ import annotations

import json
import os
import signal
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _browser_smoke_payload() -> dict:
    return {
        "dataset": {"dataset_id": "browser_smoke"},
        "video": {"name": "synthetic.npy", "width": 12, "height": 10, "frames": 4, "framePattern": "frames/frame_%03d.png"},
        "parameters": {"eventZThreshold": 2.4},
        "rois": [
            {
                "id": 1,
                "area": 16,
                "centroid": [6, 5],
                "events": [{"frame": 2, "z": 3.2}],
                "dffTrace": [0.0, 0.15, 0.9, 0.2],
                "mask": [[5, 4], [6, 4], [5, 5], [6, 5]],
            }
        ],
        "discovery": {"evidenceMaps": [], "suggestions": []},
    }


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"{path} is not a valid PNG screenshot")
    return struct.unpack(">II", header[16:24])


def _diagnostic_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _close_capture_pipes(process: subprocess.Popen[str]) -> None:
    # A sandboxed or snap-confined browser can leave descendants holding the
    # capture pipes even after its parent is killed. Closing our handles keeps
    # the opt-in test bounded without masking the render failure.
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


@unittest.skipUnless(
    os.environ.get("NEUROBENCH_BROWSER_SMOKE") == "1" and shutil.which("firefox"),
    "Set NEUROBENCH_BROWSER_SMOKE=1 with Firefox installed to run the browser smoke test",
)
class WorkbenchBrowserSmokeTests(unittest.TestCase):
    def test_generated_workbench_renders_in_real_browser(self):
        from neurobench.workbench.builder import build_workbench
        from tools import build_neuron_workbench_v2 as legacy_builder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            home.mkdir()
            review_path = root / "review_data.json"
            review_path.write_text(json.dumps(_browser_smoke_payload()), encoding="utf-8")

            paths = build_workbench(
                app_dir=root / "app",
                review_data_path=review_path,
                dataset_id="browser_smoke",
                html_template=legacy_builder.HTML_TEMPLATE,
                dataset_manifest={"dataset_id": "browser_smoke", "paths": {"review_data": str(review_path)}},
                css_fallback=legacy_builder.CSS,
                js_fallback=legacy_builder.JS,
            )
            screenshot = root / "workbench.png"
            profile = root / "firefox-profile"
            profile.mkdir()
            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "XDG_CACHE_HOME": str(root / "cache"),
                    "XDG_CONFIG_HOME": str(root / "config"),
                    "MOZ_HEADLESS": "1",
                }
            )
            process = subprocess.Popen(
                [
                    "firefox",
                    "--headless",
                    "--no-remote",
                    "--new-instance",
                    "--profile",
                    str(profile),
                    "--window-size",
                    "1280,900",
                    f"--screenshot={screenshot}",
                    paths["index"].as_uri(),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=45)
            except subprocess.TimeoutExpired as timeout_error:
                captured = [
                    _diagnostic_text(timeout_error.output),
                    _diagnostic_text(timeout_error.stderr),
                ]
                cleanup_errors: list[str] = []
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except (OSError, PermissionError, ProcessLookupError) as error:
                    cleanup_errors.append(f"process-group cleanup: {error}")
                    try:
                        process.kill()
                    except (OSError, PermissionError, ProcessLookupError) as fallback_error:
                        cleanup_errors.append(f"parent-process cleanup: {fallback_error}")

                try:
                    stdout, stderr = process.communicate(timeout=5)
                    captured.extend((stdout, stderr))
                except subprocess.TimeoutExpired as cleanup_timeout:
                    captured.extend(
                        (
                            _diagnostic_text(cleanup_timeout.output),
                            _diagnostic_text(cleanup_timeout.stderr),
                        )
                    )
                    cleanup_errors.append(
                        "browser descendants retained capture pipes after forced cleanup"
                    )
                    _close_capture_pipes(process)

                diagnostics = "\n".join(
                    part.strip()[-2000:]
                    for part in (*captured, *cleanup_errors)
                    if part and part.strip()
                )
                self.fail(
                    "isolated Firefox render exceeded the 45-second bound"
                    + (f":\n{diagnostics}" if diagnostics else "")
                )

            diagnostics = "\n".join(
                part[-2000:]
                for part in (stdout.strip(), stderr.strip())
                if part
            )
            self.assertEqual(process.returncode, 0, diagnostics)
            self.assertTrue(
                screenshot.is_file(),
                "isolated Firefox exited without producing a screenshot"
                + (f":\n{diagnostics}" if diagnostics else ""),
            )
            width, height = _png_size(screenshot)
            size = screenshot.stat().st_size

        self.assertGreaterEqual(width, 1000)
        self.assertGreaterEqual(height, 700)
        self.assertGreater(size, 5000, "rendered workbench screenshot is suspiciously small")


if __name__ == "__main__":
    unittest.main()
