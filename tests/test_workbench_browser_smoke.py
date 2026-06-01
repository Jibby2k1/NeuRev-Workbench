from __future__ import annotations

import json
import os
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
            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "XDG_CACHE_HOME": str(root / "cache"),
                    "XDG_CONFIG_HOME": str(root / "config"),
                    "MOZ_HEADLESS": "1",
                }
            )
            result = subprocess.run(
                [
                    "firefox",
                    "--headless",
                    "--window-size",
                    "1280,900",
                    f"--screenshot={screenshot}",
                    paths["index"].as_uri(),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=45,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(screenshot.is_file())
            width, height = _png_size(screenshot)
            size = screenshot.stat().st_size

        self.assertGreaterEqual(width, 1000)
        self.assertGreaterEqual(height, 700)
        self.assertGreater(size, 5000, "rendered workbench screenshot is suspiciously small")


if __name__ == "__main__":
    unittest.main()
