from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from neurobench.dashboards.manifest import (
    DashboardManifestError,
    discover_dashboard_manifests,
    load_dashboard_manifest,
    summarize_dashboard_manifest,
    write_dashboard_manifest,
)


class DashboardManifestTests(unittest.TestCase):
    def test_write_load_discover_and_summarize_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            payload = {
                "schema_version": 1,
                "dashboard_id": "demo_dashboard",
                "dashboard_type": "neuron_workbench_gamma_cfar",
                "dataset_id": "demo",
                "entrypoint": "app/index.html",
                "serve_command": "python tools/serve_neuron_workbench.py --app-dir app",
                "gamma_cfar": {"recommended_runs": ["run_001"]},
                "review_app": {"app_dir": "app"},
            }

            path = write_dashboard_manifest(root, payload)

            self.assertEqual(path.name, "dashboard_manifest.json")
            self.assertEqual(load_dashboard_manifest(root)["dashboard_id"], "demo_dashboard")
            self.assertEqual(discover_dashboard_manifests(tmp), [path])
            self.assertEqual(
                summarize_dashboard_manifest(payload),
                {
                    "dashboard_id": "demo_dashboard",
                    "dashboard_type": "neuron_workbench_gamma_cfar",
                    "dataset_id": "demo",
                    "entrypoint": "app/index.html",
                    "serve_command": "python tools/serve_neuron_workbench.py --app-dir app",
                    "recommended_runs": ["run_001"],
                    "app_dir": "app",
                },
            )

    def test_manifest_requires_core_fields(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(DashboardManifestError):
                write_dashboard_manifest(Path(tmp), {"schema_version": 1, "dashboard_id": "missing"})


if __name__ == "__main__":
    unittest.main()
