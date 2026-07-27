from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest


def _make_app(root: Path, relative: str, dataset_id: str) -> Path:
    app = root / relative
    app.mkdir(parents=True)
    files = {
        "review_data.json": json.dumps({"dataset": {"dataset_id": dataset_id}, "video": {"frames": 1}}),
        "annotations.json": '{"schema_version": 3}\n',
        "architecture_runs.json": json.dumps({"schema_version": 1, "dataset_id": dataset_id, "runs": []}) + "\n",
        "index.html": "<html></html>\n",
        "workbench.css": ".app {}\n",
        "workbench.js": "const ready = true;\n",
    }
    for name, content in files.items():
        (app / name).write_text(content, encoding="utf-8")
    return app


class WorkbenchBaselineTests(unittest.TestCase):
    def test_capture_baseline_hashes_locked_files_without_mutation(self):
        from neurobench.workbench.baseline import capture_baseline, write_baseline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = _make_app(root, "Outputs/NeuronReview/demo/app", "demo")
            before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            baseline = capture_baseline(root, app_dirs=[app])
            output = write_baseline(root / "baseline.json", baseline)
            after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.name != "baseline.json")

        self.assertEqual(before, after)
        self.assertEqual(output.name, "baseline.json")
        self.assertEqual(baseline["locked_contracts"]["annotations_preserved"], True)
        self.assertEqual(baseline["apps"][0]["dataset_id"], "demo")
        self.assertEqual(baseline["apps"][0]["files"]["annotations.json"]["size_bytes"], 22)
        self.assertEqual(len(baseline["stable_identity"]["sha256"]), 64)

    def test_default_discovery_includes_catalog_apps_outside_neuron_review(self):
        from neurobench.workbench.baseline import capture_baseline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = _make_app(root, "Outputs/GammaCFAR/spon_ca/app", "spon_ca")

            baseline = capture_baseline(root)

        self.assertEqual([item["dataset_id"] for item in baseline["apps"]], ["spon_ca"])
        self.assertEqual(baseline["apps"][0]["app_dir"], app.relative_to(root).as_posix())

    def test_relative_app_paths_resolve_under_root_and_outside_paths_are_rejected(self):
        from neurobench.workbench.baseline import capture_baseline

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            _make_app(root, "Outputs/NeuronReview/demo/app", "demo")
            outside = _make_app(Path(outside_tmp), "app", "outside")

            baseline = capture_baseline(root, app_dirs=[Path("Outputs/NeuronReview/demo/app")])
            with self.assertRaisesRegex(ValueError, "outside workspace root"):
                capture_baseline(root, app_dirs=[outside])

        self.assertEqual(baseline["apps"][0]["dataset_id"], "demo")

    def test_capture_fails_when_a_protected_file_is_missing(self):
        from neurobench.workbench.baseline import capture_baseline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = _make_app(root, "Outputs/NeuronReview/demo/app", "demo")
            (app / "workbench.js").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "workbench.js"):
                capture_baseline(root, app_dirs=[app])

    def test_write_refuses_overwrite_unless_explicit(self):
        from neurobench.workbench.baseline import capture_baseline, write_baseline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = _make_app(root, "Outputs/NeuronReview/demo/app", "demo")
            baseline = capture_baseline(root, app_dirs=[app])
            target = write_baseline(root / "baseline.json", baseline)
            first = target.read_bytes()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_baseline(target, baseline)
            write_baseline(target, baseline, overwrite=True)
            overwritten = target.read_bytes()

        self.assertEqual(overwritten, first)

    def test_diff_ignores_capture_time_but_verify_finds_protected_byte_change(self):
        from neurobench.workbench.baseline import capture_baseline, diff_baselines, verify_baseline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = _make_app(root, "Outputs/NeuronReview/demo/app", "demo")
            baseline = capture_baseline(root, app_dirs=[app])
            later = json.loads(json.dumps(baseline))
            later["captured_at"] = "2099-01-01T00:00:00Z"
            timestamp_only = diff_baselines(baseline, later)

            (app / "annotations.json").write_bytes(b'{"schema_version":3,"changed":true}\n')
            verified = verify_baseline(root, baseline)

        self.assertTrue(timestamp_only["match"])
        self.assertFalse(verified["match"])
        self.assertTrue(any("annotations.json" in change["path"] for change in verified["changes"]))

    def test_cli_roots_capture_and_supports_verify_and_diff(self):
        from neurobench.cli.main import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = _make_app(root, "Outputs/NeuronReview/demo/app", "demo")
            output = io.StringIO()
            with redirect_stdout(output):
                captured = main(
                    [
                        "workbench",
                        "baseline",
                        "--root",
                        str(root),
                        "--app-dir",
                        "Outputs/NeuronReview/demo/app",
                        "--output",
                        "locks/before.json",
                    ]
                )
                verified = main(
                    [
                        "workbench",
                        "baseline",
                        "--root",
                        str(root),
                        "--verify",
                        "locks/before.json",
                    ]
                )
                (app / "annotations.json").write_bytes(b'{"schema_version":3,"changed":true}\n')
                changed = main(
                    [
                        "workbench",
                        "baseline",
                        "--root",
                        str(root),
                        "--verify",
                        "locks/before.json",
                    ]
                )
                main(
                    [
                        "workbench",
                        "baseline",
                        "--root",
                        str(root),
                        "--app-dir",
                        "Outputs/NeuronReview/demo/app",
                        "--output",
                        "locks/after.json",
                    ]
                )
                different = main(
                    [
                        "workbench",
                        "baseline",
                        "--root",
                        str(root),
                        "--diff",
                        "locks/before.json",
                        "locks/after.json",
                    ]
                )

            before_exists = (root / "locks" / "before.json").is_file()

        self.assertEqual(captured, 0)
        self.assertEqual(verified, 0)
        self.assertEqual(changed, 1)
        self.assertEqual(different, 1)
        self.assertTrue(before_exists)


if __name__ == "__main__":
    unittest.main()
