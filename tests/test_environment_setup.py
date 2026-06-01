from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class EnvironmentSetupTests(unittest.TestCase):
    def test_portable_environment_files_have_no_absolute_prefix(self):
        for path in ["environment.yml", "environment.cpu.yml", "environment.gpu.yml"]:
            text = _text(path)
            self.assertNotIn("prefix:", text)
            self.assertNotIn("/home/", text)
            self.assertIn("python=3.10", text)
            self.assertIn("pytest", text)

    def test_cpu_environment_omits_gpu_only_dependencies(self):
        text = _text("environment.cpu.yml")

        self.assertNotIn("pytorch-cuda", text)
        self.assertNotIn("cupy", text)
        self.assertIn("cpuonly", text)

    def test_requirements_dev_includes_test_and_schema_dependencies(self):
        packages = {
            line.strip().lower()
            for line in _text("requirements-dev.txt").splitlines()
            if line.strip() and not line.startswith("#")
        }

        self.assertIn("pytest", packages)
        self.assertIn("jsonschema", packages)
        self.assertIn("numpy", packages)
        self.assertIn("torch", packages)


if __name__ == "__main__":
    unittest.main()
