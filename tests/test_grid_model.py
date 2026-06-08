from __future__ import annotations

import unittest
from neurobench.manifests import load_json


class GridModelTests(unittest.TestCase):
    def test_grid_model_validates_example(self):
        from neurobench.models.grid import GridSpec
        grid = GridSpec.from_dict(load_json("examples/grid_spec_32x32.example.json"))
        grid.validate()
        self.assertEqual(grid.to_dict()["region_count"], 1024)
