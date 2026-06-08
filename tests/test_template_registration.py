from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class TemplateRegistrationTests(unittest.TestCase):
    def test_recovers_known_translation(self):
        from neurobench.algorithms.template_matching import _shift_image, estimate_rigid_registration

        y, x = np.mgrid[0:48, 0:64]
        template = np.exp(-((x - 32) ** 2 + (y - 24) ** 2) / 90.0).astype(np.float32)
        source = _shift_image(template, dy=3, dx=-4)
        result = estimate_rigid_registration(source, template, transform_model="translation")
        dx, dy = result["transform"]["translation_px"]

        self.assertLessEqual(abs(dx - 4), 2.0)
        self.assertLessEqual(abs(dy + 3), 2.0)

    def test_recovers_small_rotation_and_boundary_warning(self):
        from neurobench.algorithms.template_matching import _warp_similarity, estimate_rigid_registration

        y, x = np.mgrid[0:48, 0:64]
        template = (np.exp(-((x - 32) ** 2 + (y - 24) ** 2) / 90.0) + 0.3 * np.exp(-((x - 40) ** 2 + (y - 15) ** 2) / 30.0)).astype(np.float32)
        source = _warp_similarity(template, template.shape, rotation_deg=4.0)
        result = estimate_rigid_registration(source, template, rotation_range_deg=(-6, 6), rotation_step_deg=1.0)
        self.assertLessEqual(abs(result["transform"]["rotation_deg"] + 4.0), 2.0)

        boundary = estimate_rigid_registration(source, template, rotation_range_deg=(-2, 2), rotation_step_deg=1.0)
        self.assertTrue(boundary["qc"]["best_angle_at_boundary"])
