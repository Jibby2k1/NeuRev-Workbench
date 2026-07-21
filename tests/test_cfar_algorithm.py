from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def require_numpy():
    if np is None:
        raise unittest.SkipTest("numpy is not installed in this Python environment")


class CfarAlgorithmTests(unittest.TestCase):
    def test_gamma_cfar_detects_local_spots_on_heterogeneous_background(self):
        require_numpy()
        from neurobench.algorithms.cfar import robust_local_cfar

        frames, height, width = 3, 28, 34
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        gradient = (0.08 * x_grid + 0.03 * y_grid).astype(np.float32)
        video = np.repeat(gradient[None, :, :], frames, axis=0)
        video[1, 9, 7] += 8.0
        video[1, 20, 27] += 8.0

        result = robust_local_cfar(video, pfa=0.05, guard_px=1, training_radius_px=5)

        self.assertTrue(result["mask"][1, 9, 7])
        self.assertTrue(result["mask"][1, 20, 27])
        self.assertGreater(result["score"][1, 9, 7], result["threshold_z"])
        self.assertLess(result["active_fraction"], 0.02)

    def test_gamma_cfar_mask_returns_boolean_candidate_mask(self):
        require_numpy()
        from neurobench.algorithms.cfar import gamma_cfar_mask

        video = np.zeros((2, 16, 16), dtype=np.float32)
        video[0, 8, 8] = 5.0

        mask = gamma_cfar_mask(video, pfa=0.10, guard_px=1, training_radius_px=4)

        self.assertEqual(mask.shape, video.shape)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertTrue(mask[0, 8, 8])

    def test_cfar_rejects_invalid_probability_and_epsilon(self):
        require_numpy()
        from neurobench.algorithms.cfar import robust_local_cfar

        video = np.zeros((1, 8, 8), dtype=np.float32)
        for pfa in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(pfa=pfa):
                with self.assertRaisesRegex(ValueError, "pfa"):
                    robust_local_cfar(video, pfa=pfa)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            robust_local_cfar(video, epsilon=0.0)

    def test_cfar_rejects_invalid_training_ring(self):
        require_numpy()
        from neurobench.algorithms.cfar import robust_local_cfar

        video = np.zeros((1, 8, 8), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "training_radius_px"):
            robust_local_cfar(video, guard_px=3, training_radius_px=3)


if __name__ == "__main__":
    unittest.main()
