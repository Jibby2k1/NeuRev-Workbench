import numpy as np

from neurobench.algorithms.quiet_calibration import (
    EmpiricalQuietTail,
    QuietRobustStandardizer,
    energy_mapping_bank,
    group_energy,
)


def test_quiet_standardizer_uses_training_mask_only() -> None:
    video = np.zeros((6, 2, 2), dtype=np.float32)
    video[:4] = np.asarray([0, 1, 2, 3])[:, None, None]
    video[4:] = 1000
    fitted = QuietRobustStandardizer(minimum_samples=4).fit(
        video, np.asarray([True, True, True, True, False, False])
    )
    np.testing.assert_allclose(fitted.center_, 1.5)
    assert float(np.min(fitted.transform(video[4:]))) > 100


def test_empirical_tail_is_finite_monotone_and_add_one_smoothed() -> None:
    tail = EmpiricalQuietTail().fit(np.asarray([0.0, 1.0, 2.0]))
    probability = tail.survival_probability(np.asarray([-1.0, 1.0, 3.0]))
    np.testing.assert_allclose(probability, [1.0, 0.75, 0.25])
    surprise = tail.surprise(np.asarray([-1.0, 1.0, 3.0]), log_base=10)
    assert np.all(np.diff(surprise) >= 0)
    assert np.isfinite(surprise).all()


def test_energy_mappings_and_group_energy_are_nonnegative() -> None:
    values = np.asarray([[-2.0, 1.0], [3.0, -4.0]], dtype=np.float32)
    bank = energy_mapping_bank(values, bounded_kappa=1.0, huber_delta=1.5)
    np.testing.assert_array_equal(bank["signed"], values)
    for name in ("absolute", "raw_square", "bounded_square", "huber_energy"):
        assert np.all(bank[name] >= 0)
    np.testing.assert_allclose(group_energy(values, axis=1), [5.0, 25.0])
