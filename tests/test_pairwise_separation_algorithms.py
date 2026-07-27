import numpy as np

from neurobench.algorithms.pairwise_separation import (
    adaptive_difference,
    center_and_whiten_2d,
    cs_parzen_independence,
    estimate_quiet_gain,
    fit_shared_background_nmf,
    fit_cs_parzen_ica,
    fit_infomax_tanh_ica,
    fixed_difference,
    quiet_difference_stats,
    standardized_positive_mask,
)


def test_fixed_and_adaptive_difference_background_contract():
    rng = np.random.default_rng(2)
    base = rng.uniform(10, 100, size=(10, 8))
    frames = np.stack((base, base), axis=0)
    assert np.all(fixed_difference(frames)[1] == 0)
    alpha = 1.08
    pairs0 = np.stack([base + i for i in range(6)])
    pairs1 = alpha * pairs0
    estimate, diagnostics = estimate_quiet_gain(pairs0, pairs1)
    assert abs(estimate - alpha) < 1e-6
    sequence = np.stack((pairs0[0], pairs1[0]))
    assert np.mean(np.abs(adaptive_difference(sequence, estimate)[1])) < np.mean(np.abs(fixed_difference(sequence)[1]))
    assert diagnostics["rejected_pairs"] == 0


def test_quiet_floor_and_one_sided_binary_mask():
    differences = np.zeros((8, 3, 4), np.float32)
    differences[:, 1, 1] = np.arange(8)
    stats = quiet_difference_stats(differences)
    later = np.zeros((3, 3, 4), np.float32)
    later[1, 0, 0] = 100
    later[1, 0, 1] = -100
    _, mask = standardized_positive_mask(later, stats, 3, undefined_leading_frames=1)
    assert set(np.unique(mask)) <= {0, 1}
    assert mask[1, 0, 0] == 1 and mask[1, 0, 1] == 0
    assert not mask[0].any()


def test_whitening_identifies_full_rank_and_flags_rank_one():
    rng = np.random.default_rng(5)
    samples = rng.normal(size=(2, 2000)); samples[1] += 0.4 * samples[0]
    z, fit = center_and_whiten_2d(samples)
    np.testing.assert_allclose(z @ z.T / z.shape[1], np.eye(2), atol=1e-6)
    assert fit.identifiable
    _, rank_one = center_and_whiten_2d(np.vstack((samples[0], samples[0])))
    assert not rank_one.identifiable


def test_cs_parzen_is_block_size_invariant():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(2, 128))
    a, _ = cs_parzen_independence(values, 0.35, block_rows=17)
    b, _ = cs_parzen_independence(values, 0.35, block_rows=64)
    assert np.isfinite(a)
    assert abs(a - b) < 1e-12


def test_infomax_and_cs_recover_non_gaussian_sources_up_to_ambiguity():
    rng = np.random.default_rng(42)
    sources = np.vstack((
        rng.laplace(size=800),
        rng.uniform(-1.7, 1.7, size=800),
    ))
    mixed = np.asarray([[1.0, 0.45], [0.3, 1.0]]) @ sources
    whitened, _ = center_and_whiten_2d(mixed)
    infomax = fit_infomax_tanh_ica(
        whitened, max_iterations=500, learning_rate=0.03, tolerance=1e-7
    )
    cs = fit_cs_parzen_ica(
        whitened[:, :128], whitened[:, :256], bandwidth=0.35,
        block_rows=32, screen_step_degrees=5,
        refine_half_width_degrees=3, refine_step_degrees=0.5,
    )
    for fit in (infomax, cs):
        correlations = np.abs(np.corrcoef(fit.demixing @ whitened, sources)[:2, 2:])
        assert np.min(np.max(correlations, axis=1)) > 0.9


def test_constrained_nmf_is_nonnegative_and_ignores_negative_change():
    previous = np.ones((12, 9), np.float32)
    current = previous.copy(); current[3:5, 4:7] += 0.8
    fit = fit_shared_background_nmf(previous, current, 1.0, activity_l1=0.05)
    assert fit.background.min() >= 0 and fit.activity.min() >= 0
    assert fit.activity[3:5, 4:7].mean() > 0
    decay = fit_shared_background_nmf(previous, previous * 0.8, 1.0, activity_l1=0.05)
    assert not decay.activity.any()
