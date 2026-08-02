import numpy as np

from neurobench.experiments.information_source_separation.parzen_native import (
    fit_spatial_stochastic_parzen_noisy_posterior,
)


def test_native_parzen_track_is_finite_deterministic_and_closes():
    rng = np.random.default_rng(4)
    movie = rng.normal(size=(32, 9, 9)).astype(np.float32)
    movie[20:25, 3:6, 3:6] += 3
    kwargs = dict(quiet_frames=16, patch_size=5, rank=3, noise_scale=1.0,
                  seed=8, device="cpu", sample_count=256)
    left = fit_spatial_stochastic_parzen_noisy_posterior(movie, **kwargs)
    right = fit_spatial_stochastic_parzen_noisy_posterior(movie, **kwargs)
    assert np.isfinite(left["signal"]).all()
    assert np.array_equal(left["signal"], right["signal"])
    assert np.max(np.abs(movie-left["signal"]-left["remainder"])) < 1e-6
    assert left["method_id"] != "exact_infomax"
