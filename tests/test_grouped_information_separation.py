import numpy as np

from neurobench.algorithms.grouped_information_separation import (
    fit_group_energy_hsic_isa,
)


def test_grouped_hsic_isa_is_bounded_deterministic_and_nonincreasing() -> None:
    rng = np.random.default_rng(23)
    count = 160
    phase = np.linspace(0, 8 * np.pi, count)
    latent = np.stack([
        np.sin(phase), np.cos(phase),
        np.sign(np.sin(0.37 * phase)), np.abs(np.sin(0.37 * phase)),
    ])
    mixing = rng.normal(size=(6, 4))
    observations = mixing @ latent + 0.02 * rng.normal(size=(6, count))
    first = fit_group_energy_hsic_isa(
        observations, rank=4, group_size=2, angle_step_degrees=15,
        max_sweeps=2, max_fit_samples=96, seed=7,
    )
    second = fit_group_energy_hsic_isa(
        observations, rank=4, group_size=2, angle_step_degrees=15,
        max_sweeps=2, max_fit_samples=96, seed=7,
    )
    history = first.diagnostics["group_dependence_history"]
    assert first.method_id == "group_energy_hsic_isa"
    assert history[-1] <= history[0] + 1e-12
    assert first.diagnostics["qualification"].startswith("bounded_group")
    assert np.allclose(first.sources, second.sources)
