import numpy as np

from neurobench.experiments.unsupervised_ica_eval.pc_mitl_benchmark import (
    apply_whitener,
    evaluate_cell,
    evaluate_fixed_cell,
    fit_train_whitener,
    generate_fixture,
    matched_recovery,
)


def test_t10_matching_resolves_permutation_and_sign():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(100, 4))
    recovered = reference[:, [2, 0, 3, 1]] * np.asarray([-1, 1, -1, 1])
    result = matched_recovery(reference, recovered)
    assert np.isclose(result["mean_absolute_correlation"], 1.0)
    assert np.isclose(result["minimum_absolute_correlation"], 1.0)


def test_t11_generator_is_deterministic_and_has_independent_source_streams():
    first = generate_fixture(sample_count=128, component_count=4, source_family="sparse_calcium", condition_number=5, noise_std=0.1, seed=7)
    second = generate_fixture(sample_count=128, component_count=4, source_family="sparse_calcium", condition_number=5, noise_std=0.1, seed=7)
    assert first.fingerprint == second.fingerprint
    assert np.array_equal(first.sources, second.sources)
    assert first.sources.shape == first.event_mask.shape == (128, 4)
    assert first.mixing.shape == (4, 4)
    assert all(not np.array_equal(first.event_mask[:, 0], first.event_mask[:, column]) for column in range(1, 4))
    assert np.isclose(np.linalg.cond(first.mixing), 5.0)


def test_t12_whitening_fits_train_only_and_cell_is_paired():
    fixture = generate_fixture(sample_count=128, component_count=2, source_family="laplace", condition_number=2, noise_std=0.02, seed=8)
    state = fit_train_whitener(fixture.observed[:76])
    changed = fixture.observed.copy(); changed[76:] += 1000
    changed_state = fit_train_whitener(changed[:76])
    assert state.train_data_hash == changed_state.train_data_hash
    assert np.array_equal(state.mean, changed_state.mean)
    whitened = apply_whitener(fixture.observed, state)
    assert np.allclose(whitened[:76].mean(axis=0), 0, atol=1e-12)
    rows = evaluate_cell(fixture, bandwidth_candidates=[1.0], train_stop=76, validation_stop=102, steps=3, learning_rate=0.02)
    assert {row["method"] for row in rows} == {"cs_parzen", "matrix_tc_alpha2"}
    assert len({row["fixture_fingerprint"] for row in rows}) == 1
    assert len({row["train_data_hash"] for row in rows}) == 1


def test_confirmation_uses_locked_bandwidths_without_validation_selection():
    fixture = generate_fixture(sample_count=128, component_count=4, source_family="laplace", condition_number=2, noise_std=0.02, seed=9)
    locked = {"cs_parzen": 1.0, "matrix_tc_alpha2": 0.5}
    rows = evaluate_fixed_cell(fixture, method_bandwidths=locked, train_stop=90, steps=3, learning_rate=0.02)
    assert {row["method"]: row["locked_bandwidth"] for row in rows} == locked
    assert all(np.isfinite(row["test_mean_absolute_correlation"]) for row in rows)
