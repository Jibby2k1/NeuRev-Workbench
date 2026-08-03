import numpy as np

from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_noise import (
    fit_joint_quiet_noise_model,
    joint_cs_divergence,
    qualify_noise_candidate,
)


def test_joint_quiet_model_retains_correlation_and_whitens():
    rng = np.random.default_rng(7)
    common = rng.normal(size=4000)
    views = {
        "scale_5": common + 0.2 * rng.normal(size=4000),
        "scale_7": 0.8 * common + 0.2 * rng.normal(size=4000),
        "scale_15": 0.4 * common + 0.3 * rng.normal(size=4000),
    }
    joint = fit_joint_quiet_noise_model(views)
    independent = fit_joint_quiet_noise_model(views, model_kind="independent_scale_noise")
    assert abs(joint.covariance[0, 1]) > 0.2
    assert independent.covariance[0, 1] == 0
    whitened_covariance = joint.inverse_sqrt @ joint.covariance @ joint.inverse_sqrt
    np.testing.assert_allclose(whitened_covariance, np.eye(3), atol=1e-5)
    aligned = np.column_stack([views[key] for key in joint.view_ids])
    assert joint_cs_divergence(aligned, joint) < 0.1


def test_noise_qualification_requires_every_check():
    keys = {
        "joint_cs_divergence", "covariance_error", "temporal_acf_energy",
        "spatial_acf_energy", "event_locked_energy", "motion_edge_correlation",
        "closure_max_normalized",
    }
    limits = {key: 0.1 for key in keys}
    passing = {key: 0.05 for key in keys}
    assert qualify_noise_candidate(passing, limits) == "qualified_measurement_noise"
    passing["event_locked_energy"] = 0.2
    assert qualify_noise_candidate(passing, limits) == "noise_candidate"
