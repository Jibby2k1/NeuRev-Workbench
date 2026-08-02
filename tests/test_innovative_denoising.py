import numpy as np

from neurobench.algorithms.advanced_denoising import dense_ica_denoise
from neurobench.algorithms.innovative_denoising import (
    apply_blindspot_linear_model,
    bounded_mixture,
    cross_scale_consensus_shrinkage,
    fit_blindspot_linear_model,
    graph_edge_aware_diffusion,
    local_noise_psd_wiener,
    morphology_conditioned_shrinkage,
    pareto_front_indices,
    selected_component_nmf,
    select_diverse_pareto_rows,
    tempered_residual_posterior,
)
from neurobench.algorithms.spatial_patch_ica import SpatialPatchICAModel


def _movie() -> np.ndarray:
    rng = np.random.default_rng(41)
    values = 0.3 * rng.normal(size=(24, 24, 26))
    yy, xx = np.ogrid[:24, :26]
    blob = np.exp(-((xx - 13) ** 2 + (yy - 12) ** 2) / 8)
    values[10:18] += np.linspace(0.2, 2, 8)[:, None, None] * blob
    return values.astype(np.float32)


def test_local_morphology_graph_and_consensus_outputs_align() -> None:
    source = _movie()
    outputs = [
        local_noise_psd_wiener(
            source,
            quiet_count=8,
            tile_size=16,
            overlap_fraction=0.5,
            noise_multiplier=0.75,
            frequency_smoothing_sigma=1,
            transfer_floor=0.1,
            alpha=0.5,
        ),
        morphology_conditioned_shrinkage(
            source,
            center_sigma_px=1,
            ring_sigma_px=2,
            crowd_sigma_px=4,
            isolated_threshold_z=0.5,
            crowded_threshold_z=0.75,
            gate_temperature_z=0.25,
            gain_floor=0.25,
            alpha=0.5,
        ),
        graph_edge_aware_diffusion(
            source,
            quiet_count=8,
            radius=1,
            signal_bandwidth_z=1,
            guide_bandwidth_z=1,
            iterations=1,
            alpha=0.5,
            device="cpu",
            frame_batch_size=4,
        ),
        cross_scale_consensus_shrinkage(
            source,
            spatial_scales_px=[0.75, 1.5, 3],
            agreement_power=2,
            evidence_threshold_z=0.75,
            gain_floor=0.25,
            alpha=0.5,
        ),
    ]
    for output in outputs:
        assert output.shape == source.shape
        assert np.isfinite(output).all()
        assert not np.array_equal(output, source)


def test_selected_nmf_and_blindspot_are_reproducible_and_bounded() -> None:
    source = _movie()
    first, diagnostics = selected_component_nmf(
        source,
        window_frames=8,
        rank=2,
        iterations=4,
        minimum_spatial_concentration=0.02,
        minimum_temporal_dynamics=0.25,
        selection_temperature=0.1,
        alpha=0.25,
        seed=7,
        device="cpu",
    )
    second, _ = selected_component_nmf(
        source,
        window_frames=8,
        rank=2,
        iterations=4,
        minimum_spatial_concentration=0.02,
        minimum_temporal_dynamics=0.25,
        selection_temperature=0.1,
        alpha=0.25,
        seed=7,
        device="cpu",
    )
    np.testing.assert_allclose(first, second)
    assert 0 <= diagnostics["component_keep_mean"] <= 1
    model = fit_blindspot_linear_model(
        source,
        radius=1,
        sample_count=512,
        ridge=0.01,
        seed=3,
        fit_frame_count=8,
    )
    blind = apply_blindspot_linear_model(
        source,
        model,
        alpha=0.5,
        correction_limit_z=0.4,
        device="cpu",
        frame_batch_size=4,
    )
    assert np.max(np.abs(blind - source)) <= 0.20001


def test_tempered_posterior_and_bounded_mixture_keep_identity_limits() -> None:
    source = _movie()
    posterior = 0.25 * source
    tempered = tempered_residual_posterior(
        source,
        posterior,
        activity_threshold_z=1,
        temperature_z=0.5,
        posterior_authority=0.5,
        correction_limit_z=0.4,
    )
    assert np.max(np.abs(tempered - source)) <= 0.20001
    mixture = bounded_mixture(
        source,
        [0.5 * source, 0.75 * source],
        [0.25, 0.25],
        correction_limit_z=0.3,
    )
    assert np.max(np.abs(mixture - source)) <= 0.30001
    np.testing.assert_allclose(
        bounded_mixture(source, [posterior], [0], correction_limit_z=1),
        source,
    )


def test_pareto_front_and_diverse_selection_are_stable() -> None:
    rows = [
        {"id": "a", "recall": 0.7, "noise": 0.5, "selection_score": 1.0},
        {"id": "b", "recall": 0.8, "noise": 0.8, "selection_score": 1.1},
        {"id": "c", "recall": 0.6, "noise": 0.9, "selection_score": 0.5},
        {"id": "d", "recall": 0.75, "noise": 0.4, "selection_score": 1.2},
    ]
    assert pareto_front_indices(
        rows, maximize=["recall"], minimize=["noise"]
    ) == [1, 3]
    selected = select_diverse_pareto_rows(
        rows,
        maximize=["recall"],
        minimize=["noise"],
        count=2,
    )
    assert {row["id"] for row in selected} == {"b", "d"}


def test_asymmetric_component_dynamics_is_causal_and_finite() -> None:
    source = _movie()[:, :12, :14]
    patch = 3
    dimension = patch * patch
    model = SpatialPatchICAModel(
        patch_size=patch,
        rank=dimension,
        patch_mean=np.zeros(dimension, dtype=np.float32),
        analysis_filters=np.eye(dimension, dtype=np.float32),
        synthesis_atoms=np.eye(dimension, dtype=np.float32),
        component_scale=np.ones(dimension, dtype=np.float32),
        explained_variance_ratio=np.full(
            dimension, 1 / dimension, dtype=np.float32
        ),
        fastica_iterations=1,
        fastica_converged=True,
        fastica_final_delta=0,
    )
    output, diagnostics = dense_ica_denoise(
        source,
        model,
        mode="asymmetric",
        asymmetric_rise_gain=0.9,
        asymmetric_decay_gain=0.1,
        asymmetric_innovation_threshold_z=0.5,
        asymmetric_innovation_temperature_z=0.25,
        device="cpu",
    )
    assert output.shape == source.shape
    assert np.isfinite(output).all()
    assert diagnostics["frame_batch_size"] == 1
