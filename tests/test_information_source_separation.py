import numpy as np

from neurobench.algorithms.information_source_separation import (
    fit_kernel_hsic_pairwise_rotation,
    fit_knn_mi_pairwise_rotation,
    fit_multilag_sobi,
    knn_mutual_information,
    normalized_hsic,
)
from neurobench.experiments.information_source_separation.synthetic import (
    make_spatiotemporal_fixture,
)
from neurobench.metrics.source_separation import (
    aligned_source_metrics,
    footprint_metrics,
    trace_fidelity_metrics,
)


def _ar_mixture(seed: int = 7, count: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sources = np.zeros((3, count), dtype=np.float64)
    innovations = np.stack([
        rng.laplace(size=count),
        rng.standard_t(df=5, size=count),
        rng.uniform(-np.sqrt(3), np.sqrt(3), size=count),
    ])
    for source, coefficient in enumerate((0.15, 0.70, 0.96)):
        for frame in range(1, count):
            sources[source, frame] = coefficient * sources[source, frame - 1] + innovations[source, frame]
    sources -= sources.mean(axis=1, keepdims=True)
    sources /= sources.std(axis=1, keepdims=True)
    mixing = np.asarray([
        [1.0, 0.4, -0.2],
        [0.2, 1.0, 0.5],
        [-0.3, 0.25, 1.0],
    ])
    return mixing @ sources, sources


def test_multilag_sobi_recovers_distinct_temporal_sources() -> None:
    observations, truth = _ar_mixture()
    result = fit_multilag_sobi(
        observations, rank=3, lags=(1, 2, 4, 8, 16), max_sweeps=100
    )
    metrics = aligned_source_metrics(truth, result.sources)
    assert result.method_id == "multilag_sobi"
    assert result.diagnostics["relative_subspace_closure_error"] < 1e-10
    assert metrics["mean_absolute_correlation"] > 0.94
    assert metrics["worst_absolute_correlation"] > 0.88


def test_bounded_information_rotators_report_qualified_finite_results() -> None:
    observations, _ = _ar_mixture(count=320)
    hsic = fit_kernel_hsic_pairwise_rotation(
        observations,
        rank=3,
        max_fit_samples=96,
        angle_step_degrees=15,
        max_sweeps=2,
    )
    mi = fit_knn_mi_pairwise_rotation(
        observations,
        rank=3,
        max_fit_samples=160,
        neighbors=5,
        angle_step_degrees=15,
        max_sweeps=2,
    )
    for result in (hsic, mi):
        assert np.isfinite(result.sources).all()
        assert np.isfinite(result.objective)
        assert result.diagnostics["qualification"] == "bounded_pairwise_rotation_reference"
        assert result.diagnostics["relative_subspace_closure_error"] < 1e-10


def test_dependence_estimators_distinguish_dependency() -> None:
    rng = np.random.default_rng(19)
    independent_left = rng.normal(size=180)
    independent_right = rng.normal(size=180)
    dependent_right = independent_left**2 + 0.05 * rng.normal(size=180)
    assert normalized_hsic(independent_left, dependent_right) > normalized_hsic(
        independent_left, independent_right
    )
    assert knn_mutual_information(independent_left, dependent_right) > knn_mutual_information(
        independent_left, independent_right
    )


def test_spatiotemporal_fixtures_close_and_include_unresolved_cases() -> None:
    for case_id in ("isolated", "overlap", "motion_edge", "saturation", "unresolved"):
        fixture = make_spatiotemporal_fixture(
            case_id, seed=13, frame_count=128, shape=(10, 12)
        )
        closure = (
            fixture.observation
            - fixture.background
            - fixture.neural_signal
            - fixture.structured_artifact
            - fixture.measurement_noise
        )
        assert float(np.max(np.abs(closure))) < 1e-5
        assert fixture.footprints.shape == (3, 10, 12)
        assert fixture.traces.shape == (3, 128)
    assert not make_spatiotemporal_fixture(
        "unresolved", seed=13, frame_count=128, shape=(10, 12)
    ).identifiable


def test_truth_metrics_handle_permutation_sign_scale_timing_and_footprints() -> None:
    truth = np.asarray([[0, 0, 1, 2, 1, 0], [0, 1, 0, 0, 0, 0]], dtype=float)
    estimate = np.stack([-3 * truth[1] + 2, 0.5 * truth[0] - 1])
    metrics = aligned_source_metrics(truth, estimate)
    assert metrics["mean_absolute_correlation"] > 0.999
    assert metrics["mean_aligned_nmse"] < 1e-12
    fidelity = trace_fidelity_metrics(truth[0], truth[0])
    assert fidelity["peak_retention"] == 1.0
    assert fidelity["area_retention"] == 1.0
    assert fidelity["onset_error_frames"] == 0
    footprint = np.zeros((9, 9), dtype=float)
    footprint[3:6, 3:6] = 1
    spatial = footprint_metrics(footprint, footprint)
    assert spatial["footprint_iou"] == 1.0
    assert spatial["centroid_error_px"] == 0.0
