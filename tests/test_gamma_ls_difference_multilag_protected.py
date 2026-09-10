from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neurobench.experiments.gamma_ls_difference.gpu_representations import (
    DeviceRepresentationMap,
    V5_LAGS,
    frozen_v5_residual_group_representation,
    multilag_energy_normalized_difference_representation,
    pca_whitened_delay_total_energy_representation,
)
from neurobench.experiments.gamma_ls_difference.multilag_protected import (
    CONFIRMATION_SAMPLES,
    CS_PARZEN_BANDWIDTHS,
    MULTILAG_ARMS,
    SAMPLE_SEEDS,
    SCREEN_SAMPLES,
    _assert_acquisition_raw_output,
    _read_selection_checkpoint,
    _representation_for_arm,
    audit_historical_v5_surface,
    eligible_delay_current_frames,
    fit_cs_parzen_delay_embedding,
    fit_full_rank_delay_whitening,
    fit_from_dict,
    guard_safe_selection_bursts,
    multilag_design,
    sample_fold_delay_observations,
    select_fold_contexts,
    select_fold_models,
)
from neurobench.experiments.gamma_ls_difference.protected import ContextLane
from neurobench.experiments.gamma_ls_difference.screen import FoldContract
from neurobench.algorithms.multilag_msica import TemporalMSICAFit


def _fold() -> FoldContract:
    return FoldContract(
        training_fold=1,
        heldout_burst="1",
        training_bursts=("2", "3", "4"),
        heldout_guard_ui=(90, 120),
        quiet_half_a_ui=(1, 50),
        quiet_half_b_ui=(51, 89),
    )


def test_bounded_design_counts_and_raw_domain_are_explicit() -> None:
    design = multilag_design()
    assert design["lags"] == [0, 1, 2, 4, 8, 16]
    assert design["input_domain"] == "acquisition_raw"
    assert tuple(design["arms"]) == MULTILAG_ARMS
    assert design["unique_whitening_models"] == 12
    assert design["whitening_computations"] == 12
    assert design["cs_parzen_rotation_fits"] == 24
    assert design["gamma_selection_maps"] == 36
    assert design["protected_context_arm_cells"] == 12
    assert design["protected_operating_point_cells"] == 360
    assert design["pairwise_cs_objective_calls_total_upper_bound"] == 44_280


def test_delay_guard_checks_every_lag_not_only_current_frame() -> None:
    bursts = {"1": (100, 110), "2": (121, 140), "3": (150, 160), "4": (170, 180)}
    frames = eligible_delay_current_frames(_fold(), bursts)
    # t=121 is outside the held-out guard, but t-1 through t-16 are not all out.
    assert frames["2"].tolist() == [137, 138, 139, 140]
    assert frames["3"][0] == 150
    selection = guard_safe_selection_bursts(_fold(), bursts)
    assert selection["1"] == [100, 110]
    assert selection["2"] == [137, 140]
    assert selection["3"] == [150, 160]
    with pytest.raises(ValueError, match="exactly"):
        eligible_delay_current_frames(_fold(), bursts, lags=(0, 1, 2, 4, 8))


def test_fold_sampling_is_balanced_disjoint_deterministic_and_raw() -> None:
    frames, height, width = 240, 6, 8
    movie = (
        np.arange(frames, dtype=np.float64)[:, None, None] * 1000.0
        + np.arange(height * width, dtype=np.float64).reshape(1, height, width)
    ).astype(np.float32)
    bursts = {"1": (90, 110), "2": (130, 145), "3": (160, 175), "4": (190, 205)}
    mask = np.ones((height, width), dtype=bool)
    first = sample_fold_delay_observations(
        movie, fold=_fold(), bursts=bursts, anatomy_mask=mask, seed=17
    )
    second = sample_fold_delay_observations(
        movie, fold=_fold(), bursts=bursts, anatomy_mask=mask, seed=17
    )
    assert first.screen.shape == (6, SCREEN_SAMPLES)
    assert first.confirmation.shape == (6, CONFIRMATION_SAMPLES)
    np.testing.assert_array_equal(first.screen, second.screen)
    np.testing.assert_array_equal(first.confirmation, second.confirmation)
    assert first.manifest == second.manifest
    assert first.manifest["screen_confirmation_disjoint"] is True
    assert first.manifest["input_domain"] == "acquisition_raw"
    assert set(first.manifest["counts_by_training_burst"]) == {"2", "3", "4"}
    assert all(
        row["screen_samples"] == 128 and row["confirmation_samples"] == 128
        for row in first.manifest["counts_by_training_burst"].values()
    )
    # The synthetic camera value advances exactly 1000 per raw frame.
    np.testing.assert_allclose(first.screen[0] - first.screen[1], 1000.0)
    np.testing.assert_allclose(first.screen[0] - first.screen[5], 16_000.0)


def test_historical_surface_only_freezes_grid_basis_not_matrices(tmp_path: Path) -> None:
    rows = []
    for bandwidth, gain in ((0.25, 0.18), (0.35, 0.04)):
        token = str(bandwidth).replace(".", "p")
        rows.append(
            {
                "config_id": f"delay_embedding__cs_parzen__long__bandwidth-{token}",
                "formulation": "delay_embedding",
                "objective_family": "cs_parzen",
                "profile": "long",
                "parameter": {"bandwidth": bandwidth},
                "held_out_gain_fraction": gain,
                "fit": {
                    "lags": list(V5_LAGS),
                    "baseline_objective": 0.5,
                    "objective": 0.5 * (1.0 - gain),
                    "converged": True,
                    "demixing": np.eye(6).tolist(),
                    "diagnostics": {"accepted_updates": 3, "sweeps": 2},
                },
            }
        )
    path = tmp_path / "surface.json"
    path.write_text(
        json.dumps(
            {
                "complete": True,
                "selection_labels_used": False,
                "expansion_rows": rows,
            }
        ),
        encoding="utf-8",
    )
    audit = audit_historical_v5_surface(
        path,
        selected_config_id="delay_embedding__cs_parzen__long__bandwidth-0p25",
    )
    assert [row["bandwidth"] for row in audit["bounded_grid_basis"]] == [0.25, 0.35]
    assert audit["fit_matrices_loaded_for_protected_projection"] is False
    assert "demixing" not in json.dumps(audit)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection_labels_used"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="used labels"):
        audit_historical_v5_surface(
            path,
            selected_config_id="delay_embedding__cs_parzen__long__bandwidth-0p25",
        )


def test_context_role_is_exactly_one_per_fold() -> None:
    plan = {}
    for fold in range(1, 5):
        plan[fold] = (
            ContextLane(
                role="size_sufficient_candidate",
                context_id=f"fixture_{fold}",
                half_width_px=19,
                guard_radius_px=5,
                shape=9.0,
                mode_fraction_of_half_width=1.0,
                selection_source="fold_contexts.json",
            ),
            ContextLane(
                role="larger_support_comparator",
                context_id=f"larger_{fold}",
                half_width_px=31,
                guard_radius_px=7,
                shape=9.0,
                mode_fraction_of_half_width=1.0,
                selection_source="fold_contexts.json",
            ),
        )
    selected = select_fold_contexts(plan, role="size_sufficient_candidate")
    assert set(selected) == {1, 2, 3, 4}
    assert all(lane.half_width_px == 19 for lane in selected.values())
    with pytest.raises(ValueError, match="exactly one"):
        select_fold_contexts(plan, role="training_best_context")


def _selection_rows() -> list[dict[str, object]]:
    rows = []
    for bandwidth in CS_PARZEN_BANDWIDTHS:
        for seed_index, seed in enumerate(SAMPLE_SEEDS):
            rows.append(
                {
                    "training_fold": 1,
                    "bandwidth": bandwidth,
                    "sample_seed": seed,
                    "sample_identity_sha256": f"sample-{seed}",
                    "whitening_sha256": f"white-{seed}",
                    "pca_mean_positive_tail_contrast": float(seed_index),
                    "pca_minimum_quiet_swap_positive_tail_contrast": float(
                        seed_index
                    ),
                    "ica_mean_positive_tail_contrast": (
                        10.0 if bandwidth == 0.35 and seed_index == 1 else 1.0
                    ),
                    "ica_minimum_quiet_swap_positive_tail_contrast": 1.0,
                    "converged": True,
                    "numerical_clamps": 0,
                    "positive_coordinates_used": False,
                    "positive_identities_used": False,
                }
            )
    return rows


def test_pca_and_ica_fit_selection_are_independent_and_fail_closed() -> None:
    rows = _selection_rows()
    pca, ica = select_fold_models(rows, fold=1)
    assert pca["sample_seed"] == SAMPLE_SEEDS[2]
    assert pca["bandwidth"] == CS_PARZEN_BANDWIDTHS[0]
    assert ica["sample_seed"] == SAMPLE_SEEDS[1]
    assert ica["bandwidth"] == 0.35
    with pytest.raises(ValueError, match="exact frozen"):
        select_fold_models(rows[:-1], fold=1)
    rows = _selection_rows()
    for row in rows:
        row["converged"] = False
    with pytest.raises(ValueError, match="no converged"):
        select_fold_models(rows, fold=1)


def test_delay_fit_uses_full_rank_whitening_and_analytic_component_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neurobench.experiments.gamma_ls_difference.multilag_protected as module

    def objective(values: np.ndarray, bandwidth: float, **_: object) -> SimpleNamespace:
        covariance = float(np.cov(values.T, bias=True)[0, 1])
        return SimpleNamespace(objective=abs(covariance), numerical_clamps=0)

    monkeypatch.setattr(module, "cs_parzen_objective", objective)
    rng = np.random.default_rng(31)
    screen = rng.normal(size=(6, SCREEN_SAMPLES))
    confirmation = rng.normal(size=(6, CONFIRMATION_SAMPLES))
    fit = fit_cs_parzen_delay_embedding(
        screen,
        confirmation,
        bandwidth=0.25,
        backend="cpu",
        angle_step_degrees=45.0,
        max_sweeps=1,
    )
    assert fit.lags == V5_LAGS
    assert np.linalg.matrix_rank(fit.whitening) == 6
    assert len(fit.residual_indices) == 4
    assert set(fit.residual_indices) == set(range(6)) - {
        fit.persistence_index,
        fit.innovation_index,
    }
    effective = fit.rotation @ fit.whitening
    normalized = effective / np.linalg.norm(effective, axis=1, keepdims=True)
    common = np.ones(6) / np.sqrt(6)
    difference = np.asarray([1, -1, 0, 0, 0, 0]) / np.sqrt(2)
    assert fit.persistence_index == int(np.argmax(np.abs(normalized @ common)))
    remaining = [index for index in range(6) if index != fit.persistence_index]
    assert fit.innovation_index == max(
        remaining, key=lambda index: abs(float(normalized[index] @ difference))
    )
    assert fit.diagnostics["full_rotation_total_energy_relative_max_error"] < 1e-12
    restored = fit_from_dict(fit.to_dict())
    np.testing.assert_allclose(restored.demixing, fit.demixing)


def test_one_whitening_is_reused_across_the_two_bandwidth_rotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neurobench.experiments.gamma_ls_difference.multilag_protected as module

    def objective(values: np.ndarray, bandwidth: float, **_: object) -> SimpleNamespace:
        covariance = float(np.cov(values.T, bias=True)[0, 1])
        return SimpleNamespace(objective=abs(covariance), numerical_clamps=0)

    monkeypatch.setattr(module, "cs_parzen_objective", objective)
    rng = np.random.default_rng(37)
    screen = rng.normal(size=(6, SCREEN_SAMPLES))
    confirmation = rng.normal(size=(6, CONFIRMATION_SAMPLES))
    whitening = fit_full_rank_delay_whitening(screen, confirmation)
    fits = [
        fit_cs_parzen_delay_embedding(
            screen,
            confirmation,
            bandwidth=bandwidth,
            whitening_fit=whitening,
            backend="cpu",
            angle_step_degrees=45.0,
            max_sweeps=1,
        )
        for bandwidth in CS_PARZEN_BANDWIDTHS
    ]
    assert all(
        fit.diagnostics["whitening_reused_across_bandwidths"] is True
        for fit in fits
    )
    np.testing.assert_array_equal(fits[0].center, fits[1].center)
    np.testing.assert_array_equal(fits[0].whitening, fits[1].whitening)
    changed = screen.copy()
    changed[0, 0] += 1.0
    with pytest.raises(ValueError, match="exact samples"):
        fit_cs_parzen_delay_embedding(
            changed,
            confirmation,
            bandwidth=CS_PARZEN_BANDWIDTHS[0],
            whitening_fit=whitening,
            backend="cpu",
            angle_step_degrees=45.0,
            max_sweeps=1,
        )


def test_fit_selection_resume_is_bound_to_the_exact_model(
    tmp_path: Path,
) -> None:
    contract = {
        "training_fold": 1,
        "arm_model_sha256": "model-a",
        "input_domain": "acquisition_raw",
    }
    metric = {
        "mean_positive_tail_contrast": 0.3,
        "minimum_quiet_swap_positive_tail_contrast": 0.2,
        "quiet_swap_positive_tail_contrasts": [0.2, 0.4],
        "gamma_runtime_ms_per_frame": 1.0,
        "selection_uses_training_burst_windows": True,
        "positive_coordinates_used": False,
        "positive_identities_used": False,
    }
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps({"contract": contract, "metric": metric}),
        encoding="utf-8",
    )
    assert _read_selection_checkpoint(path, contract) == metric
    with pytest.raises(RuntimeError, match="contract changed"):
        _read_selection_checkpoint(
            path,
            {**contract, "arm_model_sha256": "model-b"},
        )


def test_every_representation_output_must_attest_shared_raw_domain() -> None:
    raw = torch.arange(20 * 2 * 3, dtype=torch.float32).reshape(20, 2, 3)
    source_ui = torch.arange(84, 104, dtype=torch.int64)
    review_ui = torch.arange(100, 104, dtype=torch.int64)
    representation = DeviceRepresentationMap(
        values=torch.ones((4, 2, 3), dtype=torch.float32),
        source_frame_indices=review_ui.clone(),
        diagnostics={"input_domain": "acquisition_raw"},
    )
    _assert_acquisition_raw_output(
        representation,
        raw=raw,
        source_frame_ui=source_ui,
        review_frame_ui=review_ui,
    )
    wrong = DeviceRepresentationMap(
        values=representation.values,
        source_frame_indices=representation.source_frame_indices,
        diagnostics={"input_domain": "causal_gaussian_then_ema"},
    )
    with pytest.raises(AssertionError, match="acquisition_raw"):
        _assert_acquisition_raw_output(
            wrong,
            raw=raw,
            source_frame_ui=source_ui,
            review_frame_ui=review_ui,
        )


def test_bounded_raw_domain_projection_matches_full_tensor_references() -> None:
    rng = np.random.default_rng(47)
    raw = torch.from_numpy(rng.normal(size=(37, 5, 7)).astype(np.float32))
    source_ui = torch.arange(80, 117, dtype=torch.int64)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    whitening = np.diag(np.linspace(0.5, 1.5, 6))
    fit = TemporalMSICAFit(
        formulation="delay_embedding",
        objective_family="cs_parzen",
        objective_parameter={"bandwidth": 0.25},
        lags=V5_LAGS,
        lag_weights=(),
        center=np.linspace(-0.2, 0.2, 6),
        whitening=whitening,
        rotation=q,
        demixing=q @ whitening,
        objective=0.2,
        baseline_objective=0.3,
        persistence_index=0,
        innovation_index=1,
        residual_indices=(2, 3, 4, 5),
        component_signs=(1, 1, 1, 1, 1, 1),
        converged=True,
        diagnostics={
            "component_rule": "analytic_fixture",
            "positive_coordinates_used": False,
            "positive_identities_used": False,
        },
    )
    frozen = {"fit": fit.to_dict()}
    expected = {
        "difference_multilag_energy_normalized": (
            multilag_energy_normalized_difference_representation(
                raw, source_frame_indices=source_ui
            )
        ),
        "pca_whitened_delay_total_energy": (
            pca_whitened_delay_total_energy_representation(
                raw, frozen, source_frame_indices=source_ui
            )
        ),
        "cs_parzen_delay_residual": frozen_v5_residual_group_representation(
            raw, frozen, source_frame_indices=source_ui
        ),
    }
    for arm, reference in expected.items():
        actual = _representation_for_arm(
            raw,
            source_ui,
            arm=arm,
            fit=None if arm == "difference_multilag_energy_normalized" else fit,
            chunk_frames=7,
        )
        torch.testing.assert_close(actual.values, reference.values, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(
            actual.source_frame_indices, reference.source_frame_indices
        )
        assert actual.diagnostics["input_domain"] == "acquisition_raw"
        assert actual.diagnostics["bounded_projection"] is True
