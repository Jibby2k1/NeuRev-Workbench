from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest
import torch

from neurobench.experiments.gamma_ls_difference import gpu_representations
from neurobench.experiments.gamma_ls_difference import conditioning_sensitivity as sensitivity


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = (
    REPOSITORY
    / "examples/spon_ca_burst_gamma_ls_conditioning_sensitivity_v1.example.json"
)


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> sensitivity.ConditioningSensitivityConfig:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    return sensitivity.ConditioningSensitivityConfig.load(EXAMPLE)


def test_manifest_freezes_compact_nested_design(config) -> None:
    assert sensitivity.GAUSSIAN_SIGMAS == (0.0, 0.5, 1.0, 1.5)
    assert sensitivity.EMA_ALPHAS == (1.0, 2.0 / 3.0, 0.4, 0.25)
    assert sensitivity.SCALE_FLOOR_PERCENTILES == (5.0, 10.0, 20.0)
    assert len(sensitivity.enumerate_settings()) == 48
    assert len(sensitivity.conditioning_grid_rows()) == 144
    assert config.payload["downstream"]["protected_evaluation_in_this_run"] is False
    assert config.payload["scientific_audit"]["enabled"] is True


def test_manifest_rejects_changed_grid(tmp_path: Path, monkeypatch) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["design"]["gaussian_sigma_px"][-1] = 2.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    with pytest.raises(
        sensitivity.ConditioningSensitivityConfigError, match="sigma grid changed"
    ):
        sensitivity.ConditioningSensitivityConfig.load(changed)


def test_raw_lane_semantics_do_not_call_historical_conditioning_acquisition_raw() -> None:
    assert (
        sensitivity.raw_lane_semantics("raw", 0.0, 1.0)
        == "acquisition_raw_float32"
    )
    assert (
        sensitivity.raw_lane_semantics("raw", 1.0, 0.4)
        == "historically_conditioned_raw_lane"
    )
    assert (
        sensitivity.raw_lane_semantics("raw", 0.5, 1.0)
        == "conditioned_level_raw_representation_lane"
    )
    assert (
        sensitivity.raw_lane_semantics("difference_signed", 0.0, 1.0)
        == "not_a_raw_representation_lane"
    )


def test_sigma0_alpha1_is_exact_float32_acquisition_input() -> None:
    values = torch.arange(4 * 7 * 9, dtype=torch.float32).reshape(4, 7, 9)
    actual = sensitivity.causal_condition_dense(
        values, sigma_px=0.0, ema_alpha=1.0
    )
    torch.testing.assert_close(actual, values, rtol=0, atol=0)
    assert actual.data_ptr() != values.data_ptr()


def test_historical_anchor_matches_maintained_conditioner() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn((7, 19, 23), generator=generator, dtype=torch.float32)
    actual = sensitivity.causal_condition_dense(
        values, sigma_px=1.0, ema_alpha=0.4
    )
    expected = gpu_representations.causal_preprocess_common_input(values).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_sigma_chunking_preserves_all_four_exact_causal_ema_states(
    tmp_path: Path, monkeypatch
) -> None:
    movie = (
        torch.arange(11 * 7 * 9, dtype=torch.int64).reshape(11, 7, 9) % 4096
    ).to(torch.uint16).numpy()
    path = tmp_path / "movie.npy"
    import numpy as np

    np.save(path, movie, allow_pickle=False)
    monkeypatch.setattr(sensitivity, "EXPECTED_MOVIE_SHAPE", (11, 7, 9))
    monkeypatch.setattr(sensitivity, "REVIEW_INTERVAL_UI", (6, 11))
    monkeypatch.setattr(sensitivity, "RETAINED_INTERVAL_UI", (5, 11))
    outputs, timings = sensitivity._stream_sigma_all_ema_history_to_device(
        path,
        sigma_px=0.5,
        chunk_frames=4,
        device=torch.device("cpu"),
        heartbeat=None,
    )
    source = torch.from_numpy(movie.astype("float32"))
    for alpha in sensitivity.EMA_ALPHAS:
        expected = sensitivity.causal_condition_dense(
            source, sigma_px=0.5, ema_alpha=alpha
        )[4:]
        torch.testing.assert_close(outputs[alpha], expected, rtol=0, atol=0)
        assert timings[alpha][
            "candidate_runtime_charges_full_h2d_and_spatial_cost"
        ] is True


def _selector_rows(*, fold: int = 1, representation: str = "raw"):
    rows = []
    for index, setting in enumerate(sensitivity.enumerate_settings()):
        # The first row has the highest contrast. A deliberately slow clone of
        # its objectives below is dominated and must not enter the front.
        contrast = 1.0 - index / 1000.0
        rows.append(
            {
                "representation": representation,
                "training_fold": fold,
                "heldout_burst": str(fold),
                "support_context_id": f"ctx_fold_{fold}",
                **setting.as_dict(),
                "raw_lane_semantics": sensitivity.raw_lane_semantics(
                    representation,
                    setting.gaussian_sigma_px,
                    setting.causal_ema_alpha,
                ),
                "mean_event_positive_tail": 2.0 + contrast,
                "mean_quiet_positive_tail": 2.0,
                "mean_positive_tail_contrast": contrast,
                "minimum_swap_positive_tail_contrast": contrast - 0.01,
                "online_inference_runtime_ms_per_frame": 0.2 + index / 10000.0,
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "burst_windows_used": True,
            }
        )
    return rows


def test_selector_is_fold_representation_local_and_runtime_pareto() -> None:
    rows = _selector_rows()
    # Setting 1 gets identical contrast to setting 0 but is slower, so it is
    # strictly dominated by setting 0 in the three-objective contract.
    rows[1]["mean_positive_tail_contrast"] = rows[0][
        "mean_positive_tail_contrast"
    ]
    rows[1]["minimum_swap_positive_tail_contrast"] = rows[0][
        "minimum_swap_positive_tail_contrast"
    ]
    rows[1]["online_inference_runtime_ms_per_frame"] = 0.9
    selected, layers = sensitivity.select_training_setting(rows)
    assert selected["setting_id"] == rows[0]["setting_id"]
    assert layers[rows[0]["setting_id"]] == 0
    assert layers[rows[1]["setting_id"]] > 0


def test_selector_rejects_coordinate_or_identity_content() -> None:
    rows = _selector_rows()
    rows[0]["canonical_roi_id"] = "forbidden"
    with pytest.raises(
        sensitivity.ConditioningSensitivityUnavailable,
        match="coordinate/identity field",
    ):
        sensitivity.select_training_setting(rows)


def _support_candidate(fold: int) -> dict[str, object]:
    return {
        "context_id": f"ctx_fold_{fold}",
        "half_width_px": 11 + fold,
        "guard_radius_px": 5,
        "shape": 5.0,
        "mode_fraction_of_half_width": 0.5,
        "mode_radius_px": (11 + fold) * 0.5,
        "support": "radial_disk",
        "padding": "valid_renormalized_zero",
        "eligible_primary": True,
        "selection_basis": "fixture_training_only",
    }


def _write_support_artifact(root: Path, *, contaminated: bool = False) -> None:
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps({"status": "complete_support_screen_only"}), encoding="utf-8"
    )
    (root / "validation.json").write_text(
        json.dumps(
            {
                "status": (
                    "passed_support_screen_artifact_contract_scientific_audit_pending"
                )
            }
        ),
        encoding="utf-8",
    )
    folds = []
    for fold in sensitivity.OUTER_FOLDS:
        candidate = _support_candidate(fold)
        if contaminated and fold == 1:
            candidate["x_px"] = 12
        folds.append(
            {
                "training_fold": fold,
                "heldout_burst": str(fold),
                "support_candidate_context": candidate,
                "selection_uses_positive_coordinates": False,
                "selection_uses_positive_identities": False,
            }
        )
    (root / "fold_contexts.json").write_text(
        json.dumps(
            {
                "selection_scope": "outer_training_fold_only",
                "selection_uses_positive_coordinates": False,
                "selection_uses_positive_identities": False,
                "burst_windows_used": True,
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )
    (root / "artifact_index.json").write_text(
        json.dumps(sensitivity._artifact_index(root)), encoding="utf-8"
    )


def test_support_verifier_uses_each_fold_candidate_and_rejects_content(
    config, tmp_path
) -> None:
    clean = tmp_path / "clean"
    _write_support_artifact(clean)
    contexts, provenance = sensitivity._verify_support_screen(
        replace(config, support_screen_path=clean)
    )
    assert set(contexts) == set(sensitivity.OUTER_FOLDS)
    assert provenance["fold_support_candidate_context_ids"] == {
        str(fold): f"ctx_fold_{fold}" for fold in sensitivity.OUTER_FOLDS
    }

    contaminated = tmp_path / "contaminated"
    _write_support_artifact(contaminated, contaminated=True)
    with pytest.raises(
        sensitivity.ConditioningSensitivityUnavailable,
        match="coordinate/identity field",
    ):
        sensitivity._verify_support_screen(
            replace(config, support_screen_path=contaminated)
        )


def test_support_verifier_rejects_stale_index(config, tmp_path) -> None:
    root = tmp_path / "support"
    _write_support_artifact(root)
    (root / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        sensitivity.ConditioningSensitivityUnavailable,
        match="changed after freeze",
    ):
        sensitivity._verify_support_screen(replace(config, support_screen_path=root))


def _mock_execution() -> sensitivity.SensitivityExecution:
    fold_rows = []
    swap_rows = []
    timing_rows = []
    for fold in sensitivity.OUTER_FOLDS:
        for representation in sensitivity.REPRESENTATIONS:
            for index, setting in enumerate(sensitivity.enumerate_settings()):
                contrast = 1.0 - index / 1000.0
                minimum = contrast - 0.002
                common = {
                    "representation": representation,
                    "training_fold": fold,
                    "heldout_burst": str(fold),
                    "support_context_id": f"ctx_fold_{fold}",
                    **setting.as_dict(),
                    "raw_lane_semantics": sensitivity.raw_lane_semantics(
                        representation,
                        setting.gaussian_sigma_px,
                        setting.causal_ema_alpha,
                    ),
                }
                fold_rows.append(
                    {
                        **common,
                        "mean_event_positive_tail": 2.0 + contrast,
                        "mean_quiet_positive_tail": 2.0,
                        "mean_positive_tail_contrast": contrast,
                        "minimum_swap_positive_tail_contrast": minimum,
                        "online_inference_runtime_ms_per_frame": 0.1 + index / 10000,
                        "positive_coordinates_used": False,
                        "positive_identities_used": False,
                        "burst_windows_used": True,
                    }
                )
                for swap_index, swap in enumerate(sensitivity.QUIET_SWAPS):
                    swap_contrast = minimum if swap_index == 0 else contrast + 0.002
                    swap_rows.append(
                        {
                            **common,
                            "quiet_swap": swap,
                            "positive_tail_contrast": swap_contrast,
                            "positive_coordinates_used": False,
                            "positive_identities_used": False,
                            "burst_windows_used": True,
                        }
                    )
                timing_rows.append(
                    {
                        "representation": representation,
                        "training_fold": fold,
                        "setting_id": setting.setting_id,
                        "online_inference_runtime_ms_per_frame": 0.1
                        + index / 10000,
                    }
                )
    return sensitivity.SensitivityExecution(
        fold_rows=tuple(fold_rows),
        swap_rows=tuple(swap_rows),
        timing_rows=tuple(timing_rows),
        execution_summary={
            "resolved_device": "cuda:0",
            "dense_scoring_device": "cuda",
            "fold_metric_cells": 576,
            "quiet_swap_rows": 1152,
        },
    )


def _mock_preflight() -> dict[str, object]:
    return {
        "preflight_sha256": "1" * 64,
        "base_preflight_artifact_index_sha256": "2" * 64,
        "support_artifact_index_sha256": "3" * 64,
        "support_fold_contexts_sha256": "4" * 64,
        "runtime": {"resolved_device": "cuda:0"},
        "fold_contexts": {},
    }


def test_mock_run_writes_reconciled_selection_and_deterministic_seal(
    config, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sensitivity, "_verify_sensitivity_preflight", lambda *args, **kwargs: _mock_preflight()
    )
    monkeypatch.setattr(
        sensitivity, "_execute_device_sensitivity", lambda *args, **kwargs: _mock_execution()
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    summary_first = sensitivity.run_conditioning_sensitivity(
        config, sensitivity_preflight_dir=tmp_path / "preflight", output_dir=first
    )
    summary_second = sensitivity.run_conditioning_sensitivity(
        config, sensitivity_preflight_dir=tmp_path / "preflight", output_dir=second
    )
    assert summary_first["design_counts"]["fold_metric_cells"] == 576
    assert summary_first["design_counts"]["quiet_swap_rows"] == 1152
    assert summary_first["selection_sha256"] == summary_second["selection_sha256"]
    selection = json.loads(
        (first / "fold_selected_settings.json").read_text(encoding="utf-8")
    )
    assert len(selection["fold_representation_selections"]) == 12
    assert selection["positive_coordinates_used"] is False
    assert sensitivity.verify_selection_seal(selection) == selection["selection_sha256"]
    selection["fold_representation_selections"][0]["training_fold"] = 99
    with pytest.raises(ValueError, match="changed after sealing"):
        sensitivity.verify_selection_seal(selection)
    assert sensitivity._verify_indexed_artifact(first, role="test")[
        "verified_artifact_count"
    ] > 10


def test_run_fails_before_output_mutation_on_stale_preflight(
    config, tmp_path, monkeypatch
) -> None:
    output = tmp_path / "output"

    def stale(*args, **kwargs):
        raise sensitivity.ConditioningSensitivityUnavailable("fixture stale preflight")

    monkeypatch.setattr(sensitivity, "_verify_sensitivity_preflight", stale)
    with pytest.raises(
        sensitivity.ConditioningSensitivityUnavailable, match="stale preflight"
    ):
        sensitivity.run_conditioning_sensitivity(
            config,
            sensitivity_preflight_dir=tmp_path / "preflight",
            output_dir=output,
        )
    assert not output.exists()


def test_sensitivity_preflight_fails_before_output_on_stale_base_preflight(
    config, tmp_path, monkeypatch
) -> None:
    output = tmp_path / "new-preflight"

    def stale(*args, **kwargs):
        raise sensitivity.ConditioningSensitivityUnavailable("fixture stale base")

    monkeypatch.setattr(
        sensitivity, "_verify_base_preflight_without_annotation_reads", stale
    )
    with pytest.raises(
        sensitivity.ConditioningSensitivityUnavailable, match="stale base"
    ):
        sensitivity.run_sensitivity_preflight(
            config,
            base_preflight_dir=tmp_path / "base",
            output_dir=output,
        )
    assert not output.exists()


def test_noncolliding_output_fails_without_calling_executor(
    config, tmp_path, monkeypatch
) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    called = False

    def executor(*args, **kwargs):
        nonlocal called
        called = True
        return _mock_execution()

    monkeypatch.setattr(sensitivity, "_execute_device_sensitivity", executor)
    with pytest.raises(FileExistsError, match="output exists"):
        sensitivity.run_conditioning_sensitivity(
            config,
            sensitivity_preflight_dir=tmp_path / "preflight",
            output_dir=output,
        )
    assert called is False


def test_module_has_no_sparse_coordinate_or_identity_reader_api() -> None:
    source = inspect.getsource(sensitivity)
    assert "_read_sparse_positives" not in source
    assert 'source_paths["protected_labels_v1"]' not in source
    assert 'source_paths["latest_labels_v7"]' not in source
    assert "strict_separated_nms" not in source
