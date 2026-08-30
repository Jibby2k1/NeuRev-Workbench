from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import neurobench.experiments.neuron_identifiability.jepa_pilot as jepa_pilot_module
from neurobench.experiments.neuron_identifiability.jepa_data import RobustNormalization
from neurobench.experiments.neuron_identifiability.jepa_pilot import (
    JEPAMaskedPredictionErrorScorer,
    JEPAPilotError,
    MAEMaskedReconstructionErrorScorer,
    deterministic_full_coverage_masks,
    frozen_handcrafted_score,
    preflight_jepa_pilot,
    run_jepa_pilot,
)
from neurobench.experiments.neuron_identifiability.spatiotemporal_jepa import (
    MaskedPixelAutoencoder,
    SpatiotemporalJEPA,
    tiny_smoke_configuration,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "examples/spatiotemporal_jepa_representation_v1.example.json"


def _normalization() -> RobustNormalization:
    return RobustNormalization(
        center=100.0,
        scale=10.0,
        unscaled_mad=10.0 / 1.4826,
        scale_was_floored=False,
        fit_recording_ids=("060126_01_rest",),
        uniform_frame_indices=(("060126_01_rest", (0, 1)),),
        spatial_stride=1,
    )


def test_metadata_preflight_is_read_only_and_keeps_authorization_boundary(tmp_path: Path) -> None:
    output = tmp_path / "preflight-must-not-create"

    result = preflight_jepa_pilot(
        CONFIG,
        repository_root=REPOSITORY,
        output_root=output,
        device="cpu",
    )

    assert result["status"] == "passed_metadata_preflight"
    assert result["scientific_execution_ready"] is False
    assert "execution_authorized" in result["scientific_execution_blockers"]
    assert "live_preflight_performed" in result["scientific_execution_blockers"]
    assert "all_source_hashes_verified" in result["scientific_execution_blockers"]
    assert "registered_NREV_EXP_0025_motion_evidence_linked" in result["scientific_execution_blockers"]
    assert "registered_protocol_freeze_evidence_linked" in result["scientific_execution_blockers"]
    assert result["motion_audit"]["status"] == "not_run_metadata_only_preflight"
    assert result["checks"]["canonical_cuda_precision_is_bfloat16"] is True
    assert result["checks"]["float16_forbidden"] is True
    assert result["checks"]["implementation_hash_contract_matches"] is True
    assert not output.exists()
    assert not output.with_name(output.name + ".partial").exists()


def test_preflight_fails_closed_on_implementation_hash_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    sources = config["implementation_hash_contract"][
        "comparator_and_evaluator_suite"
    ]["sources"]
    next(row for row in sources if row["path"].endswith("jepa_training.py"))[
        "sha256"
    ] = "0" * 64
    drifted = tmp_path / "drifted-config.json"
    drifted.write_text(json.dumps(config), encoding="utf-8")
    result = preflight_jepa_pilot(
        drifted,
        repository_root=REPOSITORY,
        output_root=tmp_path / "unused-output",
        device="cpu",
    )
    assert result["status"] == "failed"
    assert result["checks"]["implementation_hash_contract_matches"] is False
    assert "implementation_hash_contract_matches" in result["failed_checks"]


def test_mask_schedule_is_deterministic_and_covers_every_token() -> None:
    encoder, mask = tiny_smoke_configuration()
    video = torch.zeros((1, 1, 4, 8, 8), dtype=torch.float32)

    first = deterministic_full_coverage_masks(
        video, mask, encoder.patch_size, seed=17
    )
    second = deterministic_full_coverage_masks(
        video, mask, encoder.patch_size, seed=17
    )

    assert torch.equal(first, second)
    counts = first.sum(dim=0)
    assert tuple(counts.shape) == (1, 2, 4, 4)
    assert int(counts.min()) >= 1


def test_learned_primary_scorers_apply_frozen_training_normalization() -> None:
    encoder, mask = tiny_smoke_configuration()
    jepa = SpatiotemporalJEPA(encoder, mask, initialization_seed=4)
    mae = MaskedPixelAutoencoder(encoder, mask, initialization_seed=4)
    raw = np.linspace(75, 125, 4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    normalization = _normalization()
    normalized = normalization.apply(raw)

    jepa_from_raw = JEPAMaskedPredictionErrorScorer(
        jepa, device="cpu", seed=81, normalization=normalization
    )(raw)
    jepa_from_normalized = JEPAMaskedPredictionErrorScorer(
        jepa, device="cpu", seed=81
    )(normalized)
    mae_from_raw = MAEMaskedReconstructionErrorScorer(
        mae, device="cpu", seed=81, normalization=normalization
    )(raw)
    mae_from_normalized = MAEMaskedReconstructionErrorScorer(
        mae, device="cpu", seed=81
    )(normalized)

    assert np.allclose(jepa_from_raw, jepa_from_normalized, rtol=1e-6, atol=1e-6)
    assert np.allclose(mae_from_raw, mae_from_normalized, rtol=1e-6, atol=1e-6)


def test_handcrafted_head_is_affine_invariant_and_source_off_fitted() -> None:
    rng = np.random.default_rng(29)
    movie = rng.normal(400, 7, size=(20, 16, 16)).astype(np.float32)

    baseline = frozen_handcrafted_score(movie)
    shifted_and_scaled = frozen_handcrafted_score(3.0 * movie + 200.0)

    assert np.allclose(shifted_and_scaled, baseline, rtol=2e-5, atol=2e-3)
    source_on = movie.copy()
    source_on[7:15, 6:10, 6:10] += 20.0
    fitted = frozen_handcrafted_score.fit_source_off(movie)
    assert np.allclose(fitted(movie), baseline)
    assert not np.allclose(fitted(source_on), frozen_handcrafted_score(source_on))


def _primary_table_rows() -> list[dict[str, object]]:
    methods = {
        "compact_jepa_latent_temporal_change": 0.8,
        "masked_pixel_autoencoder_latent_temporal_change": 0.6,
        "frozen_random_encoder_latent_temporal_change": 0.5,
        "frozen_handcrafted_stack": 0.7,
    }
    rows: list[dict[str, object]] = []
    for recording in ("recording-a", "recording-b"):
        for source_count in (1, 2, 4):
            fixture_id = f"{recording}-sources-{source_count}"
            for method, recall in methods.items():
                rows.append(
                    {
                        "training_seed": 1001,
                        "fixture_id": fixture_id,
                        "background_recording_id": recording,
                        "background_window_id": f"{recording}-window",
                        "injection_seed": 3101,
                        "source_count": source_count,
                        "method": method,
                        "source_on_recovery": {"recall": recall},
                    }
                )
    return rows


def test_primary_summary_aggregates_source_counts_and_bootstraps_selected_max() -> None:
    config = jepa_pilot_module.load_jepa_pilot_config(CONFIG)
    summary = jepa_pilot_module._primary_effect_summary(
        _primary_table_rows(),
        config=config,
        evaluated_training_seeds=[1001],
        full_registered_grid=False,
        bootstrap_draws=1_000,
    )
    seed = summary["training_seed_results"]["1001"]
    assert seed["strongest_registered_comparator"] == "frozen_handcrafted_stack"
    assert seed["paired_jepa_minus_strongest"]["observed_mean"] == pytest.approx(0.1)
    assert seed["paired_jepa_minus_strongest"][
        "strongest_comparator_reselected_inside_each_draw"
    ] is True
    assert len(seed["source_count_aggregated_cluster_rows"]) == 2
    assert all(
        row["source_counts_aggregated"] == 3
        for row in seed["source_count_aggregated_cluster_rows"]
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unbalanced"])
def test_primary_summary_rejects_missing_or_duplicate_fixture_method(mutation: str) -> None:
    config = jepa_pilot_module.load_jepa_pilot_config(CONFIG)
    rows = _primary_table_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[-1]))
    else:
        rows[-1] = {**rows[-1], "source_count": 99}
    expected = "exactly one primary" if mutation != "unbalanced" else "metadata"
    with pytest.raises(JEPAPilotError, match=expected):
        jepa_pilot_module._primary_effect_summary(
            rows,
            config=config,
            evaluated_training_seeds=[1001],
            full_registered_grid=False,
            bootstrap_draws=1_000,
        )


def test_smoke_writes_non_scientific_artifact_contract(tmp_path: Path) -> None:
    output = tmp_path / "jepa-smoke"

    summary = run_jepa_pilot(
        CONFIG,
        repository_root=REPOSITORY,
        mode="smoke",
        output_root=output,
        device="cpu",
        steps=1,
        max_training_seeds=1,
        injection_cells_per_background=1,
    )

    assert summary["status"] == "passed_non_scientific_smoke_contract_only"
    assert summary["scientific_completion"] is False
    assert summary["scientific_promotion_allowed"] is False
    assert summary["paired_injection"]["background_recording_ids"] == [
        "060126_10_rest",
        "060126_12_left",
        "060126_15_right",
        "spon_ca_burst_3_hindbrain_to_tail_488_20ms",
    ]
    for name in (
        "REPORT.md",
        "status.json",
        "summary.json",
        "validation.json",
        "llm_context.json",
        "artifact_index.json",
        "mask_sweep_coverage.json",
        "nuisance_audit.json",
        "paired_injection_results.tsv",
    ):
        assert (output / name).is_file()
    assert not output.with_name(output.name + ".partial").exists()
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["checks"]["all_mask_sweep_tokens_observed"] is True
    assert validation["checks"]["learned_scorers_apply_frozen_training_normalization"] is True
    assert validation["checks"]["scientific_audit_complete"] is False
    audit = json.loads((output / "scientific_audit_status.json").read_text(encoding="utf-8"))
    assert audit["scientific_audit_complete"] is False
    background = json.loads((output / "background_window_manifest.json").read_text(encoding="utf-8"))
    assert background["burst_interval_exclusion_used_for_spon"] is False
    assert background["burst_interval_use"] == "not_applicable_synthetic_smoke"
    assert background["roi_identity_labels_used"] is False
    nuisance_text = (output / "nuisance_audit.json").read_text(encoding="utf-8")
    assert "dtype_rail_fraction" in nuisance_text
    assert "unresolved_sensor_bit_depth_unknown" in nuisance_text

    with pytest.raises(JEPAPilotError, match="preflight failed"):
        run_jepa_pilot(
            CONFIG,
            repository_root=REPOSITORY,
            mode="smoke",
            output_root=output,
            device="cpu",
            steps=1,
        )


def test_failure_preserves_nonresumable_partial_with_explicit_claim_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "jepa-failed-smoke"

    def fail_fixture_build(*args: object, **kwargs: object) -> object:
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(jepa_pilot_module, "_make_fixtures", fail_fixture_build)
    with pytest.raises(RuntimeError, match="deliberate test failure"):
        run_jepa_pilot(
            CONFIG,
            repository_root=REPOSITORY,
            mode="smoke",
            output_root=output,
            device="cpu",
            steps=1,
        )

    partial = output.with_name(output.name + ".partial")
    status = json.loads((partial / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed_preserved_nonresumable_partial"
    assert status["resume_supported"] is False
    assert "same_manifest_resume_validation" in status["unresolved_claim_run_gates"]
