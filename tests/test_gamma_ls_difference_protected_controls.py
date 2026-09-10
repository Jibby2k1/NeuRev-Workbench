from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neurobench.experiments.gamma_ls_difference.config import GammaLSDifferenceConfig
from neurobench.experiments.gamma_ls_difference.protected import FIXED_ARMS, QUIET_SWAPS
from neurobench.experiments.gamma_ls_difference.screen import FoldContract
from neurobench.experiments.gamma_ls_difference import protected_controls as controls


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _seal_fixture(root: Path) -> None:
    _write_json(root / "artifact_index.json", controls._artifact_index(root))


def _support_fixture(root: Path) -> None:
    root.mkdir()
    _write_json(
        root / "summary.json",
        {
            "status": "complete_support_screen_only",
            "named_controls_executed": [
                {
                    "control_id": control_id,
                    "eligible_primary": False,
                }
                for control_id in (
                    "legacy_exact_n9_mode35_w23_eps64",
                    *controls.CONTROL_IDS,
                )
            ],
        },
    )
    _write_json(
        root / "validation.json",
        {
            "status": "passed_support_screen_artifact_contract_scientific_audit_pending",
            "checks": {"three_named_controls_executed": True},
        },
    )
    _write_json(
        root / "claim_boundary.json",
        {
            "protected_recall_computed": False,
            "positive_coordinates_used": False,
            "positive_identities_used": False,
        },
    )
    _write_json(root / "fold_contexts.json", {"fixture": True})
    swaps = []
    for control_id in controls.CONTROL_IDS:
        for representation in FIXED_ARMS:
            for fold in range(1, 5):
                for swap in QUIET_SWAPS:
                    swaps.append(
                        {
                            "context_id": control_id,
                            "representation": representation,
                            "training_fold": fold,
                            "quiet_swap": swap,
                            "positive_tail_contrast": 0.1 + fold / 100,
                            "positive_coordinates_used": False,
                            "positive_identities_used": False,
                        }
                    )
    _write_tsv(root / "control_quiet_swap_rows.tsv", swaps)
    latency = [
        {
            "family": "radial_gamma_ls",
            "context_id": "radial-h11",
            "half_width_px": 11,
            "p50_ms": 1.8,
        },
        {
            "family": "signed_square_annulus_ls",
            "context_id": controls.CONTROL_IDS[0],
            "half_width_px": 11,
            "p50_ms": 1.6,
        },
        {
            "family": "maintained_positive_clipped_box_cfar",
            "context_id": controls.CONTROL_IDS[1],
            "half_width_px": 11,
            "p50_ms": 0.2,
        },
    ]
    _write_tsv(root / "repeated_single_frame_latency.tsv", latency)
    _seal_fixture(root)


def _folds() -> tuple[FoldContract, ...]:
    return tuple(
        FoldContract(
            training_fold=fold,
            heldout_burst=str(fold),
            training_bursts=tuple(str(item) for item in range(1, 5) if item != fold),
            heldout_guard_ui=(100 + 10 * fold, 110 + 10 * fold),
            quiet_half_a_ui=(1, 50),
            quiet_half_b_ui=(51, 100),
        )
        for fold in range(1, 5)
    )


def _protected_fixture(
    root: Path,
    *,
    fold_contexts_sha256: str,
    radial_role: str = "size_sufficient_candidate",
) -> None:
    root.mkdir()
    roles = {
        str(fold): [
            {
                "role": radial_role,
                "support": "radial_disk",
                "padding": "valid_renormalized_zero",
            }
        ]
        for fold in range(1, 5)
    }
    _write_json(
        root / "run_contract.json",
        {
            "run_type": "protected_outer_fold_two_frame_representation",
            "movie_sha256": "a" * 64,
            "context_selection": {
                "source_sha256": fold_contexts_sha256,
                "positive_coordinates_used": False,
                "positive_identities_used": False,
                "context_roles_by_fold": roles,
            },
        },
    )
    _write_json(
        root / "candidate_seal.json",
        {"positive_coordinates_used": False, "positive_identities_used": False},
    )
    for filename in (
        "candidates_label_sealed.tsv",
        "threshold_calibration.tsv",
        "protected_v1_observation_matches.tsv",
    ):
        (root / filename).write_text("fixture\n", encoding="utf-8")
    _write_json(root / "summary.json", {"label_derived": True})
    _write_json(root / "validation.json", {"label_derived": True})
    _write_json(root / "claim_boundary.json", {"label_derived": True})
    _seal_fixture(root)


def _match_rows(*, radial: bool) -> list[dict[str, object]]:
    rows = []
    role = "radial" if radial else "simple-control"
    for swap in QUIET_SWAPS:
        for burst in range(1, 5):
            for budget in (20, 40, 58, 80, 100):
                for identity in ("roi-a", "roi-b"):
                    rows.append(
                        {
                            "cohort": "protected_v1",
                            "context_role": role,
                            "representation": "raw",
                            "quiet_swap": swap,
                            "nms_distance_px": 6,
                            "target_nms_peaks_per_pseudo_burst": 1.0,
                            "burst_id": burst,
                            "candidate_budget": budget,
                            "observation_id": f"{burst}-{identity}",
                            "canonical_roi_id": identity,
                            "matched": True if radial else identity == "roi-a",
                        }
                    )
    return rows


def test_support_audit_requires_protected_comparison_and_reports_speed(tmp_path: Path) -> None:
    root = tmp_path / "support"
    _support_fixture(root)
    audit = controls.audit_support_control_need(root)
    assert audit["decision"] == "protected_sparse_positive_comparison_required"
    assert len(audit["controls"]) == 2
    positive_box = next(
        row for row in audit["controls"] if row["control_id"] == controls.CONTROL_IDS[1]
    )
    assert positive_box["positive_contrast_cell_count"] == 24
    assert positive_box["repeated_single_frame_p50_ms"] == pytest.approx(0.2)
    assert audit["radial_h11_repeated_single_frame_p50_ms"] == pytest.approx(1.8)


def test_support_audit_fails_closed_on_indexed_content_change(tmp_path: Path) -> None:
    root = tmp_path / "support"
    _support_fixture(root)
    with (root / "claim_boundary.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed size|changed content"):
        controls.audit_support_control_need(root)


def test_protected_preseal_gate_does_not_open_label_derived_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    support = tmp_path / "fold_contexts.json"
    support.write_text("fixture", encoding="utf-8")
    evidence = {"fold_contexts_sha256": controls._sha256(support)}
    root = tmp_path / "protected"
    _protected_fixture(root, fold_contexts_sha256=evidence["fold_contexts_sha256"])
    protected_names = {
        "protected_v1_observation_matches.tsv",
        "summary.json",
        "validation.json",
        "claim_boundary.json",
    }
    original = Path.read_text

    def guarded(path: Path, *args, **kwargs):
        if path.name in protected_names:
            raise AssertionError(f"opened protected content before control seal: {path.name}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    metadata = controls._protected_index_metadata(
        root,
        radial_context_role="size_sufficient_candidate",
        support_evidence=evidence,
    )
    assert metadata["label_derived_artifacts_opened_before_control_seal"] is False


@pytest.mark.parametrize("control_id", controls.CONTROL_IDS)
def test_control_candidates_use_complete_crossfit_grid_without_labels(
    control_id: str,
) -> None:
    rng = np.random.default_rng(17)
    values = torch.from_numpy(rng.normal(size=(180, 29, 31)).astype(np.float32))
    # Make held-out windows visibly non-quiet without referring to coordinates as labels.
    values[110:145, 14, 15] += 12.0
    frame_ui = np.arange(1, 181, dtype=np.int64)
    bursts = {
        "1": [111, 114],
        "2": [121, 124],
        "3": [131, 134],
        "4": [141, 144],
    }
    candidates, calibrations, timing = controls.build_control_candidates(
        values,
        representation_name="raw",
        control_id=control_id,
        frame_ui=frame_ui,
        folds=_folds(),
        bursts=bursts,
        scale_floor_percentile=10.0,
        chunk_frames=32,
    )
    assert len(calibrations) == 4 * 2 * 3 * 5
    assert {
        (row["training_fold"], row["quiet_swap"], row["nms_distance_px"])
        for row in calibrations
    } == {
        (fold, swap, nms)
        for fold in range(1, 5)
        for swap in QUIET_SWAPS
        for nms in (4, 6, 8)
    }
    assert candidates
    assert all(row["interpretation_before_label_join"] == "unknown_candidate" for row in candidates)
    assert timing["positive_coordinates_used"] is False
    assert set(timing["score_hashes_by_quiet_swap"]) == set(QUIET_SWAPS)


def test_candidate_hash_binds_candidate_scores() -> None:
    row = {
        "training_fold": 1,
        "burst_id": 1,
        "context_role": "control",
        "representation": "raw",
        "quiet_swap": QUIET_SWAPS[0],
        "nms_distance_px": 6,
        "target_nms_peaks_per_pseudo_burst": 1.0,
        "threshold_z": 2.0,
        "candidate_rank": 1,
        "occupancy_score": 0.5,
        "x_px": 4,
        "y_px": 5,
    }
    before = controls._candidate_hashes([row])
    changed = {**row, "occupancy_score": 0.6}
    after = controls._candidate_hashes([changed])
    assert before["candidate_universe_sha256"] == after["candidate_universe_sha256"]
    assert before["candidate_score_stream_sha256"] != after["candidate_score_stream_sha256"]


def test_clustered_bootstrap_is_exactly_paired_and_reproducible() -> None:
    radial = _match_rows(radial=True)
    simple = _match_rows(radial=False)
    first = controls.paired_radial_control_bootstrap(
        radial,
        simple,
        seed=9,
        replicates=200,
        expected_cluster_count=2,
    )
    second = controls.paired_radial_control_bootstrap(
        radial,
        simple,
        seed=9,
        replicates=200,
        expected_cluster_count=2,
    )
    assert first == second
    assert len(first) == 3  # two quiet swaps plus their crossfit average
    assert all(row["radial_minus_control_b58_macro_recall"] == pytest.approx(0.5) for row in first)
    assert all(row["radial_minus_control_budget_auc"] == pytest.approx(0.5) for row in first)
    assert all(row["automatic_pipeline_selection_authorized"] is False for row in first)


def test_clustered_bootstrap_rejects_unpaired_observation_universe() -> None:
    radial = _match_rows(radial=True)
    simple = _match_rows(radial=False)[:-1]
    with pytest.raises(ValueError, match="exact paired universe"):
        controls.paired_radial_control_bootstrap(
            radial,
            simple,
            replicates=10,
            expected_cluster_count=2,
        )


def test_quiet_burden_comparison_preserves_empirical_not_pfa_boundary() -> None:
    common = {
        "training_fold": 1,
        "heldout_burst": 1,
        "representation": "raw",
        "quiet_swap": QUIET_SWAPS[0],
        "nms_distance_px": 6,
        "target_nms_peaks_per_pseudo_burst": 1.0,
    }
    radial = [{**common, "context_role": "radial", "heldout_quiet_peaks_per_pseudo_burst": 0.75}]
    simple = [{**common, "context_role": "simple", "heldout_quiet_peaks_per_pseudo_burst": 1.0}]
    rows = controls.quiet_burden_comparison_rows(radial, simple)
    assert rows[0]["control_minus_radial_quiet_burden"] == pytest.approx(0.25)
    assert rows[0]["empirical_quiet_burden_only_not_pfa"] is True


def test_explicit_boolean_parser_never_treats_false_string_as_truthy() -> None:
    assert controls._parse_bool("False", field="matched") is False
    assert controls._parse_bool("TRUE", field="matched") is True
    with pytest.raises(ValueError, match="explicit true/false"):
        controls._parse_bool("unknown", field="matched")


def test_runner_fails_before_any_output_mutation_when_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    config = GammaLSDifferenceConfig.load(EXAMPLE)
    output = tmp_path / "protected-controls"

    def blocked(*args, **kwargs):
        raise RuntimeError("stale fixture preflight")

    monkeypatch.setattr(controls, "verify_matching_preflight", blocked)
    with pytest.raises(RuntimeError, match="stale fixture preflight"):
        controls.run_protected_control_comparison(
            config,
            preflight_dir=tmp_path / "preflight",
            support_dir=tmp_path / "support",
            protected_dir=tmp_path / "protected",
            output_dir=output,
            radial_context_role="size_sufficient_candidate",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".*protected-controls*"))
