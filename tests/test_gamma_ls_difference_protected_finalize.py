from __future__ import annotations

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.protected import (
    _macro_metric,
    clustered_bootstrap_contrasts,
)
from neurobench.experiments.gamma_ls_difference.protected_finalize import (
    ORIGINAL_FAILURE_FRAGMENT,
    _bootstrap_weight_matrix,
    _original_scalar_failure,
    vectorized_clustered_bootstrap_contrasts,
)


def _randomized_match_rows(seed: int = 7351) -> list[dict[str, object]]:
    """Small complete protected grain with the real 26-identity denominator."""

    rng = np.random.default_rng(seed)
    identities = [f"roi-{index:02d}" for index in range(26)]
    arms = ("learned", "control_a", "control_b")
    rows: list[dict[str, object]] = []
    for identity_index, identity in enumerate(identities):
        burst = identity_index % 4 + 1
        latent_rank = int(rng.integers(1, 111))
        for arm_index, arm in enumerate(arms):
            arm_rank = latent_rank + 7 * arm_index + int(rng.integers(-4, 5))
            for budget in (20, 40, 58, 80, 100):
                rows.append(
                    {
                        "cohort": "protected_v1",
                        "context_role": "fixture_context",
                        "representation": arm,
                        "quiet_swap": "quiet_a_train_quiet_b_eval",
                        "nms_distance_px": 6,
                        "target_nms_peaks_per_pseudo_burst": 1.0,
                        "burst_id": burst,
                        "candidate_budget": budget,
                        "canonical_roi_id": identity,
                        "matched": bool(arm_rank <= budget),
                    }
                )
    return rows


def test_vectorized_bootstrap_reproduces_scalar_rows_and_rng() -> None:
    rows = _randomized_match_rows()
    controls = ("control_a", "control_b")
    scalar = clustered_bootstrap_contrasts(
        rows,
        learned_arm="learned",
        controls=controls,
        seed=991,
        replicates=31,
    )
    vectorized = vectorized_clustered_bootstrap_contrasts(
        rows,
        learned_arm="learned",
        controls=controls,
        seed=991,
        replicates=31,
    )
    assert len(scalar) == len(vectorized) == 4
    floating_fields = {
        "learned_budget_auc",
        "control_budget_auc",
        "budget_auc_delta",
        "budget_auc_delta_ci95_low",
        "budget_auc_delta_ci95_high",
        "learned_b58_macro_recall",
        "control_b58_macro_recall",
        "b58_macro_recall_delta",
        "b58_delta_ci95_low",
        "b58_delta_ci95_high",
    }
    for scalar_row, vectorized_row in zip(scalar, vectorized, strict=True):
        assert scalar_row.keys() == vectorized_row.keys()
        for field in scalar_row:
            if field in floating_fields:
                assert vectorized_row[field] == pytest.approx(
                    scalar_row[field], rel=0.0, abs=2e-15
                )
            else:
                assert vectorized_row[field] == scalar_row[field]

    identities = sorted({str(row["canonical_roi_id"]) for row in rows})
    expected_rng = np.random.default_rng(991)
    expected = np.zeros((31, 26), dtype=np.float64)
    lookup = {identity: index for index, identity in enumerate(identities)}
    for replicate in range(31):
        sampled = expected_rng.choice(identities, size=26, replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        for identity, count in zip(unique.tolist(), counts.tolist(), strict=True):
            expected[replicate, lookup[identity]] = count
    np.testing.assert_array_equal(
        _bootstrap_weight_matrix(identities, seed=991, replicates=31),
        expected,
    )


def test_vectorized_path_removes_observed_type_weight_failure() -> None:
    rows = _randomized_match_rows()
    identities = sorted({str(row["canonical_roi_id"]) for row in rows})
    with pytest.raises(
        TypeError,
        match=r"unsupported operand type\(s\) for \*: 'type' and 'bool'",
    ):
        _macro_metric(
            rows,
            arm="learned",
            cluster_weights={identity: int for identity in identities},
        )

    result = vectorized_clustered_bootstrap_contrasts(
        rows,
        learned_arm="learned",
        controls=("control_a", "control_b"),
        seed=17,
        replicates=7,
    )
    assert len(result) == 4
    assert all(row["cluster_count"] == 26 for row in result)


def test_vectorized_bootstrap_rejects_non_boolean_join_rows() -> None:
    rows = _randomized_match_rows()
    rows[0]["matched"] = "False"
    with pytest.raises(TypeError, match="in-memory boolean"):
        vectorized_clustered_bootstrap_contrasts(
            rows,
            learned_arm="learned",
            controls=("control_a", "control_b"),
            seed=5,
            replicates=3,
        )


def test_original_failure_survives_a_failed_finalizer_retry() -> None:
    original = f'TypeError("{ORIGINAL_FAILURE_FRAGMENT}")'
    assert (
        _original_scalar_failure(
            {
                "status": "interrupted_or_failed_resumable",
                "error": original,
            }
        )
        == original
    )
    assert (
        _original_scalar_failure(
            {
                "status": "finalizer_failed_resumable",
                "error": "a later disk error",
                "original_failure": original,
            }
        )
        == original
    )
    assert _original_scalar_failure({"status": "complete"}) is None
