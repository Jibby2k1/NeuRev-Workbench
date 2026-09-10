import json

import pytest

from neurobench.experiments.ica_whitening_evaluation.real_config import (
    RealDataConfig, RealDataConfigError,
)


def test_real_config_requires_nonblocking_amendment(tmp_path):
    payload = {
        "schema_version": 1, "experiment_id": "x",
        "parent_config": "parent.json", "synthetic_registry": "registry.tsv",
        "decision_amendment": "amendment.json", "output_dir": "out",
        "proposals": {
            "lanes": ["raw_activity", "signed_temporal_difference", "spatial_highpass"],
            "temporal_pool": "lme0.25", "per_lane_per_burst": 32,
            "proposal_nms_distance_px": 3, "union_separation_px": 3,
            "evaluation_budgets": [10, 20], "match_radius_px": 6,
        },
        "fitting": {"eligibility": "synthetic_all_numerically_resolved_only",
                    "maximum_fit_samples": 128, "activity_fraction": .5, "shard_count": 2},
        "controls": ["raw_direct", "signed_temporal_difference", "rank_matched_pca",
                     "random_rotation", "separable_control"],
        "synthetic_warning_non_blocking": False,
    }
    path = tmp_path / "config.json"; path.write_text(json.dumps(payload))
    with pytest.raises(RealDataConfigError, match="amendment"):
        RealDataConfig.from_json(path)
