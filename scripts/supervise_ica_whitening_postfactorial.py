#!/usr/bin/env python3
"""Wait for the complete factorial, then merge, analyze, and confirm finalists."""
from __future__ import annotations

import argparse
import json
import time

from neurobench.experiments.ica_whitening_evaluation.real_analysis import (
    merge_single_stage_whitening,
)
from neurobench.experiments.ica_whitening_evaluation.real_config import RealDataConfig
from neurobench.experiments.ica_whitening_evaluation.real_factorial_analysis import (
    run_complete_factorial_analysis,
)
from neurobench.experiments.ica_whitening_evaluation.real_finalist_confirmation import (
    run_finalist_confirmation,
)
from neurobench.experiments.ica_whitening_evaluation.real_scientific_audit import (
    run_scientific_audit,
)
from neurobench.experiments.ica_whitening_evaluation.independent_confirmation import (
    run_independent_confirmation,
)
from neurobench.experiments.ica_whitening_evaluation.completion_audit import (
    write_completion_audit,
)
from neurobench.experiments.ica_whitening_evaluation.concluding_report import (
    write_concluding_report,
)


GEOMETRIES = (
    "spatial", "temporal", "joint_spatiotemporal",
    "spatial_then_temporal", "temporal_then_spatial",
)


def _all_validated(config: RealDataConfig) -> bool:
    for geometry in GEOMETRIES:
        tag = geometry.upper()
        for shard in range(config.fitting.shard_count):
            path = (config.output_dir / "stages"
                    / f"S2C_{tag}_WHITENING_SHARD_{shard:02d}_OF_{config.fitting.shard_count:02d}"
                    / "validation.json")
            if not path.is_file() or json.loads(path.read_text()).get("status") != "pass":
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    config = RealDataConfig.from_json(args.config)
    while not _all_validated(config):
        time.sleep(args.poll_seconds)
    for geometry in GEOMETRIES:
        destination = (config.output_dir / "stages"
                       / f"S2C_{geometry.upper()}_WHITENING_MERGED")
        if not destination.exists():
            merge_single_stage_whitening(config, whitening_geometry=geometry)
    s3 = config.output_dir / "stages" / "S3_COMPLETE_FACTORIAL_ANALYSIS"
    if not s3.exists():
        run_complete_factorial_analysis(config)
    s4 = config.output_dir / "stages" / "S4_FINALIST_CONFIRMATION"
    if not (s4 / "validation.json").exists():
        run_finalist_confirmation(config, preflight_dir=args.preflight)
    s5 = config.output_dir / "stages" / "S5_SCIENTIFIC_AUDIT"
    if not (s5 / "inventory.json").exists():
        run_scientific_audit(config, preflight_dir=args.preflight)
    independent_contract = config.output_dir.parent / "independent_confirmation_contract_v1"
    s6 = config.output_dir / "stages" / "S6_INDEPENDENT_CONFIRMATION"
    if independent_contract.is_dir() and not (s6 / "validation.json").exists():
        run_independent_confirmation(config, contract_dir=independent_contract)
    write_completion_audit(
        config.output_dir.parent,
        config.output_dir / "completion_audit_latest.json",
    )
    write_concluding_report(
        config.output_dir.parent,
        config.output_dir / "CONCLUDING_REPORT.md",
    )


if __name__ == "__main__":
    main()
