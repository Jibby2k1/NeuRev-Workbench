"""Failure-isolated final runner for corrected native Stage 4."""
from __future__ import annotations

import argparse
import json

from . import conclusive_stage4_v2 as stage4_v2
from . import conclusive_stage4_v3 as stage4_v3
from .conclusive_config import ConclusiveBatchConfig


_base_run_method = stage4_v2.run_method


def safe_run_method(fixture, method, scientific, config, artifact_dir):
    try:
        return _base_run_method(fixture, method, scientific, config, artifact_dir)
    except Exception as exc:
        return {}, {"converged": False, "reported_method_id": method["method_id"],
                    "unresolved": True,
                    "diagnostics": {"failure_type": type(exc).__name__,
                                    "failure_message": str(exc)[:2000],
                                    "preserved_artifact_path": str(artifact_dir)}}


def run(config, *, maximum_fits=None):
    stage4_v2.run_method = safe_run_method
    return stage4_v3.run(config, maximum_fits=maximum_fits)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-fits", type=int)
    args=parser.parse_args(argv)
    print(json.dumps(run(ConclusiveBatchConfig.load(args.config),
                         maximum_fits=args.maximum_fits), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
