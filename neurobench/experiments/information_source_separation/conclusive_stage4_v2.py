"""Missing-source correction for native semi-synthetic Stage 4."""
from __future__ import annotations

import argparse
import json

from . import conclusive_stage4 as stage4_v1
from .conclusive_config import ConclusiveBatchConfig


_base_run_method = stage4_v1._run_method
_base_valid = stage4_v1._valid


def run_method(fixture, method, scientific, config, artifact_dir):
    metrics, execution = _base_run_method(
        fixture, method, scientific, config, artifact_dir
    )
    if method["method_id"] in {"caiman_cnmf", "caiman_cnmfe"}:
        component_count = int(execution["diagnostics"].get(
            "component_count", len(metrics.get("matches", []))
        ))
        metrics["matched_source_fraction"] = min(
            component_count, len(fixture.traces)
        ) / len(fixture.traces)
    else:
        metrics["matched_source_fraction"] = 1.0
    return metrics, execution


def valid(metrics, config):
    return bool(
        _base_valid(metrics, config)
        and metrics.get("matched_source_fraction", 0.0) == 1.0
    )


def run(config, *, maximum_fits=None):
    stage4_v1._run_method = run_method
    stage4_v1._valid = valid
    return stage4_v1.run(config, maximum_fits=maximum_fits)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-fits", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(run(ConclusiveBatchConfig.load(args.config),
                         maximum_fits=args.maximum_fits), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
