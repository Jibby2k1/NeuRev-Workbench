"""Frozen equivalence-margin correction for selective-risk Stage 2."""
from __future__ import annotations

import argparse
import json
from typing import Any

from . import conclusive_stage2 as stage2_v1
from .conclusive_config import ConclusiveBatchConfig


def selected_methods(stage1: dict[str, Any]) -> list[dict[str, Any]]:
    """Require convergence, then prefer lower rank within the 0.01 margin."""
    selected = []
    for method_id in sorted({row["method_id"] for row in stage1["summaries"]}):
        rows = [row for row in stage1["summaries"] if row["method_id"] == method_id]
        converged = [row for row in rows if row["converged_fraction"] >= 0.95]
        if not converged:
            continue
        best_recovery = max(row["mean_absolute_correlation"] for row in converged)
        equivalent = [row for row in converged
                      if row["mean_absolute_correlation"] >= best_recovery-0.01]
        equivalent.sort(key=lambda row: (
            int(row["parameters"].get("rank", 10**6)),
            len(row["parameters"].get("lags", [])),
            row["mean_absolute_crosstalk"],
            json.dumps(row["parameters"], sort_keys=True),
        ))
        selected.append({"method_id": method_id,
                         "parameters": equivalent[0]["parameters"]})
    return selected


def run(config: ConclusiveBatchConfig, *, maximum_rows: int | None = None):
    stage2_v1._selected_methods = selected_methods
    return stage2_v1.run(config, maximum_rows=maximum_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-rows", type=int)
    args = parser.parse_args(argv)
    payload = run(ConclusiveBatchConfig.load(args.config), maximum_rows=args.maximum_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
