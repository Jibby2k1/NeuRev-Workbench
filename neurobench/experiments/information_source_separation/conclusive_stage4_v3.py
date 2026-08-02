"""Corrected temporal-resampling contract for native Stage 4."""
from __future__ import annotations

import argparse
import json

from . import conclusive_stage4 as stage4_v1
from . import conclusive_stage4_v2 as stage4_v2
from .conclusive_config import ConclusiveBatchConfig
from .semi_synthetic_v2 import make_real_background_fixture_v2


def run(config, *, maximum_fits=None):
    stage4_v1.make_real_background_fixture = make_real_background_fixture_v2
    return stage4_v2.run(config, maximum_fits=maximum_fits)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-fits", type=int)
    args=parser.parse_args(argv)
    print(json.dumps(run(ConclusiveBatchConfig.load(args.config),
                         maximum_fits=args.maximum_fits), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
