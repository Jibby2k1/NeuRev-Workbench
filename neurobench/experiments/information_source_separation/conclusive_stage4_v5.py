"""Native-best Stage 4 with a CNMF-E-credible 32x32 spatial field."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json

from . import conclusive_stage4 as stage4_v1
from . import conclusive_stage4_v4 as stage4_v4
from .config import InformationSeparationConfig as ScientificConfig
from .conclusive_config import ConclusiveBatchConfig


class _ExpandedNativeConfigLoader:
    @staticmethod
    def load(path):
        config=ScientificConfig.load(path)
        return replace(config, semi_synthetic={**config.semi_synthetic,
                                               "crop_size_px":32})


def run(config, *, maximum_fits=None):
    stage4_v1.InformationSeparationConfig=_ExpandedNativeConfigLoader
    return stage4_v4.run(config, maximum_fits=maximum_fits)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--maximum-fits",type=int)
    args=parser.parse_args(argv)
    print(json.dumps(run(ConclusiveBatchConfig.load(args.config),
                         maximum_fits=args.maximum_fits),indent=2,sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
