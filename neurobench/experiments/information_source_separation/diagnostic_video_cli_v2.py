"""Even-dimension compatibility wrapper for the diagnostic MP4 suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration_config import CalibrationConfig
from . import diagnostic_videos as videos


def _even_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return np.pad(
        frame,
        ((0, height % 2), (0, width % 2), (0, 0)),
        mode="constant",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    original = videos._generated_frame
    videos._generated_frame = lambda *values, **kwargs: _even_frame(
        original(*values, **kwargs)
    )
    payload = videos.generate_diagnostic_suite(
        CalibrationConfig.load(args.config),
        calibration_root=args.calibration_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
