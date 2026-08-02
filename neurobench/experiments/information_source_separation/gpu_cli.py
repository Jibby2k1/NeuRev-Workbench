"""CLI for parity-gated CUDA source-separation screen execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import InformationSeparationConfig
from .gpu_screen import audit_cuda_screen, run_cuda_generated_screen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "resume"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    config = InformationSeparationConfig.load(args.config)
    if args.action == "preflight":
        payload = audit_cuda_screen(
            config, output_dir=args.output_dir, device=args.device
        )
    else:
        payload = run_cuda_generated_screen(
            config, output_dir=args.output_dir, device=args.device,
            resume=args.action == "resume",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
