"""Command-line entry point for the Gamma-LS difference/ICA experiment."""
from __future__ import annotations

import argparse
import json

from .config import GammaLSDifferenceConfig
from .preflight import run_preflight, verify_matching_preflight
from .protected import (
    ProtectedRepresentationUnavailable,
    run_protected_representation_experiment,
)
from .screen import CudaScreenUnavailable, run_gpu_screen
from .smoke import CudaSmokeUnavailable, run_cuda_smoke
from .streaming_benchmark import (
    DEFAULT_ARRIVAL_QUEUE_CAPACITY,
    DEFAULT_SOURCE_RING_FRAMES,
    MINIMUM_PRODUCTION_DURATION_SECONDS,
    STREAMING_ARMS,
    StreamingBenchmarkUnavailable,
    run_streaming_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the GPU Gamma-LS ablation and run implemented stages."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="Audit sources, resources, and CUDA without running the grid."
    )
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--artifact-dir", required=True)
    smoke = subparsers.add_parser(
        "smoke", help="Run the bounded label-free GPU pipeline smoke after ready preflight."
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--preflight-dir", required=True)
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--start-frame-ui", type=int, default=1800)
    smoke.add_argument("--frame-count", type=int, default=32)
    screen = subparsers.add_parser(
        "screen",
        help=(
            "Run burst-window-supervised, sparse-coordinate-free fold-local "
            "G1/G2 screening after ready preflight."
        ),
    )
    screen.add_argument("--config", required=True)
    screen.add_argument("--preflight-dir", required=True)
    screen.add_argument("--output-dir", required=True)
    protected = subparsers.add_parser(
        "protected",
        help=(
            "Refit the exact outer-fold two-frame PCA/CS-Parzen grid, seal "
            "cross-fitted candidates, and evaluate protected v1 plus v7 sensitivity."
        ),
    )
    protected.add_argument("--config", required=True)
    protected.add_argument("--preflight-dir", required=True)
    protected.add_argument("--context-selection-dir", required=True)
    protected.add_argument("--output-dir", required=True)
    streaming = subparsers.add_parser(
        "streaming-benchmark",
        help=(
            "Run paced 1-kHz causal GPU lanes for at least 60 seconds each, "
            "plus a separately identified batch-throughput frontier."
        ),
    )
    streaming.add_argument("--config", required=True)
    streaming.add_argument("--preflight-dir", required=True)
    streaming.add_argument("--screen-dir", required=True)
    streaming.add_argument("--output-dir", required=True)
    streaming.add_argument(
        "--context-id", default="gamma_h11_g5_n9_m1"
    )
    streaming.add_argument(
        "--arms",
        nargs="+",
        choices=STREAMING_ARMS,
        default=list(STREAMING_ARMS),
    )
    streaming.add_argument(
        "--duration-seconds",
        type=float,
        default=MINIMUM_PRODUCTION_DURATION_SECONDS,
    )
    streaming.add_argument(
        "--queue-capacity", type=int, default=DEFAULT_ARRIVAL_QUEUE_CAPACITY
    )
    streaming.add_argument(
        "--source-ring-frames", type=int, default=DEFAULT_SOURCE_RING_FRAMES
    )
    streaming.add_argument("--stream-start-ui", type=int, default=1900)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = GammaLSDifferenceConfig.load(args.config)
    if args.command == "preflight":
        payload = run_preflight(config, artifact_dir=args.artifact_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "screen":
        try:
            payload = run_gpu_screen(
                config,
                preflight_dir=args.preflight_dir,
                output_dir=args.output_dir,
                device=str(config.payload["resources"]["device"]),
            )
        except (RuntimeError, CudaScreenUnavailable) as error:
            print(
                json.dumps(
                    {
                        "status": "blocked_gpu_preflight_or_runtime",
                        "output_mutated": False,
                        "reason": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 3
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "protected":
        try:
            payload = run_protected_representation_experiment(
                config,
                preflight_dir=args.preflight_dir,
                context_selection_dir=args.context_selection_dir,
                output_dir=args.output_dir,
                device=str(config.payload["resources"]["device"]),
            )
        except (RuntimeError, ProtectedRepresentationUnavailable) as error:
            print(
                json.dumps(
                    {
                        "status": "blocked_gpu_preflight_or_runtime",
                        "output_mutated": False,
                        "reason": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 3
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "smoke":
        try:
            verify_matching_preflight(config, args.preflight_dir, require_gpu_ready=True)
            payload = run_cuda_smoke(
                config.source_paths["movie"],
                source_id=str(config.payload["sources"]["movie"]),
                output_dir=args.output_dir,
                start_frame_ui=args.start_frame_ui,
                frame_count=args.frame_count,
                device=str(config.payload["resources"]["device"]),
                spatial_sigma_px=float(
                    config.payload["preprocessing"]["gaussian_sigma_px"]
                ),
                ema_alpha=float(config.payload["preprocessing"]["ema_alpha"]),
            )
        except (RuntimeError, CudaSmokeUnavailable) as error:
            print(
                json.dumps(
                    {
                        "status": "blocked_gpu_preflight_or_runtime",
                        "output_mutated": False,
                        "reason": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 3
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "streaming-benchmark":
        try:
            payload = run_streaming_benchmark(
                config,
                preflight_dir=args.preflight_dir,
                screen_dir=args.screen_dir,
                output_dir=args.output_dir,
                context_id=args.context_id,
                arms=args.arms,
                duration_seconds=args.duration_seconds,
                arrival_interval_ms=float(
                    config.payload["efficiency"]["streaming_deadline_ms"]
                ),
                arrival_queue_capacity=args.queue_capacity,
                source_ring_frames=args.source_ring_frames,
                stream_start_ui=args.stream_start_ui,
                device=str(config.payload["resources"]["device"]),
            )
        except (RuntimeError, StreamingBenchmarkUnavailable) as error:
            print(
                json.dumps(
                    {
                        "status": "blocked_gpu_preflight_or_runtime",
                        "output_mutated": False,
                        "reason": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 3
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
