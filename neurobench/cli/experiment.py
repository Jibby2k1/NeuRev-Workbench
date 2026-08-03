"""Thin CLI for focused, manifest-driven experiment workflows."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_DEFAULT_CPU_THREADS = 2
_MAX_CPU_THREADS = 8


def add_experiment_subcommands(subparsers) -> None:
    """Register experiment workflows without importing heavy runtime modules."""
    root = subparsers.add_parser(
        "experiment",
        help="Preflight and run focused resource-bounded experiments.",
        description="Focused resource-bounded experiment workflows.",
    )
    workflows = root.add_subparsers(dest="experiment_workflow", required=True)
    difference = workflows.add_parser(
        "frame-difference",
        help="Generate globally normalized signed derivative TIFF stacks.",
    )
    difference_actions = difference.add_subparsers(dest="experiment_action", required=True)
    difference_preflight = difference_actions.add_parser("preflight", help="Validate inputs and output resources.")
    difference_preflight.add_argument("--config", required=True)
    difference_preflight.set_defaults(func=_run_frame_difference_preflight)
    difference_run = difference_actions.add_parser("run", help="Write chunked derivative BigTIFF stacks.")
    difference_run.add_argument("--config", required=True)
    difference_run.set_defaults(func=_run_frame_difference)
    smoothed_difference = workflows.add_parser(
        "smoothed-frame-difference",
        help="Generate smoothed signed derivatives with global and quiet-MAD views.",
    )
    smoothed_actions = smoothed_difference.add_subparsers(dest="experiment_action", required=True)
    smoothed_preflight = smoothed_actions.add_parser("preflight", help="Validate smoothing and TIFF resource bounds.")
    smoothed_preflight.add_argument("--config", required=True)
    smoothed_preflight.set_defaults(func=_run_smoothed_difference_preflight)
    smoothed_run = smoothed_actions.add_parser("run", help="Write four smoothed derivative BigTIFFs.")
    smoothed_run.add_argument("--config", required=True)
    smoothed_run.set_defaults(func=_run_smoothed_difference)
    activity_gate = workflows.add_parser(
        "activity-gate",
        help="Generate bounded derivative-energy and artifact-attenuated review TIFFs.",
    )
    activity_actions = activity_gate.add_subparsers(dest="experiment_action", required=True)
    activity_preflight = activity_actions.add_parser("preflight", help="Validate review interval and resources.")
    activity_preflight.add_argument("--config", required=True)
    activity_preflight.set_defaults(func=_run_activity_gate_preflight)
    activity_run = activity_actions.add_parser("run", help="Write four activity-gated review TIFFs.")
    activity_run.add_argument("--config", required=True)
    activity_run.set_defaults(func=_run_activity_gate)
    activity_benchmark = workflows.add_parser(
        "activity-gate-benchmark",
        help="Compare Raw Direct with offline and causal artifact-gated inputs.",
    )
    benchmark_actions = activity_benchmark.add_subparsers(dest="experiment_action", required=True)
    benchmark_preflight = benchmark_actions.add_parser("preflight", help="Validate labels, inputs, and resource bounds.")
    benchmark_preflight.add_argument("--config", required=True)
    benchmark_preflight.set_defaults(func=_run_activity_gate_benchmark_preflight)
    benchmark_run = benchmark_actions.add_parser("run", help="Run the six-lane paired detection comparison.")
    benchmark_run.add_argument("--config", required=True)
    benchmark_run.set_defaults(func=_run_activity_gate_benchmark)
    proposal = workflows.add_parser(
        "causal-proposal-program",
        help="Run the checkpointed Spon Ca Burst causal proposal breadth/depth program.",
    )
    proposal_actions = proposal.add_subparsers(dest="experiment_action", required=True)
    proposal_preflight = proposal_actions.add_parser(
        "preflight", help="Validate the preregistered design, inputs, labels, and resource headroom."
    )
    proposal_preflight.add_argument("--config", required=True)
    proposal_preflight.set_defaults(func=_run_causal_proposal_preflight)
    proposal_run = proposal_actions.add_parser(
        "run", help="Execute the guarded overnight proposal program."
    )
    proposal_run.add_argument("--config", required=True)
    proposal_run.set_defaults(func=_run_causal_proposal_program)
    pairwise = workflows.add_parser(
        "pairwise-separation",
        help="Preflight or run bounded adjacent-frame binary, ICA, and constrained-NMF lanes.",
    )
    pairwise_actions = pairwise.add_subparsers(dest="experiment_action", required=True)
    pairwise_preflight = pairwise_actions.add_parser("preflight", help="Write read-only validation artifacts to a new explicit directory.")
    pairwise_preflight.add_argument("--config", required=True)
    pairwise_preflight.add_argument("--artifact-dir", type=Path, required=True)
    pairwise_preflight.set_defaults(func=_run_pairwise_preflight)
    pairwise_run = pairwise_actions.add_parser("run", help="Run after an explicitly reviewed matching preflight.")
    pairwise_run.add_argument("--config", required=True)
    pairwise_run.add_argument("--preflight-dir", type=Path, required=True)
    pairwise_run.set_defaults(func=_run_pairwise_separation)
    fusion_preflight = pairwise_actions.add_parser("fusion-preflight", help="Validate bounded Raw Direct and pairwise-feature fusion.")
    fusion_preflight.add_argument("--config", required=True)
    fusion_preflight.add_argument("--artifact-dir", type=Path, required=True)
    fusion_preflight.set_defaults(func=_run_pairwise_fusion_preflight)
    fusion_run = pairwise_actions.add_parser("fusion-run", help="Run bounded additive, soft-gate, and scalar-tuned fusion.")
    fusion_run.add_argument("--config", required=True)
    fusion_run.add_argument("--preflight-dir", type=Path, required=True)
    fusion_run.set_defaults(func=_run_pairwise_fusion)
    soma = workflows.add_parser(
        "soma-excitation",
        help="Evaluate dark-soma zones and frozen model transfer.",
    )
    actions = soma.add_subparsers(dest="experiment_action", required=True)

    preflight = actions.add_parser("preflight", help="Validate inputs and resource caps.")
    preflight.add_argument("--config", required=True, help="Soma-excitation JSON manifest.")
    preflight.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Permit preflight inspection of an existing output directory; never overwrites it.",
    )
    preflight.add_argument("--output-json", help="Optional path for the preflight JSON.")
    preflight.set_defaults(func=_run_soma_preflight)

    run = actions.add_parser("run", help="Run the bounded CPU experiment.")
    run.add_argument("--config", required=True, help="Soma-excitation JSON manifest.")
    run.set_defaults(func=_run_soma_experiment)

    learnable = workflows.add_parser("learnable-contrast", help="Train and evaluate guarded weakly-supervised contrast kernels.")
    learnable_actions = learnable.add_subparsers(dest="experiment_action", required=True)
    lc_preflight = learnable_actions.add_parser("preflight", help="Validate labels, source data, coordinates, and CUDA resources.")
    lc_preflight.add_argument("--config", required=True)
    lc_preflight.add_argument("--artifact-dir", type=Path, required=True)
    lc_preflight.set_defaults(func=_run_learnable_contrast_preflight)
    lc_diag = learnable_actions.add_parser("diagnostic", help="Run the 2x2x2 spatiotemporal factorial diagnostic.")
    lc_diag.add_argument("--config", required=True)
    lc_diag.set_defaults(func=_run_learnable_contrast_diagnostic)
    lc_direct = learnable_actions.add_parser("direct-tuning", help="Tune a detector initialized from the raw-direct baseline.")
    lc_direct.add_argument("--config", required=True)
    lc_direct.set_defaults(func=_run_learnable_direct_tuning)
    lc_multi_preflight = learnable_actions.add_parser(
        "multi-cfar-preflight",
        help="Validate the morphology-aware multiscale CFAR screen.",
    )
    lc_multi_preflight.add_argument("--config", required=True)
    lc_multi_preflight.add_argument("--artifact-dir", type=Path, required=True)
    lc_multi_preflight.set_defaults(func=_run_multi_cfar_preflight)
    lc_multi = learnable_actions.add_parser(
        "multi-cfar",
        help="Run the checkpointed morphology-aware multiscale CFAR screen.",
    )
    lc_multi.add_argument("--config", required=True)
    lc_multi.set_defaults(func=_run_multi_cfar)
    lc_multi_video = learnable_actions.add_parser(
        "multi-cfar-videos",
        help="Generate standalone diagnostic videos for a fixed CFAR expert.",
    )
    lc_multi_video.add_argument("--config", required=True)
    lc_multi_video.add_argument("--results-json", type=Path, required=True)
    lc_multi_video.add_argument("--output-dir", type=Path, required=True)
    lc_multi_video.add_argument("--expert-id")
    lc_multi_video.add_argument("--fps", type=float, default=10.0)
    lc_multi_video.set_defaults(func=_run_multi_cfar_videos)
    lc_run = learnable_actions.add_parser("run", help="Run the guarded CUDA learnable-contrast experiment.")
    lc_run.add_argument("--config", required=True)
    lc_run.set_defaults(func=_run_learnable_contrast_experiment)
    latent = workflows.add_parser(
        "latent-dynamics",
        help="Preflight and run stable AR(1) denoising and feature benchmarks.",
    )
    latent_actions = latent.add_subparsers(dest="experiment_action", required=True)
    latent_preflight = latent_actions.add_parser("preflight", help="Write collision-safe read-only validation artifacts.")
    latent_preflight.add_argument("--config", required=True)
    latent_preflight.add_argument("--artifact-dir", type=Path, required=True)
    latent_preflight.set_defaults(func=_run_latent_dynamics_preflight)
    latent_synthetic = latent_actions.add_parser("synthetic", help="Run deterministic synthetic falsification cases.")
    latent_synthetic.add_argument("--output-dir", type=Path, required=True)
    latent_synthetic.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 19, 29, 37])
    latent_synthetic.set_defaults(func=_run_latent_dynamics_synthetic)
    latent_run = latent_actions.add_parser("run", help="Run a matching reviewed CPU preflight.")
    latent_run.add_argument("--config", required=True)
    latent_run.add_argument("--preflight-dir", type=Path, required=True)
    latent_run.set_defaults(func=_run_latent_dynamics)
    latent_benchmark = latent_actions.add_parser("feature-benchmark", help="Read the completed feature benchmark artifact.")
    latent_benchmark.add_argument("--run-dir", type=Path, required=True)
    latent_benchmark.set_defaults(func=_run_latent_dynamics_feature_benchmark)
    representation = workflows.add_parser("representation", help="Benchmark PCA, spatial ICA, autoencoders, and optional UMAP.")
    representation_actions = representation.add_subparsers(dest="experiment_action", required=True)
    representation_preflight = representation_actions.add_parser("preflight", help="Write guarded representation preflight artifacts.")
    representation_preflight.add_argument("--config", required=True)
    representation_preflight.add_argument("--artifact-dir", type=Path, required=True)
    representation_preflight.set_defaults(func=_run_representation_preflight)
    representation_run = representation_actions.add_parser("run", help="Run the matching reviewed representation benchmark.")
    representation_run.add_argument("--config", required=True)
    representation_run.add_argument("--preflight-dir", type=Path, required=True)
    representation_run.set_defaults(func=_run_representation_benchmark)
    dependent = workflows.add_parser(
        "dependent-multiscale",
        help="Build and audit reversible dependent multiscale decompositions.",
    )
    dependent_actions = dependent.add_subparsers(dest="experiment_action", required=True)
    dependent_synthetic = dependent_actions.add_parser("synthetic", help="Run generated-only W3 baselines.")
    dependent_synthetic.add_argument("--output-dir", type=Path, required=True)
    dependent_synthetic.set_defaults(func=_run_dependent_multiscale_synthetic)
    dependent_preflight = dependent_actions.add_parser("preflight", help="Write a guarded read-only-source audit.")
    dependent_preflight.add_argument("--config", required=True)
    dependent_preflight.set_defaults(func=_run_dependent_multiscale_preflight)
    dependent_smoke = dependent_actions.add_parser("smoke", help="Run the tiny generated suite without artifacts.")
    dependent_smoke.set_defaults(func=_run_dependent_multiscale_smoke)
    dependent_run = dependent_actions.add_parser("run", help="Require reviewed preflight and explicit real-data authorization.")
    dependent_run.add_argument("--config", required=True)
    dependent_run.set_defaults(func=_run_dependent_multiscale)
    dependent_report = dependent_actions.add_parser("report", help="Read completed metrics without writes.")
    dependent_report.add_argument("--run-dir", type=Path, required=True)
    dependent_report.set_defaults(func=_run_dependent_multiscale_report)
    architecture = workflows.add_parser(
        "parzen-architecture-visuals",
        help="Compare four stochastic-Parzen state architectures on Spon Ca.",
    )
    architecture_actions = architecture.add_subparsers(
        dest="experiment_action",
        required=True,
    )
    architecture_preflight = architecture_actions.add_parser(
        "preflight",
        help="Read-only validation of data, output collision, and resources.",
    )
    architecture_preflight.add_argument("--config", required=True)
    architecture_preflight.set_defaults(
        func=_run_parzen_architecture_preflight
    )
    architecture_run = architecture_actions.add_parser(
        "run",
        help="Write background and dynamics/noise TIFFs for all four lanes.",
    )
    architecture_run.add_argument("--config", required=True)
    architecture_run.set_defaults(func=_run_parzen_architecture_visuals)
    architecture_grid = workflows.add_parser(
        "parzen-architecture-grid",
        help="Screen stochastic-state architectures and evaluate cross-fitted finalists.",
    )
    architecture_grid_actions = architecture_grid.add_subparsers(
        dest="experiment_action", required=True,
    )
    architecture_grid_preflight = architecture_grid_actions.add_parser(
        "preflight", help="Write the read-only grid and resource audit.",
    )
    architecture_grid_preflight.add_argument("--config", required=True)
    architecture_grid_preflight.set_defaults(func=_run_parzen_architecture_grid_preflight)
    architecture_grid_run = architecture_grid_actions.add_parser(
        "run", help="Run the matching broad screen and full-field finalists.",
    )
    architecture_grid_run.add_argument("--config", required=True)
    architecture_grid_run.set_defaults(func=_run_parzen_architecture_grid)
    signal_split = workflows.add_parser(
        "parzen-signal-split",
        help="Split Parzen Innovation with a noisy-Parzen posterior diagnostic.",
    )
    signal_split_actions = signal_split.add_subparsers(
        dest="experiment_action", required=True,
    )
    signal_split_preflight = signal_split_actions.add_parser(
        "preflight", help="Write the read-only split and resource audit.",
    )
    signal_split_preflight.add_argument("--config", required=True)
    signal_split_preflight.set_defaults(func=_run_parzen_signal_split_preflight)
    signal_split_run = signal_split_actions.add_parser(
        "run", help="Screen posterior settings and write signal/noise TIFFs.",
    )
    signal_split_run.add_argument("--config", required=True)
    signal_split_run.set_defaults(func=_run_parzen_signal_split)
    denoise_audit = workflows.add_parser(
        "parzen-denoise-audit",
        help="Sequentially audit denoising families on Parzen Innovation.",
    )
    denoise_audit_actions = denoise_audit.add_subparsers(
        dest="experiment_action", required=True,
    )
    denoise_audit_preflight = denoise_audit_actions.add_parser(
        "preflight", help="Write the read-only method and resource audit.",
    )
    denoise_audit_preflight.add_argument("--config", required=True)
    denoise_audit_preflight.set_defaults(func=_run_parzen_denoise_audit_preflight)
    denoise_audit_run = denoise_audit_actions.add_parser(
        "run", help="Run all denoising methods sequentially and write audit TIFFs.",
    )
    denoise_audit_run.add_argument("--config", required=True)
    denoise_audit_run.set_defaults(func=_run_parzen_denoise_audit)
    spatial_ica = workflows.add_parser(
        "parzen-spatial-ica",
        help="Compare patch and dense spatial ICA on accepted Parzen Innovation.",
    )
    spatial_ica_actions = spatial_ica.add_subparsers(
        dest="experiment_action", required=True,
    )
    spatial_ica_preflight = spatial_ica_actions.add_parser(
        "preflight", help="Write the read-only spatial ICA and resource audit.",
    )
    spatial_ica_preflight.add_argument("--config", required=True)
    spatial_ica_preflight.set_defaults(func=_run_parzen_spatial_ica_preflight)
    spatial_ica_run = spatial_ica_actions.add_parser(
        "run", help="Run the C1-C3 patch/convolutional ICA architecture screen.",
    )
    spatial_ica_run.add_argument("--config", required=True)
    spatial_ica_run.set_defaults(func=_run_parzen_spatial_ica)
    advanced_denoising = workflows.add_parser(
        "advanced-denoising-program",
        help="Run staged breadth/depth tuning across ten denoising families.",
    )
    advanced_denoising_actions = advanced_denoising.add_subparsers(
        dest="experiment_action", required=True,
    )
    advanced_denoising_preflight = advanced_denoising_actions.add_parser(
        "preflight", help="Write the read-only design and resource audit.",
    )
    advanced_denoising_preflight.add_argument("--config", required=True)
    advanced_denoising_preflight.set_defaults(
        func=_run_advanced_denoising_preflight
    )
    advanced_denoising_run = advanced_denoising_actions.add_parser(
        "run", help="Run breadth, full-field semifinals, and finalist TIFFs.",
    )
    advanced_denoising_run.add_argument("--config", required=True)
    advanced_denoising_run.set_defaults(func=_run_advanced_denoising_program)
    innovation_denoising = workflows.add_parser(
        "innovation-denoising-v3",
        help="Run bounded local, component, graph, blind-spot, and Pareto-mixture denoisers.",
    )
    innovation_denoising_actions = innovation_denoising.add_subparsers(
        dest="experiment_action", required=True,
    )
    innovation_denoising_preflight = innovation_denoising_actions.add_parser(
        "preflight", help="Write the read-only v3 design and resource audit.",
    )
    innovation_denoising_preflight.add_argument("--config", required=True)
    innovation_denoising_preflight.set_defaults(
        func=_run_innovation_denoising_preflight
    )
    innovation_denoising_run = innovation_denoising_actions.add_parser(
        "run", help="Run v3 breadth, full-field, mixture, and gated seed stages.",
    )
    innovation_denoising_run.add_argument("--config", required=True)
    innovation_denoising_run.set_defaults(func=_run_innovation_denoising_program)
    feature_utility = workflows.add_parser(
        "feature-utility",
        help="Generate and cross-fit bounded activity features on Spon Ca.",
    )
    feature_utility_actions = feature_utility.add_subparsers(
        dest="experiment_action", required=True,
    )
    feature_utility_preflight = feature_utility_actions.add_parser(
        "preflight", help="Write the read-only feature and resource audit.",
    )
    feature_utility_preflight.add_argument("--config", required=True)
    feature_utility_preflight.set_defaults(func=_run_feature_utility_preflight)
    feature_utility_run = feature_utility_actions.add_parser(
        "run", help="Generate features and run nested utility evaluation.",
    )
    feature_utility_run.add_argument("--config", required=True)
    feature_utility_run.set_defaults(func=_run_feature_utility)
    innovation_ranker = workflows.add_parser(
        "innovation-ranker",
        help="Run nested proposal, cut-morphology, calibration, and fine-tuning experiments.",
    )
    innovation_ranker_actions = innovation_ranker.add_subparsers(
        dest="experiment_action", required=True,
    )
    innovation_ranker_preflight = innovation_ranker_actions.add_parser(
        "preflight", help="Write the read-only nested-ranker resource audit.",
    )
    innovation_ranker_preflight.add_argument("--config", required=True)
    innovation_ranker_preflight.set_defaults(
        func=_run_innovation_ranker_preflight
    )
    innovation_ranker_run = innovation_ranker_actions.add_parser(
        "run", help="Run screening, nested tuning, confirmations, and diagnostics.",
    )
    innovation_ranker_run.add_argument("--config", required=True)
    innovation_ranker_run.set_defaults(func=_run_innovation_ranker)
    innovation_ranker_video = innovation_ranker_actions.add_parser(
        "missed-video",
        help="Overlay identities missed by a completed ranker on the raw video.",
    )
    innovation_ranker_video.add_argument("--config", required=True)
    innovation_ranker_video.add_argument("--ranker-root", required=True)
    innovation_ranker_video.add_argument("--output-dir", required=True)
    innovation_ranker_video.add_argument("--fps", type=float, default=10.0)
    innovation_ranker_video.add_argument(
        "--box-half-size-px", type=int, default=9
    )
    innovation_ranker_video.add_argument(
        "--zoom-padding-px", type=int, default=36
    )
    innovation_ranker_video.add_argument(
        "--recovery-field", default="linear_recovered"
    )
    innovation_ranker_video.set_defaults(
        func=_run_innovation_ranker_missed_video
    )
    patch_information = workflows.add_parser(
        "patch-information",
        help="Evaluate Principe-aligned local Renyi, CS, and correntropy features.",
    )
    patch_information_actions = patch_information.add_subparsers(
        dest="experiment_action", required=True,
    )
    patch_information_preflight = patch_information_actions.add_parser(
        "preflight", help="Write the read-only patch-information resource audit.",
    )
    patch_information_preflight.add_argument("--config", required=True)
    patch_information_preflight.set_defaults(
        func=_run_patch_information_preflight
    )
    patch_information_run = patch_information_actions.add_parser(
        "run", help="Run fixed ITL screens and nested candidate ranking.",
    )
    patch_information_run.add_argument("--config", required=True)
    patch_information_run.set_defaults(func=_run_patch_information)
    patch_information_video = patch_information_actions.add_parser(
        "video", help="Generate every review-frame map for one frozen ITL feature.",
    )
    patch_information_video.add_argument("--config", required=True)
    patch_information_video.add_argument(
        "--feature-id", default="cs_quiet__p7__bw0p5"
    )
    patch_information_video.add_argument("--output-dir", type=Path, required=True)
    patch_information_video.set_defaults(func=_run_patch_information_video)
    multiscale_preflight = patch_information_actions.add_parser(
        "multiscale-preflight",
        help="Audit the frozen multiscale CS-divergence experiment.",
    )
    multiscale_preflight.add_argument("--config", required=True)
    multiscale_preflight.set_defaults(
        func=_run_multiscale_information_preflight
    )
    multiscale_run = patch_information_actions.add_parser(
        "multiscale-run",
        help="Run multiscale CS maps, fusions, and held-out evaluation.",
    )
    multiscale_run.add_argument("--config", required=True)
    multiscale_run.set_defaults(func=_run_multiscale_information)
    scientific_audit_preflight = patch_information_actions.add_parser(
        "scientific-audit-preflight",
        help="Audit acquisition/noise, morphology, and structured-feature resources.",
    )
    scientific_audit_preflight.add_argument("--config", required=True)
    scientific_audit_preflight.set_defaults(
        func=_run_scientific_feature_audit_preflight
    )
    scientific_audit_run = patch_information_actions.add_parser(
        "scientific-audit-run",
        help="Run the checkpointed Spon scientific feature audit.",
    )
    scientific_audit_run.add_argument("--config", required=True)
    scientific_audit_run.set_defaults(func=_run_scientific_feature_audit)


def _run_soma_preflight(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.soma_excitation import (
        SomaExcitationConfig,
        build_soma_excitation_preflight,
    )

    config = SomaExcitationConfig.load_json(args.config)
    payload = build_soma_excitation_preflight(
        config,
        allow_existing_output=bool(args.allow_existing_output),
    )
    if args.output_json:
        _atomic_json(Path(args.output_json), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_frame_difference_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.frame_difference import FrameDifferenceConfig, preflight

    payload = preflight(FrameDifferenceConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_frame_difference(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.frame_difference import FrameDifferenceConfig, run

    payload = run(FrameDifferenceConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_smoothed_difference_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.smoothed_frame_difference import SmoothedDifferenceConfig, preflight

    payload = preflight(SmoothedDifferenceConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_smoothed_difference(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.smoothed_frame_difference import SmoothedDifferenceConfig, run

    payload = run(SmoothedDifferenceConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_activity_gate_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.activity_gated_video import ActivityGateConfig, preflight

    payload = preflight(ActivityGateConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_activity_gate(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.activity_gated_video import ActivityGateConfig, run

    payload = run(ActivityGateConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_activity_gate_benchmark_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.activity_gate_benchmark import (
        ActivityGateBenchmarkConfig,
        preflight,
    )

    payload = preflight(ActivityGateBenchmarkConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_activity_gate_benchmark(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.activity_gate_benchmark import (
        ActivityGateBenchmarkConfig,
        run,
    )

    payload = run(ActivityGateBenchmarkConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_causal_proposal_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.causal_proposal_program import (
        CausalProposalProgramConfig,
        preflight,
    )

    payload = preflight(CausalProposalProgramConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_causal_proposal_program(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.causal_proposal_program import (
        CausalProposalProgramConfig,
        run,
    )

    payload = run(CausalProposalProgramConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_pairwise_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.pairwise_separation import PairwiseSeparationConfig, preflight
    payload = preflight(PairwiseSeparationConfig.load(args.config), artifact_dir=args.artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_pairwise_separation(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.pairwise_separation.config import PairwiseSeparationConfig
    from neurobench.experiments.pairwise_separation.runner import run
    payload = run(PairwiseSeparationConfig.load(args.config), preflight_dir=args.preflight_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_pairwise_fusion_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.pairwise_separation.fusion import PairwiseFusionConfig, preflight
    payload = preflight(PairwiseFusionConfig.load(args.config), artifact_dir=args.artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_pairwise_fusion(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.pairwise_separation.fusion import PairwiseFusionConfig, run
    payload = run(PairwiseFusionConfig.load(args.config), preflight_dir=args.preflight_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_soma_experiment(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.soma_excitation.runner import (
        run_soma_excitation_experiment,
    )

    payload = run_soma_excitation_experiment(args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _configure_resource_environment_from_manifest(config_path: str | Path) -> int:
    """Set process limits from JSON before importing a scientific dependency."""
    source = Path(config_path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The soma-excitation JSON config must contain one object.")
    resources = payload.get("resources")
    if resources is None:
        resources = {}
    if not isinstance(resources, dict):
        raise ValueError("resources must be a JSON object.")
    raw_threads = resources.get("cpu_threads", _DEFAULT_CPU_THREADS)
    if isinstance(raw_threads, bool):
        raise ValueError("resources.cpu_threads must be an integer, not a boolean.")
    try:
        cpu_threads = int(raw_threads)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"resources.cpu_threads must be an integer; got {raw_threads!r}."
        ) from exc
    if isinstance(raw_threads, float) and not raw_threads.is_integer():
        raise ValueError(
            f"resources.cpu_threads must be an integer; got {raw_threads!r}."
        )
    if not 1 <= cpu_threads <= _MAX_CPU_THREADS:
        raise ValueError(
            "resources.cpu_threads must be between "
            f"1 and {_MAX_CPU_THREADS}; got {cpu_threads}."
        )
    value = str(cpu_threads)
    for name in _THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return cpu_threads


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _configure_cuda_resource_environment(config_path: str | Path) -> int:
    """Set bounded CPU library threads without hiding the requested CUDA device."""
    payload = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    resources = payload.get("resources", {})
    raw = resources.get("cpu_threads", 4)
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 24:
        raise ValueError("resources.cpu_threads must be an integer between 1 and 24")
    for name in _THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = str(raw)
    return raw


def _run_learnable_contrast_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast import Config, preflight
    payload = preflight(Config.load(args.config), artifact_dir=args.artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_learnable_contrast_experiment(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast import Config, run
    payload = run(Config.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_learnable_contrast_diagnostic(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast.diagnostic import DiagnosticConfig, run
    payload = run(DiagnosticConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_learnable_direct_tuning(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast.direct_tuning import DirectTuningConfig, run
    payload = run(DirectTuningConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_multi_cfar_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast.multihypothesis import (
        MultiCFARConfig,
        preflight,
    )

    payload = preflight(MultiCFARConfig.load(args.config), artifact_dir=args.artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_multi_cfar(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast.multihypothesis import (
        MultiCFARConfig,
        run,
    )

    payload = run(MultiCFARConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_multi_cfar_videos(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.learnable_contrast.diagnostic_video import generate
    from neurobench.experiments.learnable_contrast.multihypothesis import MultiCFARConfig

    payload = generate(
        MultiCFARConfig.load(args.config),
        results_json=args.results_json.resolve(),
        output_dir=args.output_dir.resolve(),
        expert_id=args.expert_id,
        fps=args.fps,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_latent_dynamics_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.latent_dynamics import LatentDynamicsConfig, preflight
    payload = preflight(LatentDynamicsConfig.load(args.config), artifact_dir=args.artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_latent_dynamics_synthetic(args) -> int:
    from neurobench.experiments.latent_dynamics.runner import run_synthetic
    payload = run_synthetic(args.output_dir, seeds=tuple(args.seeds))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_latent_dynamics(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.latent_dynamics.config import LatentDynamicsConfig
    from neurobench.experiments.latent_dynamics.runner import run
    payload = run(LatentDynamicsConfig.load(args.config), preflight_dir=args.preflight_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_latent_dynamics_feature_benchmark(args) -> int:
    from neurobench.experiments.latent_dynamics.runner import feature_benchmark
    payload = feature_benchmark(args.run_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_dependent_multiscale_synthetic(args) -> int:
    from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_program import synthetic
    payload = synthetic(args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _run_dependent_multiscale_preflight(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_config import DependentMultiscaleConfig
    from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_program import preflight
    payload = preflight(DependentMultiscaleConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_dependent_multiscale_smoke(args) -> int:
    from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_program import generated_smoke
    payload = generated_smoke()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _run_dependent_multiscale(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_config import DependentMultiscaleConfig
    from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_program import run
    payload = run(DependentMultiscaleConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_dependent_multiscale_report(args) -> int:
    from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_program import report
    payload = report(args.run_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_representation_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.representation_benchmark import RepresentationBenchmarkConfig, preflight
    payload = preflight(RepresentationBenchmarkConfig.load(args.config), artifact_dir=args.artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_representation_benchmark(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.representation_benchmark.config import RepresentationBenchmarkConfig
    from neurobench.experiments.representation_benchmark.runner import run
    payload = run(RepresentationBenchmarkConfig.load(args.config), preflight_dir=args.preflight_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_architecture_preflight(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.architecture_config import (
        ArchitectureVisualConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.architecture_visuals import (
        preflight,
    )

    payload = preflight(ArchitectureVisualConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_architecture_visuals(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.architecture_config import (
        ArchitectureVisualConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.architecture_visuals import (
        run,
    )

    payload = run(ArchitectureVisualConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_architecture_grid_preflight(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.innovation_grid_config import (
        InnovationGridConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.innovation_grid import preflight

    payload = preflight(InnovationGridConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_architecture_grid(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.innovation_grid_config import (
        InnovationGridConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.innovation_grid import run

    payload = run(InnovationGridConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_signal_split_preflight(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.signal_noise_config import (
        SignalNoiseConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.signal_noise_split import preflight

    payload = preflight(SignalNoiseConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_signal_split(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.signal_noise_config import (
        SignalNoiseConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.signal_noise_split import run

    payload = run(SignalNoiseConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_denoise_audit_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.denoise_audit import preflight
    from neurobench.experiments.hierarchical_parzen_ica.denoise_audit_config import (
        DenoiseAuditConfig,
    )

    payload = preflight(DenoiseAuditConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_denoise_audit(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.denoise_audit import run
    from neurobench.experiments.hierarchical_parzen_ica.denoise_audit_config import (
        DenoiseAuditConfig,
    )

    payload = run(DenoiseAuditConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_spatial_ica_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.spatial_ica_config import (
        SpatialICAConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.spatial_ica_screen import (
        preflight,
    )

    payload = preflight(SpatialICAConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_parzen_spatial_ica(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.spatial_ica_config import (
        SpatialICAConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.spatial_ica_screen import run

    payload = run(SpatialICAConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_advanced_denoising_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.advanced_denoising_program import (
        preflight,
    )
    from neurobench.experiments.hierarchical_parzen_ica.denoising_program_config import (
        DenoisingProgramConfig,
    )

    payload = preflight(DenoisingProgramConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_advanced_denoising_program(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.advanced_denoising_program import (
        run,
    )
    from neurobench.experiments.hierarchical_parzen_ica.denoising_program_config import (
        DenoisingProgramConfig,
    )

    payload = run(DenoisingProgramConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

def _run_innovation_denoising_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.innovation_denoising_config import (
        InnovationDenoisingConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.innovation_denoising_program import (
        preflight,
    )

    payload = preflight(InnovationDenoisingConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_innovation_denoising_program(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.innovation_denoising_config import (
        InnovationDenoisingConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.innovation_denoising_program import (
        run,
    )

    payload = run(InnovationDenoisingConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_feature_utility_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.feature_utility_config import (
        FeatureUtilityConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.feature_utility_program import (
        preflight,
    )

    payload = preflight(FeatureUtilityConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_feature_utility(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.feature_utility_config import (
        FeatureUtilityConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.feature_utility_program import run

    payload = run(FeatureUtilityConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_innovation_ranker_preflight(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_config import (
        InnovationRankerConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_program import (
        preflight,
    )

    payload = preflight(InnovationRankerConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_innovation_ranker(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_config import (
        InnovationRankerConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_program import (
        run,
    )

    payload = run(InnovationRankerConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_innovation_ranker_missed_video(args) -> int:
    _configure_resource_environment_from_manifest(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.innovation_ranker_config import (
        InnovationRankerConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.missed_neuron_video import (
        generate,
    )

    payload = generate(
        InnovationRankerConfig.load(args.config),
        ranker_root=Path(args.ranker_root),
        output_dir=Path(args.output_dir),
        recovery_field=str(args.recovery_field),
        fps=float(args.fps),
        box_half_size_px=int(args.box_half_size_px),
        zoom_padding_px=int(args.zoom_padding_px),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_patch_information_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.patch_information_config import (
        PatchInformationConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.patch_information_program import (
        preflight,
    )

    payload = preflight(PatchInformationConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_patch_information(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.patch_information_config import (
        PatchInformationConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.patch_information_program import (
        run,
    )

    payload = run(PatchInformationConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_patch_information_video(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.patch_information_config import (
        PatchInformationConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.patch_information_video import (
        generate,
    )

    payload = generate(
        PatchInformationConfig.load(args.config),
        feature_id=str(args.feature_id),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_multiscale_information_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.multiscale_information_config import (
        MultiscaleInformationConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.multiscale_information_program import (
        preflight,
    )

    payload = preflight(MultiscaleInformationConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_multiscale_information(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.multiscale_information_config import (
        MultiscaleInformationConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.multiscale_information_program import (
        run,
    )

    payload = run(MultiscaleInformationConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_scientific_feature_audit_preflight(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_config import (
        ScientificAuditConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_program import (
        preflight,
    )

    payload = preflight(ScientificAuditConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_scientific_feature_audit(args) -> int:
    _configure_cuda_resource_environment(args.config)
    from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_config import (
        ScientificAuditConfig,
    )
    from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_program import (
        run,
    )

    payload = run(ScientificAuditConfig.load(args.config))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
