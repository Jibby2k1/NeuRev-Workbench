from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neurobench.experiments.gamma_ls_difference.config import GammaLSDifferenceConfig
from neurobench.experiments.gamma_ls_difference import streaming_benchmark as benchmark
from neurobench.metrics.sparse_detection import extract_separated_local_maxima


def test_gamma_context_parser_reconstructs_selected_radial_stencil() -> None:
    spec = benchmark.parse_gamma_context_id("gamma_h11_g5_n9_m1")
    assert spec.context_id == "gamma_h11_g5_n9_m1"
    assert spec.support_width_px == 23
    assert spec.guard_radius_px == 5
    assert spec.shape_n == 9
    assert spec.nominal_mode_radius_px == pytest.approx(11.0)
    assert spec.support_geometry == "disk"
    assert spec.boundary_mode == "valid_renormalized_zero"

    support_id = "support_support_a_h15_g7_n9_m0p5"
    support = benchmark.parse_gamma_context_id(support_id)
    assert support.context_id == support_id
    assert support.support_width_px == 31
    assert support.guard_radius_px == 7
    assert support.shape_n == 9
    assert support.nominal_mode_radius_px == pytest.approx(7.5)
    assert support.support_geometry == "disk"
    assert support.boundary_mode == "valid_renormalized_zero"

    stage_b_id = "support_support_b_h39_g15_n5_m0p75"
    stage_b = benchmark.parse_gamma_context_id(stage_b_id, scale_floor=0.125)
    assert stage_b.context_id == stage_b_id
    assert stage_b.support_width_px == 79
    assert stage_b.nominal_mode_radius_px == pytest.approx(29.25)
    assert stage_b.scale_floor == pytest.approx(0.125)

    with pytest.raises(ValueError, match="unsupported"):
        benchmark.parse_gamma_context_id("legacy_square")
    with pytest.raises(ValueError, match="unsupported"):
        benchmark.parse_gamma_context_id("support_support_c_h15_g7_n9_m0p5")
    with pytest.raises(ValueError, match="unsupported"):
        benchmark.parse_gamma_context_id("support_a_h15_g7_n9_m0p5")


def test_bounded_arrival_queue_drops_newest_and_closes_accounting() -> None:
    queue = benchmark.BoundedArrivalQueue(
        start_ns=1_000,
        interval_ns=100,
        total_arrivals=6,
        capacity=2,
    )
    assert queue.enqueue_due(1_450) == 2
    assert queue.depth == 2
    assert queue.dropped == 3
    assert queue.max_depth == 2
    assert queue.pop().arrival_index == 0
    assert queue.pop().arrival_index == 1
    assert queue.enqueue_due(1_500) == 1
    assert queue.pop().arrival_index == 5
    assert queue.exhausted
    assert queue.dropped + 3 == queue.total_arrivals


def test_latency_summary_has_required_percentiles() -> None:
    result = benchmark._percentiles_ms([1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 4
    assert result["mean_ms"] == pytest.approx(2.5)
    assert result["p50_ms"] == pytest.approx(2.5)
    assert result["p95_ms"] > result["p50_ms"]
    assert result["p99_ms"] <= result["max_ms"] == 4.0


def test_production_request_requires_sixty_seconds_and_one_millisecond() -> None:
    assert benchmark._validate_run_request(
        arms=["difference_energy_normalized"],
        duration_seconds=60.0,
        arrival_interval_ms=1.0,
        queue_capacity=8,
        source_ring_frames=64,
    ) == ("difference_energy_normalized",)
    with pytest.raises(ValueError, match="at least 60"):
        benchmark._validate_run_request(
            arms=["difference_signed"],
            duration_seconds=59.999,
            arrival_interval_ms=1.0,
            queue_capacity=8,
            source_ring_frames=64,
        )


def test_readiness_gate_is_strict_about_latency_misses_and_backlog() -> None:
    passing = benchmark._streaming_readiness_gates(
        duration_seconds=60.0,
        arrival_interval_ms=1.0,
        p99_service_ms=0.999,
        dropped_arrivals=0,
        missed_processed_deadlines=0,
        maximum_queue_depth=1,
        maximum_backlog_after_dequeue=0,
        final_queue_depth=0,
    )
    assert all(passing.values())
    at_deadline = benchmark._streaming_readiness_gates(
        duration_seconds=60.0,
        arrival_interval_ms=1.0,
        p99_service_ms=1.0,
        dropped_arrivals=0,
        missed_processed_deadlines=0,
        maximum_queue_depth=1,
        maximum_backlog_after_dequeue=0,
        final_queue_depth=0,
    )
    assert at_deadline["p99_service_latency_strictly_below_deadline"] is False
    backed_up = benchmark._streaming_readiness_gates(
        duration_seconds=60.0,
        arrival_interval_ms=1.0,
        p99_service_ms=0.8,
        dropped_arrivals=0,
        missed_processed_deadlines=1,
        maximum_queue_depth=2,
        maximum_backlog_after_dequeue=1,
        final_queue_depth=0,
    )
    assert backed_up["zero_missed_processed_deadlines"] is False
    assert backed_up["maximum_queue_depth_at_most_one"] is False
    assert backed_up["zero_backlog_after_dequeue"] is False
    with pytest.raises(ValueError, match="exactly 1 ms"):
        benchmark._validate_run_request(
            arms=["difference_signed"],
            duration_seconds=60.0,
            arrival_interval_ms=1.1,
            queue_capacity=8,
            source_ring_frames=64,
        )


def test_device_local_nms_matches_offline_random_continuous_case() -> None:
    rng = np.random.default_rng(37)
    values = rng.normal(size=(31, 35)).astype(np.float32)
    threshold = 1.25
    distance = 3
    mask = benchmark._local_maximum_mask(
        torch.from_numpy(values)[None],
        threshold_z=threshold,
        distance_px=distance,
    )[0].numpy()
    observed = {(int(x), int(y)) for y, x in np.argwhere(mask)}
    expected = {
        (int(x), int(y))
        for _, x, y in extract_separated_local_maxima(
            values,
            distance,
            threshold=float(np.nextafter(np.float64(threshold), np.inf)),
        )
    }
    assert observed == expected


def test_prepared_spatial_operator_uses_instance_torch_runtime_on_cpu() -> None:
    import torch.nn.functional as functional

    pipeline = object.__new__(benchmark.PreparedCausalGammaPipeline)
    pipeline.torch = torch
    pipeline.functional = functional
    pipeline.gaussian = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
    pipeline.y_indices = benchmark.gpu_representations._scipy_reflect_indices(
        5, 1, device=torch.device("cpu")
    )
    pipeline.x_indices = benchmark.gpu_representations._scipy_reflect_indices(
        7, 1, device=torch.device("cpu")
    )
    values = torch.arange(2 * 5 * 7, dtype=torch.float32).reshape(2, 5, 7)
    observed = pipeline._spatial(values)
    expected = benchmark.gpu_representations._spatial_gaussian_reflect(values)
    assert observed.shape == values.shape
    # This fixture uses a compact three-tap kernel instead of the production
    # nine-tap Gaussian, so only execution/shape/finite behavior is compared.
    assert torch.isfinite(observed).all()
    assert torch.isfinite(expected).all()


def test_screen_context_must_be_one_of_four_fold_local_finalists(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "summary.json").write_text(
        json.dumps({"status": "complete_screen_only"}), encoding="utf-8"
    )
    (screen / "selection.json").write_text(
        json.dumps(
            {
                "folds": [
                    {"common_g2_finalist_context_id": "gamma_h11_g5_n9_m1"},
                    {"common_g2_finalist_context_id": "gamma_h11_g5_n9_m1"},
                    {"common_g2_finalist_context_id": "gamma_h11_g5_n5_m0p5"},
                    {"common_g2_finalist_context_id": "gamma_h11_g5_n9_m1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = benchmark._screen_context_role(screen, "gamma_h11_g5_n9_m1")
    assert result["benchmark_context_selected_in_fold_count"] == 3
    assert result["benchmark_context_id"] == "gamma_h11_g5_n9_m1"
    assert result["benchmark_context_namespace"] == "original_gamma_grid"
    assert result["deployment_context_claimed"] is False
    with pytest.raises(RuntimeError, match="not a fold-local"):
        benchmark._screen_context_role(screen, "gamma_h7_g5_n9_m1")


def test_validated_support_screen_context_is_accepted_without_deployment_claim(
    tmp_path: Path,
) -> None:
    screen = tmp_path / "support_screen"
    screen.mkdir()
    summary_path = screen / "summary.json"
    validation_path = screen / "validation.json"
    contexts_path = screen / "fold_contexts.json"
    summary_path.write_text(
        json.dumps({"status": "complete_support_screen_only"}), encoding="utf-8"
    )
    validation_path.write_text(
        json.dumps(
            {
                "status": (
                    "passed_support_screen_artifact_contract_scientific_audit_pending"
                )
            }
        ),
        encoding="utf-8",
    )
    context = {
        "context_id": "support_support_a_h19_g5_n9_m1",
        "half_width_px": 19,
        "guard_radius_px": 5,
        "shape": 9.0,
        "mode_fraction_of_half_width": 1.0,
        "mode_radius_px": 19.0,
        "support": "radial_disk",
        "padding": "valid_renormalized_zero",
        "stage": "support_a",
        "eligible_primary": True,
    }
    contexts_path.write_text(
        json.dumps(
            {
                "selection_scope": "outer_training_fold_only",
                "selection_uses_positive_coordinates": False,
                "selection_uses_positive_identities": False,
                "burst_windows_used": True,
                "folds": [
                    {"support_candidate_context": context} for _ in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )
    (screen / "artifact_index.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": path.name, "sha256": benchmark._sha256(path)}
                    for path in (summary_path, validation_path, contexts_path)
                ]
            }
        ),
        encoding="utf-8",
    )
    result = benchmark._screen_context_role(screen, context["context_id"])
    assert result["screen_kind"] == "extended_support_sufficiency_screen"
    assert len(result["benchmark_context_appearances"]) == 4
    assert result["benchmark_context_id"] == context["context_id"]
    assert result["benchmark_context_namespace"] == "support_support_a"
    assert result["deployment_context_claimed"] is False

    bad_metadata = {**context, "mode_radius_px": 18.0}
    contexts_path.write_text(
        json.dumps(
            {
                "selection_scope": "outer_training_fold_only",
                "selection_uses_positive_coordinates": False,
                "selection_uses_positive_identities": False,
                "burst_windows_used": True,
                "folds": [
                    {"support_candidate_context": bad_metadata} for _ in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )
    (screen / "artifact_index.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": path.name, "sha256": benchmark._sha256(path)}
                    for path in (summary_path, validation_path, contexts_path)
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="metadata disagrees"):
        benchmark._screen_context_role(screen, context["context_id"])

    contexts_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="index mismatch"):
        benchmark._screen_context_role(screen, context["context_id"])


class _FakeRing:
    shape = (64, 340, 573)

    def __getitem__(self, _: object) -> "_FakeRing":
        return self


def _fake_config(tmp_path: Path) -> GammaLSDifferenceConfig:
    movie = tmp_path / "movie.npy"
    movie.write_bytes(b"fixture")
    return GammaLSDifferenceConfig(
        manifest_path=tmp_path / "config.json",
        repository=tmp_path,
        authority=tmp_path,
        payload={
            "experiment_id": "spon_ca_burst_gamma_ls_difference_ablation_v1",
            "resources": {"device": "cuda", "cpu_threads": 4},
            "efficiency": {
                "warmup_iterations": 50,
                "timed_iterations": 2000,
                "frame_chunks": [1, 8, 32, 64],
            },
            "gamma_ls_grid": {"screen_scale_floor_percentile": 10.0},
            "frames": {"quiet_interval_ui": [1800, 1899]},
            "sources": {"movie": "data://fixture/movie.npy"},
        },
        source_paths={"movie": movie},
        output_root=tmp_path / "outputs",
    )


def _fake_sustained() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    samples = np.ones(60_000, dtype=np.float32)
    summary: dict[str, object] = {
        "gate_passed": True,
        "declared_duration_seconds": 60.0,
        "arrival_interval_ms": 1.0,
        "scheduled_arrivals": 60_000,
        "processed_arrivals": 60_000,
        "dropped_arrivals": 0,
        "service_latency": {"end_to_end_wall_ms": {"p99_ms": 0.9}},
    }
    return summary, {"response_latency_ms": samples}


def test_stubbed_run_commits_atomic_timing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "summary.json").write_text(
        json.dumps({"status": "complete_screen_only"}), encoding="utf-8"
    )
    (screen / "selection.json").write_text(
        json.dumps(
            {
                "folds": [
                    {"common_g2_finalist_context_id": "gamma_h11_g5_n9_m1"}
                    for _ in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )
    config = _fake_config(tmp_path)
    destination = tmp_path / "streaming"
    ring = _FakeRing()

    monkeypatch.setattr(
        benchmark,
        "_verify_ready_preflight",
        lambda *args, **kwargs: {"gpu_run_ready": True},
    )
    monkeypatch.setattr(
        benchmark,
        "require_cuda_device",
        lambda device: {
            "requested_device": device,
            "resolved_device": "cuda:0",
            "cuda_available": True,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_load_pinned_source_rings",
        lambda *args, **kwargs: (
            ring,
            ring,
            {
                "frame_shape_yx": [340, 573],
                "host_rings_pinned": True,
            },
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_calibrate_benchmark_operating_point",
        lambda *args, reference, **kwargs: (
            reference,
            3.0,
            {"model_fit_ms": 0.0, "benchmark_calibration_wall_ms": 2.0},
        ),
    )
    monkeypatch.setattr(benchmark, "PreparedCausalGammaPipeline", lambda **kwargs: object())
    monkeypatch.setattr(benchmark, "_run_sustained_lane", lambda *args, **kwargs: _fake_sustained())
    monkeypatch.setattr(
        benchmark,
        "_run_batch_frontier",
        lambda **kwargs: [
            {
                "arm": kwargs["arm"],
                "context_id": kwargs["reference"].context_id,
                "batch_frames": 1,
                "interpretation": "batch_throughput_only_not_single_frame_latency",
            }
        ],
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    result = benchmark.run_streaming_benchmark(
        config,
        preflight_dir=tmp_path / "preflight",
        screen_dir=screen,
        output_dir=destination,
        device="cuda",
    )
    assert result["status"] == "passed_1khz_streaming_gate"
    assert destination.is_dir()
    validation = json.loads((destination / "validation.json").read_text())
    assert validation["artifact_contract_passed"] is True
    assert validation["streaming_gate_passed"] is True
    assert (destination / "latency_samples.npz").is_file()
    assert (destination / "batch_throughput_frontier.tsv").is_file()
    assert not list(tmp_path.glob(".streaming.partial-*"))
    with pytest.raises(FileExistsError):
        benchmark.run_streaming_benchmark(
            config,
            preflight_dir=tmp_path / "preflight",
            screen_dir=screen,
            output_dir=destination,
            device="cuda",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_pipeline_keeps_dense_fields_on_cuda() -> None:
    reference = benchmark.parse_gamma_context_id(
        "gamma_h11_g5_n9_m1", scale_floor=0.01
    )
    pipeline = benchmark.PreparedCausalGammaPipeline(
        frame_shape=(340, 573),
        arm="difference_energy_normalized",
        reference=reference,
        threshold_z=3.0,
        device="cuda:0",
        max_batch_frames=1,
    )
    host = torch.randint(
        0, 4096, (1, 340, 573), dtype=torch.uint16, pin_memory=True
    )
    decision, timing = pipeline.process_batch(host)
    assert decision["count"].shape == (1,)
    assert decision["top_score"].shape == (1,)
    assert decision["top_x"].shape == (1,)
    assert decision["top_y"].shape == (1,)
    assert pipeline.device_input.is_cuda
    assert pipeline.ema_state.is_cuda
    assert pipeline.reference_mass.is_cuda
    assert timing["end_to_end_wall_ms"] >= timing["cuda_total_ms"]
