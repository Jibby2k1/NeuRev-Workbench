from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neurobench.experiments.gamma_ls_difference.grid import enumerate_g1_contexts
from neurobench.experiments.gamma_ls_difference.config import GammaLSDifferenceConfig
from neurobench.experiments.gamma_ls_difference import gpu_representations
from neurobench.experiments.gamma_ls_difference import screen


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples/spon_ca_burst_gamma_ls_difference_ablation_v1.example.json"


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> GammaLSDifferenceConfig:
    monkeypatch.setenv("NEUROBENCH_DATA_ROOT", str(REPOSITORY))
    return GammaLSDifferenceConfig.load(EXAMPLE)


def test_fold_contracts_exclude_one_burst_and_split_quiet_exactly(config) -> None:
    folds = screen.build_fold_contracts(config)
    assert [fold.heldout_burst for fold in folds] == ["1", "2", "3", "4"]
    assert all(len(fold.training_bursts) == 3 for fold in folds)
    assert folds[0].training_bursts == ("2", "3", "4")
    assert folds[0].heldout_guard_ui == (1993, 2036)
    assert all(fold.quiet_half_a_ui == (1800, 1849) for fold in folds)
    assert all(fold.quiet_half_b_ui == (1850, 1899) for fold in folds)


def test_positive_tail_is_frozen_higher_order_statistic() -> None:
    values = torch.arange(-2.0, 18.0).reshape(1, 4, 5)
    # ceil(.9 * 20) selects the 18th one-based order statistic after clipping.
    tail = screen._positive_tail(values, 0.9)
    assert tail.item() == 15.0


def test_scale_floor_uses_only_positive_selected_values() -> None:
    values = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[100.0, 100.0], [100.0, 100.0]],
        ]
    )
    floor = screen._positive_scale_floor(
        values, torch.tensor([True, False]), percentile=50.0
    )
    assert floor.item() == 2.0


def test_context_cell_fits_and_records_both_quiet_swaps(config) -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(8, 19, 21)).astype(np.float32)
    values[1] *= 8.0
    representation = torch.from_numpy(values)
    fold = screen.FoldContract(
        training_fold=1,
        heldout_burst="1",
        training_bursts=("2", "3", "4"),
        heldout_guard_ui=(100, 120),
        quiet_half_a_ui=(1, 1),
        quiet_half_b_ui=(2, 2),
    )
    aggregate, swaps, _ = screen._evaluate_context_arm(
        representation,
        representation_name="raw",
        context=enumerate_g1_contexts(config)[0],
        stage="g1",
        folds=(fold,),
        frame_ui=torch.arange(1, 9),
        bursts={"1": [100, 101], "2": [3, 3], "3": [4, 4], "4": [5, 5]},
        scale_floor_percentile=10.0,
        tail_quantile=0.9,
        chunk_frames=4,
    )
    assert len(aggregate) == 1
    assert [row["quiet_swap"] for row in swaps] == list(screen.QUIET_SWAPS)
    assert swaps[0]["scale_floor"] != swaps[1]["scale_floor"]
    assert all(row["positive_coordinates_used"] is False for row in swaps)
    assert all(row["burst_windows_used"] is True for row in swaps)


def test_chunked_full_history_matches_dense_causal_preprocessing(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    source = rng.normal(size=(12, 9, 11)).astype(np.float32)
    movie = tmp_path / "movie.npy"
    np.save(movie, source, allow_pickle=False)
    actual, diagnostics = screen._stream_common_history_to_device(
        movie,
        review_start_ui=5,
        review_stop_ui=12,
        chunk_frames=3,
        device=torch.device("cpu"),
        heartbeat=None,
    )
    dense = gpu_representations.causal_preprocess_common_input(
        torch.from_numpy(source)
    ).values
    # Common predecessor is UI 4 (zero-based row 3), through UI 12.
    torch.testing.assert_close(actual, dense[3:12], rtol=2e-6, atol=2e-6)
    assert diagnostics["movie_h2d_transfer_count"] == 4
    assert diagnostics["history_chunk_count"] == 4
    assert diagnostics["ema_state_carried_across_chunks"] is True


def test_cuda_unavailable_fails_before_output_mutation(
    tmp_path: Path,
    config: GammaLSDifferenceConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "screen"
    monkeypatch.setattr(
        screen,
        "_verify_ready_preflight",
        lambda *args, **kwargs: {"gpu_run_ready": True},
    )

    def blocked(_: str):
        raise screen.CudaScreenUnavailable("fixture CUDA unavailable")

    monkeypatch.setattr(screen, "_require_cuda", blocked)
    with pytest.raises(screen.CudaScreenUnavailable, match="fixture CUDA unavailable"):
        screen.run_gpu_screen(
            config,
            preflight_dir=tmp_path / "preflight",
            output_dir=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".screen.partial-*"))


def _rows(count: int, stage: str) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "stage": stage,
            "context_id": f"context_{index}",
            "training_fold": index % 4 + 1,
        }
        for index in range(count)
    )


def _fake_execution() -> screen.ScreenExecution:
    return screen.ScreenExecution(
        g1_rows=_rows(108, "g1"),
        g1_swap_rows=_rows(216, "g1"),
        g2_rows=_rows(216, "g2"),
        g2_swap_rows=_rows(432, "g2"),
        fold_selections=tuple(
            {
                "training_fold": fold,
                "heldout_burst": str(fold),
                "common_g2_finalist_context_id": f"gamma_fold_{fold}",
            }
            for fold in range(1, 5)
        ),
        runtime={"resolved_device": "cuda:0", "cuda_available": True},
        timings={
            "movie_h2d_transfer_count": 37,
            "history_chunk_count": 37,
            "ema_state_carried_across_chunks": True,
        },
        peak_memory={"max_memory_allocated_bytes": 1024},
    )


def test_stubbed_screen_commits_atomic_fold_local_artifact(
    tmp_path: Path,
    config: GammaLSDifferenceConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "screen"
    monkeypatch.setattr(
        screen,
        "_verify_ready_preflight",
        lambda *args, **kwargs: {
            "gpu_run_ready": True,
            "movie_sha256": "abc",
            "label_files_reopened": False,
        },
    )
    monkeypatch.setattr(
        screen,
        "_require_cuda",
        lambda device: {"resolved_device": "cuda:0", "cuda_available": True},
    )
    monkeypatch.setattr(
        screen,
        "_source_contract",
        lambda cfg: {
            "source_id": "data://fixture.npy",
            "labels_opened": False,
            "positive_coordinates_used": False,
            "positive_identities_used": False,
            "burst_windows_used": True,
        },
    )
    monkeypatch.setattr(
        screen, "_execute_device_screen", lambda *args, **kwargs: _fake_execution()
    )

    summary = screen.run_gpu_screen(
        config,
        preflight_dir=tmp_path / "preflight",
        output_dir=output,
    )
    assert summary["status"] == "complete_screen_only"
    assert output.is_dir()
    assert not list(tmp_path.glob(".screen.partial-*"))
    assert not list(output.rglob("*.partial"))
    selection = json.loads((output / "selection.json").read_text())
    assert selection["deployment_context_id"] is None
    assert selection["across_fold_pooling_used_for_protected_selection"] is False
    boundary = json.loads((output / "claim_boundary.json").read_text())
    assert boundary["fully_label_free_claimed"] is False
    assert boundary["burst_windows_used"] is True
    assert boundary["positive_coordinates_used"] is False
    validation = json.loads((output / "validation.json").read_text())
    assert validation["checks"]["g1_has_108_metric_cells"] is True
    assert validation["checks"]["g2_has_216_metric_cells"] is True
