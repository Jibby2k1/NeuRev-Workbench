from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neurobench.algorithms.gamma_local_standardization import GammaReferenceSpec
from neurobench.experiments.gamma_ls_difference import smoke
from neurobench.experiments.gamma_ls_difference.cuda_runtime import (
    CudaRuntimeUnavailable,
    cuda_unavailable_message,
)
from neurobench.experiments.gamma_ls_difference.representations import (
    CausalPreprocessingConfig,
    causal_preprocess_common_input,
)


def _write_movie(path: Path, *, frames: int = 6, height: int = 25, width: int = 27) -> np.ndarray:
    values = np.arange(frames * height * width, dtype=np.uint16).reshape(
        frames, height, width
    )
    np.save(path, values, allow_pickle=False)
    return values


def _fake_execution() -> smoke.SmokeExecution:
    arrays: dict[str, np.ndarray] = {
        "gamma_reference_kernel": np.full((23, 23), 1.0 / (23 * 23), np.float32)
    }
    diagnostics: dict[str, dict[str, object]] = {}
    timings: dict[str, object] = {
        "h2d_ms": 0.1,
        "common_preprocessing_ms": 0.2,
        "arms": {},
    }
    for index, arm in enumerate(smoke.SMOKE_ARMS):
        arrays[f"{arm}_representation_preview"] = np.full(
            (2, 8, 8), index, dtype=np.float32
        )
        arrays[f"{arm}_gamma_ls_preview"] = np.full(
            (2, 8, 8), index + 0.5, dtype=np.float32
        )
        diagnostics[arm] = {
            "preview_finite": True,
            "representation_device_before_gamma_ls": "cuda:0",
            "gamma_ls_device_before_d2h": "cuda:0",
        }
        timings["arms"][arm] = {
            "representation_ms": 0.01,
            "gamma_ls_ms": 0.02,
            "diagnostic_d2h_ms": 0.01,
        }
    return smoke.SmokeExecution(
        runtime={"resolved_device": "cuda:0", "cuda_available": True},
        timings_ms=timings,
        arm_diagnostics=diagnostics,
        arrays=arrays,
        peak_memory={"max_memory_allocated_bytes": 1024},
    )


def test_default_smoke_reference_is_guarded_radial_gamma() -> None:
    spec = smoke.default_smoke_reference()
    assert spec.context_id == "gamma_h11_g3_n5_m0p75"
    assert spec.support_width_px == 23
    assert spec.guard_radius_px == 3
    assert spec.support_geometry == "disk"
    assert spec.boundary_mode == "valid_renormalized_zero"
    assert spec.nominal_mode_radius_px == pytest.approx(8.25)


def test_memmap_window_uses_explicit_one_based_bounds(tmp_path: Path) -> None:
    path = tmp_path / "movie.npy"
    source = _write_movie(path)
    window, diagnostics = smoke._load_memmap_window(
        path,
        start_frame_ui=2,
        frame_count=3,
        support_width_px=23,
        source_id="data://fixture/movie.npy",
    )
    np.testing.assert_array_equal(window, source[1:4].astype(np.float32))
    assert diagnostics["load_mode"] == "numpy_memmap_then_bounded_copy"
    assert diagnostics["window_start_frame_ui"] == 2
    assert diagnostics["window_stop_frame_ui_inclusive"] == 4
    assert diagnostics["window_start_index_zero_based"] == 1
    assert diagnostics["window_stop_index_zero_based_exclusive"] == 4
    assert diagnostics["labels_read"] is False
    assert diagnostics["source_id"] == "data://fixture/movie.npy"
    assert "path" not in diagnostics


def test_common_torch_preprocessing_matches_numpy_including_boundaries() -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(size=(5, 21, 23)).astype(np.float32)
    expected = causal_preprocess_common_input(
        values,
        config=CausalPreprocessingConfig(
            spatial_sigma_px=1.0,
            ema_alpha=0.4,
            gaussian_truncate=4.0,
        ),
    ).values
    actual = smoke._common_preprocess_on_device(
        torch.from_numpy(values),
        spatial_sigma_px=1.0,
        ema_alpha=0.4,
        gaussian_truncate=4.0,
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_three_gpu_formulas_are_common_frame_aligned_on_cpu_fixture() -> None:
    common = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[3.0, 8.0], [12.0, 16.0]],
        ]
    )
    raw = smoke._representation_on_device(common, "raw", epsilon=1e-8)
    signed = smoke._representation_on_device(
        common, "difference_signed", epsilon=1e-8
    )
    energy = smoke._representation_on_device(
        common, "difference_energy_normalized", epsilon=1e-8
    )
    assert raw.shape == signed.shape == energy.shape == (2, 2, 2)
    torch.testing.assert_close(raw, common[1:])
    torch.testing.assert_close(signed, common[1:] - common[:-1])
    torch.testing.assert_close(
        energy,
        signed / torch.sqrt(common[:-1].square() + common[1:].square() + 1e-8),
    )


def test_cuda_unavailable_leaves_requested_output_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = tmp_path / "movie.npy"
    _write_movie(movie)
    output = tmp_path / "blocked"

    def blocked(_: str) -> dict[str, object]:
        raise smoke.CudaSmokeUnavailable("fixture CUDA unavailable")

    monkeypatch.setattr(smoke, "_require_cuda", blocked)
    with pytest.raises(smoke.CudaSmokeUnavailable, match="fixture CUDA unavailable"):
        smoke.run_cuda_smoke(
            movie, output_dir=output, start_frame_ui=1, frame_count=3
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".blocked.partial-*"))


def test_cuda_runtime_diagnostic_distinguishes_kernel_driver_from_toolkit() -> None:
    message = cuda_unavailable_message(
        torch,
        "cuda",
        device_nodes=(),
        kernel_release="fixture-kernel",
    )
    assert f"torch={torch.__version__}" in message
    assert f"torch CUDA build={torch.version.cuda}" in message
    assert "running kernel=fixture-kernel" in message
    assert "/dev/nvidia*=absent" in message
    assert "driver module matching the running kernel" in message
    assert "user-space toolkit cannot create the missing device nodes" in message


def test_smoke_preserves_actionable_shared_cuda_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(_: str) -> dict[str, object]:
        raise CudaRuntimeUnavailable("fixture actionable diagnostic")

    monkeypatch.setattr(smoke, "require_cuda_device", blocked)
    with pytest.raises(
        smoke.CudaSmokeUnavailable, match="fixture actionable diagnostic"
    ):
        smoke._require_cuda("cuda")


def test_stubbed_smoke_commits_small_atomic_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = tmp_path / "movie.npy"
    _write_movie(movie)
    output = tmp_path / "smoke"
    monkeypatch.setattr(
        smoke,
        "_require_cuda",
        lambda device: {
            "requested_device": device,
            "resolved_device": "cuda:0",
            "cuda_available": True,
        },
    )
    monkeypatch.setattr(smoke, "_execute_cuda", lambda *args, **kwargs: _fake_execution())

    summary = smoke.run_cuda_smoke(
        movie,
        output_dir=output,
        source_id="data://fixture/movie.npy",
        start_frame_ui=2,
        frame_count=3,
    )
    assert summary["status"] == "complete_runtime_smoke_only"
    assert summary["source"]["labels_read"] is False
    assert summary["source"]["source_id"] == "data://fixture/movie.npy"
    assert "path" not in summary["source"]
    assert summary["scientific_scope"]["scientific_audit_complete"] is False
    assert output.is_dir()
    assert not list(output.rglob("*.partial"))
    assert not list(tmp_path.glob(".smoke.partial-*"))

    validation = json.loads((output / "validation.json").read_text())
    assert validation["checks"]["three_frozen_smoke_arms_present"] is True
    assert validation["checks"]["labels_read"] is False
    artifact_index = json.loads((output / "artifact_index.json").read_text())
    assert "summary.json" in artifact_index["artifacts"]
    assert "arrays/raw_gamma_ls_preview.npy" in artifact_index["artifacts"]
    loaded = np.load(output / "arrays/raw_gamma_ls_preview.npy", allow_pickle=False)
    assert loaded.shape == (2, 8, 8)

    with pytest.raises(FileExistsError):
        smoke.run_cuda_smoke(movie, output_dir=output, start_frame_ui=2, frame_count=3)


def test_invalid_reference_fails_before_output_mutation(tmp_path: Path) -> None:
    movie = tmp_path / "movie.npy"
    _write_movie(movie)
    output = tmp_path / "invalid_reference"
    invalid = GammaReferenceSpec.from_mode(
        "unguarded",
        support_width_px=23,
        shape_n=5.0,
        mode_radius_px=8.25,
        guard_radius_px=0,
        support_geometry="disk",
    )
    with pytest.raises(ValueError, match="explicit guard"):
        smoke.run_cuda_smoke(movie, output_dir=output, reference=invalid)
    assert not output.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_tiny_pipeline_stays_device_resident(tmp_path: Path) -> None:
    movie = tmp_path / "movie.npy"
    rng = np.random.default_rng(19)
    np.save(
        movie,
        rng.integers(0, 4096, size=(4, 25, 27), dtype=np.uint16),
        allow_pickle=False,
    )
    output = tmp_path / "cuda_smoke"
    summary = smoke.run_cuda_smoke(
        movie,
        output_dir=output,
        start_frame_ui=1,
        frame_count=4,
        device="cuda:0",
    )
    assert summary["runtime"]["cuda_available"] is True
    for arm in smoke.SMOKE_ARMS:
        diagnostics = json.loads((output / "diagnostics" / f"{arm}.json").read_text())
        assert diagnostics["representation_device_before_gamma_ls"].startswith("cuda")
        assert diagnostics["gamma_ls_device_before_d2h"].startswith("cuda")
