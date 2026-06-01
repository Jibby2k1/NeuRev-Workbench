"""CPU-safe CFAR-style local adaptive thresholding."""
from __future__ import annotations

from typing import Any

from neurobench.pipelines.devices import resolve_device


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for CFAR thresholding.") from exc
    return np


def _validate_video(video: Any):
    np = _load_numpy()
    array = np.asarray(video, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"CFAR expects a frame-first 3D video array, got shape {array.shape}.")
    return np, array


def _box_mean(array, radius: int):
    if radius < 0:
        raise ValueError("Box radius must be non-negative.")
    try:
        from scipy.ndimage import uniform_filter

        size = (1, 2 * radius + 1, 2 * radius + 1)
        return uniform_filter(array, size=size, mode="nearest")
    except ModuleNotFoundError:
        return _box_mean_numpy(array, radius)


def _box_mean_torch(tensor, radius: int):
    if radius < 0:
        raise ValueError("Box radius must be non-negative.")
    if radius == 0:
        return tensor.clone()
    import torch.nn.functional as functional  # type: ignore

    window = 2 * radius + 1
    frames = tensor.unsqueeze(1)
    padded = functional.pad(frames, (radius, radius, radius, radius), mode="replicate")
    return functional.avg_pool2d(padded, kernel_size=window, stride=1).squeeze(1)


def _box_mean_cupy(array, radius: int):
    if radius < 0:
        raise ValueError("Box radius must be non-negative.")
    if radius == 0:
        return array.copy()
    try:
        from cupyx.scipy.ndimage import uniform_filter  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional CuPy install
        raise RuntimeError("CuPy CUDA CFAR requires cupyx.scipy.ndimage.uniform_filter.") from exc
    size = (1, 2 * radius + 1, 2 * radius + 1)
    return uniform_filter(array, size=size, mode="nearest")


def _box_mean_numpy(array, radius: int):
    np = _load_numpy()
    if radius == 0:
        return array.astype(np.float32, copy=True)
    pad = ((0, 0), (radius, radius), (radius, radius))
    padded = np.pad(array, pad, mode="edge")
    integral = np.pad(padded, ((0, 0), (1, 0), (1, 0)), mode="constant")
    cumsum = integral.cumsum(axis=1).cumsum(axis=2)
    height, width = array.shape[1:]
    window = 2 * radius + 1
    total = (
        cumsum[:, window : window + height, window : window + width]
        - cumsum[:, :height, window : window + width]
        - cumsum[:, window : window + height, :width]
        + cumsum[:, :height, :width]
    )
    return (total / float(window**2)).astype(np.float32, copy=False)


def _normal_tail_multiplier(pfa: float):
    np = _load_numpy()
    clipped = float(np.clip(pfa, np.finfo(np.float32).tiny, 1.0))
    return float(np.sqrt(max(0.0, -2.0 * np.log(clipped))))


def _cfar_numpy(array, *, pfa: float, guard_px: int, training_radius_px: int, epsilon: float) -> dict[str, Any]:
    np = _load_numpy()
    evidence = np.maximum(array, 0.0).astype(np.float32, copy=False)
    outer_area = float((2 * training_radius_px + 1) ** 2)
    guard_area = float((2 * guard_px + 1) ** 2)
    training_area = outer_area - guard_area
    outer_mean = _box_mean(evidence, training_radius_px)
    outer_sq_mean = _box_mean(evidence * evidence, training_radius_px)
    guard_mean = _box_mean(evidence, guard_px)
    guard_sq_mean = _box_mean(evidence * evidence, guard_px)

    local_mean = ((outer_mean * outer_area) - (guard_mean * guard_area)) / training_area
    local_sq_mean = ((outer_sq_mean * outer_area) - (guard_sq_mean * guard_area)) / training_area
    local_var = np.maximum(local_sq_mean - (local_mean * local_mean), 0.0)
    local_std = np.sqrt(local_var + epsilon).astype(np.float32, copy=False)
    score = np.maximum((evidence - local_mean) / (local_std + epsilon), 0.0).astype(np.float32, copy=False)
    threshold_z = _normal_tail_multiplier(float(pfa))
    mask = score >= threshold_z
    return {
        "mask": mask,
        "score": score,
        "local_mean": local_mean.astype(np.float32, copy=False),
        "local_std": local_std,
        "threshold_z": threshold_z,
        "active_fraction": float(np.mean(mask)),
    }


def _cfar_torch_cuda(array, *, pfa: float, guard_px: int, training_radius_px: int, epsilon: float) -> dict[str, Any]:
    np = _load_numpy()
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - guarded by device resolution
        raise RuntimeError("Torch CUDA backend was selected, but torch could not be imported.") from exc
    if not bool(torch.cuda.is_available()):  # pragma: no cover - guarded by device resolution
        raise RuntimeError("Torch CUDA backend was selected, but CUDA is no longer available.")

    threshold_z = _normal_tail_multiplier(float(pfa))
    with torch.no_grad():
        evidence = torch.as_tensor(array, dtype=torch.float32, device="cuda").clamp_min(0.0)
        outer_area = float((2 * training_radius_px + 1) ** 2)
        guard_area = float((2 * guard_px + 1) ** 2)
        training_area = outer_area - guard_area
        outer_mean = _box_mean_torch(evidence, training_radius_px)
        outer_sq_mean = _box_mean_torch(evidence * evidence, training_radius_px)
        guard_mean = _box_mean_torch(evidence, guard_px)
        guard_sq_mean = _box_mean_torch(evidence * evidence, guard_px)
        local_mean = ((outer_mean * outer_area) - (guard_mean * guard_area)) / training_area
        local_sq_mean = ((outer_sq_mean * outer_area) - (guard_sq_mean * guard_area)) / training_area
        local_var = (local_sq_mean - (local_mean * local_mean)).clamp_min(0.0)
        local_std = torch.sqrt(local_var + float(epsilon))
        score = ((evidence - local_mean) / (local_std + float(epsilon))).clamp_min(0.0)
        mask = score >= threshold_z
        active_fraction = float(mask.float().mean().item())
        return {
            "mask": mask.detach().cpu().numpy().astype(np.bool_, copy=False),
            "score": score.detach().cpu().numpy().astype(np.float32, copy=False),
            "local_mean": local_mean.detach().cpu().numpy().astype(np.float32, copy=False),
            "local_std": local_std.detach().cpu().numpy().astype(np.float32, copy=False),
            "threshold_z": threshold_z,
            "active_fraction": active_fraction,
        }


def _cfar_cupy_cuda(array, *, pfa: float, guard_px: int, training_radius_px: int, epsilon: float) -> dict[str, Any]:
    np = _load_numpy()
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover - guarded by device resolution
        raise RuntimeError("CuPy CUDA backend was selected, but cupy could not be imported.") from exc

    threshold_z = _normal_tail_multiplier(float(pfa))
    evidence = cp.maximum(cp.asarray(array, dtype=cp.float32), 0.0)
    outer_area = float((2 * training_radius_px + 1) ** 2)
    guard_area = float((2 * guard_px + 1) ** 2)
    training_area = outer_area - guard_area
    outer_mean = _box_mean_cupy(evidence, training_radius_px)
    outer_sq_mean = _box_mean_cupy(evidence * evidence, training_radius_px)
    guard_mean = _box_mean_cupy(evidence, guard_px)
    guard_sq_mean = _box_mean_cupy(evidence * evidence, guard_px)
    local_mean = ((outer_mean * outer_area) - (guard_mean * guard_area)) / training_area
    local_sq_mean = ((outer_sq_mean * outer_area) - (guard_sq_mean * guard_area)) / training_area
    local_var = cp.maximum(local_sq_mean - (local_mean * local_mean), 0.0)
    local_std = cp.sqrt(local_var + float(epsilon)).astype(cp.float32, copy=False)
    score = cp.maximum((evidence - local_mean) / (local_std + float(epsilon)), 0.0).astype(cp.float32, copy=False)
    mask = score >= threshold_z
    return {
        "mask": cp.asnumpy(mask).astype(np.bool_, copy=False),
        "score": cp.asnumpy(score).astype(np.float32, copy=False),
        "local_mean": cp.asnumpy(local_mean).astype(np.float32, copy=False),
        "local_std": cp.asnumpy(local_std).astype(np.float32, copy=False),
        "threshold_z": threshold_z,
        "active_fraction": float(cp.mean(mask).get()),
    }


def robust_local_cfar(
    video,
    *,
    pfa: float = 0.001,
    guard_px: int = 2,
    training_radius_px: int = 11,
    epsilon: float = 1e-6,
    device: str = "cpu",
) -> dict[str, Any]:
    """Score positive excursions against an annular local background estimate.

    The training ring estimates local mean and variance while excluding a guard
    box around each test pixel. This keeps bright local structures from forcing
    one global threshold while reducing the chance that a candidate trains on
    its own signal.
    """
    device_spec = resolve_device(device)
    np, array = _validate_video(video)
    guard_px = int(guard_px)
    training_radius_px = int(training_radius_px)
    if guard_px < 0:
        raise ValueError("guard_px must be non-negative.")
    if training_radius_px <= guard_px:
        raise ValueError("training_radius_px must be larger than guard_px.")
    epsilon = float(epsilon)
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    evidence = np.maximum(array, 0.0).astype(np.float32, copy=False)
    if device_spec.resolved == "cuda":
        if device_spec.backend == "torch_cuda":
            result = _cfar_torch_cuda(
                evidence,
                pfa=pfa,
                guard_px=guard_px,
                training_radius_px=training_radius_px,
                epsilon=epsilon,
            )
        elif device_spec.backend == "cupy_cuda":
            result = _cfar_cupy_cuda(
                evidence,
                pfa=pfa,
                guard_px=guard_px,
                training_radius_px=training_radius_px,
                epsilon=epsilon,
            )
        else:  # pragma: no cover - resolve_device constrains backends
            raise RuntimeError(f"Unsupported CUDA backend '{device_spec.backend}'.")
    else:
        result = _cfar_numpy(
            evidence,
            pfa=pfa,
            guard_px=guard_px,
            training_radius_px=training_radius_px,
            epsilon=epsilon,
        )

    result["device"] = device_spec.as_dict()
    return result


def gamma_cfar_mask(
    video,
    *,
    pfa: float = 0.001,
    guard_px: int = 2,
    training_radius_px: int = 11,
    epsilon: float = 1e-6,
    device: str = "cpu",
):
    """Return only the boolean candidate mask from :func:`robust_local_cfar`."""
    return robust_local_cfar(
        video,
        pfa=pfa,
        guard_px=guard_px,
        training_radius_px=training_radius_px,
        epsilon=epsilon,
        device=device,
    )["mask"]
