"""Version-agnostic CUDA device resolution for the Gamma-LS experiment.

PyTorch wheels carry their own CUDA user-space runtime.  This gate therefore
selects a logical CUDA device without pinning a toolkit release, then verifies
that the host driver has exposed a usable device before any output is mutated.
"""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Any


class CudaRuntimeUnavailable(RuntimeError):
    """Raised when PyTorch cannot use the requested logical CUDA device."""


def cuda_unavailable_message(
    torch: Any,
    device: str,
    *,
    device_nodes: tuple[str, ...] | None = None,
    kernel_release: str | None = None,
) -> str:
    """Build an actionable host/runtime diagnostic without assuming a distro."""

    nodes = (
        tuple(sorted(str(path) for path in Path("/dev").glob("nvidia*")))
        if device_nodes is None
        else tuple(device_nodes)
    )
    kernel = platform.release() if kernel_release is None else kernel_release
    backend_built = bool(torch.backends.cuda.is_built())
    build = str(torch.version.cuda) if torch.version.cuda is not None else "none"
    node_state = ",".join(nodes) if nodes else "absent"
    message = (
        f"PyTorch cannot use requested device {device!r} "
        f"(torch={torch.__version__}, torch CUDA build={build}, "
        f"CUDA backend built={backend_built}, running kernel={kernel}, "
        f"/dev/nvidia*={node_state})."
    )
    if not backend_built:
        return message + " Install a CUDA-enabled PyTorch build."
    if not nodes:
        return message + (
            " The NVIDIA kernel driver has not exposed a device; install or load "
            "a driver module matching the running kernel and restart the host. "
            "Changing the code's CUDA selector or user-space toolkit cannot create "
            "the missing device nodes."
        )
    return message + " Check the loaded NVIDIA driver and container/device permissions."


def require_cuda_device(device: str) -> dict[str, Any]:
    """Resolve and probe one logical CUDA device using the installed PyTorch build."""

    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - runtime dependent
        raise CudaRuntimeUnavailable("PyTorch is not installed") from error

    try:
        resolved = torch.device(device)
    except Exception as error:
        raise CudaRuntimeUnavailable(f"invalid CUDA device {device!r}") from error
    if resolved.type != "cuda":
        raise CudaRuntimeUnavailable(
            f"the Gamma-LS GPU runner requires a CUDA device, got {device!r}"
        )
    if not torch.cuda.is_available():
        raise CudaRuntimeUnavailable(cuda_unavailable_message(torch, device))

    try:
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        count = int(torch.cuda.device_count())
        if index < 0 or index >= count:
            raise CudaRuntimeUnavailable(
                f"CUDA device index {index} is outside the {count} visible device(s)"
            )
        torch.cuda.set_device(index)
        resolved_name = f"cuda:{index}"
        probe = torch.empty((1,), dtype=torch.float32, device=resolved_name)
        del probe
        torch.cuda.synchronize(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        properties = torch.cuda.get_device_properties(index)
    except CudaRuntimeUnavailable:
        raise
    except Exception as error:  # pragma: no cover - runtime dependent
        raise CudaRuntimeUnavailable(
            cuda_unavailable_message(torch, device)
            + f" CUDA allocation/synchronization failed: {type(error).__name__}."
        ) from error

    return {
        "requested_device": device,
        "resolved_device": resolved_name,
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "device_name": torch.cuda.get_device_name(index),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "visible_device_count": count,
        "free_vram_bytes_before": int(free_bytes),
        "total_vram_bytes": int(total_bytes),
        "cuda_available": True,
    }


__all__ = [
    "CudaRuntimeUnavailable",
    "cuda_unavailable_message",
    "require_cuda_device",
]
