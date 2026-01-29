# core/pipelines.py
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from .filters import (
    GammaKernelFeatures,
    specify_gamma_kernel,
    generate_gamma_kernel,
)
from .detection import CFAR


def _select_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_stats(name: str, t: torch.Tensor):
    v = t.detach()
    finite = torch.isfinite(v)
    total = v.numel()
    fin = int(finite.sum().item())
    if fin > 0:
        vf = v[finite]
        mn = float(vf.min().item())
        mx = float(vf.max().item())
        me = float(vf.mean().item())
        logging.info(f"[diag] {name}: min={mn:.4g} max={mx:.4g} mean={me:.4g} finite={fin}/{total}")
    else:
        logging.info(f"[diag] {name}: no finite values ({total} elements)")


def _gamma_st_reduce_to_single_channel(
    video_t: torch.Tensor,
    st_cfg: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Apply Gamma ST on (T,H,W) -> (T,1,H,W); if no kernels are requested, return identity.
    """
    assert video_t.ndim == 3, f"Expected (T,H,W); got {tuple(video_t.shape)}"
    video_t = video_t.to(device=device, dtype=torch.float32)

    # Build kernels based on half-decay radii
    t_decay = float(st_cfg.get("t_decay", 0.0))
    s_decay = float(st_cfg.get("s_decay", 0.0))

    t_k = None
    s_k = None
    if t_decay > 0:
        t_n, t_mu = specify_gamma_kernel("center-peaked", half_decay_radius=t_decay)
        t_k = generate_gamma_kernel(1, t_n, t_mu, 19, device=device).unsqueeze(0)
    if s_decay > 0:
        s_n, s_mu = specify_gamma_kernel("center-peaked", half_decay_radius=s_decay)
        s_k = generate_gamma_kernel(2, s_n, s_mu, (23, 23), device=device).unsqueeze(0)

    # 🔑 Identity fallback if both kernels are None
    if t_k is None and s_k is None:
        single = video_t  # identity
        out = single.unsqueeze(1).contiguous()  # (T,1,H,W)
        return out

    # Otherwise, use your Gamma feature bank
    model = GammaKernelFeatures("spatio_temporal").to(device).eval()
    with torch.no_grad():
        feats = model(video_t, temporal_kernels=t_k, kernels_2d=s_k)  # -> (C,T,H,W) expected
        if feats.dim() == 3:  # some versions may return (T,H,W)
            single = feats
        else:
            single = feats.mean(dim=0)  # (T,H,W)
        out = single.unsqueeze(1).contiguous()
        return out


def _ensure_tchw(video_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy (T,H,W) -> torch (T,H,W) float32 on device."""
    if video_np.ndim != 3:
        raise ValueError(f"Expected video as (T,H,W); got {video_np.shape}")
    return torch.from_numpy(video_np.astype(np.float32, copy=False)).to(device)


def run_single_stage(
    video_np: np.ndarray,
    p: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    dev = _select_device(device)
    vid_t = _ensure_tchw(video_np, dev)

    # ST1
    x1_tchw = _gamma_st_reduce_to_single_channel(vid_t, p["st1"], dev)
    if p.get("debug"):
        _log_stats("ST1", x1_tchw)

    # CFAR1 (float32; no autocast)
    cfar1 = CFAR(p["cfar1"]).to(dev).eval()
    with torch.no_grad():
        z1, _mask_ignored, _std1, _mean1 = cfar1(x1_tchw.float())  # (T,1,H,W)
    if p.get("debug"):
        _log_stats("z1", z1)

    # Single global threshold on z1
    final_mask = (z1 > float(p["z_thresh"])).to(torch.uint8)

    return {
        "arch": "single",
        "features1": x1_tchw.squeeze(1).detach().cpu().numpy(),
        "z1": z1.squeeze(1).detach().cpu().numpy(),
        "final_mask": final_mask.squeeze(1).detach().cpu().numpy(),
    }


def run_two_stage(
    video_np: np.ndarray,
    p: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    dev = _select_device(device)
    vid_t = _ensure_tchw(video_np, dev)

    # ST1
    x1_tchw = _gamma_st_reduce_to_single_channel(vid_t, p["st1"], dev)
    if p.get("debug"):
        _log_stats("ST1", x1_tchw)

    # CFAR1 -> z1
    cfar1 = CFAR(p["cfar1"]).to(dev).eval()
    with torch.no_grad():
        z1, _mask1, _std1, _mean1 = cfar1(x1_tchw.float())
    if p.get("debug"):
        _log_stats("z1", z1)

    # ST2 consumes NON-BINARIZED z1
    z1_thw = z1.squeeze(1)  # (T,H,W)
    x2_tchw = _gamma_st_reduce_to_single_channel(z1_thw, p["st2"], dev)
    if p.get("debug"):
        _log_stats("ST2", x2_tchw)

    # CFAR2 -> z2
    cfar2 = CFAR(p["cfar2"]).to(dev).eval()
    with torch.no_grad():
        z2, _mask2_ignored, _std2, _mean2 = cfar2(x2_tchw.float())
    if p.get("debug"):
        _log_stats("z2", z2)

    final_mask = (z2 > float(p["z_thresh"])).to(torch.uint8)

    return {
        "arch": "two_stage",
        "features1": x1_tchw.squeeze(1).detach().cpu().numpy(),
        "z1": z1.squeeze(1).detach().cpu().numpy(),
        "features2": x2_tchw.squeeze(1).detach().cpu().numpy(),
        "z2": z2.squeeze(1).detach().cpu().numpy(),
        "final_mask": final_mask.squeeze(1).detach().cpu().numpy(),
    }
