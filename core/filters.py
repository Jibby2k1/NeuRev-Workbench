# core/filters.py
"""Core spatio-temporal filtering and feature extraction algorithms."""
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cupy as cp
    from tqdm import tqdm
except ImportError:
    cp = None
    tqdm = lambda x, **kwargs: x

# --- Gamma Filter Components ---

class GammaKernelFeatures(nn.Module):
    """Applies separable spatio-temporal gamma kernel convolutions."""
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def forward(self, video: torch.Tensor, **kwargs):
        is_3d = video.dim() == 3
        vid = video.unsqueeze(0) if is_3d else video
        
        temporal_kernels = kwargs.get('temporal_kernels')
        temporal_features = self._temporal_conv(vid, temporal_kernels) if temporal_kernels is not None else vid.unsqueeze(1)
        
        kernels_2d = kwargs.get('kernels_2d')
        feats = self._separable_spatial_conv(temporal_features, kernels_2d) if kernels_2d is not None else temporal_features
        
        return feats.squeeze(0) if is_3d else feats

    def _temporal_conv(self, video_tensor, temporal_kernels):
        B, T, H, W = video_tensor.shape
        N_k = temporal_kernels.shape[0]
        padding = temporal_kernels.shape[-1] // 2
        reshaped = video_tensor.permute(0, 2, 3, 1).reshape(B * H * W, 1, T)
        convolved = F.conv1d(reshaped, temporal_kernels.unsqueeze(1), padding=padding)
        return convolved.view(B, H, W, N_k, T).permute(0, 3, 4, 1, 2)

    def _separable_spatial_conv(self, temporal_features, spatial_kernels):
        B, N_t, T, H, W = temporal_features.shape
        N_s = spatial_kernels.shape[0]
        padding = (spatial_kernels.shape[-2] // 2, spatial_kernels.shape[-1] // 2)
        reshaped = temporal_features.permute(0, 2, 1, 3, 4).reshape(B * T, N_t, H, W)
        convolved = F.conv2d(reshaped, spatial_kernels.unsqueeze(1).repeat(N_t, 1, 1, 1), padding=padding, groups=N_t)
        return convolved.view(B, T, N_t * N_s, H, W).permute(0, 2, 1, 3, 4)

def specify_gamma_kernel(spec_type: str, **kwargs) -> Tuple[float, float]:
    """Specifies gamma kernel parameters from high-level descriptions."""
    if spec_type == 'center-peaked':
        half_decay = float(kwargs['half_decay_radius'])
        if half_decay <= 0: return 0.0, 0.0
        n = 1.0
        mu = np.log(2) / half_decay
        return n, mu
    raise ValueError(f"Unknown gamma kernel spec_type: {spec_type}")

def generate_gamma_kernel(dim: int, n: float, mu: float, shape, device=None) -> torch.Tensor:
    """Generates a normalized gamma kernel."""
    if dim == 1:
        coords = torch.arange(-shape // 2 + 1, shape // 2 + 1, dtype=torch.float32, device=device)
        r = torch.abs(coords)
    else:
        y = torch.arange(-shape[0] // 2 + 1, shape[0] // 2 + 1, device=device)
        x = torch.arange(-shape[1] // 2 + 1, shape[1] // 2 + 1, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        r = torch.sqrt(xx**2 + yy**2)
    
    k = r.pow(n - 1) * torch.exp(-mu * r)
    k[r == 0] = 0.0 if n > 1 else 1.0 # Handle center point
    
    norm = torch.sum(k)
    return k / norm if norm > 1e-9 else k

# --- Kalman Filter Components (if used) ---

def apply_kalman_mcc_filter(x_prev: np.ndarray, frame: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    e = frame - x_prev; norm2 = np.sum(e * e, axis=-1, keepdims=True)
    K = np.exp(-norm2 / (2 * sigma**2)); grad = (1 / sigma**2) * K * e
    return x_prev + mu * grad

def full_kalman_mcc_filter(frames: np.ndarray, sigma: float, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    max_val = np.iinfo(frames.dtype).max if np.issubdtype(frames.dtype, np.integer) else 1.0
    frames_norm = frames.astype(np.float32) / max_val
    if frames_norm.ndim == 3: frames_norm = frames_norm[..., None]
    T, H, W, C = frames_norm.shape; bg = np.zeros((T, H, W, C), dtype=np.float32)
    x_prev = frames_norm[0]; bg[0] = x_prev
    for t in tqdm(range(1, T), desc="Kalman-MCC (CPU)", leave=False, ncols=80):
        x_prev = apply_kalman_mcc_filter(x_prev, frames_norm[t], sigma, mu); bg[t] = x_prev
    bg_uint16 = np.clip(bg * max_val, 0, max_val).astype(np.uint16)
    if C == 1: bg_uint16 = bg_uint16[..., 0]
    original_uint16 = frames.astype(np.uint16) if frames.dtype != np.uint16 else frames
    diff = np.abs(original_uint16 - bg_uint16)
    return bg_uint16, diff.astype(np.uint16)

def apply_kalman_mcc_filter_gpu(x_prev: cp.ndarray, frame: cp.ndarray, sigma: float, mu: float) -> cp.ndarray:
    e = frame - x_prev; norm2 = cp.sum(e * e, axis=-1, keepdims=True)
    K = cp.exp(-norm2 / (2 * cp.power(cp.asarray(sigma), 2))); grad = (1 / cp.power(cp.asarray(sigma), 2)) * K * e
    return x_prev + mu * grad

def full_kalman_mcc_filter_gpu_batches(frames: np.ndarray, sigma: float, mu: float, num_batches: int) -> Tuple[np.ndarray, np.ndarray]:
    if cp is None: raise ImportError("cupy is not installed. Cannot use GPU Kalman filter.")
    max_val = np.iinfo(frames.dtype).max if np.issubdtype(frames.dtype, np.integer) else 1.0
    T, H, W = frames.shape[:3]; has_channel = (frames.ndim == 4)
    bg_out, diff_out = np.zeros_like(frames, dtype=np.uint16), np.zeros_like(frames, dtype=np.uint16)
    first_frame = frames[0].astype(np.float32) / max_val
    if not has_channel: first_frame = first_frame[..., None]
    x_prev = cp.asarray(first_frame); bg0 = cp.clip(x_prev * max_val, 0, max_val).astype(cp.uint16)
    bg_out[0] = cp.asnumpy(bg0[..., 0] if not has_channel else bg0)
    batch_size = int(np.ceil(T / num_batches)) if num_batches > 0 else T
    for b in range(num_batches):
        start, end = b * batch_size, min(T, (b + 1) * batch_size)
        if start >= end: continue
        batch_cpu = frames[start:end].astype(np.float32) / max_val
        if not has_channel: batch_cpu = batch_cpu[..., None]
        batch_gpu = cp.asarray(batch_cpu); original_frames_u16_gpu = cp.asarray(frames[start:end].astype(np.uint16))
        if not has_channel: original_frames_u16_gpu = original_frames_u16_gpu[..., None]
        for i in range(end - start):
            t = start + i
            if t == 0: continue
            x_prev = apply_kalman_mcc_filter_gpu(x_prev, batch_gpu[i], sigma, mu)
            bg_u16 = cp.clip(x_prev * max_val, 0, max_val).astype(cp.uint16)
            bg_out[t] = cp.asnumpy(bg_u16[..., 0] if not has_channel else bg_u16)
            diff_u16 = cp.abs(original_frames_u16_gpu[i] - bg_u16)
            diff_out[t] = cp.asnumpy(diff_u16[..., 0] if not has_channel else diff_u16)
        del batch_gpu, original_frames_u16_gpu; cp.get_default_memory_pool().free_all_blocks()
    return bg_out, diff_out


# --- Feature Map Dispatcher ---

def get_feature_map(video_np: np.ndarray, params: Dict, device: torch.device) -> Tuple[np.ndarray, str]:
    """
    Dispatcher function to compute feature maps based on specified filter type.
    """
    filter_type = params['varied_param']['filter_type']
    
    if filter_type == 'gamma':
        model = GammaKernelFeatures("spatio_temporal").to(device).eval()
        video_tensor_gpu = torch.from_numpy(video_np.astype(np.float32)).to(device)
        
        t_decay = params['varied_param']['t_decay']
        s_decay = params['varied_param']['s_decay']
        
        t_p = specify_gamma_kernel('center-peaked', half_decay_radius=t_decay)
        s_p = specify_gamma_kernel('center-peaked', half_decay_radius=s_decay)
        
        t_k = generate_gamma_kernel(1, *t_p, 19, device=device).unsqueeze(0) if t_decay > 0 else None
        s_k = generate_gamma_kernel(2, *s_p, (23, 23), device=device).unsqueeze(0) if s_decay > 0 else None
        
        with torch.amp.autocast(device_type=device.type), torch.no_grad():
            features_gpu = model(video_tensor_gpu, temporal_kernels=t_k, kernels_2d=s_k)
            if features_gpu.dim() == 4 and features_gpu.shape[0] == 1:
                features_gpu = features_gpu.squeeze(0)
                
        features_np = features_gpu.cpu().numpy()
        feature_name = "S.T. Enhanced"
        del video_tensor_gpu, features_gpu
        return features_np, feature_name
        
    elif filter_type == 'kalman_mcc':
        sigma = float(params['varied_param']['sigma'])
        mu    = float(params['varied_param']['mu'])
        try:
            import cupy as cp  # noqa: F401
            bg_u16, _ = full_kalman_mcc_filter_gpu_batches(video_np, sigma=sigma, mu=mu, num_batches=4)
        except Exception:
            bg_u16, _ = full_kalman_mcc_filter(video_np, sigma=sigma, mu=mu)

        vmax = np.iinfo(video_np.dtype).max if np.issubdtype(video_np.dtype, np.integer) else 1.0
        raw_f = video_np.astype(np.float32) / vmax
        bg_f  = bg_u16.astype(np.float32) / 65535.0
        feat  = np.clip(raw_f - bg_f, -1.0, 1.0)  # residual feature
        return feat.astype(np.float32), "MCC Residual"

    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")