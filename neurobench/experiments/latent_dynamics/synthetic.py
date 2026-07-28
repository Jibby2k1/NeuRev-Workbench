"""Deterministic falsification fixtures for latent-dynamics inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    latent: np.ndarray
    observation: np.ndarray
    event_interval: tuple[int,int] | None
    parameters: dict[str,Any]


def generate_synthetic_case(
    case_id: str,
    *,
    seed: int=7,
    frames: int=160,
    signals: int=8,
    gamma: float=0.92,
    process_std: float=0.12,
    observation_std: float=0.35,
) -> SyntheticCase:
    if frames<20 or signals<1 or not 0<=gamma<1 or min(process_std,observation_std)<0:
        raise ValueError("Invalid synthetic dimensions or stable/noise parameters")
    rng=np.random.default_rng(seed); latent=np.zeros((frames,signals),dtype=np.float64); event=None
    if case_id in {"stable_ar1","model_mismatch","heteroscedastic","gain_offset_drift"}:
        for t in range(1,frames): latent[t]=gamma*latent[t-1]+rng.normal(0,process_std,signals)
    elif case_id in {"transient","two_events","impulsive_outlier","motion_edge"}:
        starts=[frames//3] if case_id!="two_events" else [frames//3,frames//3+12]
        for start in starts:
            drive=np.zeros(frames); drive[start]=3.0
            for t in range(1,frames): latent[t]=gamma*latent[t-1]+drive[t]
        event=(starts[0],min(frames,starts[-1]+30))
        if case_id=="motion_edge":
            latent[:]=0; latent[starts[0]:starts[0]+2,:signals//2 or 1]=2; event=(starts[0],starts[0]+2)
    elif case_id=="slow_ramp":
        start=frames//4; stop=3*frames//4; latent[start:stop]=np.linspace(0,2,stop-start)[:,None]; latent[stop:]=2; event=(start,stop)
    elif case_id in {"constant","pure_noise","identity"}:
        latent[:]=1.0 if case_id in {"constant","identity"} else 0.0
    else: raise ValueError(f"Unknown synthetic case: {case_id}")
    std=np.full((frames,signals),observation_std)
    if case_id=="heteroscedastic": std*=np.linspace(.5,2,frames)[:,None]
    observation=latent+rng.normal(0,std)
    if case_id=="identity": observation=latent.copy()
    if case_id=="impulsive_outlier": observation[frames//2,0]+=12
    if case_id=="gain_offset_drift":
        gain=np.linspace(.9,1.1,frames)[:,None]; offset=np.linspace(-.5,.5,frames)[:,None]; observation=observation*gain+offset
    return SyntheticCase(case_id,latent.astype(np.float32),observation.astype(np.float32),event,
                         {"seed":seed,"frames":frames,"signals":signals,"gamma":gamma,
                          "process_std":process_std,"observation_std":observation_std})


def synthetic_suite(seeds=(7,13,19,29,37)) -> list[SyntheticCase]:
    kinds=("constant","stable_ar1","transient","slow_ramp","two_events","pure_noise",
           "impulsive_outlier","heteroscedastic","gain_offset_drift","motion_edge","model_mismatch")
    return [generate_synthetic_case(kind,seed=seed) for seed in seeds for kind in kinds]
