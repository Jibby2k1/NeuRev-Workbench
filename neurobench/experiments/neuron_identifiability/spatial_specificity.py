"""Prespecified site-level spatial specificity with matched controls."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .contracts import ObservationRecord
from .trace_extraction import Geometry, _disk_trace, _ring_trace


def specificity(center: float, annulus: float) -> float:
    return float((center-annulus)/(abs(center)+abs(annulus)+np.finfo(float).eps))


def matched_control_rows(video: np.ndarray, records: list[ObservationRecord], geometry: Geometry, *, seed: int = 20260822) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sites = {(round(r.x_px),round(r.y_px)) for r in records}
    by_site: dict[str, tuple[np.ndarray,np.ndarray]] = {}
    rows = []
    excluded = [(max(0,r.start_zero-30),min(len(video),r.stop_zero_exclusive+30)) for r in records]
    for record in records:
        if record.observation_site_id not in by_site:
            by_site[record.observation_site_id] = (
                _disk_trace(video,record.x_px,record.y_px,geometry.center_radius),
                _ring_trace(video,record.x_px,record.y_px,geometry.annulus_inner,geometry.annulus_outer),
            )
        center_trace, annulus_trace = by_site[record.observation_site_id]
        duration = record.stop_zero_exclusive-record.start_zero
        base = slice(max(0,record.start_zero-20),record.start_zero); event = slice(record.start_zero,record.stop_zero_exclusive)
        center_amp = float(np.mean(center_trace[event])-np.median(center_trace[base])); annulus_amp = float(np.mean(annulus_trace[event])-np.median(annulus_trace[base])); observed_native=center_amp-annulus_amp; observed = specificity(center_amp,annulus_amp)
        candidates = [s for s in range(20,len(video)-duration-1) if all(s+duration<a or s>b for a,b in excluded)]
        quiet_start = int(candidates[int(stable_index(record.observation_id,len(candidates)))])
        qb=slice(quiet_start-20,quiet_start); qe=slice(quiet_start,quiet_start+duration)
        quiet_center=float(np.mean(center_trace[qe])-np.median(center_trace[qb])); quiet_annulus=float(np.mean(annulus_trace[qe])-np.median(annulus_trace[qb])); quiet_native=quiet_center-quiet_annulus; quiet = specificity(quiet_center,quiet_annulus)
        offsets=[(-32,0),(32,0),(0,-32),(0,32),(-24,-24),(24,24),(-24,24),(24,-24)]
        spatial=[]
        target_baseline=float(np.median(center_trace[base]))
        for dx,dy in offsets:
            x,y=record.x_px+dx,record.y_px+dy
            if min(x,y)<geometry.outer_outer or x>=video.shape[2]-geometry.outer_outer or y>=video.shape[1]-geometry.outer_outer: continue
            if any((x-sx)**2+(y-sy)**2<geometry.outer_outer**2 for sx,sy in sites): continue
            ct=_disk_trace(video,x,y,geometry.center_radius); at=_ring_trace(video,x,y,geometry.annulus_inner,geometry.annulus_outer)
            cb=float(np.median(ct[base])); ca=float(np.mean(ct[event])-cb); aa=float(np.mean(at[event])-np.median(at[base]))
            spatial.append((abs(cb-target_baseline),specificity(ca,aa),ca-aa,x,y))
        spatial.sort(key=lambda item:item[0]); selected=spatial[0] if spatial else (float("nan"),float("nan"),float("nan"),float("nan"),float("nan"))
        rows.append({"observation_id":record.observation_id,"observation_site_id":record.observation_site_id,"burst_id":record.burst_id,"geometry":f"center_r{geometry.center_radius}_annulus_r{geometry.annulus_inner}_{geometry.annulus_outer}","observed_specificity":observed,"quiet_shift_specificity":quiet,"matched_spatial_specificity":selected[1],"observed_native_center_minus_annulus":observed_native,"quiet_native_center_minus_annulus":quiet_native,"matched_spatial_native_center_minus_annulus":selected[2],"spatial_baseline_difference":selected[0],"matched_x_px":selected[3],"matched_y_px":selected[4],"observed_minus_quiet":observed-quiet,"observed_minus_spatial":observed-selected[1],"native_observed_minus_quiet":observed_native-quiet_native,"native_observed_minus_spatial":observed_native-selected[2]})
    return rows


def stable_index(text: str, size: int) -> int:
    import hashlib
    return int(hashlib.sha256(text.encode()).hexdigest()[:12],16)%size


def clustered_interval(rows: list[dict[str,Any]], key: str, seed: int, draws: int=5000) -> dict[str,Any]:
    grouped: dict[str,list[float]]=defaultdict(list)
    for row in rows:
        value=float(row[key])
        if np.isfinite(value): grouped[str(row["observation_site_id"])].append(value)
    site_values=np.asarray([np.mean(values) for values in grouped.values()]); rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(site_values,(draws,len(site_values)),replace=True),axis=1)
    return {"site_count":len(site_values),"mean":float(np.mean(site_values)),"ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
