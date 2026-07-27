"""Bounded fusion of Raw Direct with continuous pairwise activity evidence."""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from neurobench.algorithms.pairwise_separation import quiet_difference_stats, standardized_positive_mask
from neurobench.experiments.frame_difference import _atomic_json, _available_ram_mib, _sha256
from neurobench.experiments.learnable_contrast import core as v1
from neurobench.metrics.sparse_detection import temporal_pool

from .config import PairwiseSeparationConfig
from .evaluation import QUIET_DURATIONS, QUIET_STARTS, evaluate_lane, event_intervals


RAW_EXPECTED = 0.6056159420289855


@dataclass(frozen=True)
class PairwiseFusionConfig:
    schema_version: int
    experiment_id: str
    pairwise_config: Path
    pairwise_run_dir: Path
    output_dir: Path
    features: tuple[str, ...]
    additive_lambdas: tuple[float, ...]
    gate_floors: tuple[float, ...]
    feature_clip_z: float
    learned_max_lambda: float
    learned_learning_rate: float
    learned_epochs: int
    learned_l2: float
    visualizations: tuple[dict[str, Any], ...]
    cpu_threads: int
    max_ram_mib: int
    min_free_disk_mib: int
    max_output_mib: int

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseFusionConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        allowed = {
            "schema_version", "experiment_id", "pairwise_config", "pairwise_run_dir",
            "output_dir", "features", "additive_lambdas", "gate_floors",
            "feature_clip_z", "learning", "visualizations", "resources",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown fusion fields: {sorted(unknown)}")
        root = source.parent
        learning, resources = raw["learning"], raw["resources"]
        config = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            pairwise_config=(root / raw["pairwise_config"]).resolve(),
            pairwise_run_dir=(root / raw["pairwise_run_dir"]).resolve(),
            output_dir=(root / raw["output_dir"]).resolve(),
            features=tuple(str(x) for x in raw["features"]),
            additive_lambdas=tuple(float(x) for x in raw["additive_lambdas"]),
            gate_floors=tuple(float(x) for x in raw["gate_floors"]),
            feature_clip_z=float(raw["feature_clip_z"]),
            learned_max_lambda=float(learning["max_lambda"]),
            learned_learning_rate=float(learning["learning_rate"]),
            learned_epochs=int(learning["epochs"]),
            learned_l2=float(learning["l2_to_raw_initialization"]),
            visualizations=tuple(dict(x) for x in raw["visualizations"]),
            cpu_threads=int(resources["cpu_threads"]),
            max_ram_mib=int(resources["max_ram_mib"]),
            min_free_disk_mib=int(resources["min_free_disk_mib"]),
            max_output_mib=int(resources["max_output_mib"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        allowed_features = {"fixed_binary_difference", "adaptive_binary_difference", "infomax_tanh_ica"}
        if self.schema_version != 1 or not self.features or set(self.features) - allowed_features:
            raise ValueError("Fusion version 1 requires the declared fixed/adaptive/InfoMax features")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Fusion features must be unique")
        if not self.additive_lambdas or min(self.additive_lambdas) <= 0 or max(self.additive_lambdas) > 0.5:
            raise ValueError("Additive lambdas must be in (0, 0.5]")
        if not self.gate_floors or not all(0 < x < 1 for x in self.gate_floors):
            raise ValueError("Soft-gate floors must be in (0,1)")
        if self.feature_clip_z <= 0 or not 0 < self.learned_max_lambda <= 0.5:
            raise ValueError("Feature and learned-lambda bounds must be positive")
        if not 0 < self.learned_learning_rate <= 0.01 or not 1 <= self.learned_epochs <= 2000 or self.learned_l2 < 0:
            raise ValueError("Learning schedule exceeds the bounded tuning contract")
        if not 1 <= self.cpu_threads <= 8 or min(self.max_ram_mib, self.min_free_disk_mib, self.max_output_mib) <= 0:
            raise ValueError("Invalid fusion resource envelope")
        for spec in self.visualizations:
            if set(spec) != {"kind", "feature", "value", "filename"} or spec["feature"] not in self.features:
                raise ValueError("Invalid visualization specification")
            if spec["kind"] not in {"soft_gate", "additive"} or Path(spec["filename"]).name != spec["filename"]:
                raise ValueError("Visualization kind/filename is invalid")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("pairwise_config", "pairwise_run_dir", "output_dir"):
            payload[key] = str(payload[key])
        return payload


def soft_gate(original: np.ndarray, feature_unit: np.ndarray, floor: float) -> np.ndarray:
    if original.shape != feature_unit.shape or not 0 < floor < 1:
        raise ValueError("Soft-gate inputs must align and floor must be in (0,1)")
    return np.asarray(original, dtype=np.float32) * (floor + (1 - floor) * np.clip(feature_unit, 0, 1))


def additive_fusion(raw_residual: np.ndarray, feature_unit: np.ndarray, weight: float, feature_scale: float) -> np.ndarray:
    if raw_residual.shape != feature_unit.shape or not 0 <= weight <= 0.5 or feature_scale <= 0:
        raise ValueError("Invalid additive fusion inputs")
    return np.asarray(raw_residual, dtype=np.float32) + weight * feature_scale * np.clip(feature_unit, 0, 1)


def fit_bounded_lambda(
    raw_positive: np.ndarray,
    feature_positive: np.ndarray,
    raw_negative: np.ndarray,
    feature_negative: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
    maximum: float,
) -> tuple[float, list[float]]:
    """Tune one nonnegative residual weight with a pairwise logistic objective."""
    rp, fp, rn, fn = (np.asarray(x, dtype=np.float64).ravel() for x in
                      (raw_positive, feature_positive, raw_negative, feature_negative))
    if min(len(rp), len(rn)) == 0 or len(rp) != len(fp) or len(rn) != len(fn):
        raise ValueError("Positive and quiet-negative tuning samples are required")
    # Deterministic balanced comparisons avoid a quadratic materialization.
    count = max(len(rp), len(rn))
    pi = np.arange(count) % len(rp); ni = np.arange(count) % len(rn)
    base_delta = rp[pi] - rn[ni]; feature_delta = fp[pi] - fn[ni]
    weight = 0.0; history = []
    for _ in range(epochs):
        margin = np.clip(base_delta + weight * feature_delta, -40, 40)
        gradient = float(np.mean(-feature_delta / (1 + np.exp(margin))) + 2 * l2 * weight)
        weight = float(np.clip(weight - learning_rate * gradient, 0, maximum))
        history.append(float(np.mean(np.logaddexp(0, -margin)) + l2 * weight**2))
    return weight, history


def _raw_residual(raw: np.ndarray, quiet_count: int) -> tuple[np.ndarray, float]:
    baseline = np.median(raw[:quiet_count], axis=0)
    low, high = np.percentile(raw[:quiet_count, ::4, ::4], [1, 99.9])
    scale = max(float(high - low), 1e-6)
    return np.maximum((raw - baseline) / scale, 0).astype(np.float32), scale


def _feature_unit(path: Path, quiet_count: int, pairwise: PairwiseSeparationConfig, clip_z: float) -> np.ndarray:
    activity = np.load(path, mmap_mode="r", allow_pickle=False)
    lag = pairwise.preprocessing.lag_frames
    stats = quiet_difference_stats(activity[lag:quiet_count], floor_percentile=pairwise.thresholding.quiet_mad_floor_percentile)
    z, _ = standardized_positive_mask(activity, stats, pairwise.thresholding.primary_z_threshold, undefined_leading_frames=lag)
    return np.clip(np.maximum(z, 0) / clip_z, 0, 1).astype(np.float32)


def _residual_after_gate(gated: np.ndarray, quiet_count: int) -> np.ndarray:
    baseline = np.median(gated[:quiet_count], axis=0)
    low, high = np.percentile(gated[:quiet_count, ::4, ::4], [1, 99.9])
    return np.maximum((gated - baseline) / max(float(high - low), 1e-6), 0).astype(np.float32)


def _write_uint16_tiff(path: Path, values: np.ndarray, description: dict[str, Any], *, scale: float | None = None) -> None:
    if scale is None:
        scale = max(float(np.percentile(values[::4, ::4, ::4], 99.5)), 1e-6)
    temporary = path.with_suffix(path.suffix + ".partial")
    with tifffile.TiffWriter(temporary, bigtiff=True) as writer:
        for index, frame in enumerate(values):
            page = np.rint(np.clip(frame, 0, scale) / scale * 65535).astype(np.uint16)
            writer.write(page, photometric="minisblack", contiguous=True, metadata=None,
                         description=json.dumps(description | {"display_scale": [0, scale]}) if index == 0 else None)
    temporary.replace(path)


def preflight(config: PairwiseFusionConfig, *, artifact_dir: str | Path) -> dict[str, Any]:
    destination = Path(artifact_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Fusion preflight exists: {destination}")
    pairwise = PairwiseSeparationConfig.load(config.pairwise_config)
    required = [config.pairwise_run_dir / "run_state.json", config.pairwise_run_dir / "metrics.json",
                config.pairwise_run_dir / "label_projection_overlay.png"]
    for feature in config.features:
        required.append(config.pairwise_run_dir / "methods" / feature / "continuous_activity.npy")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    state = json.loads(required[0].read_text())
    if state.get("status") != "complete" or config.output_dir.exists():
        raise RuntimeError("Pairwise source run must be complete and fusion output unused")
    video = np.load(pairwise.source_video, mmap_mode="r", allow_pickle=False)
    frames = pairwise.frames.review_end_ui - pairwise.frames.review_start_ui + 1
    expected_shape = (frames, video.shape[1], video.shape[2])
    for feature in config.features:
        shape = np.load(config.pairwise_run_dir / "methods" / feature / "continuous_activity.npy", mmap_mode="r").shape
        if tuple(shape) != expected_shape:
            raise ValueError(f"Feature shape mismatch for {feature}: {shape}")
    output_bytes = len(config.visualizations) * int(np.prod(expected_shape)) * 2
    output_mib = math.ceil(output_bytes / 2**20)
    disk_mib = shutil.disk_usage(config.output_dir.parent).free // 2**20
    ready = output_mib <= config.max_output_mib and disk_mib >= config.min_free_disk_mib + output_mib and _available_ram_mib() >= config.max_ram_mib
    payload = {"schema_version":1, "experiment_id":config.experiment_id, "ready":ready,
        "pairwise_run":str(config.pairwise_run_dir), "source_shape":list(video.shape),
        "review_shape":list(expected_shape), "fixed_comparisons":1+len(config.features)*(len(config.additive_lambdas)+len(config.gate_floors)),
        "learned_outer_folds":len(config.features)*4, "visualization_count":len(config.visualizations),
        "resources":{"estimated_output_mib":output_mib,"output_cap_mib":config.max_output_mib,
                     "disk_free_mib":disk_mib,"ram_available_mib":_available_ram_mib(),"ram_cap_mib":config.max_ram_mib,
                     "cpu_threads":config.cpu_threads},
        "interpretation":"Derivative/ICA evidence is auxiliary; unmatched event candidates remain unknown, not false positives.",
        "inputs":[{"path":str(path),"bytes":path.stat().st_size,"sha256":_sha256(path)} for path in required],
    }
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config.pairwise_run_dir / "label_projection_overlay.png", destination / "label_projection_overlay.png")
    _atomic_json(destination / "config.resolved.json", config.to_dict()); _atomic_json(destination / "preflight.json", payload)
    if not ready:
        raise RuntimeError(f"Fusion preflight is not ready: {payload}")
    return payload


def _tuning_samples(raw_residual: np.ndarray, feature: np.ndarray, labels: list[dict[str, Any]], pairwise: PairwiseSeparationConfig, held_out: int) -> tuple[np.ndarray, ...]:
    intervals = event_intervals(labels, pairwise.frames.review_start_ui)
    positives_raw=[]; positives_feature=[]
    for burst,(start,stop) in intervals.items():
        if burst == held_out: continue
        rmap=temporal_pool(raw_residual[start:stop],"lme0.25"); fmap=feature[start:stop].max(axis=0)
        for row in (x for x in labels if int(x["burst_id"])==burst):
            y,x=int(row["y_px"]),int(row["x_px"]); ys=slice(max(0,y-2),y+3); xs=slice(max(0,x-2),x+3)
            positives_raw.append(float(rmap[ys,xs].max())); positives_feature.append(float(fmap[ys,xs].max()))
    quiet_raw=[]; quiet_feature=[]
    for start,duration in zip(QUIET_STARTS,QUIET_DURATIONS):
        rmap=temporal_pool(raw_residual[start:start+duration],"lme0.25"); fmap=feature[start:start+duration].max(axis=0)
        # Deterministic full-field hard negatives from the strongest Raw Direct pixels.
        indices=np.argpartition(rmap.ravel(),-256)[-256:]
        quiet_raw.extend(rmap.ravel()[indices].tolist()); quiet_feature.extend(fmap.ravel()[indices].tolist())
    return tuple(np.asarray(x,dtype=np.float64) for x in (positives_raw,positives_feature,quiet_raw,quiet_feature))


def run(config: PairwiseFusionConfig, *, preflight_dir: str | Path) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]=str(config.cpu_threads)
    if config.output_dir.exists(): raise FileExistsError(config.output_dir)
    audit=json.loads((Path(preflight_dir)/"preflight.json").read_text()); resolved=json.loads((Path(preflight_dir)/"config.resolved.json").read_text())
    if not audit.get("ready") or resolved != json.loads(json.dumps(config.to_dict())):
        raise ValueError("Fusion preflight does not match the requested configuration")
    pairwise=PairwiseSeparationConfig.load(config.pairwise_config); started=time.perf_counter()
    config.output_dir.mkdir(parents=True,exist_ok=False); _atomic_json(config.output_dir/"run_state.json",{"status":"running"})
    try:
        shutil.copy2(Path(preflight_dir)/"preflight.json",config.output_dir/"preflight.json"); shutil.copy2(Path(preflight_dir)/"label_projection_overlay.png",config.output_dir/"label_projection_overlay.png")
        _atomic_json(config.output_dir/"config.resolved.json",config.to_dict())
        video=np.load(pairwise.source_video,mmap_mode="r",allow_pickle=False); labels=v1.load_labels(pairwise.labels_tsv); f=pairwise.frames
        raw=np.asarray(video[f.review_start_ui-1:f.review_end_ui],dtype=np.float32); quiet_count=f.quiet_end_ui-f.quiet_start_ui+1
        raw_residual,_=_raw_residual(raw,quiet_count); raw_result,_,_=evaluate_lane("raw_direct",raw_residual,labels,pairwise,binary=False)
        valid=abs(raw_result["mean_recall"]-RAW_EXPECTED)<1e-12
        feature_scale=max(float(np.percentile(raw_residual[:quiet_count,::4,::4],99.5)),1e-6)
        rows=[raw_result]; learned=[]; visualization_dir=config.output_dir/"review_tiffs"; visualization_dir.mkdir()
        visual_specs={(x["kind"],x["feature"],float(x["value"])):x["filename"] for x in config.visualizations}
        for feature_name in config.features:
            feature=_feature_unit(config.pairwise_run_dir/"methods"/feature_name/"continuous_activity.npy",quiet_count,pairwise,config.feature_clip_z)
            for weight in config.additive_lambdas:
                lane=f"additive__{feature_name}__lambda_{weight:g}"; score=additive_fusion(raw_residual,feature,weight,feature_scale)
                result,_,_=evaluate_lane(lane,score,labels,pairwise,binary=False); result.update({"kind":"additive","feature":feature_name,"value":weight}); rows.append(result)
                filename=visual_specs.get(("additive",feature_name,weight))
                if filename: _write_uint16_tiff(visualization_dir/filename,score,{"kind":"additive_score","feature":feature_name,"lambda":weight,"axes":"TYX"})
                del score
            for floor in config.gate_floors:
                gated=soft_gate(raw,feature,floor); score=_residual_after_gate(gated,quiet_count)
                lane=f"soft_gate__{feature_name}__floor_{floor:g}"; result,_,_=evaluate_lane(lane,score,labels,pairwise,binary=False); result.update({"kind":"soft_gate","feature":feature_name,"value":floor}); rows.append(result)
                filename=visual_specs.get(("soft_gate",feature_name,floor))
                if filename: _write_uint16_tiff(visualization_dir/filename,gated,{"kind":"original_soft_gate","feature":feature_name,"floor":floor,"axes":"TYX"},scale=4095.0)
                del gated,score
            for held_out in sorted(event_intervals(labels,f.review_start_ui)):
                samples=_tuning_samples(raw_residual,feature,labels,pairwise,held_out)
                weight,history=fit_bounded_lambda(*samples,learning_rate=config.learned_learning_rate,epochs=config.learned_epochs,l2=config.learned_l2,maximum=config.learned_max_lambda)
                score=additive_fusion(raw_residual,feature,weight,feature_scale); result,_,_=evaluate_lane(f"learned__{feature_name}__outer_{held_out}",score,labels,pairwise,binary=False)
                fold=next(x for x in result["outer_folds"] if x["burst_id"]==held_out)
                learned.append({"feature":feature_name,"held_out_burst":held_out,"lambda":weight,"initial_lambda":0.0,
                    "learning_rate":config.learned_learning_rate,"epochs":config.learned_epochs,"loss_initial":history[0],"loss_final":history[-1],"held_out":fold})
                del score
            del feature
        # Leakage-safe fixed-spec selection: select on the other three bursts.
        nested=[]
        candidates=[x for x in rows if x["lane"]!="raw_direct"]
        for held_out in (1,2,3,4):
            def key(row):
                training=[x for x in row["outer_folds"] if x["burst_id"]!=held_out]
                return (np.mean([x["recall"] for x in training]),-sum(x["candidates"] for x in training),row["lane"])
            chosen=max(candidates,key=key); fold=next(x for x in chosen["outer_folds"] if x["burst_id"]==held_out)
            nested.append({"held_out_burst":held_out,"selected_lane":chosen["lane"],**fold})
        metrics={"schema_version":1,"raw_direct_anchor_valid":valid,"raw_direct":raw_result,"fixed_comparisons":rows,
            "nested_fixed_selection":{"outer_folds":nested,"mean_recall":float(np.mean([x["recall"] for x in nested])),"pooled_recall":sum(x["matched"] for x in nested)/79},
            "learned_scalar_outer_folds":learned,"precision_contract":"Unmatched event candidates remain unknown; known-label candidate fraction is not precision."}
        _atomic_json(config.output_dir/"metrics.json",metrics)
        best=max(candidates,key=lambda x:(x["mean_recall"],-x["total_event_candidates"],x["lane"]))
        summary={"schema_version":1,"experiment_id":config.experiment_id,"status":"complete" if valid else "invalid_baseline",
            "raw_direct_mean_recall":raw_result["mean_recall"],"best_descriptive_lane":best["lane"],"best_descriptive_mean_recall":best["mean_recall"],
            "nested_mean_recall":metrics["nested_fixed_selection"]["mean_recall"],"visualization_files":sorted(x["filename"] for x in config.visualizations),"elapsed_seconds":time.perf_counter()-started}
        _atomic_json(config.output_dir/"experiment_summary.json",summary); _atomic_json(config.output_dir/"run_state.json",{"status":"complete","elapsed_seconds":summary["elapsed_seconds"]})
        return summary
    except Exception as exc:
        _atomic_json(config.output_dir/"run_state.json",{"status":"failed","error":repr(exc)}); raise
