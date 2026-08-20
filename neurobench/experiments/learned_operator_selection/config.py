"""Strict manifest for the learned whitening/operator-selection program."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

class LearnedOperatorConfigError(ValueError): pass

def _strict(value: Any, fields: set[str], scope: str) -> dict[str, Any]:
    if not isinstance(value, Mapping): raise LearnedOperatorConfigError(f"{scope} must be an object")
    payload = dict(value); unknown, missing = set(payload)-fields, fields-set(payload)
    if unknown or missing: raise LearnedOperatorConfigError(f"{scope}: unknown={sorted(unknown)} missing={sorted(missing)}")
    return payload

def _path(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value: raise LearnedOperatorConfigError(f"{field} must be a non-empty path")
    path = Path(value).expanduser(); return (base/path).resolve() if not path.is_absolute() else path.resolve()

@dataclass(frozen=True)
class FrameConfig: review_start_ui:int; review_end_ui:int; quiet_start_ui:int; quiet_end_ui:int; frame_period_ms:float
@dataclass(frozen=True)
class EvaluationConfig:
    temporal_pool:str; nms_distance_px:int; match_radius_px:int; fixed_candidates_per_burst:int; quiet_false_peaks_per_map:float
    expected_q1_macro_kpr:float; expected_q1_matches:int; expected_q1_candidates:int; expected_fixed_matches:int; expected_fixed_macro_kpr:float; exact_tolerance:float
@dataclass(frozen=True)
class DesignConfig: sampler:str; master_seed:int; master_size:int; allowed_prefix_sizes:tuple[int,...]; default_max_prefix_size:int; lhs_confirmation_size:int
@dataclass(frozen=True)
class OperatorBounds:
    spatial_width_px:tuple[int,int]; temporal_width_frames:tuple[int,int]; spatial_whitening_exponent:tuple[float,float]; temporal_whitening_exponent:tuple[float,float]
    shrinkage:tuple[float,float]; eigen_floor_ratio:tuple[float,float]; raw_preserving_blend:tuple[float,float]; maximum_joint_dimension:int; maximum_condition_number:float; minimum_effective_rank_fraction:float
@dataclass(frozen=True)
class StageGates: minimum_macro_delta:float; minimum_pooled_match_delta:int; minimum_nonnegative_bursts:int; maximum_rescues_per_branch:int; permitted_ica_ranks:tuple[int,...]
@dataclass(frozen=True)
class SeedConfig: screen:tuple[int,...]; confirmation:tuple[int,...]
@dataclass(frozen=True)
class ScientificAuditConfig: enabled:bool; opt_out_reason:str
@dataclass(frozen=True)
class ResourceConfig: device:str; cpu_threads:int; max_ram_mib:int; min_free_disk_mib:int; max_output_mib:int; gpu_reserve_mib:int

@dataclass(frozen=True)
class LearnedOperatorConfig:
    schema_version:int; experiment_id:str; source_video:Path; labels_tsv:Path; label_summary:Path; output_dir:Path
    frames:FrameConfig; evaluation:EvaluationConfig; design:DesignConfig; operator_bounds:OperatorBounds; stage_gates:StageGates
    model_seeds:SeedConfig; scientific_audit:ScientificAuditConfig; resources:ResourceConfig

    @classmethod
    def load(cls, source: str|Path) -> "LearnedOperatorConfig":
        manifest=Path(source).expanduser().resolve(); raw=_strict(json.loads(manifest.read_text()), {"schema_version","experiment_id","source_video","labels_tsv","label_summary","output_dir","frames","evaluation","design","operator_bounds","stage_gates","model_seeds","scientific_audit","resources"}, "top-level"); base=manifest.parent
        frames=_strict(raw["frames"],set(FrameConfig.__dataclass_fields__),"frames"); evaluation=_strict(raw["evaluation"],set(EvaluationConfig.__dataclass_fields__),"evaluation")
        design=_strict(raw["design"],set(DesignConfig.__dataclass_fields__),"design"); bounds=_strict(raw["operator_bounds"],set(OperatorBounds.__dataclass_fields__),"operator_bounds")
        gates=_strict(raw["stage_gates"],set(StageGates.__dataclass_fields__),"stage_gates"); seeds=_strict(raw["model_seeds"],set(SeedConfig.__dataclass_fields__),"model_seeds")
        audit=_strict(raw["scientific_audit"],set(ScientificAuditConfig.__dataclass_fields__),"scientific_audit"); resources=_strict(raw["resources"],set(ResourceConfig.__dataclass_fields__),"resources")
        config=cls(raw["schema_version"],raw["experiment_id"],_path(base,raw["source_video"],"source_video"),_path(base,raw["labels_tsv"],"labels_tsv"),_path(base,raw["label_summary"],"label_summary"),_path(base,raw["output_dir"],"output_dir"),FrameConfig(**frames),EvaluationConfig(**evaluation),DesignConfig(**{k:v for k,v in design.items() if k!="allowed_prefix_sizes"},allowed_prefix_sizes=tuple(design["allowed_prefix_sizes"])),OperatorBounds(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in bounds.items()}),StageGates(**{k:v for k,v in gates.items() if k!="permitted_ica_ranks"},permitted_ica_ranks=tuple(gates["permitted_ica_ranks"])),SeedConfig(tuple(seeds["screen"]),tuple(seeds["confirmation"])),ScientificAuditConfig(**audit),ResourceConfig(**resources))
        config.validate(); return config

    def validate(self) -> None:
        f,e,d,b,g,a,r=self.frames,self.evaluation,self.design,self.operator_bounds,self.stage_gates,self.scientific_audit,self.resources
        if self.schema_version!=1 or not self.experiment_id: raise LearnedOperatorConfigError("schema_version must be 1 and experiment_id non-empty")
        if not (1<=f.review_start_ui==f.quiet_start_ui<=f.quiet_end_ui<f.review_end_ui) or f.frame_period_ms!=20.0: raise LearnedOperatorConfigError("invalid frozen frame contract")
        if (e.temporal_pool,e.nms_distance_px,e.match_radius_px,e.fixed_candidates_per_burst,e.quiet_false_peaks_per_map)!=("lme0.25",6,6,58,1.0): raise LearnedOperatorConfigError("primary evaluator contract differs")
        if e.exact_tolerance<=0: raise LearnedOperatorConfigError("exact_tolerance must be positive")
        if d.sampler!="scrambled_sobol" or d.master_size!=64 or d.allowed_prefix_sizes!=(8,16,32,64) or d.default_max_prefix_size>32 or d.lhs_confirmation_size!=16: raise LearnedOperatorConfigError("invalid nested design contract")
        for pair in (b.spatial_width_px,b.temporal_width_frames,b.spatial_whitening_exponent,b.temporal_whitening_exponent,b.shrinkage,b.eigen_floor_ratio,b.raw_preserving_blend):
            if pair[0]>pair[1]: raise LearnedOperatorConfigError("operator lower bound exceeds upper bound")
        if b.maximum_joint_dimension>128 or b.maximum_condition_number>1e6 or b.minimum_effective_rank_fraction<.25: raise LearnedOperatorConfigError("joint whitening numerical gate exceeds contract")
        if (g.minimum_macro_delta,g.minimum_pooled_match_delta,g.minimum_nonnegative_bursts,g.maximum_rescues_per_branch)!=(.02,2,3,1) or g.permitted_ica_ranks!=(8,16): raise LearnedOperatorConfigError("promotion/rescue/ICA gates differ")
        if a.enabled or "essential-metrics-only sequential screening on 2026-08-19" not in a.opt_out_reason: raise LearnedOperatorConfigError("explicit screening audit opt-out missing")
        if r.device not in {"cpu","cuda"} or not 1<=r.cpu_threads<=8 or min(r.max_ram_mib,r.min_free_disk_mib,r.max_output_mib,r.gpu_reserve_mib)<=0: raise LearnedOperatorConfigError("invalid resource envelope")

    def to_dict(self) -> dict[str,Any]:
        payload=asdict(self)
        for key in ("source_video","labels_tsv","label_summary","output_dir"): payload[key]=str(getattr(self,key))
        return json.loads(json.dumps(payload))
