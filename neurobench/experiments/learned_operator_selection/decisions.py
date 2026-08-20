"""Stage state, dependencies, and machine-readable S0 decisions."""
from __future__ import annotations
from enum import Enum
from typing import Any

class Stage(str, Enum):
    S0_BASELINE = "S0_BASELINE"
    S1_OPERATOR_SCREEN = "S1_OPERATOR_SCREEN"
    S2_GLOBAL_MIXTURE = "S2_GLOBAL_MIXTURE"
    S3_CONTINUOUS_SCALE = "S3_CONTINUOUS_SCALE"
    S4_ADAPTIVE_SELECTION = "S4_ADAPTIVE_SELECTION"
    S5_ICA_UTILITY = "S5_ICA_UTILITY"
    S6_FINAL_AUDIT = "S6_FINAL_AUDIT"

ALLOWED_STATES = {"not_started", "preflight_ready", "running", "completed", "failed_operationally", "advance", "advance_with_caution", "diagnostic_rescue", "stop_branch", "stop_program", "awaiting_human_review"}
DEPENDENCIES = {
    Stage.S0_BASELINE: (), Stage.S1_OPERATOR_SCREEN: (Stage.S0_BASELINE,),
    Stage.S2_GLOBAL_MIXTURE: (Stage.S1_OPERATOR_SCREEN,),
    Stage.S3_CONTINUOUS_SCALE: (Stage.S2_GLOBAL_MIXTURE,),
    Stage.S4_ADAPTIVE_SELECTION: (Stage.S3_CONTINUOUS_SCALE,),
    Stage.S5_ICA_UTILITY: (Stage.S0_BASELINE,), Stage.S6_FINAL_AUDIT: (Stage.S5_ICA_UTILITY,),
}

def initial_program_state(experiment_id: str) -> dict[str, Any]:
    return {"schema_version": 1, "experiment_id": experiment_id, "stages": {stage.value: {"state": "not_started", "rescue_consumed": False} for stage in Stage}}

def prerequisite_decision(stage: Stage, state: dict[str, Any]) -> dict[str, Any]:
    required = [item.value for item in DEPENDENCIES[stage]]
    missing = [item for item in required if state.get("stages", {}).get(item, {}).get("state") not in {"advance", "advance_with_caution"}]
    return {"permitted": not missing, "required": required, "blocking": missing}

def s0_decision(summary: dict[str, Any], *, rescue_consumed: bool = False) -> dict[str, Any]:
    failed = [name for name, value in summary["invariants"].items() if not value]
    decision = "advance" if not failed else ("stop_program" if rescue_consumed else "diagnostic_rescue")
    return {
        "stage": Stage.S0_BASELINE.value, "decision": decision,
        "primary_metric": "Macro-KPR@58", "comparator": "documented_raw_direct_invariants",
        "delta_primary": summary["macro_kpr_at_58"] - summary["expected"]["macro_kpr_at_58"],
        "pooled_match_delta": summary["matches_at_58"] - summary["expected"]["matches_at_58"],
        "burst_delta_vector": summary["burst_delta_vector"],
        "numerical_health": "pass" if summary["invariants"]["finite_outputs"] else "fail",
        "preservation_status": "not_applicable_raw_direct_anchor", "reason_codes": failed,
        "rescue_consumed": rescue_consumed,
        "next_allowed_stages": [Stage.S1_OPERATOR_SCREEN.value, Stage.S5_ICA_UTILITY.value] if not failed else [],
    }
