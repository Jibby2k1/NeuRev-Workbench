from neurobench.experiments.learned_operator_selection.decisions import Stage, initial_program_state, prerequisite_decision, s0_decision

def _summary(delta=0.0, matches=52):
    return {"macro_kpr_at_58":.6572463768115942+delta,"matches_at_58":matches,"expected":{"macro_kpr_at_58":.6572463768115942,"matches_at_58":52},"burst_delta_vector":[0,0,0,0],"invariants":{"finite_outputs":True,"baseline_exact":delta==0 and matches==52}}

def test_s0_advance_and_single_rescue_boundary() -> None:
    assert s0_decision(_summary())["decision"]=="advance"
    assert s0_decision(_summary(.019),rescue_consumed=False)["decision"]=="diagnostic_rescue"
    assert s0_decision(_summary(.019),rescue_consumed=True)["decision"]=="stop_program"

def test_stage_dependency_reads_decision_state() -> None:
    state=initial_program_state("x")
    assert prerequisite_decision(Stage.S1_OPERATOR_SCREEN,state)["permitted"] is False
    state["stages"][Stage.S0_BASELINE.value]["state"]="advance"
    assert prerequisite_decision(Stage.S1_OPERATOR_SCREEN,state)["permitted"] is True
