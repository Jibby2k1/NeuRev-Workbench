"""Chronology, registration coordinates, and sealed shared-readout checks."""
import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import control_study as study


def test_declared_matrix_retains_both_operator_designs_and_raw_controls():
    arms=study.study_arms()
    assert len(arms)==28
    names={arm["arm_id"] for arm in arms}
    assert {"s0_a1_current_A", "s0_a1_difference_A", "centered_direct_Z", "centered_serial_Z",
            "legacy_carrier", "legacy_residual", "motion_Z", "global_offset_Z"} <= names
    for arm in arms:
        if arm["arm_id"] != "guard7_disk_Z" and not arm.get("historical"):
            assert arm["guard"]==0


@pytest.mark.parametrize("sigma,alpha",[(0,1),(0,.4),(1,1),(1,.4)])
def test_preprocessing_cannot_see_future_and_has_cold_zero(sigma,alpha):
    raw=np.random.default_rng(31).normal(100,3,(12,25,29)).astype(np.float32)
    changed=raw.copy();changed[8:]+=1000
    first=list(study.conditioned_frames(raw,sigma,alpha))
    second=list(study.conditioned_frames(changed,sigma,alpha))
    for a,b in zip(first[:8],second[:8]):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(first[0][1],0)
    if sigma==0 and alpha==1:
        np.testing.assert_array_equal(first[7][0],raw[7])
        np.testing.assert_array_equal(first[7][1],raw[7]-raw[6])


def test_registration_and_inverse_map_preserve_original_object_position():
    template=np.random.default_rng(5).normal(size=(35,39)).astype(np.float32)
    frame=np.roll(template,(3,-4),axis=(0,1))
    assert study.rigid_shift(frame,template)==(-3,4)
    transforms=[]
    registered=list(study.corrected_frames([frame],template,"motion",transforms))[0]
    np.testing.assert_array_equal(registered[5:-5,5:-5],template[5:-5,5:-5])
    restored=list(study.source_coordinate_frames([registered],transforms))[0]
    np.testing.assert_array_equal(restored[8:-8,8:-8],frame[8:-8,8:-8])
    assert study.rigid_shift(np.ones((20,21)),np.ones((20,21)))==(0,0)


def test_fixed_template_global_offset_is_causal():
    template=np.arange(100,dtype=np.float32).reshape(10,10)
    transforms=[]
    corrected=list(study.corrected_frames([template+9,template-4],template,"global_offset",transforms))
    np.testing.assert_array_equal(corrected,[template,template])
    assert [x["offset"] for x in transforms]==[9,-4]


def test_shared_display_scales_do_not_change_with_conditioner():
    arms={a["arm_id"]:a for a in study.study_arms()}
    assert study.display_group(arms["s0_a1_current_Z"],"Score")==study.display_group(arms["centered_serial_Z"],"Score")
    assert study.display_group(arms["s0_a1_difference_A"],"Input")==study.display_group(arms["motion_Z"],"Input")
    assert study.display_group(arms["legacy_carrier"],"Score") != study.display_group(arms["legacy_residual"],"Score")


def test_panel_order_survives_sorted_json_sealing():
    stages={name:{"path":name+".npy"} for name in ("Raw","Input","Mean","Score")}
    restored=json.loads(json.dumps(stages,sort_keys=True))
    assert list(study.ordered_stage_paths(restored))==["Raw","Input","Mean","Score"]
    restored.pop("Mean")
    assert list(study.ordered_stage_paths(restored))==["Raw","Input","Score"]


def test_score_arm_emits_only_application_frames_and_seals_outputs(tmp_path):
    # Quiet and application are deliberately different. A known later peak
    # must not raise the setup cutoff or leak into setup proposal rows.
    rng=np.random.default_rng(44)
    values=rng.normal(size=(560,32,36)).astype(np.float32)
    values[100:,16,18]+=10
    source=tmp_path/"source.npy";np.save(source,values)
    inputs={"files":{k:study.binding(source) for k in ("Raw","legacy_residual","legacy_carrier")}}
    (tmp_path/"protocol.json").write_text(json.dumps({"burst_windows":{"1":[1900,1905]}}))
    arm=next(a for a in study.study_arms() if a["arm_id"]=="legacy_carrier")
    seal=study.score_arm(tmp_path,arm,inputs,"cpu",8)
    folder=tmp_path/"arms/legacy_carrier"
    calibration=json.loads((folder/"calibration.json").read_text())
    for op in calibration["operating_points"]:
        assert op["calibration_source_frames_ui"]==list(range(1801,1900))
        assert op["calibration_proposal_count"]<=int(99*op["target_proposals_per_frame"])
    rows=study.read_tsv(folder/"q1/candidates.tsv")
    assert rows and min(int(x["source_frame_ui"]) for x in rows)==1900
    assert max(int(x["source_frame_ui"]) for x in rows)==2359
    assert any(x["x_px"]=="18" and x["y_px"]=="16" for x in rows)
    for item in seal["files"]:study.verify(item)
    assert study.score_arm(tmp_path,arm,inputs,"cpu",8)==seal
    with source.open("ab") as stream:stream.write(b"changed")
    with pytest.raises(ValueError,match="Changed sealed"):
        study.score_arm(tmp_path,arm,inputs,"cpu",8)
