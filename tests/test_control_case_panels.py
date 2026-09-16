"""Focused contracts for the deterministic postprocessing-only case panels."""
from copy import deepcopy

import pytest

from neurobench.experiments.gamma_ls_difference import control_case_panels as panels


def _cohort():
    experts = [{"observation_id":f"b1_roi_{i}","canonical_roi_id":f"roi_{i}","burst_id":1,
                "x_px":1.6,"y_px":1.2,"source_start_ui":1807,"source_stop_ui":1810} for i in range(5)]
    a,z = [],[]
    for i,expert in enumerate(experts):
        shared = {key:expert[key] for key in ("observation_id","canonical_roi_id","burst_id","x_px","y_px")}
        shared.update(window_start_frame_ui=1807,window_stop_frame_ui=1810,
                      matched_site_id="site_1",nearest_site_id="site_1")
        a.append({**shared,"matched":i in (0,1,4)})
        z.append({**shared,"matched":i in (0,2,4)})
    return experts,a,z


def test_selection_is_first_id_per_stratum_and_preserves_the_full_cohort():
    experts,a,z = _cohort()
    membership,selection = panels.stratify_occurrences(experts[::-1],a[::-1],z[::-1])
    assert len(membership) == 5
    assert [row["observation_id"] for row in selection] == ["b1_roi_0","b1_roi_1","b1_roi_2","b1_roi_3"]
    assert [row["occurrence_count"] for row in selection] == [2,1,1,1]
    assert membership[-1]["selected_for_panel"] is False
    assert panels.stratify_occurrences(experts,a,z) == (membership,selection)


def test_empty_strata_are_explicit_and_do_not_borrow_an_example():
    experts,a,z = _cohort()
    for row in a+z:
        row["matched"] = False
    membership,selection = panels.stratify_occurrences(experts,a,z)
    assert [row["selection_status"] for row in selection] == ["empty_stratum"]*3+["selected"]
    assert [row["observation_id"] for row in selection] == ["","","","b1_roi_0"]
    assert sum(row["selected_for_panel"] for row in membership) == 1


@pytest.mark.parametrize("damage",["duplicate","missing","coordinate","window","boolean"])
def test_stratum_join_rejects_identity_geometry_and_outcome_damage(damage):
    experts,a,z = deepcopy(_cohort())
    if damage == "duplicate":a.append(a[0])
    elif damage == "missing":z.pop()
    elif damage == "coordinate":z[0]["x_px"] += 1
    elif damage == "window":a[0]["window_start_frame_ui"] += 1
    else:z[0]["matched"] = "unknown"
    with pytest.raises(ValueError):
        panels.stratify_occurrences(experts,a,z)


def test_saved_occurrence_table_must_agree_with_the_recomputed_join():
    _,a,_ = _cohort()
    expected = [{**a[0],"optional":None}]
    saved = [{key:"" if value is None else str(value) for key,value in expected[0].items()}]
    panels._compare_occurrence_table(saved,expected)
    saved[0]["matched"] = "False"
    with pytest.raises(ValueError,match="disagrees"):
        panels._compare_occurrence_table(saved,expected)


def test_exact_center_trace_uses_source_frames_yx_and_five_frame_context():
    import numpy as np
    raw = np.arange(20*4*5,dtype=np.float32).reshape(20,4,5)
    a = raw/20
    m = np.ones_like(a)
    sigma = np.ones_like(a)*2
    contrast = a-m
    arrays = {"Raw":raw,"A":a,"M":m,"sigma":sigma,"contrast":contrast,"Z":contrast/(sigma+1e-6)}
    experts,ar,zr = _cohort()
    row = panels.stratify_occurrences(experts,ar,zr)[0][0]
    result = panels.case_traces(arrays,row,source_start=1800,source_stop=1819,floor=.5,epsilon=1e-6)
    assert result["frames"] == list(range(1802,1816))
    assert result["pixel_xy"] == [2,1]
    np.testing.assert_array_equal(result["values"]["Raw"],raw[2:16,1,2])
    arrays["Z"] = arrays["Z"]+1
    with pytest.raises(ValueError,match="Z trace differs"):
        panels.case_traces(arrays,row,source_start=1800,source_stop=1819,floor=.5,epsilon=1e-6)


def test_context_clips_to_available_data_without_clipping_the_burst():
    assert panels.frame_window(1801,1803,1800,1808) == list(range(1800,1809))
    with pytest.raises(ValueError,match="fully inside"):
        panels.frame_window(1799,1803,1800,1808)


def test_frozen_q1_threshold_must_be_unique_and_finite():
    point = {"target_proposals_per_frame":1,"threshold_z":2.25}
    assert panels._threshold({"operating_points":[point]}) == 2.25
    for points in ([point,point],[],[{**point,"threshold_z":float("nan")} ]):
        with pytest.raises(ValueError):
            panels._threshold({"operating_points":points})


def test_completed_panel_resume_rejects_missing_or_changed_artifacts(tmp_path):
    selection = [{"stratum":name,"selection_status":"empty_stratum","observation_id":"","occurrence_count":0}
                 for name in panels.STRATA]
    selection[-1].update(selection_status="selected",observation_id="b1_roi_0",occurrence_count=79)
    artifacts = []
    names = ["panel_contract.json","all79_strata.tsv","selection.tsv","README.md"]
    names.extend(f"neither_matched__b1_roi_0{suffix}" for suffix in (".png",".json",".tsv"))
    for name in names:
        path = tmp_path/name
        path.write_text("sealed")
        artifacts.append(panels._record(path))
    manifest = {"full_cohort_occurrence_count":79,"panel_count":1,"selected_occurrence_ids":["b1_roi_0"],
                "selection":selection,"artifacts":artifacts}
    panels._verify_panel_inventory(tmp_path,manifest,selection)
    with pytest.raises(ValueError,match="inventory"):
        panels._verify_panel_inventory(tmp_path,{**manifest,"artifacts":artifacts[:-1]},selection)
    (tmp_path/"README.md").write_text("altered")
    with pytest.raises(ValueError,match="changed"):
        panels._verify_panel_inventory(tmp_path,manifest,selection)
