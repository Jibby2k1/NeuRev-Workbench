"""Postprocessing verifies seals/joins and never changes frozen predictions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference import control_review as review


def _arms():
    result = []
    for sigma in (0,1):
        for alpha in (1.0,.4):
            for representation in ("current","difference"):
                source = f"s{sigma}_a{alpha:g}_{representation}"
                for readout in (("A","contrast","Z") if (sigma,alpha)==(1,.4) else ("A","Z")):
                    result.append(dict(arm_id=f"{source}_{readout}",source=source,sigma=sigma,alpha=alpha,
                        representation=representation,correction="none",design="point",reference="gamma",
                        guard=0,geometry="square",readout=readout))
    result.extend(dict(arm_id=name,source=name,readout="native",historical=True) for name in ("legacy_residual","legacy_carrier"))
    base = dict(source="s1_a0.4_difference",sigma=1,alpha=.4,representation="difference",correction="none",reference="gamma",guard=0,geometry="square")
    for design, names in (("direct",("A","contrast","Z")),("serial",("Z",))):
        result.extend(dict(base,arm_id=f"centered_{design}_{name}",design=design,readout=name) for name in names)
    result.append(dict(base,arm_id="uniform_reference_Z",design="point",readout="Z",reference="uniform"))
    result.append(dict(base,arm_id="guard7_disk_Z",design="point",readout="Z",guard=7,geometry="disk"))
    result.extend(dict(base,arm_id=f"{name}_Z",source=f"{name}_difference",design="point",readout="Z",correction=name) for name in ("motion","global_offset"))
    return result


def _json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


def _rows(path, values, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or (list(values[0]) if values else ["empty"]),delimiter="\t")
        writer.writeheader();writer.writerows(values)


def _binding(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size)


def _complete_fixture(root):
    from neurobench.experiments.gamma_ls_difference.two_stencil_evaluation import evaluate_occurrence_windows
    windows={b:[1900+10*(b-1),1901+10*(b-1)] for b in range(1,5)}
    labels=[]
    for i in range(79):
        burst=min(4,i//20+1)
        labels.append(dict(observation_id=f"b{burst}__occ_{i:03d}",canonical_roi_id=f"roi_{i%26+1:03d}",burst_id=burst,
            source_start_ui=windows[burst][0],source_stop_ui=windows[burst][1],x_px=10+12*(i%26),y_px=20,
            temporal_extent_semantics="configured_burst_window_not_per_roi_onset"))
    labels_path=root/"expert_occurrences.tsv";_rows(labels_path,labels)
    helper=Path(review.__file__).with_name("two_stencil_evaluation.py")
    _json(root/"protocol.json",dict(arms=_arms(),target_proposals_per_frame=list(review.TARGETS),application_ui=[1900,2359],
        code_bindings=[_binding(helper)],labels=_binding(labels_path),burst_windows=windows))
    dense=root/"sealed_dense_stage.npy";dense.write_bytes(b"unused sealed dense stage fixture; must never be opened")
    dense_record=_binding(dense)
    labels=review._rows(labels_path)
    joined=evaluate_occurrence_windows([],labels,burst_intervals_ui=windows)
    seals=[];online=[];diagnostics=[]
    for arm in _arms():
        name=arm["arm_id"];folder=root/"arms"/name
        calibration=folder/"calibration.json"
        _json(calibration,dict(operating_points=[dict(target_proposals_per_frame=q,threshold_z=1,calibration_proposal_count=0) for q in review.TARGETS]))
        files=[_binding(calibration)]
        for q in review.TARGETS:
            qroot=folder/f"q{q:g}";candidate=qroot/"candidates.tsv"
            _rows(candidate,[],fields=["proposal_id","cell_id","target_proposals_per_frame","threshold_z","source_frame_ui","x_px","y_px","score"])
            files.append(_binding(candidate))
            for key in ("occurrence_rows","site_rows","membership_rows","burst_summaries"):_rows(qroot/f"{key}.tsv",joined[key])
            _json(qroot/"summary.json",joined["summary"])
            online.append(dict(arm_id=name,q=q,**joined["summary"]))
        sealed=folder/"sealed.json"
        _json(sealed,dict(arm=arm,stages={"Raw":dense_record,"Input":dense_record,"Score":dense_record},files=files,
            application_ui=[1900,2359],calibration_score_ui=[1801,1899]))
        seals.append(_binding(sealed))
        diagnostics.extend(dict(arm_id=name,observation_id=row["observation_id"],category="no_local_exceedance",local_peak=0,threshold=1,margin=-1) for row in labels)
    _json(root/"candidate_seal.json",dict(status="ALL_ARMS_SEALED_BEFORE_OUTCOME_JOIN",arms=28,seals=seals))
    _rows(root/"online_summary.tsv",online);_rows(root/"q1_stage_diagnostics.tsv",diagnostics)
    _json(root/"evaluation_complete.json",dict(status="NUMERICAL_PASS",arm_count=28,online_operating_points=140,diagnostic_occurrences=2212))
    return dense


def test_components_are_metadata_selected_and_only_guard_disk_is_confounded():
    from collections import Counter
    pairs=review.component_pairs(_arms())
    assert len(pairs)==48
    assert Counter(row["component"] for row in pairs)==dict(spatial_smoothing=8,temporal_smoothing=8,
        temporal_representation=9,readout=15,target_pooling=3,reference_construction=1,
        reference_weights=1,motion_correction=1,global_offset=1,guard_and_support=1)
    assert all("legacy" not in row["left_arm_id"]+row["right_arm_id"] for row in pairs)
    confounded=[row for row in pairs if row["confounded"]]
    assert len(confounded)==1 and confounded[0]["changed_factors"]=="guard,geometry"
    assert confounded[0]["left_arm_id"]=="guard7_disk_Z"
    assert review.component_pairs(list(reversed(_arms())))==pairs


@pytest.mark.parametrize("name,valid", [
    ("s0_a0.4_current_A", True), ("centered_direct_Z", True),
    ("", False), (".", False), ("..", False), ("../arm", False),
    ("arm/child", False), ("arm\\child", False), ("/arm", False),
])
def test_arm_identifier_allows_decimal_parameters_but_not_path_traversal(name,valid):
    assert review._valid_arm_id(name) is valid


def test_paired_counts_keep_gains_losses_and_false_strings_straight():
    left=[dict(observation_id=k,matched=k in "ab") for k in "abcd"]
    right=[dict(observation_id=k,matched=str(k in "bc")) for k in "abcd"]
    assert review.paired_counts(left,right)==dict(known_occurrence_count=4,both=1,gained=1,lost=1,neither=1,left_matched=2,right_matched=2,delta_matched=0)
    with pytest.raises(ValueError,match="same known occurrences"):review.paired_counts(left,right[:3])
    with pytest.raises(ValueError,match="Duplicate"):review.paired_counts(left+left[:1],right)


def _occurrence(matched=False):
    return dict(observation_id="o",canonical_roi_id="r",burst_id=1,source_start_ui=10,source_stop_ui=20,
                x_px=.25,y_px=0,matched=matched,matched_site_id="s" if matched else "")


def _candidate(x=5,frame=15):
    return dict(proposal_id="p",x_px=x,y_px=0,source_frame_ui=frame,score=2)


def _site(x=5):return dict(site_id="s",burst_id=1,x_px=x,y_px=0)


@pytest.mark.parametrize("category,matched,peak,candidates,sites,assignments",[
    ("matched",True,2,[_candidate()],[_site()],{"s":"o"}),
    ("no_score_exceedance",False,1,[],[],{}),
    ("no_retained_frame_candidate_within6",False,2,[_candidate(frame=21)],[],{}),
    ("no_representative_within6",False,2,[_candidate()],[_site(10)],{"s":"other"}),
    ("one_to_one_assignment_competition",False,2,[_candidate()],[_site()],{"s":"other"}),
])
def test_all_five_conditional_categories(category,matched,peak,candidates,sites,assignments):
    old="matched" if matched else ("no_local_exceedance" if peak<=1 else "exceedance_unmatched")
    row=review.classify_q1_occurrence(_occurrence(matched),dict(observation_id="o",local_peak=peak,threshold=1,margin=peak-1,category=old),candidates,sites,assignments)
    assert row["diagnostic_category"]==category
    assert row["onset_latency_identified"] is False
    assert row["unmatched_candidates"]=="unknown_not_negative"
    if category=="one_to_one_assignment_competition":assert json.loads(row["nearby_site_assigned_observations"])=={"s":"other"}


def test_attrition_rejects_unassigned_representative_and_stale_peak():
    diagnostic=dict(observation_id="o",local_peak=2,threshold=1,margin=1,category="exceedance_unmatched")
    with pytest.raises(ValueError,match="unassigned nearby"):
        review.classify_q1_occurrence(_occurrence(),diagnostic,[_candidate()],[_site()],{})
    with pytest.raises(ValueError,match="contradicts the local"):
        review.classify_q1_occurrence(_occurrence(),dict(diagnostic,local_peak=1,margin=0),[_candidate()],[_site()],{"s":"other"})


def test_complete_review_is_read_only_and_does_not_hash_dense_arrays(tmp_path,monkeypatch):
    dense=_complete_fixture(tmp_path)
    before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    original_sha=review._sha
    def restricted_sha(path):
        assert Path(path)!=dense,"postprocessing read a dense stage array"
        return original_sha(path)
    monkeypatch.setattr(review,"_sha",restricted_sha)
    result=review.run_review(tmp_path)
    assert result["status"]=="PASS"
    assert result["component_pair_count"]==48 and result["paired_operating_point_count"]==240
    assert result["q1_occurrence_count"]==2212
    assert result["saved_numeric_joins_recomputed_and_equal"]
    assert not result["dense_stage_arrays_rehashed"] and not result["predictions_changed"]
    for path,digest in before.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    paired=review._rows(tmp_path/"review/paired_components.tsv")
    assert all(row["neither"]=="79" and row["gained"]==row["lost"]=="0" for row in paired)
    counts=review._rows(tmp_path/"review/q1_attrition_counts.tsv")
    assert len(counts)==28 and all(row["no_score_exceedance"]=="79" for row in counts)
    assert review.run_review(tmp_path)==result


@pytest.mark.parametrize("damage,pattern",[("candidate","Changed sealed artifact"),("labels","Changed sealed artifact"),
    ("summary","numeric summary differs"),("join","occurrence_rows differs"),("online","Combined online summary"),
    ("diagnostic","local-peak margin"),("incomplete","All 28")])
def test_postprocessing_gate_rejects_mutated_or_incomplete_outputs(tmp_path,damage,pattern):
    _complete_fixture(tmp_path)
    arm=tmp_path/"arms"/_arms()[0]["arm_id"]
    if damage=="candidate":
        with (arm/"q1/candidates.tsv").open("a") as stream:stream.write("changed\n")
    elif damage=="labels":
        with (tmp_path/"expert_occurrences.tsv").open("a") as stream:stream.write("changed\n")
    elif damage=="summary":
        path=arm/"q1/summary.json";payload=review._json(path);payload["matched_known_positive_count"]=1;_json(path,payload)
    elif damage=="join":
        path=arm/"q1/occurrence_rows.tsv";rows=review._rows(path);rows[0]["matched"]="True";_rows(path,rows)
    elif damage=="online":
        path=tmp_path/"online_summary.tsv";rows=review._rows(path);rows[0]["matched_known_positive_count"]="1";_rows(path,rows)
    elif damage=="diagnostic":
        path=tmp_path/"q1_stage_diagnostics.tsv";rows=review._rows(path);rows[0]["margin"]="9";_rows(path,rows)
    elif damage=="incomplete":
        path=tmp_path/"evaluation_complete.json";payload=review._json(path);payload["status"]="RUNNING";_json(path,payload)
    with pytest.raises(ValueError,match=pattern):review.run_review(tmp_path)
    assert not (tmp_path/"review").exists()


def test_plot_groups_use_plain_labels_and_keep_all_fixed_curves():
    conditioning,operators=review.plot_groups(_arms())
    assert len(conditioning)==len(operators)==4
    assert all({label for _,label in curves}=={"Neither","Spatial only","Temporal only","Both"} for _,curves in conditioning)
    assert "Current brightness" in conditioning[0][0]
    assert "without local standardization" in conditioning[0][0]
    assert "Signed change" in conditioning[3][0] and "with Gamma-LS" in conditioning[3][0]
    assert [label for _,label in operators[1][1]]==["Centered target average","Target minus reference","Locally standardized"]
    assert "outer support" in operators[2][0]
    assert all(len(curves)==3 for _,curves in operators)
