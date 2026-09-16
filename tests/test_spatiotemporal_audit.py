"""Tiny truth/no-truth adapter fixtures, including actual lossless media."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import spatiotemporal_audit as audit


def _fixture(tmp_path, *, with_truth=True, with_candidates=True):
    raw = np.full((8,24,28),8.123456789,dtype=np.float64)
    score = np.zeros(raw.shape,dtype=np.float32);score[4,12,13] = np.float32(2.3456789)
    paths = {}
    for key in ("Raw","Input","A","Score"):
        path = tmp_path/f"{key}.npy";np.save(path,raw if key == "Raw" else score);paths[key] = path
    linked = tmp_path/"M.npy";np.save(linked,score)
    rows = [{"proposal_id":"p1","source_frame_ui":105,"x_px":13,"y_px":12,"score":float(score[4,12,13]),
             "threshold_z":1,"target_proposals_per_frame":.069}] if with_candidates else []
    candidates = tmp_path/"candidates.json";candidates.write_text(json.dumps(rows))
    experts = [{"canonical_roi_id":"synthetic_roi_01","observation_id":"event_01","burst_id":1,
                "source_start_ui":103,"source_stop_ui":106,"x_px":13,"y_px":12}] if with_truth else []
    truth = tmp_path/"experts.json";truth.write_text(json.dumps(experts))
    numeric = tmp_path/"all_thresholds.json";numeric.write_text('{"thresholds":[0,1,2],"scope":"fixture"}')
    return {"root":tmp_path/"audit","stage_paths":paths,"display_limits":{key:([0,10] if key == "Raw" else [-4,4]) for key in paths},
        "signed_stage_keys":["Input","A","Score"],"source_frames_ui":list(range(101,109)),
        "candidates_path":candidates,"expert_occurrences":truth,
        "operating_point":{"threshold":1,"threshold_z":1,"target_proposals_per_frame":.069,
                           "application_source_start_ui":103,"application_source_stop_ui":108},
        "source_binding":{"stage_sha256":{key:audit._digest(path) for key,path in paths.items()},
            "candidates_sha256":audit._digest(candidates),"expert_occurrences_sha256":audit._digest(truth),
            "truth_mode":"fully_synthetic","original_source_offset_xy":[30,40],
            "numeric_stage_bindings":{"M":{"path":str(linked),"sha256":audit._digest(linked),"size_bytes":linked.stat().st_size}}},
        "numeric_threshold_paths":{"all_thresholds":numeric},"context_frames":1}


def test_json_preserves_exact_scores_and_empty_truth_inventory(tmp_path):
    config = _fixture(tmp_path,with_truth=False)
    rows = audit._rows(config["candidates_path"])
    assert rows[0]["score"] == float(np.float32(2.3456789))
    plan = audit.audit_inventory_plan(config["candidates_path"],config["expert_occurrences"],config["source_frames_ui"])
    assert plan["expert_roi_count"] == plan["expert_occurrence_count"] == 0
    assert plan["model_roi_count"] == 1 and plan["expected_video_count"] == 3
    assert "N/A" in plan["expert_applicability"]
    assert audit._descriptive_evaluation(rows,[])["summary"]["descriptive_window_site_coverage"] is None


def test_overlapping_truth_windows_are_descriptive_not_primary_assignments():
    candidates = [{"proposal_id":"p1","source_frame_ui":7,"x_px":10,"y_px":10,"score":4}]
    experts = [{"canonical_roi_id":"roi1","observation_id":"e1","burst_id":1,"source_start_ui":4,"source_stop_ui":9,"x_px":10,"y_px":10},
               {"canonical_roi_id":"roi1","observation_id":"e2","burst_id":2,"source_start_ui":6,"source_stop_ui":12,"x_px":10,"y_px":10}]
    result = audit._descriptive_evaluation(candidates,experts)
    assert len(result["occurrence_rows"]) == 2
    assert result["summary"]["emitted_frame_proposal_count"] == 1
    assert result["summary"]["descriptive_window_site_matched_occurrence_count"] == 2
    assert not result["summary"]["primary_classification_metric"]
    assert len({row["site_id"] for row in result["site_rows"]}) == 2


@pytest.mark.parametrize("field,value,match",[("width",719,"canvas"),("height",166,"canvas"),
    ("r_frame_rate","50/1","frame rate"),("duration","0.31","duration"),("nb_frames","4","frame count")])
def test_probe_geometry_and_timing_reject_tampering(field,value,match):
    record = {"path":"/audit/videos/model_sequential_full_field.mp4","source_frames_ui":[101,106,108],
        "probe":{"width":720,"height":184,"r_frame_rate":"10/1","duration":"0.300000","nb_frames":"3"}}
    audit._validate_video_geometry(record,stage_count=4,source_fps=50,fullfield_step=5)
    record["probe"][field] = value
    with pytest.raises(ValueError,match=match):audit._validate_video_geometry(record,stage_count=4,source_fps=50,fullfield_step=5)


@pytest.mark.parametrize("stage_count",[3,4])
@pytest.mark.parametrize("long_title",[False,True])
def test_trace_text_is_inside_canvas_without_title_axis_overlap(tmp_path,stage_count,long_title):
    keys = ("Raw","Input","Score") if stage_count == 3 else ("Raw","Input","A","Score")
    values = {key:np.sin(np.arange(464)/17)+index for index,key in enumerate(keys)}
    title = "Synthetic source truth | ROI01\nExact pixel (60,64); all464 source samples"
    if long_title:
        title = ("Synthetic source truth | "+"crowded__seed20260914__event02__"*7+" | window/site matched=False\n"
                 "UI180–435 (window185–430, context±5); nearest burst_1__site_00001; assigned unavailable")
    audit._trace_plot(tmp_path/"trace.png",values,list(range(1,465)),title=title,spans=[(185,430)],color="#46dc7d",threshold=3.7)
    figure,axes,_,title_artist = audit._TRACE_CANVASES[tuple(keys)]
    old_dpi = figure.dpi
    try:
        figure.set_dpi(105);figure.canvas.draw();renderer = figure.canvas.get_renderer()
        canvas = figure.bbox
        artists = [title_artist]
        for axis in axes:
            artists += [axis.xaxis.label,axis.yaxis.label,axis.xaxis.get_offset_text(),axis.yaxis.get_offset_text()]
            # Locators keep extra ticks outside the data limits; Axis.draw does
            # not paint those, even though the cached Text says visible=True.
            for dimension,limits in ((axis.xaxis,axis.get_xlim()),(axis.yaxis,axis.get_ylim())):
                for tick in dimension.get_major_ticks():
                    if min(limits) <= tick.get_loc() <= max(limits):artists += [tick.label1,tick.label2]
            if axis.get_legend():artists += axis.get_legend().get_texts()
        for artist in artists:
            if not artist.get_visible() or not artist.get_text():continue
            bounds = artist.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.y0 >= 0,artist.get_text()
            assert bounds.x1 <= canvas.width and bounds.y1 <= canvas.height,artist.get_text()
        assert title_artist.get_window_extent(renderer).y0 >= axes[0].get_window_extent(renderer).y1+5
        assert "".join(title_artist.get_text().split()) == "".join(title.split())
    finally:figure.set_dpi(old_dpi)


@pytest.mark.parametrize("damage,match",[("wrong_target","Candidate target"),("nonfinite_threshold","finite frozen threshold"),
    ("missing_truth_mode","truth_mode"),("changed_linked_stage","Numeric linked stage"),("missing_numeric","descriptive string keys")])
def test_bad_contract_never_reaches_video(tmp_path,monkeypatch,damage,match):
    config = _fixture(tmp_path)
    if damage == "wrong_target":config["operating_point"]["target_proposals_per_frame"] = .5
    elif damage == "nonfinite_threshold":config["operating_point"]["threshold"] = float("nan")
    elif damage == "missing_truth_mode":config["source_binding"].pop("truth_mode")
    elif damage == "changed_linked_stage":Path(config["source_binding"]["numeric_stage_bindings"]["M"]["path"]).write_bytes(b"changed")
    elif damage == "missing_numeric":config["numeric_threshold_paths"] = {}
    monkeypatch.setattr(audit,"_write_video",lambda *args,**kwargs:pytest.fail("invalid contract reached media"))
    with pytest.raises(ValueError,match=match):audit.run_spatiotemporal_audit(**config)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),reason="FFmpeg required")
@pytest.mark.parametrize("with_truth,with_candidates",[(True,True),(False,True),(False,False)])
def test_complete_small_media_and_bound_resume(tmp_path,with_truth,with_candidates):
    config = _fixture(tmp_path,with_truth=with_truth,with_candidates=with_candidates)
    result = audit.run_spatiotemporal_audit(**config)
    root = config["root"]
    assert result["scientific_audit_complete"]
    assert result["video_count"] == 2+int(with_truth)+int(with_candidates)
    assert result["expert_occurrence_count"] == int(with_truth)
    assert result["model_roi_count"] == int(with_candidates)
    assert result["truth_semantics"]["truth_mode"] == "fully_synthetic"
    coverage = json.loads((root/"coverage_manifest.json").read_text())
    assert "q_coverage" not in coverage
    assert set(coverage["numeric_threshold_evidence"]) == {"all_thresholds"}
    videos = json.loads((root/"video_manifest.json").read_text())["videos"]
    assert all(video["all_frames_rgb_byte_exact"] and video["all_frames_source_palette_pure"] for video in videos)
    assert all(video["probe"]["pix_fmt"] == "gbrp" for video in videos)
    assert all(video["caption_footer_height_px"] == 18 for video in videos)
    fullfields = [video for video in videos if "full_field" in video["path"]]
    assert all(video["probe"]["width"] == 720 and video["probe"]["height"] == 184 for video in fullfields)
    assert all(video["source_frames_ui"] == [101,106,108] for video in fullfields)
    if not with_truth:
        expert = next(video for video in videos if "/1_Expert_Annotations/" in video["path"])
        assert "Expert N/A" in expert["truth_caption"]
        overview = json.loads((root/"3_Comparison/overview_metadata.json").read_text())
        assert overview["source_frames_ui"] == list(range(103,109))
        assert not list((root/"1_Expert_Annotations/metadata").glob("*.json"))
    else:
        rows = list(csv.DictReader((root/"1_Expert_Annotations/exact_pixel_traces/synthetic_roi_01.csv").open()))
        assert len(rows) == 8 and float(rows[0]["Raw"]) == 8.123456789
        metadata = json.loads((root/"1_Expert_Annotations/metadata/synthetic_roi_01.json").read_text())
        assert metadata["original_source_offset_xy"] == [30,40]
    assert audit.run_spatiotemporal_audit(**config) == result
    linked = Path(config["source_binding"]["numeric_stage_bindings"]["M"]["path"])
    linked.write_bytes(b"changed after complete")
    with pytest.raises(ValueError,match="size mismatch|SHA-256 mismatch"):
        audit.run_spatiotemporal_audit(**config)
