"""Bounded fixtures; no recording/movie is loaded by these tests."""
import json

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import spatiotemporal_real_review as review


def _protocol(tmp_path):
    raw = tmp_path / "source.bin"
    raw.write_bytes(b"source binding fixture only; prepare must not load movie pixels")
    labels = tmp_path / "labels.tsv"
    rows = [{"observation_id":f"obs_{index:03d}","canonical_roi_id":f"roi_{index%26:03d}",
             "x_px":100+index%40,"y_px":120+index%8} for index in range(79)]
    review._table(labels,rows,list(rows[0]))
    protocol = tmp_path / "protocol.json"
    review._json(protocol,{"source":review._binding(raw),"labels":review._binding(labels)})
    return protocol


def test_geometry_and_exact_source_frame_map():
    panels = review.panel_geometry()
    assert [(p["x0"],p["x1"],p["y0"],p["y1"]) for p in panels] == [(94,222,106,234),(222,350,106,234),(350,478,106,234)]
    assert min(p["minimum_available_spatial_halo_px"] for p in panels) == 94
    rows = review.frame_grid(panels,(1800,2359))
    assert len(rows) == 1680
    assert rows[0]["source_numpy_index"] == 1799
    assert rows[559]["source_frame_ui"] == 2359
    assert rows[559]["evaluation_array_row"] == 559
    with pytest.raises(ValueError):review.panel_geometry(width=380)


def test_preparation_preserves_all_labels_and_pending_state(tmp_path):
    protocol = _protocol(tmp_path)
    output = tmp_path / "review"
    manifest = review.prepare_review_package(protocol,output)
    assert manifest["known_occurrence_count"] == 79
    assert manifest["known_identity_count"] == 26
    assert len(review._rows(output/"known_occurrence_panels.tsv")) == 79
    assert sum(len(row["observation_ids"]) for row in review._read(output/"known_center_locations.json")) == 79
    assert manifest["contract"]["descriptive_calibration_ui"] == [1,1799]
    assert manifest["contract"]["detector_setup_ui"] == [1600,1799]
    assert manifest["contract"]["physical"]["reported_bandwidth_exceeds_nyquist"]
    assert not review.validate_annotation_coverage(output)["real_precision_eligible"]
    assert review.prepare_review_package(protocol,output) == manifest
    with (output/"frame_grid.tsv").open("a") as handle:handle.write("tamper\n")
    with pytest.raises(ValueError,match="Bound source changed"):review.prepare_review_package(protocol,output)


def _annotation_fixture(tmp_path, *, uncertainty=False):
    panel = {"panel_id":"panel_01","x0":10,"x1":14,"y0":20,"y1":24,"width":4,"height":4}
    grid = review.frame_grid([panel],(1800,1801))
    review._table(tmp_path/"frame_grid.tsv",grid,list(grid[0]))
    reviews = [{"panel_id":"panel_01","source_frame_ui":frame,"status":"complete",
        "source_inventory_complete":True,"active_regions_complete":True,"uncertainty_explicit":True,
        "reviewer_id":"fixture-human","reviewed_at":"2026-09-14T00:00:00Z","notes":""} for frame in (1800,1801)]
    review._table(tmp_path/"frame_reviews.tsv",reviews,review.FRAME_REVIEW_FIELDS)
    for name,fields in (("source_inventory.tsv",review.SOURCE_FIELDS),("active_regions.tsv",review.ACTIVE_FIELDS),
                        ("uncertain_areas.tsv",review.UNCERTAIN_FIELDS),("fluorescence_events.tsv",review.EVENT_FIELDS)):
        review._table(tmp_path/name,[],fields)
    review._table(tmp_path/"panel_characterization.tsv",[{"panel_id":"panel_01","notes":"quiet fixture"}],("panel_id","notes"))
    if uncertainty:
        rows = [{"panel_id":"panel_01","source_frame_ui":frame,"uncertainty_id":"whole_panel",
            "polygon_xy_json":json.dumps([[10,20],[14,20],[14,24],[10,24]]),"reason":"unresolvable","reviewer_id":"fixture-human"} for frame in (1800,1801)]
        review._table(tmp_path/"uncertain_areas.tsv",rows,review.UNCERTAIN_FIELDS)
    review._json(tmp_path/"manifest.json",{"contract":{"panels":[panel],"evaluation_ui":[1800,1801]},
        "immutable_artifacts":[review._binding(tmp_path/"frame_grid.tsv")]})
    acceptance = {"accepted":True,"accepted_by":"fixture-human","accepted_at":"2026-09-14T00:00:00Z",
        "complete_source_inventory":True,"complete_active_region_coverage":True,"uncertainty_explicit":True,
        "frame_grid_sha256":review._sha(tmp_path/"frame_grid.tsv"),
        "annotation_file_sha256":{name:review._sha(tmp_path/name) for name in review.MUTABLE_REVIEW_FILES}}
    review._json(tmp_path/"annotation_acceptance.json",acceptance)


def test_exhaustive_acceptance_is_hash_bound_and_does_not_compute_precision(tmp_path):
    _annotation_fixture(tmp_path)
    result = review.validate_annotation_coverage(tmp_path)
    assert result["accepted_exhaustive_coverage"] and result["real_precision_eligible"]
    assert result["real_precision"] is None
    assert result["evaluable_pixel_frames"] == 32
    with (tmp_path/"frame_reviews.tsv").open("a") as handle:handle.write("\n")
    with pytest.raises(ValueError,match="hashes differ"):review.validate_annotation_coverage(tmp_path)


def test_all_uncertain_pixels_cannot_become_negative_material(tmp_path):
    _annotation_fixture(tmp_path,uncertainty=True)
    result = review.validate_annotation_coverage(tmp_path)
    assert result["uncertain_excluded_pixel_frames"] == 32
    assert result["evaluable_pixel_frames"] == 0
    assert not result["real_precision_eligible"]


def test_duplicate_frame_coverage_rejected(tmp_path):
    _annotation_fixture(tmp_path)
    path = tmp_path/"frame_reviews.tsv"
    rows = review._rows(path);rows.append(rows[0])
    review._table(path,rows,review.FRAME_REVIEW_FIELDS)
    with pytest.raises(ValueError,match="exactly once"):review.validate_annotation_coverage(tmp_path)


def test_missing_activity_does_not_create_indicator_fit():
    result = review.fit_isolated_transients(np.ones(300),source_start_ui=1)
    assert result["status"] == "unidentified" and result["events"] == []
    assert not result["indicator_impulse_response_identified"]


@pytest.mark.parametrize("kind",["rise","decay"])
def test_sampled_exponential_profile_recovers_resolved_fixture(kind):
    times = np.arange(80)*.02
    values = 4*(1-np.exp(-times/.2)) if kind == "rise" else 4*np.exp(-times/.2)
    fit = review._exponential_fit(values,.02,kind)
    assert fit["status"] == "descriptive_fit"
    assert fit["tau_ms"] == pytest.approx(200,rel=.04)
    assert "not_indicator_irf" in fit["reason"]


def test_abrupt_rise_and_right_censoring_are_explicit():
    trace = .01*np.sin(np.arange(300)*1.7)
    trace[100:140] += 2*np.exp(-np.arange(40)/7)
    result = review.fit_isolated_transients(trace,source_start_ui=101)
    event = max(result["events"],key=lambda row:row["peak_above_baseline"])
    assert event["peak_frame_ui"] == 201
    assert event["fast_rise_unresolved"] is True
    assert event["rise_fit"]["tau_ms"] is None
    trace = .01*np.sin(np.arange(300)*1.7);trace[290:] += 2
    result = review.fit_isolated_transients(trace,source_start_ui=1)
    event = max(result["events"],key=lambda row:row["peak_above_baseline"])
    assert event["right_censored"]
    assert event["decay_fit"]["tau_ms"] is None


def test_radial_width_is_descriptive_and_static_profile_has_no_activity_width():
    yy,xx = np.indices((25,25));image = np.exp(-((xx-12)**2+(yy-12)**2)/(2*3**2))
    profile = review.radial_profile(image,center_x=12,center_y=12,pixel_size_um=.5)
    assert profile["status"] == "described"
    assert profile["fwhm_um"] == pytest.approx(2.35482*3*.5,rel=.08)
    static = review.radial_profile(image,center_x=12,center_y=12,pixel_size_um=.5,activity_image=False)
    assert static["fwhm_um"] is None
