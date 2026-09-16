"""Coverage, deterministic site grouping, and tiny complete media audit checks."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil

import pytest

from neurobench.experiments.gamma_ls_difference.two_stencil_audit import (
    _artifact_cached,
    _checkpoint_artifact,
    _canonical_digest,
    _panel,
    _source_palette,
    _validate_lossless_video,
    _write_video,
    audit_inventory_plan,
    context_source_frames,
    identity_crop,
    panel_image_geometry,
    panel_marker_position,
    run_two_stencil_audit,
    spatial_review_sites,
    verify_completed_audit,
)


def proposal(identifier: str, x: int, y: int, frame: int, score: float) -> dict:
    return {"proposal_id": identifier, "x_px": x, "y_px": y,
            "source_frame_ui": frame, "score": score, "threshold_z": 1.0,
            "target_proposals_per_frame": 1.0}


def expert(identifier: str, occurrence: str, burst: int, start: int, stop: int, x: float = 20, y: float = 20) -> dict:
    return {"canonical_roi_id": identifier, "observation_id": occurrence,
            "burst_id": burst, "source_start_ui": start, "source_stop_ui": stop,
            "x_px": x, "y_px": y,
            "temporal_extent_semantics": "configured_burst_window_not_per_roi_onset"}


def test_sites_are_nontransitive_actual_representatives_and_permutation_stable():
    rows = [proposal("a", 0, 20, 1, 5), proposal("b", 5, 20, 2, 4),
            proposal("c", 10, 20, 3, 3), proposal("d", 5, 20, 4, 2)]
    sites, assignments = spatial_review_sites(rows)
    assert [site["representative_proposal_id"] for site in sites] == ["a", "c"]
    assert sites[0]["member_proposal_ids"] == ["a", "b", "d"]
    assert len(assignments) == len(rows)
    assert spatial_review_sites(list(reversed(rows))) == (sites, assignments)
    assert all("not_biological_identity" in site["interpretation"] for site in sites)


def test_inventory_covers_every_occurrence_and_every_model_site_without_sampling():
    candidates = [proposal("a", 20, 20, 7, 5), proposal("b", 22, 20, 18, 4),
                  proposal("c", 35, 20, 20, 3)]
    labels = [expert("roi_001", "b1_roi1", 1, 6, 8), expert("roi_001", "b2_roi1", 2, 18, 20),
              expert("roi_002", "b1_roi2", 1, 6, 8, 35, 20)]
    plan = audit_inventory_plan(candidates, labels, range(1, 24))
    assert plan["expert_roi_count"] == 2
    assert plan["expert_occurrence_count"] == 3
    assert plan["model_roi_count"] == 2
    assert plan["all_model_proposal_count"] == 3
    assert plan["expected_video_count"] == 6
    assert plan["fullfield_source_frames_ui"] == [1, 6, 11, 16, 21, 23]
    assert set(range(6, 9)) | set(range(18, 21)) <= set(plan["expert_closeup_source_frames_ui"]["roi_001"])
    assert {7, 18} <= set(plan["model_closeup_source_frames_ui"]["review_site_00001"])
    assert plan["no_model_anchor_sampling"]


def test_context_frame_union_preserves_source_gaps_and_does_not_duplicate_frames():
    assert context_source_frames([(4, 5), (5, 7), (15, 16)], list(range(1, 20)), 1) == [3, 4, 5, 6, 7, 8, 14, 15, 16, 17]


@pytest.mark.parametrize("source_wh,panel_wh,fitted_wh,padding", [
    ((573, 340), (180, 136), [180, 107], [0, 14, 0, 15]),
    ((48, 48), (144, 112), [112, 112], [16, 0, 16, 0]),
    ((49, 49), (144, 112), [112, 112], [16, 0, 16, 0]),
    ((340, 573), (180, 136), [81, 136], [49, 0, 50, 0]),
])
def test_panel_fit_preserves_complete_extent_and_marker_pixel_centers(source_wh, panel_wh, fitted_wh, padding):
    geometry = panel_image_geometry(*source_wh, *panel_wh)
    assert geometry["fitted_size_wh"] == fitted_wh
    assert geometry["padding_left_top_right_bottom"] == padding
    assert not geometry["source_region_cropped_during_fit"]
    for size, fitted in zip(source_wh, fitted_wh):
        assert abs(size * geometry["nominal_uniform_scale"] - fitted) <= .5
    crop = (10, 20, 10 + source_wh[0], 20 + source_wh[1])
    center = panel_marker_position(10 + (source_wh[0] - 1) / 2,
                                   20 + (source_wh[1] - 1) / 2, crop, geometry)
    assert center == pytest.approx((padding[0] + (fitted_wh[0] - 1) / 2,
                                    padding[1] + (fitted_wh[1] - 1) / 2))
    assert panel_marker_position(crop[0] - .5, crop[1] - .5, crop, geometry) == pytest.approx(
        (padding[0] - .5, padding[1] - .5)
    )
    assert panel_marker_position(crop[2] - .5, crop[3] - .5, crop, geometry) == pytest.approx(
        (padding[0] + fitted_wh[0] - .5, padding[1] + fitted_wh[1] - .5)
    )


def test_panel_raster_letterboxes_square_and_places_marker_in_fitted_image():
    import numpy as np
    frame = np.ones((49, 49), dtype=np.float32)
    kwargs = dict(source_ui=2, source_fps=50, color=(70, 220, 125),
                  panel_width=144, panel_height=112)
    image = _panel({"Raw": frame}, {"Raw": [0, 1]}, markers=[], **kwargs)
    assert image.shape == (142, 576, 3)
    assert np.all(image[30:, :16] == 0)
    assert np.all(image[30:, 128:144] == 0)
    assert np.all(image[30:, 16:128] == 255)
    marked = _panel({"Raw": frame}, {"Raw": [0, 1]},
                    markers=[{"x_px": 24, "y_px": 24}], **kwargs)
    y, x = np.nonzero(np.all(marked == (70, 220, 125), axis=2))
    assert ((x.min() + x.max()) / 2, (y.min() + y.max()) / 2) == pytest.approx((71.5, 85.5), abs=1)


def test_inventory_rejects_partial_labels_and_supports_zero_candidate_state():
    labels = [expert("roi_001", "b1_roi1", 1, 5, 8)]
    with pytest.raises(ValueError, match="fully covered"):
        audit_inventory_plan([], labels, range(1, 7))
    plan = audit_inventory_plan([], labels, range(1, 12))
    assert plan["model_roi_count"] == 0
    assert plan["expected_video_count"] == 3
    assert plan["all_occurrences_covered"]


def test_identity_coordinate_variants_keep_actual_anchor_all_coordinates_and_crop():
    labels = [expert("roi_010", "b2_roi10", 2, 15, 17, 23, 21),
              expert("roi_010", "b1_roi15", 1, 4, 8, 22, 20),
              expert("roi_010", "b1_roi10", 1, 4, 8, 20, 20)]
    plan = audit_inventory_plan([], labels, range(1, 21))
    location = plan["unique_experts"]["roi_010"]
    assert plan["expert_roi_count"] == 1
    assert plan["expert_occurrence_count"] == 3
    assert location["trace_anchor_observation_id"] == "b1_roi10"
    assert (location["x_px"], location["y_px"]) == (20, 20)
    assert len(location["coordinate_variants"]) == 3
    assert location == audit_inventory_plan([], list(reversed(labels)), range(1, 21))["unique_experts"]["roi_010"]
    assert identity_crop(location, 100, 100, radius=5) == (15, 15, 29, 27)


def test_completed_audit_checks_external_source_hashes(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("original")
    output = tmp_path / "audit"
    output.mkdir()
    manifest = output / "source_manifest.json"
    manifest.write_text(json.dumps({"sources": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}))
    index = output / "artifact_index.json"
    index.write_text(json.dumps({"artifacts": [{"path": "source_manifest.json", "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}]}))
    verify_completed_audit(output)
    source.write_text("changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_completed_audit(output)


def test_figure_checkpoint_reuses_only_identical_bytes(tmp_path: Path):
    artifact = tmp_path / "trace.csv"
    artifact.write_text("source_frame_ui,Z\n1,0.0\n")
    checkpoint = {}
    checkpoint_path = tmp_path / "figure_checkpoint.json"
    assert not _artifact_cached(artifact, checkpoint)
    _checkpoint_artifact(artifact, checkpoint, checkpoint_path)
    assert _artifact_cached(artifact, checkpoint)
    restored = json.loads(checkpoint_path.read_text())
    assert _artifact_cached(artifact, restored)
    artifact.write_text("source_frame_ui,Z\n1,9.9\n")
    assert not _artifact_cached(artifact, restored)


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_cached_video_repairs_thumbnail_without_reencoding(tmp_path: Path, monkeypatch, damage):
    import numpy as np
    from PIL import Image
    import neurobench.experiments.gamma_ls_difference.two_stencil_audit as module
    monkeypatch.setattr(module, "_load_deps", lambda: (np, Image, None, None))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("cached video was reencoded"))
    video = tmp_path / "cached.mp4"
    video.write_bytes(b"previously verified encoded video fixture")
    thumbnail = video.with_suffix(".png")
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    Image.fromarray(image).save(thumbnail)
    thumbnail_hash = hashlib.sha256(thumbnail.read_bytes()).hexdigest()
    record = {"path": str(video), "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
              "source_frames_sha256": _canonical_digest([1, 2]), "thumbnail": str(thumbnail),
              "thumbnail_sha256": thumbnail_hash, "encoding_revision": "encoding_v2",
              "all_frames_rgb_byte_exact": True}
    checkpoint = {str(video): record}
    if damage == "missing":
        thumbnail.unlink()
    else:
        Image.fromarray(np.zeros_like(image)).save(thumbnail)
    rendered = []
    def frame_factory(frame):
        rendered.append(frame)
        return image
    video_mtime = video.stat().st_mtime_ns
    kwargs = dict(fps=50, marker_sample_index=1, annotation_section="expert",
                  checkpoint=checkpoint, checkpoint_path=tmp_path / "checkpoint.json")
    repaired = _write_video(video, [1, 2], frame_factory, **kwargs)
    assert rendered == [2]
    assert video.stat().st_mtime_ns == video_mtime
    assert repaired["thumbnail_sha256"] == hashlib.sha256(thumbnail.read_bytes()).hexdigest() == thumbnail_hash
    assert json.loads((tmp_path / "checkpoint.json").read_text())[str(video)]["thumbnail_sha256"] == thumbnail_hash
    assert _write_video(video, [1, 2], frame_factory, **kwargs) == repaired
    assert rendered == [2]  # intact thumbnail is reused too


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required for media validation")
def test_lossless_rgb_preserves_orange_on_noisy_gray_and_checks_every_frame(tmp_path: Path):
    import numpy as np
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(10092026)
    frames = []
    for index in range(8):
        gray = rng.integers(0, 256, (96, 128), dtype=np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        image = Image.fromarray(rgb)
        if index != 3:
            ImageDraw.Draw(image).ellipse((60 + index, 44, 68 + index, 52), outline=(255, 145, 35), width=2)
        frames.append(np.asarray(image))
    path = tmp_path / "noisy_orange.mp4"
    source_ui = list(range(101, 109))
    result = _write_video(path, source_ui, lambda ui: frames[ui - 101], fps=50,
                          marker_sample_index=0, annotation_section="model",
                          checkpoint={}, checkpoint_path=tmp_path / "checkpoint.json")
    assert result["all_frames_rgb_byte_exact"]
    assert result["all_frames_source_palette_pure"] and result["all_frames_decoded_palette_pure"]
    assert result["all_frames_marker_expectations_pass"]
    assert result["decoded_frame_count"] == 8
    assert result["probe"]["profile"] == "High 4:4:4 Predictive"
    assert result["probe"]["pix_fmt"] == "gbrp"
    assert result["source_rgb_stream_sha256"] == result["decoded_rgb_stream_sha256"]
    assert result["source_palette_summary"]["frames_with_own_marker"] == 7
    assert result["source_palette_summary"]["minimum_own_marker_pixels_per_frame"] == 0
    assert result["source_palette_summary"]["checked_frame_count"] == 8
    assert "source_rgb_frames" not in result  # per-frame hashes remain bounded in RAM
    assert result["source_rgb_stream_sha256"] == hashlib.sha256(b"".join(frame.tobytes() for frame in frames)).hexdigest()
    changed_source_record = [{"source_frame_ui": ui, "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                              **_source_palette(frame, "model")} for ui, frame in zip(source_ui, frames)]
    changed_source_record[6]["rgb_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source UI 107"):
        _validate_lossless_video(path, width=128, height=96,
                                 source_frames=changed_source_record, annotation_section="model")


def test_source_palette_rejects_real_green_overlay_in_model_frame():
    import numpy as np
    image = np.full((20, 20, 3), 127, dtype=np.uint8)
    image[5:9, 5:9] = [70, 220, 125]
    with pytest.raises(ValueError, match="Source palette violation"):
        _source_palette(image, "model")
    assert _source_palette(image, "expert")["own_marker_pixels"] == 16
    image[5, 5] = [110, 171, 132]  # even a compression-like nondeclared source hue fails
    with pytest.raises(ValueError, match="Source palette violation"):
        _source_palette(image, "expert")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required for media validation")
def test_tiny_complete_audit_preserves_every_frame_in_traces_and_resumes(tmp_path: Path):
    import numpy as np
    frames = list(range(1, 13))
    raw = np.ones((12, 40, 48), dtype=np.float32) * 4
    raw[3:8, 19:22, 19:22] += np.arange(5, dtype=np.float32)[:, None, None]
    z = np.zeros_like(raw)
    z[3:8, 20, 20] = [0, 2, 3, 2, 0]
    paths = {}
    for name in ("Raw", "X", "A", "M", "sigma", "contrast", "Z"):
        array = raw if name in {"Raw", "A", "M"} else (np.ones_like(raw) if name == "sigma" else z)
        paths[name] = tmp_path / f"{name}.npy"
        np.save(paths[name], array)
    labels = [expert("roi_001", "b1_roi1", 1, 4, 8)]
    candidates = [proposal("a", 20, 20, 6, 3), proposal("b", 21, 20, 7, 2)]
    numeric_root = tmp_path / "numeric_q1"
    numeric_root.mkdir()
    (numeric_root / "summary.json").write_text(json.dumps({"target_proposals_per_frame": 1.0}))
    kwargs = dict(stage_paths=paths, signed_stage_keys=["X", "contrast", "Z"],
                  display_limits={name: ([-5, 5] if name in {"X", "contrast", "Z"} else [0, 10]) for name in paths},
                  source_frames_ui=frames, source_binding={"cell_id": "tiny_fixture", "stage_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}},
                  operating_point={"target_proposals_per_frame": 1.0, "threshold_z": 1.0},
                  fullnumeric_q_paths={"1": numeric_root}, candidate_rows=candidates,
                  expert_occurrences=labels, context_frames=1)
    output = tmp_path / "audit"
    result = run_two_stencil_audit(output, **kwargs)
    assert result["scientific_audit_complete"]
    assert result["expert_occurrence_count"] == 1
    assert result["model_roi_count"] == 1
    assert result["video_count"] == 4
    figure_checkpoint = json.loads((output / "figure_checkpoint.json").read_text())
    assert len(figure_checkpoint) == 8  # two ROI CSV/PNG/JSON triplets + comparison PNG/JSON
    videos = json.loads((output / "video_manifest.json").read_text())["videos"]
    assert all(video["full_decode_pass"] and video["encoded_marker_separation_pass"] for video in videos)
    assert all(video["all_frames_rgb_byte_exact"] and video["all_frames_marker_expectations_pass"]
               and video["source_palette_summary"]["checked_frame_count"] == video["frame_count"] for video in videos)
    fullfield = [video for video in videos if "full_field" in video["path"]]
    assert all(video["source_frames_ui"] == [1, 6, 11, 12] for video in fullfield)
    assert all(video["panel_image_geometry"]["fitted_size_wh"] == [163, 136]
               and not video["panel_image_geometry"]["source_region_cropped_during_fit"]
               for video in fullfield)
    roi_metadata = json.loads((output / "1_Expert_Annotations/metadata/roi_001.json").read_text())
    assert "panel_image_geometry" in roi_metadata
    trace = output / "1_Expert_Annotations/exact_pixel_traces/roi_001.csv"
    assert len(trace.read_text().splitlines()) == 13
    comparison = json.loads((output / "3_Comparison/trace_comparisons/b1_roi1.json").read_text())
    assert comparison["comparison_plot_source_frames_ui"] == list(range(3, 10))
    assert comparison["correlation_and_lag_window_ui_inclusive"] == [4, 8]
    assert comparison["comparison_plot_start_frame_ui"] == 3
    assert comparison["comparison_plot_stop_frame_ui"] == 9
    original_time = (output / "status.json").stat().st_mtime_ns
    assert run_two_stencil_audit(output, **kwargs) == result
    assert (output / "status.json").stat().st_mtime_ns == original_time
