import json
from pathlib import Path

import numpy as np
import tifffile

from neurobench.experiments.pairwise_separation.config import PairwiseSeparationConfig
from neurobench.experiments.pairwise_separation.artifacts import write_lane
from neurobench.experiments.pairwise_separation.preflight import preflight
from neurobench.experiments.pairwise_separation.runner import run


ROOT=Path(__file__).resolve().parents[1]


def test_unresolved_ica_lane_explicitly_omits_activity_and_masks(tmp_path):
    lane = {
        "continuous_components": None,
        "binary_mask": None,
        "fit": {
            "status": "unresolved_component",
            "artifact_omissions": [
                "continuous_activity.npy", "binary_mask.npy", "binary_mask.tif"
            ],
        },
        "diagnostics": {"component_selection": "unresolved"},
        "timing": {"fit_and_apply_seconds": 0.0},
    }
    write_lane(tmp_path, "ica_unresolved", lane, write_tiff=True)
    target = tmp_path / "methods" / "ica_unresolved"
    fit = json.loads((target / "fit.json").read_text())
    assert fit["artifact_omissions"] == [
        "continuous_activity.npy", "binary_mask.npy", "binary_mask.tif"
    ]
    assert (target / "parameters.npz").is_file()
    assert not (target / "continuous_activity.npy").exists()
    assert not (target / "binary_mask.npy").exists()
    assert not (target / "binary_mask.tif").exists()


def test_tiny_runner_writes_all_resolved_artifacts_without_partials(tmp_path):
    rng=np.random.default_rng(9); video=rng.normal(100,2,size=(90,20,22)).astype(np.float32)
    video[65:72,8:12,9:13]+=np.linspace(0,80,7)[:,None,None]
    video=np.clip(video,0,4095).astype(np.uint16)
    np.save(tmp_path/"source.npy",video); tifffile.imwrite(tmp_path/"source.tif",video,photometric="minisblack")
    header="burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\tstop_frame_zero_exclusive\tpoint_index\troi_identity\tx_px\ty_px\trecurrence_count\n"
    (tmp_path/"labels.tsv").write_text(header+"1\t66\t72\t65\t72\t1\tr1\t11\t10\t1\n")
    (tmp_path/"design.md").write_text("# tiny\n")
    raw=json.loads((ROOT/"examples/spon_ca_burst_pairwise_separation.example.json").read_text())
    raw.update({"experiment_id":"tiny","source_video":"source.npy","source_tiff":"source.tif","labels_tsv":"labels.tsv","design_document":"design.md","output_dir":"run"})
    raw["frames"]={"review_start_ui":1,"review_end_ui":90,"quiet_start_ui":1,"quiet_end_ui":60,"frame_period_ms":20}
    raw["sampling"].update({"screen_samples":16,"confirm_samples":32,"screen_angle_step_degrees":30,"refine_half_width_degrees":1,"refine_angle_step_degrees":1,"pairwise_diagnostic_frames_ui":[20,40,60]})
    raw["methods"]["infomax_tanh_ica"].update({"max_iterations":10,"initial_angles_degrees":[0,45]})
    raw["methods"]["cs_parzen_ica"].update({"kernel_block_rows":16})
    raw["methods"]["shared_background_nmf"].update({"max_iterations":10})
    raw["thresholding"]["write_binary_tiff"]=True
    raw["evaluation"].update({"nms_distance_px":2,"match_radii_px":[2,4,6],"candidate_review_rows":30})
    raw["resources"].update({"cpu_threads":1,"frame_chunk":8,"kernel_block_rows":16,"max_ram_mib":1,"min_free_disk_mib":1,"max_output_mib":20})
    path=tmp_path/"config.json"; path.write_text(json.dumps(raw)); config=PairwiseSeparationConfig.load(path)
    preflight(config,artifact_dir=tmp_path/"preflight")
    summary=run(config,preflight_dir=tmp_path/"preflight")
    assert summary["implementation_status"]=="complete"
    assert set(summary["methods"])=={"fixed_binary_difference","adaptive_binary_difference","infomax_tanh_ica","cs_parzen_ica","shared_background_nmf"}
    assert (config.output_dir/"metrics.json").is_file() and (config.output_dir/"report.md").is_file()
    assert not list(config.output_dir.rglob("*.partial"))
    for figure in ("method_comparison_montage.png", "objective_by_angle.png", "recall_candidate_tradeoff.png", "quiet_burst_occupancy.png"):
        assert (config.output_dir/"figures"/figure).is_file()
    for method in summary["methods"]:
        fit=json.loads((config.output_dir/"methods"/method/"fit.json").read_text())
        assert fit["method_id"] == method
        assert (config.output_dir/"methods"/method/"parameters.npz").is_file()
        if fit["status"] not in {"unidentifiable","unresolved_component"}:
            assert (config.output_dir/"methods"/method/"binary_mask.npy").is_file()
            assert (config.output_dir/"methods"/method/"binary_mask.tif").is_file()
            signed = config.output_dir/"methods"/method/"continuous_activity_signed.tif"
            positive = config.output_dir/"methods"/method/"positive_z.tif"
            assert signed.is_file() and positive.is_file()
            view = tifffile.imread(signed)
            assert view.dtype == np.uint16 and view.shape == video.shape
