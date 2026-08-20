import json
from pathlib import Path
import numpy as np
import pytest
from neurobench.experiments.learned_operator_selection.config import LearnedOperatorConfig
from neurobench.experiments.learned_operator_selection.decisions import Stage
from neurobench.experiments.learned_operator_selection.preflight import preflight
from neurobench.experiments.learned_operator_selection.runner import run_stage

ROOT=Path(__file__).resolve().parents[1]

def _fixture(tmp_path: Path) -> Path:
    payload=json.loads((ROOT/"examples/spon_ca_burst_learned_operator_selection.example.json").read_text())
    movie=tmp_path/"movie.npy"; np.save(movie,np.zeros((120,24,24),dtype=np.float32))
    labels=tmp_path/"labels.tsv"; rows=["burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\tstop_frame_zero_exclusive\tpoint_index\troi_identity\tx_px\ty_px\trecurrence_count"]
    for burst,(start,stop) in enumerate(((21,30),(41,50),(61,70),(81,90)),1): rows.append(f"{burst}\t{start}\t{stop}\t{start-1}\t{stop}\t1\troi_{burst:03d}\t12\t12\t1")
    labels.write_text("\n".join(rows)+"\n"); summary=tmp_path/"summary.json"; summary.write_text(json.dumps({"total_point_window_labels":4,"unique_roi_coordinates":4}))
    payload.update(source_video=str(movie),labels_tsv=str(labels),label_summary=str(summary),output_dir=str(tmp_path/"program"))
    payload["frames"]={"review_start_ui":1,"review_end_ui":100,"quiet_start_ui":1,"quiet_end_ui":20,"frame_period_ms":20.0}; payload["resources"]["max_ram_mib"]=1; payload["resources"]["min_free_disk_mib"]=1
    path=tmp_path/"config.json"; path.write_text(json.dumps(payload)); return path

def test_preflight_is_collision_safe_and_later_stage_refuses(tmp_path: Path) -> None:
    config=LearnedOperatorConfig.load(_fixture(tmp_path)); artifact=tmp_path/"preflight"
    result=preflight(config,artifact_dir=artifact); assert result["ready"] and (artifact/"label_projection_overlay.png").is_file()
    with pytest.raises(FileExistsError): preflight(config,artifact_dir=artifact)
    with pytest.raises(RuntimeError,match="identical config and stage"):
        run_stage(config,preflight_dir=artifact,stage=Stage.S1_OPERATOR_SCREEN)
