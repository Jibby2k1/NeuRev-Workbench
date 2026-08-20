import json
from pathlib import Path
import numpy as np
from neurobench.experiments.learned_operator_selection.config import LearnedOperatorConfig
from neurobench.experiments.learned_operator_selection.data import outer_burst_splits, ui_inclusive_to_zero_half_open
from neurobench.experiments.learned_operator_selection.evaluation import deterministic_peaks

ROOT=Path(__file__).resolve().parents[1]

def test_frame_conversion_and_outer_splits() -> None:
    assert ui_inclusive_to_zero_half_open(1800,2359)==(1799,2359)
    labels=[{"burst_id":burst} for burst in (1,2,3,4)]
    splits=outer_burst_splits(labels)
    assert len(splits)==4 and all(len(row["inner_rotations"])==3 for row in splits)
    assert 1 not in splits[0]["training_bursts"]

def test_score_ties_are_ordered_by_y_then_x() -> None:
    score=np.zeros((15,15),dtype=np.float32); score[4,9]=3; score[4,5]=3; score[9,4]=3
    assert deterministic_peaks(score,1,limit=3)==[(3.0,5,4),(3.0,9,4),(3.0,4,9)]

def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    payload=json.loads((ROOT/"examples/spon_ca_burst_learned_operator_selection.example.json").read_text())
    payload["evaluation"]["learned_radius"]=7; path=tmp_path/"bad.json"; path.write_text(json.dumps(payload))
    try: LearnedOperatorConfig.load(path)
    except ValueError as exc: assert "unknown" in str(exc)
    else: raise AssertionError("unknown config field was accepted")
