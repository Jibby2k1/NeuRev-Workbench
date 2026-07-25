from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from neurobench.experiments.learnable_contrast.core import Config, load_labels


def test_normalized_labels_match_summary():
    root=Path(__file__).resolve().parents[1]
    labels=root/"Inputs/Spon Ca Burst/labels/labels_normalized.tsv"
    summary=root/"Inputs/Spon Ca Burst/labels/label_summary.json"
    if not labels.exists(): pytest.skip("local ignored labels unavailable")
    rows=load_labels(labels); meta=json.loads(summary.read_text())
    assert len(rows)==meta["total_point_window_labels"]==79
    assert len({r["roi_identity"] for r in rows})==meta["unique_roi_coordinates"]==27
    assert {b:sum(r["burst_id"]==b for r in rows) for b in range(1,5)}=={1:15,2:20,3:21,4:23}
    assert all(0<=r["x_px"]<573 and 0<=r["y_px"]<340 for r in rows)


def test_model_forward_is_finite_and_shape_preserving():
    torch=pytest.importorskip("torch")
    from neurobench.experiments.learnable_contrast.core import _model_class
    model=_model_class()(21,True)
    x=torch.rand(3,1,29,29)
    y=model(x)
    assert y.shape==x.shape
    assert torch.isfinite(y).all()
    kt,kr=model.kernels()
    assert torch.allclose(kt.sum(),torch.tensor(1.0),atol=1e-6)
    assert torch.allclose(kr.sum(),torch.tensor(1.0),atol=1e-6)


def test_main_cli_registers_learnable_contrast_workflow():
    from neurobench.cli.main import build_parser
    args=build_parser(active_command="experiment").parse_args(["experiment","learnable-contrast","preflight","--config","x.json","--artifact-dir","out"])
    assert args.experiment_workflow=="learnable-contrast"
    assert args.experiment_action=="preflight"
    assert callable(args.func)
