from __future__ import annotations
import numpy as np
from neurobench.experiments.learnable_contrast.diagnostic import factor_matrix,_robust_whiten

def test_factor_matrix_is_complete_and_unique():
    rows=factor_matrix()
    assert len(rows)==8
    assert len({r['combination_id'] for r in rows})==8
    assert {r['input'] for r in rows}=={'raw_quiet_residual','kalman_spatiotemporal'}
    assert {r['objective'] for r in rows}=={'legacy_raw_score','stabilized_log_score'}
    assert {r['initialization'] for r in rows}=={'fixed_guarded','jittered_guarded'}

def test_quiet_whitening_is_positive_bounded_and_quiet_only():
    rng=np.random.default_rng(3); x=rng.normal(5,2,size=(12,5,6)).astype(np.float32); x[10:]+=20
    z,summary=_robust_whiten(x,10,8)
    assert z.shape==x.shape and z.dtype==np.float32
    assert z.min()>=0 and z.max()<=8
    assert summary['scale_floor']>0

def test_main_cli_registers_spatiotemporal_diagnostic():
    from neurobench.cli.main import build_parser
    args=build_parser(active_command='experiment').parse_args(['experiment','learnable-contrast','diagnostic','--config','matrix.json'])
    assert args.experiment_workflow=='learnable-contrast'
    assert args.experiment_action=='diagnostic'
    assert callable(args.func)
