import pytest

from neurobench.experiments.gamma_ls_difference import necessity_report as r


def test_discovery_pool_uses_counts_not_mean_percentages():
    rows=[]
    for tp,fp,active,events,recovered in [(1,0,2,1,1),(1,99,8,3,0)]:
        rows.append(dict(true_positive_count=tp,false_positive_count=fp,active_region_frame_count=active,
            event_count=events,proposal_count=tp+fp,recovered_event_count=recovered,eligible_area_px=100,exposure_seconds=2,
            deadline_rows=[dict(deadline_ms=d,event_count=events,recovered_by_deadline=recovered) for d in r.DEADLINES]))
    p=r.pool(rows)
    assert p['precision']==pytest.approx(2/101)
    assert p['sensitivity']==.2 and p['coverage']==.25
    assert p['false_proposals_per_10000um2_second']==9900
    assert p['deadline_rows'][-1]['event_count']==4


def test_monitoring_pool_has_separate_roi_frame_denominator():
    row=dict(true_positive_count=2,false_positive_count=3,false_negative_count=6,true_negative_count=9,roi_frame_count=20,
        deadline_rows=[dict(deadline_ms=d,event_count=2,recovered_by_deadline=1) for d in r.DEADLINES])
    p=r.monitoring_pool([row,row])
    assert p['roi_frame_count']==40 and p['precision']==.4 and p['sensitivity']==.25
    assert p['inactive_roi_frame_exceedance_rate']==.25
    assert p['deadline_rows'][-1]['event_count']==4


def test_zero_opportunities_do_not_become_zero_precision():
    assert r.ratio(0,0) is None
    assert r.pct(None)=='undefined'
