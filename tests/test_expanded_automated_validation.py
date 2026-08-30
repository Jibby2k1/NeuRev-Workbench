import numpy as np
from neurobench.experiments.neuron_identifiability.expanded_automated_validation import coverage_risk,trace_stats
def test_selective_risk_has_requested_coverages():
 rows=coverage_risk(np.array([0,0,1,1]),np.array([.01,.4,.6,.99]))
 assert [r['coverage'] for r in rows]==[.25,.5,.75,1.0]
 assert rows[0]['error_rate']==0
def test_trace_stats_scale_invariant_peak_mad():
 x=np.r_[np.linspace(-1,1,40),np.linspace(0,4,40),np.zeros(40)]
 assert np.isclose(trace_stats(x)['peak_mad'],trace_stats(3*x)['peak_mad'])
