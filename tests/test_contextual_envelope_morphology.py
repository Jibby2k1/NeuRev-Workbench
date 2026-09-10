import numpy as np
from neurobench.experiments.neuron_identifiability.contextual_envelope_morphology import aligned,halfmax_width,operators,valid_quiet_starts

def test_operator_identities_and_bounds():
    a=np.array([0,.2,.7,1,.4],dtype=np.float32)[:,None,None];f=operators(a)
    assert set(f)=={"A","A2","U","C","P","H"};np.testing.assert_allclose(f["A2"],f["A"]**2);np.testing.assert_allclose(f["P"],f["A"]*f["U"]);np.testing.assert_allclose(f["H"],f["A"]*f["C"]);assert np.all(f["C"]<=1+1e-6);assert np.all(f["H"]<=f["A"]+1e-6)

def test_alignment_and_width():
    x=np.array([0,1,3,3,1,0],dtype=float);z=aligned(x,2);assert z[20]==3;assert halfmax_width(x,2)==2

def test_quiet_starts_require_complete_window():
    m=np.array([1,1,0,1,1,1],dtype=bool);np.testing.assert_array_equal(valid_quiet_starts(m,2),[0,3,4])
