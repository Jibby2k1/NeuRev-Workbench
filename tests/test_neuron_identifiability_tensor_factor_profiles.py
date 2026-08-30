import numpy as np
from neurobench.experiments.neuron_identifiability.tensor_factor_profiles import _align,_canonical,_eta_squared
def test_component_alignment_handles_permutation_and_sign():
 a=np.array([[1.,0.],[0.,1.],[1.,1.]]);b=np.ones((4,2));c=np.array([[1.,0.],[0.,1.]])
 ref=_canonical((a,b,c))[0];aligned,corrs=_align(ref,(a[:,[1,0]]*np.array([-1,1]),b[:,[1,0]],c[:,[1,0]]*np.array([-1,1])))
 assert min(corrs)>.99
def test_eta_squared_detects_separated_groups():
 assert _eta_squared(np.array([0.,0.,2.,2.]),np.array([1,1,2,2]))==1.0
