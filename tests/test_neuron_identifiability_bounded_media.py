import numpy as np
import pytest

from neurobench.experiments.neuron_identifiability.bounded_field_media import scale_uint8


def test_bounded_media_scaling_is_fixed_and_clipped():
    result=scale_uint8(np.array([-1.,0.,5.,10.,11.]),0.,10.)
    assert result.tolist()==[0,0,128,255,255]
    with pytest.raises(ValueError): scale_uint8(np.zeros(1),1.,1.)
