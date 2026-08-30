from __future__ import annotations

import numpy as np

from neurobench.experiments.neuron_identifiability.spatial_specificity import clustered_interval,specificity


def test_specificity_is_bounded_and_signed() -> None:
    assert 0<specificity(5,1)<1
    assert -1<specificity(1,5)<0
    assert specificity(0,0)==0


def test_clustered_interval_resamples_sites_not_occurrences() -> None:
    rows=[]
    for site,value in (("a",1.0),("b",3.0),("c",5.0)):
        rows.extend([{"observation_site_id":site,"delta":value}]*4)
    result=clustered_interval(rows,"delta",seed=4,draws=1000)
    assert result["site_count"]==3
    assert result["mean"]==3.0
