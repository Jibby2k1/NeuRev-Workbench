import numpy as np

from neurobench.experiments.unsupervised_ica_eval.core import (
    align_components, analytic_baselines, fit_two_frame_ica,
    representation_fingerprint, stability_summary, temporal_block_shuffle,
)
from neurobench.experiments.unsupervised_ica_eval.traces import candidate_signature_summary, paired_trace_metrics


def _mixture(seed=2, n=5000):
    rng=np.random.default_rng(seed); sources=np.column_stack((rng.laplace(size=n),rng.uniform(-2,2,size=n)))
    return sources @ np.asarray([[1.0,.7],[.3,1.2]]).T


def test_fit_baselines_alignment_stability_and_freeze_are_deterministic():
    x=_mixture(); fits=[fit_two_frame_ica(x,seed=s) for s in (1,2,3)]
    summary=stability_summary(fits,top_fraction=.05)
    assert summary["mean_activation_correlation"] > .99
    assert summary["mean_absolute_direction_cosine"] > .99
    baseline=analytic_baselines(x,random_seed=4)
    assert set(("difference_signed","difference_standardized","difference_energy_normalized","random_rotation_0")) <= set(baseline)
    flipped=fits[0].activations[:,::-1]*np.asarray([-1,1])
    aligned,order=align_components(fits[0].activations,flipped)
    assert order == (1,0) and np.allclose(aligned,fits[0].activations)
    assert representation_fingerprint(fits[0],{"top_fraction":.01}) == representation_fingerprint(fits[0],{"top_fraction":.01})


def test_shuffle_breaks_pairing_and_trace_utilities_equal_weight_rois():
    x=_mixture(n=200); shuffled=temporal_block_shuffle(x,block_size=10,seed=7)
    assert shuffled.shape == x.shape and not np.array_equal(shuffled[:,1],x[:,1])
    raw=np.r_[np.zeros(5),[1,3,1,0]]; pipeline=np.r_[np.zeros(5),[1,6,1,0]]
    metrics=paired_trace_metrics(raw,pipeline,pre_event=5)
    assert metrics["pipeline_signed_peak"] == 6 and metrics["peak_time_offset_samples"] == 0
    base=np.asarray([0,0,1,3,1,0],float)
    traces=np.stack((base,base*.9,base*1.1,np.roll(base,1)))
    result=candidate_signature_summary(traces,np.asarray(["a","a","a","b"]))
    assert result["roi_count"] == 2 and len(result["population_template"]) == 6
