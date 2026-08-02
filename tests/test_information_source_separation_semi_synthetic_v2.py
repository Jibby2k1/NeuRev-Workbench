import numpy as np

from neurobench.experiments.information_source_separation.semi_synthetic_v2 import make_real_background_fixture_v2


def test_v2_rescales_all_truth_events_into_short_crop(tmp_path):
    path=tmp_path/'movie.npy'
    np.save(path, np.random.default_rng(1).normal(100,2,size=(120,16,16)).astype(np.float32))
    fixture=make_real_background_fixture_v2(path,quiet_start_ui=1,quiet_end_ui=100,
        crop_origin_xy=(0,0),crop_size_px=16,amplitude=1,seed=2,morphology_case='isolated')
    assert np.all(np.max(fixture.traces,axis=1)>0)
    assert fixture.metadata['fixture_contract_version'] == 2
    assert np.max(np.abs(fixture.observation-fixture.native_background-fixture.injected_neural_signal)) < 1e-5
