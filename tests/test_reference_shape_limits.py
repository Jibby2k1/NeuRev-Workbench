"""Small mathematical checks; no simulation movies or detector evaluation."""
import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference.reference_shape_limits import (
    center_shape_limit, conditioned_footprint, derive_table,
)


def weights():
    target = np.zeros((3, 3)); target[1, 1] = 1
    return target, np.full((3, 3), 1/9)


def test_conditioning_matches_explicit_finite_gaussian_at_used_edge():
    raw, u = conditioned_footprint(3., 7)
    x = np.arange(-4, 5, dtype=np.float64)
    g = np.exp(-x*x/2); g /= g.sum()
    for y, x in ((11, 11), (4, 4), (18, 18)):
        expected = np.sum(raw[y-4:y+5, x-4:x+5] * g[:, None] * g[None, :])
        assert u[y, x] == pytest.approx(expected, rel=2e-15)


def test_neural_cutoff_is_circular_inclusive_and_artifact_has_no_cutoff():
    finite, _ = conditioned_footprint(1., 8, circular_truncate_sigma=4.)
    artifact, _ = conditioned_footprint(3., 15)
    c = len(finite)//2
    assert finite[c, c+4] > 0
    assert finite[c+1, c+4] == 0
    c = len(artifact)//2
    assert artifact[c, c+13] > 0
    assert artifact[c, c] == 1


def test_shape_limit_equals_direct_weighted_score_for_positive_amplitudes():
    target, reference = weights()
    _, u = conditioned_footprint(1., 1, circular_truncate_sigma=4.)
    result = center_shape_limit(target, reference, u)
    c = len(u)//2; patch = u[c-1:c+2, c-1:c+2]
    for background in (0., 3.):
        for amplitude in (1., 2., 20.):
            field = background + amplitude*patch
            a = np.sum(target*field); m = np.sum(reference*field)
            spread = np.sqrt(np.sum(reference*(field-m)**2))
            assert (a-m)/spread == pytest.approx(result['positive_amplitude_shape_limit'], rel=2e-13)


def test_fixed_floor_breaks_small_amplitude_invariance_then_limit_is_exact():
    target, reference = weights()
    _, u = conditioned_footprint(1., 1)
    row = center_shape_limit(target, reference, u)
    floor = .5
    onset_amplitude = floor/row['reference_center_spread_gain']
    scores = [b*row['contrast_center_gain']/max(b*row['reference_center_spread_gain'], floor)
              for b in (onset_amplitude/2, onset_amplitude, onset_amplitude*10)]
    assert scores == pytest.approx([row['positive_amplitude_shape_limit']/2,
                                   row['positive_amplitude_shape_limit'],
                                   row['positive_amplitude_shape_limit']])


def test_zero_variance_and_unnormalized_weights_are_rejected():
    target, reference = weights()
    with pytest.raises(ValueError, match='Positive reference shape variance'):
        center_shape_limit(target, target, np.ones((3, 3)))
    with pytest.raises(ValueError, match='unit mass'):
        center_shape_limit(target, reference*2, np.ones((3, 3)))


def test_table_preserves_inputs_and_common_full_halo():
    target, reference = weights(); copies = target.copy(), reference.copy()
    rows, arrays = derive_table(target, {'fixture': reference})
    assert len(rows) == 3
    assert all(r['raw_square_half_width_px'] == 5 for r in rows)
    assert {r['shape_id'] for r in rows} == {'neural_sigma1', 'neural_sigma2', 'artifact_sigma3'}
    assert np.array_equal(target, copies[0]) and np.array_equal(reference, copies[1])
    arrays['reference_fixture'][0, 0] = 100
    assert np.array_equal(reference, copies[1])
