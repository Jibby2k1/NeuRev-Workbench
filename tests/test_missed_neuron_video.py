from neurobench.experiments.hierarchical_parzen_ica.missed_neuron_video import (
    _zoom_box,
    frame_identity_status,
    missed_identity_records,
)


def _audit(burst_id, identity, recovered):
    return {
        "burst_id": burst_id,
        "roi_identity": identity,
        "linear_recovered": recovered,
        "oracle_union_recoverable": "true",
    }


def _label(burst_id, identity, start, stop, x, y):
    return {
        "burst_id": burst_id,
        "roi_identity": identity,
        "start_frame_zero": start,
        "stop_frame_zero_exclusive": stop,
        "x_px": x,
        "y_px": y,
    }


def test_missed_identity_records_keeps_all_bursts_for_missed_identity():
    audit = [
        _audit(1, "roi_a", "false"),
        _audit(2, "roi_a", "true"),
        _audit(1, "roi_b", "true"),
    ]
    labels = [
        _label(1, "roi_a", 10, 20, 30, 40),
        _label(2, "roi_a", 30, 40, 31, 41),
        _label(1, "roi_b", 10, 20, 50, 60),
    ]

    identities, records = missed_identity_records(
        audit, labels, recovery_field="linear_recovered"
    )

    assert identities == ["roi_a"]
    assert len(records) == 2
    assert frame_identity_status(records, "roi_a", 9) == ("inactive", None)
    assert frame_identity_status(records, "roi_a", 10) == (
        "active_missed",
        1,
    )
    assert frame_identity_status(records, "roi_a", 30) == (
        "active_recovered",
        2,
    )


def test_zoom_box_is_even_when_missed_identity_is_near_boundary():
    x0, y0, x1, y1 = _zoom_box({"roi_a": (99, 79)}, (80, 100), 5)

    assert 0 <= x0 < x1 <= 100
    assert 0 <= y0 < y1 <= 80
    assert (x1 - x0) % 2 == 0
    assert (y1 - y0) % 2 == 0
