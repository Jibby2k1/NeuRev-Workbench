"""Small metadata fixtures; no production arrays or media rendering."""
import copy
from types import SimpleNamespace

import pytest

from neurobench.experiments.gamma_ls_difference import reference_media as media


def states():
    return [dict(case_id=f"case{i}", arm_id=f"mean{mean}_n{n}", study="reference",
                 input_mode="level", readout="Z", window=3, calibration_method="global",
                 reused_audit=i < 12 and mean == "1" and n == 9)
            for i in range(21) for mean in ("2of3", "1", "4of3") for n in (3, 9, 15)]


def test_full_matrix_and_only_central_crowding_reuse(monkeypatch):
    value = states()
    monkeypatch.setattr(media, "_api", lambda: SimpleNamespace(cells=lambda protocol: value))
    assert len(media.matrix({})) == 189
    assert sum(media.reused(c) for c in value) == 12


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_grid", "wrong_reuse", "wrong_input"])
def test_matrix_rejects_incomplete_or_semantically_wrong_states(monkeypatch, change):
    value = states()
    if change == "duplicate":
        value[-1] = value[-2]
    elif change == "missing":
        value.pop()
    elif change == "wrong_grid":
        value[0]["arm_id"] = "mean2of3_n4"
    elif change == "wrong_reuse":
        value[0]["reused_audit"] = True
        next(c for c in value if c["arm_id"] == "mean1_n9")["reused_audit"] = False
    else:
        value[0]["input_mode"] = "difference"
    monkeypatch.setattr(media, "_api", lambda: SimpleNamespace(cells=lambda protocol: value))
    with pytest.raises(ValueError):
        media.matrix({})


def test_frozen_crowding_limits_preserved_exactly_and_no_alias_mutation():
    original = dict(Raw=[94.41658752441407, 106.89186404418945],
                    level=dict(X=[0., 103.24390411376953], A=[0., 103.24390411376953], Z=[-8., 8.]))
    existing = {f"case{i}": copy.deepcopy(original) for i in range(12)}
    common = media.common_crowding_limits(existing, sorted(existing))
    assert common["Raw"] == original["Raw"]
    nulls = media.extend_null_limits(common, [dict(raw_low=88., raw_high=118., input_high=114., a_high=113.)])
    assert nulls == dict(Raw=[88., 118.], Input=[0., 114.], A=[0., 114.], Score=[-8., 8.])
    assert existing["case0"] == original
    assert common["Raw"] == original["Raw"]
    existing["case1"]["level"]["A"][1] += 1
    with pytest.raises(ValueError, match="identical frozen"):
        media.common_crowding_limits(existing, sorted(existing))


def test_null_extension_uses_common_range_when_no_extension_is_needed():
    common = dict(Raw=[90., 110.], Input=[0., 104.], A=[0., 104.], Score=[-8., 8.])
    assert media.extend_null_limits(common, [dict(raw_low=91., raw_high=109., input_high=102., a_high=103.)]) == common
    with pytest.raises(ValueError, match="Nonfinite"):
        media.extend_null_limits(common, [dict(raw_low=91., raw_high=109., input_high=float("nan"), a_high=103.)])


def test_media_operations_refuse_completed_root(tmp_path):
    (tmp_path / "completion_manifest.json").write_text('{"status":"PASS"}')
    before = list(tmp_path.iterdir())
    with pytest.raises(FileExistsError, match="immutable"):
        media.forecast(tmp_path)
    with pytest.raises(FileExistsError, match="immutable"):
        media.media(tmp_path)
    assert list(tmp_path.iterdir()) == before
