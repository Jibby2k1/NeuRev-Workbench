"""Small campaign metadata fixtures; no production arrays or rendered media."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from neurobench.experiments.gamma_ls_difference import noise_media as media


def protocol_and_states():
    cases = []
    for seed in (16, 17, 18):
        for v in (0, 1):
            for s in (0, 1):
                for t in (0, 1):
                    old = (v, s, t) == (0, 0, 0)
                    name = f"stationary{seed}" if old else f"v{v}s{s}t{t}_{seed}"
                    cases.append(dict(case_id=name, seed=seed, kind="factorial", V=v, S=s, T=t,
                                      reused_dataset=old, baseline_case_id=name if old else None))
        cases.append(dict(case_id=f"legacy{seed}", seed=seed, kind="legacy", V=None, S=None, T=None,
                          reused_dataset=True, baseline_case_id=f"legacy{seed}"))
    protocol = dict(cases=cases, references=[dict(arm_id=a) for a in sorted(media.REFERENCE_ARMS)],
                    expected_cells=81, expected_curve_rows=1620)
    states = [dict(case_id=c["case_id"], arm_id=r["arm_id"], study="noise", input_mode="level",
                   readout="Z", window=3, calibration_method="global", reused_audit=c["reused_dataset"],
                   baseline_case_id=c["baseline_case_id"], baseline_arm=r["arm_id"] if c["reused_dataset"] else None,
                   case_kind=c["kind"], seed=c["seed"], V=c["V"], S=c["S"], T=c["T"])
              for c in cases for r in protocol["references"]]
    return protocol, states


def test_complete_factorial_and_explicit_reuse_owner(monkeypatch):
    protocol, states = protocol_and_states()
    monkeypatch.setattr(media, "_api", lambda: SimpleNamespace(cells=lambda p: states))
    assert media.matrix(protocol) == states
    assert sum(media.reused(c) for c in states) == 18
    assert [len(states[w::3]) for w in range(3)] == [27, 27, 27]


@pytest.mark.parametrize("change", ["duplicate", "missing", "grid", "reuse_owner", "input", "factor", "new_reuse"])
def test_matrix_rejects_missing_states_and_changed_scientific_scope(monkeypatch, change):
    protocol, states = protocol_and_states()
    if change == "duplicate":
        states[-1] = states[-2]
    elif change == "missing":
        states.pop()
    elif change == "grid":
        protocol["references"][0]["arm_id"] = "mean1_n15"
    elif change == "reuse_owner":
        states[0]["baseline_arm"] = "wrong_reference"
    elif change == "input":
        states[0]["input_mode"] = "difference"
    elif change == "factor":
        protocol["cases"][1]["S"] = 1
    else:
        states[3]["reused_audit"] = True
        states[0]["reused_audit"] = False
    monkeypatch.setattr(media, "_api", lambda: SimpleNamespace(cells=lambda p: states))
    with pytest.raises(ValueError):
        media.matrix(protocol)


def test_shared_frozen_null_range_and_explicit_pooled_extension():
    common = dict(Raw=[87., 118.], Input=[0., 115.], A=[0., 115.], Score=[-8., 8.])
    existing = {f"case{i}": copy.deepcopy(common) for i in range(6)}
    original = copy.deepcopy(existing)
    baseline = media.common_reuse_limits(existing, sorted(existing))
    assert media.extend_null_limits(baseline, [dict(raw_low=90., raw_high=117., input_high=110., a_high=110.)]) == common
    expanded = media.extend_null_limits(baseline, [dict(raw_low=80., raw_high=122., input_high=120., a_high=119.)])
    assert expanded == dict(Raw=[80., 122.], Input=[0., 120.], A=[0., 120.], Score=[-8., 8.])
    assert existing == original
    existing["case1"]["A"][1] += 1
    with pytest.raises(ValueError, match="identical frozen"):
        media.common_reuse_limits(existing, sorted(existing))


def test_nonfinite_preview_rejected():
    common = dict(Raw=[87., 118.], Input=[0., 115.], A=[0., 115.], Score=[-8., 8.])
    with pytest.raises(ValueError, match="Nonfinite"):
        media.extend_null_limits(common, [dict(raw_low=90., raw_high=117., input_high=float("nan"), a_high=110.)])


def test_display_resume_passes_plain_bindings_to_strict_runner_verifier(tmp_path, monkeypatch):
    protocol, states = protocol_and_states()
    source = tmp_path / "stage.bin"
    source.write_bytes(b"bound display preview")
    record = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(), size_bytes=source.stat().st_size)
    saved = dict(protocol_sha256="frozen", baseline_display=record,
                 preview_sources=[dict(record, case_id="case", stage="A")])
    (tmp_path / "display_contract.json").write_text(json.dumps(saved))
    checked = []
    def strict_verify(value):
        assert value == record
        checked.append(value)
    api = SimpleNamespace(load=lambda root: protocol, cells=lambda p: states,
                          read=lambda path: json.loads(path.read_text()), sha256=lambda path: "frozen", verify=strict_verify)
    monkeypatch.setattr(media, "_api", lambda: api)
    assert media.display(tmp_path) == saved
    assert len(checked) == 2


def test_completed_root_is_immutable(tmp_path):
    (tmp_path / "completion_manifest.json").write_text('{"status":"PASS"}')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for fn in (media.forecast, media.media, media.display):
        with pytest.raises(FileExistsError, match="immutable"):
            fn(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
