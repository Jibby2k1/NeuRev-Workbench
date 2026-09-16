"""Small campaign metadata fixtures; no production arrays or rendered media."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from neurobench.experiments.gamma_ls_difference import background_media as media


def protocol_and_states():
    cases, logical = [], []
    for seed in (16, 17, 18):
        for background in ("sloped", "flat"):
            for normalization in ("raw", "conditioned"):
                for v in (0, 1):
                    for s in (0, 1):
                        for t in (0, 1):
                            norm = "raw" if s == t == 0 else normalization
                            name = f"{background}_{norm}_v{v}s{s}t{t}_{seed}"
                            old = background == "sloped" and norm == "raw"
                            factors = dict(seed=seed, background=background, normalization=normalization, V=v, S=s, T=t)
                            logical.append(dict(factors, canonical_case_id=name))
                            if norm != normalization:
                                continue
                            cases.append(dict(factors, case_id=name, kind="factorial", reused_dataset=old,
                                              baseline_case_id=name if old else None))
    protocol = dict(cases=cases, logical_cases=logical,
                    references=[dict(arm_id=a) for a in sorted(media.REFERENCE_ARMS)],
                    expected_cells=252, expected_datasets=84, expected_curve_rows=5040, reused_audits=72, new_audits=180)
    states = [dict(case_id=c["case_id"], arm_id=r["arm_id"], study="background", input_mode="level",
                   readout="Z", window=3, calibration_method="global", reused_audit=c["reused_dataset"],
                   baseline_case_id=c["baseline_case_id"], baseline_arm=r["arm_id"] if c["reused_dataset"] else None,
                   case_kind=c["kind"], **{k:c[k] for k in media.FACTOR_FIELDS})
              for c in cases for r in protocol["references"]]
    return protocol, states


def test_complete_factorial_and_explicit_reuse_owner(monkeypatch):
    protocol, states = protocol_and_states()
    monkeypatch.setattr(media, "_api", lambda: SimpleNamespace(cells=lambda p: states))
    assert media.matrix(protocol) == states
    assert sum(media.reused(c) for c in states) == 72
    assert [len(states[w::3]) for w in range(3)] == [84, 84, 84]
    mapping = media.logical_mapping(protocol)
    assert len(mapping) == 288
    assert len({(r["canonical_case_id"], r["arm_id"]) for r in mapping}) == 252


@pytest.mark.parametrize("change", ["wrong_seed", "wrong_background", "missing", "duplicate"])
def test_logical_alias_must_resolve_to_exact_factor_condition(monkeypatch, change):
    protocol, states = protocol_and_states()
    row = next(r for r in protocol["logical_cases"] if r["normalization"] == "conditioned" and r["S"] == r["T"] == 0)
    if change in ("wrong_seed", "wrong_background"):
        row["canonical_case_id"] = next(c["case_id"] for c in protocol["cases"]
            if c["normalization"] == "raw" and c["V"] == row["V"] and c["S"] == c["T"] == 0
            and (c["seed"] != row["seed"] if change == "wrong_seed" else c["background"] != row["background"]))
    elif change == "missing":
        protocol["logical_cases"].pop()
    else:
        protocol["logical_cases"][-1] = protocol["logical_cases"][0]
    monkeypatch.setattr(media, "_api", lambda: SimpleNamespace(cells=lambda p: states))
    with pytest.raises(ValueError, match="[Ll]ogical"):
        media.matrix(protocol)


def schedule_rows():
    _, states = protocol_and_states()
    costs = {"mean2of3_n3": 20, "mean1_n9": 200, "mean4of3_n9": 2000}
    return [dict(c, reused=media.reused(c), model_closeup_frames=costs[c["arm_id"]]) for c in states]


def test_count_capped_schedule_balances_wide_reference_tail_without_losing_states():
    rows = schedule_rows()
    original = copy.deepcopy(rows)
    for workers in (1, 2, 3):
        bins = media.balanced_partitions(rows, workers)
        assert bins == media.balanced_partitions(list(reversed(rows)), workers)
        keys = [tuple(k) for b in bins for k in b["assigned_keys"]]
        assert len(keys) == len(set(keys)) == 252
        assert set(keys) == {(r["case_id"], r["arm_id"]) for r in rows}
        assert all(b["cells"] == 252 // workers for b in bins)
        loads = [b["estimated_new_model_closeup_frames"] for b in bins]
        assert sum(loads) == sum(r["model_closeup_frames"] for r in rows if not r["reused"])
        assert max(loads) - min(loads) <= 2000
    assert rows == original


def test_schedule_rejects_same_count_key_swap_and_invalid_frame_cost():
    rows = schedule_rows()
    forecast = dict(cells=rows, worker_partitions={"3": media.balanced_partitions(rows, 3)})
    assert media.frozen_partition(forecast, 0, 3)["cells"] == 84
    bins = forecast["worker_partitions"]["3"]
    bins[0]["assigned_keys"][0], bins[1]["assigned_keys"][0] = bins[1]["assigned_keys"][0], bins[0]["assigned_keys"][0]
    with pytest.raises(ValueError, match="assignment differs"):
        media.frozen_partition(forecast, 0, 3)
    rows[0]["model_closeup_frames"] = -1
    with pytest.raises(ValueError, match="frame estimate"):
        media.balanced_partitions(rows, 3)


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
        states[-1]["reused_audit"] = True
        states[0]["reused_audit"] = False
    monkeypatch.setattr(media, "_api", lambda: SimpleNamespace(cells=lambda p: states))
    with pytest.raises(ValueError):
        media.matrix(protocol)


def test_individual_frozen_ranges_and_explicit_shared_extension():
    common = dict(Raw=[87., 118.], Input=[0., 115.], A=[0., 115.], Score=[-8., 8.])
    existing = {f"case{i}": copy.deepcopy(common) for i in range(24)}
    existing["case1"]["A"][1] = 119.
    existing["case2"]["Raw"][0] = 86.
    original = copy.deepcopy(existing)
    originals = media.reused_case_limits(existing, sorted(existing))
    baseline = media.pooled_reuse_limits(originals)
    assert baseline == dict(Raw=[86., 118.], Input=[0., 119.], A=[0., 119.], Score=[-8., 8.])
    expanded = media.extend_null_limits(baseline, [dict(raw_low=80., raw_high=122., input_high=120., a_high=119.)])
    assert expanded == dict(Raw=[80., 122.], Input=[0., 120.], A=[0., 120.], Score=[-8., 8.])
    assert existing == original == originals
    originals["case1"]["A"][1] = 999.
    assert existing == original


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
