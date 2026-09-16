"""Latency definitions and parity gates, without GPU or recording pixels."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import spatiotemporal_latency as latency


def test_fixed_kernel_matrix_and_frame_accounting():
    specs = latency.central_specifications()
    assert [row["time_scale_ms"] for row in specs] == [0,20,60,200]
    assert all(row["radius_px"] == 7.5 and row["target_sigma_px"] == 1 and row["reference_n"] == 9 for row in specs)
    frames = list(range(1400,2360))[slice(*latency.SOURCE_ROWS)]
    assert frames == list(range(1960,2088))
    assert frames[latency.DISCARDED_FRAMES:] == list(range(1992,2088))
    assert latency.WORKER_TIMEOUT_S < 60


class FakeClock:
    def __init__(self):self.now = 0.;self.sleeps = []
    def __call__(self):return self.now
    def sleep(self,duration):self.sleeps.append(duration);self.now += duration
    def add(self,duration):self.now += duration


def test_processing_latency_and_backlog_are_separate_and_pacing_sleeps():
    clock = FakeClock()
    durations = [.001,.003,.030,.002]
    def stream():
        for index,duration in enumerate(durations):
            clock.add(duration)
            yield index,index+1,{}
    def finalize(stages,frame):
        clock.add(.001)
        return {},[dict(source_frame_ui=frame)]
    def sink(index,stages,candidates):clock.add(.004)
    rows,startup = latency._measure_stream(iter(stream()),finalize,sink,[10,11,12,13],
        discarded_frames=1,clock=clock,sleeper=clock.sleep)
    measured = rows[1:]
    assert [row["processing_ms"] for row in measured] == pytest.approx([4,31,3])
    assert [row["bookkeeping_ms"] for row in measured] == pytest.approx([4,4,4])
    assert clock.sleeps == pytest.approx([.012])
    assert measured[2]["start_lateness_ms"] == pytest.approx(15)
    assert measured[2]["release_to_result_ms"] == pytest.approx(18)
    assert measured[2]["completion_lateness_ms"] == 0
    summary = latency.timing_summary(rows)
    assert summary["processing_deadline_exceedance_count"] == 1
    assert summary["release_to_result_deadline_exceedance_count"] == 1
    assert summary["processing_ms"]["max"] == pytest.approx(31)
    assert startup["first_next_normalize_nms_ms"] == pytest.approx(2)


def test_incorrect_iterator_grain_is_rejected():
    with pytest.raises(ValueError,match="exactly one"):
        latency._measure_stream(iter([(0,2,{})]),lambda *args:({},[]),lambda *args:None,
                                [1,2],discarded_frames=1)


def test_float32_rounding_precedes_score_as_in_the_sealed_runner():
    a = np.full((1,15,17),1.00000007,dtype=np.float64)
    m = np.full_like(a,1.00000001)
    variance = np.full_like(a,.25)
    stages = latency._normalize_frame({"A":a,"M":m,"variance":variance},[0,0,15,17],.6)
    assert all(array.dtype == np.float32 for array in stages.values())
    expected = (a[0].astype(np.float32)-m[0].astype(np.float32))/np.float32(.6)
    np.testing.assert_array_equal(stages["Score"],expected)
    assert not np.allclose(stages["Score"],((a-m)/.6)[0],rtol=1e-4,atol=1e-9)


def test_fixed_stage_tolerance_rejects_outside_bound_and_nonfinite():
    expected = {key:np.ones((2,3,4),dtype=np.float32) for key in ("A","M","Spread","Score")}
    actual = {key:value.copy() for key,value in expected.items()}
    actual["Score"][0,0,0] += np.float32(1e-5)
    assert latency.compare_stages(actual,expected)["pass"]
    actual["Score"][0,0,0] += np.float32(1e-3)
    comparison = latency.compare_stages(actual,expected)
    assert not comparison["pass"] and comparison["stages"]["Score"]["violation_count"] == 1
    actual["M"][0,0,0] = np.nan
    assert not latency.compare_stages(actual,expected)["pass"]


def test_candidate_coordinate_changes_cannot_hide_within_score_tolerance():
    original = {"source_frame_ui":1992,"x_px":20,"y_px":12,"score":1.5}
    assert latency.compare_candidates([original],[original])["pass"]
    result = latency.compare_candidates([{**original,"x_px":21}],[original])
    assert not result["pass"] and result["sealed_only_frame_y_x"] == [[1992,12,20]]
    with pytest.raises(ValueError,match="Nonfinite"):
        latency.compare_candidates([{**original,"score":float("nan")}],[original])


def test_worker_timeout_is_bounded_and_preserves_diagnostics(tmp_path,monkeypatch):
    job = tmp_path/"job.json";job.write_text("{}")
    def timeout(command,**kwargs):
        assert kwargs["timeout"] == 55
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        raise subprocess.TimeoutExpired(command,55,output=b"started",stderr=b"fixture timeout")
    monkeypatch.setattr(latency.subprocess,"run",timeout)
    result = latency._invoke_worker(job)
    assert result["status"] == "TIMEOUT" and result["worker_killed_and_reaped"]
    assert (tmp_path/"stderr.log").read_text() == "fixture timeout"
    assert json.loads((tmp_path/"process.json").read_text())["timeout_s"] == 55
    with pytest.raises(ValueError,match="below60"):
        latency._invoke_worker(job,timeout_s=60)


def test_bound_attempt_resume_rejects_modified_frame_timings(tmp_path):
    contract = {"fixture":True};digest = latency._canonical(contract)
    job = {"contract":contract,"case":{"spec_id":"fixture"},"repetition":1}
    latency._write(tmp_path/"job.json",job)
    result = {"status":"PASS","job_sha256":latency._sha(tmp_path/"job.json"),"contract_sha256":digest,
              "spec_id":"fixture","repetition":1,"stage_comparison":{"pass":True},"candidate_comparison":{"pass":True}}
    latency._write(tmp_path/"result.json",result)
    (tmp_path/"frame_timings.tsv").write_text("source_frame_ui\tprocessing_ms\n1992\t5\n")
    latency._write(tmp_path/"complete.json",{"status":"PASS","contract_sha256":digest,
        "artifacts":[latency._binding(tmp_path/name) for name in ("job.json","result.json","frame_timings.tsv")]})
    assert latency._complete_attempt(tmp_path,digest) == result
    (tmp_path/"frame_timings.tsv").write_text("changed")
    with pytest.raises(ValueError,match="changed"):
        latency._complete_attempt(tmp_path,digest)
