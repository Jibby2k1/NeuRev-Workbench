"""Bounded paced replay of four frozen kernels, one input frame per call.

This measures kernel computation, host/device transfers, float32 score formation,
and maintained NMS. It excludes conditioning, acquisition, downstream decisions,
and actuator hardware. It is not a biological metric or a full control benchmark.
CUDA is used only by an explicitly invoked worker, never at import/preflight.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


TIMES_MS = (0,20,60,200)
SOURCE_ROWS = (560,688)
DISCARDED_FRAMES = 32
PERIOD_S = .020
REPETITIONS = 2
WORKER_TIMEOUT_S = 55.0
FLOAT32_TOLERANCE = {"rtol":2e-5,"atol":2e-6,
    "rule":"abs(streamed-sealed)<=atol+rtol*abs(sealed), at every measured crop pixel; no tolerance fitting"}
SCOPE = {
    "included":"next(cuda iter_chunks chunk_frames=1), transfers in both directions, full-field valid-mass normalization, crop, float32 A/M/Spread, frozen-floor Score, exact maintained NMS",
    "excluded":"Gaussian/EMA/difference conditioning, source acquisition/exposure, transport from camera, decision logic, communications, actuator hardware, biological delay",
    "not_full_control_readiness":True,"biological_metrics_computed":False,
    "pacing":"nominal release every20ms; sleep until release, never busy-wait; backlog is retained without dropping frames",
    "lateness":"observed wall-time schedule includes measured buffer-copy bookkeeping; correctness comparisons are deferred until after pacing",
    "information_delay":"kernel lag support/marginals are recorded separately; a causal history length is not a measured neural-event delay",
    "file_cache_state":"uncontrolled OS page cache; repetitions are fresh worker processes, not independent acquisitions"}


def _read(path: str | Path) -> Any:return json.loads(Path(path).read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n");temporary.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def _binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path":str(path),"size_bytes":path.stat().st_size,"sha256":_sha(path)}


def _verify(record: Mapping[str, Any]) -> Path:
    path = Path(record["path"])
    if _binding(path) != {key:record[key] for key in ("path","size_bytes","sha256")}:
        raise ValueError(f"Frozen benchmark source/artifact changed: {path}")
    return path


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t")
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def central_specifications() -> list[dict[str, Any]]:
    return [{"radius_px":7.5,"time_scale_ms":value,"target_sigma_px":1.0,
             "reference_n":9.0,"dt_ms":20.0,"construction":"joint"} for value in TIMES_MS]


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(map(float,values))
    if not ordered:raise ValueError("Cannot summarize an empty timing sequence")
    ordinal = (len(ordered)-1)*probability;left = int(math.floor(ordinal));right = int(math.ceil(ordinal))
    return ordered[left]+(ordinal-left)*(ordered[right]-ordered[left])


def timing_summary(rows: Sequence[Mapping[str, Any]], *, period_s: float = PERIOD_S) -> dict[str, Any]:
    measured = [row for row in rows if row["measured"]]
    if not measured:raise ValueError("No measured frames after startup")
    result = {"measured_frame_count":len(measured),"period_ms":period_s*1000,
        "processing_deadline_exceedance_count":sum(row["processing_ms"] > period_s*1000 for row in measured),
        "release_to_result_deadline_exceedance_count":sum(row["completion_lateness_ms"] > 0 for row in measured)}
    for key in ("processing_ms","start_lateness_ms","release_to_result_ms","completion_lateness_ms","bookkeeping_ms"):
        values = [float(row[key]) for row in measured]
        result[key] = {"p50":_quantile(values,.5),"p95":_quantile(values,.95),"p99":_quantile(values,.99),"max":max(values)}
    result["processing_deadline_exceedance_fraction"] = result["processing_deadline_exceedance_count"]/len(measured)
    result["release_to_result_deadline_exceedance_fraction"] = result["release_to_result_deadline_exceedance_count"]/len(measured)
    return result


def _measure_stream(iterator: Any, finalize: Callable, sink: Callable, source_frames_ui: Sequence[int], *,
                    discarded_frames: int, period_s: float = PERIOD_S,
                    clock: Callable = time.perf_counter, sleeper: Callable = time.sleep) -> tuple[list[dict[str, Any]],dict[str, Any]]:
    """Time pipeline calls; defer comparisons and use real sleeping for pacing."""
    if not 0 < discarded_frames < len(source_frames_ui) or not math.isfinite(period_s) or period_s <= 0:
        raise ValueError("A nonempty warmup, measured interval, and positive pacing period are required")
    started = clock();origin = None;rows = []
    for index,frame in enumerate(source_frames_ui):
        measured = index >= discarded_frames
        if measured and origin is None:origin = clock()
        release = origin+(index-discarded_frames)*period_s if measured else None
        if measured:
            remaining = release-clock()
            if remaining > 0:sleeper(remaining)
        begin = clock()
        start,stop,stages = next(iterator)
        if start != index or stop != index+1:raise ValueError("Benchmark requires exactly one next-frame output")
        values,candidates = finalize(stages,int(frame))
        finish = clock()
        if measured:sink(index-discarded_frames,values,candidates)
        bookkept = clock()
        rows.append({"source_frame_ui":int(frame),"local_input_row":index,"measured":measured,
            "processing_ms":(finish-begin)*1000,"proposal_count":len(candidates),
            "scheduled_release_elapsed_ms":(release-started)*1000 if measured else None,
            "start_lateness_ms":max(0,begin-release)*1000 if measured else None,
            "release_to_result_ms":(finish-release)*1000 if measured else None,
            "completion_lateness_ms":max(0,finish-release-period_s)*1000 if measured else None,
            "bookkeeping_ms":(bookkept-finish)*1000})
    try:next(iterator)
    except StopIteration:pass
    else:raise ValueError("Iterator emitted more frames than the sealed source map")
    return rows,{"first_next_normalize_nms_ms":rows[0]["processing_ms"],
        "discarded_frame_count":discarded_frames,"warmup_and_iterator_startup_ms":(origin-started)*1000,
        "stream_loop_wall_ms":(clock()-started)*1000}


def _normalize_frame(stages: Mapping[str, Any], crop_yxyx: Sequence[int], floor: float) -> dict[str, Any]:
    """Match the sealed runner's float32 storage before subtraction/division."""
    import numpy as np
    if not math.isfinite(floor) or floor <= 0:raise ValueError("The sealed positive scale floor is required")
    y0,x0,y1,x1 = map(int,crop_yxyx)
    selection = (0,slice(y0,y1),slice(x0,x1))
    result = {name:np.asarray(stages[name][selection],dtype=np.float32) for name in ("A","M")}
    result["Spread"] = np.sqrt(stages["variance"][selection]).astype(np.float32)
    result["Score"] = (result["A"]-result["M"])/np.maximum(result["Spread"],floor)
    if any(array.dtype != np.float32 or not np.isfinite(array).all() for array in result.values()):
        raise ValueError("Streaming score stages must be finite float32")
    return result


def compare_stages(actual: Mapping[str, Any], expected: Mapping[str, Any], *, tolerance: Mapping[str, Any] = FLOAT32_TOLERANCE) -> dict[str, Any]:
    """Pixelwise float32 agreement with frozen bounds, never adjusted to results."""
    import numpy as np
    if set(actual) != {"A","M","Spread","Score"} or set(expected) != set(actual):
        raise ValueError("Compare all four sealed numeric stages")
    rows = {}
    for name in actual:
        first,second = np.asarray(actual[name]),np.asarray(expected[name])
        if first.shape != second.shape or first.dtype != np.float32 or second.dtype != np.float32:
            raise ValueError("Stage shape or float32 storage mismatch")
        difference = np.abs(first.astype(np.float64)-second.astype(np.float64))
        allowed = float(tolerance["atol"])+float(tolerance["rtol"])*np.abs(second.astype(np.float64))
        finite = bool(np.isfinite(first).all() and np.isfinite(second).all())
        violations = int(np.count_nonzero(~np.isfinite(difference)|(difference > allowed)))
        rows[name] = {"pass":finite and violations == 0,"pixel_count":int(first.size),"violation_count":violations,
            "maximum_absolute_error":float(difference.max()) if finite else None,
            "maximum_fraction_of_allowed_error":float(np.max(difference/allowed)) if finite else None}
    return {"pass":all(row["pass"] for row in rows.values()),"tolerance":dict(tolerance),"stages":rows}


def compare_candidates(actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require the same coordinate/frame proposals; score values use fixed bounds."""
    key = lambda row:(int(row["source_frame_ui"]),int(row["y_px"]),int(row["x_px"]))
    a,b = {key(row):float(row["score"]) for row in actual},{key(row):float(row["score"]) for row in expected}
    if len(a) != len(actual) or len(b) != len(expected):raise ValueError("Duplicate frame/pixel NMS proposal")
    if any(not math.isfinite(score) for score in (*a.values(),*b.values())):raise ValueError("Nonfinite candidate score")
    score_fail = [coordinate for coordinate in a.keys() & b.keys()
        if abs(a[coordinate]-b[coordinate]) > FLOAT32_TOLERANCE["atol"]+FLOAT32_TOLERANCE["rtol"]*abs(b[coordinate])]
    return {"pass":set(a) == set(b) and not score_fail,"streamed_proposal_count":len(actual),"sealed_proposal_count":len(expected),
        "streamed_only_frame_y_x":[list(key) for key in sorted(a.keys()-b.keys())],
        "sealed_only_frame_y_x":[list(key) for key in sorted(b.keys()-a.keys())],
        "score_disagreement_frame_y_x":[list(key) for key in sorted(score_fail)],
        "comparison_unit":"source frame plus integer pixel; not biological identity or event"}


def _contract(study_root: Path) -> dict[str, Any]:
    """Bind source and four frozen references before any CUDA worker starts."""
    protocol_path = study_root/"protocol.json";preflight_path = study_root/"preflight.json"
    protocol = _read(protocol_path)
    if _sha(protocol_path) != _read(preflight_path)["protocol_sha256"]:raise ValueError("Study protocol changed")
    completion = study_root/"computation_complete.json"
    if _read(completion).get("status") != "PASS":raise ValueError("Full matrix computation must finish before latency benchmarking")
    source_records = [_binding(path) for path in (protocol_path,preflight_path,completion)]
    for record in [protocol["source"],*protocol["code_bindings"]]:_verify(record);source_records.append(dict(record))
    dataset = study_root/"datasets"/"real"
    prepared_path = dataset/"prepared.json";prepared = _read(prepared_path)
    for name in ("input","metadata"):_verify(prepared[name]);source_records.append(dict(prepared[name]))
    source_records.append(_binding(prepared_path))
    metadata = _read(dataset/"metadata.json")
    frames = list(map(int,metadata["source_frames_ui"]))
    if frames != list(range(1400,2360)) or frames[SOURCE_ROWS[0]:SOURCE_ROWS[1]] != list(range(1960,2088)):
        raise ValueError("Unexpected real source map for the fixed latency interval")
    cases = []
    for spec in central_specifications():
        identifier = f"R7.5_T{spec['time_scale_ms']:g}_S1_n9_joint"
        cell = study_root/"cells"/"real"/identifier
        seal_path = cell/"sealed.json";seal = _read(seal_path)
        if seal.get("status") != "SEALED_BEFORE_LABEL_JOIN":raise ValueError("Expected frozen full-run stages")
        source_records.append(_binding(seal_path))
        for record in [seal["calibration"],seal["audit_candidates"],*[seal["stages"][key] for key in ("A","M","Spread","Score")]]:
            _verify(record);source_records.append(dict(record))
        operating = _read(seal["calibration"]["path"])
        if any(not math.isfinite(float(operating[key])) for key in ("threshold","scale_floor")) or operating["scale_floor"] <= 0:
            raise ValueError("Finite frozen threshold and positive scale floor are required")
        if seal["kernel"]["spec"] != spec:raise ValueError("Sealed central kernel differs from the fixed benchmark matrix")
        if int(seal["kernel"]["maximum_lag_frames"]) >= DISCARDED_FRAMES:
            raise ValueError("Discarded startup does not exceed the kernel's complete causal history")
        cases.append({"spec":spec,"spec_id":identifier,"kernel":seal["kernel"],"operating_point":operating,
            "stage_bindings":{key:seal["stages"][key] for key in ("A","M","Spread","Score")},
            "candidate_binding":seal["audit_candidates"],"cell_seal":_binding(seal_path)})
    # Include dependencies not already bound by the study, including this new source.
    module_path = Path(__file__).resolve()
    code_paths = [module_path,module_path.parents[2]/"algorithms"/"gamma_spatiotemporal.py",
        module_path.with_name("two_stencil_evaluation.py"),module_path.with_name("evaluation.py")]
    code = [_binding(path) for path in code_paths]
    unique = {record["path"]:record for record in source_records}
    return {"schema_version":1,"study_root":str(study_root),"sources":list(unique.values()),"code":code,
        "source_input":dict(prepared["input"]),"source_frames_ui":frames[SOURCE_ROWS[0]:SOURCE_ROWS[1]],
        "source_array_rows_half_open":list(SOURCE_ROWS),"measured_source_frames_ui":frames[SOURCE_ROWS[0]+DISCARDED_FRAMES:SOURCE_ROWS[1]],
        "crop_yxyx":metadata["evaluation_box_yxyx"],"original_source_offset_xy":metadata["original_source_offset_xy"],
        "discarded_frames":DISCARDED_FRAMES,"period_s":PERIOD_S,"repetitions":REPETITIONS,
        "worker_timeout_s":WORKER_TIMEOUT_S,"worker_process_per_repetition":True,"device":"cuda","chunk_frames":1,
        "float32_tolerance":dict(FLOAT32_TOLERANCE),"cases":cases,"scope":dict(SCOPE)}


def _run_worker(job_path: Path) -> None:
    """One bounded repetition; called only in a separately timed CUDA process."""
    worker_started = time.perf_counter();job = _read(job_path)
    contract = job["contract"];case = job["case"]
    for record in contract["code"]:_verify(record)
    import numpy as np
    import torch
    from neurobench.algorithms.gamma_spatiotemporal import GammaSTSpec,build_kernels,describe,iter_chunks
    from .two_stencil_evaluation import extract_frame_candidates
    if not torch.cuda.is_available():raise RuntimeError("CUDA is unavailable; no CPU fallback for this latency benchmark")
    kernel = build_kernels(GammaSTSpec(**case["spec"]))
    if describe(kernel) != case["kernel"]:raise ValueError("Rebuilt benchmark kernel differs from the sealed full run")
    source = np.load(contract["source_input"]["path"],mmap_mode="r",allow_pickle=False)
    begin,end = contract["source_array_rows_half_open"]
    if source.shape != (960,340,573) or source.dtype.kind not in "fui":raise ValueError("Unexpected source input shape/storage")
    values = source[begin:end]
    measured_count = len(contract["measured_source_frames_ui"])
    y0,x0,y1,x1 = map(int,contract["crop_yxyx"])
    buffer = {key:np.empty((measured_count,y1-y0,x1-x0),dtype=np.float32) for key in ("A","M","Spread","Score")}
    proposed = [];op = case["operating_point"]
    def finalize(stages,frame):
        converted = _normalize_frame(stages,contract["crop_yxyx"],float(op["scale_floor"]))
        candidates = extract_frame_candidates(converted["Score"],source_frame_ui=frame,threshold_z=float(op["threshold"]),
            cell_id=case["spec_id"],frame_interval_ms=20.)
        return converted,candidates
    def sink(index,stages,candidates):
        for key in buffer:buffer[key][index] = stages[key]
        proposed.extend(candidates)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    setup_done = time.perf_counter()
    iterator = iter_chunks(values,kernel,chunk_frames=1,device="cuda")
    rows,startup = _measure_stream(iterator,finalize,sink,contract["source_frames_ui"],
        discarded_frames=contract["discarded_frames"],period_s=contract["period_s"])
    pipeline_done = time.perf_counter()
    reference = {key:np.load(record["path"],mmap_mode="r",allow_pickle=False)[begin+DISCARDED_FRAMES:end]
                 for key,record in case["stage_bindings"].items()}
    comparison = compare_stages(buffer,reference)
    measured_frames = set(contract["measured_source_frames_ui"])
    expected_candidates = [row for row in _read(case["candidate_binding"]["path"]) if int(row["source_frame_ui"]) in measured_frames]
    candidate_comparison = compare_candidates(proposed,expected_candidates)
    comparison_done = time.perf_counter()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = {"status":"PASS" if comparison["pass"] and candidate_comparison["pass"] else "NUMERIC_PARITY_FAILED",
        "job_sha256":_sha(job_path),"contract_sha256":_canonical(contract),"spec_id":case["spec_id"],"repetition":job["repetition"],
        "timing":timing_summary(rows,period_s=contract["period_s"]),"startup":{**startup,
            "imports_kernel_input_setup_ms":(setup_done-worker_started)*1000,
            "worker_start_to_paced_loop_end_ms":(pipeline_done-worker_started)*1000,
            "deferred_correctness_comparison_ms":(comparison_done-pipeline_done)*1000},
        "stage_comparison":comparison,"candidate_comparison":candidate_comparison,
        "buffer_bytes":sum(array.nbytes for array in buffer.values()),"gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "hardware":{"device_name":device.name,"device_total_memory_bytes":device.total_memory,
                    "torch_version":str(torch.__version__),"torch_cuda_version":torch.version.cuda,"cpu_affinity":sorted(os.sched_getaffinity(0))},
        "kernel_information_support":{"maximum_history_ms":case["kernel"]["maximum_history_ms"],
            "target_mean_lag_ms":case["kernel"]["target"]["mean_lag_ms"],"reference_mean_lag_ms":case["kernel"]["reference"]["mean_lag_ms"]},
        "source_frames_ui":contract["source_frames_ui"],"measured_source_frames_ui":contract["measured_source_frames_ui"],
        "operating_point":op,"scope":dict(SCOPE)}
    output = job_path.parent
    _table(output/"frame_timings.tsv",rows);_write(output/"proposals.json",proposed)
    _write(output/"result.json",result)
    print(json.dumps({"status":result["status"],"spec_id":case["spec_id"],"repetition":job["repetition"],"timing":result["timing"]}),flush=True)


def _invoke_worker(job_path: Path, *, timeout_s: float = WORKER_TIMEOUT_S) -> dict[str, Any]:
    if not 0 < timeout_s < 60:raise ValueError("Each worker timeout must be below60seconds")
    command = [sys.executable,"-m",__name__.replace("__main__","neurobench.experiments.gamma_ls_difference.spatiotemporal_latency"),"--worker-job",str(job_path)]
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    started = time.monotonic()
    try:
        process = subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=environment,timeout=timeout_s)
        stdout,stderr = process.stdout,process.stderr
        outcome = {"status":"EXITED","returncode":process.returncode,"worker_command":command}
    except subprocess.TimeoutExpired as error:
        # subprocess.run kills and waits for the child on timeout. No next GPU
        # repetition starts before that process has been reaped.
        stdout,stderr = error.stdout or "",error.stderr or ""
        if isinstance(stdout,bytes):stdout = stdout.decode(errors="replace")
        if isinstance(stderr,bytes):stderr = stderr.decode(errors="replace")
        outcome = {"status":"TIMEOUT","returncode":None,"worker_command":command,
            "timeout_s":timeout_s,"worker_killed_and_reaped":True}
    (job_path.parent/"stdout.log").write_text(stdout);(job_path.parent/"stderr.log").write_text(stderr)
    outcome["subprocess_wall_seconds"] = time.monotonic()-started
    _write(job_path.parent/"process.json",outcome)
    return outcome


def _complete_attempt(attempt: Path, contract_digest: str) -> dict[str, Any]:
    manifest = _read(attempt/"complete.json")
    if manifest.get("status") != "PASS" or manifest["contract_sha256"] != contract_digest:
        raise ValueError("Repetition completion disagrees with frozen benchmark contract")
    for record in manifest["artifacts"]:_verify(record)
    result = _read(attempt/"result.json")
    job = _read(attempt/"job.json")
    if (result.get("status") != "PASS" or result["job_sha256"] != _sha(attempt/"job.json") or
        result["contract_sha256"] != contract_digest or _canonical(job["contract"]) != contract_digest or
        result["spec_id"] != job["case"]["spec_id"] or result["repetition"] != job["repetition"] or
        not result["stage_comparison"]["pass"] or not result["candidate_comparison"]["pass"]):
        raise ValueError("Repetition result is not bound to its exact job")
    return result


def run(study_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Run eight serial CUDA repetitions; failed attempts remain intact."""
    study,output = Path(study_root).resolve(),Path(output_root).resolve()
    contract = _contract(study);digest = _canonical(contract)
    contract_path = output/"contract.json"
    if contract_path.exists():
        if _read(contract_path) != contract:raise ValueError("Latency contract changed; select a new output directory")
    elif output.exists() and any(output.iterdir()):raise FileExistsError("Refusing an unrelated output directory")
    else:_write(contract_path,contract)
    status_path = output/"status.json"
    if status_path.exists() and _read(status_path).get("complete"):
        for record in _read(output/"artifact_index.json")["artifacts"]:_verify(record)
        summary = _read(output/"summary.json")
        expected = {(case["spec_id"],repetition) for case in contract["cases"] for repetition in range(1,REPETITIONS+1)}
        actual = {(row["spec_id"],row["repetition"]) for row in summary["results"]}
        if summary.get("status") != "PASS" or summary["contract_sha256"] != digest or actual != expected or len(summary["results"]) != len(expected):
            raise ValueError("Completed benchmark inventory disagrees with its fixed matrix")
        for case,rep in sorted(expected):
            completed = list((output/case/f"repetition_{rep:02d}").glob("attempt_*/complete.json"))
            if len(completed) != 1:raise ValueError("Completed repetition missing or ambiguous")
            result = _complete_attempt(completed[0].parent,digest)
            if result != next(row for row in summary["results"] if (row["spec_id"],row["repetition"]) == (case,rep)):
                raise ValueError("Benchmark summary differs from a sealed repetition")
        return summary
    results = []
    _write(output/"status.json",{"status":"RUNNING","complete":False,"contract_sha256":digest})
    for case in contract["cases"]:
        for repetition in range(1,REPETITIONS+1):
            group = output/case["spec_id"]/f"repetition_{repetition:02d}"
            attempts = sorted(group.glob("attempt_*")) if group.exists() else []
            complete = [attempt for attempt in attempts if (attempt/"complete.json").exists()]
            if len(complete) > 1:raise ValueError("Ambiguous repeated complete benchmark attempts")
            if complete:
                result = _complete_attempt(complete[0],digest);results.append(result);continue
            attempt = group/f"attempt_{len(attempts)+1:02d}"
            attempt.mkdir(parents=True,exist_ok=False)
            job = {"contract":contract,"case":case,"repetition":repetition}
            _write(attempt/"job.json",job)
            process = _invoke_worker(attempt/"job.json",timeout_s=WORKER_TIMEOUT_S)
            result_path = attempt/"result.json"
            if process["status"] != "EXITED" or process["returncode"] != 0 or not result_path.exists() or _read(result_path).get("status") != "PASS":
                _write(output/"status.json",{"status":"FAILED","complete":False,"contract_sha256":digest,
                    "failed_attempt":str(attempt),"process":process,"completed_repetition_count":len(results)})
                raise RuntimeError(f"Bounded latency repetition failed; preserved evidence in {attempt}")
            _write(attempt/"complete.json",{"status":"PASS","contract_sha256":digest,
                "artifacts":[_binding(path) for path in sorted(attempt.iterdir()) if path.is_file()]})
            result = _complete_attempt(attempt,digest);results.append(result)
            print(json.dumps({"status":"LATENCY_REPETITION_COMPLETE","spec_id":case["spec_id"],"repetition":repetition,
                              "completed_repetitions":len(results),"timing":result["timing"]}),flush=True)
    summary = {"status":"PASS","complete":True,"contract_sha256":digest,"repetition_count":len(results),
        "all_stage_comparisons_pass":all(result["stage_comparison"]["pass"] for result in results),
        "all_nms_comparisons_pass":all(result["candidate_comparison"]["pass"] for result in results),
        "results":results,"scope":dict(SCOPE),"biological_metric_or_control_readiness_claim":False}
    _write(output/"summary.json",summary)
    _write(output/"status.json",{"status":"PASS","complete":True,"contract_sha256":digest,"repetition_count":len(results)})
    _write(output/"artifact_index.json",{"artifacts":[_binding(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_index.json"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--worker-job",type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job:_run_worker(args.worker_job)
    elif args.root and args.output:print(json.dumps(run(args.root,args.output),sort_keys=True))
    else:parser.error("--root and --output are required")


if __name__ == "__main__":main()
