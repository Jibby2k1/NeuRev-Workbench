"""Shared server-side helpers for the Neurobench workbench."""
from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from neurobench.architecture_runs import as_run_manifest
from neurobench.llm_planning import proposal_set_to_architecture_manifest, validate_proposal_set
from neurobench.pipeline_catalog import normalize_pipeline
from neurobench.workbench.materialize import materialize_virtual_roi_traces


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP_DIR = PROJECT_ROOT / "Outputs/NeuronReview/calcium_video_2/app"
DEFAULT_FIJI = Path("/home/jibby2k1/.local/bin/fiji")
MAX_LOG_LINES = 300
ALLOWED_BACKENDS = {"auto", "fiji_groovy", "python_gpu"}
GENERATION_STAGES = "all"


class GenerationJob:
    """Mutable status record for a generated workbench run."""

    def __init__(self, *, app_dir: Path, payload: dict[str, Any]) -> None:
        self.job_id = uuid.uuid4().hex[:12]
        self.app_dir = app_dir.resolve()
        self.output_app_dir = self.app_dir / "generated_runs" / safe_run_id(
            str(payload.get("run_id") or "current_review_pipeline")
        )
        self.output_root = self.output_app_dir / "pipeline_outputs"
        self.payload = dict(payload)
        self.run_id = str(payload.get("run_id") or "current_review_pipeline")
        self.dataset_id = str(payload.get("dataset_id") or self.app_dir.parent.name)
        self.backend = str(payload.get("backend") or "auto")
        self.preview = bool(payload.get("preview"))
        self.status = "queued"
        self.stage = "queued"
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.return_code: int | None = None
        self.outputs: dict[str, str] = {}
        self.error = ""
        self.log_lines: list[str] = []

    def append_log(self, line: str) -> None:
        self.log_lines.append(line.rstrip())
        if len(self.log_lines) > MAX_LOG_LINES:
            self.log_lines = self.log_lines[-MAX_LOG_LINES:]

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "backend": self.backend,
            "preview": self.preview,
            "output_app_dir": str(self.output_app_dir),
            "status": self.status,
            "stage": self.stage,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "outputs": self.outputs,
            "error": self.error,
            "log_tail": self.log_lines[-80:],
        }


class JobRegistry:
    """Thread-safe in-memory registry for local generation jobs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, GenerationJob] = {}

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [job.as_dict() for job in self.jobs.values()]

    def get(self, job_id: str) -> GenerationJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def active_for(self, app_dir: Path, run_id: str) -> GenerationJob | None:
        with self.lock:
            for job in self.jobs.values():
                if job.app_dir == app_dir.resolve() and job.run_id == run_id and job.status in {"queued", "running"}:
                    return job
        return None

    def add(self, job: GenerationJob) -> None:
        with self.lock:
            self.jobs[job.job_id] = job


JOBS = JobRegistry()


def safe_run_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
    return cleaned.strip("._") or "run"


def rel_to_app(app_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(app_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def owner_token_required() -> bool:
    return bool(os.environ.get("NEUROBENCH_OWNER_TOKEN"))


def owner_token_matches(value: str | None) -> bool:
    expected = os.environ.get("NEUROBENCH_OWNER_TOKEN")
    if not expected:
        return True
    return value == expected


def threshold_tag(value: float) -> str:
    return f"{round(value * 10):03d}"


def generation_labels(payload: dict[str, Any]) -> tuple[str, str]:
    sigma = payload.get("sigma_label")
    seed = payload.get("component_seed_z")
    grow = payload.get("component_grow_z")
    min_area = payload.get("component_min_area_px")
    if sigma is None:
        sigma = "06"
    sigma_label = f"sigma{sigma}" if not str(sigma).startswith("sigma") else str(sigma)
    if seed is not None or grow is not None or min_area is not None:
        seed_v = float(seed if seed is not None else 2.0)
        grow_v = float(grow if grow is not None else 1.1)
        min_v = int(float(min_area if min_area is not None else 4))
        preset_tag = f"run_seed{threshold_tag(seed_v)}_grow{threshold_tag(grow_v)}_min{min_v}"
    else:
        preset_tag = "balanced_seed017_grow009_min3"
    return sigma_label, preset_tag


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resolve_materialization_raw_video(app_dir: Path, payload: dict[str, Any], review_data: dict[str, Any]) -> Path:
    raw_video = payload.get("raw_video")
    if raw_video:
        raw_path = Path(str(raw_video)).expanduser()
        if not raw_path.is_absolute():
            raw_path = (PROJECT_ROOT / raw_path).resolve()
        return raw_path
    manifest_path = app_dir / "dataset_manifest.generated.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        raw = (manifest.get("paths") or {}).get("raw_video")
        if raw:
            return Path(str(raw)).expanduser().resolve()
    dataset_id = infer_dataset_id(app_dir, review_data)
    inferred = find_raw_video(review_data, dataset_id)
    if inferred is None:
        raise RuntimeError("Could not infer raw video path. Add raw_video to the materialization request.")
    return inferred.resolve()


def load_run(app_dir: Path, run_id: str) -> dict[str, Any] | None:
    path = app_dir / "architecture_runs.json"
    if not path.exists():
        return None
    manifest = as_run_manifest(load_json(path))
    return next((dict(run) for run in manifest.get("runs", []) if run.get("run_id") == run_id), None)


def run_generation_params(run: dict[str, Any] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if not run:
        return params
    for stage in normalize_pipeline(run.get("pipeline") or []):
        stage_id = stage.get("stage_id") or stage.get("op") or stage.get("name")
        stage_params = dict(stage.get("params") or {})
        if stage_id == "temporal_highpass_gaussian" and "sigma_frames" in stage_params:
            params["sigma_frames"] = stage_params["sigma_frames"]
            sigma_frames = float(stage_params["sigma_frames"])
            params["sigma_label"] = f"{int(sigma_frames):02d}" if sigma_frames.is_integer() else f"{round(sigma_frames * 10):03d}"
        if stage_id == "component_filter":
            for key in ("seed_z", "grow_z", "min_area_px", "max_area_px"):
                if key in stage_params:
                    params[f"component_{key}"] = stage_params[key]
        if stage_id in {"robust_kalman_positive_innovation", "trace_event_scoring", "candidate_event_pipeline"} and "event_threshold_z" in stage_params:
            params["event_threshold_z"] = stage_params["event_threshold_z"]
    return params


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.executable,
        "fiji": "",
        "fiji_available": False,
        "owner_token_required": owner_token_required(),
        "modules": {},
        "gpu": {"torch": False, "cuda": False, "cupy": False},
    }
    fiji = Path(shutil.which("fiji") or DEFAULT_FIJI)
    report["fiji"] = str(fiji)
    report["fiji_available"] = bool(fiji.exists())
    modules = {}
    for name in ["PIL", "numpy", "scipy", "tifffile", "torch", "cupy"]:
        modules[name] = importlib.util.find_spec(name) is not None
    report["modules"] = modules
    if modules.get("torch"):
        try:
            import torch  # type: ignore

            report["gpu"]["torch"] = True
            report["gpu"]["cuda"] = bool(torch.cuda.is_available())
            report["gpu"]["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        except Exception as exc:
            report["gpu"]["torch_error"] = str(exc)
    report["gpu"]["cupy"] = bool(modules.get("cupy"))
    return report


def infer_dataset_id(app_dir: Path, review_data: dict[str, Any] | None = None) -> str:
    params = (review_data or {}).get("parameters") or {}
    return str(params.get("datasetId") or app_dir.parent.name)


def find_raw_video(review_data: dict[str, Any], dataset_id: str) -> Path | None:
    video_name = review_data.get("video", {}).get("name")
    candidates: list[Path] = []
    if video_name:
        candidates.extend(PROJECT_ROOT.glob(f"Inputs/**/{video_name}"))
    candidates.extend(PROJECT_ROOT.glob(f"Inputs/**/*{dataset_id}*.tif"))
    candidates.extend(PROJECT_ROOT.glob(f"Inputs/**/*{dataset_id}*.tiff"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def generated_dataset_manifest(app_dir: Path, payload: dict[str, Any], *, output_app_dir: Path | None = None) -> Path:
    output_app_dir = (output_app_dir or app_dir).resolve()
    review_data_path = app_dir / "review_data.json"
    review_data = load_json(review_data_path) if review_data_path.exists() else {}
    dataset_id = str(payload.get("dataset_id") or infer_dataset_id(app_dir, review_data))
    raw_video = payload.get("raw_video")
    raw_path = Path(raw_video).expanduser() if raw_video else find_raw_video(review_data, dataset_id)
    if raw_path is None:
        raise RuntimeError("Could not infer raw video path. Add raw_video to the generation request or dataset manifest.")
    if not raw_path.is_absolute():
        raw_path = (PROJECT_ROOT / raw_path).resolve()
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "name": review_data.get("video", {}).get("name") or raw_path.name,
        "frame_rate_hz": float(payload.get("frame_rate_hz") or 5.0),
        "pixel_size_microns": float(payload.get("pixel_size_microns") or 0.5),
        "paths": {
            "raw_video": str(raw_path),
            "app_dir": str(output_app_dir),
            "review_data": str(output_app_dir / "review_data.json"),
            "annotations": str(output_app_dir / "annotations.json"),
            "architecture_runs": str(app_dir / "architecture_runs.json"),
        },
    }
    out = output_app_dir / "dataset_manifest.generated.json"
    atomic_write_json(out, manifest)
    return out


def ensure_run_record(
    app_dir: Path,
    run_id: str,
    dataset_id: str,
    status: str,
    *,
    output_app_dir: Path | None = None,
    output_root: Path | None = None,
) -> None:
    path = app_dir / "architecture_runs.json"
    manifest = as_run_manifest(load_json(path)) if path.exists() else {"schema_version": 1, "dataset_id": dataset_id, "runs": []}
    runs = list(manifest.get("runs") or [])
    run = next((item for item in runs if item.get("run_id") == run_id), None)
    if run is None:
        run = {"schema_version": 1, "run_id": run_id, "dataset_id": dataset_id, "label": run_id.replace("_", " "), "pipeline": []}
        runs.append(run)
    run["execution"] = dict(run.get("execution") or {}, status=status)
    if output_root is not None:
        run["execution"]["output_root"] = str(output_root)
    artifacts = dict(run.get("artifacts") or {})
    artifact_app_dir = (output_app_dir or app_dir).resolve()
    artifacts.update(
        {
            "review_data": rel_to_app(app_dir, artifact_app_dir / "review_data.json"),
            "app_url": rel_to_app(app_dir, artifact_app_dir / "index.html"),
            "frames": rel_to_app(app_dir, artifact_app_dir / "frames"),
        }
    )
    run["artifacts"] = artifacts
    manifest["runs"] = runs
    manifest["dataset_id"] = manifest.get("dataset_id") or dataset_id
    atomic_write_json(path, manifest)


def import_llm_proposals_into_app(app_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and merge an LLM proposal set into a workbench architecture manifest."""

    proposal = payload.get("proposal") or payload.get("proposal_set") or payload
    if not isinstance(proposal, dict):
        raise ValueError("LLM proposal import payload must be an object or contain a proposal object.")
    max_combinations = payload.get("max_combinations")
    max_combinations_int = int(max_combinations) if max_combinations is not None else None
    try:
        validated = validate_proposal_set(proposal, max_combinations=max_combinations_int)
    except Exception as exc:
        raise ValueError(f"Invalid LLM proposal set: {exc}") from exc
    path = app_dir / "architecture_runs.json"
    base = as_run_manifest(load_json(path)) if path.exists() else None
    manifest = proposal_set_to_architecture_manifest(
        validated,
        base_manifest=base,
        max_combinations=max_combinations_int,
    )
    atomic_write_json(path, manifest)
    proposal_set_id = str(validated.get("proposal_set_id") or "")
    proposal_run_ids = [
        str(run.get("run_id"))
        for run in manifest.get("runs", [])
        if (run.get("artifacts") or {}).get("proposal_set_id") == proposal_set_id
    ]
    template_ids = [
        str(template.get("id"))
        for template in manifest.get("saved_pipelines", [])
        if template.get("proposal_set_id") == proposal_set_id
    ]
    return {
        "ok": True,
        "architecture_runs": str(path),
        "proposal_set_id": proposal_set_id,
        "run_ids": proposal_run_ids,
        "saved_pipeline_ids": template_ids,
        "validation_report": validated.get("validation_report", {}),
    }


def run_process(job: GenerationJob, command: list[str], *, stage: str, env: dict[str, str] | None = None) -> int:
    job.stage = stage
    job.append_log("+ " + " ".join(shlex.quote(str(part)) for part in command))
    proc = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        job.append_log(line)
    proc.wait()
    return int(proc.returncode)


def export_known_intermediates(job: GenerationJob) -> None:
    app_dir = job.app_dir
    dataset_id = job.dataset_id
    run_id = job.run_id
    sigma_label, preset_tag = generation_labels(job.payload)
    output_root = job.output_root
    specs = [
        ("temporal_highpass_gaussian", "Temporal high-pass", output_root / "HighPass" / dataset_id / f"{dataset_id}_hp_gaussian_{sigma_label}f_float32.tif"),
        ("event_preserving_noise_suppression", "Event-preserving denoise/z", output_root / "EventPreservingNoiseSuppression" / dataset_id / f"{dataset_id}_{sigma_label}_positive_local_z_float32.tif"),
        ("robust_positive_local_z", "Robust positive local-z", output_root / "CandidateEventPipeline" / dataset_id / f"{dataset_id}_{sigma_label}_robust_positive_z_float32.tif"),
        ("component_filter", "Candidate mask", output_root / "CandidateEventPipeline" / dataset_id / f"{dataset_id}_{sigma_label}_{preset_tag}_mask.tif"),
        ("trace_event_scoring", "Temporal candidate mask", output_root / "TemporalCandidateScoring" / dataset_id / f"{dataset_id}_{sigma_label}_{preset_tag}_score_ge_050_mask.tif"),
    ]
    for stage_id, label, tif_path in specs:
        if not tif_path.exists():
            job.append_log(f"skip intermediate {stage_id}: missing {tif_path}")
            continue
        out_dir = app_dir / "generated_runs" / safe_run_id(run_id) / "intermediates" / stage_id
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools/export_intermediate_frames.py"),
            "--input-tif",
            str(tif_path),
            "--out-dir",
            str(out_dir),
            "--architecture-runs",
            str(app_dir / "architecture_runs.json"),
            "--run-id",
            run_id,
            "--stage-id",
            stage_id,
            "--label",
            label,
        ]
        code = run_process(job, cmd, stage=f"export {stage_id}")
        if code != 0:
            job.append_log(f"intermediate export failed for {stage_id} with exit code {code}")


def execute_generation_job(job: GenerationJob) -> None:
    job.status = "running"
    job.started_at = time.time()
    try:
        env = environment_report()
        if job.backend not in ALLOWED_BACKENDS:
            raise RuntimeError(f"Unsupported backend: {job.backend}")
        if job.backend == "python_gpu" and not env.get("gpu", {}).get("cuda"):
            job.status = "blocked"
            job.error = "Python GPU backend requested, but Torch CUDA is not available."
            job.append_log(job.error)
            return
        if job.backend == "python_gpu":
            job.append_log("Python GPU generation is not yet a full Review builder; using whitelisted Fiji/Groovy review generation after GPU readiness check.")
        run = load_run(job.app_dir, job.run_id)
        job.payload.update(run_generation_params(run))
        job.output_app_dir.mkdir(parents=True, exist_ok=True)
        manifest = generated_dataset_manifest(job.app_dir, job.payload, output_app_dir=job.output_app_dir)
        job.dataset_id = load_json(manifest)["dataset_id"]
        ensure_run_record(job.app_dir, job.run_id, job.dataset_id, "running", output_app_dir=job.output_app_dir, output_root=job.output_root)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools/run_neuron_review_pipeline.py"),
            "--dataset-manifest",
            str(manifest),
            "--output-root",
            str(job.output_root),
            "--architecture-runs",
            str(job.app_dir / "architecture_runs.json"),
            "--run-id",
            job.run_id,
            "--stages",
            str(job.payload.get("stages") or GENERATION_STAGES),
        ]
        code = run_process(job, cmd, stage="review pipeline")
        job.return_code = code
        if code != 0:
            job.status = "failed"
            job.error = f"review pipeline exited with code {code}"
            ensure_run_record(job.app_dir, job.run_id, job.dataset_id, "failed", output_app_dir=job.output_app_dir, output_root=job.output_root)
            return
        if job.payload.get("generate_intermediates", True):
            export_known_intermediates(job)
        ensure_run_record(job.app_dir, job.run_id, job.dataset_id, "completed", output_app_dir=job.output_app_dir, output_root=job.output_root)
        job.outputs = {
            "review_data": str(job.output_app_dir / "review_data.json"),
            "architecture_runs": str(job.app_dir / "architecture_runs.json"),
            "app_url": str(job.output_app_dir / "index.html"),
        }
        job.status = "completed"
        job.stage = "completed"
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.append_log(f"ERROR: {exc}")
        try:
            ensure_run_record(job.app_dir, job.run_id, job.dataset_id, "failed", output_app_dir=job.output_app_dir, output_root=job.output_root)
        except Exception:
            pass
    finally:
        job.finished_at = time.time()


class WorkbenchHandler(BaseHTTPRequestHandler):
    """HTTP handler for static workbench files, autosave, and generation jobs."""

    app_dir: Path
    root_dir: Path | None = None
    POST_HANDLERS = {
        ("jobs", "generate-view"): "_handle_generation_post",
        ("jobs", "generate-preview"): "_handle_generation_post",
        ("materialize-traces",): "_handle_materialize_traces_post",
        ("llm-proposals", "import"): "_handle_llm_proposal_import_post",
    }

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str, *, include_body: bool = True) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Neurobench-Owner-Token")
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any] | list[Any], *, include_body: bool = True) -> None:
        self._send(status, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n", "application/json", include_body=include_body)

    def _safe_path(self) -> Path | None:
        parsed = urlparse(self.path)
        rel = unquote(parsed.path).lstrip("/")
        if not rel:
            rel = "index.html"
        root = (self.root_dir or self.app_dir).resolve()
        candidate = (root / rel).resolve()
        if candidate == root or root not in candidate.parents:
            return None
        return candidate

    def _safe_put_path(self, parsed_path: str) -> Path | None:
        rel = unquote(parsed_path).lstrip("/")
        if self.root_dir is None:
            if rel not in {"annotations.json", "architecture_runs.json"}:
                return None
            return (self.app_dir / rel).resolve()
        parts = Path(rel).parts
        if len(parts) != 3 or parts[1] != "app" or parts[2] not in {"annotations.json", "architecture_runs.json"}:
            return None
        root = self.root_dir.resolve()
        candidate = (root / rel).resolve()
        if root not in candidate.parents:
            return None
        return candidate

    def _api_app_dir(self, parsed_path: str) -> Path | None:
        rel = unquote(parsed_path).lstrip("/")
        if self.root_dir is None:
            if not (rel == "api" or rel.startswith("api/")):
                return None
            return self.app_dir.resolve()
        parts = Path(rel).parts
        if len(parts) >= 3 and parts[1] == "app" and parts[2] == "api":
            root = self.root_dir.resolve()
            candidate = (root / parts[0] / "app").resolve()
            if root in candidate.parents and candidate.exists():
                return candidate
        return None

    def _api_tail(self, parsed_path: str) -> tuple[str, ...] | None:
        rel = unquote(parsed_path).lstrip("/")
        parts = list(Path(rel).parts)
        if "api" not in parts:
            return None
        return tuple(parts[parts.index("api") + 1 :])

    def _read_post_payload(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 2_000_000:
            self._send_json(413, {"error": "invalid request size"})
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return None
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "payload must be an object"})
            return None
        return payload

    def do_OPTIONS(self) -> None:
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:
        if self._serve_api_get():
            return
        self._serve_file(include_body=True)

    def do_HEAD(self) -> None:
        if self._serve_api_get(include_body=False):
            return
        self._serve_file(include_body=False)

    def _serve_api_get(self, *, include_body: bool = True) -> bool:
        parsed = urlparse(self.path)
        app_dir = self._api_app_dir(parsed.path)
        if app_dir is None:
            return False
        rel = unquote(parsed.path).lstrip("/")
        parts = list(Path(rel).parts)
        api_index = parts.index("api")
        tail = parts[api_index + 1 :]
        if tail == ["environment"]:
            self._send_json(200, environment_report(), include_body=include_body)
            return True
        if tail == ["jobs"]:
            self._send_json(200, {"jobs": JOBS.list()}, include_body=include_body)
            return True
        if len(tail) == 2 and tail[0] == "jobs":
            job = JOBS.get(tail[1])
            if job is None:
                self._send_json(404, {"error": "job not found"}, include_body=include_body)
                return True
            self._send_json(200, job.as_dict(), include_body=include_body)
            return True
        self._send_json(404, {"error": "unknown api endpoint"}, include_body=include_body)
        return True

    def _materialize_traces(self, app_dir: Path, payload: dict[str, Any]) -> None:
        review_path = app_dir / "review_data.json"
        if not review_path.exists():
            self._send_json(404, {"error": "review_data.json not found"})
            return
        review_data = load_json(review_path)
        annotations_path = app_dir / "annotations.json"
        if isinstance(payload.get("annotations"), dict):
            annotations = dict(payload["annotations"])
        elif annotations_path.exists():
            annotations = load_json(annotations_path)
        else:
            annotations = {"version": 3, "schema_version": 3, "settings": {}, "virtualRois": {}, "runs": {}}
        try:
            raw_path = resolve_materialization_raw_video(app_dir, payload, review_data)
            if not raw_path.exists():
                self._send_json(404, {"error": f"raw video not found: {raw_path}"})
                return
            result = materialize_virtual_roi_traces(
                review_data=review_data,
                annotations=annotations,
                raw_video_path=raw_path,
                run_id=payload.get("run_id") or (annotations.get("settings") or {}).get("activeRunId"),
                roi_ids=payload.get("roi_ids") or None,
                outer_radius_px=int(payload.get("outer_radius_px") or 15),
                neuropil_weight=float(payload.get("neuropil_weight") or 0.7),
                event_threshold_z=float(payload.get("event_threshold_z") or 2.4),
                kalman_gain=float(payload.get("kalman_gain") or 0.06),
                spike_gain=float(payload.get("spike_gain") or 0.008),
                negative_gain=float(payload.get("negative_gain") or 0.11),
            )
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})
            return
        atomic_write_json(annotations_path, result["annotations"])
        self._send_json(
            200,
            {
                "ok": True,
                "raw_video": str(raw_path),
                "materialized_ids": result["materialized_ids"],
                "annotations": result["annotations"],
            },
        )

    def _serve_file(self, *, include_body: bool) -> None:
        path = self._safe_path()
        if path is None:
            self._send(403, b"Forbidden\n", "text/plain", include_body=include_body)
            return
        if not path.exists() or not path.is_file():
            self._send(404, b"Not found\n", "text/plain", include_body=include_body)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype, include_body=include_body)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        out = self._safe_put_path(parsed.path)
        if out is None:
            self._send(404, b"Only per-dataset annotations.json and architecture_runs.json can be updated\n", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 20_000_000:
            self._send(413, b"Invalid request size\n", "text/plain")
            return
        raw = self.rfile.read(length)
        try:
            parsed_json = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send(400, f"Invalid JSON: {exc}\n".encode(), "text/plain")
            return
        if out.name == "architecture_runs.json":
            try:
                parsed_json = as_run_manifest(parsed_json)
            except Exception as exc:
                self._send(400, f"Invalid architecture run manifest: {exc}\n".encode(), "text/plain")
                return
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.tmp")
        tmp.write_text(json.dumps(parsed_json, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, out)
        self._send(200, b'{"ok":true}\n', "application/json")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        app_dir = self._api_app_dir(parsed.path)
        if app_dir is None:
            self._send_json(404, {"error": "unknown api endpoint"})
            return
        tail = self._api_tail(parsed.path)
        handler_name = self.POST_HANDLERS.get(tail or ())
        if handler_name is None:
            self._send_json(404, {"error": "unknown api endpoint"})
            return
        if not owner_token_matches(self.headers.get("X-Neurobench-Owner-Token")):
            self._send_json(401, {"error": "owner token required"})
            return
        payload = self._read_post_payload()
        if payload is None:
            return
        getattr(self, handler_name)(app_dir, payload, tail or ())

    def _handle_materialize_traces_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        self._materialize_traces(app_dir, payload)

    def _handle_llm_proposal_import_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        try:
            self._send_json(200, import_llm_proposals_into_app(app_dir, payload))
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_generation_post(self, app_dir: Path, payload: dict[str, Any], tail: tuple[str, ...]) -> None:
        if tail == ("jobs", "generate-preview"):
            payload["preview"] = True
            payload.setdefault("stages", "high-pass,event-denoise,candidates,temporal-scoring,review-data,workbench")
        backend = str(payload.get("backend") or "auto")
        if backend not in ALLOWED_BACKENDS:
            self._send_json(400, {"error": f"unsupported backend: {backend}"})
            return
        run_id = str(payload.get("run_id") or "current_review_pipeline")
        if not payload.get("force"):
            active = JOBS.active_for(app_dir, run_id)
            if active is not None:
                self._send_json(409, {"error": "generation already running for this run", "job": active.as_dict()})
                return
        job = GenerationJob(app_dir=app_dir, payload=payload)
        JOBS.add(job)
        thread = threading.Thread(target=execute_generation_job, args=(job,), daemon=True)
        thread.start()
        self._send_json(202, job.as_dict())


def configure_workbench_handler(
    *,
    app_dir: Path = DEFAULT_APP_DIR,
    root_dir: Path | None = None,
) -> tuple[type[WorkbenchHandler], Path]:
    """Validate serving roots and configure the shared handler class."""

    if root_dir:
        root_dir = root_dir.resolve()
        if not (root_dir / "index.html").exists():
            raise SystemExit(f"index.html not found in {root_dir}")
        WorkbenchHandler.root_dir = root_dir
        WorkbenchHandler.app_dir = root_dir
        return WorkbenchHandler, root_dir
    app_dir = app_dir.resolve()
    if not (app_dir / "index.html").exists():
        raise SystemExit(f"index.html not found in {app_dir}")
    WorkbenchHandler.root_dir = None
    WorkbenchHandler.app_dir = app_dir
    return WorkbenchHandler, app_dir


def create_workbench_server(
    *,
    app_dir: Path = DEFAULT_APP_DIR,
    root_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> tuple[ThreadingHTTPServer, Path]:
    handler, served = configure_workbench_handler(app_dir=app_dir, root_dir=root_dir)
    return ThreadingHTTPServer((host, port), handler), served


def serve_workbench(
    *,
    app_dir: Path = DEFAULT_APP_DIR,
    root_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server, served = create_workbench_server(app_dir=app_dir, root_dir=root_dir, host=host, port=port)
    print(f"Serving {served}")
    print(f"Open http://{host}:{port}/")
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the neuron workbench with local autosave.")
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR)
    parser.add_argument("--root-dir", type=Path, default=None, help="Serve a multi-dataset Outputs/NeuronReview root.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve_workbench(app_dir=args.app_dir, root_dir=args.root_dir, host=args.host, port=args.port)
