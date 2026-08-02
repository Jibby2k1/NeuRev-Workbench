"""Collision-safe subprocess adapter for exact external CaImAn fits."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .cnmf_adapter import require_caiman_backend


def fit_caiman_movie(
    movie: np.ndarray,
    *,
    method_id: str,
    parameters: dict[str, Any],
    output_dir: str | Path,
    python_executable: str | Path | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Fit exact external CaImAn into a new, auditable artifact directory."""
    audit = require_caiman_backend("1.13.1", python_executable=python_executable)
    values = np.asarray(movie, dtype=np.float32)
    if values.ndim != 3 or not values.size or not np.isfinite(values).all():
        raise ValueError("movie must be a finite non-empty TYX array")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(str(destination)+".partial")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)
    input_path = staging/"input.npy"
    config_path = staging/"worker_config.json"
    np.save(input_path, values)
    worker_config = {
        "method_id": str(method_id),
        "n_processes": int(parameters.get("n_processes", 1)),
        "K": int(parameters["K"]), "gSig": list(map(int, parameters["gSig"])),
        "p": int(parameters.get("p", 1)), "fr": float(parameters.get("fr", 50.0)),
        "decay_time": float(parameters.get("decay_time", 0.4)),
        "min_corr": float(parameters.get("min_corr", 0.5)),
        "min_pnr": float(parameters.get("min_pnr", 2.0)),
    }
    config_path.write_text(json.dumps(worker_config, indent=2, sort_keys=True)+"\n")
    worker = Path(__file__).with_name("caiman_fit_worker.py").resolve()
    environment = dict(os.environ)
    environment.update({"MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    result_dir = staging/"result"
    completed = subprocess.run(
        [str(audit["python_executable"]), str(worker), "--input", str(input_path),
         "--config", str(config_path), "--output-dir", str(result_dir)],
        capture_output=True, text=True, check=False, timeout=int(timeout_seconds),
        env=environment,
    )
    (staging/"worker.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (staging/"worker.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"CaImAn worker failed; preserved diagnostics at {staging}")
    payload = json.loads((result_dir/"fit.json").read_text(encoding="utf-8"))
    payload["backend_audit"] = audit
    (staging/"fit.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    staging.replace(destination)
    return payload
