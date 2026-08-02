"""External-environment worker for exact CaImAn CNMF and CNMF-E fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(input_path: Path, config_path: Path, output_dir: Path) -> dict:
    from caiman.source_extraction.cnmf import cnmf

    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"method_id", "n_processes", "K", "gSig", "p", "fr",
                "decay_time", "min_corr", "min_pnr"}
    if set(config) != required:
        raise ValueError("CaImAn worker configuration fields differ")
    method_id = str(config["method_id"])
    if method_id not in {"caiman_cnmf", "caiman_cnmfe"}:
        raise ValueError("method_id must be caiman_cnmf or caiman_cnmfe")
    images = np.load(input_path, allow_pickle=False).astype(np.float32, copy=False)
    if images.ndim != 3 or not np.isfinite(images).all():
        raise ValueError("CaImAn input must be finite TYX")
    images = np.ascontiguousarray(images-images.min())
    common = dict(
        n_processes=int(config["n_processes"]), k=int(config["K"]),
        gSig=list(map(int, config["gSig"])), p=int(config["p"]),
        fr=float(config["fr"]), decay_time=float(config["decay_time"]),
        min_corr=float(config["min_corr"]), min_pnr=float(config["min_pnr"]),
        dview=None, ssub=1, tsub=1, check_nan=True, do_merge=True,
        merge_thresh=0.8, normalize_init=True,
    )
    if method_id == "caiman_cnmfe":
        common.update(method_init="corr_pnr", gnb=0, ring_size_factor=1.4,
                      low_rank_background=False)
    else:
        common.update(method_init="greedy_roi", gnb=2,
                      low_rank_background=True)
    started = perf_counter()
    model = cnmf.CNMF(**common)
    model.fit(images)
    runtime = perf_counter()-started
    estimates = model.estimates
    spatial = estimates.A.tocsc() if sparse.issparse(estimates.A) else sparse.csc_matrix(estimates.A)
    traces = np.asarray(estimates.C, dtype=np.float32)
    background_spatial = np.asarray(estimates.b, dtype=np.float32)
    background_temporal = np.asarray(estimates.f, dtype=np.float32)
    neural_matrix = spatial @ traces
    background_matrix = background_spatial @ background_temporal
    dims = images.shape[1:]
    neural = np.asarray(neural_matrix.T).reshape((len(images),)+dims, order="F")
    background = np.asarray(background_matrix.T).reshape((len(images),)+dims, order="F")
    residual = images-neural-background
    output_dir.mkdir(parents=True, exist_ok=False)
    sparse.save_npz(output_dir/"spatial_components.npz", spatial)
    np.savez_compressed(
        output_dir/"temporal_components.npz", C=traces,
        S=np.asarray(estimates.S, dtype=np.float32),
        YrA=np.asarray(estimates.YrA, dtype=np.float32), b=background_spatial,
        f=background_temporal,
    )
    np.save(output_dir/"neural_reconstruction.npy", neural.astype(np.float32))
    np.save(output_dir/"background_reconstruction.npy", background.astype(np.float32))
    np.save(output_dir/"residual.npy", residual.astype(np.float32))
    payload = {
        "schema_version": 1, "status": "caiman_fit_complete",
        "method_id": method_id, "component_count": int(traces.shape[0]),
        "shape_tyx": list(images.shape), "runtime_seconds": runtime,
        "relative_closure_error": float(np.linalg.norm(images-neural-background-residual)/max(np.linalg.norm(images), np.finfo(float).eps)),
        "relative_residual_norm": float(np.linalg.norm(residual)/max(np.linalg.norm(images-images.mean(axis=0)), np.finfo(float).eps)),
        "input_sha256": _sha256(input_path), "configuration": config,
        "accepted_component_indices": list(range(int(traces.shape[0]))),
        "rejected_component_indices": [],
        "component_quality_status": "unfiltered_worker_output_requires_development_selection",
    }
    (output_dir/"fit.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.config, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
