"""Static technical fixtures for inspecting direct/serial LS mechanisms.

The batch axis indexes independent synthetic spatial scenes, not time. These
diagnostics contain no detection thresholds, candidate matching, event-level
accuracy, biological labels, or scientific-promotion decision.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from neurobench.algorithms.gamma_two_stencil import (
    TwoStencilSpec, two_stencil_local_standardization,
)


SEED = 20260910
FIXTURE_FLOOR = 0.1
STAGES = {"A": "target_response", "M": "reference_mean", "sigma": "reference_std", "contrast": "contrast", "Z": "values"}


@dataclass(frozen=True)
class FixtureScene:
    scene_id: str
    values: np.ndarray
    background: np.ndarray
    description: str
    components: tuple[dict[str, Any], ...] = ()


def build_fixture_scenes(size: int = 64, seed: int = SEED) -> tuple[FixtureScene, ...]:
    """Eight deterministic fields in arbitrary input units; no biological truth."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 32:
        raise ValueError("fixture size must be an integer >=32")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("fixture seed must be a nonnegative integer")
    yy, xx = np.indices((size, size), dtype=np.float64)
    center = size // 2
    background = np.full((size, size), 4.0, dtype=np.float64)

    def component(x, y, sigma, amplitude):
        return {"x_px": int(x), "y_px": int(y), "sigma_px": float(sigma), "amplitude": float(amplitude)}

    def source_scene(name, components, description):
        image = background.copy()
        for part in components:
            image += part["amplitude"] * np.exp(
                -((xx - part["x_px"]) ** 2 + (yy - part["y_px"]) ** 2) / (2 * part["sigma_px"] ** 2)
            )
        return FixtureScene(name, image, background.copy(), description, tuple(components))

    gradient = 4.0 + 0.025 * (xx - (size - 1) / 2) + 0.015 * (yy - (size - 1) / 2)
    return (
        FixtureScene("constant", background.copy(), background.copy(), "Constant field of4 fixture units."),
        source_scene("compact_positive", [component(center, center, 1.5, 1)], "One positive compact Gaussian addition."),
        source_scene("compact_negative", [component(center, center, 1.5, -1)], "One negative compact Gaussian addition; sign-control partner."),
        source_scene("two_nearby", [component(center - 3, center, 1.5, 1), component(center + 3, center, 1.5, .8)], "Two imposed component centers6px apart; the combined field is not separately labeled as two detections."),
        source_scene("extended_positive", [component(center, center, 8, 1)], "Extended positive Gaussian field, sigma8px."),
        FixtureScene("gradient", gradient, gradient.copy(), "Linear background gradient; no imposed source component."),
        FixtureScene("noise", background + np.random.default_rng(seed).normal(0, .15, (size, size)), background.copy(), "Seeded Gaussian noise, standard deviation0.15 fixture units; not biological negative truth."),
        source_scene("near_boundary", [component(2, 2, 1.5, 1)], "Positive Gaussian centered at(2,2), partially outside the finite image."),
    )


def fixture_cells(include_uniform: bool = False) -> tuple[tuple[str, TwoStencilSpec], ...]:
    """Six Gamma controls, optionally six matched uniform-reference controls."""
    families = ("gamma", "uniform") if include_uniform else ("gamma",)
    return tuple(
        (f"{design}_{family}_g{guard}", TwoStencilSpec(
            design=design, reference_family=family, guard_radius_px=float(guard),
            scale_floor=FIXTURE_FLOOR, epsilon=1e-6,
        ))
        for family in families for design in ("point", "direct", "serial") for guard in (0, 7)
    )


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(scene: FixtureScene, cell_id: str, stages: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row: dict[str, Any] = {"scene_id": scene.scene_id, "cell_id": cell_id,
        "interpretation": "descriptive_static_fixture_response_not_detection_accuracy",
        "all_stages_finite": all(bool(np.isfinite(value).all()) for value in stages.values())}
    for name, values in stages.items():
        row[name + "_min"] = float(values.min())
        row[name + "_max"] = float(values.max())
        row[name + "_rms"] = float(np.sqrt(np.mean(values ** 2)))
    center_rows = []
    for index, part in enumerate(scene.components):
        x, y = part["x_px"], part["y_px"]
        input_change = float(scene.values[y, x] - scene.background[y, x])
        center_rows.append({
            "scene_id": scene.scene_id, "cell_id": cell_id, "component_index": index,
            "constructed_center_x_px": x, "constructed_center_y_px": y,
            "component_amplitude": part["amplitude"], "expected_polarity": 1 if part["amplitude"] > 0 else -1,
            "observed_Z_sign_at_center": int(np.sign(stages["Z"][y, x])),
            "sign_matches_constructed_component": bool(np.sign(stages["Z"][y, x]) == np.sign(part["amplitude"])),
            "input_minus_background_at_center": input_change,
            "A_minus_background_at_center": float(stages["A"][y, x] - scene.background[y, x]),
            "M_minus_background_at_center": float(stages["M"][y, x] - scene.background[y, x]),
            "target_response_relative_to_input_change_at_center": float((stages["A"][y, x] - scene.background[y, x]) / input_change),
            **{name + "_at_center": float(values[y, x]) for name, values in stages.items()},
            "interpretation": "response_at_constructed_component_center_not_individual_source_recovery",
        })
    if scene.components:
        sign = 1 if scene.components[0]["amplitude"] > 0 else -1
        # One global extremum is a spatial-response diagnostic, not multi-source recall.
        y, x = np.unravel_index(np.argmax(sign * stages["Z"]), stages["Z"].shape)
        row.update(signed_global_extremum_x_px=int(x), signed_global_extremum_y_px=int(y),
            signed_global_extremum_value=float(stages["Z"][y, x]),
            global_extremum_distance_to_nearest_constructed_center_px=min(
                float(np.hypot(x - p["x_px"], y - p["y_px"])) for p in scene.components),
            localization_interpretation="distance_of_one_global_signed_extremum_not_candidate_or_event_localization_accuracy")
    return row, center_rows


def _render(root: Path, scenes: tuple[FixtureScene, ...], packets: dict[str, dict[str, np.ndarray]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = []
    columns = ("X", *STAGES)
    cell_ids = tuple(packets)
    for scene_index, scene in enumerate(scenes):
        fig, axes = plt.subplots(len(cell_ids), len(columns), figsize=(15, 2.2 * len(cell_ids)), squeeze=False)
        for col, quantity in enumerate(columns):
            values = [scene.values] if quantity == "X" else [packets[cell][quantity][scene_index] for cell in cell_ids]
            lo, hi = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
            if quantity in {"contrast", "Z"}:
                bound = max(abs(lo), abs(hi), .1 if quantity == "contrast" else 1.0)
                lo, hi = -bound, bound
            elif quantity == "sigma":
                lo, hi = 0.0, max(hi, FIXTURE_FLOOR)
            elif hi - lo < 1e-8:
                lo, hi = lo - .1, hi + .1
            for row, cell_id in enumerate(cell_ids):
                ax = axes[row, col]
                values = scene.values if quantity == "X" else packets[cell_id][quantity][scene_index]
                ax.imshow(values, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
                if row == 0:
                    ax.set_title(f"{quantity}\n[{lo:.3g}, {hi:.3g}]", fontsize=9)
                if col == 0:
                    ax.set_ylabel(cell_id, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"{scene.scene_id}: {scene.description}\nStatic technical fixture; shared scale per quantity across arms; no detection threshold", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .95))
        path = root / f"scene_{scene.scene_id}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        images.append(path.name)
    fig, axes = plt.subplots(len(cell_ids), 2, figsize=(7, 2.4 * len(cell_ids)), squeeze=False)
    for row, cell_id in enumerate(cell_ids):
        for col, key in enumerate(("target_kernel", "reference_kernel")):
            kernel = packets[cell_id][key]
            display = kernel
            if kernel.shape == (1, 1):
                # A 1x1 imshow would fill the panel and falsely resemble a box
                # average. Display the delta on the common reference lattice.
                display = np.zeros_like(packets[cell_id]["reference_kernel"])
                display[display.shape[0] // 2, display.shape[1] // 2] = kernel[0, 0]
            ax = axes[row, col]
            ax.imshow(display, cmap="gray", vmin=0, vmax=float(kernel.max()), interpolation="nearest")
            ax.set_title(f"{cell_id}: {key}\nnormalized mass={kernel.sum():.6f}; max={kernel.max():.4g}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Actual discrete kernels; each heatmap uses its labeled weight range", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(root / "kernels.png", dpi=110)
    plt.close(fig)
    return images + ["kernels.png"]


def run_fixture_diagnostics(
    output_dir: str | Path, *, device: str = "cpu", include_uniform: bool = False,
    render: bool = True, size: int = 64, seed: int = SEED,
) -> dict[str, Any]:
    """Write a new immutable technical fixture packet; caller schedules resources."""
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError("fixture diagnostics require a new output root")
    scenes = build_fixture_scenes(size=size, seed=seed)
    cells = fixture_cells(include_uniform=include_uniform)
    root.mkdir(parents=True)
    inputs = np.stack([scene.values for scene in scenes])
    _npz(root / "inputs.npz", X=inputs, background=np.stack([scene.background for scene in scenes]))
    metadata = {
        "schema_version": 1, "purpose": "static_technical_operator_fixtures",
        "seed": seed, "shape_scene_y_x": list(inputs.shape),
        "batch_axis": "independent_static_scenes_not_time", "dtype": "float64",
        "input_units": "arbitrary_fixture_units", "device": device,
        "scale_floor": FIXTURE_FLOOR, "scale_floor_role": "fixed_fixture_unit_stabilizer_not_estimated_from_biological_data",
        "epsilon": 1e-6, "thresholding_or_nms_applied": False,
        "target_scale_rule": "discrete_radial_second_moment_match_to_Gaussian_sigma1_truncate4_on_31px_square",
        "scenes": [{"scene_id": s.scene_id, "description": s.description, "components": list(s.components)} for s in scenes],
        "cells": [{"cell_id": cell_id, "spec": asdict(spec)} for cell_id, spec in cells],
        "scientific_promotion": False, "biological_accuracy_claimed": False,
        "source_hashes": {"fixture_module": _sha256(Path(__file__)), "operator_module": _sha256(Path(__file__).resolve().parents[2] / "algorithms/gamma_two_stencil.py")},
    }
    _json(root / "protocol.json", metadata)
    packets = {}
    metrics, center_metrics = [], []
    for cell_id, spec in cells:
        result = two_stencil_local_standardization(inputs, spec, device=device, chunk_frames=1)
        packet = {name: getattr(result, attribute).cpu().numpy() for name, attribute in STAGES.items()}
        packet.update(target_kernel=result.target_kernel.cpu().numpy(), reference_kernel=result.reference_kernel.cpu().numpy())
        _npz(root / f"stages_{cell_id}.npz", **packet)
        _json(root / f"operator_{cell_id}.json", result.diagnostics)
        packets[cell_id] = packet
        for scene_index, scene in enumerate(scenes):
            row, centers = _metrics(scene, cell_id, {name: packet[name][scene_index] for name in STAGES})
            metrics.append(row)
            center_metrics.extend(centers)
        del result
    _json(root / "stage_metrics.json", metrics)
    _json(root / "constructed_center_metrics.json", center_metrics)
    _tsv(root / "stage_metrics.tsv", metrics)
    _tsv(root / "constructed_center_metrics.tsv", center_metrics)
    constant_rows = [row for row in metrics if row["scene_id"] == "constant"]
    constant_contrast_ok = all(max(abs(row["contrast_min"]), abs(row["contrast_max"])) < 1e-8 for row in constant_rows)
    finite = all(row["all_stages_finite"] for row in metrics)
    images = _render(root, scenes, packets) if render else []
    (root / "README.md").write_text(
        "# Two-stencil static fixture diagnostics\n\n"
        "These are constructed spatial fields in arbitrary units, not a biological detection benchmark. "
        "The batch axis indexes independent scenes, not time. No threshold, NMS, event-matching, precision, recall, or F1 is computed.\n\n"
        "The fixed floor is0.1 fixture units and epsilon is1e-6; neither is calibrated on biological data. "
        "Serial reference moments use A; direct/point reference moments use X. Each convolution renormalizes over valid image pixels. "
        "Construction-center ratios and the distance of one signed global extremum are descriptive response diagnostics, not source-recovery estimates.\n\n"
        "NPZ files contain exact float64 inputs, kernels, A, M, sigma, contrast, and Z. "
        "Scene figures share a display range within each quantity across arms; kernel plots label their individual ranges. "
        "Numerical validation does not complete visual QA or promote the biological method.\n"
    )
    artifacts = [{"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
                 for path in sorted(root.iterdir()) if path.is_file()]
    validation = {
        "status": "NUMERICAL_DIAGNOSTICS_PASS" if finite and constant_contrast_ok else "FAIL",
        "all_stages_finite": finite, "constant_field_contrast_pass": constant_contrast_ok,
        "constant_field_contrast_tolerance": 1e-8,
        "constant_field_max_abs_contrast": max(max(abs(row["contrast_min"]), abs(row["contrast_max"])) for row in constant_rows),
        "scene_count": len(scenes), "cell_count": len(cells), "metric_row_count": len(metrics),
        "image_count": len(images), "visual_qa": "pending" if render else "not_rendered",
        "scope": "technical_static_fixtures_not_a_detection_or_biological_benchmark",
        "scientific_promotion": False, "biological_accuracy_claimed": False,
        "scientific_audit_applicability": "static_unit_fixtures_have_no_recording_timeline_or_expert_model_annotation_population; full_experiment_audits_are_separate",
        "artifacts": artifacts,
    }
    _json(root / "validation.json", validation)
    return {"output_dir": str(root), **validation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-uniform", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    result = run_fixture_diagnostics(args.output, device=args.device,
        include_uniform=args.include_uniform, render=not args.no_render)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "NUMERICAL_DIAGNOSTICS_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
