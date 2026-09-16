"""Idealized centered shape limits from frozen spatial stencil weights.

This is a mathematical kernel diagnostic. It does not load movies, truth,
calibrations or candidate files, and does not run the detector or evaluate it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import gaussian_filter


SHAPES = (("neural_sigma1", 1.0, 4.0),
          ("neural_sigma2", 2.0, 4.0),
          ("artifact_sigma3", 3.0, None))


def _binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _verify(record):
    if _binding(record["path"]) != record:
        raise RuntimeError(f"Source binding changed: {record['path']}")


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _kernel(value):
    value = np.asarray(value, dtype=np.float64)
    if (value.ndim != 2 or value.shape[0] != value.shape[1]
            or value.shape[0] % 2 != 1 or not np.isfinite(value).all()
            or (value < 0).any()
            or not math.isclose(float(value.sum()), 1., rel_tol=0, abs_tol=2e-14)):
        raise ValueError("Weights must be an odd square of finite nonnegative unit mass")
    return value


def conditioned_footprint(sigma_px, weight_half_width_px, *, circular_truncate_sigma=None):
    """Sample a unit-peak source and the fixed sigma1/truncate4 pre-Gaussian.

    The raw square includes the complete four-pixel conditioning halo for all
    requested stencil samples. The untruncated source is evaluated everywhere
    needed by the finite filters; no four-sigma artifact cutoff is introduced.
    Float64 evaluates the generator's mathematical shape, not its float32 bytes.
    """
    if not math.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError("sigma_px must be finite and positive")
    if (isinstance(weight_half_width_px, bool)
            or not isinstance(weight_half_width_px, int) or weight_half_width_px < 0):
        raise ValueError("weight_half_width_px must be a nonnegative integer")
    if circular_truncate_sigma is not None and (
            not math.isfinite(circular_truncate_sigma) or circular_truncate_sigma <= 0):
        raise ValueError("circular truncation must be positive or None")
    half = weight_half_width_px + 4
    coordinates = np.arange(-half, half + 1, dtype=np.float64)
    radius2 = coordinates[:, None] ** 2 + coordinates[None, :] ** 2
    raw = np.exp(-radius2 / (2 * sigma_px ** 2))
    if circular_truncate_sigma is not None:
        raw *= radius2 <= (circular_truncate_sigma * sigma_px) ** 2
    conditioned = gaussian_filter(raw, sigma=1., truncate=4., mode="reflect")
    return raw, conditioned


def center_shape_limit(target, reference, conditioned):
    """Return center moments and the positive-amplitude, inactive-floor limit."""
    target, reference = _kernel(target), _kernel(reference)
    u = np.asarray(conditioned, dtype=np.float64)
    if (u.ndim != 2 or u.shape[0] != u.shape[1] or u.shape[0] % 2 != 1
            or not np.isfinite(u).all() or u.shape[0] < max(target.shape[0], reference.shape[0])):
        raise ValueError("Conditioned footprint must be a finite centered odd square covering both kernels")
    center = u.shape[0] // 2

    def crop(kernel):
        half = kernel.shape[0] // 2
        return u[center-half:center+half+1, center-half:center+half+1]

    a = float(np.sum(target * crop(target), dtype=np.float64))
    ur = crop(reference)
    m = float(np.sum(reference * ur, dtype=np.float64))
    second = float(np.sum(reference * ur * ur, dtype=np.float64))
    variance = second - m * m
    stable_variance = float(np.sum(reference * (ur-m) ** 2, dtype=np.float64))
    if not math.isclose(variance, stable_variance, rel_tol=2e-12, abs_tol=2e-16):
        raise ValueError("Reference variance suffered numerical cancellation")
    if variance <= 0:
        raise ValueError("Positive reference shape variance is required for a finite shape limit")
    spread = math.sqrt(variance)
    return dict(target_center_gain=a, reference_center_mean_gain=m,
                reference_center_second_moment_gain=second,
                reference_center_variance_gain=variance,
                reference_center_spread_gain=spread,
                contrast_center_gain=a-m,
                positive_amplitude_shape_limit=(a-m)/spread)


def derive_table(target, references):
    """Compute exactly three declared centered shapes for each supplied kernel."""
    target = _kernel(target)
    if not references:
        raise ValueError("At least one reference kernel is required")
    references = {name: _kernel(value) for name, value in references.items()}
    half = max(target.shape[0] // 2, *(k.shape[0] // 2 for k in references.values()))
    arrays = {"target": target.copy()}
    arrays.update({"reference_"+name: value.copy() for name, value in references.items()})
    rows = []
    for shape_id, sigma, truncation in SHAPES:
        raw, u = conditioned_footprint(sigma, half, circular_truncate_sigma=truncation)
        arrays["raw_"+shape_id], arrays["conditioned_"+shape_id] = raw, u
        for arm, reference in references.items():
            rows.append(dict(arm_id=arm, shape_id=shape_id, raw_sigma_px=sigma,
                raw_circular_truncation_sigma=truncation, raw_peak=1.,
                center_x_px=0, center_y_px=0, target_width_px=target.shape[0],
                reference_width_px=reference.shape[0], raw_square_half_width_px=half+4,
                **center_shape_limit(target, reference, u)))
    return rows, arrays


README = """# Idealized reference shape limits

This separate mathematical diagnostic uses the nine frozen reference arrays
and the baseline target. It loads no movie, annotation, calibration or candidate
data and performs no detector run, threshold choice or classification.

For a constant background c and one fixed spatial shape, write the conditioned
input as X=c+b u, where u is the raw unit-peak footprint after the fixed spatial
Gaussian (sigma1px, truncate4). Let a=(Kt*u)(0), m=(Kr*u)(0), and
v=(Kr*u²)(0)-m². Both stencils have unit mass. Therefore A=c+b a,
M=c+b m, A-M=b(a-m), and Spread²=b²v. With b>0 and a fixed positive floor f,

    Z(b) = b(a-m) / max(b sqrt(v), f).
    lim[b→+infinity] Z(b) = (a-m)/sqrt(v), for v>0.

The ratio is already exact when b sqrt(v)>=f in this noiseless model.
If the floor scales with b, that conclusion need not hold. This derivation
requires a positive nonzero shape variance; it does not define the v=0 case.
Spread measures reference variability, including signal structure; it is not
the uncertainty of the contrast or evidence of a standard-normal score.

The neural sigma1 and sigma2 shapes use the generator's inclusive circular
radius4sigma cutoff. The sigma3 artifact is an untruncated Gaussian, matching
its generator's formula. The finite raw square includes the complete four-pixel
conditioning halo for every stencil sample, so its edge does not truncate any
contribution used in the center calculation. Shapes are centered on an integer
pixel and evaluated in float64. The stencil arrays are copied exactly; these
ideal real-valued footprint samples are not a replay of float32 movie arithmetic.

A static shape under a linear EMA can be written with an effective scalar
amplitude. A moving shape generally cannot: EMA mixes differently centered
footprints. The actual artifact has motion and fractional-pixel centers.
Noise, the experimental background gradient, other sources, finite amplitudes,
floor activation and transient history alter the finite-data score. In
particular, the reference's variance includes gradient/noise/cross terms.
The result is a center score, not a spatial maximum, NMS proposal or event
metric. Neither amplitude-independent real-data behavior nor biological
performance follows from these limits.

`shape_limits.tsv` and `shape_limits.json` contain all27 moments/limits.
`geometry_inputs.npz` contains the exact stencil arrays and all sampled raw and
conditioned shapes. `source_capsule/` preserves the generator, its focused
mathematical tests, the frozen relevant source files, protocol, preflight and
nine reference .npy files. `manifest.json` binds all outputs and sources.
Reproduce from the repository with its scientific Python environment:

    python -m neurobench.experiments.gamma_ls_difference.reference_shape_limits --root ROOT --output NEW_OUTPUT --tests-xml TESTS_XML

The destination must be new. This is an analytical supplement to the existing
study, whose simulation and media audit completion remain separate.
"""


def build_capsule(root, output=None, tests_xml=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root / "geometry_limits"
    if (root / "completion_manifest.json").exists():
        raise RuntimeError("Preserve completed study roots")
    if output.exists():
        raise FileExistsError("A new geometry diagnostic destination is required")
    protocol_path, preflight_path = root / "protocol.json", root / "preflight.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("experiment") != "gamma_reference_mean_distance_order":
        raise ValueError("Expected the frozen reference-geometry study")
    if _binding(protocol_path)["sha256"] != json.loads(preflight_path.read_text())["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    here = Path(__file__).resolve()
    repo = here.parents[3]
    test = repo / "tests/test_reference_shape_limits.py"
    dependency_names = {"gamma_spatiotemporal.py", "gamma_two_stencil.py", "control_study.py",
                        "followup_fixtures.py", "reference_fixtures.py"}
    dependencies = [b for b in protocol["code_bindings"] if Path(b["path"]).name in dependency_names]
    if {Path(b["path"]).name for b in dependencies} != dependency_names:
        raise RuntimeError("Missing frozen footprint or stencil dependency")
    for b in dependencies + protocol["kernel_bindings"]:
        _verify(b)
    from neurobench.algorithms.gamma_spatiotemporal import GammaSTSpec, build_kernels
    baseline = build_kernels(GammaSTSpec())
    references = {Path(b["path"]).stem: np.load(b["path"], allow_pickle=False)
                  for b in protocol["kernel_bindings"]}
    if len(references) != 9 or set(references) != {a["arm_id"] for a in protocol["references"]}:
        raise RuntimeError("Expected exactly the nine frozen reference kernels")
    central = references["mean1_n9"]
    if central.dtype != baseline.reference.dtype or central.shape != baseline.reference[0].shape or central.tobytes() != baseline.reference[0].tobytes():
        raise RuntimeError("Baseline builder does not exactly reproduce the central reference")
    rows, arrays = derive_table(baseline.target[0], references)
    tests = None
    if tests_xml is not None:
        tests_xml = Path(tests_xml).resolve()
        suites = [e for e in ET.parse(tests_xml).getroot().iter("testsuite")]
        tests = {key: sum(int(s.get(key, "0")) for s in suites)
                 for key in ("tests", "failures", "errors", "skipped")}
        if not suites or tests["tests"] < 1 or any(tests[k] for k in ("failures", "errors", "skipped")):
            raise RuntimeError("Focused mathematical tests must all pass")
    output.mkdir(parents=True)
    _write(output / "shape_limits.json", dict(schema_version=1, kind="idealized_kernel_geometry",
        detector_experiment=False, source_center_xy=[0, 0], score_scope="center only; no NMS",
        precision="float64 mathematical footprints; exact frozen kernel arrays",
        conditioning=dict(sigma_px=1., truncate=4., mode="reflect", complete_halo=True),
        background="constant", noise=False, motion=False, fractional_center=False,
        fixed_floor_inactive=True, label_access=False, threshold_selection=False,
        scientific_promotion=False, rows=rows))
    with (output / "shape_limits.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output / "geometry_inputs.npz", **arrays)
    (output / "README.md").write_text(README)
    sources = [_binding(here), _binding(test), _binding(protocol_path), _binding(preflight_path)] + dependencies + protocol["kernel_bindings"]
    capsule = output / "source_capsule"; capsule.mkdir()
    copies = []
    for b in sources:
        target = capsule / Path(b["path"]).name
        if target.exists():
            raise RuntimeError("Duplicate source capsule basename")
        shutil.copyfile(b["path"], target)
        actual = _binding(target)
        if (actual["sha256"], actual["size_bytes"]) != (b["sha256"], b["size_bytes"]):
            raise RuntimeError("Source capsule copy differs")
        copies.append(dict(original=b, copy=actual))
    if tests_xml is not None:
        shutil.copyfile(tests_xml, output / "focused_tests.xml")
    _write(capsule / "manifest.json", dict(status="PASS", source_copies=copies))
    import scipy
    _write(output / "validation.json", dict(status="PASS", kernel_count=9, shape_count=3,
        row_count=len(rows), exact_central_reference_parity=True,
        moments_finite=all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v, float)),
        focused_tests=tests, test_input=None if tests_xml is None else _binding(tests_xml),
        runtime=dict(numpy=np.__version__, scipy=scipy.__version__),
        validation_scope="source bindings, kernel parity and mathematical fixture tests; no media/detector evaluation"))
    _write(output / "manifest.json", dict(status="PASS", source_bindings=sources,
        interpretation="Mathematical shape limits only; no new detector evidence or scientific promotion",
        artifact_bindings=[_binding(f) for f in sorted(output.rglob("*")) if f.is_file()]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tests-xml", type=Path)
    args = parser.parse_args()
    print(build_capsule(args.root, args.output, args.tests_xml))


if __name__ == "__main__":
    main()
