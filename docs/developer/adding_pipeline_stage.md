# Adding A Pipeline Stage

This guide describes the current Neurobench path for turning a pipeline idea
into a runnable Architecture Lab stage. The same stage definition is used by the
dashboard, dry-run validation, local execution, run manifests, and tests, so new
stages should follow one artifact contract from the start.

## Stage Contract

A stage should answer four questions before code is added:

- What artifact does it consume?
- What artifact does it produce?
- Which parameters should be visible and sweepable?
- Is it runnable now, externally imported, or only planned?

Common artifact names currently include `raw_video`, `highpass_video`,
`smoothed_video`, `z_stack`, `candidate_mask`, `roi_candidates`,
`ranked_candidates`, `roi_traces`, and `candidate_events`.

Prefer a CPU-safe, deterministic implementation first. GPU acceleration can be
added later, but the default test path should remain runnable in the standard
development environment.

## 1. Put Algorithm Logic In A Small Module

If the stage has nontrivial math, put the core function under
`neurobench/algorithms/`, `neurobench/discovery/`, or another focused package
before wiring it into the executor. Keep the function independent of manifests,
paths, and UI state.

For example, CFAR logic lives in `neurobench/algorithms/cfar.py`, while the
executor only handles loading arrays, calling the function, writing artifacts,
and registering provenance.

## 2. Add The Catalog Entry

Add a `PipelineStage` entry to `STAGE_CATALOG` in
`neurobench/pipeline_catalog.py`:

```python
"example_stage": PipelineStage(
    stage_id="example_stage",
    label="Example stage",
    order=45,
    default_params={"strength": 1.0},
    param_ranges={"strength": ParameterRange(minimum=0.0, maximum=10.0)},
    description="Short action-oriented description of the stage.",
),
```

Then add `_STAGE_METADATA` for dashboard grouping, artifact flow, why-to-use
text, and real-time expectations:

```python
"example_stage": {
    "availability": "implemented",
    "ui_group": "preprocessing",
    "type": "filtering",
    "input": "highpass_video",
    "output": "example_video",
    "expected_qc_outputs": ("example_frame", "example_summary_trace"),
    "why_use_it": "Explains the practical review or detection problem it helps solve.",
    "real_time_profile": {
        "mode": "streaming",
        "latency_budget_ms": 2.0,
        "stateful": False,
        "adaptive": False,
        "closed_loop_candidate": True,
    },
},
```

Add parameter help to `_PARAMETER_DOCS` for every visible parameter. The
Architecture Lab uses this text to explain what each control means and why it
matters.

Important: `availability="implemented"` makes the stage executable in
`StageRegistry`, but the executor still needs a `_STAGE_RUNNERS` entry. Do not
mark a stage as implemented until a local runner and tests exist. Otherwise
`dry_run_pipeline(..., require_executable=True)` can accept it, but
`execute_pipeline(...)` will fail with `NotImplementedError`.

Use these availability values consistently:

- `implemented`: locally executable through `neurobench.pipelines.executor`
- `external_import`: available only by importing artifacts from another tool
- `planned`: visible as a roadmap item but not executable

## 3. Wire The Executor Runner

Local stage runners live in `neurobench/pipelines/executor.py`. A runner receives
the normalized step, the artifact map produced by previous stages, and an
`ArtifactStore`. It returns the output artifact key and output path.

Minimal runner pattern:

```python
def _run_example_stage(
    step: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    store: ArtifactStore,
) -> tuple[str, Path]:
    np = _load_numpy()
    source = _require_artifact(artifacts, "highpass_video", step)
    video = _load_npy(source).astype(np.float32, copy=False)
    strength = float(step["params"].get("strength", 1.0))

    result = (video * strength).astype(np.float32, copy=False)
    out = store.artifact_path("preprocessing", "example_video.npy")
    np.save(out, result)
    store.register_file(
        out,
        artifact_id="example_video.v1",
        kind="example_video",
        producer_stage=str(step["stage_id"]),
        summary={"shape": [int(value) for value in result.shape], "strength": strength},
    )
    return "example_video", out
```

Then add it to `_STAGE_RUNNERS`:

```python
_STAGE_RUNNERS = {
    ...
    "example_stage": _run_example_stage,
}
```

Use stable artifact filenames and stable artifact IDs. The manifest is the
source of truth for later reports, exports, and dashboard comparisons.

## 4. Test The Stage

Add focused tests at three levels:

- Algorithm test: validates the pure function on a tiny synthetic array.
- Dry-run test: validates parameter defaults, ranges, and artifact flow.
- Execution test: runs a synthetic pipeline and checks the manifest plus output
  artifact.

Minimal execution test pattern:

```python
def test_synthetic_pipeline_executes_example_stage(self):
    require_numpy()
    from neurobench.data.synthetic import generate_synthetic_calcium_dataset
    from neurobench.pipelines.executor import dry_run_pipeline, execute_pipeline

    dataset = generate_synthetic_calcium_dataset(include_impulse_artifact=False)
    with tempfile.TemporaryDirectory() as tmp:
        paths = dataset.write(Path(tmp) / "fixture", dataset_id="synthetic_example")
        spec = {
            "schema_version": 1,
            "dataset_id": "synthetic_example",
            "run_id": "synthetic_pipeline_example",
            "pipeline": [
                {"id": "source", "stage_id": "source_video_import", "params": {"source": paths["video"]}},
                {"id": "highpass", "stage_id": "temporal_highpass_gaussian", "params": {"sigma_frames": 2.0}},
                {"id": "example", "stage_id": "example_stage", "params": {"strength": 0.5}},
            ],
        }

        plan = dry_run_pipeline(spec, validate_artifacts=True)
        result = execute_pipeline(spec, run_root=Path(tmp) / "run")

        self.assertIn("example_video", plan["available_artifacts"])
        self.assertEqual(result["status"], "completed")
```

The current tested minimal real stage pattern is
`source_video_import -> temporal_highpass_gaussian -> spatial_gaussian ->
gamma_cfar`. See
`tests/test_pipeline_executor.py::test_synthetic_pipeline_executes_spatial_gaussian_and_gamma_cfar`
for the live reference.

## 5. Validate From The CLI

Use dry-run first, then execute:

```bash
python -m neurobench.cli.main run dry-run pipeline_spec.json --validate-artifacts
python -m neurobench.cli.main run execute pipeline_spec.json --run-root Outputs/runs/example_stage
```

For parameter sweeps, keep the same artifact contract and use:

```bash
python -m neurobench.cli.main run sweep sweep_spec.json --run-root Outputs/sweeps/example_stage
```

## 6. Update User-Facing Surfaces

If the stage should appear in the Architecture Lab or Process Lab, confirm that
the catalog entry includes:

- a clear `label`
- concise `description`
- useful `why_use_it`
- complete `parameter_docs`
- realistic `real_time_profile`
- `expected_qc_outputs` when the stage can produce previewable intermediate data
  or a diagnostic summary. Locally runnable stages are expected to declare at
  least one Process Lab output so dry-run plans and the dashboard can show a
  complete inspection checklist.

The Review page should only depend on stable run artifacts, not private runner
state.

## Common Pitfalls

- Marking a stage `implemented` before adding `_STAGE_RUNNERS`.
- Changing an artifact kind or filename after reports or exports already depend
  on it.
- Hiding parameter meaning in code instead of `_PARAMETER_DOCS`.
- Requiring GPU, Fiji, or lab data for the default test path.
- Returning broad masks or merged components without summary metrics that let a
  reviewer see what changed.
- Adding dashboard controls without a matching pipeline spec parameter.

## Completion Checklist

Before calling a stage ready:

- The pure algorithm has a focused test.
- The catalog entry has metadata, parameter docs, and real-time profile.
- `dry_run_pipeline(..., validate_artifacts=True)` passes on a synthetic spec.
- `execute_pipeline(...)` writes the expected artifact and manifest record.
- The artifact summary contains enough context for reports and comparison.
- The README or relevant workflow docs mention the stage when it changes a user
  workflow.
