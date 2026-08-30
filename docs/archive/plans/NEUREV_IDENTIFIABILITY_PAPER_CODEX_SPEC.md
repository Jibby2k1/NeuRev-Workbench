# NeuRev Sparse-Positive Neuron Identifiability Paper Program

## Autonomous Codex implementation, analysis, validation, and manuscript-export specification

**Target repository:** NeuRev Workbench  
**Target study:** `Spon Ca Burst / 3 hindbrain to tail 488 20ms`  
**Target publication:** *Journal of Neuroscience Methods*, full Research Article  
**Primary scientific framing:** statistical characterization of shared burst dynamics, neuron-specific observability, spatial specificity, and identifiability limits under sparse-positive annotation  
**Default execution mode:** continue autonomously through every non-blocked stage; collect human decisions and request them only after all unaffected work has completed  
**Scientific status of existing numerical results:** provisional until the identity/geometry repair and protected rerun are complete

---

# 0. Codex operating directive

Execute this document as a complete research-program handoff, not as a suggestion list.

Codex must:

1. Read `AGENTS.md`, `docs/CODEBASE_NAVIGATION.md`, and `docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md` before modifying code or creating a run.
2. Preserve all user changes, ignored data, completed output roots, archived logs, and existing scientific evidence.
3. Never restart the historical grid128 sweep, widen an existing feature sweep, or launch an unrelated experiment.
4. Build a new, versioned program under the maintained `neurobench/` package.
5. Use the canonical video and all available labels, but never treat an unlabeled pixel or candidate as a negative.
6. Separate immutable spatial-site identity from potentially merged canonical-neuron identity before any trace extraction or aggregation.
7. Complete every analysis that is scientifically valid without unresolved human decisions.
8. Generate review material for unresolved identity, timing, and bounded-field annotation decisions, but defer the questions until the end of the autonomous run.
9. Stop only for a fatal scientific or operational condition defined in this document. A missing optional dataset or optional dependency is not fatal; degrade gracefully and document the omission.
10. Produce machine-readable stage summaries, validation records, source maps, manuscript macros, figures, tables, and a final aggregate review packet.
11. Keep current findings explicitly provisional until the protected rerun passes.
12. Do not describe completion as scientific success. Apply the declared continue, downgrade, and stop criteria.

This specification authorizes implementation, CPU-compatible tests, bounded smoke runs, the full canonical analysis on the available canonical recording, and optional CUDA acceleration when it does not alter the numerical contract. It does **not** authorize unrelated historical sweeps or expansion of the feature search space.

---

# 1. Scientific objective and paper thesis

The current project should not be framed primarily as a supervised neuron detector. The dataset does not support that claim because it contains one canonical recording, four shared burst intervals, sparse positive labels, incomplete negative coverage, and no independent recording-level test cohort.

The paper should instead test the following model of an event-aligned ROI trace:

\[
Y_{ibs}(\tau)
=
\mu(\tau)
+
U_s(\tau)
+
V_b(\tau)
+
\beta_s N_{ibs}(\tau)
+
E_{ibs}(\tau),
\]

where:

- \(i\) indexes an occurrence;
- \(b\) indexes a burst;
- \(s\) indexes an immutable spatial observation site;
- \(\tau\) is time relative to the event window;
- \(\mu(\tau)\) is shared recording-level burst structure;
- \(U_s(\tau)\) is site- or neuron-associated temporal deviation;
- \(V_b(\tau)\) is burst-specific deviation;
- \(N_{ibs}(\tau)\) is local annulus/common-mode activity;
- \(\beta_s\) is site-specific local-background coupling;
- \(E_{ibs}(\tau)\) is residual measurement variation.

For scalar metrics \(m_{ibs}\), use the corresponding variance-component model:

\[
m_{ibs}=\alpha+u_s+v_b+\epsilon_{ibs}.
\]

The paper must answer:

1. Which temporal patterns are real relative to matched quiet-time controls?
2. Which patterns are shared across the recording or local tissue?
3. Which measurements are spatially concentrated at the labeled site?
4. Are amplitude, SNR, timing, morphology, or detectability stable properties of particular sites or proposed neurons?
5. Does any representation preserve neuron-relevant information rather than merely detect temporal change?
6. Which conclusions remain identifiable under positive-unlabeled annotation?
7. What additional annotation would convert currently unidentified quantities, especially precision, into estimable quantities?

The intended paper-level conclusion is conditional, not predetermined:

> Sparse zebrafish fluorescence events may exhibit a strong shared burst process, heterogeneous local residuals, and repeatable site-dependent observability. Two-frame ICA may recover temporal differentiation rather than independent neuronal sources. Spatial context and incomplete-label-aware evaluation are therefore necessary to distinguish change detection from neuron-specific identification.

Codex must allow the data to weaken or reject any clause of this thesis.

---

# 2. Repository facts that constrain the implementation

Treat the following as current audit facts to verify during preflight:

- The uploaded snapshot contains one canonical recording contract and four labeled burst intervals.
- Existing external-assay artifacts report 79 inclusive labeled occurrences.
- Existing artifacts contain 27 distinct spatial sites but 26 proposed canonical identities because original `roi_015` is provisionally mapped to `roi_010`.
- `roi_010` and `roi_015` retain different coordinates, approximately `(388.694, 138.907)` and `(389.057, 142.057)`.
- `neurobench/experiments/unsupervised_ica_eval/external_assay.py` preserves both traces by keying on `(canonical_roi_id, x, y)`.
- `signature_assay.py` and `residual_signature_assay.py` currently build `geometry` dictionaries keyed only by canonical ROI identity. This silently overwrites one of the two spatial sites after the proposed merge and invalidates identity-sensitive signature aggregation.
- Existing hard-ROI documentation reports 25 target observations awaiting final adjudication, including identity, morphology, and burst-2 timing questions.
- Existing full-field scientific-feature evidence promotes a compact confirmation panel rather than another broad search: carrier, `coherence_w15`, `propagation_lag2_w15`, `radial_cs_shell`, and one variance-stabilized auxiliary lane.
- Existing two-frame ICA evidence indicates near-equivalence to signed temporal difference.
- Existing labels are sparse positives. Unmatched candidates are unknown, not false positives.
- Existing output roots must never be overwritten.
- UI frame intervals are one-based and inclusive. NumPy intervals are zero-based and half-open. Coordinates are `x=column`, `y=row`.

The implementation must verify these facts from files rather than assume them silently. Any discrepancy belongs in `PRELIGHT_FINDINGS.json` and `REVIEW_REQUIRED.md`.

---

# 3. Non-negotiable scientific rules

## 3.1 Unit of analysis

Pixels and frames are repeated measurements, not independent biological units. The main dependency hierarchy is:

```text
frame samples -> occurrence -> immutable spatial site -> recording
                         \-> burst
```

All inferential summaries must respect site and burst clustering. Do not report a frame-level or pixel-level p-value as if it represented independent neurons or animals.

## 3.2 Recording-level scope

The canonical recording is an intensive case study. It supports within-recording methodological and measurement claims. It does not support population-level claims across fish, preparations, microscopes, sessions, or laboratories.

Any other videos are an unlabeled acquisition-QC/failure-regime cohort unless they independently satisfy a frozen annotation and evaluation contract.

## 3.3 Positive-unlabeled interpretation

Allowed metrics outside an exhaustively reviewed bounded field:

- known-positive recall;
- matched-count and per-burst recall;
- candidate burden;
- proposal-union coverage;
- rank of the nearest candidate;
- timing/localization/NMS/proposal failure decomposition;
- paired trace fidelity;
- label-shift and spatial-shift statistics.

Prohibited outside an exhaustively reviewed bounded field:

- precision;
- false-positive rate;
- specificity;
- negative predictive value;
- statements that unmatched candidates are artifacts.

## 3.4 Discovery/evaluation separation

Freeze operators, feature parameters, candidate proposals, budgets, NMS rules, equivalence margins, and primary metrics before label-driven confirmatory evaluation. Existing selected features may be used as frozen candidates. Do not widen the search based on the same four bursts.

## 3.5 Amplitude preservation

Primary trace analysis must preserve original timing and amplitude. Peak alignment, per-trace maximum scaling, and amplitude normalization are secondary visualization or shape-only analyses and must be labeled accordingly.

## 3.6 Identity preservation

Trace extraction must never use `canonical_neuron_id` as the sole key. A possible merge is an aggregation hypothesis, not a geometry definition.

## 3.7 Audit completeness

Every final promoted lane must satisfy `docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`. Analytic stages that do not generate model candidates still require an equivalent compact evidence index, trace inventory, source map, validation report, and representative figures.

---

# 4. Authorization and autonomous behavior

## 4.1 Continue automatically

Continue to the next stage when:

- required inputs for that stage are present;
- the current stage passes numerical and scientific validity checks; or
- the next stage has a defined degraded mode that remains interpretable.

Do not pause merely because:

- an optional video is absent;
- a reviewer decision is pending but site-level analysis remains valid;
- a conditional functional/tensor analysis fails its entry gate;
- CUDA is unavailable;
- a secondary representation is unavailable;
- a manuscript figure remains a placeholder.

## 4.2 Aggregate human decisions at the end

Do not interrupt execution for reviewable but nonfatal questions. Generate the relevant panels and add a structured item to `DECISIONS_REQUESTED.yaml`. Continue all unaffected stages.

Human feedback is expected for:

- final ROI 010/015 identity disposition and, if merged, the consolidated geometry rule;
- burst-2 onset/peak/end adjudication;
- the remaining targeted hard-ROI rows;
- bounded-field reviewer classifications;
- author list, ethics, funding, and data-release statements;
- any proposed equivalence margin that scientific collaborators reject.

## 4.3 Fatal stop conditions

Stop the program only if one of these occurs and no documented fallback is valid:

1. The canonical video cannot be loaded or has non-finite/corrupted data.
2. Coordinate or frame conventions cannot be resolved without risking systematic misalignment.
3. The label table has duplicate observation IDs, impossible intervals, or coordinates outside the video after all known schema adapters are tried.
4. A requested output root already exists and a new versioned root cannot be allocated.
5. The identity repair cannot preserve all original observations and spatial sites.
6. Synthetic and regression tests reveal a remaining silent geometry collapse.
7. A mandatory primary statistic is mathematically undefined for the available sample structure.
8. Source hashes change during a run.
9. Disk/RAM conditions make atomic output generation unsafe.

On fatal stop, still write all common stage artifacts and a precise recovery instruction.

---

# 5. Target repository additions

Create a maintained package rather than placing paper logic in ad hoc scripts:

```text
neurobench/experiments/neuron_identifiability/
  __init__.py
  __main__.py
  cli.py
  config.py
  contracts.py
  discovery.py
  preflight.py
  identity.py
  trace_extraction.py
  trace_metrics.py
  trace_atlas.py
  acquisition_qc.py
  scalar_models.py
  spatial_specificity.py
  functional_models.py
  tensor_models.py
  measurement_phenotypes.py
  bounded_field.py
  representation_confirmation.py
  paper_exports.py
  reporting.py
  validation.py

examples/
  spon_ca_burst_neuron_identifiability_paper_v1.example.json

docs/workflows/
  spon_ca_burst_neuron_identifiability_paper.md

docs/research/
  SPON_CA_BURST_NEURON_IDENTIFIABILITY_PAPER_V1_RESULTS.md

tests/
  test_neuron_identifiability_contracts.py
  test_neuron_identifiability_identity.py
  test_neuron_identifiability_traces.py
  test_neuron_identifiability_acquisition_qc.py
  test_neuron_identifiability_statistics.py
  test_neuron_identifiability_spatial.py
  test_neuron_identifiability_tensor.py
  test_neuron_identifiability_paper_exports.py
```

Add a `paper` optional dependency group to `pyproject.toml` only for dependencies that are genuinely used:

```toml
paper = [
  "statsmodels>=0.14",
  "scikit-learn>=1.6",
]

tensor = [
  "tensorly>=0.9",
]
```

Do not make the base test suite require CUDA or tensor dependencies. Optional tests must use explicit skips with a reason.

---

# 6. Program output layout and common stage contract

Default root:

```text
Outputs/NeuronIdentifiability/
  spon_ca_burst_identifiability_paper_v1/
```

If it exists, allocate `_v2`, `_v3`, and so on. Never reuse a completed root.

Required layout:

```text
<root>/
  program_config.resolved.json
  program_state.json
  source_manifest.json
  source_hashes.json
  command_history.log
  FINAL_REPORT.md
  FINAL_METRICS.json
  FINAL_VALIDATION_REPORT.md
  FINAL_FIGURE_INDEX.md
  REVIEW_REQUIRED.md
  DECISIONS_REQUESTED.yaml
  manuscript/
    results_macros.tex
    result_source_map.json
    tables/
    figures/
    supplement/
  00_preflight/
  01_identity_geometry/
  02_label_timing_contract/
  03_trace_atlas/
  04_acquisition_qc/
  05_scalar_observability/
  06_spatial_specificity/
  07_functional_tensor/
  08_measurement_phenotypes/
  09_bounded_field_annotation/
  10_representation_confirmation/
  11_manuscript_exports/
  12_reproducibility_release/
```

Every stage directory must contain:

```text
STAGE_SUMMARY.md
METRICS.json
VALIDATION_REPORT.md
FIGURE_INDEX.md
REVIEW_REQUIRED.md
DECISIONS_REQUESTED.yaml
artifact_index.json
llm_context.json
status.json
```

`status.json` must include:

```json
{
  "schema_version": 1,
  "stage": "03_trace_atlas",
  "status": "complete|complete_degraded|blocked|failed",
  "scientific_decision": "advance|advance_degraded|hold|stop_branch|fatal_stop",
  "started_at": "ISO-8601",
  "completed_at": "ISO-8601 or null",
  "input_hashes": {},
  "output_hashes": {},
  "warnings": [],
  "blocked_dependencies": [],
  "next_stage": "04_acquisition_qc"
}
```

Write all JSON/TSV/CSV/TeX files atomically. Use deterministic filenames and stable IDs. A rerun with identical inputs and configuration must produce the same numerical tables within declared floating-point tolerances.

---

# 7. Configuration and CLI contract

## 7.1 Required CLI

```bash
PYTHON=.venv-neurobench/bin/python
[ -x "$PYTHON" ] || PYTHON=python

$PYTHON -m neurobench.experiments.neuron_identifiability preflight \
  --config examples/spon_ca_burst_neuron_identifiability_paper_v1.example.json

$PYTHON -m neurobench.experiments.neuron_identifiability run \
  --config examples/spon_ca_burst_neuron_identifiability_paper_v1.example.json

$PYTHON -m neurobench.experiments.neuron_identifiability run \
  --config ... \
  --from-stage 03_trace_atlas \
  --through-stage 08_measurement_phenotypes \
  --resume

$PYTHON -m neurobench.experiments.neuron_identifiability paper-export \
  --run-root Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1 \
  --overleaf-root paper/overleaf_jnm
```

Required flags:

- `--dry-run`
- `--resume`
- `--from-stage`
- `--through-stage`
- `--cpu-only`
- `--max-workers`
- `--memory-limit-gib`
- `--allow-provisional-labels`
- `--strict`

`--allow-provisional-labels` permits analyses that remain valid at the immutable site level. It must not silently promote provisional canonical-neuron results.

## 7.2 Input discovery order

Canonical video candidates, in order:

1. Configured explicit path.
2. `Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy`
3. A cache derived from `Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif`, with hash-verified provenance.

Label candidates, in order:

1. Configured final adjudication TSV.
2. `Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv`
3. Latest valid versioned final adjudication TSV under `Outputs/HardROIAdjudication/`.
4. Provisional adjudication draft, clearly classified as provisional.
5. `Inputs/Spon Ca Burst/labels/labels_normalized.tsv`, adapted to immutable original-site view.

Other-video candidates:

- Configured manifest;
- TIFF/NPY files under `Inputs/Spon Ca Burst/` that are not the canonical recording;
- dataset manifests already registered in the workbench.

Codex must create `input_discovery_report.json` with every candidate, selection reason, hash, dimensions, and status.

---

# 8. Central data contracts

## 8.1 Observation record

Create one validated dataclass or equivalent model used by every new analysis:

```python
@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    burst_id: int
    original_roi_id: str
    observation_site_id: str
    canonical_neuron_id: str | None
    x_px: float
    y_px: float
    geometry_kind: str
    geometry_hash: str
    original_start_frame_ui: int
    original_end_frame_ui: int
    event_onset_ui: int | None
    event_peak_ui: int | None
    event_end_ui: int | None
    include_confirmed: bool
    include_inclusive: bool
    review_status: str
    disposition: str
```

Rules:

- `observation_id` is occurrence-specific.
- `observation_site_id` is immutable across proposed identity merges. For the current data it should distinguish original ROI 010 and 015.
- `canonical_neuron_id` is an analysis grouping label, not a trace key.
- `geometry_hash` includes the spatial mask or center/radius specification.
- Original UI timing remains preserved even when adjudicated timing exists.
- No loader may discard excluded or unresolved rows without writing an exclusion table.

## 8.2 Trace key

```text
(observation_site_id, geometry_hash, video_hash, trace_channel, channel_parameters_hash)
```

A canonical-neuron identifier alone is invalid.

## 8.3 Analysis views

Every analysis must state its view:

- `original_site_original_timing` — mandatory invariant baseline;
- `original_site_adjudicated_timing` — when available;
- `canonical_proposed_original_timing` — sensitivity only while provisional;
- `canonical_confirmed_adjudicated_timing` — final confirmatory view after review;
- `canonical_inclusive_adjudicated_timing` — sensitivity view.

Do not collapse two sites into one occurrence unless an adjudicated merge and geometry rule exist. Until then, site-level results are primary.

---

# 9. Stage 00 — preflight and static repository audit

## Goal

Verify inputs, environment, output safety, current evidence, test baseline, and exact stage feasibility without changing scientific data.

## Required actions

1. Read all mandatory workflow documents.
2. Record `git status --short`, current branch/commit when available, and whether the checkout is dirty.
3. Select the Python interpreter according to `AGENTS.md`.
4. Check free disk, RAM, CPU count, active Python/GPU jobs, CUDA availability, and GPU memory.
5. Discover inputs and hash them.
6. Verify video shape, dtype, finite values, intensity range, saturation ceiling candidates, and memory-map support.
7. Load all available label tables through schema adapters.
8. Count observations, original sites, proposed canonical identities, bursts, and review states.
9. Generate a projection overlay using original spatial sites.
10. Run focused tests for:
   - unsupervised ICA evaluation;
   - hard-ROI adjudication;
   - scientific feature audit;
   - scientific audit contracts.
11. Run `pytest --collect-only -q` and classify optional dependency failures separately from code failures.
12. Inspect existing `summary.json`, `llm_context.json`, and result documentation before opening media.
13. Refuse any output collision.

## Current expected baseline to verify

The uploaded snapshot may show:

- focused tests mostly passing, with one path-existence failure because large feature-panel outputs are absent from the snapshot;
- full collection failing on an unconditional `import cupy` in `tests/test_quantized_coactivity.py` when CuPy is unavailable;
- missing final adjudication and some ignored outputs.

Do not “fix” a missing ignored scientific artifact by fabricating it. Convert path-dependent tests into fixture-driven tests where appropriate, while retaining separate integration tests guarded by explicit data-availability markers.

## Pass criteria

- Canonical video and at least the original sparse-positive labels are readable.
- All coordinates are inside the video.
- All intervals are valid after explicit frame conversion.
- Source hashes are stable across preflight.
- A new output root is allocated.
- No mandatory code test fails for a reason unrelated to missing optional data/dependency.

## Degraded continuation

If only the raw TIFF exists, build a versioned memory-mapped cache and record the conversion hash.

If final adjudication is absent, continue with immutable site-level analysis and mark canonical conclusions provisional.

If other videos are absent, skip multi-video QC and continue canonical QC.

## Fatal stop

Apply Section 4.3.

---

# 10. Stage 01 — identity and geometry repair

## Goal

Eliminate the canonical-ID geometry overwrite and establish one identity contract for all future analyses.

## Required code changes

1. Refactor the label reader so it returns both original/site identity and canonical identity.
2. Introduce `observation_site_id` and `geometry_hash`.
3. Replace code patterns such as:

```python
geometry = {row["roi_id"]: (row["x"], row["y"]) for row in rows}
```

with site-indexed geometry:

```python
geometry = {
    row.observation_site_id: Geometry(...)
    for row in rows
}
```

4. Build trace dictionaries by immutable site/geometry key.
5. Group site-level repeatability by `observation_site_id`.
6. Group canonical-neuron sensitivity analyses only after trace extraction.
7. Update `signature_assay.py`, `residual_signature_assay.py`, and any shared loader they depend on.
8. Preserve backward-compatible output columns while adding explicit:
   - `original_roi_id`;
   - `observation_site_id`;
   - `canonical_neuron_id`;
   - `geometry_hash`;
   - `analysis_view`.
9. Regenerate affected results in a new output root. Never replace `raw_signature_null_generalization_v1` or `roi_minus_annulus_signature_v1`.

## Mandatory tests

### Synthetic duplicate-canonical test

Create two spatial sites with the same canonical neuron ID and different coordinates. Assert:

- two geometries are retained;
- two traces are extracted;
- no dictionary overwrite occurs;
- site-level event windows differ when the video differs at the two sites;
- canonical grouping occurs only after extraction.

### Current ROI 010/015 regression test

When current labels are available, assert:

- 79 observation rows are preserved;
- 27 distinct `observation_site_id` values are present;
- 26 proposed `canonical_neuron_id` values are present;
- ROI 010 and ROI 015 have different geometry hashes;
- all eight ROI 010/015 occurrences remain traceable to their original site;
- no summary calls 26 values “spatial ROIs” without qualification.

### Round-trip test

Load -> serialize -> load must preserve all identities, coordinates, timing, review fields, and ordering.

## Outputs

- `identity_contract.json`
- `observation_crosswalk.tsv`
- `site_geometry.tsv`
- `canonical_grouping.tsv`
- `geometry_collision_audit.tsv`
- before/after regression metrics
- review panel showing ROI 010 and ROI 015 separately

## Pass criteria

- No silent collapse.
- All 79 occurrences and 27 sites remain addressable.
- A canonical grouping can contain multiple sites without changing trace extraction.
- Affected tests pass.

## Human decision generated, not immediately requested

Create an item asking whether ROI 010/015 are:

- two neurons;
- one neuron with ROI 010 geometry;
- one neuron with ROI 015 geometry;
- one neuron with an explicit union/manual mask;
- unresolved.

Until answered, keep site-level analysis primary and proposed canonical aggregation secondary.

---

# 11. Stage 02 — authoritative label and timing contract

## Goal

Create a versioned, non-destructive label state that separates identity, activity, timing, morphology, context, and review provenance.

## Required actions

1. Discover every label/adjudication source.
2. Validate the existing `ADJUDICATION_FIELDS` contract.
3. Build a source-precedence table without deleting lower-precedence sources.
4. Create the analysis views defined in Section 8.3.
5. Verify that every target hard-ROI row is either adjudicated or explicitly pending/provisional.
6. Produce a burst-2 timing diagnostic using label-independent local traces:
   - Raw site mean;
   - annulus;
   - ROI-minus-annulus residual;
   - signed difference;
   - frozen ICA;
   - coherence and recurrence if available.
7. Generate automated onset/peak/end suggestions using a conservative algorithm, but never write them into final adjudication fields automatically.
8. Run all later analyses with original timing and, when available, adjudicated timing.
9. Predeclare a burst-2 exclusion sensitivity analysis regardless of final adjudication.

## Timing suggestion algorithm

Use a deterministic local trace method:

- baseline window: configurable, default 20 frames before the original interval;
- robust center: median;
- robust scale: MAD × 1.4826;
- onset candidate: first sustained threshold crossing in a bounded lookback window;
- peak candidate: maximum amplitude-preserving residual within the review window;
- end candidate: return below a lower hysteresis threshold for a fixed duration;
- no detector rank or match outcome may enter the suggestion.

Store all thresholds and candidate confidence.

## Outputs

- `label_source_inventory.tsv`
- `label_view_summary.json`
- one TSV per analysis view
- `burst_2_timing_suggestions.tsv`
- one trace panel per burst-2 labeled occurrence
- `hard_roi_review_packet/`

## Pass criteria

- No original label is overwritten.
- Every analysis row records its source and view.
- Timing conversion is tested at UI/NumPy boundaries.
- All later stages can run using original-site/original-timing view.

## Degraded continuation

Pending adjudication does not block site-level analysis. Mark canonical and adjudicated-timing results provisional or unavailable.

---

# 12. Stage 03 — complete Raw-to-pipeline trace atlas

## Goal

Create the central occurrence-level evidence package requested by the research group: synchronized Raw and relevant pipeline traces at every labeled site and burst.

## Required trace channels

Mandatory:

1. Raw ROI/site mean in native intensity units.
2. Baseline-subtracted Raw.
3. `ΔF/F0` when the baseline is positive and stable; otherwise explicit `not_applicable`.
4. Robustly standardized Raw using pre-event median and MAD.
5. Local annulus mean, default radii 3-6 px.
6. Outer ring mean, default radii 7-10 px.
7. ROI-minus-annulus residual in native intensity units.
8. Annulus-minus-outer-ring spatial control.
9. Signed adjacent-frame difference.
10. Frozen two-frame ICA output with verified orientation.
11. Frozen carrier/direct score trace if available.
12. `coherence_w15` trace if reconstructable.
13. `propagation_lag2_w15` trace if reconstructable.

Optional but useful:

- motion magnitude or registration residual;
- saturation fraction;
- variance-stabilized auxiliary trace;
- local radial-shell statistics.

## Required temporal views

For every occurrence and channel:

- full-duration trace;
- padded burst-context trace;
- event-centered trace with original timing;
- event-centered trace with adjudicated timing when available;
- burst-2 lookback trace;
- no primary peak alignment.

## Required occurrence metrics

At minimum:

- baseline median, mean, MAD, slope, and autocorrelation;
- signed peak amplitude;
- absolute peak amplitude;
- robust peak SNR;
- event area under curve;
- positive area and signed area;
- rise time;
- time to peak;
- decay half-life when identifiable;
- full width at half maximum when identifiable;
- peak frame UI and zero-based index;
- Raw/pipeline peak-time offset;
- Raw/pipeline trace correlation;
- event-window and baseline-window energy;
- annulus coupling correlation and regression coefficient;
- normalized center-versus-annulus specificity;
- saturation fraction;
- boundary clipping flags;
- missing/undefined reason fields.

## Atlas figures

For every occurrence, generate:

1. Full trace page.
2. Event-centered overlay.
3. Local spatial context image with center, annulus, outer ring, and nearby labels.
4. A compact review card containing essential metrics.

Generate population summaries:

- occurrence-by-time heatmaps in native baseline-subtracted units;
- robust-z heatmaps;
- Raw, annulus, and residual event-triggered median with site-clustered intervals;
- neuron/site-by-burst amplitude and SNR matrices;
- burst-specific distributions;
- Raw-versus-pipeline fidelity plots.

## Data storage

Use one tidy occurrence-channel table plus compressed NumPy arrays:

```text
traces.npz
trace_index.tsv
occurrence_metrics.tsv
site_metrics.tsv
burst_metrics.tsv
```

Do not write thousands of redundant uncompressed arrays.

## Pass criteria

- Every included occurrence has every mandatory available channel.
- Trace counts match the selected label view.
- No trace is keyed only by canonical identity.
- All figure panels use synchronized frame conventions.
- Native-amplitude and normalized views are clearly separated.
- NaN/undefined values have documented reasons.
- Hashes link trace rows to source video, geometry, parameters, and code revision.

## Stop/downgrade criteria

If a pipeline channel is unavailable, do not reconstruct it from display-clipped media. Mark it unavailable and continue with mandatory Raw/annulus/residual/difference/ICA channels.

---

# 13. Stage 04 — acquisition physics and unlabeled video QC

## Goal

Characterize measurement noise, saturation, drift, spatial discontinuities, and recording regimes before interpreting neuron-level differences.

## Canonical recording analyses

### Signal-dependent noise

For quiet adjacent-frame pairs, fit the descriptive Poisson-Gaussian relation:

\[
\operatorname{Var}(Y_{t+1}-Y_t\mid \bar Y)
\approx 2(\sigma_r^2+\alpha \bar Y).
\]

Report:

- slope \(\alpha\);
- nonnegative intercept \(2\sigma_r^2\);
- weighted and unweighted goodness of fit;
- residual pattern;
- fit sensitivity to saturation removal;
- full, left-field, right-field, and annotated-region fits.

Do not claim physical sensor parameters unless acquisition calibration justifies them.

### Temporal structure

Calculate:

- global and field-specific traces;
- autocorrelation;
- power spectral density;
- slow drift and bleaching slope;
- frame-difference distribution;
- robust change points;
- quiet/event variance ratio;
- stationarity diagnostics in bounded windows.

### Spatial structure

Calculate:

- mean, median, variance, saturation, and temporal-coherence maps;
- row/column discontinuity scores;
- the provisional `x=286` boundary and an automated boundary scan;
- left/right field correlations;
- local effective rank;
- motion or registration proxies;
- distance-to-boundary covariates for each site.

## Other videos

For every available noncanonical video, calculate label-free QC only:

- dimensions, dtype, duration, frame period if known;
- intensity distribution;
- saturation and clipping;
- signal-dependent noise fit;
- drift;
- motion proxy;
- spatial coherence;
- effective rank;
- global burstiness;
- local transient density;
- field discontinuities;
- similarity to the canonical recording in a standardized QC feature space.

Classify videos as acquisition regimes, not biological successes/failures. Suggested labels:

- `canonical_like`;
- `low_dynamic_range`;
- `high_saturation`;
- `high_motion`;
- `strong_field_discontinuity`;
- `weak_transient_content`;
- `unresolved`.

## Outputs

- canonical QC report;
- multi-video QC table;
- noise-model tables and figures;
- field-boundary analysis;
- per-site acquisition covariates;
- acquisition-regime embedding/cluster figure only if at least five videos exist.

## Pass criteria

- All reported neuron/site metrics can be joined to acquisition covariates.
- Noise fits include residual diagnostics.
- Saturated samples are reported, not silently discarded.
- Other videos are not used as biological validation without labels.

---

# 14. Stage 05 — scalar neuron/site observability and variance decomposition

## Goal

Test whether particular spatial sites or proposed neurons exhibit repeatable amplitude, SNR, timing, spatial specificity, or detectability across bursts.

## Primary metrics

Use a prespecified compact panel:

- Raw signed peak amplitude;
- Raw robust peak SNR;
- Raw event area;
- ROI-minus-annulus peak amplitude;
- ROI-minus-annulus robust SNR;
- normalized spatial specificity;
- time to peak;
- full width at half maximum where valid;
- annulus coupling coefficient;
- known-positive match indicator at frozen budgets 20, 40, and 58.

Secondary metrics require false-discovery-rate control.

## Primary descriptive model

For each scalar metric:

\[
m_{sb}=\mu+\gamma_b+u_s+\epsilon_{sb},
\]

where burst is treated as a fixed nuisance effect for the primary conditional site ICC because there are only four bursts, and site is a random intercept when estimable.

Report:

\[
\mathrm{ICC}_{site\mid burst}
=
\frac{\sigma_s^2}{\sigma_s^2+\sigma_\epsilon^2}.
\]

Do not hide singular fits. Use the following fallback sequence:

1. REML random-intercept model.
2. Alternative optimizer and scaled metric.
3. Nonparametric variance decomposition using site and burst medians.
4. Rank-repeatability analysis only, with model marked non-identifiable.

## Rank stability

For all burst pairs with at least ten shared sites, calculate:

- Spearman correlation;
- Kendall correlation;
- bootstrap interval resampled by site;
- influence of each site;
- sensitivity excluding burst 2.

## Multiway uncertainty

Use site-clustered bootstrap as primary. Add a two-way site/burst bootstrap as a sensitivity analysis, acknowledging that four bursts limit resolution. Report leave-one-burst-out estimates explicitly.

## Evidence levels for stable observability

These are measurement claims within one recording.

### O3 — strong within-recording stability

- median pairwise Spearman correlation at least 0.70;
- at least four of six burst pairs exceed 0.60;
- conditional ICC point estimate at least 0.60;
- site-bootstrap ICC lower bound above 0.30;
- direction remains compatible after excluding burst 2.

### O2 — moderate/suggestive stability

- median pairwise Spearman at least 0.50 or ICC at least 0.40;
- no major sign reversal;
- uncertainty remains broad.

### O1 — weak/inconsistent

- isolated positive pairs but median below 0.50 or burst-2 sensitivity dominates.

### O0 — no repeatable observability detected

- median pairwise Spearman below 0.30 and ICC below 0.25, or model invalid.

These thresholds classify evidence; they are not universal biological cutoffs.

## Required figures

- site-by-burst matrices for amplitude, SNR, residual amplitude, and specificity;
- pairwise burst rank plots;
- site random-effect or shrinkage estimates;
- ICC summary with uncertainty;
- burst-2 sensitivity comparison;
- acquisition-covariate relationship plots.

## Pass criteria

- All metrics have explicit units/normalization.
- Site identity is immutable.
- Canonical-neuron analysis is a separate sensitivity view.
- Results do not treat 79 occurrences as 79 independent neurons.

---

# 15. Stage 06 — spatial specificity and local source evidence

## Goal

Determine whether event-associated activity is more spatially concentrated at the labeled site than in matched nearby tissue.

## Prespecified geometry

Primary geometry:

- center disk/square equivalent to current radius 2 px ROI mean;
- local annulus 3-6 px;
- outer ring 7-10 px.

Sensitivity geometries:

- center radii 1, 2, and 3 px;
- annulus offsets adjusted to remain nonoverlapping;
- no post-hoc choice of the best radius as the confirmatory result.

## Event-minus-baseline map

For each occurrence:

\[
D_{ib}(x,y)
=
\operatorname{mean}_{t\in W_{event}}Y_t(x,y)
-
\operatorname{median}_{t\in W_{pre}}Y_t(x,y).
\]

Calculate:

- radial activation profile;
- center-to-annulus difference and ratio;
- center-to-outer-ring difference;
- footprint radius;
- spatial compactness;
- eccentricity;
- local Moran-type autocorrelation or an equivalent bounded statistic;
- nearest-neighbor and crowding measures;
- overlap with nearby labeled sites;
- event-map repeatability across bursts.

## Primary specificity statistic

\[
S_{ib}
=
\frac{A^{center}_{ib}-A^{annulus}_{ib}}
{|A^{center}_{ib}|+|A^{annulus}_{ib}|+\varepsilon}.
\]

Report native-unit differences alongside \(S\).

## Controls

- same-site quiet-time shifts preserving duration and autocorrelation structure;
- matched spatial shifts within the same field and similar baseline intensity;
- annulus-minus-outer-ring residual;
- optional rotated/translated geometry controls that avoid known labels but remain “unknown,” not negatives.

## Superiority and equivalence

For correlation-difference analyses, predeclare `0.10` correlation units as the primary practical equivalence margin, with `0.05` and `0.15` sensitivity margins. Use paired site-level bootstrap confidence intervals and a TOST-style conclusion:

- superiority if the lower interval bound exceeds `+0.10`;
- practical equivalence if the entire interval lies within `[-0.10, +0.10]`;
- inferentially unresolved otherwise.

For amplitude specificity, calibrate the primary practical margin from quiet-time site-versus-annulus variability before event results are inspected. Freeze the margin and source calculation in the resolved configuration.

## Required language

- Failure to reject zero is not equivalence.
- Annulus subtraction may remove spatially broad biological activity as well as nuisance activity.
- A localized residual is evidence of local measurement structure, not proof of a single neuron.

## Pass criteria

- Spatial controls are intensity- and field-matched.
- Results are aggregated at site level.
- Multiple radii are reported as sensitivity, not cherry-picked.
- The ROI 010/015 geometry remains separated until adjudicated.

---

# 16. Stage 07 — conditional functional and tensor decomposition

This stage is conditionally promoted. Failure to enter it does not block the paper.

## Entry gates

Enter when:

- trace-atlas validation passes;
- at least 15 immutable sites have at least three occurrences;
- at least 10 sites are observed in all four bursts for complete-tensor analysis;
- event windows share a valid common time grid;
- no unresolved timing conversion error remains.

Otherwise write `stop_branch_insufficient_repeated_structure` and continue.

## Functional mixed-effects analysis

Use amplitude-preserving, non-peak-aligned event traces. Represent functions with a fixed B-spline or functional-PCA basis chosen using reconstruction criteria that do not optimize the biological hypothesis.

For basis coefficient \(k\):

\[
c_{sbk}=\mu_k+u_{sk}+v_{bk}+e_{sbk}.
\]

Reconstruct:

- population waveform;
- site-specific deviations;
- burst-specific deviations;
- residual waveform;
- simultaneous bootstrap bands.

Run separately for:

- Raw baseline-subtracted traces;
- annulus traces;
- ROI-minus-annulus residuals;
- difference/ICA only as change-detection comparisons.

Primary functional estimands:

- fraction of integrated functional variance attributed to site, burst, and residual;
- held-out reconstruction correlation;
- site-function repeatability;
- difference in variance attribution after annulus subtraction.

## Tensor analysis

Construct a complete tensor on sites present in all four bursts:

\[
\mathcal{X}_{sbt}.
\]

Test CP/Tucker ranks 1-4. Use nested leave-one-burst-out or masked-entry reconstruction, not in-sample variance alone.

Rank-one model:

\[
\mathcal{X}_{sbt}\approx a_s c_b h_t.
\]

Evaluate:

- held-out reconstruction error;
- variance explained;
- component stability under site bootstrap;
- factor sign/scale alignment;
- whether added components capture stable structure or noise.

## Promotion logic

### Shared rank-one structure supported

- rank 1 explains at least 70% of centered event-tensor variance;
- rank 2 improves held-out reconstruction by less than 10 percentage points;
- temporal factor cosine stability exceeds 0.90 across bootstrap runs.

### Stable richer structure supported

- rank 2 or higher improves held-out reconstruction by at least 10 percentage points;
- added temporal factors have bootstrap cosine stability at least 0.80;
- factors are not driven by one burst or one site;
- interpretation survives burst-2 exclusion.

### No stable low-rank structure

- components are unstable, boundary-driven, or do not improve held-out reconstruction.

Do not label factors as cell types without independent biological evidence.

---

# 17. Stage 08 — measurement phenotypes and known-positive recoverability

## Goal

Explain recurrent easy and difficult labeled observations using continuous measurement characteristics rather than inventing biological classes.

## Site/occurrence profile

Build a profile containing:

```text
Raw amplitude
Raw SNR
residual amplitude
residual SNR
annulus coupling
spatial specificity
radial compactness
center/ring score and confidence
crowding / nearest-neighbor distance
baseline intensity
saturation
field/boundary position
burst timing
trace repeatability
frozen detector match status and rank
```

Use “measurement phenotype” or “observability phenotype.” Never call an unsupervised cluster a neuronal subtype.

## Primary analyses

1. Recurrent hard-site table across bursts.
2. Univariable site-clustered associations with known-positive recovery.
3. Compact multivariable model only when events-per-parameter is adequate.
4. Leave-one-burst-out prediction of match status at frozen budgets.
5. Separate failure-mode models for proposal, ranking, localization, timing, and NMS where counts permit.

## Model constraints

- Outcome is known-positive recovery, not truth of unmatched candidates.
- Use at most three prespecified predictors in a primary model.
- Prefer penalized or Firth-style logistic regression when separation occurs.
- Use site-clustered uncertainty.
- Do not report training AUC as evidence.
- Report leave-one-burst-out AUC/Brier score only when each fold contains both outcomes.
- Use FDR for exploratory univariable screens.

## Exploratory visualization

A two-dimensional embedding or clustering may be produced only after the continuous analysis. It must show bootstrap stability and be labeled exploratory.

## Pass criteria

- Every predictor is computed without using the recovery label.
- Site and burst dependence are respected.
- No model equates “missed” with “not a neuron.”
- Results explicitly identify whether acquisition covariates explain apparent neuron differences.

---

# 18. Stage 09 — nonblocking exhaustive bounded-field annotation study

## Goal

Create the smallest additional annotation effort capable of estimating local precision and characterizing unmatched candidates.

This stage prepares and ingests annotation. It must not block the rest of the paper.

## Automatic region proposal

Choose a bounded region without using detector scores:

1. Restrict to the documented annotated/right acquisition field.
2. Compute the bounding box of immutable labeled sites.
3. Add a fixed margin, default 32 px.
4. Quantize to a reproducible square or rectangle no larger than a configured review burden, default 192×192 or 256×192.
5. Freeze the region before revealing model candidates.
6. If the region is too large, select a deterministic spatial tile maximizing known-label coverage per area, using labels only for coverage—not model scores.

Record the selection rule and limitations. Local precision applies only to this enriched region.

## Blinded review packet

Generate:

- Raw full-field and crop videos for all four bursts;
- matched quiet windows;
- event-minus-baseline views;
- no model overlays in the first-pass packet;
- independent reviewer tables;
- a second-pass candidate panel only after first-pass exhaustive marking is frozen.

Required reviewer classes:

- confirmed neuron;
- probable neuron;
- activity visible, identity uncertain;
- artifact;
- background;
- unresolved.

Required attributes:

- center/ring morphology;
- isolated/crowded context;
- visibility confidence;
- onset/peak/end;
- spatial-boundary confidence;
- reviewer ID and timestamp.

## Agreement and adjudication

Calculate:

- raw agreement;
- class-specific agreement;
- Cohen/Fleiss kappa only where appropriate;
- prevalence-aware agreement sensitivity;
- coordinate matching agreement;
- timing agreement;
- adjudication changes.

Do not automatically resolve disagreements.

## Post-adjudication evaluation

Only after bounded-field truth is frozen, calculate:

- local precision-recall;
- candidate taxonomy;
- hidden-positive rate among previously unmatched candidates;
- score calibration;
- reviewer-confidence sensitivity;
- confirmed-only and inclusive views.

## If reviewers are unavailable

Complete packet generation, freeze the region and candidate set, write the pending decision, and continue. The manuscript must label precision as pending and avoid substituting a surrogate.

---

# 19. Stage 10 — frozen compact representation confirmation

## Goal

Confirm a compact interpretable feature panel without another broad sweep.

## Frozen lanes

1. Raw/carrier reference.
2. Signed adjacent-frame difference.
3. Frozen two-frame ICA as an equivalence control.
4. ROI-minus-annulus residual.
5. `coherence_w15`.
6. `propagation_lag2_w15`.
7. `radial_cs_shell` when a quantitative, non-display-clipped source is available.
8. One variance-stabilized auxiliary-carrier lane when reconstructable.

Do not add additional features unless a fatal implementation issue makes one lane unavailable; any replacement requires a documented rationale and remains exploratory.

## Evaluation regimes

- native proposals;
- identical frozen proposal union for ranking-only comparison;
- budgets 20, 40, and 58 primary;
- budgets 80 and 100 secondary;
- NMS radii 4, 6, and 8 px sensitivity, with 6 px primary;
- leave-one-burst-out selection only where selection is necessary;
- original-site/original-timing primary invariant;
- adjudicated/canonical views as sensitivity or final view when available.

## Trace-fidelity guardrails

A feature may improve recovery while damaging trace interpretation. Report:

- Raw/pipeline event correlation;
- amplitude ratio;
- area ratio;
- peak-time offset;
- onset sensitivity;
- baseline variance change.

The signed difference and pairwise ICA are expected to act as onset/change features rather than denoised fluorescence traces. Do not require them to preserve amplitude, but do not describe them as denoisers.

## Confirmation levels

### C3 — confirmed compact utility

- macro known-positive recall improves by at least `+0.03` at budget 20;
- improvement is nonnegative in at least three of four bursts;
- no catastrophic drop greater than `-0.05` in any burst;
- identical-proposal analysis demonstrates ranking utility or native-proposal analysis demonstrates proposal utility;
- results survive small NMS sensitivity;
- no prohibited precision claim.

### C2 — limited/context-dependent utility

- positive aggregate change but below `+0.03`, or concentrated in one burst;
- useful for a defined failure mode or secondary budget.

### C1 — descriptive feature only

- no recovery improvement but interpretable scientific statistic.

### C0 — no demonstrated utility

- unstable or consistently harmful.

## Pairwise ICA conclusion

Retain the operator-identification assay as a negative/clarifying result. If repaired results again show near-equivalence to signed difference, state that two-frame ICA does not create new source-identifying information in this setting.

---

# 20. Stage 11 — manuscript figures, tables, and result macros

## Goal

Export every paper claim from validated result tables rather than manual transcription.

## Required result macro generation

Generate `manuscript/results_macros.tex` with one macro per manuscript value. Example:

```tex
\newcommand{\DatasetOccurrenceCount}{79}
\newcommand{\PairwiseICADifferenceRSquared}{0.9938}
\newcommand{\RawTemplateHeldOutCorrelation}{0.8256}
```

Also generate `result_source_map.json` containing:

- macro name;
- rendered value;
- source file;
- JSON pointer or table query;
- source hash;
- analysis view;
- provisional/final status;
- code revision;
- run root.

The paper must not contain manually copied final numbers.

## Required main figures

Use these stable filenames:

1. `fig01_dataset_annotation_structure.pdf`
2. `fig02_acquisition_physics.pdf`
3. `fig03_trace_atlas.pdf`
4. `fig04_shared_specific_decomposition.pdf`
5. `fig05_site_observability.pdf`
6. `fig06_spatial_phenotype_failures.pdf`
7. `fig07_detection_implications.pdf`

## Figure content

### Figure 1 — dataset and annotation structure

- mean/projection image;
- 27 original sites;
- proposed 26 canonical identities shown separately;
- four burst windows;
- review-state summary;
- acquisition field boundary.

### Figure 2 — acquisition physics

- variance versus intensity;
- saturation map;
- left/right field comparison;
- temporal drift/PSD;
- optional multi-video QC comparison.

### Figure 3 — trace atlas

Representative high-, medium-, and low-observability sites plus burst 2:

- Raw;
- annulus;
- residual;
- signed difference;
- ICA;
- coherence/recurrence when available.

### Figure 4 — shared versus specific structure

- population Raw template;
- annulus template;
- residual template;
- scalar/functional variance components;
- tensor factors if promoted.

### Figure 5 — stable observability

- site-by-burst amplitude/SNR matrices;
- rank correlations;
- ICCs;
- burst-2 sensitivity.

### Figure 6 — spatial phenotype and failure taxonomy

- center/ring, isolated/crowded, compact/diffuse examples;
- radial profiles;
- proposal/ranking/localization/timing/NMS failures.

### Figure 7 — detection implications

- recall versus candidate budget;
- identical-proposal ranking comparison;
- fidelity tradeoffs;
- bounded-field precision when available.

## Required tables

1. Dataset/annotation contract.
2. Acquisition-QC metrics.
3. Primary hypotheses and outcomes.
4. Variance/repeatability summary.
5. Spatial specificity/equivalence summary.
6. Frozen representation confirmation.
7. Limitations and identifiable claims.

## Supplement

- one trace page per site and occurrence;
- all radii/window sensitivities;
- detailed review tables;
- model diagnostics;
- source and environment manifests;
- bounded-field protocol;
- complete failure taxonomy.

## Validation

- all figures are generated from machine-readable data;
- captions state the analysis view and sample units;
- no table duplicates an incompatible label view;
- manuscript macros agree with source hashes;
- provisional values are visibly marked until final rerun.

---

# 21. Stage 12 — reproducibility and release package

## Goal

Produce a self-contained scientific record suitable for collaborators and manuscript preparation.

## Required outputs

- exact environment report;
- code revision and dirty-state record;
- source-data hashes;
- label-source hashes;
- resolved config;
- command history;
- all stage status files;
- final artifact index;
- final LLM context index;
- manuscript macro source map;
- figure-generation commands;
- test results;
- reproducibility checklist;
- known missing artifacts;
- data/code availability draft.

## Test policy

Run:

1. new unit tests;
2. relevant focused existing tests;
3. full collection with optional dependency classification;
4. CPU smoke run on synthetic data;
5. tiny real-data smoke window;
6. full canonical run;
7. deterministic rerun of a compact subset to verify numerical stability.

Do not require CuPy for CPU test collection. Replace unconditional optional imports in tests with `pytest.importorskip` or a fixture-based CPU path, while preserving explicit GPU integration tests.

## Release gate

A paper package may be called `analysis_complete` only when:

- identity/geometry repair passed;
- trace atlas passed;
- acquisition QC passed;
- scalar and spatial primary analyses passed or reached a valid null conclusion;
- every claimed number has a source macro;
- all unresolved human decisions are listed;
- no unsupported precision or cross-recording claim appears;
- scientific-audit requirements are met for promoted detector lanes.

A package may be called `manuscript_ready` only after:

- final adjudication or explicit author decision to publish site-level results;
- ethics/funding/authorship/data statements are completed;
- bounded-field precision is either complete or explicitly removed from claims;
- all provisional macros are regenerated as final;
- coauthors approve the claim matrix.

---

# 22. Primary hypotheses, estimands, and decision matrix

| ID | Hypothesis | Primary estimand | Current status to verify | Final decision language |
|---|---|---|---|---|
| H1 | Labeled intervals contain unusual transient activity | event-minus-block-shift score | supported provisionally | supported / not supported |
| H2 | Activity is spatially concentrated at labeled sites | center-minus-annulus amplitude and equivalence tests | partial | superiority / equivalent / unresolved |
| H3 | Sites have stable observability across bursts | burst-adjusted ICC and rank stability | exploratory | O0-O3 evidence level |
| H4 | Sites possess reproducible temporal morphology beyond shared tissue | functional site variance and held-out residual correlation | not established | supported / not supported / underpowered |
| H5 | Measurement phenotype explains known-positive misses | leave-one-burst-out recovery model | suggestive | predictive / descriptive / unsupported |
| H6 | Two-frame ICA adds information beyond differentiation | analytic equivalence and event-score correlation | rejected provisionally | equivalent / distinguishable |
| H7 | Compact spatial/contextual statistics improve ranking | protected fixed-budget recall and identical proposals | supported provisionally | C0-C3 level |
| H8 | Local precision is estimable | bounded-field exhaustive annotation | pending | estimated locally / unavailable |

Codex must update this table from results and preserve negative findings.

---

# 23. Essential metrics schema

At minimum, `FINAL_METRICS.json` must contain:

```json
{
  "schema_version": 1,
  "scope": "single_recording_within_recording_methodological_case_study",
  "data": {
    "recording_count": 1,
    "burst_count": 4,
    "occurrence_count_by_view": {},
    "original_site_count": null,
    "canonical_identity_count_by_view": {},
    "frame_count": null,
    "frame_interval_ms": null
  },
  "identity": {
    "geometry_collision_count_before": null,
    "geometry_collision_count_after": null,
    "roi_010_015_status": "pending|separate|merged|unresolved"
  },
  "acquisition": {
    "noise_model": {},
    "saturation": {},
    "field_boundary": {},
    "drift": {},
    "other_video_qc": []
  },
  "traces": {
    "channel_inventory": {},
    "occurrence_metrics": {},
    "fidelity": {}
  },
  "observability": {
    "metrics": {},
    "evidence_levels": {},
    "burst_2_sensitivity": {}
  },
  "spatial_specificity": {
    "center_annulus": {},
    "equivalence": {},
    "radial_profiles": {}
  },
  "functional_tensor": {
    "entry_gate": {},
    "functional_variance": {},
    "tensor_rank": {}
  },
  "measurement_phenotypes": {
    "recurrent_hard_sites": [],
    "recoverability_models": {}
  },
  "representations": {
    "pairwise_ica_equivalence": {},
    "frozen_panel": {},
    "known_positive_only": true
  },
  "bounded_field": {
    "status": "pending|complete|not_run",
    "region": {},
    "agreement": {},
    "precision_recall": {}
  },
  "claims": {
    "supported": [],
    "not_supported": [],
    "unresolved": [],
    "prohibited": []
  }
}
```

---

# 24. Testing details

## 24.1 Contract tests

- invalid frame ranges rejected;
- UI/NumPy conversion round-trip;
- coordinates use x=column/y=row;
- duplicate observation IDs rejected;
- source hashes stable;
- atomic writes leave no partial final files;
- label view exclusions are explicit.

## 24.2 Geometry tests

- canonical merge does not collapse sites;
- annulus masks are nonoverlapping and boundary-correct;
- union/manual masks have deterministic hashes;
- geometry near image boundaries is either clipped with recorded area or rejected according to config.

## 24.3 Statistical tests

Use synthetic data with known:

- site random effect;
- burst effect;
- shared waveform;
- site waveform deviation;
- annulus contamination;
- rank-one and rank-two tensors;
- missing observations;
- singular/no-effect cases.

Assert recovery within tolerances and correct downgrade behavior.

## 24.4 Positive-unlabeled tests

- unmatched candidates cannot be labeled false positive by any public metric function;
- precision functions require an exhaustive-field truth flag;
- candidate burden remains available.

## 24.5 Paper export tests

- every macro has a source-map entry;
- missing source yields a visible missing/provisional marker;
- TeX special characters are escaped;
- figure index paths exist or compile-safe placeholders are used;
- manuscript compilation command is recorded.

---

# 25. Work packages and delegation

Codex may delegate bounded work to subagents, but one integration owner must enforce contracts.

## Work package A — contracts and identity repair

Own:

- central dataclasses;
- loaders;
- site/canonical crosswalk;
- signature-assay repair;
- regression tests.

Do not modify statistical interpretation.

## Work package B — trace atlas and acquisition QC

Own:

- trace extraction;
- channel reconstruction;
- occurrence metrics;
- atlas figures;
- noise/drift/field QC.

Do not select features using labels.

## Work package C — statistics

Own:

- mixed models;
- clustered bootstrap;
- rank repeatability;
- equivalence tests;
- functional/tensor conditional stages.

Must provide synthetic validation.

## Work package D — spatial and phenotype analysis

Own:

- radial profiles;
- spatial controls;
- measurement profiles;
- recovery/failure models.

Must keep unmatched candidates unknown.

## Work package E — bounded-field annotation

Own:

- deterministic region proposal;
- blinded packets;
- reviewer schemas;
- agreement/adjudication ingestion;
- local precision only after frozen truth.

## Work package F — frozen representation confirmation

Own:

- exact compact lanes;
- native and identical proposals;
- budget/NMS sensitivity;
- fidelity guardrails;
- scientific audit package.

No grid widening.

## Work package G — manuscript exports

Own:

- result macros;
- source map;
- figures/tables;
- Overleaf synchronization;
- compilation validation;
- claim matrix.

## Integration owner

Must:

- review every interface;
- run all tests;
- resolve duplicated utilities;
- ensure output roots are new;
- ensure no stage silently changes label view;
- ensure manuscript claims match final validation.

---

# 26. How Codex must react as results return

## After identity repair

- If 27 sites and all observations are preserved, rerun affected signature analyses.
- If counts differ, diagnose source-view differences before proceeding.
- Do not wait for ROI 010/015 adjudication; run site-level analyses and queue the decision.

## After trace atlas

- If mandatory Raw/annulus/residual traces pass, continue.
- If optional feature traces are unavailable, mark them unavailable and continue.
- If burst 2 shows materially different timing, preserve both original and suggested windows and queue review.

## After scalar observability

- If O2/O3, promote stable observability as a main result.
- If O0/O1, retain heterogeneity as a descriptive result and avoid a stable-neuron-gain claim.
- Always inspect whether baseline intensity or acquisition field explains the effect.

## After spatial specificity

- If center superiority is established, state localized measurement evidence.
- If equivalence is established, state that the tested support does not isolate site-specific morphology.
- If unresolved, report uncertainty and do not convert it to a null conclusion.

## After functional/tensor stage

- Promote only stable held-out structure.
- If rank one dominates, formulate a shared waveform × site gain × burst strength model.
- If richer components are stable, describe measurement modes, not cell types.
- If unstable, stop the branch without blocking the paper.

## After phenotype/recovery modeling

- Promote only leave-one-burst-out or otherwise protected associations.
- If one burst drives performance, label it context-dependent.
- Never infer candidate truth from recovery models.

## After bounded-field packet generation

- Continue all other stages.
- Add reviewer tasks to the final decision file.
- Once completed labels are later supplied, rerun only bounded-field metrics and dependent manuscript exports.

## After representation confirmation

- Promote a feature only at C2/C3 according to the frozen criteria.
- Preserve the pairwise ICA equivalence result even if it is negative.
- Do not describe `propagation_lag2_w15` as causal propagation.

---

# 27. Final user-feedback behavior

At the end of the autonomous run, Codex must present a compact summary and request only unresolved decisions. It must not ask the user to inspect raw directories without direct paths to review artifacts.

Generate `DECISIONS_REQUESTED.yaml` using this structure:

```yaml
schema_version: 1
run_root: Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1
items:
  - id: identity_roi_010_015
    status: pending
    question: >-
      Should original sites roi_010 and roi_015 be treated as separate neurons,
      one merged neuron, or unresolved?
    options:
      - separate
      - merged_use_roi_010_geometry
      - merged_use_roi_015_geometry
      - merged_union_geometry
      - unresolved
    recommended_option: unresolved
    evidence:
      - 01_identity_geometry/review/roi_010_015_overlay.png
      - 03_trace_atlas/review/roi_010_015_trace_comparison.png
    blocks:
      - final_canonical_neuron_count
      - canonical_functional_analysis
    does_not_block:
      - original_site_analysis
  - id: burst_2_timing
    status: pending
    question: >-
      Accept, edit, or reject the generated burst-2 onset/peak/end proposals.
    evidence:
      - 02_label_timing_contract/burst_2_review/index.html
    blocks:
      - adjudicated_timing_view
    does_not_block:
      - original_timing_view
```

`REVIEW_REQUIRED.md` must contain:

- the scientific consequence of each choice;
- the recommended conservative default;
- exact artifact paths;
- a copy-paste response format;
- no already-resolved question.

The final chat prompt should resemble:

```text
All non-blocked stages are complete. Please review the five decisions in
DECISIONS_REQUESTED.yaml. The conservative defaults preserve original sites and
original timing. Reply with the decision IDs and selected options; only affected
stages need to be rerun.
```

---

# 28. Final acceptance checklist

## Repository integrity

- [ ] `AGENTS.md` and scientific-audit rules were followed.
- [ ] No completed output root was overwritten.
- [ ] Source files and labels were not edited in place.
- [ ] Optional dependency tests do not break CPU collection.

## Identity and labels

- [ ] 79 current occurrences are preserved or any difference is explained.
- [ ] 27 original spatial sites are independently addressable in the current canonical data.
- [ ] Proposed canonical identities are separated from geometry.
- [ ] ROI 010/015 cannot overwrite each other.
- [ ] Original and adjudicated timing views remain available.

## Trace atlas

- [ ] Raw, annulus, residual, difference, and ICA traces exist for every included occurrence.
- [ ] Full and event-centered views are synchronized.
- [ ] Native amplitude is preserved in primary analyses.
- [ ] Every metric has units and provenance.

## Statistics

- [ ] Site/burst dependence is respected.
- [ ] Four-burst limitations are explicit.
- [ ] Singular models downgrade transparently.
- [ ] Equivalence is tested with margins, not inferred from nonsignificance.
- [ ] Functional/tensor results are held-out and stable if promoted.

## Positive-unlabeled evaluation

- [ ] Unmatched candidates remain unknown outside the bounded field.
- [ ] No full-field precision or specificity is reported.
- [ ] Known-positive recovery is clearly named.
- [ ] Bounded-field precision is local and confidence-sensitive.

## Representations

- [ ] No broad feature expansion occurred.
- [ ] Pairwise ICA is compared analytically with signed difference.
- [ ] Contextual features use frozen parameters.
- [ ] Detection utility and trace fidelity are reported separately.

## Manuscript

- [ ] Every numerical claim uses a generated macro.
- [ ] Every macro has a source-map record.
- [ ] Current provisional values were replaced or visibly marked.
- [ ] Main figures follow the stable filename contract.
- [ ] Limitations match the actual analysis scope.
- [ ] Generative-AI disclosure is included and reviewed by authors.
- [ ] Ethics, funding, CRediT, data, and code statements are completed.

---

# 29. Immediate implementation order

Codex should begin in this order:

1. Create the new package/config/CLI skeleton and common stage artifact writer.
2. Implement and test the central observation/site/canonical identity contract.
3. Repair the two signature assays and add the ROI 010/015 regression test.
4. Run preflight and generate the source/label inventory.
5. Generate the complete trace atlas.
6. Run canonical acquisition QC and any available unlabeled-video QC.
7. Run scalar observability and spatial-specificity analyses.
8. Conditionally run functional/tensor analysis.
9. Run measurement-phenotype/recoverability analysis.
10. Generate the bounded-field annotation packet without blocking.
11. Confirm the frozen representation panel.
12. Export manuscript macros, figures, tables, reports, and review decisions.
13. Run release validation and produce the final collaborator-facing summary.

The default immediate command after implementation is:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.neuron_identifiability run \
  --config examples/spon_ca_burst_neuron_identifiability_paper_v1.example.json \
  --resume
```

Do not begin by training a new neural network, widening ICA dimensionality, or launching a hyperparameter sweep. The first scientific deliverable is the repaired identity-aware trace and measurement analysis.
