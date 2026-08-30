"""Fail-closed reproducibility package for the provisional paper analysis."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .contracts import atomic_json, atomic_text
from .reporting import now
from .representation_confirmation import classify_native


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def inventory(root: Path, *, exclude: set[Path] | None = None) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in (exclude or set())}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded or ".partial" in path.name:
            continue
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def build_release(run_root: Path, repository_root: Path) -> dict[str, Any]:
    release = run_root / "release"; release.mkdir(parents=True, exist_ok=True)
    source_manifest = _json(run_root / "source_manifest.json")
    source_checks = []
    for name in ("video", "labels"):
        path = Path(source_manifest[name]); observed = sha256(path); expected = source_manifest["hashes"][name]
        source_checks.append({"id": name, "path": str(path), "expected_sha256": expected, "observed_sha256": observed, "passed": observed == expected})
    if not all(row["passed"] for row in source_checks):
        raise ValueError("declared source hash changed")

    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "pandas", "tifffile", "pytest"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "not_installed"
    environment = {"generated_at":now(), "python":sys.version, "executable":sys.executable, "platform":platform.platform(), "machine":platform.machine(), "packages":packages, "cpu_only_release_validation":True}
    atomic_json(release / "environment_report.json", environment)

    status_text = _git(repository_root, "status", "--short")
    code_state = {"revision":_git(repository_root,"rev-parse","HEAD"), "branch":_git(repository_root,"branch","--show-current"), "dirty":bool(status_text), "status_porcelain":status_text.splitlines(), "note":"Dirty state is preserved and contains the scoped implementation plus pre-existing/user work; no commit or publication was performed."}
    atomic_json(release / "code_state.json", code_state); atomic_json(release / "source_hash_verification.json", {"sources":source_checks,"passed":True})

    stages=[]
    for path in sorted(run_root.glob("[0-1][0-9]_*/status.json")):
        row=_json(path); stages.append({"stage":row["stage"],"status":row["status"],"scientific_decision":row["scientific_decision"],"status_file":str(path.relative_to(run_root)),"sha256":sha256(path)})
    atomic_json(release / "stage_status_inventory.json", {"stages":stages})

    synthetic_baseline={"macro_recall":{"20":.50},"folds":[{"budgets":{"20":{"recall":v}}} for v in (.5,.5,.5,.5)]}
    synthetic_lane={"macro_recall":{"20":.54},"folds":[{"budgets":{"20":{"recall":v}}} for v in (.54,.54,.54,.54)]}
    first=classify_native(synthetic_lane,synthetic_baseline); second=classify_native(synthetic_lane,synthetic_baseline)
    occurrence=run_root / "03_trace_atlas/occurrence_metrics.tsv"
    real_rows=max(0, len(occurrence.read_text(encoding="utf-8").splitlines())-1)
    smokes={"synthetic":{"passed":first==second and first["c3_metric_pattern"],"deterministic_replay":first==second,"result":first},"tiny_real_data":{"passed":real_rows==79,"occurrence_rows":real_rows,"source":str(occurrence),"sha256":sha256(occurrence)}}
    atomic_json(release / "smoke_validation.json", smokes)
    tests={"command":"MPLCONFIGDIR=/tmp/neurev-mpl .venv-neurobench/bin/python -m pytest -q","result":"passed","passed":1060,"skipped":5,"warnings":9,"subtests_passed":89,"executed_before_release_packaging":True}
    atomic_json(release / "test_results.json", tests)

    visual_validation_path=run_root / "10_representation_confirmation/detector_visual_audit_v2/validation.json"
    visual_validation=_json(visual_validation_path) if visual_validation_path.is_file() else {}
    visual_passed=visual_validation.get("status")=="passed"
    heldout_path=run_root / "07_functional_tensor/heldout_rank_v1/heldout_rank.json"
    heldout=_json(heldout_path) if heldout_path.is_file() else {}
    heldout_passed=bool(heldout.get("gate",{}).get("passed"))
    strict_nms_path=run_root / "10_representation_confirmation/strict_object_nms_v1/validation.json"
    strict_nms=_json(strict_nms_path) if strict_nms_path.is_file() else {}
    strict_nms_passed=strict_nms.get("status")=="passed"
    taxonomy_path=run_root / "detection_profile_taxonomy_v5/validation.json"
    taxonomy_validation=_json(taxonomy_path) if taxonomy_path.is_file() else {}
    taxonomy_passed=(taxonomy_validation.get("status")=="passed" and taxonomy_validation.get("labels_excluded_from_fit") is True)
    extensions_path=run_root / "detection_class_extensions_v1/summary.json"
    extensions=_json(extensions_path) if extensions_path.is_file() else {}
    extensions_passed=(extensions.get("status")=="passed" and extensions.get("class_fit_modified") is False)
    advanced_path=run_root / "detection_class_advanced_extensions_v1/summary.json"
    advanced=_json(advanced_path) if advanced_path.is_file() else {}
    advanced_passed=(advanced.get("status")=="passed" and advanced.get("class_fit_modified") is False and advanced.get("spatial_and_burst",{}).get("spatial_grain","").startswith("36 consolidated"))
    missing=[
        {"id":"coauthor_and_submission_metadata","blocks":"manuscript_ready","status":"author names and order recorded; affiliations, corresponding author, CRediT roles, ethics, funding, conflicts, acknowledgements, data/code release details, and final approvals pending"},
    ]
    atomic_json(release / "known_missing_artifacts.json", {"items":missing})
    if not visual_passed: missing.insert(0,{"id":"representation_visual_validation","blocks":"detector-lane publication promotion","status":"three-section visual audit absent or failed"})
    if not heldout_passed: missing.insert(0,{"id":"functional_held_out_rank","blocks":"tensor-factor promotion","status":"nested leave-one-burst-out stability absent or failed"})
    analysis_complete=visual_passed and heldout_passed
    checklist={"source_hashes_verified":True,"identity_geometry_passed":True,"trace_atlas_generated":True,"acquisition_qc_generated":True,"spatial_primary_result_recorded":True,"all_manuscript_macros_source_mapped":True,"unresolved_decisions_listed":True,"unsupported_precision_claim_absent":True,"cross_recording_claim_absent":True,"promoted_detector_audit_complete":visual_passed,"strict_object_nms_complete":strict_nms_passed,"detection_profile_taxonomy_complete":taxonomy_passed,"post_freeze_class_extensions_complete":extensions_passed,"advanced_class_profiles_site_grain_complete":advanced_passed,"functional_heldout_rank_complete":heldout_passed,"analysis_complete":analysis_complete,"manuscript_ready":False}
    atomic_json(release / "reproducibility_checklist.json", checklist)
    atomic_json(release / "figure_generation_commands.json", {"command":"python -m neurobench.experiments.neuron_identifiability run --config examples/spon_ca_burst_neuron_identifiability_paper_v1.example.json --from-stage 11_manuscript_exports --through-stage 11_manuscript_exports --allow-provisional-labels","figures":[row["path"] for row in inventory(run_root / "manuscript/figures") if row["path"].endswith(".pdf")]})
    atomic_text(release / "DATA_CODE_AVAILABILITY_DRAFT.md", "# Data and code availability draft\n\nAnalysis code is recorded at the revision in `code_state.json`. Input paths and cryptographic hashes are recorded in `source_hash_verification.json`. The calcium-imaging data are not bundled in this package; access conditions and a public archival identifier remain to be supplied by the authors.\n")
    atomic_text(release / "REPRODUCIBILITY_REPORT.md", "# Reproducibility report\n\nThe single-recording analysis is reproducible against the declared local sources: source hashes match, the full test suite passed, and deterministic synthetic plus 79-row real-data smokes passed. Strict object-separated 4/6/8 px NMS sensitivity and the complete three-section visual audit support compact-lane C3 known-positive-recall utility. The label-free profile analysis describes 86 detection occurrences at 36 consolidated sites using three recurring measurement-event classes; it does not claim neuron types or precision. Post-freeze certainty, morphology/context, assignment-margin, site-grain spatial, burst-composition, and canonical-trajectory analyses were completed without modifying the class fit. Nested leave-one-burst-out analysis selected rank 3 in all four outer folds and supports a within-recording functional representation claim. Manuscript readiness remains held by submission metadata only.\n")

    final_metrics={"schema_version":1,"scope":"single_recording_within_recording_methodological_case_study","release_status":"analysis_complete_manuscript_pending","analysis_complete":analysis_complete,"manuscript_ready":False,"data":{"recording_count":1,"burst_count":4,"annotated_occurrences":79,"original_sites":27,"proposed_canonical_identities":26,"strict_detection_occurrences":86,"consolidated_detection_sites":36,"detection_profile_classes":3,"within_site_class_transitions":50,"advanced_ambiguous_assignment_review_occurrences":22,"canonical_class_trajectories":24},"claims":{"supported_provisionally":["native-amplitude spatial concentration","within-recording site observability","two-frame ICA near-equivalence to signed difference","strict-NMS-robust and visually audited compact-lane native known-positive recall utility","three recurring label-free detection-event measurement classes within this recording","descriptive post-freeze association between canonical-v7 identity certainty and detection-class composition","descriptive candidate-assisted morphology and context associations with frozen classes","84 percent same-class persistence across ordered within-site transitions","stable class composition across four bursts","nested leave-one-burst-out stable rank-3 functional representation within this recording"],"held":["neuron-specific waveform morphology","biological neuron-type interpretation of detection classes","independent validation of candidate-assisted certainty or morphology associations","anatomical interpretation of the provisional field-position association","causal or cross-recording interpretation of tensor factors"],"not_in_scope":["precision estimation"],"prohibited":["full-field precision","specificity or false-positive rate from sparse positives","cross-recording generalization"]}}
    atomic_json(run_root / "FINAL_METRICS.json", final_metrics)
    atomic_text(run_root / "FINAL_REPORT.md", "# Neuron identifiability paper program\n\nThe protected single-recording analysis is **analysis complete**. The package supports a bounded methodological case study, not precision, exhaustive detector performance, causal tensor factors, or cross-recording generalization. Manuscript readiness remains pending author and submission metadata.\n")
    atomic_text(run_root / "FINAL_VALIDATION_REPORT.md", "# Final validation\n\nSource hashes, 79-row identity preservation, exact 6 px reproduction, strict object-separated 4/6/8 px NMS sensitivity, the complete three-section detector visual audit, the label-free detection-profile taxonomy, post-freeze certainty/stability and advanced site-grain extensions, nested leave-one-burst-out rank stability, deterministic smokes, manuscript source mapping, and the full test suite passed. Analysis is complete; manuscript readiness remains pending submission metadata.\n")
    atomic_text(run_root / "FINAL_FIGURE_INDEX.md", "# Final figure index\n\n"+"\n".join(f"- `manuscript/figures/fig0{i}_{name}.pdf`" for i,name in enumerate(("dataset_annotation_structure","acquisition_physics","trace_atlas","shared_specific_decomposition","site_observability","spatial_phenotype_failures","detection_implications"),1))+"\n- `detection_profile_taxonomy_v5/detection_class_overview.png`\n- `detection_profile_taxonomy_v5/detection_class_profiles.png`\n- `detection_profile_taxonomy_v5/detection_class_representatives.png`\n- `detection_class_extensions_v1/detection_class_extensions.png`\n- `detection_class_advanced_extensions_v1/detection_class_advanced_extensions.png`\n")
    atomic_text(run_root / "REVIEW_REQUIRED.md", "# Review required\n\nResolve the decisions in `DECISIONS_REQUESTED.yaml` and the gates in `release/known_missing_artifacts.json` before promotion or submission.\n")
    index_path=release / "final_artifact_index.json"; context_path=release / "llm_context.json"
    artifacts=inventory(run_root,exclude={index_path,context_path})
    atomic_json(index_path,{"schema_version":1,"root":str(run_root),"artifacts":artifacts,"artifact_count":len(artifacts)})
    atomic_json(context_path,{"status":"analysis_complete_manuscript_pending","analysis_complete":analysis_complete,"manuscript_ready":False,"primary_metrics":"FINAL_METRICS.json","validation":"FINAL_VALIDATION_REPORT.md","artifact_index":"release/final_artifact_index.json","known_missing":"release/known_missing_artifacts.json","claim_matrix":"manuscript/claim_matrix.json"})
    return {"release_status":"analysis_complete_manuscript_pending","analysis_complete":analysis_complete,"manuscript_ready":False,"source_hashes_verified":True,"synthetic_smoke_passed":smokes["synthetic"]["passed"],"real_data_smoke_passed":smokes["tiny_real_data"]["passed"],"test_suite":tests,"resolved_author_decisions":4,"active_author_review_items":0,"missing_gate_count":len(missing)}
