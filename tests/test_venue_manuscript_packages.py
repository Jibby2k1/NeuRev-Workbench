from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "overleaf_jnm"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_text(venue: str) -> str:
    shared = [
        PAPER / "venue_drafts/shared/02_materials_and_methods.tex",
        PAPER / "venue_drafts/shared/03_results.tex",
        PAPER / "venue_drafts/shared/06_supplementary_operator_note.tex",
    ]
    if venue == "jnm":
        paths = [
            PAPER / "main_jnm_identity_safe_draft.tex",
            PAPER / "venue_drafts/jnm/01_introduction.tex",
            *shared,
            PAPER / "venue_drafts/jnm/04_discussion.tex",
            PAPER / "venue_drafts/shared/05_conclusion.tex",
        ]
    else:
        paths = [
            PAPER / "main_neuroinformatics_identity_safe_draft.tex",
            PAPER / "venue_drafts/neuroinformatics/01_introduction.tex",
            shared[0],
            PAPER / "venue_drafts/neuroinformatics/02_system_and_methods.tex",
            shared[1],
            PAPER / "venue_drafts/neuroinformatics/03_software_validation.tex",
            PAPER / "venue_drafts/neuroinformatics/04_discussion.tex",
            PAPER / "venue_drafts/neuroinformatics/05_conclusion.tex",
            shared[2],
        ]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_generated_facts_reconcile_high_risk_denominators() -> None:
    manifest = json.loads((PAPER / "macros/venue_results_manifest.json").read_text())
    facts = manifest["facts"]
    assert int(facts["VenueFactSourceCount"]) == len(manifest["sources"])
    assert int(facts["VenueStrictRecovered"]) + int(facts["VenueStrictMissed"]) == int(
        facts["VenueCohortOccurrences"]
    )
    assert int(facts["VenueSameIdentityCollision"]) + int(
        facts["VenueOtherIdentityCollision"]
    ) == int(facts["VenueIdentityCollision"])
    assert int(facts["VenueReviewDefinite"]) + int(facts["VenueReviewProbable"]) == int(
        facts["VenueReviewLikely"]
    )
    assert int(facts["VenueTraceClassOneSites"]) + int(
        facts["VenueTraceClassTwoSites"]
    ) == int(facts["VenueCohortSites"])
    assert "feature_atlas_v1" in manifest["excluded"]
    assert manifest["schema_version"] == 2
    for source in manifest["sources"]:
        path = ROOT / source["path"]
        assert len(source["sha256"]) == 64
        int(source["sha256"], 16)
        assert source["size_bytes"] > 0
        local_only = Path(source["path"]).parts[0] == "Outputs"
        assert source["availability"] == (
            "local_metadata_only" if local_only else "repository"
        )
        if local_only and not path.exists():
            continue
        assert path.is_file()
        assert path.stat().st_size == source["size_bytes"]
        assert _sha256(path) == source["sha256"]


def test_both_manuscripts_have_complete_main_visual_story() -> None:
    for venue in ("jnm", "neuroinformatics"):
        text = _source_text(venue)
        assert "\\DraftFigure{" not in text
        assert text.count("\\includegraphics[width=\\textwidth]{figures/venue/fig0") == 6
        assert "per-lane" in text
        assert "any-lane" in text
        assert "total budget of 58" not in text
        assert "frozen 58-candidate budget" not in text
        assert "0.9844" not in text
        assert "0.9797" not in text
        assert "34/78" not in text
        assert "35/78" not in text


def test_venue_figure_manifest_binds_current_main_assets() -> None:
    manifest = json.loads((PAPER / "figures/venue/venue_figure_manifest.json").read_text())
    output_names = {Path(row["path"]).name for row in manifest["outputs"]}
    required = {
        "fig01_jnm_identity_contract.png",
        "fig01_neuroinformatics_architecture.png",
        "fig02_measurement_representations.png",
        "fig03_known_positive_evaluation.png",
        "fig04_candidate_review.png",
        "fig05_identity_aware_misses.png",
        "fig06_exact_truth_validation.png",
    }
    assert required <= output_names
    assert "fig01_dataset_annotation_structure.pdf" not in output_names
    assert manifest["schema_version"] == 2
    for row in manifest["sources"]:
        assert len(row["sha256"]) == 64
        int(row["sha256"], 16)
        assert row["size_bytes"] > 0
        local_only = Path(row["path"]).parts[0] == "Outputs"
        assert row["availability"] == (
            "local_metadata_only" if local_only else "repository"
        )
        path = ROOT / row["path"]
        # Private/local evidence is hash-bound metadata in a source-only clone.
        # When it is available on the research workstation, verify it too.
        if local_only and not path.exists():
            continue
        assert path.is_file()
        assert path.stat().st_size == row["size_bytes"]
        assert _sha256(path) == row["sha256"]
    for row in manifest["outputs"]:
        path = ROOT / row["path"]
        assert path.is_file()
        assert _sha256(path) == row["sha256"]


def test_neuroinformatics_has_venue_specific_software_body() -> None:
    main = (PAPER / "main_neuroinformatics_identity_safe_draft.tex").read_text()
    assert "venue_drafts/neuroinformatics/02_system_and_methods" in main
    assert "venue_drafts/neuroinformatics/03_software_validation" in main
    assert "venue_drafts/neuroinformatics/05_conclusion" in main
    assert "venue_drafts/shared/05_conclusion" not in main


def test_result_map_preserves_distinct_units_and_budgets() -> None:
    text = (PAPER / "EXPERIMENT_INCLUSION_MATRIX_2026-09-07.md").read_text()
    for required in (
        "106 occurrence geometries at 50 immutable sites",
        "three independently truncated per-burst B58 lists",
        "93/102",
        "per-burst B100",
        "Feature Atlas used a different",
        "not precision",
    ):
        assert required in text
