#!/usr/bin/env python3
"""Build flat, upload-ready Overleaf packages for the two venue drafts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
PAPER_ROOT = BASE.parent
DEFAULT_OUTPUT = PAPER_ROOT / "overleaf_uploads" / "2026-09-07"
VENDOR = BASE / "vendor" / "springer_nature_2024_v3_1"
FIGURE_ROOT = BASE / "figures" / "venue"


BODY_INPUTS = {
    "venue_drafts/jnm/01_introduction": BASE
    / "venue_drafts/jnm/01_introduction.tex",
    "venue_drafts/neuroinformatics/01_introduction": BASE
    / "venue_drafts/neuroinformatics/01_introduction.tex",
    "venue_drafts/neuroinformatics/02_system_and_methods": BASE
    / "venue_drafts/neuroinformatics/02_system_and_methods.tex",
    "venue_drafts/neuroinformatics/03_software_validation": BASE
    / "venue_drafts/neuroinformatics/03_software_validation.tex",
    "venue_drafts/neuroinformatics/05_conclusion": BASE
    / "venue_drafts/neuroinformatics/05_conclusion.tex",
    "venue_drafts/shared/02_materials_and_methods": BASE
    / "venue_drafts/shared/02_materials_and_methods.tex",
    "venue_drafts/shared/03_results": BASE / "venue_drafts/shared/03_results.tex",
    "venue_drafts/jnm/04_discussion": BASE / "venue_drafts/jnm/04_discussion.tex",
    "venue_drafts/neuroinformatics/04_discussion": BASE
    / "venue_drafts/neuroinformatics/04_discussion.tex",
    "venue_drafts/shared/05_conclusion": BASE
    / "venue_drafts/shared/05_conclusion.tex",
    "venue_drafts/shared/06_supplementary_operator_note": BASE
    / "venue_drafts/shared/06_supplementary_operator_note.tex",
    "sections/06_declarations": BASE / "sections/06_declarations.tex",
    "macros/provisional": BASE / "macros/provisional.tex",
    "macros/venue_results": BASE / "macros/venue_results.tex",
}


COMMON_FIGURES = (
    "fig02_measurement_representations.png",
    "fig03_known_positive_evaluation.png",
    "fig04_candidate_review.png",
    "fig05_identity_aware_misses.png",
    "fig06_exact_truth_validation.png",
)


UNUSED_JNM_INPUTS = (
    "macros/notation",
    "macros/results_macros",
    "macros/full_trace_feature_panel",
    "macros/automated_feature_validation",
    "macros/feature_deep_dives",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inline_inputs(source: Path, allowed_keys: set[str]) -> str:
    text = source.read_text(encoding="utf-8")
    for key in UNUSED_JNM_INPUTS:
        text = text.replace(f"\\input{{{key}}}\n", "")
    for key in sorted(allowed_keys, key=len, reverse=True):
        marker = f"\\input{{{key}}}"
        body = BODY_INPUTS[key].read_text(encoding="utf-8").rstrip()
        text = text.replace(marker, f"% BEGIN INLINED: {key}\n{body}\n% END INLINED: {key}")
    text = text.replace("figures/venue/", "")
    text = text.replace("\\graphicspath{{figures/}}", "\\graphicspath{{./}}")
    return text


def _validate_main(text: str, venue: str) -> None:
    if "\\input{" in text or "\\include{" in text:
        raise ValueError(f"{venue}: unresolved input/include remains")
    if text.count("{") != text.count("}"):
        raise ValueError(f"{venue}: unbalanced braces")
    if text.count("\\begin{") != text.count("\\end{"):
        raise ValueError(f"{venue}: unbalanced environments")
    if "/home/" in text or "external://" in text:
        raise ValueError(f"{venue}: nonportable path remains")
    if "79 occurrences" in text or "27 sites" in text:
        raise ValueError(f"{venue}: stale cohort count remains")
    if "\\DraftFigure{" in text:
        raise ValueError(f"{venue}: draft figure placeholder remains")
    integrated_figure_count = sum(
        "\\includegraphics" in line and "{fig0" in line for line in text.splitlines()
    )
    if integrated_figure_count != 6:
        raise ValueError(f"{venue}: expected exactly six integrated figures")
    for forbidden in (
        "total budget of 58",
        "frozen 58-candidate budget",
        "0.9844",
        "0.9797",
        "34/78",
        "35/78",
        "79 occurrences",
        "27 sites",
    ):
        if forbidden in text:
            raise ValueError(f"{venue}: forbidden stale or excluded statement remains: {forbidden}")
    for required in (
        "\\VenueCohortOccurrences",
        "\\VenueCohortSites",
        "per-lane",
        "any-lane",
        "\\VenueStrictRecovered",
        "\\VenueReviewLikely",
        "fig06_exact_truth_validation.png",
    ):
        if required not in text:
            raise ValueError(f"{venue}: missing required current statement: {required}")


def _copy_figures(target: Path, venue_figure: str) -> None:
    names = (venue_figure,) + COMMON_FIGURES
    for name in names:
        source = FIGURE_ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / name)
    master_path = FIGURE_ROOT / "venue_figure_manifest.json"
    master = json.loads(master_path.read_text(encoding="utf-8"))
    selected = set(names)
    package_manifest = {
        "schema_version": 1,
        "purpose": "six integrated main figures for one venue package",
        "master_manifest_sha256": _sha256(master_path),
        "sources": [
            row for row in master["sources"] if "feature_atlas" not in row["path"]
        ],
        "outputs": [
            row for row in master["outputs"] if Path(row["path"]).name in selected
        ],
        "excluded": {
            "figS1_feature_atlas_presubmission_audit.png":
                "not a main-paper figure; canonical registration and media audit incomplete"
        },
    }
    (target / "venue_figure_manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(BASE / "macros/venue_results_manifest.json", target / "venue_results_manifest.json")


def _write_readme(target: Path, venue: str, template_note: str) -> None:
    target.write_text(
        "# Overleaf upload instructions\n\n"
        f"Venue draft: {venue}\n\n"
        "1. In Overleaf, choose New Project -> Upload Project.\n"
        "2. Upload the ZIP archive supplied with this directory.\n"
        "3. Confirm that `main.tex` is the main document.\n"
        "4. Set the compiler to pdfLaTeX.\n"
        "5. Compile and inspect all red author-decision placeholders.\n"
        "6. Confirm that six venue-only PNG figures, both source manifests, and "
        "`RESULTS_AND_EVIDENCE_MAP.md` are present.\n\n"
        f"{template_note}\n\n"
        "The narrative, tables, and six main figures are integrated and package-checked. "
        "This remains a working manuscript, not a submission-ready release. Author "
        "affiliations, corresponding-author details, ethics, funding, CRediT "
        "roles, data/code identifiers, references, and final adjudication remain "
        "open. The manuscript intentionally does not report 13/18 as detector "
        "precision or promote the Feature Atlas point estimate as confirmed.\n",
        encoding="utf-8",
    )


def _write_manifest(directory: Path) -> None:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _zip_flat(directory: Path, archive: Path) -> None:
    fixed_time = (2026, 9, 7, 12, 0, 0)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.name, fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            out.writestr(info, path.read_bytes())


def _prepare_clean_dir(path: Path) -> None:
    resolved = path.resolve()
    output_root = DEFAULT_OUTPUT.parent.resolve()
    if output_root not in resolved.parents:
        raise ValueError(f"refusing to replace unexpected path: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def build(output: Path) -> list[Path]:
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError("only the versioned default output root is supported")

    jnm_dir = output / "journal_of_neuroscience_methods"
    neuro_dir = output / "neuroinformatics"
    output.mkdir(parents=True, exist_ok=True)
    _prepare_clean_dir(jnm_dir)
    _prepare_clean_dir(neuro_dir)

    shared = {
        "venue_drafts/shared/02_materials_and_methods",
        "venue_drafts/shared/03_results",
        "venue_drafts/shared/05_conclusion",
        "venue_drafts/shared/06_supplementary_operator_note",
        "sections/06_declarations",
        "macros/provisional",
        "macros/venue_results",
    }

    jnm_text = _inline_inputs(
        BASE / "main_jnm_identity_safe_draft.tex",
        shared
        | {"venue_drafts/jnm/01_introduction", "venue_drafts/jnm/04_discussion"},
    )
    _validate_main(jnm_text, "Journal of Neuroscience Methods")
    (jnm_dir / "main.tex").write_text(jnm_text, encoding="utf-8")
    _copy_figures(jnm_dir, "fig01_jnm_identity_contract.png")
    shutil.copy2(BASE / "references.bib", jnm_dir / "references.bib")
    shutil.copy2(BASE / "highlights_jnm_identity_safe.txt", jnm_dir / "highlights.txt")
    shutil.copy2(
        BASE / "EXPERIMENT_INCLUSION_MATRIX_2026-09-07.md",
        jnm_dir / "RESULTS_AND_EVIDENCE_MAP.md",
    )
    _write_readme(
        jnm_dir / "README.md",
        "Journal of Neuroscience Methods",
        "The manuscript uses Elsevier's official `elsarticle` class with "
        "author--year references. Overleaf supplies this class.",
    )
    _write_manifest(jnm_dir)

    neuro_text = _inline_inputs(
        BASE / "main_neuroinformatics_identity_safe_draft.tex",
        shared
        | {
            "venue_drafts/neuroinformatics/01_introduction",
            "venue_drafts/neuroinformatics/02_system_and_methods",
            "venue_drafts/neuroinformatics/03_software_validation",
            "venue_drafts/neuroinformatics/04_discussion",
            "venue_drafts/neuroinformatics/05_conclusion",
        },
    )
    _validate_main(neuro_text, "Neuroinformatics")
    (neuro_dir / "main.tex").write_text(neuro_text, encoding="utf-8")
    _copy_figures(neuro_dir, "fig01_neuroinformatics_architecture.png")
    shutil.copy2(BASE / "references.bib", neuro_dir / "references.bib")
    shutil.copy2(
        BASE / "highlights_neuroinformatics_identity_safe.txt",
        neuro_dir / "highlights.txt",
    )
    shutil.copy2(
        BASE / "EXPERIMENT_INCLUSION_MATRIX_2026-09-07.md",
        neuro_dir / "RESULTS_AND_EVIDENCE_MAP.md",
    )
    shutil.copy2(VENDOR / "sn-jnl.cls", neuro_dir / "sn-jnl.cls")
    shutil.copy2(VENDOR / "sn-mathphys-ay.bst", neuro_dir / "sn-mathphys-ay.bst")
    shutil.copy2(VENDOR / "TEMPLATE_SOURCE.md", neuro_dir / "TEMPLATE_SOURCE.md")
    _write_readme(
        neuro_dir / "README.md",
        "Neuroinformatics",
        "The package includes Springer Nature `sn-jnl.cls` template version "
        "3.1 (December 2024) and the `sn-mathphys-ay` bibliography style.",
    )
    _write_manifest(neuro_dir)

    archives = [
        output / "NeuRev_JNM_identity_safe_Overleaf_2026-09-07.zip",
        output / "NeuRev_Neuroinformatics_identity_safe_Overleaf_2026-09-07.zip",
    ]
    _zip_flat(jnm_dir, archives[0])
    _zip_flat(neuro_dir, archives[1])
    return archives


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    archives = build(args.output)
    for archive in archives:
        print(f"{_sha256(archive)}  {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
