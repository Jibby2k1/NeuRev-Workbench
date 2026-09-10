"""Generate gate-separated concluding remarks from validated ICA artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "unavailable"


def build_concluding_report(experiment_root: str | Path) -> str:
    root = Path(experiment_root).expanduser().resolve()
    real = root / "real_data_v1"; stages = real / "stages"
    audit = _json(real / "completion_audit_latest.json") or {}
    s3 = _json(stages / "S3_COMPLETE_FACTORIAL_ANALYSIS" / "summary.json")
    s4 = _json(stages / "S4_FINALIST_CONFIRMATION" / "summary.json")
    s5 = _json(stages / "S5_SCIENTIFIC_AUDIT" / "summary.json")
    s5v = _json(stages / "S5_SCIENTIFIC_AUDIT" / "validation.json")
    s6 = _json(stages / "S6_INDEPENDENT_CONFIRMATION" / "summary.json")
    s6v = _json(stages / "S6_INDEPENDENT_CONFIRMATION" / "validation.json")
    gates = audit.get("gates", {})
    lines = [
        "# ICA and Whitening Evaluation Concluding Report", "",
        f"**Overall gate state:** {audit.get('status', 'incomplete')}  ",
        f"**Completion claim allowed:** {str(bool(audit.get('completion_claim_allowed', False))).lower()}", "",
        "This report is generated from validated artifacts. A missing or failed gate is reported as pending; "
        "it is never inferred from implementation intent.", "",
        "## Computational findings", "",
    ]
    if gates.get("complete_factorial_exact_coverage") and s3:
        lines.extend([
            f"- Exact coverage passed for {int(s3['observed_fit_count']):,} unique eligible fits.",
            f"- {int(s3['real_data_converged_count']):,} fits converged on the real recording; "
            f"{int(s3['finalist_eligible_count']):,} also passed the whitening-resolution finalist gate.",
            "- The mixed-implementation spatial run is excluded; only the clean rerun contributes evidence.",
        ])
    else:
        lines.append("- Pending: exact 30,891-fit factorial coverage has not passed.")
    lines.extend(["", "## Within-recording sparse-positive findings", ""])
    if gates.get("fold_specific_protected_confirmation") and s4:
        protected = s4["protected_held_out_performance"]
        interval = protected["burst_bootstrap_95_interval"]
        lines.extend([
            f"- Fold-specific protected macro known-positive recall at budget 58: "
            f"{_fmt(protected['macro_held_out_burst_recall_at_58'])} "
            f"(burst-bootstrap 95% interval {_fmt(interval[0])} to {_fmt(interval[1])}).",
            "- Each burst was evaluated only for configurations selected without that burst's labels.",
            "- Seeds and multiple finalists within a burst are robustness checks, not independent biological replicates.",
            "- Unmatched candidates remain unknown; precision, specificity, and false-positive rate are not identified.",
        ])
    else:
        lines.append("- Pending: fold-specific protected refits and burst-level aggregation have not passed.")
    lines.extend(["", "## Learned-parameter and interpretability findings", ""])
    if gates.get("conditional_statistics_and_interpretability") and s3:
        lines.extend([
            "- Frequency-response distributions are summarized against ICA rank and temporal/spatial support.",
            "- Whitening condition and effective-rank distributions are summarized by geometry, scope, support, and blend.",
            "- Factor effects are stratified by ICA family and whitening geometry to reduce conditional-design confounding.",
            "- Scrambled Sobol coordinates support descriptive sensitivity summaries, not formal Sobol indices.",
            "- Individual component interpretation remains conditional on the S4 seed-alignment stability gate.",
        ])
    else:
        lines.append("- Pending: complete learned-response and conditional-factor artifacts have not passed.")
    lines.extend(["", "## Scientific-audit and promotion findings", ""])
    if gates.get("all_finalist_scientific_visual_audit") and s5 and s5v:
        lines.extend([
            f"- All {int(s5['finalist_count'])} finalist packages passed inventory, decode, and frozen visual inspection.",
            f"- Promotion allowed by the scientific audit: {str(bool(s5v['promotion_allowed_by_audit'])).lower()}.",
            "- Visual passage does not establish biological source identity or precision.",
        ])
    else:
        lines.append("- Pending: promotion is false until every finalist package passes decode and frozen visual inspection.")
    lines.extend(["", "## Independent-recording findings", ""])
    if gates.get("independent_confirmation_exact_coverage") and s6 and s6v:
        lines.extend([
            f"- Independent confirmation passed exact coverage on recording `{s6['recording_id']}` "
            f"for {int(s6['fit_seed_count'])} finalist-seed refits.",
            f"- The label-blind universe contained {int(s6['candidate_count']):,} candidates; every score vector "
            "was frozen before sparse-positive labels were joined.",
        ])
        for finalist in s6["finalists"]:
            lines.append(
                f"- `{finalist['source_fit_id']}`: mean pooled known-positive recall at 58 per block "
                f"{_fmt(finalist['mean_pooled_known_positive_recall_at_58'])}; "
                f"MRR {_fmt(finalist['mean_reciprocal_rank'])}."
            )
        lines.extend([
            "- This is one independent recording and does not establish population-level generalization.",
            "- Precision, specificity, false-positive rate, and biological source identity remain unidentified.",
        ])
    else:
        lines.append("- Pending: exact independent-recording confirmation has not passed.")
    lines.extend(["", "## Final claim boundary", ""])
    if audit.get("completion_claim_allowed"):
        lines.append(
            "All requested computational and audit gates passed. Conclusions remain limited to sparse-positive "
            "retrieval and the explicitly tested recordings; synthetic truth recovery did not validate biological source identity."
        )
    else:
        incomplete = ", ".join(audit.get("incomplete_gates", [])) or "unknown gates"
        lines.append(f"No completion or promotion claim is allowed. Incomplete gates: {incomplete}.")
    return "\n".join(lines) + "\n"


def write_concluding_report(experiment_root: str | Path, destination: str | Path) -> str:
    text = build_concluding_report(experiment_root)
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return text
