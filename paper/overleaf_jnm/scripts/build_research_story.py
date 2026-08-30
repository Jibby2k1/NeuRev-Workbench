#!/usr/bin/env python3
"""Build synchronized research-story views from one canonical registry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "story" / "research_story.yaml"
GENERATED = ROOT / "generated"
REPOSITORY_ROOT = ROOT.parents[1]


def escape_tex(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        text = text.replace(old, new)
    return text


def registry() -> dict:
    payload = yaml.safe_load(REGISTRY.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported research-story schema")
    return payload


def _portable_artifact(
    artifact: str,
    *,
    capsules: list[str],
    verify_live_artifacts: bool,
    owner: str,
) -> None:
    artifact_path = (ROOT / artifact).resolve()
    if artifact_path.is_file():
        return
    local_output = "Outputs" in artifact_path.parts
    capsule_paths = [(ROOT / capsule).resolve() for capsule in capsules]
    if not verify_live_artifacts and local_output and capsule_paths:
        try:
            repository_path = artifact_path.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError as exc:
            raise FileNotFoundError(f"{owner}: artifact escapes repository: {artifact}") from exc
        registered = False
        for capsule_path in capsule_paths:
            if not capsule_path.is_file():
                continue
            capsule = json.loads(capsule_path.read_text())
            if any(row.get("repository_path") == repository_path for row in capsule.get("artifacts", [])):
                registered = True
                break
        if registered:
            return
        raise FileNotFoundError(f"{owner}: local artifact is not registered by its evidence capsule: {artifact}")
    if local_output and not capsules:
        raise FileNotFoundError(f"{owner}: local artifact has no portable evidence capsule: {artifact}")
    raise FileNotFoundError(f"{owner}: {artifact}")


def validate(payload: dict, *, verify_live_artifacts: bool = False) -> None:
    for group in ("claims", "experiments", "next_experiments"):
        rows = payload.get(group, [])
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs in {group}")
    for document in payload["documents"].values():
        if not (ROOT / document["entrypoint"]).is_file():
            raise FileNotFoundError(document["entrypoint"])
    for row in payload["claims"]:
        capsules = row.get("evidence_capsules", [])
        for artifact in row["evidence"]:
            _portable_artifact(
                artifact,
                capsules=capsules,
                verify_live_artifacts=verify_live_artifacts,
                owner=f"claim {row['id']}",
            )
    for row in payload["experiments"]:
        capsule = row.get("evidence_capsule")
        for artifact in row["artifacts"]:
            _portable_artifact(
                artifact,
                capsules=[capsule] if capsule else [],
                verify_live_artifacts=verify_live_artifacts,
                owner=f"experiment {row['id']}",
            )
    for row in payload["next_experiments"]:
        if row.get("plan") and not (ROOT / row["plan"]).is_file():
            raise FileNotFoundError(f"next experiment {row['id']}: {row['plan']}")


def status_tex(payload: dict) -> str:
    row_end = r"\\"
    rows = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\begin{tabularx}{\linewidth}{>{\bfseries}p{0.20\linewidth} X p{0.22\linewidth}}",
        r"\toprule", "Claim & Current interpretation & Status and scope " + row_end, r"\midrule",
    ]
    for claim in payload["claims"]:
        rows.append(f"{escape_tex(claim['id'].replace('_', ' ').title())} & {escape_tex(claim['statement'])} & "
                    f"{escape_tex(claim['status'])}: {escape_tex(claim['scope'])} " + row_end)
    rows += [r"\bottomrule", r"\end{tabularx}",
             r"\caption{Generated current-state claim ledger. Status is scoped to the stated evidence boundary.}",
             r"\label{tab:overview-status}", r"\end{table}", ""]
    return "\n".join(rows)


def experiment_tex(payload: dict) -> str:
    rows = [r"\section{Experiment story}",
            "This document is generated from the canonical research-story registry. Each entry separates the question, design, result or bounded engineering history, limitation, and next decision; status lines distinguish validated evidence from non-claim-bearing screens."]
    for index, item in enumerate(payload["experiments"], 1):
        rows += [f"\\subsection{{{index}. {escape_tex(item['id'].replace('_', ' ').title())}}}",
                 f"\\textbf{{Question.}} {escape_tex(item['question'])}",
                 f"\\textbf{{Design.}} {escape_tex(item['design'])}",
                 f"\\textbf{{Status.}} {escape_tex(item['status'])}",
                 f"\\textbf{{Finding.}} {escape_tex(item['finding'])}",
                 f"\\textbf{{Boundary.}} {escape_tex(item['limitation'])}",
                 f"\\textbf{{Next decision.}} {escape_tex(item['next_decision'])}"]
    rows += [r"\section{Prioritized next experiments}", r"\begin{enumerate}"]
    for item in sorted(payload["next_experiments"], key=lambda row: row["priority"]):
        suffix = f" Plan: \\texttt{{{escape_tex(item['plan'])}}}." if item.get("plan") else ""
        rows.append(f"\\item {escape_tex(item['objective'])}{suffix}")
    rows += [r"\end{enumerate}", ""]
    return "\n\n".join(rows)


def experiment_md(payload: dict) -> str:
    rows = ["# Experiment story", "", "Generated from the repository research registry through `story/research_story.yaml`; do not edit by hand.", "", "Entries include validated evidence-bearing experiments and explicitly bounded engineering-run history. Status and boundary fields must not be collapsed.", ""]
    for index, item in enumerate(payload["experiments"], 1):
        rows += [f"## {index}. {item['id'].replace('_', ' ').title()}", "",
                 f"- **Question:** {item['question']}", f"- **Design:** {item['design']}",
                 f"- **Status:** `{item['status']}`", f"- **Finding:** {item['finding']}",
                 f"- **Boundary:** {item['limitation']}", f"- **Next decision:** {item['next_decision']}",
                 "- **Artifacts:** " + ", ".join(f"`{path}`" for path in item["artifacts"]), ""]
    rows += ["## Prioritized next experiments", ""]
    for item in sorted(payload["next_experiments"], key=lambda row: row["priority"]):
        suffix = f" Plan: `{item['plan']}`." if item.get("plan") else ""
        rows.append(f"{item['priority']}. **{item['id'].replace('_', ' ').title()}:** {item['objective']}{suffix}")
    return "\n".join(rows) + "\n"


def story_index(payload: dict) -> str:
    digest = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    rows = ["# Research story index", "", "Generated from the repository research registry through `story/research_story.yaml`; do not edit by hand.", "",
            f"Registry SHA-256: `{digest}`", "", "## Document profiles", ""]
    for name, doc in payload["documents"].items():
        rows.append(f"- **{name}:** `{doc['entrypoint']}` — {doc['purpose']}")
    rows += ["", "## Shared source-of-truth rules", "",
             "- Scientific prose remains canonical in `main.tex` and its section files.",
            "- Claim status, evidence membership, and experiment decisions are canonical in `../../research/registry/`; the paper story is a generated compatibility view.",
             "- Generated files must be refreshed after changing the registry or evidence artifacts.",
             "- A generated view may simplify prose but must not widen a claim's registered scope.", ""]
    return "\n".join(rows)


def outputs(payload: dict) -> dict[Path, str]:
    return {
        GENERATED / "story_status_table.tex": status_tex(payload),
        GENERATED / "experiment_story.tex": experiment_tex(payload),
        ROOT / "EXPERIMENT_STORY.md": experiment_md(payload),
        ROOT / "STORY_INDEX.md": story_index(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated views are stale")
    parser.add_argument(
        "--verify-live-artifacts",
        action="store_true",
        help="require ignored local output artifacts in addition to portable evidence capsules",
    )
    args = parser.parse_args()
    payload = registry(); validate(payload, verify_live_artifacts=args.verify_live_artifacts); expected = outputs(payload)
    stale = [path for path, text in expected.items() if not path.is_file() or path.read_text() != text]
    if args.check:
        if stale:
            raise SystemExit("stale research-story outputs: " + ", ".join(str(path.relative_to(ROOT)) for path in stale))
        print(f"research story current: {len(payload['claims'])} claims, {len(payload['experiments'])} experiments")
        return 0
    for path, text in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    print(f"updated {len(expected)} story views from {REGISTRY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
