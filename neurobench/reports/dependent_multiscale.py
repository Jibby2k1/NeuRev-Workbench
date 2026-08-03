"""Generated-only report rendering for dependent multiscale baselines."""
from __future__ import annotations

from collections.abc import Mapping


def render_generated_report(metrics: Mapping[str, object]) -> str:
    gates = metrics["gates"]
    return "\n".join((
        "# Dependent multiscale generated baseline", "",
        "## Scientific anchors", "",
        "Pairwise ICA is not the target. Local low-rank amplitude structure is an initialization, not a physical-source claim. Quiet-relative compact information is the strongest direct ITL evidence; broad scales remain context. The accepted scientific carrier remains separate from this candidate decomposition.", "",
        "## Result", "",
        f"The generated-only W3 baseline evaluated {metrics['fixture_count']} exact-truth fixtures. C1 numerical reconstruction: **{gates['C1_numerical_reconstruction']}**. C2 attribution is **not evaluated for promotion** because the matrix-Renyi group objective belongs to W4.", "",
        f"Maximum normalized closure error: `{metrics['maximum_normalized_closure']:.3e}`.", "",
        "All residuals retain the name `noise_candidate`; none are qualified measurement noise.", "",
        "## Scope boundary", "",
        "No Spon data, semi-synthetic injection, GPU execution, or scientific-carrier replacement was run. The exact next work package is W4: groupwise matrix-Renyi dependence, nuisance residualization tests, and joint quiet-residual objective integration.", "",
    ))
