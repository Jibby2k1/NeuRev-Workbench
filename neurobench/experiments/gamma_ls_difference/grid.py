"""Coordinate-free, deterministic Gamma-LS successive-halving design.

This module deliberately stops at context enumeration and selection.  It does
not read labels, fit the local-standard-deviation floor, or process a movie.
Scale floors therefore do not belong to these shared geometry objects. The
caller supplies one metric row for every context/representation/fold cell. The
screen uses human-declared burst *windows*, but never positive coordinates or
identities.  Selection is always local to one outer training fold; no helper in
this module is permitted to pool all four folds for a protected decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Any, Mapping, Sequence


SCREEN_REPRESENTATIONS = (
    "raw",
    "difference_signed",
    "difference_energy_normalized",
)
TRAINING_FOLDS = (1, 2, 3, 4)
EXPECTED_G1_CONTEXTS = 9
MAX_G2_CONTEXTS = 18
EXPECTED_G1_EVALUATIONS = 108
MAX_G2_EVALUATIONS = 216


class GammaGridError(ValueError):
    """Raised when the frozen Gamma-LS grid or coordinate-free rows drift."""


@dataclass(frozen=True, order=True)
class RadiusGuardPair:
    """A radial support/guard pair retained between G1 and G2."""

    half_width_px: int
    guard_radius_px: int


@dataclass(frozen=True)
class GammaContext:
    """One eligible modern radial Gamma-LS context."""

    context_id: str
    stage: str
    half_width_px: int
    guard_radius_px: int
    shape: float
    mode_fraction_of_half_width: float
    mode_radius_px: float
    support: str = "radial_disk"
    padding: str = "valid_renormalized_zero"
    eligible_primary: bool = True

    @property
    def radius_guard_pair(self) -> RadiusGuardPair:
        return RadiusGuardPair(self.half_width_px, self.guard_radius_px)

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "stage": self.stage,
            "half_width_px": self.half_width_px,
            "guard_radius_px": self.guard_radius_px,
            "shape": self.shape,
            "mode_fraction_of_half_width": self.mode_fraction_of_half_width,
            "mode_radius_px": self.mode_radius_px,
            "support": self.support,
            "padding": self.padding,
            "eligible_primary": self.eligible_primary,
        }


@dataclass(frozen=True)
class DiagnosticControl:
    """Metadata for a diagnostic that is ineligible for primary selection."""

    control_id: str
    family: str
    metadata: Mapping[str, Any]
    eligible_primary: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "family": self.family,
            "eligible_primary": False,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContextAggregate:
    """Three-representation summary for one outer training fold."""

    stage: str
    context_id: str
    training_fold: int
    mean_positive_tail_contrast: float
    minimum_representation_positive_tail_contrast: float
    mean_runtime_ms_per_frame: float
    pareto_layer: int
    cell_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "context_id": self.context_id,
            "training_fold": self.training_fold,
            "mean_positive_tail_contrast": self.mean_positive_tail_contrast,
            "minimum_representation_positive_tail_contrast": (
                self.minimum_representation_positive_tail_contrast
            ),
            "mean_runtime_ms_per_frame": self.mean_runtime_ms_per_frame,
            "pareto_layer": self.pareto_layer,
            "cell_count": self.cell_count,
        }


@dataclass(frozen=True)
class G1Selection:
    """Exactly two geometry pairs retained by the G1 screen."""

    retained_pairs: tuple[RadiusGuardPair, RadiusGuardPair]
    retained_context_ids: tuple[str, str]
    ranked_contexts: tuple[ContextAggregate, ...]


@dataclass(frozen=True)
class G2Selection:
    """One common Gamma-LS context retained by the G2 screen."""

    finalist_context_id: str
    ranked_contexts: tuple[ContextAggregate, ...]


@dataclass(frozen=True)
class SuccessiveHalvingSelection:
    """Frozen G1 geometry reduction followed by one common G2 finalist."""

    g1: G1Selection
    g2: G2Selection
    g2_contexts: tuple[GammaContext, ...]


def _grid_mapping(config: Any) -> Mapping[str, Any]:
    payload = getattr(config, "payload", config)
    if not isinstance(payload, Mapping):
        raise GammaGridError("config must be a manifest or mapping")
    grid = payload.get("gamma_ls_grid", payload)
    if not isinstance(grid, Mapping):
        raise GammaGridError("gamma_ls_grid must be a mapping")
    return grid


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise GammaGridError(f"{name} must be finite and positive")
    return result


def _number_token(value: float) -> str:
    decimal = Decimal(str(float(value))).normalize()
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def _context_id(
    half_width_px: int,
    guard_radius_px: int,
    shape: float,
    mode_fraction: float,
) -> str:
    return (
        f"gamma_h{half_width_px}_g{guard_radius_px}"
        f"_n{_number_token(shape)}_m{_number_token(mode_fraction)}"
    )


def _validate_pair(pair: RadiusGuardPair) -> None:
    if pair.half_width_px <= 0 or pair.guard_radius_px <= 0:
        raise GammaGridError("modern Gamma-LS radii and guards must be positive")
    if pair.guard_radius_px >= pair.half_width_px:
        raise GammaGridError("guard radius must be smaller than kernel half-width")


def _modern_context(
    *,
    stage: str,
    pair: RadiusGuardPair,
    shape: float,
    mode_fraction: float,
    grid: Mapping[str, Any],
) -> GammaContext:
    _validate_pair(pair)
    shape_value = _finite_positive(shape, "Gamma shape")
    mode_value = _finite_positive(mode_fraction, "mode fraction")
    if grid.get("circular_support") is not True:
        raise GammaGridError("primary Gamma-LS contexts require circular support")
    if grid.get("padding") != "valid_renormalized_zero":
        raise GammaGridError("modern Gamma-LS padding contract changed")
    return GammaContext(
        context_id=_context_id(
            pair.half_width_px,
            pair.guard_radius_px,
            shape_value,
            mode_value,
        ),
        stage=stage,
        half_width_px=pair.half_width_px,
        guard_radius_px=pair.guard_radius_px,
        shape=shape_value,
        mode_fraction_of_half_width=mode_value,
        mode_radius_px=mode_value * pair.half_width_px,
    )


def enumerate_g1_contexts(config: Any) -> tuple[GammaContext, ...]:
    """Enumerate the frozen 3-by-3 guarded radial G1 grid.

    Scale floors are deliberately absent from geometry objects.  They are fit
    per context/representation/fold/quiet-swap by the screen executor.
    """

    grid = _grid_mapping(config)
    half_widths = tuple(int(value) for value in grid.get("g1_half_width_px", ()))
    guards = tuple(int(value) for value in grid.get("g1_guard_radius_px", ()))
    shape = float(grid.get("g1_shape", float("nan")))
    mode_fraction = float(
        grid.get("g1_mode_fraction_of_half_width", float("nan"))
    )
    retain = int(grid.get("g1_retain_radius_guard_pairs", -1))
    if half_widths != (7, 11, 15) or guards != (1, 3, 5):
        raise GammaGridError("G1 must use half-widths 7/11/15 and guards 1/3/5")
    if shape != 5.0 or mode_fraction != 0.75 or retain != 2:
        raise GammaGridError("G1 shape, mode fraction, or retention count changed")

    contexts = tuple(
        _modern_context(
            stage="g1",
            pair=RadiusGuardPair(half_width, guard),
            shape=shape,
            mode_fraction=mode_fraction,
            grid=grid,
        )
        for half_width in half_widths
        for guard in guards
    )
    if len(contexts) != EXPECTED_G1_CONTEXTS:
        raise AssertionError("frozen G1 grid must contain exactly nine contexts")
    if len({context.context_id for context in contexts}) != len(contexts):
        raise AssertionError("G1 context identifiers must be unique")
    return contexts


def enumerate_g2_contexts(
    config: Any,
    retained_pairs: Sequence[RadiusGuardPair | tuple[int, int]],
) -> tuple[GammaContext, ...]:
    """Enumerate G2 for exactly two G1-retained radius/guard pairs."""

    grid = _grid_mapping(config)
    parsed = tuple(
        pair
        if isinstance(pair, RadiusGuardPair)
        else RadiusGuardPair(int(pair[0]), int(pair[1]))
        for pair in retained_pairs
    )
    if len(parsed) != 2 or len(set(parsed)) != 2:
        raise GammaGridError("G2 requires exactly two distinct retained G1 pairs")
    valid_g1_pairs = {
        context.radius_guard_pair
        for context in enumerate_g1_contexts(config)
    }
    if not set(parsed).issubset(valid_g1_pairs):
        raise GammaGridError("G2 retained pairs must come from the frozen G1 grid")
    shapes = tuple(float(value) for value in grid.get("g2_shape", ()))
    mode_fractions = tuple(
        float(value) for value in grid.get("g2_mode_fraction_of_half_width", ())
    )
    if shapes != (2.0, 5.0, 9.0) or mode_fractions != (0.5, 0.75, 1.0):
        raise GammaGridError("G2 must use shapes 2/5/9 and mode fractions 0.5/0.75/1")

    contexts = tuple(
        _modern_context(
            stage="g2",
            pair=pair,
            shape=shape,
            mode_fraction=mode_fraction,
            grid=grid,
        )
        for pair in sorted(parsed)
        for shape in shapes
        for mode_fraction in mode_fractions
    )
    if len(contexts) > MAX_G2_CONTEXTS:
        raise AssertionError("G2 exceeded the frozen maximum of 18 contexts")
    if len(contexts) != 18 or len({context.context_id for context in contexts}) != 18:
        raise AssertionError("two retained pairs must yield 18 unique G2 contexts")
    return contexts


def legacy_exact_diagnostic(config: Any) -> DiagnosticControl:
    """Return the exact historical anchor, outside both primary grids."""

    grid = _grid_mapping(config)
    if grid.get("include_legacy_exact_anchor") is not True:
        raise GammaGridError("legacy_exact diagnostic must remain enabled")
    legacy = grid.get("legacy_exact")
    expected = {
        "half_width_px": 11,
        "guard_radius_px": 0,
        "shape": 9.0,
        "mode_radius_px": 35.0,
        "circular_support": False,
        "additive_epsilon": 64.0,
    }
    if legacy != expected:
        raise GammaGridError("legacy_exact diagnostic contract changed")
    return DiagnosticControl(
        control_id="legacy_exact_h11_g0_n9_m35_eps64",
        family="legacy_exact_gamma_local_standardization",
        metadata={
            **expected,
            "support": "square_23x23_center_only_exclusion",
            "padding": "reflect",
            "scale_floor_source": "fixed_additive_epsilon_not_fitted",
        },
    )


def square_box_control_metadata(config: Any) -> DiagnosticControl:
    """Return metadata only for the ineligible square outer-minus-guard control."""

    grid = _grid_mapping(config)
    if grid.get("include_square_box_control") is not True:
        raise GammaGridError("square-box diagnostic must remain enabled")
    square = grid.get("square_box_control")
    expected = {"outer_half_width_px": 11, "guard_half_width_px": 3}
    if square != expected:
        raise GammaGridError("square-box diagnostic contract changed")
    return DiagnosticControl(
        control_id="square_box_outer_h11_guard_h3",
        family="square_outer_minus_guard_cfar",
        metadata={
            **expected,
            "support": "square_outer_minus_square_guard",
            "role": "named_diagnostic_control_only",
        },
    )


def assert_screen_design_counts(
    g1_contexts: Sequence[GammaContext],
    g2_contexts: Sequence[GammaContext],
) -> dict[str, int]:
    """Assert the frozen 3-representation by 4-fold screen dimensions."""

    if len(g1_contexts) != EXPECTED_G1_CONTEXTS:
        raise GammaGridError("G1 design must contain exactly nine contexts")
    if len(g2_contexts) > MAX_G2_CONTEXTS:
        raise GammaGridError("G2 design exceeds 18 contexts")
    g1_cells = len(g1_contexts) * len(SCREEN_REPRESENTATIONS) * len(TRAINING_FOLDS)
    g2_cells = len(g2_contexts) * len(SCREEN_REPRESENTATIONS) * len(TRAINING_FOLDS)
    if g1_cells != EXPECTED_G1_EVALUATIONS:
        raise AssertionError("G1 design must contain exactly 108 evaluations")
    if g2_cells > MAX_G2_EVALUATIONS:
        raise AssertionError("G2 design exceeded 216 evaluations")
    return {
        "g1_contexts": len(g1_contexts),
        "g1_evaluations": g1_cells,
        "g2_contexts": len(g2_contexts),
        "g2_evaluations": g2_cells,
        "g2_max_evaluations": MAX_G2_EVALUATIONS,
    }


_METRIC_ROW_KEYS = {
    "stage",
    "context_id",
    "representation",
    "training_fold",
    "event_positive_tail",
    "quiet_positive_tail",
    "runtime_ms_per_frame",
    "positive_coordinates_used",
    "positive_identities_used",
    "burst_windows_used",
}


def _validated_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    contexts: Sequence[GammaContext],
    *,
    stage: str,
    training_fold: int,
) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        raise GammaGridError(f"{stage.upper()} selection rows cannot be empty")
    context_ids = {context.context_id for context in contexts}
    if len(context_ids) != len(contexts):
        raise GammaGridError("eligible context identifiers must be unique")
    grouped: dict[str, list[dict[str, Any]]] = {
        context_id: [] for context_id in context_ids
    }
    seen: set[tuple[str, str, int]] = set()
    for source in rows:
        if not isinstance(source, Mapping):
            raise GammaGridError("each selection row must be a mapping")
        actual = set(source)
        if actual != _METRIC_ROW_KEYS:
            missing = sorted(_METRIC_ROW_KEYS - actual)
            unknown = sorted(actual - _METRIC_ROW_KEYS)
            raise GammaGridError(
                f"invalid coordinate-free metric row fields: missing={missing}; unknown={unknown}"
            )
        if (
            source["positive_coordinates_used"] is not False
            or source["positive_identities_used"] is not False
        ):
            raise GammaGridError(
                "Gamma context selection cannot use positive coordinates or identities"
            )
        if source["burst_windows_used"] is not True:
            raise GammaGridError(
                "Gamma context selection must disclose its human-declared burst windows"
            )
        if source["stage"] != stage:
            raise GammaGridError(f"all rows must belong to stage {stage!r}")
        context_id = str(source["context_id"])
        if context_id not in context_ids:
            raise GammaGridError(f"row references ineligible context {context_id!r}")
        representation = str(source["representation"])
        if representation not in SCREEN_REPRESENTATIONS:
            raise GammaGridError(
                "selection accepts only raw, difference_signed, and "
                "difference_energy_normalized"
            )
        fold = int(source["training_fold"])
        if fold not in TRAINING_FOLDS or fold != source["training_fold"]:
            raise GammaGridError("training_fold must be an integer in 1..4")
        if fold != training_fold:
            raise GammaGridError(
                "selection rows must contain only the requested outer training fold"
            )
        event_tail = float(source["event_positive_tail"])
        quiet_tail = float(source["quiet_positive_tail"])
        runtime = float(source["runtime_ms_per_frame"])
        if not (math.isfinite(event_tail) and math.isfinite(quiet_tail)):
            raise GammaGridError("positive-tail metrics must be finite")
        if not math.isfinite(runtime) or runtime <= 0:
            raise GammaGridError("runtime_ms_per_frame must be finite and positive")
        key = (context_id, representation, fold)
        if key in seen:
            raise GammaGridError(f"duplicate metric cell: {key!r}")
        seen.add(key)
        grouped[context_id].append(
            {
                "representation": representation,
                "training_fold": fold,
                "contrast": event_tail - quiet_tail,
                "runtime_ms_per_frame": runtime,
            }
        )

    expected_cells = {
        (context_id, representation, fold)
        for context_id in context_ids
        for representation in SCREEN_REPRESENTATIONS
        for fold in (training_fold,)
    }
    if seen != expected_cells:
        missing = sorted(expected_cells - seen)
        extra = sorted(seen - expected_cells)
        raise GammaGridError(
            f"incomplete or unbalanced {stage.upper()} design: "
            f"missing={missing[:5]}; extra={extra[:5]}"
        )
    return grouped


def _pareto_layers(values: Sequence[tuple[float, float, str]]) -> dict[str, int]:
    """Assign deterministic nondominated layers (contrast max, runtime min)."""

    remaining = {context_id: (contrast, runtime) for contrast, runtime, context_id in values}
    layers: dict[str, int] = {}
    layer = 0
    while remaining:
        front: list[str] = []
        for context_id, (contrast, runtime) in remaining.items():
            dominated = any(
                other_id != context_id
                and other_contrast >= contrast
                and other_runtime <= runtime
                and (other_contrast > contrast or other_runtime < runtime)
                for other_id, (other_contrast, other_runtime) in remaining.items()
            )
            if not dominated:
                front.append(context_id)
        if not front:  # pragma: no cover - finite scalar partial order guarantees a front
            raise AssertionError("Pareto layer construction stalled")
        for context_id in sorted(front):
            layers[context_id] = layer
            del remaining[context_id]
        layer += 1
    return layers


def rank_coordinate_free_contexts(
    rows: Sequence[Mapping[str, Any]],
    contexts: Sequence[GammaContext],
    *,
    stage: str,
    training_fold: int,
) -> tuple[ContextAggregate, ...]:
    """Rank one fold by three-arm contrast, runtime Pareto layer, then ID."""

    if training_fold not in TRAINING_FOLDS:
        raise GammaGridError("training_fold must be an integer in 1..4")
    grouped = _validated_metric_rows(
        rows, contexts, stage=stage, training_fold=training_fold
    )
    raw: list[dict[str, Any]] = []
    for context_id, cells in grouped.items():
        contrasts = [cell["contrast"] for cell in cells]
        if len(contrasts) != len(SCREEN_REPRESENTATIONS):
            raise AssertionError("validated screen lost representation balance")
        raw.append(
            {
                "context_id": context_id,
                "mean_contrast": sum(contrasts) / len(contrasts),
                "minimum_representation_contrast": min(contrasts),
                "mean_runtime": sum(cell["runtime_ms_per_frame"] for cell in cells)
                / len(cells),
                "cell_count": len(cells),
            }
        )
    layers = _pareto_layers(
        [
            (row["mean_contrast"], row["mean_runtime"], row["context_id"])
            for row in raw
        ]
    )
    aggregates = [
        ContextAggregate(
            stage=stage,
            context_id=row["context_id"],
            training_fold=training_fold,
            mean_positive_tail_contrast=float(row["mean_contrast"]),
            minimum_representation_positive_tail_contrast=float(
                row["minimum_representation_contrast"]
            ),
            mean_runtime_ms_per_frame=float(row["mean_runtime"]),
            pareto_layer=layers[row["context_id"]],
            cell_count=int(row["cell_count"]),
        )
        for row in raw
    ]
    return tuple(
        sorted(
            aggregates,
            key=lambda row: (
                row.pareto_layer,
                -row.mean_positive_tail_contrast,
                row.mean_runtime_ms_per_frame,
                row.context_id,
            ),
        )
    )


def select_g1_radius_guard_pairs(
    rows: Sequence[Mapping[str, Any]],
    contexts: Sequence[GammaContext],
    *,
    training_fold: int,
) -> G1Selection:
    """Retain exactly two G1 radius/guard pairs without sparse-positive data."""

    if len(contexts) != EXPECTED_G1_CONTEXTS or any(
        context.stage != "g1" or not context.eligible_primary for context in contexts
    ):
        raise GammaGridError("G1 selection requires the nine eligible G1 contexts")
    ranked = rank_coordinate_free_contexts(
        rows, contexts, stage="g1", training_fold=training_fold
    )
    selected = ranked[:2]
    if len(selected) != 2:
        raise GammaGridError("G1 must retain exactly two contexts")
    by_id = {context.context_id: context for context in contexts}
    pairs = tuple(by_id[row.context_id].radius_guard_pair for row in selected)
    if len(set(pairs)) != 2:
        raise AssertionError("G1 selected duplicate radius/guard pairs")
    return G1Selection(
        retained_pairs=(pairs[0], pairs[1]),
        retained_context_ids=(selected[0].context_id, selected[1].context_id),
        ranked_contexts=ranked,
    )


def select_g2_common_finalist(
    rows: Sequence[Mapping[str, Any]],
    contexts: Sequence[GammaContext],
    *,
    training_fold: int,
) -> G2Selection:
    """Freeze one common G2 context for every downstream representation."""

    if len(contexts) != 18 or any(
        context.stage != "g2" or not context.eligible_primary for context in contexts
    ):
        raise GammaGridError("G2 selection requires the 18 eligible G2 contexts")
    ranked = rank_coordinate_free_contexts(
        rows, contexts, stage="g2", training_fold=training_fold
    )
    if not ranked:
        raise GammaGridError("G2 produced no common finalist")
    return G2Selection(finalist_context_id=ranked[0].context_id, ranked_contexts=ranked)


def select_successive_halving(
    config: Any,
    *,
    g1_rows: Sequence[Mapping[str, Any]],
    g2_rows: Sequence[Mapping[str, Any]],
    training_fold: int,
) -> SuccessiveHalvingSelection:
    """Apply one fold's frozen G1-to-G2 coordinate-free selection rule."""

    g1_contexts = enumerate_g1_contexts(config)
    g1 = select_g1_radius_guard_pairs(
        g1_rows, g1_contexts, training_fold=training_fold
    )
    g2_contexts = enumerate_g2_contexts(config, g1.retained_pairs)
    assert_screen_design_counts(g1_contexts, g2_contexts)
    g2 = select_g2_common_finalist(
        g2_rows, g2_contexts, training_fold=training_fold
    )
    return SuccessiveHalvingSelection(g1=g1, g2=g2, g2_contexts=g2_contexts)


__all__ = [
    "DiagnosticControl",
    "EXPECTED_G1_CONTEXTS",
    "EXPECTED_G1_EVALUATIONS",
    "G1Selection",
    "G2Selection",
    "GammaContext",
    "GammaGridError",
    "MAX_G2_CONTEXTS",
    "MAX_G2_EVALUATIONS",
    "RadiusGuardPair",
    "SCREEN_REPRESENTATIONS",
    "SuccessiveHalvingSelection",
    "TRAINING_FOLDS",
    "assert_screen_design_counts",
    "enumerate_g1_contexts",
    "enumerate_g2_contexts",
    "legacy_exact_diagnostic",
    "rank_coordinate_free_contexts",
    "select_g1_radius_guard_pairs",
    "select_g2_common_finalist",
    "select_successive_halving",
    "square_box_control_metadata",
]
