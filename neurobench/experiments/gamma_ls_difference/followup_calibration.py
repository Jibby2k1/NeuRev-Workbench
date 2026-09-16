"""Setup-only regional order statistics, applied after unchanged global NMS."""
from __future__ import annotations

import math
import numpy as np
from .necessity_study import cutoff, BUDGETS, REFERENCE_AREA_PX


def regions(shape, rows=2, columns=3, border=6):
    h, w = map(int, shape)
    if min(h, w) <= 2 * border or rows < 1 or columns < 1:
        raise ValueError("Empty region geometry")
    ys = [border + i * (h - 2 * border) // rows for i in range(rows + 1)]
    xs = [border + i * (w - 2 * border) // columns for i in range(columns + 1)]
    result = [dict(region_id=f"r{i}c{j}", box_yxyx=[ys[i], xs[j], ys[i+1], xs[j+1]],
                   eligible_area_px=(ys[i+1]-ys[i])*(xs[j+1]-xs[j]))
              for i in range(rows) for j in range(columns)]
    if any(r["eligible_area_px"] <= 0 for r in result):
        raise ValueError("Grid contains empty regions")
    return result


def region_index(x, y, grid):
    found = [i for i, r in enumerate(grid) if r["box_yxyx"][1] <= x < r["box_yxyx"][3]
             and r["box_yxyx"][0] <= y < r["box_yxyx"][2]]
    if len(found) != 1:
        raise ValueError("Point must be in exactly one eligible region")
    return found[0]


def allocate_budget(budget, areas):
    """Hamilton allocation with integer remainders and row-major tie breaking."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("Budget must be a nonnegative integer")
    if not areas or any(isinstance(a, bool) or not isinstance(a, int) or a <= 0 for a in areas):
        raise ValueError("Positive integer areas required")
    total = sum(areas)
    allocated = [budget * a // total for a in areas]
    order = sorted(range(len(areas)), key=lambda i: (-(budget * areas[i] % total), i))
    for i in order[:budget - sum(allocated)]:
        allocated[i] += 1
    assert sum(allocated) == budget
    return allocated


def regional_plan(setup_peaks, grid, frame_count):
    if frame_count < 1:
        raise ValueError("Setup must contain frames")
    populations = [[] for _ in grid]
    for row in setup_peaks:
        populations[region_index(row["x_px"], row["y_px"], grid)].append(row["score"])
    area = sum(r["eligible_area_px"] for r in grid)
    plan = []
    for q in BUDGETS:
        total = math.floor(q * area / REFERENCE_AREA_PX * frame_count)
        budgets = allocate_budget(total, [r["eligible_area_px"] for r in grid])
        fitted = []
        for region, values, budget in zip(grid, populations, budgets):
            tau, observed = cutoff(values, budget)
            fitted.append(dict(region, threshold=tau, setup_proposal_budget=budget,
                               setup_proposal_count=observed, positive_setup_peaks=len(values)))
        plan.append(dict(threshold_id=f"q{q:g}", setup_budget_per_reference_area_frame=q,
                         threshold=None, regional_thresholds=fitted, setup_proposal_budget=total,
                         setup_proposal_count=sum(r["setup_proposal_count"] for r in fitted)))
    for name, tau in (("all_positive", 0.), ("no_output", None)):
        fitted = [dict(r, threshold=tau, setup_proposal_budget=None,
                       setup_proposal_count=len(v) if tau == 0 else 0)
                  for r, v in zip(grid, populations)]
        plan.append(dict(threshold_id=name, setup_budget_per_reference_area_frame=None,
                         threshold=tau, regional_thresholds=fitted, setup_proposal_budget=None,
                         setup_proposal_count=sum(r["setup_proposal_count"] for r in fitted)))
    return plan


def select(prefix, setting, target=None):
    """Preserve original score order/IDs/ranks; never NMS or rerank a margin."""
    selected = []
    grid = setting.get("regional_thresholds")
    for row in prefix:
        r = grid[region_index(row["x_px"], row["y_px"], grid)] if grid else setting
        tau = r["threshold"]
        if tau is not None and row["score"] > tau:
            selected.append(dict(row, threshold_z=tau, target_proposals_per_frame=target,
                                 calibration_region_id=r.get("region_id", "global")))
    return selected
