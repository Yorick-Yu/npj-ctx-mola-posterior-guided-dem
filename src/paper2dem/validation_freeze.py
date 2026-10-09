"""Blind bridge-validation decisions required before the G5 test unlock."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .e7_prefix import E7_PATH_COUNTS, E7_STEP_COUNTS
from .evaluation import curve_areas
from .hashing import canonical_sha256
from .incremental_anchor import (
    ANCHORED_METHOD,
    ISO_METHOD,
    OT_METHOD,
    IncrementalAnchorPolicy,
    anchored_curve,
)
from .training import FORMAL_SEEDS


E7_METHODS = (ISO_METHOD, OT_METHOD)
PRIMARY_CURVE_METHODS = (ISO_METHOD, ANCHORED_METHOD)
E7_BUDGETS = (20, 100)
E7_REPLICATES = tuple(range(5))
CURVE_BUDGETS = (0, 5, 10, 20, 50, 100)
TAU_GRID_STRICT_FIRST = (0.6, 0.7, 0.8, 0.9)


def _finite(row: Mapping[str, Any], name: str) -> float:
    value = float(row.get(name, np.nan))
    if not np.isfinite(value):
        raise ValueError(f"validation row {name} must be finite")
    return value


def build_e7_validation_report(
    rows: Sequence[Mapping[str, Any]], *, expected_tiles: Sequence[str]
) -> dict[str, Any]:
    """Select the cheapest M/N whose numerical doubling is already stable."""

    tiles = tuple(str(value) for value in expected_tiles)
    if len(tiles) != 12 or len(set(tiles)) != 12:
        raise ValueError("E7 selection requires exactly 12 frozen bridge_val tiles")
    expected = {
        (tile, method, budget, replicate, seed, path_count, step_count)
        for tile in tiles
        for method in E7_METHODS
        for budget in E7_BUDGETS
        for replicate in E7_REPLICATES
        for seed in FORMAL_SEEDS
        for path_count in E7_PATH_COUNTS
        for step_count in E7_STEP_COUNTS
    }
    observed: dict[tuple[Any, ...], dict[str, float]] = {}
    canonical_rows: list[dict[str, Any]] = []
    for raw in rows:
        key = (
            str(raw.get("tile_id", "")),
            str(raw.get("method", "")),
            int(raw.get("budget", -1)),
            int(raw.get("noise_replicate", -1)),
            int(raw.get("training_seed", -1)),
            int(raw.get("path_count", -1)),
            int(raw.get("step_count", -1)),
        )
        if key in observed:
            raise ValueError(f"duplicate E7 validation row: {key}")
        metrics = {
            "rmse": _finite(raw, "rmse"),
            "coverage_90": _finite(raw, "coverage_90"),
            "coverage_95": _finite(raw, "coverage_95"),
        }
        if metrics["rmse"] <= 0.0 or any(
            not 0.0 <= metrics[name] <= 1.0
            for name in ("coverage_90", "coverage_95")
        ):
            raise ValueError("E7 RMSE/coverage lies outside its valid range")
        observed[key] = metrics
        canonical_rows.append(
            {
                "tile_id": key[0],
                "method": key[1],
                "budget": key[2],
                "noise_replicate": key[3],
                "training_seed": key[4],
                "path_count": key[5],
                "step_count": key[6],
                **metrics,
            }
        )
    if set(observed) != expected:
        raise ValueError(
            "E7 validation grid is not exact: "
            f"missing={len(expected - set(observed))}, "
            f"unexpected={len(set(observed) - expected)}"
        )

    # A numerical choice is accepted only when every method x budget stratum is
    # stable after averaging the paired tiles, seeds, and observation replicates.
    strata: dict[tuple[str, int, int, int], dict[str, float]] = {}
    for method in E7_METHODS:
        for budget in E7_BUDGETS:
            for path_count in E7_PATH_COUNTS:
                for step_count in E7_STEP_COUNTS:
                    selected = [
                        metrics
                        for key, metrics in observed.items()
                        if key[1] == method
                        and key[2] == budget
                        and key[5] == path_count
                        and key[6] == step_count
                    ]
                    strata[(method, budget, path_count, step_count)] = {
                        name: float(np.mean([item[name] for item in selected]))
                        for name in ("rmse", "coverage_90", "coverage_95")
                    }

    def comparison(
        path_count: int, step_count: int, *, axis: str
    ) -> dict[str, float | bool | int | None]:
        if axis == "M":
            index = E7_PATH_COUNTS.index(path_count)
            if index == len(E7_PATH_COUNTS) - 1:
                return {
                    "terminal": True,
                    "compared_to": None,
                    "max_relative_rmse_change": 0.0,
                    "max_coverage90_change": 0.0,
                    "max_coverage95_change": 0.0,
                    "stable": True,
                }
            next_path = E7_PATH_COUNTS[index + 1]
            pairs = [
                (
                    strata[(method, budget, path_count, step_count)],
                    strata[(method, budget, next_path, step_count)],
                )
                for method in E7_METHODS
                for budget in E7_BUDGETS
            ]
            compared_to = next_path
        elif axis == "N":
            index = E7_STEP_COUNTS.index(step_count)
            if index == len(E7_STEP_COUNTS) - 1:
                return {
                    "terminal": True,
                    "compared_to": None,
                    "max_relative_rmse_change": 0.0,
                    "max_coverage90_change": 0.0,
                    "max_coverage95_change": 0.0,
                    "stable": True,
                }
            next_step = E7_STEP_COUNTS[index + 1]
            pairs = [
                (
                    strata[(method, budget, path_count, step_count)],
                    strata[(method, budget, path_count, next_step)],
                )
                for method in E7_METHODS
                for budget in E7_BUDGETS
            ]
            compared_to = next_step
        else:  # pragma: no cover - internal call is exhaustive
            raise AssertionError(axis)
        rmse = max(abs(left["rmse"] - right["rmse"]) / right["rmse"] for left, right in pairs)
        coverage90 = max(
            abs(left["coverage_90"] - right["coverage_90"])
            for left, right in pairs
        )
        coverage95 = max(
            abs(left["coverage_95"] - right["coverage_95"])
            for left, right in pairs
        )
        return {
            "terminal": False,
            "compared_to": compared_to,
            "max_relative_rmse_change": rmse,
            "max_coverage90_change": coverage90,
            "max_coverage95_change": coverage95,
            "stable": rmse < 0.01 and coverage90 < 0.01 and coverage95 < 0.01,
        }

    candidates: list[dict[str, Any]] = []
    for path_count in E7_PATH_COUNTS:
        for step_count in E7_STEP_COUNTS:
            m_result = comparison(path_count, step_count, axis="M")
            n_result = comparison(path_count, step_count, axis="N")
            candidates.append(
                {
                    "path_count": path_count,
                    "step_count": step_count,
                    "nominal_cost": path_count * step_count,
                    "M_doubling": m_result,
                    "N_doubling": n_result,
                    "eligible": bool(m_result["stable"] and n_result["stable"]),
                }
            )
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:  # The terminal 32x40 cell is an explicit safe fallback.
        raise AssertionError("E7 grid has no terminal fallback")
    winner = min(
        eligible,
        key=lambda item: (
            int(item["nominal_cost"]),
            int(item["path_count"]),
            int(item["step_count"]),
        ),
    )
    canonical_rows.sort(
        key=lambda item: (
            item["tile_id"],
            item["method"],
            item["budget"],
            item["noise_replicate"],
            item["training_seed"],
            item["path_count"],
            item["step_count"],
        )
    )
    return {
        "schema": "paper2.e7-validation-report/v1",
        "status": "complete",
        "role": "bridge_val",
        "test_results_inspected": False,
        "selection_rule": (
            "minimum-MxN-cost-with-rmse-change-lt-1pct-and-coverage-change-lt-1pp"
        ),
        "selected_path_count": int(winner["path_count"]),
        "selected_step_count": int(winner["step_count"]),
        "tile_ids": sorted(tiles),
        "row_count": len(canonical_rows),
        "rows_semantic_sha256": canonical_sha256(canonical_rows),
        "stratum_aggregation": "equal-weight mean over tiles, seeds, and noise replicates",
        "candidate_decisions": candidates,
    }


def build_validation_curve_freeze(
    rows: Sequence[Mapping[str, Any]], *, expected_tiles: Sequence[str]
) -> dict[str, Any]:
    """Freeze tau and the paired A/V effects from complete blind validation curves."""

    tiles = tuple(str(value) for value in expected_tiles)
    if len(tiles) != 12 or len(set(tiles)) != 12:
        raise ValueError("validation curves require exactly 12 bridge_val tiles")
    values: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for raw in rows:
        tile = str(raw.get("tile_id", ""))
        method = str(raw.get("method", ""))
        budget = int(raw.get("budget", -1))
        seed = int(raw.get("training_seed", -1))
        replicate = int(raw.get("noise_replicate", -1))
        if (
            tile not in tiles
            or method not in E7_METHODS
            or budget not in CURVE_BUDGETS
            or seed not in FORMAL_SEEDS
            or replicate not in E7_REPLICATES
        ):
            raise ValueError("validation curve row identity lies outside the frozen grid")
        mse = _finite(raw, "mse")
        coarse_mse = _finite(raw, "coarse_mse")
        if mse <= 0.0 or coarse_mse <= 0.0:
            raise ValueError("validation curve MSE values must be positive")
        values[(tile, method, budget)].append((mse, coarse_mse))
    expected_count = len(FORMAL_SEEDS) * len(E7_REPLICATES)
    expected_keys = {
        (tile, method, budget)
        for tile in tiles
        for method in E7_METHODS
        for budget in CURVE_BUDGETS
    }
    if set(values) != expected_keys or any(
        len(group) != expected_count for group in values.values()
    ):
        raise ValueError("validation curve grid is incomplete or duplicated")

    policy = IncrementalAnchorPolicy()
    curves: dict[tuple[str, str], np.ndarray] = {}
    coarse: dict[str, float] = {}
    for tile in tiles:
        coarse_values = [
            coarse_value
            for method in E7_METHODS
            for budget in CURVE_BUDGETS
            for _, coarse_value in values[(tile, method, budget)]
        ]
        if not np.allclose(coarse_values, coarse_values[0], rtol=0.0, atol=0.0):
            raise ValueError("paired validation methods do not share one coarse MSE")
        coarse[tile] = float(coarse_values[0])
        for method in E7_METHODS:
            curves[(tile, method)] = np.asarray(
                [
                    np.mean([mse for mse, _ in values[(tile, method, budget)]])
                    for budget in CURVE_BUDGETS
                ],
                dtype=np.float64,
            )
        curves[(tile, ANCHORED_METHOD)] = anchored_curve(
            budgets=CURVE_BUDGETS,
            iso_curve=curves[(tile, ISO_METHOD)],
            ot_curve=curves[(tile, OT_METHOD)],
            policy=policy,
        )

    tau_diagnostics = []
    selected_tau: float | None = None
    for tau in TAU_GRID_STRICT_FIRST:
        reached = {
            method: [
                bool(np.min(curves[(tile, method)] / coarse[tile]) <= tau)
                for tile in tiles
            ]
            for method in PRIMARY_CURVE_METHODS
        }
        leave_one_out = {
            method: [
                bool(
                    np.min(
                        np.mean(
                            [
                                curves[(tile, method)] / coarse[tile]
                                for tile in tiles
                                if tile != omitted
                            ],
                            axis=0,
                        )
                    )
                    <= tau
                )
                for omitted in tiles
            ]
            for method in PRIMARY_CURVE_METHODS
        }
        eligible = all(
            np.mean(reached[method]) >= 0.80 and all(leave_one_out[method])
            for method in PRIMARY_CURVE_METHODS
        )
        tau_diagnostics.append(
            {
                "tau": tau,
                "tile_reach_fraction": {
                    method: float(np.mean(reached[method]))
                    for method in PRIMARY_CURVE_METHODS
                },
                "all_leave_one_tile_out_curves_reach": {
                    method: all(leave_one_out[method])
                    for method in PRIMARY_CURVE_METHODS
                },
                "eligible": eligible,
            }
        )
        if eligible and selected_tau is None:
            selected_tau = tau
    if selected_tau is None:
        raise ValueError(
            "no preregistered tau is estimable on blind bridge_val; savings must remain unavailable"
        )

    effects = []
    for tile in sorted(tiles):
        iso = curve_areas(
            CURVE_BUDGETS,
            curves[(tile, ISO_METHOD)],
            coarse_mse=coarse[tile],
        )
        anchored = curve_areas(
            CURVE_BUDGETS,
            curves[(tile, ANCHORED_METHOD)],
            coarse_mse=coarse[tile],
        )
        effects.append(
            {
                "tile_id": tile,
                "delta_A": float(iso["A"] - anchored["A"]),
                "delta_V": float(anchored["V"] - iso["V"]),
            }
        )
    power_input = {
        "schema": "paper2.power-input/v1",
        "source_role": "bridge_val",
        "test_results_inspected": False,
        "rows": effects,
    }
    return {
        "schema": "paper2.validation-curve-freeze/v1",
        "status": "complete",
        "role": "bridge_val",
        "test_results_inspected": False,
        "primary_comparison": [ANCHORED_METHOD, ISO_METHOD],
        "incremental_anchor_policy": policy.to_dict(),
        "incremental_anchor_policy_sha256": policy.sha256,
        "selected_tau": selected_tau,
        "tau_selection_rule": (
            "strictest-grid-target-reached-by-both-methods-on-at-least-80pct-tiles-"
            "and-all-leave-one-tile-out-mean-curves"
        ),
        "tau_diagnostics": tau_diagnostics,
        "power_input": power_input,
        "power_input_semantic_sha256": canonical_sha256(power_input),
    }
