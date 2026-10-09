"""Blind validation selection of terminal footprint data-consistency parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .hashing import canonical_sha256
from .training import FORMAL_SEEDS


DC_TAU_GRID = (0.5, 1.0, 2.0)
DC_DELTA = 1.0e-4
DC_LAMBDA_GRID = (0.5, 1.0, 2.0)
DC_NONINFERIORITY_MARGIN = 0.02
DC_BUDGETS = (20, 100)
DC_REPLICATES = tuple(range(5))


def _metric(row: Mapping[str, Any], name: str) -> float:
    value = float(row.get(name, np.nan))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"DC validation {name} must be finite and nonnegative")
    return value


def build_dc_validation_report(
    rows: Sequence[Mapping[str, Any]], *, expected_tiles: Sequence[str]
) -> dict[str, Any]:
    """Choose D1 only when used, held-out, and far-field criteria all pass."""

    tiles = tuple(str(value) for value in expected_tiles)
    if len(tiles) != 12 or len(set(tiles)) != 12:
        raise ValueError("DC validation requires exactly 12 bridge_val tiles")
    base_keys = {
        (tile, budget, seed, replicate)
        for tile in tiles
        for budget in DC_BUDGETS
        for seed in FORMAL_SEEDS
        for replicate in DC_REPLICATES
    }
    configurations = tuple(
        (tau_dc, DC_DELTA, lambda_dc)
        for tau_dc in DC_TAU_GRID
        for lambda_dc in DC_LAMBDA_GRID
    )
    baseline: dict[tuple[str, int, int, int], dict[str, float]] = {}
    corrected: dict[
        tuple[str, int, int, int, float, float, float], dict[str, float]
    ] = {}
    canonical_rows: list[dict[str, Any]] = []
    for raw in rows:
        identity = (
            str(raw.get("tile_id", "")),
            int(raw.get("budget", -1)),
            int(raw.get("training_seed", -1)),
            int(raw.get("noise_replicate", -1)),
        )
        mode = str(raw.get("dc_mode", ""))
        metrics = {
            "used_footprint_rmse": _metric(raw, "used_footprint_rmse"),
            "heldout_clean_footprint_rmse": _metric(
                raw, "heldout_clean_footprint_rmse"
            ),
            "far_field_dem_rmse": _metric(raw, "far_field_dem_rmse"),
        }
        record: dict[str, Any] = {
            "tile_id": identity[0],
            "budget": identity[1],
            "training_seed": identity[2],
            "noise_replicate": identity[3],
            "dc_mode": mode,
            **metrics,
        }
        if mode == "D0":
            if identity in baseline:
                raise ValueError(f"duplicate D0 validation row: {identity}")
            baseline[identity] = metrics
        elif mode == "D1":
            parameters = (
                float(raw.get("tau_dc", np.nan)),
                float(raw.get("delta", np.nan)),
                float(raw.get("lambda_dc", np.nan)),
            )
            if parameters not in configurations:
                raise ValueError("D1 validation parameters lie outside the frozen grid")
            key = (*identity, *parameters)
            if key in corrected:
                raise ValueError(f"duplicate D1 validation row: {key}")
            corrected[key] = metrics
            record.update(
                {
                    "tau_dc": parameters[0],
                    "delta": parameters[1],
                    "lambda_dc": parameters[2],
                }
            )
        else:
            raise ValueError("DC validation rows must use D0 or D1")
        canonical_rows.append(record)
    expected_corrected = {
        (*identity, *configuration)
        for identity in base_keys
        for configuration in configurations
    }
    if set(baseline) != base_keys or set(corrected) != expected_corrected:
        raise ValueError(
            "DC validation grid is not exact: "
            f"missing_D0={len(base_keys - set(baseline))}, "
            f"missing_D1={len(expected_corrected - set(corrected))}, "
            f"unexpected_D0={len(set(baseline) - base_keys)}, "
            f"unexpected_D1={len(set(corrected) - expected_corrected)}"
        )

    decisions = []
    for configuration in configurations:
        paired = [
            (baseline[identity], corrected[(*identity, *configuration)])
            for identity in sorted(base_keys)
        ]
        used_ratio = float(
            np.mean(
                [
                    candidate["used_footprint_rmse"]
                    / max(source["used_footprint_rmse"], np.finfo(float).tiny)
                    for source, candidate in paired
                ]
            )
        )
        heldout_ratio = float(
            np.mean(
                [
                    candidate["heldout_clean_footprint_rmse"]
                    / max(
                        source["heldout_clean_footprint_rmse"],
                        np.finfo(float).tiny,
                    )
                    for source, candidate in paired
                ]
            )
        )
        far_ratio = float(
            np.mean(
                [
                    candidate["far_field_dem_rmse"]
                    / max(source["far_field_dem_rmse"], np.finfo(float).tiny)
                    for source, candidate in paired
                ]
            )
        )
        decisions.append(
            {
                "tau_dc": configuration[0],
                "delta": configuration[1],
                "lambda_dc": configuration[2],
                "mean_used_footprint_rmse_ratio_to_D0": used_ratio,
                "mean_heldout_clean_footprint_rmse_ratio_to_D0": heldout_ratio,
                "mean_far_field_dem_rmse_ratio_to_D0": far_ratio,
                "eligible": used_ratio < 1.0
                and heldout_ratio <= 1.0
                and far_ratio <= 1.0 + DC_NONINFERIORITY_MARGIN,
            }
        )
    eligible = [value for value in decisions if value["eligible"]]
    if not eligible:
        raise ValueError(
            "no DC configuration improves used footprints without held-out/far-field harm"
        )
    winner = min(
        eligible,
        key=lambda value: (
            float(value["mean_heldout_clean_footprint_rmse_ratio_to_D0"]),
            float(value["mean_used_footprint_rmse_ratio_to_D0"]),
            decisions.index(value),
        ),
    )
    canonical_rows.sort(
        key=lambda value: (
            value["tile_id"],
            value["budget"],
            value["training_seed"],
            value["noise_replicate"],
            value["dc_mode"],
            value.get("tau_dc", 0.0),
            value.get("lambda_dc", 0.0),
        )
    )
    return {
        "schema": "paper2.dc-validation-report/v1",
        "status": "complete",
        "role": "bridge_val",
        "test_results_inspected": False,
        "method": "OT-SB-full",
        "budgets": list(DC_BUDGETS),
        "tau_dc_grid": list(DC_TAU_GRID),
        "delta_grid": [DC_DELTA],
        "lambda_grid": list(DC_LAMBDA_GRID),
        "selection_rule": (
            "minimum-heldout-ratio-among-used-improving-heldout-nonworse-"
            "far-field-noninferior-configurations"
        ),
        "selected_tau_dc": float(winner["tau_dc"]),
        "selected_delta": float(winner["delta"]),
        "selected_lambda": float(winner["lambda_dc"]),
        "selected_noninferiority_margin": DC_NONINFERIORITY_MARGIN,
        "row_count": len(canonical_rows),
        "rows_semantic_sha256": canonical_sha256(canonical_rows),
        "candidate_decisions": decisions,
    }

