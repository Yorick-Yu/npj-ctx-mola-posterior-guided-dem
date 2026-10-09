from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .evaluation import curve_areas, laser_saving, paired_tile_bootstrap
from .incremental_anchor import (
    ANCHORED_METHOD,
    ISO_METHOD,
    OT_METHOD,
    IncrementalAnchorPolicy,
    anchored_curve,
)


PRIMARY_BUDGETS = (0, 5, 10, 20, 50, 100)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, preserving the named hypotheses."""

    if not p_values:
        return {}
    for name, value in p_values.items():
        if not name or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Holm inputs require named finite p-values in [0,1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[name] = min(running, 1.0)
    return adjusted


def _key(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("key")
    if not isinstance(value, Mapping):
        raise ValueError("evaluation record lacks a matrix key")
    return value


def primary_e3_analysis(
    records: Sequence[Mapping[str, Any]],
    *,
    tau: float,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Run the preregistered tile-equal E3 A/V and equal-accuracy comparison."""

    if not 0 < tau <= 1:
        raise ValueError("tau must lie in (0,1]")
    policy = IncrementalAnchorPolicy()
    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for record in records:
        key = _key(record)
        if key.get("experiment") != "E3":
            continue
        method = str(key["method"])
        if method not in {"coarse", ISO_METHOD, OT_METHOD}:
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping) or "mse" not in metrics:
            raise ValueError("E3 evaluation record lacks MSE")
        values[(str(key["tile_id"]), method, int(key["budget"]))].append(
            float(metrics["mse"])
        )
    tiles = sorted({tile for tile, method, _ in values if method == "coarse"})
    if len(tiles) not in {24, 30, 36}:
        raise ValueError("primary E3 analysis requires exactly 24, 30, or 36 locked tiles")
    tile_results: list[dict[str, Any]] = []
    for tile in tiles:
        coarse_values = values.get((tile, "coarse", 0), [])
        if len(coarse_values) != 1:
            raise ValueError(f"tile {tile} requires exactly one coarse K=0 cell")
        coarse_mse = float(coarse_values[0])
        curves: dict[str, list[float]] = {}
        areas: dict[str, dict[str, float]] = {}
        for method in (ISO_METHOD, OT_METHOD):
            curve = []
            for budget in PRIMARY_BUDGETS:
                samples = values.get((tile, method, budget), [])
                if not samples or not np.all(np.isfinite(samples)):
                    raise ValueError(f"tile {tile}/{method}/K={budget} is missing or non-finite")
                curve.append(float(np.mean(samples)))
            curves[method] = curve
            areas[method] = curve_areas(PRIMARY_BUDGETS, curve, coarse_mse=coarse_mse)
        curves[ANCHORED_METHOD] = anchored_curve(
            budgets=PRIMARY_BUDGETS,
            iso_curve=curves[ISO_METHOD],
            ot_curve=curves[OT_METHOD],
            policy=policy,
        ).tolist()
        areas[ANCHORED_METHOD] = curve_areas(
            PRIMARY_BUDGETS,
            curves[ANCHORED_METHOD],
            coarse_mse=coarse_mse,
        )
        saving = laser_saving(
            PRIMARY_BUDGETS,
            np.asarray(curves[ANCHORED_METHOD]) / coarse_mse,
            np.asarray(curves[ISO_METHOD]) / coarse_mse,
            tau,
        )
        tile_results.append(
            {
                "tile_id": tile,
                "coarse_mse": coarse_mse,
                "curves": curves,
                "areas": areas,
                "saving": saving,
                "A_improvement": areas[ISO_METHOD]["A"] - areas[ANCHORED_METHOD]["A"],
                "V_improvement": areas[ANCHORED_METHOD]["V"] - areas[ISO_METHOD]["V"],
            }
        )
    a_bootstrap = paired_tile_bootstrap(
        [item["A_improvement"] for item in tile_results],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    v_bootstrap = paired_tile_bootstrap(
        [item["V_improvement"] for item in tile_results],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 1,
    )
    reached = [item for item in tile_results if item["saving"]["saving_estimable"]]
    savings_estimable = len(reached) == len(tile_results)
    saving_bootstrap = None
    point_delta = None
    practical = False
    if savings_estimable:
        saving_bootstrap = paired_tile_bootstrap(
            [float(item["saving"]["saving_fraction"]) for item in reached],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 2,
        )
        point_delta = float(np.mean([float(item["saving"]["delta_k"]) for item in reached]))
        practical = saving_bootstrap["mean"] > 0.10 and point_delta >= 5.0
    primary_pass = a_bootstrap["one_sided_low"] > 0 and v_bootstrap["one_sided_low"] > 0
    saving_pass = bool(
        savings_estimable
        and saving_bootstrap is not None
        and saving_bootstrap["one_sided_low"] > 0
    )
    return {
        "schema": "paper2.primary-analysis/v1",
        "tile_count": len(tile_results),
        "tile_weighting": "equal",
        "tau": tau,
        "primary_comparison": [ANCHORED_METHOD, ISO_METHOD],
        "incremental_anchor_policy": policy.to_dict(),
        "incremental_anchor_policy_sha256": policy.sha256,
        "A_improvement": a_bootstrap,
        "V_improvement": v_bootstrap,
        "joint_primary_pass": primary_pass,
        "saving": {
            "all_tiles_estimable_without_extrapolation": savings_estimable,
            "estimable_tiles": len(reached),
            "bootstrap": saving_bootstrap,
            "mean_delta_k": point_delta,
            "one_sided_positive": saving_pass,
            "practical_threshold_pass": practical,
        },
        "claim_pass": bool(primary_pass and saving_pass and practical),
        "tile_results": tile_results,
    }


def _positive_effect_bootstrap(
    values: Sequence[float], *, replicates: int, seed: int
) -> tuple[dict[str, float], float]:
    source = np.asarray(values, dtype=np.float64)
    summary = dict(
        paired_tile_bootstrap(source, replicates=replicates, seed=seed)
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(source), size=(replicates, len(source)))
    estimates = source[indices].mean(axis=1)
    p_value = float((1 + np.count_nonzero(estimates <= 0.0)) / (replicates + 1))
    return summary, p_value


def e4_mechanism_analysis(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Run the locked E4 2x2 mobility/gate and spatial-placebo attribution tests."""

    methods = ("F00", "F10", "F01", "F11", "spatial-shift-M")
    units: dict[tuple[str, int, int, int], dict[str, float]] = defaultdict(dict)
    for record in records:
        key = _key(record)
        if key.get("experiment") != "E4":
            continue
        method = str(key["method"])
        if method not in methods:
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping) or "mse" not in metrics:
            raise ValueError("E4 evaluation record lacks MSE")
        unit = (
            str(key["tile_id"]),
            int(key["budget"]),
            int(key["noise_replicate"]),
            int(key["training_seed"]),
        )
        if method in units[unit]:
            raise ValueError(f"duplicate E4 method cell: {unit}/{method}")
        value = float(metrics["mse"])
        if not np.isfinite(value):
            raise ValueError("E4 MSE values must be finite")
        units[unit][method] = value
    tiles = sorted({unit[0] for unit in units})
    if len(tiles) not in {24, 30, 36}:
        raise ValueError("E4 attribution requires exactly 24, 30, or 36 locked tiles")
    tile_results: list[dict[str, Any]] = []
    for tile in tiles:
        tile_units = [
            (unit, values) for unit, values in units.items() if unit[0] == tile
        ]
        if len(tile_units) != 120:
            raise ValueError(f"E4 tile {tile} requires exactly 120 paired units")
        if {unit[1] for unit, _ in tile_units} != {20, 100}:
            raise ValueError(f"E4 tile {tile} lacks both registered budgets")
        contrasts: dict[str, list[float]] = defaultdict(list)
        for unit, values in tile_units:
            if set(values) != set(methods):
                raise ValueError(
                    f"E4 paired unit {unit} differs: missing={sorted(set(methods)-set(values))}"
                )
            f00, f10, f01, f11, shifted = (values[name] for name in methods)
            contrasts["Delta_M"].append(0.5 * ((f00 - f10) + (f01 - f11)))
            contrasts["Delta_G"].append(0.5 * ((f00 - f01) + (f10 - f11)))
            contrasts["Delta_MG"].append((f01 - f11) - (f00 - f10))
            contrasts["D_M_given_G1"].append(f01 - f11)
            contrasts["D_placebo"].append(shifted - f11)
        tile_results.append(
            {
                "tile_id": tile,
                **{
                    name: float(np.mean(values))
                    for name, values in sorted(contrasts.items())
                },
            }
        )
    names = ("Delta_M", "Delta_G", "Delta_MG", "D_M_given_G1", "D_placebo")
    summaries: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    for index, name in enumerate(names):
        summary, p_value = _positive_effect_bootstrap(
            [float(tile[name]) for tile in tile_results],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
        )
        summaries[name] = {"bootstrap": summary, "one_sided_p": p_value}
        raw_p[name] = p_value
    adjusted = holm_adjust(raw_p)
    for name in names:
        summaries[name]["holm_adjusted_p"] = adjusted[name]
        summaries[name]["positive_95_lower"] = (
            summaries[name]["bootstrap"]["one_sided_low"] > 0.0
        )
    significant = {
        name: bool(
            summaries[name]["positive_95_lower"]
            and summaries[name]["holm_adjusted_p"] < 0.05
        )
        for name in names
    }
    trace_matched_pass = significant["Delta_M"] and significant["Delta_G"]
    real_spatial_pass = significant["D_M_given_G1"]
    placebo_pass = significant["D_placebo"]
    attribution_pass = bool(trace_matched_pass and real_spatial_pass and placebo_pass)
    return {
        "schema": "paper2.e4-mechanism-analysis/v1",
        "tile_count": len(tile_results),
        "tile_weighting": "equal",
        "paired_units_per_tile": 120,
        "error_metric": "mse_lower_is_better",
        "contrast_sign": "positive_means_registered_OT_component_reduces_error",
        "tests": summaries,
        "significant_after_holm": significant,
        "trace_matched_2x2_pass": trace_matched_pass,
        "real_spatial_field_pass": real_spatial_pass,
        "structural_placebo_pass": placebo_pass,
        "Delta_MG_is_characterization_not_a_required_gate": True,
        "ot_geometry_attribution_pass": attribution_pass,
        "claim_policy": (
            "OT geometry attribution is confirmatory"
            if attribution_pass
            else "retain all results and downgrade OT geometry attribution"
        ),
        "tile_results": tile_results,
    }
