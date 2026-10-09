from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import t

from .hashing import canonical_sha256


@dataclass(frozen=True)
class PowerEstimate:
    group_count: int
    joint_power: float
    marginal_power_a: float
    marginal_power_v: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "group_count": self.group_count,
            "joint_power": self.joint_power,
            "marginal_power_A": self.marginal_power_a,
            "marginal_power_V": self.marginal_power_v,
        }


def _paired_effects(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, ...], np.ndarray]:
    if len(rows) < 4:
        raise ValueError("power planning requires at least four validation groups")
    tile_ids: list[str] = []
    effects: list[tuple[float, float]] = []
    for row in rows:
        tile_id = str(row.get("tile_id", ""))
        if not tile_id or tile_id in tile_ids:
            raise ValueError("power rows require unique nonempty tile IDs")
        values = (float(row.get("delta_A", np.nan)), float(row.get("delta_V", np.nan)))
        if not np.all(np.isfinite(values)):
            raise ValueError(f"power effects must be finite for tile {tile_id}")
        tile_ids.append(tile_id)
        effects.append(values)
    return tuple(tile_ids), np.asarray(effects, dtype=np.float64)


def estimate_joint_power(
    effects: np.ndarray,
    *,
    group_count: int,
    simulations: int,
    seed: int,
    alpha: float = 0.05,
    chunk_size: int = 2_000,
) -> PowerEstimate:
    """Estimate the preregistered A-and-V intersection power.

    Both columns must be oriented so that positive values favour the frozen
    shared-K0 incremental anchor: ``A_iso - A_anchor`` and
    ``V_anchor - V_iso``. The planning distribution is a
    bivariate normal fitted only to bridge-validation geographic-group effects.
    Every simulated study applies two one-sided paired t tests and succeeds only
    when both endpoints reject, preserving their fitted correlation.
    """
    values = np.asarray(effects, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 4:
        raise ValueError("effects must have shape (validation_groups, 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("effects must be finite")
    if group_count not in {24, 30, 36}:
        raise ValueError("group_count must be 24, 30, or 36")
    if simulations < 1_000:
        raise ValueError("power estimation requires at least 1000 simulations")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie between zero and 0.5")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    mean = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("validation covariance is not finite")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # A zero variance endpoint makes a t statistic undefined. A tiny, declared
    # numerical floor only regularizes simulation draws; it cannot turn a
    # non-positive mean into evidence because the rejection direction remains.
    scale = max(float(np.trace(covariance)) / 2.0, 1.0)
    floor = np.finfo(np.float64).eps * scale
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T

    generator = np.random.default_rng(seed)
    critical = float(t.ppf(1.0 - alpha, df=group_count - 1))
    joint = marginal_a = marginal_v = completed = 0
    while completed < simulations:
        count = min(chunk_size, simulations - completed)
        samples = generator.multivariate_normal(
            mean,
            covariance,
            size=(count, group_count),
            check_valid="raise",
        )
        sample_mean = np.mean(samples, axis=1)
        sample_sd = np.std(samples, axis=1, ddof=1)
        statistics = sample_mean / (sample_sd / np.sqrt(group_count))
        passed = statistics > critical
        marginal_a += int(np.sum(passed[:, 0]))
        marginal_v += int(np.sum(passed[:, 1]))
        joint += int(np.sum(np.all(passed, axis=1)))
        completed += count
    return PowerEstimate(
        group_count=group_count,
        joint_power=joint / simulations,
        marginal_power_a=marginal_a / simulations,
        marginal_power_v=marginal_v / simulations,
    )


def build_blinded_power_report(
    payload: Mapping[str, Any],
    *,
    expected_validation_tiles: Sequence[str],
    reserve_group_order: Sequence[str],
    simulations: int = 100_000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if payload.get("schema") != "paper2.power-input/v1":
        raise ValueError("power input must use paper2.power-input/v1")
    if payload.get("source_role") != "bridge_val":
        raise ValueError("power input may only use bridge_val")
    if payload.get("test_results_inspected") is not False:
        raise ValueError("power input must certify that no test result was inspected")
    tile_ids, effects = _paired_effects(payload.get("rows", ()))
    if set(tile_ids) != set(str(value) for value in expected_validation_tiles):
        raise ValueError("power rows do not exactly match the frozen bridge_val tiles")
    reserve = tuple(str(value) for value in reserve_group_order)
    if len(reserve) != 12 or len(reserve) != len(set(reserve)):
        raise ValueError("the frozen sealed reserve must contain exactly 12 unique groups")

    seed_material = {
        "protocol": "paper2-bivariate-paired-t-power/v1",
        "payload": payload,
        "simulations": simulations,
        "alpha": alpha,
    }
    seed = int(canonical_sha256(seed_material)[:16], 16) % (2**63 - 1)
    estimates = tuple(
        estimate_joint_power(
            effects,
            group_count=count,
            simulations=simulations,
            seed=seed + count,
            alpha=alpha,
        )
        for count in (24, 30, 36)
    )
    selected = next((item for item in estimates if item.joint_power >= 0.80), estimates[-1])
    sufficient = selected.joint_power >= 0.80
    promoted = selected.group_count - 24
    return {
        "schema": "paper2.power-analysis/v1",
        "status": "complete" if sufficient else "insufficient_power",
        "test_results_inspected": False,
        "source_role": "bridge_val",
        "planning_model": "bivariate-normal-paired-geographic-group-effects",
        "decision_rule": "both one-sided paired t tests pass at alpha=0.05",
        "effect_orientation": {
            "delta_A": "A_Iso-SB-cond - A_OT-SB-incremental-anchor",
            "delta_V": "V_OT-SB-incremental-anchor - V_Iso-SB-cond",
        },
        "validation_group_count": len(tile_ids),
        "simulations": simulations,
        "alpha": alpha,
        "simulation_seed": seed,
        "input_semantic_sha256": canonical_sha256(payload),
        "test_group_count": selected.group_count,
        "power": selected.joint_power,
        "promoted_reserve_group_ids": list(reserve[:promoted]),
        "estimates": [item.to_dict() for item in estimates],
        "failure_policy": (
            "unlock selected test size"
            if sufficient
            else "keep all results sealed and downgrade the primary claim to exploratory"
        ),
    }
