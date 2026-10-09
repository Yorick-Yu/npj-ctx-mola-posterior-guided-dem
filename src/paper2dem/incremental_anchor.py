"""Deterministic shared-K0 policy for attributing laser-conditioned OT gains."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .hashing import canonical_sha256


ISO_METHOD = "Iso-SB-cond"
OT_METHOD = "OT-SB-full"
ANCHORED_METHOD = "OT-SB-incremental-anchor"


@dataclass(frozen=True)
class IncrementalAnchorPolicy:
    """Select Iso at K=0 and OT at every strictly positive laser budget.

    The policy has no fitted threshold or weight.  Keeping it as a hashed
    contract prevents validation/test code from selecting a source prediction
    using reference errors or per-tile outcomes.
    """

    schema: str = "paper2.incremental-anchor-policy/v1"
    method: str = ANCHORED_METHOD
    zero_budget_source: str = ISO_METHOD
    positive_budget_source: str = OT_METHOD
    threshold: int = 0
    threshold_fitted: bool = False
    weights_fitted: bool = False
    tile_specific: bool = False
    noise_replicate_specific: bool = False

    def validate(self) -> None:
        if not (
            self.schema == "paper2.incremental-anchor-policy/v1"
            and self.method == ANCHORED_METHOD
            and self.zero_budget_source == ISO_METHOD
            and self.positive_budget_source == OT_METHOD
            and self.threshold == 0
            and self.threshold_fitted is False
            and self.weights_fitted is False
            and self.tile_specific is False
            and self.noise_replicate_specific is False
        ):
            raise ValueError("incremental-anchor policy differs from the frozen contract")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "method": self.method,
            "zero_budget_source": self.zero_budget_source,
            "positive_budget_source": self.positive_budget_source,
            "threshold": self.threshold,
            "threshold_fitted": self.threshold_fitted,
            "weights_fitted": self.weights_fitted,
            "tile_specific": self.tile_specific,
            "noise_replicate_specific": self.noise_replicate_specific,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def source_method(self, budget: int) -> str:
        self.validate()
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("laser budget must be a nonnegative integer")
        return self.zero_budget_source if budget == 0 else self.positive_budget_source


def anchored_prediction(
    *,
    budget: int,
    iso_mean: np.ndarray,
    iso_variance: np.ndarray,
    ot_mean: np.ndarray,
    ot_variance: np.ndarray,
    policy: IncrementalAnchorPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the frozen source arrays without blending or reference access."""

    contract = policy or IncrementalAnchorPolicy()
    source = contract.source_method(budget)
    arrays = {
        "iso_mean": np.asarray(iso_mean),
        "iso_variance": np.asarray(iso_variance),
        "ot_mean": np.asarray(ot_mean),
        "ot_variance": np.asarray(ot_variance),
    }
    shape = arrays["iso_mean"].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays.values()):
        raise ValueError("incremental-anchor prediction arrays must be aligned 2-D grids")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("incremental-anchor prediction arrays must be finite")
    if np.any(arrays["iso_variance"] < 0) or np.any(arrays["ot_variance"] < 0):
        raise ValueError("incremental-anchor variances must be nonnegative")
    if source == ISO_METHOD:
        return arrays["iso_mean"], arrays["iso_variance"], source
    return arrays["ot_mean"], arrays["ot_variance"], source


def anchored_curve(
    *,
    budgets: Sequence[int],
    iso_curve: Sequence[float] | np.ndarray,
    ot_curve: Sequence[float] | np.ndarray,
    policy: IncrementalAnchorPolicy | None = None,
) -> np.ndarray:
    """Materialize the frozen shared-K0 curve from registered source curves."""

    contract = policy or IncrementalAnchorPolicy()
    budget_values = tuple(budgets)
    invalid_budget = any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in budget_values
    )
    not_increasing = any(
        right <= left
        for left, right in zip(budget_values, budget_values[1:], strict=False)
    )
    if not budget_values or 0 not in budget_values or invalid_budget or not_increasing:
        raise ValueError(
            "anchored curve budgets must be strictly increasing nonnegative "
            "integers including zero"
        )
    iso = np.asarray(iso_curve, dtype=np.float64)
    ot = np.asarray(ot_curve, dtype=np.float64)
    if (
        iso.ndim != 1
        or ot.ndim != 1
        or iso.shape != ot.shape
        or len(iso) != len(budget_values)
    ):
        raise ValueError("anchored source curves must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(iso)) or not np.all(np.isfinite(ot)):
        raise ValueError("anchored source curves must be finite")
    return np.asarray(
        [
            iso[index]
            if contract.source_method(budget) == ISO_METHOD
            else ot[index]
            for index, budget in enumerate(budget_values)
        ],
        dtype=np.float64,
    )
