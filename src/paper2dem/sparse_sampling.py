from __future__ import annotations

from collections.abc import Sequence
import hashlib

import numpy as np


def _candidate_ids(values: Sequence[str], count: int) -> np.ndarray:
    result = np.asarray(values).astype(str)
    if result.shape != (count,):
        raise ValueError("candidate IDs must align one-to-one with candidates")
    if len(set(result.tolist())) != count:
        raise ValueError("candidate IDs must be unique")
    return result


def _maximum_budget(count: int, maximum_budget: int) -> int:
    if maximum_budget < 1 or maximum_budget > count:
        raise ValueError("maximum budget must lie between one and candidate count")
    return int(maximum_budget)


def deterministic_random_order(
    candidate_ids: Sequence[str], *, stream: str, maximum_budget: int
) -> np.ndarray:
    """Stable random-like order independent of NumPy implementation versions."""

    identifiers = _candidate_ids(candidate_ids, len(candidate_ids))
    maximum = _maximum_budget(len(identifiers), maximum_budget)
    ranked = sorted(
        range(len(identifiers)),
        key=lambda index: (
            hashlib.sha256(f"{stream}|{identifiers[index]}".encode()).hexdigest(),
            identifiers[index],
        ),
    )
    return np.asarray(ranked[:maximum], dtype=np.int64)


def uniform_farthest_order(
    coordinates: np.ndarray,
    candidate_ids: Sequence[str],
    *,
    maximum_budget: int,
) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("candidate coordinates must have finite shape (n, 2)")
    identifiers = _candidate_ids(candidate_ids, len(points))
    maximum = _maximum_budget(len(points), maximum_budget)
    centroid = np.mean(points, axis=0)
    distance_to_centroid = np.sum((points - centroid) ** 2, axis=1)
    farthest = float(np.max(distance_to_centroid))
    tied = np.flatnonzero(np.isclose(distance_to_centroid, farthest))
    first = min(tied.tolist(), key=lambda index: identifiers[index])
    selected = np.zeros(len(points), dtype=bool)
    selected[first] = True
    order = [first]
    minimum_distance = np.sum((points - points[first]) ** 2, axis=1)
    while len(order) < maximum:
        available = np.flatnonzero(~selected)
        best = float(np.max(minimum_distance[available]))
        tied = available[np.isclose(minimum_distance[available], best)]
        chosen = min(tied.tolist(), key=lambda index: identifiers[index])
        order.append(chosen)
        selected[chosen] = True
        minimum_distance = np.minimum(
            minimum_distance, np.sum((points - points[chosen]) ** 2, axis=1)
        )
    return np.asarray(order, dtype=np.int64)


def max_prior_variance_order(
    prior_variance: np.ndarray,
    candidate_ids: Sequence[str],
    *,
    maximum_budget: int,
) -> np.ndarray:
    variance = np.asarray(prior_variance, dtype=np.float64)
    if variance.ndim != 1 or not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError("prior variance must be a finite nonnegative vector")
    identifiers = _candidate_ids(candidate_ids, len(variance))
    maximum = _maximum_budget(len(variance), maximum_budget)
    order = sorted(
        range(len(variance)),
        key=lambda index: (-variance[index], identifiers[index]),
    )
    return np.asarray(order[:maximum], dtype=np.int64)


def _validate_covariance(covariance: np.ndarray) -> np.ndarray:
    value = np.asarray(covariance, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("candidate covariance must be square")
    if not np.all(np.isfinite(value)) or not np.allclose(value, value.T, atol=1.0e-9):
        raise ValueError("candidate covariance must be finite and symmetric")
    if np.min(np.linalg.eigvalsh(value)) < -1.0e-7:
        raise ValueError("candidate covariance must be positive semidefinite")
    return (value + value.T) * 0.5


def greedy_mutual_information_order(
    candidate_covariance: np.ndarray,
    candidate_ids: Sequence[str],
    *,
    noise_variance: float,
    maximum_budget: int,
) -> np.ndarray:
    """Greedy point-MI order using conditional candidate variance."""

    conditional = _validate_covariance(candidate_covariance).copy()
    identifiers = _candidate_ids(candidate_ids, len(conditional))
    maximum = _maximum_budget(len(conditional), maximum_budget)
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("noise variance must be finite and positive")
    selected = np.zeros(len(conditional), dtype=bool)
    order = []
    for _ in range(maximum):
        scores = np.diag(conditional).copy()
        scores[selected] = -np.inf
        best = float(np.max(scores))
        tied = np.flatnonzero(np.isclose(scores, best))
        chosen = min(tied.tolist(), key=lambda index: identifiers[index])
        order.append(chosen)
        selected[chosen] = True
        column = conditional[:, chosen].copy()
        denominator = max(float(conditional[chosen, chosen] + noise_variance), 1.0e-12)
        conditional -= np.outer(column, column) / denominator
        conditional = (conditional + conditional.T) * 0.5
    return np.asarray(order, dtype=np.int64)


def greedy_a_optimal_order(
    target_candidate_covariance: np.ndarray,
    candidate_covariance: np.ndarray,
    candidate_ids: Sequence[str],
    *,
    noise_variance: float,
    maximum_budget: int,
    target_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Footprint-ready A-optimal ordering by integrated variance reduction."""

    target_cross = np.asarray(target_candidate_covariance, dtype=np.float64).copy()
    conditional = _validate_covariance(candidate_covariance).copy()
    if target_cross.ndim != 2 or target_cross.shape[1] != len(conditional):
        raise ValueError("target-candidate covariance must align with candidates")
    if not np.all(np.isfinite(target_cross)):
        raise ValueError("target-candidate covariance must be finite")
    identifiers = _candidate_ids(candidate_ids, len(conditional))
    maximum = _maximum_budget(len(conditional), maximum_budget)
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("noise variance must be finite and positive")
    if target_weights is None:
        weights = np.full(target_cross.shape[0], 1.0 / target_cross.shape[0])
    else:
        weights = np.asarray(target_weights, dtype=np.float64)
        if (
            weights.shape != (target_cross.shape[0],)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or np.sum(weights) <= 0
        ):
            raise ValueError("target weights must be finite, nonnegative, and nonzero")
        weights = weights / np.sum(weights)
    selected = np.zeros(len(conditional), dtype=bool)
    order = []
    for _ in range(maximum):
        denominators = np.maximum(np.diag(conditional) + noise_variance, 1.0e-12)
        scores = np.sum(weights[:, None] * target_cross**2, axis=0) / denominators
        scores[selected] = -np.inf
        best = float(np.max(scores))
        tied = np.flatnonzero(np.isclose(scores, best))
        chosen = min(tied.tolist(), key=lambda index: identifiers[index])
        order.append(chosen)
        selected[chosen] = True
        candidate_column = conditional[:, chosen].copy()
        target_column = target_cross[:, chosen].copy()
        denominator = max(float(conditional[chosen, chosen] + noise_variance), 1.0e-12)
        target_cross -= np.outer(target_column, candidate_column) / denominator
        conditional -= np.outer(candidate_column, candidate_column) / denominator
        conditional = (conditional + conditional.T) * 0.5
    return np.asarray(order, dtype=np.int64)
