"""Footprint-aware Gaussian-process fusion without target covariance matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


FloatArray = NDArray[np.float64]


class Kernel(Protocol):
    def __call__(self, left: FloatArray, right: FloatArray) -> FloatArray: ...

    def diagonal(self, features: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class Matern32Kernel:
    """Positive-definite ARD Matérn-3/2 kernel used by the frozen GP prior."""

    variance: float = 1.0
    length_scale: float | ArrayLike = 1.0

    def _length_scales(self, feature_count: int) -> FloatArray:
        scales = np.asarray(self.length_scale, dtype=np.float64)
        try:
            scales = np.broadcast_to(scales, (feature_count,)).astype(np.float64, copy=False)
        except ValueError as exc:
            raise ValueError("length_scale must be scalar or match the feature count") from exc
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("length scales must be finite and strictly positive")
        if not np.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError("kernel variance must be finite and strictly positive")
        return scales

    def __call__(self, left: FloatArray, right: FloatArray) -> FloatArray:
        x = np.asarray(left, dtype=np.float64)
        y = np.asarray(right, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
            raise ValueError("kernel inputs must be 2-D with matching feature counts")
        scales = self._length_scales(x.shape[1])
        difference = x[:, None, :] / scales - y[None, :, :] / scales
        distance = np.sqrt(np.sum(difference * difference, axis=2, dtype=np.float64))
        scaled = np.sqrt(3.0) * distance
        return np.asarray(self.variance * (1.0 + scaled) * np.exp(-scaled), dtype=np.float64)

    def diagonal(self, features: FloatArray) -> FloatArray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("features must be 2-D")
        self._length_scales(values.shape[1])
        return np.full(values.shape[0], float(self.variance), dtype=np.float64)


@dataclass(frozen=True)
class GPPosterior:
    mean_residual: FloatArray
    variance: FloatArray
    corrected_dem: FloatArray
    relative_uncertainty: FloatArray
    information_gain: FloatArray
    cross_covariance: FloatArray
    observation_covariance: FloatArray
    cholesky_jitter: float


def _validate_features(value: ArrayLike, name: str) -> FloatArray:
    features = np.asarray(value, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, D) with positive dimensions")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{name} must contain only finite values")
    return features


def _noise_covariance(value: ArrayLike, count: int) -> FloatArray:
    noise = np.asarray(value, dtype=np.float64)
    result = np.zeros((count, count), dtype=np.float64)
    if noise.ndim == 0:
        if not np.isfinite(noise) or float(noise) <= 0.0:
            raise ValueError("noise variance must be strictly positive")
        if count:
            result.flat[:: count + 1] = float(noise)
    elif noise.ndim == 1:
        if noise.shape != (count,):
            raise ValueError("noise variance vector has the wrong length")
        if not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
            raise ValueError("noise variances must be finite and strictly positive")
        if count:
            result.flat[:: count + 1] = noise
    elif noise.shape == (count, count):
        if not np.all(np.isfinite(noise)):
            raise ValueError("noise covariance must be finite")
        result = 0.5 * (noise + noise.T)
        if count and np.min(np.linalg.eigvalsh(result)) <= 0.0:
            raise ValueError("noise covariance must be positive definite")
    else:
        raise ValueError("noise_covariance has the wrong shape")
    return result


def cross_covariance_to_footprints(
    query_features: ArrayLike,
    grid_features: ArrayLike,
    footprint_operator: sparse.spmatrix,
    kernel: Kernel,
    *,
    block_size: int = 4096,
) -> FloatArray:
    """Compute ``Cov(r(query), F r(grid))`` using footprint supports only.

    Kernel calls have shape ``block x footprint_support``.  In particular,
    this routine never asks the kernel to construct the full target-grid
    covariance matrix.
    """

    query = _validate_features(query_features, "query_features")
    grid = _validate_features(grid_features, "grid_features")
    if query.shape[1] != grid.shape[1]:
        raise ValueError("query and grid features must have the same feature count")
    operator = sparse.csr_matrix(footprint_operator, dtype=np.float64)
    if operator.shape[1] != grid.shape[0]:
        raise ValueError("footprint operator column count must match grid features")
    if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    count_query = query.shape[0]
    # For the usual query==grid path, this strict cap makes an accidental n x n
    # kernel allocation impossible even if a footprint spans the whole grid.
    cap = count_query
    if grid.shape[0] > 1:
        cap = min(cap, grid.shape[0] - 1)
    effective_block = max(1, min(int(block_size), cap))
    result = np.empty((count_query, operator.shape[0]), dtype=np.float64)
    for start in range(0, count_query, effective_block):
        stop = min(start + effective_block, count_query)
        query_block = query[start:stop]
        block_result = np.zeros((stop - start, operator.shape[0]), dtype=np.float64)
        for footprint_index in range(operator.shape[0]):
            left = operator.indptr[footprint_index]
            right = operator.indptr[footprint_index + 1]
            support_indices = operator.indices[left:right]
            support_weights = operator.data[left:right]
            if support_indices.size == 0:
                raise ValueError(f"footprint row {footprint_index} has no support")
            covariance = np.asarray(kernel(query_block, grid[support_indices]), dtype=np.float64)
            expected_shape = (query_block.shape[0], support_indices.size)
            if covariance.shape != expected_shape:
                raise ValueError("kernel returned an unexpected covariance shape")
            block_result[:, footprint_index] = covariance @ support_weights
        result[start:stop] = block_result
    return result


def _factor_positive_definite(matrix: FloatArray) -> tuple[tuple[FloatArray, bool], float]:
    scale = max(1.0, float(np.max(np.abs(np.diag(matrix)))))
    jitter = 0.0
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    for attempt in range(9):
        try:
            factor = cho_factor(matrix + jitter * identity, lower=True, check_finite=False)
            return factor, jitter
        except np.linalg.LinAlgError:
            jitter = scale * (10.0 ** (-12 + attempt))
    raise np.linalg.LinAlgError("observation covariance is not numerically positive definite")


def footprint_gp_posterior(
    coarse_dem: ArrayLike,
    grid_features: ArrayLike,
    footprint_operator: sparse.spmatrix,
    observations: ArrayLike,
    *,
    noise_covariance: ArrayLike,
    kernel: Kernel,
    nugget_variance: float = 0.0,
    block_size: int = 4096,
    epsilon: float = 1e-10,
    retain_cross_covariance: bool = True,
) -> GPPosterior:
    """Fuse footprint returns into a coarse DEM and return diagonal uncertainty.

    With ``retain_cross_covariance=False``, the target-by-footprint matrix is
    recomputed in query blocks and is not retained. This is the formal full-tile
    path: memory scales with ``block_size * K`` rather than ``N * K``.
    """

    coarse = np.asarray(coarse_dem, dtype=np.float64)
    original_shape = coarse.shape
    coarse_flat = coarse.reshape(-1)
    if not np.all(np.isfinite(coarse_flat)):
        raise ValueError("coarse_dem must contain only finite values")
    features = _validate_features(grid_features, "grid_features")
    if features.shape[0] != coarse_flat.size:
        raise ValueError("grid feature count must match coarse_dem size")
    operator = sparse.csr_matrix(footprint_operator, dtype=np.float64)
    if operator.shape[1] != coarse_flat.size:
        raise ValueError("footprint operator column count must match coarse_dem size")
    returns = np.asarray(observations, dtype=np.float64).reshape(-1)
    if returns.shape != (operator.shape[0],):
        raise ValueError("observations length must match footprint rows")
    if not np.all(np.isfinite(returns)):
        raise ValueError("observations must be finite")
    if not np.isfinite(nugget_variance) or nugget_variance < 0.0:
        raise ValueError("nugget_variance must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive")

    prior_variance = np.asarray(kernel.diagonal(features), dtype=np.float64).reshape(-1)
    if prior_variance.shape != (features.shape[0],):
        raise ValueError("kernel diagonal returned the wrong shape")
    if not np.all(np.isfinite(prior_variance)) or np.any(prior_variance <= 0.0):
        raise ValueError("kernel diagonal must be finite and strictly positive")

    footprint_count = operator.shape[0]
    if footprint_count == 0:
        zeros = np.zeros_like(coarse_flat)
        ones = np.ones_like(coarse_flat)
        return GPPosterior(
            mean_residual=zeros.reshape(original_shape),
            variance=prior_variance.reshape(original_shape),
            corrected_dem=coarse.copy(),
            relative_uncertainty=ones.reshape(original_shape),
            information_gain=zeros.reshape(original_shape),
            cross_covariance=np.empty((coarse_flat.size, 0), dtype=np.float64),
            observation_covariance=np.empty((0, 0), dtype=np.float64),
            cholesky_jitter=0.0,
        )

    if retain_cross_covariance:
        cross = cross_covariance_to_footprints(
            features,
            features,
            operator,
            kernel,
            block_size=block_size,
        )
        footprint_prior = np.asarray(operator @ cross, dtype=np.float64)
    else:
        support_indices = np.unique(operator.indices)
        support_cross = cross_covariance_to_footprints(
            features[support_indices],
            features,
            operator,
            kernel,
            block_size=block_size,
        )
        footprint_prior = np.asarray(
            operator[:, support_indices] @ support_cross,
            dtype=np.float64,
        )
        cross = np.empty((0, footprint_count), dtype=np.float64)
    footprint_prior = 0.5 * (footprint_prior + footprint_prior.T)
    effective_noise = _noise_covariance(noise_covariance, footprint_count)
    if nugget_variance:
        effective_noise.flat[:: footprint_count + 1] += float(nugget_variance)
    system = footprint_prior + effective_noise
    factor, jitter = _factor_positive_definite(system)

    residual_observations = returns - np.asarray(operator @ coarse_flat).reshape(-1)
    alpha = cho_solve(factor, residual_observations, check_finite=False)
    posterior_mean = np.empty_like(coarse_flat)
    posterior_variance = prior_variance.copy()
    for start in range(0, coarse_flat.size, int(block_size)):
        stop = min(start + int(block_size), coarse_flat.size)
        cross_block = (
            cross[start:stop]
            if retain_cross_covariance
            else cross_covariance_to_footprints(
                features[start:stop],
                features,
                operator,
                kernel,
                block_size=block_size,
            )
        )
        posterior_mean[start:stop] = cross_block @ alpha
        solved = cho_solve(factor, cross_block.T, check_finite=False).T
        posterior_variance[start:stop] -= np.sum(
            cross_block * solved,
            axis=1,
            dtype=np.float64,
        )
    numerical_tolerance = 1e-10 * np.maximum(1.0, prior_variance)
    if np.any(posterior_variance < -numerical_tolerance):
        raise RuntimeError("GP posterior variance is materially negative")
    posterior_variance = np.clip(posterior_variance, 0.0, prior_variance)
    relative = np.clip(
        (posterior_variance + float(epsilon)) / (prior_variance + float(epsilon)),
        0.0,
        1.0,
    )
    information_gain = 0.5 * np.log(
        (prior_variance + float(epsilon)) / (posterior_variance + float(epsilon))
    )
    corrected = coarse_flat + posterior_mean
    return GPPosterior(
        mean_residual=posterior_mean.reshape(original_shape),
        variance=posterior_variance.reshape(original_shape),
        corrected_dem=corrected.reshape(original_shape),
        relative_uncertainty=relative.reshape(original_shape),
        information_gain=information_gain.reshape(original_shape),
        cross_covariance=cross,
        observation_covariance=system,
        cholesky_jitter=jitter,
    )
