"""Joint terminal data consistency for sparse laser footprints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DataConsistencyResult:
    corrected: FloatArray
    correction: FloatArray
    residual_before: FloatArray
    residual_after: FloatArray
    mode: str


def _observation_covariance(value: ArrayLike, size: int) -> FloatArray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.ndim == 0:
        if not np.isfinite(covariance) or float(covariance) <= 0.0:
            raise ValueError("noise variance must be strictly positive")
        result = np.zeros((size, size), dtype=np.float64)
        if size:
            result.flat[:: size + 1] = float(covariance)
        return result
    if covariance.ndim == 1:
        if covariance.shape != (size,):
            raise ValueError("noise variance vector has the wrong length")
        if not np.all(np.isfinite(covariance)) or np.any(covariance <= 0.0):
            raise ValueError("noise variances must be finite and strictly positive")
        result = np.zeros((size, size), dtype=np.float64)
        if size:
            result.flat[:: size + 1] = covariance
        return result
    if covariance.shape != (size, size):
        raise ValueError("noise covariance has the wrong shape")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("noise covariance must be finite")
    result = 0.5 * (covariance + covariance.T)
    if size and np.min(np.linalg.eigvalsh(result)) <= 0.0:
        raise ValueError("noise covariance must be positive definite")
    return result


def joint_terminal_data_consistency(
    z_generated: ArrayLike,
    footprint_operator: sparse.spmatrix,
    observations: ArrayLike,
    *,
    noise_covariance: ArrayLike,
    prior_variance: ArrayLike,
    lambda_dc: float,
) -> DataConsistencyResult:
    """Solve the joint terminal proximal update through a ``K x K`` system.

    ``prior_variance`` is the diagonal of ``P_dc``.  No ``n x n`` matrix is
    constructed; the multiplication ``F diag(P) F.T`` remains sparse until
    the final observation-space matrix is formed.
    """

    generated = np.asarray(z_generated, dtype=np.float64)
    original_shape = generated.shape
    generated_flat = generated.reshape(-1)
    if not np.all(np.isfinite(generated_flat)):
        raise ValueError("z_generated must contain only finite values")
    operator = sparse.csr_matrix(footprint_operator, dtype=np.float64)
    if operator.shape[1] != generated_flat.size:
        raise ValueError("footprint operator column count does not match z_generated")
    returns = np.asarray(observations, dtype=np.float64).reshape(-1)
    if returns.shape != (operator.shape[0],):
        raise ValueError("observations length must match footprint rows")
    if not np.all(np.isfinite(returns)):
        raise ValueError("observations must contain only finite values")
    if not np.isfinite(lambda_dc) or lambda_dc <= 0.0:
        raise ValueError("lambda_dc must be strictly positive")

    diagonal = np.asarray(prior_variance, dtype=np.float64)
    try:
        diagonal = np.broadcast_to(diagonal, (generated_flat.size,)).astype(np.float64, copy=False)
    except ValueError as exc:
        raise ValueError("prior_variance must be scalar or have one value per pixel") from exc
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError("prior_variance must be finite and strictly positive")

    residual_before = returns - np.asarray(operator @ generated_flat).reshape(-1)
    if operator.shape[0] == 0:
        zeros = np.zeros_like(generated_flat)
        return DataConsistencyResult(
            corrected=generated.copy(),
            correction=zeros.reshape(original_shape),
            residual_before=residual_before,
            residual_after=residual_before.copy(),
            mode="D1",
        )

    covariance = _observation_covariance(noise_covariance, operator.shape[0])
    weighted_operator = operator.multiply(diagonal[None, :]).tocsr()
    system = np.asarray((weighted_operator @ operator.T).toarray(), dtype=np.float64)
    system += covariance / float(lambda_dc)
    system = 0.5 * (system + system.T)
    factor = cho_factor(system, lower=True, check_finite=False)
    coefficients = cho_solve(factor, residual_before, check_finite=False)
    correction_flat = diagonal * np.asarray(operator.T @ coefficients).reshape(-1)
    corrected_flat = generated_flat + correction_flat
    residual_after = returns - np.asarray(operator @ corrected_flat).reshape(-1)
    return DataConsistencyResult(
        corrected=corrected_flat.reshape(original_shape),
        correction=correction_flat.reshape(original_shape),
        residual_before=residual_before,
        residual_after=residual_after,
        mode="D1",
    )


def terminal_data_consistency(
    z_generated: ArrayLike,
    footprint_operator: sparse.spmatrix,
    observations: ArrayLike,
    *,
    noise_covariance: ArrayLike,
    mobility: ArrayLike,
    tau_dc: float,
    delta: float,
    lambda_dc: float,
    mode: str = "D1",
    point_operator: sparse.spmatrix | None = None,
) -> DataConsistencyResult:
    """Apply the pre-registered D0, D1, or D3 terminal-DC mode.

    D0 is a no-op, D1 uses the supplied physical footprint, and D3 uses the
    explicitly supplied point-observation operator while keeping the same
    returns and noise realization.
    """

    normalized_mode = mode.upper()
    if normalized_mode not in {"D0", "D1", "D3"}:
        raise ValueError("mode must be one of D0, D1, or D3")
    generated = np.asarray(z_generated, dtype=np.float64)
    physical_operator = sparse.csr_matrix(footprint_operator, dtype=np.float64)
    returns = np.asarray(observations, dtype=np.float64).reshape(-1)
    if physical_operator.shape[1] != generated.size:
        raise ValueError("footprint operator column count does not match z_generated")
    if returns.shape != (physical_operator.shape[0],):
        raise ValueError("observations length must match footprint rows")
    if normalized_mode == "D0" or physical_operator.shape[0] == 0:
        residual = returns - np.asarray(physical_operator @ generated.reshape(-1)).reshape(-1)
        return DataConsistencyResult(
            corrected=generated.copy(),
            correction=np.zeros_like(generated),
            residual_before=residual,
            residual_after=residual.copy(),
            mode=normalized_mode,
        )

    if not np.isfinite(tau_dc) or tau_dc <= 0.0:
        raise ValueError("tau_dc must be strictly positive")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be strictly positive")
    mobility_diagonal = np.asarray(mobility, dtype=np.float64)
    try:
        mobility_diagonal = np.broadcast_to(mobility_diagonal, (generated.size,))
    except ValueError as exc:
        raise ValueError("mobility must be scalar or have one value per pixel") from exc
    if not np.all(np.isfinite(mobility_diagonal)) or np.any(mobility_diagonal <= 0.0):
        raise ValueError("mobility must be finite and strictly positive")
    prior_variance = float(tau_dc) ** 2 * mobility_diagonal + float(delta)

    model_operator = physical_operator
    if normalized_mode == "D3":
        if point_operator is None:
            raise ValueError("D3 requires an explicit point_operator")
        model_operator = sparse.csr_matrix(point_operator, dtype=np.float64)
        if model_operator.shape != physical_operator.shape:
            raise ValueError("point_operator must have the same shape as footprint_operator")

    result = joint_terminal_data_consistency(
        generated,
        model_operator,
        returns,
        noise_covariance=noise_covariance,
        prior_variance=prior_variance,
        lambda_dc=lambda_dc,
    )
    return DataConsistencyResult(
        corrected=result.corrected,
        correction=result.correction,
        residual_before=returns - np.asarray(physical_operator @ generated.reshape(-1)).reshape(-1),
        residual_after=returns
        - np.asarray(physical_operator @ result.corrected.reshape(-1)).reshape(-1),
        mode=normalized_mode,
    )
