"""Analytic utilities for the uncertainty-preconditioned endpoint bridge.

The spatial preconditioner is represented exclusively by its diagonal.  This
is deliberate: a DEM patch may contain tens of thousands of pixels, so an
``n x n`` mobility or metric matrix must never be materialised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BridgeMoments:
    """Moments and scalar coefficients of an endpoint-conditioned bridge."""

    mean: FloatArray
    variance: FloatArray
    weight_x0: float
    weight_xT: float
    scalar_variance: float


def _finite_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_mobility(value: ArrayLike, shape: tuple[int, ...]) -> FloatArray:
    mobility = _finite_array(value, "mobility")
    try:
        mobility = np.broadcast_to(mobility, shape).astype(np.float64, copy=False)
    except ValueError as exc:
        raise ValueError("mobility is not broadcastable to the endpoint shape") from exc
    if np.any(mobility <= 0.0):
        raise ValueError("mobility must be strictly positive")
    return mobility


def mobility_from_uncertainty(
    omega: ArrayLike,
    *,
    g_min: float,
    alpha: float,
) -> FloatArray:
    """Map relative remaining uncertainty to the diagonal of ``G_omega``.

    ``omega`` is clipped to ``[0, 1]`` as specified by the method contract.
    The returned array is the diagonal itself, not a dense diagonal matrix.
    """

    uncertainty = _finite_array(omega, "omega")
    if not 0.0 < g_min <= 1.0:
        raise ValueError("g_min must lie in (0, 1]")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be strictly positive")
    clipped = np.clip(uncertainty, 0.0, 1.0)
    return g_min + (1.0 - g_min) * np.power(clipped, alpha)


def inverse_metric_multiply(vector: ArrayLike, mobility: ArrayLike) -> FloatArray:
    """Apply ``G_omega^{-1}`` using elementwise operations."""

    values = _finite_array(vector, "vector")
    diagonal = _positive_mobility(mobility, values.shape)
    return values / diagonal


def transport_metric_norm_squared(vector: ArrayLike, mobility: ArrayLike) -> float:
    """Return ``vector.T @ G_omega^{-1} @ vector`` without a matrix."""

    values = _finite_array(vector, "vector")
    return float(np.sum(values * inverse_metric_multiply(values, mobility), dtype=np.float64))


def endpoint_bridge_moments(
    x0: ArrayLike,
    xT: ArrayLike,
    *,
    B_t: float,
    B_bar_t: float,
    mobility: ArrayLike,
) -> BridgeMoments:
    """Return the analytic Gaussian moments conditioned on both endpoints.

    Parameters are the cumulative diffusion strengths
    ``B_t = integral_0^t beta_s ds`` and
    ``B_bar_t = integral_t^T beta_s ds``.  Endpoint values (one of the two
    strengths equal to zero) are valid and have zero conditional variance.
    """

    start = _finite_array(x0, "x0")
    end = _finite_array(xT, "xT")
    if start.shape != end.shape:
        raise ValueError("x0 and xT must have identical shapes")
    diagonal = _positive_mobility(mobility, start.shape)
    if not np.isfinite(B_t) or not np.isfinite(B_bar_t):
        raise ValueError("diffusion strengths must be finite")
    if B_t < 0.0 or B_bar_t < 0.0:
        raise ValueError("diffusion strengths must be non-negative")
    total = float(B_t + B_bar_t)
    if total <= 0.0:
        raise ValueError("B_t + B_bar_t must be strictly positive")

    weight_x0 = float(B_bar_t / total)
    weight_xT = float(B_t / total)
    scalar_variance = float(B_t * B_bar_t / total)
    mean = weight_x0 * start + weight_xT * end
    variance = scalar_variance * diagonal
    return BridgeMoments(
        mean=np.asarray(mean, dtype=np.float64),
        variance=np.asarray(variance, dtype=np.float64),
        weight_x0=weight_x0,
        weight_xT=weight_xT,
        scalar_variance=scalar_variance,
    )


def sample_endpoint_bridge(
    x0: ArrayLike,
    xT: ArrayLike,
    *,
    B_t: float,
    B_bar_t: float,
    mobility: ArrayLike,
    rng: np.random.Generator | None = None,
    epsilon: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Sample the analytic endpoint bridge and return ``(sample, epsilon)``."""

    moments = endpoint_bridge_moments(
        x0,
        xT,
        B_t=B_t,
        B_bar_t=B_bar_t,
        mobility=mobility,
    )
    if epsilon is None:
        generator = np.random.default_rng() if rng is None else rng
        noise = generator.standard_normal(moments.mean.shape, dtype=np.float64)
    else:
        noise = _finite_array(epsilon, "epsilon")
        if noise.shape != moments.mean.shape:
            raise ValueError("epsilon must have the same shape as the endpoints")
    sample = moments.mean + np.sqrt(moments.variance) * noise
    return np.asarray(sample, dtype=np.float64), np.asarray(noise, dtype=np.float64)


def whiten_bridge_noise(
    x_t: ArrayLike,
    x0: ArrayLike,
    xT: ArrayLike,
    *,
    B_t: float,
    B_bar_t: float,
    mobility: ArrayLike,
) -> FloatArray:
    """Recover the standard-normal bridge noise by diagonal whitening."""

    state = _finite_array(x_t, "x_t")
    moments = endpoint_bridge_moments(
        x0,
        xT,
        B_t=B_t,
        B_bar_t=B_bar_t,
        mobility=mobility,
    )
    if state.shape != moments.mean.shape:
        raise ValueError("x_t must have the same shape as the endpoints")
    if moments.scalar_variance <= 0.0:
        raise ValueError("bridge noise is undefined at a deterministic endpoint")
    return np.asarray((state - moments.mean) / np.sqrt(moments.variance), dtype=np.float64)


def gated_endpoint(
    xT: ArrayLike,
    residual: ArrayLike,
    mobility: ArrayLike,
    *,
    enabled: bool = True,
) -> FloatArray:
    """Apply the method's ``X_T + sqrt(G_omega) * residual`` parameterisation."""

    endpoint = _finite_array(xT, "xT")
    update = _finite_array(residual, "residual")
    if endpoint.shape != update.shape:
        raise ValueError("xT and residual must have identical shapes")
    if not enabled:
        return np.asarray(endpoint + update, dtype=np.float64)
    diagonal = _positive_mobility(mobility, endpoint.shape)
    return np.asarray(endpoint + np.sqrt(diagonal) * update, dtype=np.float64)
