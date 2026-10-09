from __future__ import annotations

import numpy as np
import pytest

from paper2dem.bridge import (
    endpoint_bridge_moments,
    gated_endpoint,
    mobility_from_uncertainty,
    sample_endpoint_bridge,
    transport_metric_norm_squared,
    whiten_bridge_noise,
)


def test_endpoint_moments_match_dense_gaussian_conditioning() -> None:
    x0 = np.asarray([2.0, -1.0, 4.0])
    xT = np.asarray([3.5, 2.0, -2.0])
    mobility = np.asarray([0.2, 0.7, 1.4])
    B_t = 0.35
    B_bar = 0.8

    moments = endpoint_bridge_moments(
        x0, xT, B_t=B_t, B_bar_t=B_bar, mobility=mobility
    )
    dense_G = np.diag(mobility)
    covariance_t = B_t * dense_G
    covariance_T = (B_t + B_bar) * dense_G
    dense_mean = x0 + covariance_t @ np.linalg.solve(covariance_T, xT - x0)
    dense_covariance = covariance_t - covariance_t @ np.linalg.solve(
        covariance_T, covariance_t
    )

    np.testing.assert_allclose(moments.mean, dense_mean, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(moments.variance, np.diag(dense_covariance), atol=1e-12)


def test_sampling_and_whitening_are_exact_inverses() -> None:
    x0 = np.linspace(-2.0, 2.0, 17)
    xT = x0[::-1]
    mobility = np.linspace(0.1, 1.0, 17)
    epsilon = np.random.default_rng(17).standard_normal(17)
    sample, returned_epsilon = sample_endpoint_bridge(
        x0,
        xT,
        B_t=0.4,
        B_bar_t=0.6,
        mobility=mobility,
        epsilon=epsilon,
    )
    recovered = whiten_bridge_noise(
        sample,
        x0,
        xT,
        B_t=0.4,
        B_bar_t=0.6,
        mobility=mobility,
    )
    np.testing.assert_allclose(returned_epsilon, epsilon, atol=0.0)
    np.testing.assert_allclose(recovered, epsilon, rtol=0.0, atol=1e-12)


def test_isotropic_degeneration_and_gate() -> None:
    omega = np.ones((3, 4))
    mobility = mobility_from_uncertainty(omega, g_min=0.03, alpha=2.0)
    np.testing.assert_array_equal(mobility, np.ones_like(omega))
    vector = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert transport_metric_norm_squared(vector, mobility) == pytest.approx(
        float(np.sum(vector**2))
    )
    endpoint = np.zeros_like(vector)
    np.testing.assert_allclose(gated_endpoint(endpoint, vector, mobility), vector)


def test_bridge_uses_diagonal_vector_not_dense_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_diag(*args: object, **kwargs: object) -> None:
        raise AssertionError("bridge implementation attempted to materialise a diagonal matrix")

    monkeypatch.setattr(np, "diag", forbidden_diag)
    mobility = mobility_from_uncertainty(np.linspace(0.0, 1.0, 20), g_min=0.05, alpha=1.5)
    moments = endpoint_bridge_moments(
        np.zeros(20), np.ones(20), B_t=0.2, B_bar_t=0.8, mobility=mobility
    )
    assert moments.mean.shape == (20,)


def test_whitening_rejects_deterministic_endpoint() -> None:
    with pytest.raises(ValueError, match="deterministic endpoint"):
        whiten_bridge_noise(
            np.ones(2),
            np.ones(2),
            np.zeros(2),
            B_t=0.0,
            B_bar_t=1.0,
            mobility=np.ones(2),
        )
