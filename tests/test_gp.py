from __future__ import annotations

import numpy as np
from scipy import sparse

from paper2dem.gp import Matern32Kernel, footprint_gp_posterior


def test_footprint_gp_matches_dense_closed_form() -> None:
    x = np.linspace(0.0, 2.0, 7)[:, None]
    kernel = Matern32Kernel(variance=1.7, length_scale=0.8)
    dense_F = np.asarray(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75],
        ]
    )
    F = sparse.csr_matrix(dense_F)
    coarse = np.linspace(10.0, 13.0, 7)
    observations = np.asarray([10.8, 12.0, 12.3])
    noise = np.asarray([0.2, 0.4, 0.3])
    nugget = 0.05

    posterior = footprint_gp_posterior(
        coarse,
        x,
        F,
        observations,
        noise_covariance=noise,
        kernel=kernel,
        nugget_variance=nugget,
        block_size=3,
    )
    K = kernel(x, x)
    K_vf = K @ dense_F.T
    K_ff = dense_F @ K @ dense_F.T
    A = K_ff + np.diag(noise + nugget)
    residual = observations - dense_F @ coarse
    expected_mean = K_vf @ np.linalg.solve(A, residual)
    expected_covariance = K - K_vf @ np.linalg.solve(A, K_vf.T)

    np.testing.assert_allclose(posterior.mean_residual, expected_mean, atol=2e-12)
    np.testing.assert_allclose(posterior.variance, np.diag(expected_covariance), atol=2e-12)
    np.testing.assert_allclose(posterior.corrected_dem, coarse + expected_mean, atol=2e-12)
    assert np.all((posterior.relative_uncertainty >= 0.0) & (posterior.relative_uncertainty <= 1.0))


def test_k_zero_returns_prior_and_coarse_dem() -> None:
    features = np.arange(6.0)[:, None]
    coarse = np.arange(6.0).reshape(2, 3)
    kernel = Matern32Kernel(variance=2.5, length_scale=2.0)
    posterior = footprint_gp_posterior(
        coarse,
        features,
        sparse.csr_matrix((0, 6)),
        np.empty(0),
        noise_covariance=np.empty(0),
        kernel=kernel,
    )
    np.testing.assert_array_equal(posterior.corrected_dem, coarse)
    np.testing.assert_array_equal(posterior.mean_residual, np.zeros_like(coarse))
    np.testing.assert_array_equal(posterior.variance, np.full_like(coarse, 2.5))
    np.testing.assert_array_equal(posterior.relative_uncertainty, np.ones_like(coarse))


def test_gp_kernel_is_never_called_for_full_target_covariance() -> None:
    n = 40
    features = np.column_stack((np.linspace(0.0, 1.0, n), np.linspace(1.0, 2.0, n)))
    F = sparse.csr_matrix(
        (
            np.full(8, 0.25),
            ([0, 0, 0, 0, 1, 1, 1, 1], [3, 4, 5, 6, 31, 32, 33, 34]),
        ),
        shape=(2, n),
    )

    class GuardedKernel(Matern32Kernel):
        def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
            if left.shape[0] == n and right.shape[0] == n:
                raise AssertionError("attempted to construct the n x n target covariance")
            return super().__call__(left, right)

    posterior = footprint_gp_posterior(
        np.zeros(n),
        features,
        F,
        [1.0, -1.0],
        noise_covariance=[0.1, 0.1],
        kernel=GuardedKernel(variance=1.0, length_scale=[0.3, 0.4]),
        block_size=1000,
    )
    assert posterior.cross_covariance.shape == (n, 2)
    assert posterior.variance.shape == (n,)


def test_real_footprint_differs_from_center_point_assumption() -> None:
    features = np.arange(5.0)[:, None]
    coarse = np.zeros(5)
    footprint = sparse.csr_matrix([[0.0, 0.25, 0.5, 0.25, 0.0]])
    point = sparse.csr_matrix([[0.0, 0.0, 1.0, 0.0, 0.0]])
    kernel = Matern32Kernel(variance=1.0, length_scale=0.7)
    physical = footprint_gp_posterior(
        coarse,
        features,
        footprint,
        [1.0],
        noise_covariance=[0.01],
        kernel=kernel,
        block_size=2,
    )
    wrong_point = footprint_gp_posterior(
        coarse,
        features,
        point,
        [1.0],
        noise_covariance=[0.01],
        kernel=kernel,
        block_size=2,
    )
    assert not np.allclose(physical.mean_residual, wrong_point.mean_residual)


def test_streaming_gp_matches_retained_cross_covariance() -> None:
    rows, columns = 5, 6
    row_grid, column_grid = np.indices((rows, columns))
    features = np.column_stack((row_grid.ravel(), column_grid.ravel())).astype(float)
    coarse = np.linspace(0.0, 2.0, rows * columns).reshape(rows, columns)
    footprint = sparse.csr_matrix(
        (
            [0.5, 0.5, 0.25, 0.5, 0.25],
            ([0, 0, 1, 1, 1], [3, 4, 20, 21, 22]),
        ),
        shape=(2, rows * columns),
    )
    arguments = dict(
        coarse_dem=coarse,
        grid_features=features,
        footprint_operator=footprint,
        observations=np.asarray([0.7, 1.4]),
        noise_covariance=np.asarray([0.1, 0.2]),
        kernel=Matern32Kernel(variance=1.3, length_scale=[2.0, 3.0]),
        block_size=7,
    )
    retained = footprint_gp_posterior(**arguments)
    streamed = footprint_gp_posterior(**arguments, retain_cross_covariance=False)
    np.testing.assert_allclose(streamed.corrected_dem, retained.corrected_dem, atol=1e-12)
    np.testing.assert_allclose(streamed.variance, retained.variance, atol=1e-12)
    np.testing.assert_allclose(
        streamed.relative_uncertainty,
        retained.relative_uncertainty,
        atol=1e-12,
    )
    assert streamed.cross_covariance.shape == (0, 2)
