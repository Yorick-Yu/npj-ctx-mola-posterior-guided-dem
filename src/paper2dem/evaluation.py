from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import integrate, ndimage, stats


def _paired_values(
    estimate: np.ndarray, reference: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if estimate.shape != reference.shape or estimate.shape != valid.shape:
        raise ValueError("estimate, reference, and mask must have identical shapes")
    valid &= np.isfinite(estimate) & np.isfinite(reference)
    if not np.any(valid):
        raise ValueError("evaluation mask has no finite pixels")
    return estimate[valid], reference[valid]


def dem_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    *,
    pixel_size_m: float,
    curvature_scale_m: float = 100.0,
) -> dict[str, float]:
    predicted, truth = _paired_values(estimate, reference, valid_mask)
    error = predicted - truth
    absolute = np.abs(error)
    elevation_range = float(np.percentile(truth, 95) - np.percentile(truth, 5))
    median_error = float(np.median(error))
    result = {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "nmad": float(1.4826 * np.median(np.abs(error - median_error))),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "nrmse_p95_p5": float(np.sqrt(np.mean(error**2)) / elevation_range)
        if elevation_range > 0
        else float("nan"),
    }
    sigma_pixels = max(curvature_scale_m / max(pixel_size_m, 1.0e-12), 0.5)
    estimate_filled = np.where(valid_mask, estimate, np.nan)
    reference_filled = np.where(valid_mask, reference, np.nan)
    # Nearest fill is used only for stencil support; metrics remain on the original mask.
    if np.any(~np.isfinite(estimate_filled)):
        indices = ndimage.distance_transform_edt(
            ~np.isfinite(estimate_filled), return_distances=False, return_indices=True
        )
        estimate_filled = estimate_filled[tuple(indices)]
    if np.any(~np.isfinite(reference_filled)):
        indices = ndimage.distance_transform_edt(
            ~np.isfinite(reference_filled), return_distances=False, return_indices=True
        )
        reference_filled = reference_filled[tuple(indices)]
    pred_gradient = np.gradient(estimate_filled, pixel_size_m)
    true_gradient = np.gradient(reference_filled, pixel_size_m)
    pred_slope = np.degrees(np.arctan(np.hypot(*pred_gradient)))
    true_slope = np.degrees(np.arctan(np.hypot(*true_gradient)))
    slope_error = (pred_slope - true_slope)[valid_mask]
    result["slope_rmse_degrees"] = float(np.sqrt(np.mean(slope_error**2)))
    result["slope_mae_degrees"] = float(np.mean(np.abs(slope_error)))
    pred_smooth = ndimage.gaussian_filter(estimate_filled, sigma=sigma_pixels)
    true_smooth = ndimage.gaussian_filter(reference_filled, sigma=sigma_pixels)
    pred_curvature = ndimage.laplace(pred_smooth) / (pixel_size_m**2)
    true_curvature = ndimage.laplace(true_smooth) / (pixel_size_m**2)
    curvature_error = (pred_curvature - true_curvature)[valid_mask]
    result["curvature_rmse"] = float(np.sqrt(np.mean(curvature_error**2)))
    return result


def curve_areas(
    budgets: Sequence[int],
    mse: Sequence[float],
    *,
    coarse_mse: float,
) -> dict[str, float]:
    k = np.asarray(budgets, dtype=np.float64)
    errors = np.asarray(mse, dtype=np.float64)
    if k.ndim != 1 or errors.shape != k.shape or len(k) < 2:
        raise ValueError("budgets and MSE must be equal one-dimensional curves")
    if k[0] != 0 or np.any(np.diff(k) <= 0) or coarse_mse <= 0:
        raise ValueError("curve requires K=0, strictly increasing budgets, and positive coarse MSE")
    if not np.all(np.isfinite(errors)):
        raise ValueError("missing/non-finite budget cells cannot be interpolated")
    normalized_budget = k / k[-1]
    absolute_area = float(integrate.trapezoid(errors / coarse_mse, normalized_budget))
    realized = (errors[0] - errors) / coarse_mse
    value_area = float(integrate.trapezoid(realized, normalized_budget))
    return {"A": absolute_area, "V": value_area}


def nonincreasing_isotonic(values: Sequence[float]) -> np.ndarray:
    """Pool-adjacent-violators fit with equal weights and a non-increasing constraint."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or not len(source) or not np.all(np.isfinite(source)):
        raise ValueError("isotonic input must be a finite vector")
    means: list[float] = []
    counts: list[int] = []
    for value in source:
        means.append(float(value))
        counts.append(1)
        while len(means) >= 2 and means[-2] < means[-1]:
            total = counts[-2] + counts[-1]
            pooled = (means[-2] * counts[-2] + means[-1] * counts[-1]) / total
            means[-2:] = [pooled]
            counts[-2:] = [total]
    return np.concatenate([np.full(count, mean) for mean, count in zip(means, counts)])


def required_budget(
    budgets: Sequence[int], normalized_error: Sequence[float], target: float
) -> float | None:
    k = np.asarray(budgets, dtype=np.float64)
    fitted = nonincreasing_isotonic(normalized_error)
    if k.shape != fitted.shape or k[0] != 0 or np.any(np.diff(k) <= 0):
        raise ValueError("budgets must start at zero and increase strictly")
    reached = np.flatnonzero(fitted <= target)
    if not len(reached):
        return None
    index = int(reached[0])
    if index == 0:
        return 0.0
    left_k, right_k = k[index - 1], k[index]
    left_error, right_error = fitted[index - 1], fitted[index]
    if left_error == right_error:
        return float(right_k)
    fraction = (left_error - target) / (left_error - right_error)
    return float(left_k + np.clip(fraction, 0.0, 1.0) * (right_k - left_k))


def laser_saving(
    budgets: Sequence[int],
    full_error: Sequence[float],
    baseline_error: Sequence[float],
    target: float,
) -> dict[str, float | bool | None]:
    full_k = required_budget(budgets, full_error, target)
    baseline_k = required_budget(budgets, baseline_error, target)
    if full_k is None or baseline_k is None:
        return {
            "target_reached": False,
            "saving_estimable": False,
            "full_budget": full_k,
            "baseline_budget": baseline_k,
            "delta_k": None,
            "saving_fraction": None,
            "undefined_reason": "target_not_reached_without_extrapolation",
        }
    delta = baseline_k - full_k
    if baseline_k == 0:
        return {
            "target_reached": True,
            "saving_estimable": False,
            "full_budget": full_k,
            "baseline_budget": baseline_k,
            "delta_k": float(delta),
            "saving_fraction": None,
            "undefined_reason": "baseline_reaches_target_at_zero_budget",
        }
    saving = delta / baseline_k
    return {
        "target_reached": True,
        "saving_estimable": True,
        "full_budget": full_k,
        "baseline_budget": baseline_k,
        "delta_k": float(delta),
        "saving_fraction": float(saving),
        "undefined_reason": None,
    }


def gaussian_probabilistic_metrics(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    predicted, reference = _paired_values(mean, truth, valid_mask)
    sigma = np.asarray(standard_deviation, dtype=np.float64)[np.asarray(valid_mask, dtype=bool)]
    if sigma.shape != predicted.shape or np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("standard deviations must be finite and positive on valid pixels")
    z = (reference - predicted) / sigma
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * z**2
    crps = sigma * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    result = {
        "gaussian_nll": float(np.mean(nll)),
        "gaussian_crps": float(np.mean(crps)),
        "sharpness_mean_std": float(np.mean(sigma)),
    }
    for level in (0.50, 0.80, 0.90, 0.95):
        critical = stats.norm.ppf(0.5 + level / 2)
        result[f"coverage_{int(level * 100)}"] = float(np.mean(np.abs(z) <= critical))
    return result


def footprint_consistency_metrics(
    estimate: np.ndarray,
    operator: Any,
    observations: np.ndarray,
    *,
    noise_variance: float | np.ndarray,
    pixel_variance: np.ndarray | None = None,
    footprint_variance_scale: float | None = None,
    footprint_variance_floor: float = 0.0,
) -> dict[str, float]:
    """Report data discrepancy on footprints already consumed by GP/DC.

    These observations are not held out.  Consequently this function never turns
    pixel marginal variance into a footprint predictive variance.  Formal
    footprint uncertainty is evaluated only from per-complete-path projections.
    """

    surface = np.asarray(estimate, dtype=np.float64)
    observed = np.asarray(observations, dtype=np.float64)
    predicted = np.asarray(operator @ surface.ravel(), dtype=np.float64)
    if predicted.shape != observed.shape or not np.all(np.isfinite(observed)):
        raise ValueError("footprint predictions and observations must be aligned and finite")
    residual = predicted - observed
    result = {
        "footprint_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "footprint_mae_m": float(np.mean(np.abs(residual))),
        "footprint_bias_m": float(np.mean(residual)),
    }
    noise = np.broadcast_to(np.asarray(noise_variance, dtype=np.float64), observed.shape)
    if np.any(noise <= 0) or not np.all(np.isfinite(noise)):
        raise ValueError("instrument noise variance must be finite and positive")
    if pixel_variance is not None:
        raise ValueError(
            "diagonal pixel variance cannot represent spatial footprint covariance"
        )
    if footprint_variance_scale is not None:
        raise ValueError(
            "used-footprint discrepancy cannot consume the held-out footprint calibrator"
        )
    standardized = residual / np.sqrt(noise)
    result.update(
        {
            "noise_scaled_discrepancy_rms": float(np.sqrt(np.mean(standardized**2))),
            "footprint_max_absolute_discrepancy_m": float(np.max(np.abs(residual))),
        }
    )
    return result


def audit_footprint_metrics(
    *,
    predicted_mean: np.ndarray,
    path_sample_variance: np.ndarray,
    clean_reference: np.ndarray,
    heldout_returns: np.ndarray,
    instrument_noise_variance: float | np.ndarray,
    calibration_scale: float | None,
    calibration_floor_variance: float = 0.0,
) -> dict[str, Any]:
    """Score clean and noisy held-out footprints without reusing reconstruction data."""

    mean = np.asarray(predicted_mean, dtype=np.float64)
    raw_variance = np.asarray(path_sample_variance, dtype=np.float64)
    clean = np.asarray(clean_reference, dtype=np.float64)
    heldout = np.asarray(heldout_returns, dtype=np.float64)
    if not (mean.shape == raw_variance.shape == clean.shape == heldout.shape):
        raise ValueError("audit footprint arrays must have identical shapes")
    if mean.ndim != 1 or not len(mean) or not np.all(
        np.isfinite(np.stack((mean, raw_variance, clean, heldout)))
    ):
        raise ValueError("audit footprint arrays must be nonempty finite vectors")
    if np.any(raw_variance < 0):
        raise ValueError("audit path variance must be non-negative")
    noise = np.broadcast_to(np.asarray(instrument_noise_variance, dtype=np.float64), mean.shape)
    if np.any(noise <= 0) or not np.all(np.isfinite(noise)):
        raise ValueError("audit instrument noise variance must be finite and positive")
    clean_error = mean - clean
    noisy_error = mean - heldout
    result: dict[str, Any] = {
        "schema": "paper2.audit-footprint-metrics/v1",
        "count": int(len(mean)),
        "variance_source": "variance_of_per_path_full_dem_footprint_projections",
        "diagonal_pixel_variance_projection": False,
        "instrument_noise_used_for_calibration": False,
        "clean": {
            "rmse_m": float(np.sqrt(np.mean(clean_error**2))),
            "mae_m": float(np.mean(np.abs(clean_error))),
            "bias_m": float(np.mean(clean_error)),
        },
        "heldout_noisy": {
            "rmse_m": float(np.sqrt(np.mean(noisy_error**2))),
            "mae_m": float(np.mean(np.abs(noisy_error))),
            "bias_m": float(np.mean(noisy_error)),
        },
        "probabilistic_available": calibration_scale is not None,
    }
    if calibration_scale is None:
        return result
    if (
        not np.isfinite(calibration_scale)
        or calibration_scale <= 0
        or not np.isfinite(calibration_floor_variance)
        or calibration_floor_variance < 0
    ):
        raise ValueError("audit footprint calibration scale/floor is invalid")
    calibrated = np.maximum(
        calibration_scale * raw_variance, calibration_floor_variance
    )
    # A zero path variance may occur for a degenerate ensemble; the frozen floor
    # must still make every formal Gaussian score well-defined.
    if np.any(calibrated <= 0):
        raise ValueError("calibrated footprint variance must be strictly positive")
    clean_probabilistic = gaussian_probabilistic_metrics(
        mean, np.sqrt(calibrated), clean, np.ones(mean.shape, dtype=bool)
    )
    total_variance = calibrated + noise
    noisy_probabilistic = gaussian_probabilistic_metrics(
        mean, np.sqrt(total_variance), heldout, np.ones(mean.shape, dtype=bool)
    )
    standardized = (heldout - mean) / np.sqrt(total_variance)
    result["clean"]["probabilistic"] = clean_probabilistic
    result["heldout_noisy"]["probabilistic"] = {
        **noisy_probabilistic,
        "standardized_bias": float(np.mean(standardized)),
        "reduced_chi_square": float(np.mean(standardized**2)),
    }
    result["calibrated_model_variance_mean_m2"] = float(np.mean(calibrated))
    result["instrument_noise_variance_mean_m2"] = float(np.mean(noise))
    result["instrument_noise_added_exactly_once"] = True
    return result


@dataclass(frozen=True)
class TileCurve:
    tile_id: str
    budgets: tuple[int, ...]
    normalized_mse: tuple[float, ...]


def tile_equal_mean(curves: Sequence[TileCurve]) -> tuple[np.ndarray, np.ndarray]:
    if not curves:
        raise ValueError("at least one tile curve is required")
    budgets = np.asarray(curves[0].budgets)
    rows = []
    for curve in curves:
        if not np.array_equal(budgets, np.asarray(curve.budgets)):
            raise ValueError("tile curves must share frozen budgets")
        rows.append(np.asarray(curve.normalized_mse, dtype=np.float64))
    return budgets, np.mean(np.stack(rows), axis=0)


def paired_tile_bootstrap(
    differences: Sequence[float], *, replicates: int = 10_000, seed: int
) -> Mapping[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap requires at least two finite tile-level paired differences")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "two_sided_low": float(np.quantile(estimates, 0.025)),
        "two_sided_high": float(np.quantile(estimates, 0.975)),
        "one_sided_low": float(np.quantile(estimates, 0.05)),
        "one_sided_high": float(np.quantile(estimates, 0.95)),
    }
