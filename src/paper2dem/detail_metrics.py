from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage


DEFAULT_DETAIL_SCALES_M = (40.0, 80.0, 160.0)
DEFAULT_LOWPASS_SCALE_M = 320.0


def _as_field(
    values: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if field.ndim != 2 or field.shape != valid.shape:
        raise ValueError("detail fields and masks must be equal two-dimensional arrays")
    valid = valid & np.isfinite(field)
    if not np.any(valid):
        raise ValueError("detail field contains no finite valid pixels")
    return field, valid


def masked_gaussian_smooth(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sigma_pixels: float,
    minimum_support: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian smoothing by normalized convolution without invalid-pixel leakage."""

    field, valid = _as_field(values, valid_mask)
    if not np.isfinite(sigma_pixels) or sigma_pixels <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    if not 0 < minimum_support <= 1:
        raise ValueError("minimum_support must lie in (0, 1]")
    numerator = ndimage.gaussian_filter(
        np.where(valid, field, 0.0), sigma=sigma_pixels, mode="constant", cval=0.0
    )
    denominator = ndimage.gaussian_filter(
        valid.astype(np.float64), sigma=sigma_pixels, mode="constant", cval=0.0
    )
    support = valid & (denominator >= minimum_support)
    smooth = np.full(field.shape, np.nan, dtype=np.float64)
    usable = denominator > 1.0e-12
    smooth[usable] = numerator[usable] / denominator[usable]
    return smooth, support


def detail_field(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    pixel_size_m: float,
    scale_m: float,
    minimum_support: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Gaussian high-pass component at one physical terrain scale."""

    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be finite and positive")
    if not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("scale_m must be finite and positive")
    field, valid = _as_field(values, valid_mask)
    smooth, support = masked_gaussian_smooth(
        field,
        valid,
        sigma_pixels=max(scale_m / pixel_size_m, 0.5),
        minimum_support=minimum_support,
    )
    detail = np.full(field.shape, np.nan, dtype=np.float64)
    detail[support] = field[support] - smooth[support]
    return detail, support


@dataclass(frozen=True)
class DetailThresholdCalibration:
    scales_m: tuple[float, ...]
    thresholds_m: tuple[float, ...]
    quantile: float
    sample_count_by_scale: tuple[int, ...]
    pixel_size_m: float
    seed: int

    SCHEMA = "paper2.multiscale-detail-thresholds/v1"

    def __post_init__(self) -> None:
        if not self.scales_m or len(self.scales_m) != len(self.thresholds_m):
            raise ValueError("detail calibration scales and thresholds must align")
        if len(self.scales_m) != len(self.sample_count_by_scale):
            raise ValueError("detail calibration sample counts must align with scales")
        if any(not np.isfinite(value) or value <= 0 for value in self.scales_m):
            raise ValueError("detail scales must be finite and positive")
        if any(not np.isfinite(value) or value < 0 for value in self.thresholds_m):
            raise ValueError("detail thresholds must be finite and nonnegative")
        if any(value < 1 for value in self.sample_count_by_scale):
            raise ValueError("each detail scale requires calibration samples")
        if not 0 < self.quantile < 1:
            raise ValueError("detail calibration quantile must lie in (0, 1)")
        if not np.isfinite(self.pixel_size_m) or self.pixel_size_m <= 0:
            raise ValueError("detail calibration pixel size must be positive")

    @property
    def threshold_by_scale(self) -> dict[float, float]:
        return dict(zip(self.scales_m, self.thresholds_m, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "scales_m": list(self.scales_m),
            "thresholds_m": list(self.thresholds_m),
            "quantile": self.quantile,
            "sample_count_by_scale": list(self.sample_count_by_scale),
            "pixel_size_m": self.pixel_size_m,
            "seed": self.seed,
            "semantics": (
                "absolute reference Gaussian high-pass magnitude quantile estimated "
                "from development-training targets only"
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DetailThresholdCalibration:
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError(f"unsupported detail calibration schema: {payload.get('schema')}")
        return cls(
            scales_m=tuple(float(value) for value in payload["scales_m"]),
            thresholds_m=tuple(float(value) for value in payload["thresholds_m"]),
            quantile=float(payload["quantile"]),
            sample_count_by_scale=tuple(
                int(value) for value in payload["sample_count_by_scale"]
            ),
            pixel_size_m=float(payload["pixel_size_m"]),
            seed=int(payload["seed"]),
        )


def calibrate_detail_thresholds(
    references: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    pixel_size_m: float,
    scales_m: Sequence[float] = DEFAULT_DETAIL_SCALES_M,
    quantile: float = 0.80,
    maximum_samples_per_field: int = 4096,
    seed: int = 20260810,
) -> DetailThresholdCalibration:
    """Calibrate significant-detail thresholds from development targets only."""

    scales = tuple(float(value) for value in scales_m)
    if not scales or len(set(scales)) != len(scales):
        raise ValueError("detail scales must be nonempty and unique")
    if maximum_samples_per_field < 1:
        raise ValueError("maximum_samples_per_field must be positive")
    generator = np.random.default_rng(seed)
    samples: dict[float, list[np.ndarray]] = {scale: [] for scale in scales}
    field_count = 0
    for values, valid_mask in references:
        field_count += 1
        for scale in scales:
            detail, support = detail_field(
                values,
                valid_mask,
                pixel_size_m=pixel_size_m,
                scale_m=scale,
            )
            available = np.abs(detail[support])
            if available.size > maximum_samples_per_field:
                chosen = generator.choice(
                    available.size, size=maximum_samples_per_field, replace=False
                )
                available = available[chosen]
            if available.size:
                samples[scale].append(available.astype(np.float32, copy=False))
    if field_count == 0:
        raise ValueError("detail calibration requires at least one reference field")
    thresholds = []
    counts = []
    for scale in scales:
        if not samples[scale]:
            raise ValueError(f"detail scale {scale:g} m has no calibration samples")
        values = np.concatenate(samples[scale]).astype(np.float64, copy=False)
        thresholds.append(float(np.quantile(values, quantile)))
        counts.append(int(values.size))
    return DetailThresholdCalibration(
        scales_m=scales,
        thresholds_m=tuple(thresholds),
        quantile=float(quantile),
        sample_count_by_scale=tuple(counts),
        pixel_size_m=float(pixel_size_m),
        seed=int(seed),
    )


def _safe_ratio(numerator: float, denominator: float, *, empty: float = 0.0) -> float:
    return empty if denominator <= 0 else float(numerator / denominator)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) <= 0 or np.std(right) <= 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def multiscale_detail_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    *,
    pixel_size_m: float,
    calibration: DetailThresholdCalibration,
    baseline: np.ndarray | None = None,
    lowpass_scale_m: float = DEFAULT_LOWPASS_SCALE_M,
) -> dict[str, Any]:
    """Measure authentic detail retention and unsupported predicted detail.

    False detail is predicted high-pass structure above a train-calibrated
    physical-scale threshold where the reference high-pass magnitude is below
    that same threshold. This keeps the definition independent of test outcomes.
    """

    predicted, valid = _as_field(estimate, valid_mask)
    truth, truth_valid = _as_field(reference, valid_mask)
    if predicted.shape != truth.shape:
        raise ValueError("estimate and reference shapes differ")
    valid &= truth_valid
    baseline_field = None
    if baseline is not None:
        baseline_field, baseline_valid = _as_field(baseline, valid)
        if baseline_field.shape != predicted.shape:
            raise ValueError("baseline shape differs from estimate")
        valid &= baseline_valid

    scale_rows: list[dict[str, Any]] = []
    for scale, threshold in calibration.threshold_by_scale.items():
        predicted_detail, predicted_support = detail_field(
            predicted, valid, pixel_size_m=pixel_size_m, scale_m=scale
        )
        true_detail, true_support = detail_field(
            truth, valid, pixel_size_m=pixel_size_m, scale_m=scale
        )
        support = predicted_support & true_support
        baseline_detail = None
        if baseline_field is not None:
            baseline_detail, baseline_support = detail_field(
                baseline_field, valid, pixel_size_m=pixel_size_m, scale_m=scale
            )
            support &= baseline_support
        if not np.any(support):
            raise ValueError(f"detail scale {scale:g} m has no common support")
        pred = predicted_detail[support]
        ref = true_detail[support]
        error = pred - ref
        true_positive_support = np.abs(ref) >= threshold
        predicted_positive_support = np.abs(pred) >= threshold
        true_positive = int(np.count_nonzero(true_positive_support))
        predicted_positive = int(np.count_nonzero(predicted_positive_support))
        intersection = true_positive_support & predicted_positive_support
        true_intersection = int(np.count_nonzero(intersection))
        false_positive = int(
            np.count_nonzero(predicted_positive_support & ~true_positive_support)
        )
        precision = _safe_ratio(
            true_intersection,
            predicted_positive,
            empty=1.0 if true_positive == 0 else 0.0,
        )
        recall = _safe_ratio(
            true_intersection,
            true_positive,
            empty=1.0,
        )
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall, empty=0.0)
        predicted_energy = float(np.sum(pred**2))
        reference_energy = float(np.sum(ref**2))
        false_detail_mask = ~true_positive_support
        false_energy = float(np.sum(pred[false_detail_mask] ** 2))
        scale_row: dict[str, Any] = {
            "scale_m": float(scale),
            "threshold_m": float(threshold),
            "valid_pixels": int(pred.size),
            "reference_detail_pixels": true_positive,
            "predicted_detail_pixels": predicted_positive,
            "detail_rmse_m": float(np.sqrt(np.mean(error**2))),
            "detail_mae_m": float(np.mean(np.abs(error))),
            "detail_correlation": _correlation(pred, ref),
            "detail_energy_ratio": _safe_ratio(
                predicted_energy, reference_energy, empty=1.0 if predicted_energy == 0 else 0.0
            ),
            "detail_support_precision": precision,
            "detail_support_recall": recall,
            "detail_support_f1": f1,
            "detail_sign_agreement": (
                float(np.mean(np.sign(pred[intersection]) == np.sign(ref[intersection])))
                if true_intersection
                else None
            ),
            "false_detail_pixels": false_positive,
            "false_detail_fraction_valid": float(false_positive / pred.size),
            "false_detail_fraction_predicted": _safe_ratio(
                false_positive, predicted_positive, empty=0.0
            ),
            "false_detail_energy_fraction": _safe_ratio(
                false_energy, predicted_energy, empty=0.0
            ),
            "false_detail_rmse_m": float(
                np.sqrt(np.mean(error[false_detail_mask] ** 2))
            )
            if np.any(false_detail_mask)
            else 0.0,
            "true_detail_rmse_m": float(
                np.sqrt(np.mean(error[true_positive_support] ** 2))
            )
            if true_positive
            else 0.0,
        }
        if baseline_detail is not None:
            base_error = baseline_detail[support] - ref
            base_rmse = float(np.sqrt(np.mean(base_error**2)))
            scale_row["baseline_detail_rmse_m"] = base_rmse
            scale_row["detail_rmse_improvement_fraction"] = _safe_ratio(
                base_rmse - scale_row["detail_rmse_m"], base_rmse, empty=0.0
            )
        scale_rows.append(scale_row)

    predicted_low, predicted_low_support = masked_gaussian_smooth(
        predicted,
        valid,
        sigma_pixels=max(lowpass_scale_m / pixel_size_m, 0.5),
    )
    true_low, true_low_support = masked_gaussian_smooth(
        truth,
        valid,
        sigma_pixels=max(lowpass_scale_m / pixel_size_m, 0.5),
    )
    low_support = predicted_low_support & true_low_support
    low_error = predicted_low[low_support] - true_low[low_support]
    result: dict[str, Any] = {
        "schema": "paper2.multiscale-detail-metrics/v1",
        "pixel_size_m": float(pixel_size_m),
        "lowpass_scale_m": float(lowpass_scale_m),
        "lowpass_rmse_m": float(np.sqrt(np.mean(low_error**2))),
        "scales": scale_rows,
        "mean_detail_rmse_m": float(
            np.mean([row["detail_rmse_m"] for row in scale_rows])
        ),
        "mean_detail_support_f1": float(
            np.mean([row["detail_support_f1"] for row in scale_rows])
        ),
        "mean_false_detail_fraction_valid": float(
            np.mean([row["false_detail_fraction_valid"] for row in scale_rows])
        ),
        "mean_false_detail_energy_fraction": float(
            np.mean([row["false_detail_energy_fraction"] for row in scale_rows])
        ),
    }
    if baseline_field is not None:
        baseline_low, baseline_low_support = masked_gaussian_smooth(
            baseline_field,
            valid,
            sigma_pixels=max(lowpass_scale_m / pixel_size_m, 0.5),
        )
        baseline_support = low_support & baseline_low_support
        baseline_low_error = baseline_low[baseline_support] - true_low[baseline_support]
        result["baseline_lowpass_rmse_m"] = float(
            np.sqrt(np.mean(baseline_low_error**2))
        )
        result["mean_detail_rmse_improvement_fraction"] = float(
            np.mean(
                [row["detail_rmse_improvement_fraction"] for row in scale_rows]
            )
        )
    return result
