from __future__ import annotations

import numpy as np

from paper2dem.detail_metrics import (
    DetailThresholdCalibration,
    calibrate_detail_thresholds,
    detail_field,
    masked_gaussian_smooth,
    multiscale_detail_metrics,
)


def _calibration(threshold: float = 0.1) -> DetailThresholdCalibration:
    return DetailThresholdCalibration(
        scales_m=(2.0, 4.0),
        thresholds_m=(threshold, threshold),
        quantile=0.8,
        sample_count_by_scale=(100, 100),
        pixel_size_m=1.0,
        seed=7,
    )


def test_masked_gaussian_smoothing_does_not_import_invalid_values() -> None:
    field = np.ones((41, 41), dtype=np.float64)
    valid = np.ones(field.shape, dtype=bool)
    valid[18:23, 18:23] = False
    field[~valid] = 1.0e9
    smooth, support = masked_gaussian_smooth(
        field, valid, sigma_pixels=1.0, minimum_support=0.5
    )
    assert np.allclose(smooth[support], 1.0)


def test_perfect_reconstruction_has_perfect_detail_metrics() -> None:
    y, x = np.indices((96, 96), dtype=np.float64)
    reference = 3.0 * np.sin(x / 3.0) + 2.0 * np.cos(y / 5.0)
    valid = np.ones(reference.shape, dtype=bool)
    metrics = multiscale_detail_metrics(
        reference,
        reference,
        valid,
        pixel_size_m=1.0,
        calibration=_calibration(),
        baseline=np.zeros_like(reference),
        lowpass_scale_m=8.0,
    )
    assert metrics["mean_detail_rmse_m"] == 0.0
    assert metrics["mean_detail_support_f1"] == 1.0
    assert metrics["mean_false_detail_fraction_valid"] == 0.0
    assert metrics["mean_detail_rmse_improvement_fraction"] == 1.0


def test_checkerboard_on_flat_reference_is_detected_as_false_detail() -> None:
    y, x = np.indices((96, 96))
    estimate = ((x + y) % 2).astype(np.float64) * 2.0 - 1.0
    reference = np.zeros_like(estimate)
    valid = np.ones(reference.shape, dtype=bool)
    metrics = multiscale_detail_metrics(
        estimate,
        reference,
        valid,
        pixel_size_m=1.0,
        calibration=_calibration(0.2),
        lowpass_scale_m=8.0,
    )
    assert metrics["mean_false_detail_fraction_valid"] > 0.9
    assert metrics["mean_false_detail_energy_fraction"] > 0.99
    assert metrics["mean_detail_support_f1"] == 0.0


def test_calibration_is_positive_and_serializable() -> None:
    y, x = np.indices((64, 64), dtype=np.float64)
    valid = np.ones((64, 64), dtype=bool)
    fields = [
        (np.sin(x / 4.0) + np.cos(y / 6.0), valid),
        (2.0 * np.sin(x / 7.0) - np.cos(y / 3.0), valid),
    ]
    calibration = calibrate_detail_thresholds(
        fields,
        pixel_size_m=1.0,
        scales_m=(2.0, 4.0),
        maximum_samples_per_field=500,
        seed=11,
    )
    assert all(value > 0 for value in calibration.thresholds_m)
    assert DetailThresholdCalibration.from_dict(calibration.to_dict()) == calibration


def test_detail_field_rejects_invalid_physical_scale() -> None:
    values = np.ones((8, 8), dtype=np.float64)
    valid = np.ones(values.shape, dtype=bool)
    try:
        detail_field(values, valid, pixel_size_m=1.0, scale_m=0.0)
    except ValueError as error:
        assert "scale_m" in str(error)
    else:
        raise AssertionError("zero detail scale should fail")
