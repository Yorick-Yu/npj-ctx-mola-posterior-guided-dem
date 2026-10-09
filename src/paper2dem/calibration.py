from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np

from .hashing import canonical_sha256, is_sha256, write_canonical_json


CALIBRATION_SCHEMA = "paper2.calibration-manifest/v2"
CALIBRATION_SELECTION_RULE = "minimum_nll_scale_floor_on_calibration_role"
CLEAN_TARGETS = {
    "pixel": "Zstar_minus_complete_path_mean",
    "footprint": "F_Zstar_minus_mean_of_per_path_F_Z",
}
VARIANCE_SOURCES = {
    "pixel": "complete_dem_path_sample_variance",
    "footprint": "variance_of_per_path_full_dem_footprint_projections",
}


@dataclass(frozen=True)
class VarianceCalibrator:
    target: str
    scale: float
    floor_variance: float

    def validate(self) -> None:
        if self.target not in {"pixel", "footprint"}:
            raise ValueError("calibrator target must be pixel or footprint")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("calibration scale must be finite and positive")
        if not np.isfinite(self.floor_variance) or self.floor_variance < 0:
            raise ValueError("variance floor must be finite and non-negative")

    def apply(self, variance: np.ndarray) -> np.ndarray:
        self.validate()
        value = np.asarray(variance, dtype=np.float64)
        if np.any(value < 0) or not np.all(np.isfinite(value)):
            raise ValueError("uncalibrated variance must be finite and non-negative")
        return np.maximum(self.scale * value, self.floor_variance)


def fit_variance_calibrator(
    variance: np.ndarray,
    residual: np.ndarray,
    valid_mask: np.ndarray,
    *,
    target: str,
    minimum_variance: float = 1.0e-8,
) -> VarianceCalibrator:
    predicted = np.asarray(variance, dtype=np.float64)
    error = np.asarray(residual, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if predicted.shape != error.shape or predicted.shape != valid.shape:
        raise ValueError("variance, residual, and mask must share shape")
    if target not in CLEAN_TARGETS:
        raise ValueError("calibration target must be pixel or footprint")
    if not np.isfinite(minimum_variance) or minimum_variance <= 0:
        raise ValueError("minimum calibration variance must be finite and positive")
    valid &= np.isfinite(predicted) & np.isfinite(error) & (predicted >= 0)
    if not np.any(valid):
        raise ValueError("calibration set has no valid values")
    stabilized = np.maximum(predicted[valid], minimum_variance)
    # Gaussian NLL optimum for one preregistered global multiplicative scale.
    # Both variance inputs and residual targets are clean reconstruction terms;
    # instrument noise is never passed to this function by the formal fitter.
    scale = float(np.mean(error[valid] ** 2 / stabilized))
    scale = max(scale, np.finfo(np.float64).eps)
    floor = float(np.quantile(error[valid] ** 2, 0.01) * 1.0e-2)
    result = VarianceCalibrator(
        target=target,
        scale=scale,
        floor_variance=max(floor, minimum_variance),
    )
    result.validate()
    return result


def calibration_manifest_payload(
    *,
    calibrator: VarianceCalibrator,
    checkpoint_sha256: str,
    calibration_asset_lock_sha256: str,
    role_manifest_sha256: str,
    sealed_reference_lock_sha256: str,
    fitting_input_sha256: str,
    method: str,
    training_seed: int,
    sample_count: int,
    tile_ids: Sequence[str],
    inference_manifest_records: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    calibrator.validate()
    tiles = tuple(str(value) for value in tile_ids)
    if len(tiles) != 12 or len(set(tiles)) != 12:
        raise ValueError("formal calibration requires exactly 12 unique calibration tiles")
    if sample_count <= 0 or training_seed < 0 or not method:
        raise ValueError("calibration identity/sample count is invalid")
    for name, digest in (
        ("checkpoint", checkpoint_sha256),
        ("asset_lock", calibration_asset_lock_sha256),
        ("role_manifest", role_manifest_sha256),
        ("sealed_reference_lock", sealed_reference_lock_sha256),
        ("fitting_input", fitting_input_sha256),
    ):
        if not is_sha256(digest):
            raise ValueError(f"calibration {name} hash is invalid")
    records = [dict(record) for record in inference_manifest_records]
    if len(records) != 12:
        raise ValueError("calibration must bind one inference manifest per calibration tile")
    if {record.get("tile_id") for record in records} != set(tiles):
        raise ValueError("calibration inference records differ from the calibration tiles")
    for record in records:
        if set(record) != {
            "tile_id",
            "inference_manifest_relative_path",
            "inference_manifest_sha256",
            "inference_manifest_semantic_sha256",
            "prediction_sha256",
            "audit_projection_sha256",
            "audit_plan_sha256",
            "sealed_reference_sha256",
        }:
            raise ValueError("calibration inference record fields differ")
        for name in (
            "inference_manifest_sha256",
            "inference_manifest_semantic_sha256",
            "prediction_sha256",
            "audit_projection_sha256",
            "audit_plan_sha256",
            "sealed_reference_sha256",
        ):
            if not is_sha256(str(record[name])):
                raise ValueError(f"calibration inference record {name} is invalid")
        relative = Path(str(record["inference_manifest_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("calibration inference manifest path must stay below paper2")
    payload: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "target": calibrator.target,
        "method": method,
        "training_seed": int(training_seed),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_asset_lock_sha256": calibration_asset_lock_sha256,
        "role_manifest_sha256": role_manifest_sha256,
        "sealed_reference_lock_sha256": sealed_reference_lock_sha256,
        "fitting_input_sha256": fitting_input_sha256,
        "sample_count": int(sample_count),
        "tile_count": len(tiles),
        "tile_ids": sorted(tiles),
        "inference_manifests": sorted(records, key=lambda value: str(value["tile_id"])),
        "scale": calibrator.scale,
        "floor_variance": calibrator.floor_variance,
        "selection_rule": CALIBRATION_SELECTION_RULE,
        "clean_error_target": CLEAN_TARGETS[calibrator.target],
        "variance_source": VARIANCE_SOURCES[calibrator.target],
        "instrument_noise_absorbed": False,
        "instrument_noise_fields_read_during_fit": False,
        "role": "uncertainty_calibration",
        "test_results_inspected": False,
        "audit_footprints_per_tile": 128,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def write_calibration_manifest(
    path: str | Path,
    *,
    calibrator: VarianceCalibrator,
    checkpoint_sha256: str,
    calibration_asset_lock_sha256: str,
    role_manifest_sha256: str,
    sealed_reference_lock_sha256: str,
    fitting_input_sha256: str,
    method: str,
    training_seed: int,
    sample_count: int,
    tile_ids: Sequence[str],
    inference_manifest_records: Sequence[Mapping[str, str]],
    paper2_root: str | Path,
) -> Path:
    return write_canonical_json(
        path,
        calibration_manifest_payload(
            calibrator=calibrator,
            checkpoint_sha256=checkpoint_sha256,
            calibration_asset_lock_sha256=calibration_asset_lock_sha256,
            role_manifest_sha256=role_manifest_sha256,
            sealed_reference_lock_sha256=sealed_reference_lock_sha256,
            fitting_input_sha256=fitting_input_sha256,
            method=method,
            training_seed=training_seed,
            sample_count=sample_count,
            tile_ids=tile_ids,
            inference_manifest_records=inference_manifest_records,
        ),
        paper2_root=paper2_root,
    )


def read_calibration_manifest(path: str | Path) -> tuple[VarianceCalibrator, Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("formal calibration manifest must use v2 clean-target schema")
    if payload.get("manifest_sha256") != expected:
        raise ValueError("calibration manifest hash mismatch")
    calibrator = VarianceCalibrator(
        target=str(payload["target"]),
        scale=float(payload["scale"]),
        floor_variance=float(payload["floor_variance"]),
    )
    calibrator.validate()
    if (
        payload.get("role") != "uncertainty_calibration"
        or payload.get("test_results_inspected") is not False
        or payload.get("instrument_noise_absorbed") is not False
        or payload.get("instrument_noise_fields_read_during_fit") is not False
        or payload.get("selection_rule") != CALIBRATION_SELECTION_RULE
        or payload.get("clean_error_target") != CLEAN_TARGETS[calibrator.target]
        or payload.get("variance_source") != VARIANCE_SOURCES[calibrator.target]
        or int(payload.get("tile_count", -1)) != 12
        or int(payload.get("audit_footprints_per_tile", -1)) != 128
    ):
        raise ValueError("calibration clean-target/role/noise contract differs")
    tile_ids = payload.get("tile_ids")
    records = payload.get("inference_manifests")
    if (
        not isinstance(tile_ids, list)
        or len(tile_ids) != 12
        or len(set(str(value) for value in tile_ids)) != 12
        or not isinstance(records, list)
        or len(records) != 12
    ):
        raise ValueError("calibration tile/inference set is not exact")
    for name in (
        "checkpoint_sha256",
        "calibration_asset_lock_sha256",
        "role_manifest_sha256",
        "sealed_reference_lock_sha256",
        "fitting_input_sha256",
    ):
        if not is_sha256(str(payload.get(name, ""))):
            raise ValueError(f"calibration {name} is invalid")
    if int(payload.get("sample_count", 0)) <= 0 or int(payload.get("training_seed", -1)) < 0:
        raise ValueError("calibration sample/seed identity is invalid")
    return calibrator, payload
