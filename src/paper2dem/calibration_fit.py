"""Formal clean-target calibration over the frozen uncertainty-calibration role."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .audit_footprints import (
    read_audit_footprint_plan,
    read_audit_path_projections,
    validate_audit_plan_binding,
)
from .calibration import fit_variance_calibrator, write_calibration_manifest
from .contracts import PlanContract, ReconstructionProblem
from .hashing import canonical_sha256, file_sha256, guard_paper2_path, read_json
from .matrix import InferenceManifest
from .models import canonical_model_identity, method_spec
from .sealed_reference import read_sealed_reference_lock


CALIBRATION_FIT_MODE_SHA256 = canonical_sha256(
    {
        "mode": "formal-clean-target-calibration-fit",
        "role": "uncertainty_calibration",
        "instrument_noise_read": False,
        "audit_projection": "per-complete-dem-path",
    }
)


def _artifact_path(
    manifest: InferenceManifest, name: str, *, root: Path
) -> Path:
    matches = [artifact for artifact in manifest.artifacts if artifact.name == name]
    if len(matches) != 1:
        raise ValueError(f"calibration inference requires one {name} artifact")
    artifact = matches[0]
    path = guard_paper2_path(root / artifact.relative_path, paper2_root=root)
    if not path.is_file() or file_sha256(path) != artifact.sha256:
        raise ValueError(f"calibration inference {name} artifact bytes differ")
    return path


def fit_formal_calibration_pair(
    *,
    paper2_root: str | Path,
    entries: Sequence[Mapping[str, Any]],
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    sealed_reference_lock_path: str | Path,
    checkpoint_manifest_path: str | Path,
    method: str,
    pixel_output_path: str | Path,
    footprint_output_path: str | Path,
) -> tuple[Path, Path]:
    """Fit pixel and footprint calibrators from exactly the same 12 clean tiles.

    Audit observation returns are intentionally absent from this API, so instrument
    noise cannot accidentally enter either fitting target.
    """

    root = Path(paper2_root).resolve()
    role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
    asset_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    reference_lock_path = guard_paper2_path(sealed_reference_lock_path, paper2_root=root)
    checkpoint_path = guard_paper2_path(checkpoint_manifest_path, paper2_root=root)
    for path in (role_path, asset_path, reference_lock_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = method_spec(method)
    if not spec.generative:
        raise ValueError("formal variance calibration is defined only for generative methods")
    identity = canonical_model_identity(method)
    roles = read_json(role_path)
    if roles.get("schema") != "paper2.role-manifest/v1":
        raise ValueError("calibration requires paper2.role-manifest/v1")
    calibration_tiles = tuple(str(value) for value in roles.get("roles", {}).get(
        "uncertainty_calibration", ()
    ))
    if len(calibration_tiles) != 12 or len(set(calibration_tiles)) != 12:
        raise ValueError("role manifest must contain exactly 12 uncertainty-calibration tiles")
    if len(entries) != 12 or {str(entry.get("tile_id")) for entry in entries} != set(
        calibration_tiles
    ):
        raise ValueError("calibration fit entries must exactly cover the frozen 12 tiles")
    checkpoint = read_json(checkpoint_path)
    if (
        checkpoint.get("status") != "complete"
        or canonical_model_identity(str(checkpoint.get("method"))) != identity
        or str(checkpoint.get("model_identity")) != identity
    ):
        raise ValueError("calibration checkpoint manifest identity/state differs")
    checkpoint_sha256 = str(checkpoint.get("checkpoint_sha256", ""))
    training_seed = int(checkpoint.get("seed", -1))
    reference_lock = read_sealed_reference_lock(
        reference_lock_path, paper2_root=root, verify_files=True
    )
    references = {str(entry["tile_id"]): entry for entry in reference_lock["entries"]}
    if not set(calibration_tiles).issubset(references):
        raise ValueError("sealed reference lock lacks calibration tiles")

    pixel_variances: list[np.ndarray] = []
    pixel_residuals: list[np.ndarray] = []
    footprint_variances: list[np.ndarray] = []
    footprint_residuals: list[np.ndarray] = []
    records: list[dict[str, str]] = []
    not_applied = canonical_sha256({"calibration_manifest": "not-applied"})
    footprint_not_applied = canonical_sha256(
        {"footprint_calibration_manifest": "not-applied"}
    )
    for raw in sorted(entries, key=lambda value: str(value["tile_id"])):
        tile_id = str(raw["tile_id"])
        required = {
            "tile_id",
            "inference_manifest",
            "problem",
            "plan",
            "audit_footprint_plan",
            "sealed_reference",
        }
        if set(raw) != required:
            raise ValueError(
                f"calibration entry fields differ for {tile_id}: "
                f"missing={sorted(required - set(raw))}, extra={sorted(set(raw) - required)}"
            )
        inference_path = guard_paper2_path(raw["inference_manifest"], paper2_root=root)
        problem_path = guard_paper2_path(raw["problem"], paper2_root=root)
        plan_path = guard_paper2_path(raw["plan"], paper2_root=root)
        audit_plan_path = guard_paper2_path(raw["audit_footprint_plan"], paper2_root=root)
        reference_path = guard_paper2_path(raw["sealed_reference"], paper2_root=root)
        inference = InferenceManifest.read(inference_path)
        trace = dict(inference.trace_hashes)
        if (
            inference.status != "complete"
            or inference.key.tile_id != tile_id
            or canonical_model_identity(inference.key.method) != identity
            or inference.key.training_seed != training_seed
            or trace.get("checkpoint_sha256") != checkpoint_sha256
            or trace.get("asset_lock_sha256") != file_sha256(asset_path)
            or trace.get("calibration_fit_mode_sha256") != CALIBRATION_FIT_MODE_SHA256
            or trace.get("calibration_manifest_sha256") != not_applied
            or trace.get("footprint_calibration_manifest_sha256")
            != footprint_not_applied
        ):
            raise ValueError(f"calibration inference identity/clean-fit trace differs: {tile_id}")
        problem = ReconstructionProblem.read(problem_path)
        plan = PlanContract.read(plan_path)
        audit_plan = read_audit_footprint_plan(audit_plan_path)
        operator = validate_audit_plan_binding(
            audit_plan, problem=problem, measurement_plan=plan
        )
        if (
            trace.get("problem_sha256") != file_sha256(problem_path)
            or trace.get("plan_sha256") != file_sha256(plan_path)
            or trace.get("audit_footprint_plan_sha256") != file_sha256(audit_plan_path)
        ):
            raise ValueError(f"calibration inference data/audit trace differs: {tile_id}")
        prediction_path = _artifact_path(inference, "prediction", root=root)
        projection_path = _artifact_path(
            inference, "audit_footprint_projection", root=root
        )
        with np.load(prediction_path, allow_pickle=False) as archive:
            mean = np.asarray(archive["mean"], dtype=np.float64)
            variance = np.asarray(archive["variance"], dtype=np.float64)
        projection_meta, _, footprint_mean, footprint_variance = (
            read_audit_path_projections(projection_path)
        )
        if (
            projection_meta.get("matrix_key_identifier") != inference.key.identifier
            or projection_meta.get("audit_plan_file_sha256") != file_sha256(audit_plan_path)
            or projection_meta.get("variance_source")
            != "complete_dem_path_ensemble_spatial_covariance"
            or int(projection_meta.get("path_count", 0)) < 2
        ):
            raise ValueError(f"calibration path projection contract differs: {tile_id}")
        locked_reference = references[tile_id]
        if (
            locked_reference["reference_relative_path"]
            != reference_path.relative_to(root).as_posix()
            or locked_reference["reference_sha256"] != file_sha256(reference_path)
            or locked_reference["problem_sha256"] != plan.problem_sha256
        ):
            raise ValueError(f"calibration reference differs from sealed lock: {tile_id}")
        with np.load(reference_path, allow_pickle=False) as archive:
            reference = np.asarray(archive["dem"], dtype=np.float64)
            reference_valid = np.asarray(archive["valid_mask"], dtype=bool)
        if mean.shape != problem.z0.shape or variance.shape != mean.shape or reference.shape != mean.shape:
            raise ValueError(f"calibration pixel grids differ: {tile_id}")
        valid = problem.valid_mask & reference_valid & np.isfinite(reference)
        valid &= np.isfinite(mean) & np.isfinite(variance) & (variance >= 0)
        if not np.any(valid):
            raise ValueError(f"calibration tile has no valid clean pixels: {tile_id}")
        pixel_variances.append(variance[valid])
        pixel_residuals.append((reference - mean)[valid])
        supported = np.unique(operator.indices)
        if not np.all(reference_valid.ravel()[supported]) or not np.all(
            np.isfinite(reference.ravel()[supported])
        ):
            raise ValueError(
                f"calibration reference lacks full audit-footprint support: {tile_id}"
            )
        reference_support = np.where(reference_valid, reference, 0.0)
        clean_footprints = np.asarray(operator @ reference_support.ravel(), dtype=np.float64)
        footprint_variances.append(footprint_variance)
        footprint_residuals.append(clean_footprints - footprint_mean)
        records.append(
            {
                "tile_id": tile_id,
                "inference_manifest_relative_path": inference_path.relative_to(root).as_posix(),
                "inference_manifest_sha256": file_sha256(inference_path),
                "inference_manifest_semantic_sha256": inference.sha256,
                "prediction_sha256": file_sha256(prediction_path),
                "audit_projection_sha256": file_sha256(projection_path),
                "audit_plan_sha256": file_sha256(audit_plan_path),
                "sealed_reference_sha256": file_sha256(reference_path),
            }
        )
    pixel_variance = np.concatenate(pixel_variances)
    pixel_residual = np.concatenate(pixel_residuals)
    footprint_variance = np.concatenate(footprint_variances)
    footprint_residual = np.concatenate(footprint_residuals)
    pixel = fit_variance_calibrator(
        pixel_variance,
        pixel_residual,
        np.ones(pixel_variance.shape, dtype=bool),
        target="pixel",
    )
    footprint = fit_variance_calibrator(
        footprint_variance,
        footprint_residual,
        np.ones(footprint_variance.shape, dtype=bool),
        target="footprint",
    )
    fitting_input_sha256 = canonical_sha256(
        {
            "schema": "paper2.calibration-fitting-input/v1",
            "method": identity,
            "training_seed": training_seed,
            "role_manifest_sha256": file_sha256(role_path),
            "asset_lock_sha256": file_sha256(asset_path),
            "sealed_reference_lock_sha256": file_sha256(reference_lock_path),
            "records": records,
            "clean_targets_only": True,
            "instrument_noise_fields_read": False,
        }
    )
    common = {
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_asset_lock_sha256": file_sha256(asset_path),
        "role_manifest_sha256": file_sha256(role_path),
        "sealed_reference_lock_sha256": file_sha256(reference_lock_path),
        "fitting_input_sha256": fitting_input_sha256,
        "method": identity,
        "training_seed": training_seed,
        "tile_ids": calibration_tiles,
        "inference_manifest_records": records,
        "paper2_root": root,
    }
    pixel_path = write_calibration_manifest(
        pixel_output_path,
        calibrator=pixel,
        sample_count=int(pixel_variance.size),
        **common,
    )
    footprint_path = write_calibration_manifest(
        footprint_output_path,
        calibrator=footprint,
        sample_count=int(footprint_variance.size),
        **common,
    )
    return pixel_path, footprint_path
