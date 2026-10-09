from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping
import os
import re
import tempfile

import fcntl

import numpy as np
from scipy import ndimage

from .contracts import AssetFile, AssetQA
from .g0 import (
    SCHEMAS,
    validate_edr_index,
    validate_mola_source_lock,
    validate_paper1_seen_lock,
    validate_processed_lock,
)
from .g0_sidecars import _conflict_groups_hash, _fixed_assignments
from .g0_raster_preparation import (
    COARSE_DEM_ALGORITHM,
    COARSE_DEM_LINEAGE_SCHEMA,
    COREGISTRATION_ALGORITHM,
    COREGISTRATION_CONTROLS_SCHEMA,
    COREGISTRATION_LINEAGE_SCHEMA,
    COREGISTRATION_MODE,
    COREGISTRATION_SOURCE,
    COREGISTRATION_TRANSFORM_FAMILY,
    FROZEN_COREGISTRATION_CONFIG,
    MIN_COREGISTRATION_ERROR_COUNT,
    RASTER_PREPARATION_SCHEMA,
    fit_global_translation,
    load_problem_grid,
    verify_mola_coarse_cube_manifest,
)
from .hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)
from .official_assets import (
    download_official_asset,
    validate_attached_pds3_fixed_length,
)
from .isis_runtime_assets import DEFAULT_LOCK as ISIS_BASE_KERNEL_LOCK
from .isis_runtime_assets import verify_isis_base_kernel_lock
from .preparation import (
    CTXPipelineConfig,
    ImageQA,
    ISD_GENERATE_MAX_ATTEMPTS,
    SPICEQL_RETRYABLE_STAGES,
    ctx_isis_commands,
    image_qa,
    is_transient_spiceql_failure,
    mola_preparation_manifest_path,
    mola_preparation_ready,
    run_ctx_isis,
    verify_ale_spiceql_compatibility,
    verify_csm_environment,
    verify_mola_preparation,
    write_prepared_manifest,
)
from .source_preparation import (
    validate_source_preparation_index,
    verify_source_preparation_manifest,
)


QA_EVIDENCE_SCHEMA = "paper2.g0.ctx-qa-evidence/v2"
QA_REJECTION_EVIDENCE_SCHEMA = "paper2.g0.ctx-qa-rejection-evidence/v2"
PRODUCT_STATE_SCHEMA = "paper2.g0.ctx-product-processing/v1"
GROUP_STATE_SCHEMA = "paper2.g0.ctx-group-processing/v1"
PROCESSING_INDEX_SCHEMA = "paper2.g0.ctx-processing-index/v1"
MAX_PROCESS_WORKERS = 4
EDR_TERMINAL_STATUSES = frozenset({"resolved", "missing", "ambiguous", "rejected"})
ALLOWED_COREGISTRATION_SOURCES = frozenset({COREGISTRATION_SOURCE})
TERMINAL_GROUP_PRODUCT_STATUSES = frozenset(
    {"source_rejected", "qa_failed", "terminal_unavailable"}
)


def _read_object(path: str | Path, *, paper2_root: Path) -> tuple[Path, dict[str, Any]]:
    resolved = guard_paper2_path(path, paper2_root=paper2_root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = read_json(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {resolved}")
    return resolved, payload


def processing_exhausted_group_ids(
    payload: Mapping[str, Any], assignments: list[Mapping[str, Any]]
) -> tuple[str, ...]:
    """Return groups whose complete fixed product order reached terminal G0 QA states."""
    fixed_by_group = {
        str(assignment["group_id"]): [
            str(assignment["representative_product_id"]),
            *(str(value) for value in assignment.get("fallback_product_ids", [])),
        ]
        for assignment in assignments
    }
    if len(fixed_by_group) != len(assignments):
        raise ValueError("processing exhaustion assignments must be unique")
    exhausted: list[str] = []
    observed: set[str] = set()
    for raw in payload.get("groups", []):
        group = dict(raw)
        group_id = str(group.get("group_id", ""))
        if group_id not in fixed_by_group or group_id in observed:
            raise ValueError(f"processing exhaustion group identity differs: {group_id}")
        observed.add(group_id)
        if group.get("state") == "complete":
            continue
        attempts = [dict(value) for value in group.get("attempts", [])]
        product_ids = [str(value.get("product_id", "")) for value in attempts]
        statuses = [str(value.get("status", "")) for value in attempts]
        if product_ids == fixed_by_group[group_id] and all(
            status in TERMINAL_GROUP_PRODUCT_STATUSES for status in statuses
        ):
            exhausted.append(group_id)
    if observed != set(fixed_by_group):
        raise ValueError("processing exhaustion index does not cover fixed assignments")
    return tuple(sorted(exhausted))


def replacement_processing_exhausted_group_ids(
    payload: Mapping[str, Any],
    assignments: list[Mapping[str, Any]],
    *,
    expected_assignments_file_sha256: str,
    expected_role_assignment_sha256: str,
    source_exhausted: tuple[str, ...],
) -> tuple[tuple[str, ...], bool]:
    """Use processing exhaustion only when it belongs to the current generation.

    A source-only replacement generation can legitimately precede creation of its
    processing index.  In that case the durable processing index may still belong
    to the parent generation.  It is safe to retain that index as lineage for the
    pruning audit only when it contains none of the groups being replaced from
    current source-QA evidence.  The pruning gate independently refuses any
    surviving cubes without a matching processing generation.
    """

    if payload.get("schema") != PROCESSING_INDEX_SCHEMA:
        raise ValueError("replacement processing-index schema differs")
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("replacement processing-index groups are invalid")
    group_ids = [str(dict(raw).get("group_id", "")) for raw in raw_groups]
    if "" in group_ids or len(group_ids) != len(set(group_ids)):
        raise ValueError("replacement processing-index group identities are invalid")
    generation_matches = (
        payload.get("assignments_file_sha256")
        == expected_assignments_file_sha256
        and payload.get("role_assignment_sha256")
        == expected_role_assignment_sha256
    )
    if generation_matches:
        return processing_exhausted_group_ids(payload, assignments), True
    overlap = sorted(set(source_exhausted).intersection(group_ids))
    if overlap:
        raise ValueError(
            "stale processing index contains current source-exhausted groups: "
            + ",".join(overlap)
        )
    return (), False


def _verify_mola_shape_sources(
    payload: Mapping[str, Any],
    *,
    config: CTXPipelineConfig,
    paper2_root: Path,
) -> dict[str, Any]:
    """Bind the local MEGR label/image bytes actually consumed by pds2isis."""
    matches = [
        dict(raw)
        for raw in payload.get("products", [])
        if raw.get("product_id") == "MEGR90N000FB" and raw.get("purpose") == "isis_shape_model"
    ]
    if len(matches) != 1:
        raise ValueError("MOLA source lock lacks one MEGR ISIS shape product")
    product = matches[0]
    assets: dict[str, Any] = {}
    for kind in ("image", "label"):
        relative_path = str(product.get(f"{kind}_relative_path", ""))
        path = guard_paper2_path(relative_path, paper2_root=paper2_root)
        if not path.is_file():
            raise FileNotFoundError(f"locked MOLA {kind} is missing: {path}")
        observed = file_sha256(path)
        if product.get(f"{kind}_sha256") != observed:
            raise ValueError(f"locked MOLA {kind} hash differs from local bytes")
        assets[kind] = {
            "relative_path": relative_path,
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    locked_label = guard_paper2_path(assets["label"]["relative_path"], paper2_root=paper2_root)
    if locked_label != config.mola_radius_label.resolve():
        raise ValueError("CTX config MOLA label differs from the locked MEGR label")
    if (
        Path(assets["image"]["relative_path"]).parent
        != Path(assets["label"]["relative_path"]).parent
    ):
        raise ValueError("locked MEGR image and label must share one local directory")
    return assets


def _write_recoverable(path: str | Path, payload: Mapping[str, Any], *, paper2_root: Path) -> Path:
    output = guard_paper2_path(path, paper2_root=paper2_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".checkpoint", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _safe_identifier(value: str, *, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", value):
        raise ValueError(f"{name} contains unsafe path characters: {value!r}")
    return value


def _relative_file_record(path: Path, *, kind: str, paper2_root: Path) -> dict[str, Any]:
    resolved = guard_paper2_path(path, paper2_root=paper2_root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return AssetFile(
        kind=kind,
        relative_path=resolved.relative_to(paper2_root).as_posix(),
        sha256=file_sha256(resolved),
        byte_count=resolved.stat().st_size,
    ).to_dict()


def _qa_dict(qa: ImageQA) -> dict[str, Any]:
    return AssetQA(
        image_coverage_fraction=qa.coverage_fraction,
        finite_dem_fraction=qa.finite_dem_fraction,
        alignment_median_pixels=qa.coreg_median_target_pixels,
        alignment_p95_pixels=qa.coreg_p95_target_pixels,
        passed=qa.passed,
        failure_reasons=qa.failures(),
    ).to_dict()


def _load_npy(path: Path, *, artifact_name: str) -> np.ndarray:
    """Load the deliberately narrow, pickle-free QA array contract."""
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{artifact_name} must use the .npy contract: {path}")
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{artifact_name} must contain exactly one NumPy array: {path}")
    return values


def _problem_contract(path: Path, *, product_id: str) -> dict[str, Any]:
    """Parse the Paper1-compatible, reference-free grid/geometry identity."""
    return dict(load_problem_grid(path, product_id=product_id).contract)


def _measure_qa_inputs(
    paths: Mapping[str, Path],
    *,
    product_id: str,
    paper2_root: Path,
    minimum_coregistration_error_count: int = MIN_COREGISTRATION_ERROR_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    isis_payload = read_json(paths["isis_run_manifest"])
    if not isinstance(isis_payload, dict):
        raise ValueError("ISIS run manifest must be a JSON object")
    if isis_payload.get("edr_sha256") != file_sha256(paths["edr"]):
        raise ValueError("QA EDR differs from the ISIS run manifest")
    if isis_payload.get("reference_dem_accessed") is not False:
        raise ValueError("ISIS run manifest accessed a forbidden reference DEM")
    projected = dict(dict(isis_payload.get("artifacts", {})).get("ctx_cam2map.cub", {}))
    projected_path = guard_paper2_path(
        str(projected.get("relative_path", "")),
        paper2_root=paper2_root,
    )
    if not projected_path.is_file() or projected.get("sha256") != file_sha256(projected_path):
        raise ValueError("QA source is not bound to the ISIS ctx_cam2map.cub artifact")

    image = _load_npy(paths["image"], artifact_name="image")
    image_mask = _load_npy(paths["image_mask"], artifact_name="image_mask")
    dem_mask = _load_npy(paths["dem_mask"], artifact_name="dem_mask")
    errors = _load_npy(paths["coregistration_errors"], artifact_name="coregistration_errors")
    coarse_dem = _load_npy(paths["coarse_dem"], artifact_name="coarse_dem")
    if image_mask.dtype != np.bool_ or dem_mask.dtype != np.bool_:
        raise ValueError("image_mask and dem_mask must be boolean .npy arrays")
    if (
        image.ndim != 2
        or coarse_dem.ndim != 2
        or image.shape != coarse_dem.shape
        or image_mask.shape != coarse_dem.shape
        or dem_mask.shape != coarse_dem.shape
    ):
        raise ValueError("image, coarse_dem and QA masks must share one non-empty 2-D grid")
    if not np.issubdtype(image.dtype, np.number) or not np.issubdtype(coarse_dem.dtype, np.number):
        raise ValueError("image and coarse_dem must use numeric dtypes")
    if not np.array_equal(image_mask, np.isfinite(image)):
        raise ValueError("image_mask must equal finite support recomputed from the grid image")
    expected_dem_mask = np.isfinite(coarse_dem)
    if not np.array_equal(dem_mask, expected_dem_mask):
        raise ValueError("dem_mask must equal the finite-value mask recomputed from coarse_dem")
    if minimum_coregistration_error_count < 0:
        raise ValueError("minimum coregistration error count cannot be negative")
    if (
        errors.ndim != 1
        or errors.size < minimum_coregistration_error_count
        or np.any(errors < 0)
        or not np.all(np.isfinite(errors))
    ):
        raise ValueError(
            "coregistration_errors must be a finite nonnegative vector with at least "
            f"{minimum_coregistration_error_count} values"
        )
    problem_contract = _problem_contract(paths["problem"], product_id=product_id)
    if list(image.shape) != problem_contract["grid_shape"]:
        raise ValueError("QA raster shape differs from the problem grid contract")
    return image_mask, dem_mask, errors, problem_contract


def _verify_ephemeral_export(
    raw: Mapping[str, Any], *, paper2_root: Path
) -> Path:
    record = dict(raw)
    if set(record) != {
        "relative_path",
        "sha256",
        "byte_count",
        "removed_after_verified_read",
        "sidecars",
    }:
        raise ValueError("ephemeral export record fields differ")
    path = guard_paper2_path(
        str(record.get("relative_path", "")), paper2_root=paper2_root
    )
    if (
        path.exists()
        or not is_sha256(str(record.get("sha256", "")))
        or int(record.get("byte_count", -1)) <= 0
        or record.get("removed_after_verified_read") is not True
    ):
        raise ValueError("ephemeral export was not hash-audited and removed")
    sidecars = record.get("sidecars")
    if not isinstance(sidecars, list):
        raise ValueError("ephemeral export sidecars must be a list")
    seen: set[Path] = set()
    for raw_sidecar in sidecars:
        sidecar = dict(raw_sidecar)
        if set(sidecar) != {"relative_path", "sha256", "byte_count"}:
            raise ValueError("ephemeral export sidecar fields differ")
        sidecar_path = guard_paper2_path(
            str(sidecar.get("relative_path", "")), paper2_root=paper2_root
        )
        if (
            sidecar_path in seen
            or sidecar_path.exists()
            or sidecar_path.parent != path.parent
            or not sidecar_path.name.startswith(f"{path.name}.")
            or not is_sha256(str(sidecar.get("sha256", "")))
            or int(sidecar.get("byte_count", -1)) <= 0
        ):
            raise ValueError("ephemeral export sidecar did not close")
        seen.add(sidecar_path)
    return path


def _verify_coarse_dem_lineage(
    manifest_path: Path,
    *,
    product_id: str,
    problem: Path,
    coarse_dem: Path,
    dem_mask: Path,
    mola_source_lock: Path,
    paper2_root: Path,
) -> dict[str, Any]:
    _, payload = _read_object(manifest_path, paper2_root=paper2_root)
    if (
        payload.get("schema") != COARSE_DEM_LINEAGE_SCHEMA
        or payload.get("state") != "complete"
        or payload.get("algorithm") != COARSE_DEM_ALGORITHM
    ):
        raise ValueError("coarse DEM lineage is not a complete frozen MOLA preparation")
    if payload.get("product_id") != product_id:
        raise ValueError("coarse DEM lineage product differs")
    if payload.get("reference_dem_accessed") is not False:
        raise ValueError("coarse DEM lineage accessed a forbidden reference DEM")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("coarse DEM lineage content hash differs")
    expected = {
        "mola_source_lock": mola_source_lock,
        "problem": problem,
        "coarse_dem": coarse_dem,
        "dem_mask": dem_mask,
    }
    artifacts = dict(payload.get("artifacts", {}))
    required_artifacts = {
        *expected,
        "mola_coarse_cube_manifest",
        "mola_coarse_cube",
    }
    if set(artifacts) != required_artifacts:
        raise ValueError("coarse DEM lineage artifact set differs")
    resolved_artifacts: dict[str, Path] = {}
    for name, raw in artifacts.items():
        record = dict(raw)
        path = guard_paper2_path(str(record.get("relative_path", "")), paper2_root=paper2_root)
        if (
            not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or int(record.get("byte_count", -1)) != path.stat().st_size
        ):
            raise ValueError(f"coarse DEM lineage artifact differs: {name}")
        resolved_artifacts[name] = path
    ephemeral_exports = payload.get("ephemeral_exports")
    if not isinstance(ephemeral_exports, dict) or set(ephemeral_exports) != {
        "mola_grid_export"
    }:
        raise ValueError("coarse DEM lineage ephemeral export set differs")
    mola_grid_export = _verify_ephemeral_export(
        dict(ephemeral_exports["mola_grid_export"]), paper2_root=paper2_root
    )
    for name, path in expected.items():
        if resolved_artifacts[name] != path:
            raise ValueError(f"coarse DEM lineage supplied artifact path differs: {name}")
    source_lock = read_json(mola_source_lock)
    megt = [
        dict(raw)
        for raw in source_lock.get("products", [])
        if raw.get("product_id") == "MEGT90N000FB"
    ]
    if len(megt) != 1 or megt[0].get("purpose") != "terrain_score":
        raise ValueError("coarse DEM lineage source lock lacks one MEGT terrain product")
    expected_source = {
        "product_id": "MEGT90N000FB",
        "image_sha256": megt[0].get("image_sha256"),
        "label_sha256": megt[0].get("label_sha256"),
        "coarse_cube_sha256": file_sha256(resolved_artifacts["mola_coarse_cube"]),
        "coarse_cube_manifest_sha256": file_sha256(
            resolved_artifacts["mola_coarse_cube_manifest"]
        ),
    }
    if payload.get("mola_source") != expected_source:
        raise ValueError("coarse DEM lineage MEGT source differs")
    verified_cube = verify_mola_coarse_cube_manifest(
        resolved_artifacts["mola_coarse_cube_manifest"],
        source_lock_path=mola_source_lock,
        product_id="MEGT90N000FB",
        paper2_root=paper2_root,
    )
    if verified_cube != resolved_artifacts["mola_coarse_cube"]:
        raise ValueError("coarse DEM lineage MOLA cube differs from its source manifest")
    if payload.get("problem_contract") != _problem_contract(problem, product_id=product_id):
        raise ValueError("coarse DEM lineage problem grid contract differs")
    expected_grid = {
        "shape": payload["problem_contract"]["grid_shape"],
        "transform_hex": payload["problem_contract"]["transform_hex"],
        "crs_wkt_sha256": payload["problem_contract"]["crs_wkt_sha256"],
    }
    if payload.get("target_grid") != expected_grid:
        raise ValueError("coarse DEM lineage target-grid identity differs")
    if payload.get("resampling") != "bilinear_values_nearest_validity":
        raise ValueError("coarse DEM lineage uses a non-frozen resampling contract")
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) != 1:
        raise ValueError("coarse DEM lineage lacks production command evidence")
    for raw in commands:
        record = dict(raw)
        argv = record.get("argv")
        if not isinstance(argv, list) or not argv or int(record.get("returncode", -1)) != 0:
            raise ValueError("coarse DEM lineage has failed/invalid command evidence")
        serialized = " ".join(str(value) for value in argv).lower()
        if any(marker in serialized for marker in ("reference", "ground_truth", "target_dem")):
            raise ValueError("coarse DEM lineage command contains a forbidden truth locator")
        if Path(str(argv[0])).name != "gdal_translate":
            raise ValueError("coarse DEM lineage contains a non-frozen export command")
        if [
            str(resolved_artifacts["mola_coarse_cube"]),
            str(mola_grid_export),
        ] != [str(value) for value in argv[-2:]]:
            raise ValueError("coarse DEM export command source/destination differs")
    operations = payload.get("operations")
    if not isinstance(operations, list) or not any(
        dict(operation).get("name") == "rasterio_reproject" for operation in operations
    ):
        raise ValueError("coarse DEM lineage lacks target-grid reprojection evidence")
    return payload


def _verify_coregistration_lineage(
    manifest_path: Path,
    *,
    product_id: str,
    source: str,
    problem: Path,
    isis_run_manifest: Path,
    image_unregistered: Path,
    image_unregistered_mask: Path,
    image: Path,
    image_mask: Path,
    errors: Path,
    paper2_root: Path,
    minimum_control_count: int = MIN_COREGISTRATION_ERROR_COUNT,
) -> dict[str, Any]:
    _, payload = _read_object(manifest_path, paper2_root=paper2_root)
    if (
        payload.get("schema") != COREGISTRATION_LINEAGE_SCHEMA
        or payload.get("state") != "complete"
        or payload.get("algorithm") != COREGISTRATION_ALGORITHM
    ):
        raise ValueError("coregistration lineage is not a complete frozen preparation")
    if payload.get("product_id") != product_id or payload.get("source") != source:
        raise ValueError("coregistration lineage identity/source differs")
    if payload.get("reference_dem_accessed") is not False:
        raise ValueError("coregistration lineage accessed a forbidden reference DEM")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("coregistration lineage content hash differs")
    expected = {
        "isis_run_manifest": isis_run_manifest,
        "problem": problem,
        "image_unregistered": image_unregistered,
        "image_unregistered_mask": image_unregistered_mask,
        "image": image,
        "image_mask": image_mask,
        "coregistration_errors": errors,
    }
    artifacts = dict(payload.get("artifacts", {}))
    required_artifacts = {
        *expected,
        "projected_cube",
        "coreg_aux",
        "tiepoints",
    }
    if set(artifacts) != required_artifacts:
        raise ValueError("coregistration lineage artifact set differs")
    resolved_artifacts: dict[str, Path] = {}
    for name, raw in artifacts.items():
        record = dict(raw)
        path = guard_paper2_path(str(record.get("relative_path", "")), paper2_root=paper2_root)
        if (
            not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or int(record.get("byte_count", -1)) != path.stat().st_size
        ):
            raise ValueError(f"coregistration lineage artifact differs: {name}")
        resolved_artifacts[name] = path
    ephemeral_exports = payload.get("ephemeral_exports")
    expected_ephemeral_names = {
        "projected_grid_export",
        "coreg_aux_grid_export",
    }
    if (
        not isinstance(ephemeral_exports, dict)
        or set(ephemeral_exports) != expected_ephemeral_names
    ):
        raise ValueError("coregistration lineage ephemeral export set differs")
    resolved_ephemeral_exports = {
        name: _verify_ephemeral_export(
            dict(ephemeral_exports[name]), paper2_root=paper2_root
        )
        for name in sorted(expected_ephemeral_names)
    }
    for name, path in expected.items():
        if resolved_artifacts[name] != path:
            raise ValueError(f"coregistration lineage supplied artifact path differs: {name}")
    if payload.get("problem_contract") != _problem_contract(problem, product_id=product_id):
        raise ValueError("coregistration lineage problem contract differs")
    expected_grid = {
        "shape": payload["problem_contract"]["grid_shape"],
        "transform_hex": payload["problem_contract"]["transform_hex"],
        "crs_wkt_sha256": payload["problem_contract"]["crs_wkt_sha256"],
    }
    if payload.get("target_grid") != expected_grid:
        raise ValueError("coregistration lineage target-grid identity differs")
    if payload.get("coreg_aux_is_primary") is not False:
        raise ValueError("coreg_aux cannot be substituted for the primary CTX image")
    if not isinstance(payload.get("coreg_aux_influenced_primary_pixels"), bool):
        raise ValueError("coregistration lineage lacks a boolean geometry-use flag")
    if (
        payload.get("coreg_aux_pixel_values_copied") is not False
        or payload.get("primary_intensity_source") != "ctx_edr_isis_cam2map_only"
        or payload.get("registration_transform_family")
        != COREGISTRATION_TRANSFORM_FAMILY
        or payload.get("rotation_scale_shear_or_local_warp_applied") is not False
    ):
        raise ValueError("coregistration lineage violates the rigid CTX-only contract")
    isis_payload = read_json(resolved_artifacts["isis_run_manifest"])
    projected = dict(dict(isis_payload.get("artifacts", {})).get("ctx_cam2map.cub", {}))
    if (
        guard_paper2_path(str(projected.get("relative_path", "")), paper2_root=paper2_root)
        != resolved_artifacts["projected_cube"]
        or projected.get("sha256") != file_sha256(resolved_artifacts["projected_cube"])
        or payload.get("projected_cube_sha256")
        != file_sha256(resolved_artifacts["projected_cube"])
        or payload.get("isis_run_manifest_sha256")
        != file_sha256(resolved_artifacts["isis_run_manifest"])
    ):
        raise ValueError("coregistration lineage is not bound to the ISIS cam2map source")
    controls = read_json(resolved_artifacts["tiepoints"])
    if not isinstance(controls, dict):
        raise ValueError("coregistration control evidence must be one JSON object")
    declared_controls_hash = controls.get("content_sha256")
    unhashed_controls = dict(controls)
    unhashed_controls.pop("content_sha256", None)
    if declared_controls_hash != canonical_sha256(unhashed_controls):
        raise ValueError("coregistration control evidence content hash differs")
    if (
        controls.get("schema") != COREGISTRATION_CONTROLS_SCHEMA
        or controls.get("product_id") != product_id
        or controls.get("source") != source
        or controls.get("problem_contract") != payload.get("problem_contract")
    ):
        raise ValueError("coregistration control evidence identity differs")
    if (
        controls.get("mode") != COREGISTRATION_MODE
        or controls.get("transform_family") != COREGISTRATION_TRANSFORM_FAMILY
        or controls.get("coreg_aux_pixel_values_copied") is not False
        or controls.get("primary_intensity_source") != "ctx_edr_isis_cam2map_only"
        or controls.get("rotation_scale_shear_or_local_warp_applied") is not False
    ):
        raise ValueError("coregistration controls violate the rigid CTX-only contract")
    if controls.get("parameters") != asdict(FROZEN_COREGISTRATION_CONFIG):
        raise ValueError("coregistration parameters differ from the frozen G0 contract")
    raw_controls = controls.get("raw_controls")
    if not isinstance(raw_controls, list):
        raise ValueError("coregistration lacks raw controls for translation fitting")
    raw_control_ids = [str(dict(control).get("control_id", "")) for control in raw_controls]
    if (
        len(set(raw_control_ids)) != len(raw_control_ids)
        or any(not value for value in raw_control_ids)
    ):
        raise ValueError("coregistration raw controls are not independent/unique")
    translation = controls.get("fitted_translation_to_apply_pixels")
    (
        expected_translation,
        expected_fit_control_ids,
        expected_fit_rejections,
        expected_translation_fit,
    ) = fit_global_translation(raw_controls, config=FROZEN_COREGISTRATION_CONFIG)
    translation_applied = expected_translation is not None
    if translation_applied:
        if (
            not isinstance(translation, list)
            or len(translation) != 2
            or not np.allclose(
                np.asarray(translation, dtype=np.float64),
                np.asarray(expected_translation, dtype=np.float64),
                rtol=0.0,
                atol=1.0e-15,
            )
            or controls.get("fit_control_ids") != expected_fit_control_ids
            or controls.get("coreg_aux_influenced_primary_pixels") is not True
        ):
            raise ValueError("global translation does not reproduce from raw controls")
    elif (
        translation is not None
        or controls.get("fit_control_ids") != []
        or controls.get("coreg_aux_influenced_primary_pixels") is not False
    ):
        raise ValueError("insufficient raw controls cannot authorize translation")
    if (
        controls.get("fit_rejections") != expected_fit_rejections
        or controls.get("translation_fit") != expected_translation_fit
    ):
        raise ValueError("robust translation-fit diagnostics do not reproduce")
    if payload.get("coreg_aux_influenced_primary_pixels") is not translation_applied:
        raise ValueError("coregistration lineage geometry-use flag differs")
    residual_controls = controls.get("residual_controls")
    if minimum_control_count < 0:
        raise ValueError("minimum control count cannot be negative")
    if (
        not isinstance(residual_controls, list)
        or len(residual_controls) < minimum_control_count
    ):
        raise ValueError(
            f"coregistration requires {minimum_control_count} measured residual controls"
        )
    residual_sufficient = len(residual_controls) >= MIN_COREGISTRATION_ERROR_COUNT
    expected_rejection_reasons = (
        [] if residual_sufficient else ["coregistration_error_count<4"]
    )
    if (
        controls.get("control_count_sufficient") is not residual_sufficient
        or controls.get("rejection_reasons") != expected_rejection_reasons
    ):
        raise ValueError("coregistration residual-control terminal state differs")
    control_ids = [str(dict(control).get("control_id", "")) for control in residual_controls]
    if len(set(control_ids)) != len(control_ids) or any(not value for value in control_ids):
        raise ValueError("coregistration residual controls are not independent/unique")
    windows: list[tuple[int, int, int, int]] = []
    required_measurements = {
        "window_bounds",
        "center_row",
        "center_column",
        "shift_row_pixels",
        "shift_column_pixels",
        "valid_fraction",
        "reference_texture_std",
        "moving_texture_std",
        "peak",
        "peak_ratio",
    }
    for raw in residual_controls:
        control = dict(raw)
        if not required_measurements.issubset(control):
            raise ValueError("coregistration residual control lacks measured feature evidence")
        bounds = tuple(int(value) for value in control["window_bounds"])
        if len(bounds) != 4:
            raise ValueError("coregistration control window must have four bounds")
        row_start, row_stop, column_start, column_stop = bounds
        if row_start < 0 or column_start < 0 or row_stop <= row_start or column_stop <= column_start:
            raise ValueError("coregistration control window bounds are invalid")
        if row_stop > expected_grid["shape"][0] or column_stop > expected_grid["shape"][1]:
            raise ValueError("coregistration control window escapes the problem grid")
        windows.append(bounds)
        numeric = [
            float(control[name])
            for name in required_measurements.difference({"window_bounds"})
        ]
        if not np.all(np.isfinite(numeric)) or float(control["valid_fraction"]) <= 0:
            raise ValueError("coregistration control measurements must be finite and supported")
    for index, first in enumerate(windows):
        for second in windows[index + 1 :]:
            row_overlap = min(first[1], second[1]) - max(first[0], second[0])
            column_overlap = min(first[3], second[3]) - max(first[2], second[2])
            if row_overlap > 0 and column_overlap > 0:
                raise ValueError("coregistration residual controls do not use independent windows")
    measured_errors = np.asarray(
        [
            np.hypot(
                float(dict(control)["shift_row_pixels"]),
                float(dict(control)["shift_column_pixels"]),
            )
            for control in residual_controls
        ],
        dtype=np.float64,
    )
    stored_errors = _load_npy(errors, artifact_name="coregistration_errors").astype(
        np.float64, copy=False
    )
    if stored_errors.shape != measured_errors.shape or not np.allclose(
        stored_errors, measured_errors, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("coregistration error vector is not derived from measured controls")
    unregistered_image = _load_npy(
        resolved_artifacts["image_unregistered"],
        artifact_name="image_unregistered",
    ).astype(np.float64, copy=False)
    unregistered_mask = _load_npy(
        resolved_artifacts["image_unregistered_mask"],
        artifact_name="image_unregistered_mask",
    ).astype(bool, copy=False)
    observed_image = _load_npy(image, artifact_name="image").astype(
        np.float32, copy=False
    )
    observed_mask = _load_npy(image_mask, artifact_name="image_mask").astype(
        bool, copy=False
    )
    if translation_applied:
        shift = tuple(float(value) for value in translation)
        if shift == (0.0, 0.0):
            expected_image = unregistered_image.copy()
            expected_mask = unregistered_mask.copy()
        else:
            expected_image = ndimage.shift(
                unregistered_image,
                shift=shift,
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            expected_mask = ndimage.shift(
                unregistered_mask.astype(np.uint8),
                shift=shift,
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            ).astype(bool)
            expected_mask &= np.isfinite(expected_image)
            expected_image[~expected_mask] = np.nan
    else:
        expected_image = unregistered_image.copy()
        expected_mask = unregistered_mask.copy()
    expected_image = np.asarray(expected_image, dtype=np.float32)
    if (
        observed_image.shape != expected_image.shape
        or not np.array_equal(observed_mask, expected_mask)
        or not np.array_equal(observed_image, expected_image, equal_nan=True)
    ):
        raise ValueError("registered CTX image does not reproduce from the rigid translation")
    registration = dict(payload.get("registration", {}))
    expected_discrete = {
        "mode": COREGISTRATION_MODE,
        "transform_family": COREGISTRATION_TRANSFORM_FAMILY,
        "coreg_aux_influenced_primary_pixels": translation_applied,
        "coreg_aux_pixel_values_copied": False,
        "primary_intensity_source": "ctx_edr_isis_cam2map_only",
        "rotation_scale_shear_or_local_warp_applied": False,
        "fitted_translation_to_apply_pixels": translation,
        "fit_control_ids": expected_fit_control_ids,
        "raw_control_count": len(controls.get("raw_controls", [])),
        "residual_control_count": len(residual_controls),
        "residual_error_count": int(measured_errors.size),
    }
    if any(registration.get(key) != value for key, value in expected_discrete.items()):
        raise ValueError("coregistration summary does not reproduce from measured controls")
    if measured_errors.size:
        expected_metrics = {
            "residual_median_target_pixels": float(np.median(measured_errors)),
            "residual_p95_target_pixels": float(np.percentile(measured_errors, 95)),
        }
        if any(
            not np.isclose(
                float(registration.get(key, np.nan)), value, rtol=0.0, atol=1.0e-15
            )
            for key, value in expected_metrics.items()
        ):
            raise ValueError("coregistration residual summary differs from measured controls")
    elif any(
        registration.get(key) is not None
        for key in ("residual_median_target_pixels", "residual_p95_target_pixels")
    ):
        raise ValueError("empty coregistration controls must have null residual summaries")
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) != 2:
        raise ValueError("coregistration lineage lacks production command evidence")
    expected_exports = (
        (
            resolved_artifacts["projected_cube"],
            resolved_ephemeral_exports["projected_grid_export"],
        ),
        (
            resolved_artifacts["coreg_aux"],
            resolved_ephemeral_exports["coreg_aux_grid_export"],
        ),
    )
    for raw, (source_path, destination_path) in zip(commands, expected_exports, strict=True):
        record = dict(raw)
        argv = record.get("argv")
        if not isinstance(argv, list) or not argv or int(record.get("returncode", -1)) != 0:
            raise ValueError("coregistration lineage has failed/invalid command evidence")
        serialized = " ".join(str(value) for value in argv).lower()
        if any(marker in serialized for marker in ("reference", "ground_truth", "target_dem")):
            raise ValueError("coregistration command contains a forbidden truth locator")
        if Path(str(argv[0])).name != "gdal_translate":
            raise ValueError("coregistration lineage contains a non-frozen export command")
        if [str(source_path), str(destination_path)] != [str(value) for value in argv[-2:]]:
            raise ValueError("coregistration export command source/destination differs")
    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise ValueError("coregistration lineage lacks measured operations")
    operation_names = {str(dict(operation).get("name", "")) for operation in operations}
    if {"rasterio_reproject", COREGISTRATION_ALGORITHM}.difference(operation_names):
        raise ValueError("coregistration lineage lacks reprojection/phase-correlation evidence")
    coregistration_operations = [
        dict(operation)
        for operation in operations
        if dict(operation).get("name") == COREGISTRATION_ALGORITHM
    ]
    if len(coregistration_operations) != 1:
        raise ValueError("coregistration lineage must contain one frozen phase-correlation step")
    coregistration_operation = coregistration_operations[0]
    if (
        coregistration_operation.get("parameters") != asdict(FROZEN_COREGISTRATION_CONFIG)
        or coregistration_operation.get("mode") != COREGISTRATION_MODE
        or coregistration_operation.get("transform_family")
        != COREGISTRATION_TRANSFORM_FAMILY
        or coregistration_operation.get("pixel_adjustment_applied")
        is not translation_applied
        or coregistration_operation.get("coreg_aux_pixel_values_copied") is not False
        or coregistration_operation.get(
            "rotation_scale_shear_or_local_warp_applied"
        )
        is not False
        or coregistration_operation.get("fitted_translation_to_apply_pixels")
        != translation
        or coregistration_operation.get("measured_control_file_sha256")
        != file_sha256(resolved_artifacts["tiepoints"])
        or coregistration_operation.get("error_vector_sha256") != file_sha256(errors)
    ):
        raise ValueError("coregistration operation parameters/output hashes differ")
    return payload


def select_primary_edr(
    product_id: str, edr_entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Resolve first then second observation without changing fixed product order."""
    observations = tuple(value for value in product_id.split("__") if value)
    if len(observations) < 2:
        raise ValueError(f"CTX product lacks a stereo observation pair: {product_id}")
    statuses: list[dict[str, str]] = []
    for index, observation_id in enumerate(observations):
        entry = edr_entries.get(observation_id)
        if entry is None:
            return {
                "status": "incomplete",
                "product_id": product_id,
                "missing_observation_id": observation_id,
            }
        status = str(entry.get("status", ""))
        statuses.append({"observation_id": observation_id, "status": status})
        if status == "resolved":
            return {
                "status": "resolved",
                "product_id": product_id,
                "observation_id": observation_id,
                "selected_url": str(entry["selected_url"]),
                "reason": "first_observation" if index == 0 else "first_missing_used_second",
                "observation_statuses": statuses,
            }
        if status not in EDR_TERMINAL_STATUSES:
            return {
                "status": "incomplete",
                "product_id": product_id,
                "observation_id": observation_id,
                "observation_statuses": statuses,
            }
        if index == 0 and status != "missing":
            return {
                "status": "terminal_unavailable",
                "product_id": product_id,
                "reason": f"first_observation_{status}_does_not_authorize_second",
                "observation_statuses": statuses,
            }
    return {
        "status": "terminal_unavailable",
        "product_id": product_id,
        "observation_statuses": statuses,
    }


def build_ctx_qa_evidence(
    *,
    output: str | Path,
    paper2_root: str | Path,
    product_id: str,
    primary_observation_id: str,
    edr: str | Path,
    isis_run_manifest: str | Path,
    problem: str | Path,
    image_unregistered: str | Path,
    image_unregistered_mask: str | Path,
    image: str | Path,
    image_mask: str | Path,
    coarse_dem: str | Path,
    dem_mask: str | Path,
    coregistration_errors: str | Path,
    mola_source_lock: str | Path,
    coarse_dem_lineage_manifest: str | Path,
    coregistration_lineage_manifest: str | Path,
    coregistration_source: str,
) -> dict[str, Any]:
    """Freeze reference-free QA measurements for one already projected CTX product."""
    root = Path(paper2_root).resolve()
    _safe_identifier(product_id, name="product_id")
    _safe_identifier(primary_observation_id, name="primary_observation_id")
    if coregistration_source not in ALLOWED_COREGISTRATION_SOURCES:
        raise ValueError(
            f"coregistration_source must be one of {sorted(ALLOWED_COREGISTRATION_SOURCES)}"
        )
    paths = {
        "edr": guard_paper2_path(edr, paper2_root=root),
        "isis_run_manifest": guard_paper2_path(isis_run_manifest, paper2_root=root),
        "problem": guard_paper2_path(problem, paper2_root=root),
        "image_unregistered": guard_paper2_path(
            image_unregistered, paper2_root=root
        ),
        "image_unregistered_mask": guard_paper2_path(
            image_unregistered_mask, paper2_root=root
        ),
        "image": guard_paper2_path(image, paper2_root=root),
        "image_mask": guard_paper2_path(image_mask, paper2_root=root),
        "coarse_dem": guard_paper2_path(coarse_dem, paper2_root=root),
        "dem_mask": guard_paper2_path(dem_mask, paper2_root=root),
        "coregistration_errors": guard_paper2_path(coregistration_errors, paper2_root=root),
        "mola_source_lock": guard_paper2_path(mola_source_lock, paper2_root=root),
        "coarse_dem_lineage_manifest": guard_paper2_path(
            coarse_dem_lineage_manifest, paper2_root=root
        ),
        "coregistration_lineage_manifest": guard_paper2_path(
            coregistration_lineage_manifest, paper2_root=root
        ),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    image_mask_values, dem_mask_values, error_values, problem_contract = _measure_qa_inputs(
        paths,
        product_id=product_id,
        paper2_root=root,
    )
    coarse_lineage = _verify_coarse_dem_lineage(
        paths["coarse_dem_lineage_manifest"],
        product_id=product_id,
        problem=paths["problem"],
        coarse_dem=paths["coarse_dem"],
        dem_mask=paths["dem_mask"],
        mola_source_lock=paths["mola_source_lock"],
        paper2_root=root,
    )
    coregistration_lineage = _verify_coregistration_lineage(
        paths["coregistration_lineage_manifest"],
        product_id=product_id,
        source=coregistration_source,
        problem=paths["problem"],
        isis_run_manifest=paths["isis_run_manifest"],
        image_unregistered=paths["image_unregistered"],
        image_unregistered_mask=paths["image_unregistered_mask"],
        image=paths["image"],
        image_mask=paths["image_mask"],
        errors=paths["coregistration_errors"],
        paper2_root=root,
    )
    qa = image_qa(image_mask_values, dem_mask_values, error_values.reshape(-1))
    artifacts = {
        name: {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    payload = {
        "schema": QA_EVIDENCE_SCHEMA,
        "product_id": product_id,
        "primary_observation_id": primary_observation_id,
        "coregistration_source": coregistration_source,
        "reference_dem_accessed": False,
        "coreg_aux_influenced_primary_pixels": coregistration_lineage[
            "coreg_aux_influenced_primary_pixels"
        ],
        "coreg_aux_pixel_values_copied": False,
        "registration_transform_family": COREGISTRATION_TRANSFORM_FAMILY,
        "primary_intensity_source": "ctx_edr_isis_cam2map_only",
        "rotation_scale_shear_or_local_warp_applied": False,
        "artifacts": artifacts,
        "problem_contract": problem_contract,
        "coarse_dem_lineage_content_sha256": coarse_lineage["content_sha256"],
        "coregistration_lineage_content_sha256": coregistration_lineage["content_sha256"],
        "measurement_counts": {
            "mask_shape": list(image_mask_values.shape),
            "mask_pixel_count": int(image_mask_values.size),
            "coregistration_error_count": int(error_values.size),
        },
        "qa": _qa_dict(qa),
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    written = write_canonical_json(output, payload, paper2_root=root)
    return read_json(written)


def _insufficient_control_qa(
    image_mask: np.ndarray,
    dem_mask: np.ndarray,
    errors: np.ndarray,
) -> AssetQA:
    error_values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if error_values.size >= MIN_COREGISTRATION_ERROR_COUNT:
        raise ValueError("insufficient-control QA cannot contain four or more controls")
    coverage = float(np.mean(image_mask))
    finite_dem = float(np.mean(dem_mask))
    median = float(np.median(error_values)) if error_values.size else None
    p95 = float(np.percentile(error_values, 95)) if error_values.size else None
    failures = ["coregistration_error_count<4"]
    if coverage < 0.95:
        failures.append("coverage_fraction<0.95")
    if finite_dem < 0.99:
        failures.append("finite_dem_fraction<0.99")
    if median is not None and median > 1.0:
        failures.append("coreg_median_target_pixels>1")
    if p95 is not None and p95 > 2.0:
        failures.append("coreg_p95_target_pixels>2")
    qa = AssetQA(
        image_coverage_fraction=coverage,
        finite_dem_fraction=finite_dem,
        alignment_median_pixels=median,
        alignment_p95_pixels=p95,
        passed=False,
        failure_reasons=tuple(failures),
    )
    qa.validate()
    return qa


def _verify_raster_rejection_manifest(
    manifest_path: Path,
    *,
    product_id: str,
    resolved: Mapping[str, Path],
    image_mask: np.ndarray,
    dem_mask: np.ndarray,
    errors: np.ndarray,
    paper2_root: Path,
) -> dict[str, Any]:
    _, payload = _read_object(manifest_path, paper2_root=paper2_root)
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("raster rejection manifest content hash differs")
    if (
        payload.get("schema") != RASTER_PREPARATION_SCHEMA
        or payload.get("state") != "qa_rejected"
        or payload.get("product_id") != product_id
        or payload.get("reference_dem_accessed") is not False
        or payload.get("coregistration_source") != COREGISTRATION_SOURCE
        or payload.get("registration_transform_family")
        != COREGISTRATION_TRANSFORM_FAMILY
        or not isinstance(payload.get("coreg_aux_influenced_primary_pixels"), bool)
        or payload.get("coreg_aux_pixel_values_copied") is not False
        or payload.get("primary_intensity_source") != "ctx_edr_isis_cam2map_only"
        or payload.get("rotation_scale_shear_or_local_warp_applied") is not False
    ):
        raise ValueError("raster rejection manifest identity/state differs")
    lineage = read_json(resolved["coregistration_lineage_manifest"])
    if payload.get("coreg_aux_influenced_primary_pixels") is not lineage.get(
        "coreg_aux_influenced_primary_pixels"
    ):
        raise ValueError("raster rejection geometry-use flag differs")
    expected_outputs = {
        "image_unregistered": resolved["image_unregistered"],
        "image_unregistered_mask": resolved["image_unregistered_mask"],
        "image": resolved["image"],
        "image_mask": resolved["image_mask"],
        "coarse_dem": resolved["coarse_dem"],
        "dem_mask": resolved["dem_mask"],
        "coregistration_errors": resolved["coregistration_errors"],
        "tiepoints": resolved["coregistration_tiepoints"],
        "coarse_lineage": resolved["coarse_dem_lineage_manifest"],
        "coregistration_lineage": resolved["coregistration_lineage_manifest"],
    }
    outputs = dict(payload.get("outputs", {}))
    if set(outputs) != set(expected_outputs):
        raise ValueError("raster rejection manifest output set differs")
    for name, expected in expected_outputs.items():
        record = dict(outputs[name])
        path = guard_paper2_path(str(record.get("relative_path", "")), paper2_root=paper2_root)
        if (
            path != expected
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or int(record.get("byte_count", -1)) != path.stat().st_size
        ):
            raise ValueError(f"raster rejection output differs: {name}")
    expected_measurements = {
        "image_coverage_fraction": float(np.mean(image_mask)),
        "finite_dem_fraction": float(np.mean(dem_mask)),
        "alignment_median_pixels": (
            float(np.median(errors)) if errors.size else None
        ),
        "alignment_p95_pixels": (
            float(np.percentile(errors, 95)) if errors.size else None
        ),
        "coregistration_error_count": int(errors.size),
    }
    if payload.get("qa_measurements") != expected_measurements:
        raise ValueError("raster rejection measurements do not reproduce")
    if payload.get("qa_rejection_reasons") != ["coregistration_error_count<4"]:
        raise ValueError("raster rejection reason differs")
    return payload


def build_ctx_qa_rejection_evidence(
    *,
    output: str | Path,
    paper2_root: str | Path,
    product_id: str,
    primary_observation_id: str,
    edr: str | Path,
    isis_run_manifest: str | Path,
    problem: str | Path,
    image_unregistered: str | Path,
    image_unregistered_mask: str | Path,
    image: str | Path,
    image_mask: str | Path,
    coarse_dem: str | Path,
    dem_mask: str | Path,
    coregistration_errors: str | Path,
    coregistration_tiepoints: str | Path,
    mola_source_lock: str | Path,
    coarse_dem_lineage_manifest: str | Path,
    coregistration_lineage_manifest: str | Path,
    raster_preparation_manifest: str | Path,
    coregistration_source: str,
) -> dict[str, Any]:
    """Freeze a terminal QA rejection when fewer than four controls are measurable."""
    root = Path(paper2_root).resolve()
    _safe_identifier(product_id, name="product_id")
    _safe_identifier(primary_observation_id, name="primary_observation_id")
    if coregistration_source != COREGISTRATION_SOURCE:
        raise ValueError("QA rejection must use the frozen coregistration source")
    paths = {
        name: guard_paper2_path(value, paper2_root=root)
        for name, value in {
            "edr": edr,
            "isis_run_manifest": isis_run_manifest,
            "problem": problem,
            "image_unregistered": image_unregistered,
            "image_unregistered_mask": image_unregistered_mask,
            "image": image,
            "image_mask": image_mask,
            "coarse_dem": coarse_dem,
            "dem_mask": dem_mask,
            "coregistration_errors": coregistration_errors,
            "coregistration_tiepoints": coregistration_tiepoints,
            "mola_source_lock": mola_source_lock,
            "coarse_dem_lineage_manifest": coarse_dem_lineage_manifest,
            "coregistration_lineage_manifest": coregistration_lineage_manifest,
            "raster_preparation_manifest": raster_preparation_manifest,
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    image_mask_values, dem_mask_values, errors, problem_contract = _measure_qa_inputs(
        paths,
        product_id=product_id,
        paper2_root=root,
        minimum_coregistration_error_count=0,
    )
    if errors.size >= MIN_COREGISTRATION_ERROR_COUNT:
        raise ValueError("QA rejection has enough controls for ordinary QA")
    coarse_lineage = _verify_coarse_dem_lineage(
        paths["coarse_dem_lineage_manifest"],
        product_id=product_id,
        problem=paths["problem"],
        coarse_dem=paths["coarse_dem"],
        dem_mask=paths["dem_mask"],
        mola_source_lock=paths["mola_source_lock"],
        paper2_root=root,
    )
    coregistration_lineage = _verify_coregistration_lineage(
        paths["coregistration_lineage_manifest"],
        product_id=product_id,
        source=coregistration_source,
        problem=paths["problem"],
        isis_run_manifest=paths["isis_run_manifest"],
        image_unregistered=paths["image_unregistered"],
        image_unregistered_mask=paths["image_unregistered_mask"],
        image=paths["image"],
        image_mask=paths["image_mask"],
        errors=paths["coregistration_errors"],
        paper2_root=root,
        minimum_control_count=0,
    )
    registration = dict(coregistration_lineage.get("registration", {}))
    if (
        registration.get("control_count_sufficient") is not False
        or registration.get("rejection_reasons") != ["coregistration_error_count<4"]
        or int(registration.get("residual_control_count", -1)) != int(errors.size)
    ):
        raise ValueError("coregistration rejection lineage does not reproduce")
    _verify_raster_rejection_manifest(
        paths["raster_preparation_manifest"],
        product_id=product_id,
        resolved=paths,
        image_mask=image_mask_values,
        dem_mask=dem_mask_values,
        errors=errors,
        paper2_root=root,
    )
    qa = _insufficient_control_qa(image_mask_values, dem_mask_values, errors)
    artifacts = {
        name: {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    controls = read_json(paths["coregistration_tiepoints"])
    payload = {
        "schema": QA_REJECTION_EVIDENCE_SCHEMA,
        "state": "qa_rejected",
        "product_id": product_id,
        "primary_observation_id": primary_observation_id,
        "coregistration_source": coregistration_source,
        "failure_stage": "coregistration_control_audit",
        "reference_dem_accessed": False,
        "coreg_aux_influenced_primary_pixels": coregistration_lineage[
            "coreg_aux_influenced_primary_pixels"
        ],
        "coreg_aux_pixel_values_copied": False,
        "registration_transform_family": COREGISTRATION_TRANSFORM_FAMILY,
        "primary_intensity_source": "ctx_edr_isis_cam2map_only",
        "rotation_scale_shear_or_local_warp_applied": False,
        "artifacts": artifacts,
        "problem_contract": problem_contract,
        "coarse_dem_lineage_content_sha256": coarse_lineage["content_sha256"],
        "coregistration_lineage_content_sha256": coregistration_lineage["content_sha256"],
        "measurement_counts": {
            "mask_shape": list(image_mask_values.shape),
            "mask_pixel_count": int(image_mask_values.size),
            "coregistration_error_count": int(errors.size),
            "rejected_control_count": len(dict(controls).get("residual_rejections", [])),
        },
        "qa": qa.to_dict(),
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    written = write_canonical_json(output, payload, paper2_root=root)
    return read_json(written)


def _load_ctx_qa_evidence(
    path: Path,
    *,
    product_id: str,
    observation_id: str,
    edr_sha256: str,
    isis_run_manifest_sha256: str,
    mola_source_lock_sha256: str,
    paper2_root: Path,
) -> tuple[dict[str, Any], ImageQA, dict[str, Path]]:
    evidence_path, payload = _read_object(path, paper2_root=paper2_root)
    if payload.get("schema") != QA_EVIDENCE_SCHEMA:
        raise ValueError(f"unsupported CTX QA evidence schema: {evidence_path}")
    if payload.get("product_id") != product_id:
        raise ValueError(f"QA evidence product differs for {product_id}")
    if payload.get("primary_observation_id") != observation_id:
        raise ValueError(f"QA evidence primary observation differs for {product_id}")
    if payload.get("reference_dem_accessed") is not False:
        raise ValueError(f"QA evidence accessed a forbidden reference DEM for {product_id}")
    if (
        not isinstance(payload.get("coreg_aux_influenced_primary_pixels"), bool)
        or payload.get("coreg_aux_pixel_values_copied") is not False
        or payload.get("registration_transform_family")
        != COREGISTRATION_TRANSFORM_FAMILY
        or payload.get("primary_intensity_source") != "ctx_edr_isis_cam2map_only"
        or payload.get("rotation_scale_shear_or_local_warp_applied") is not False
    ):
        raise ValueError(f"QA evidence violates the rigid CTX-only contract for {product_id}")
    if payload.get("coregistration_source") not in ALLOWED_COREGISTRATION_SOURCES:
        raise ValueError(f"QA evidence uses a forbidden coregistration source for {product_id}")
    declared_evidence_hash = payload.get("evidence_sha256")
    unhashed = dict(payload)
    unhashed.pop("evidence_sha256", None)
    if declared_evidence_hash != canonical_sha256(unhashed):
        raise ValueError(f"QA evidence content hash differs for {product_id}")
    required = {
        "edr",
        "isis_run_manifest",
        "problem",
        "image_unregistered",
        "image_unregistered_mask",
        "image",
        "image_mask",
        "coarse_dem",
        "dem_mask",
        "coregistration_errors",
        "mola_source_lock",
        "coarse_dem_lineage_manifest",
        "coregistration_lineage_manifest",
    }
    artifacts = dict(payload.get("artifacts", {}))
    if set(artifacts) != required:
        raise ValueError(f"QA evidence artifact set differs for {product_id}")
    resolved: dict[str, Path] = {}
    for name in sorted(required):
        record = dict(artifacts[name])
        artifact_path = guard_paper2_path(
            str(record.get("relative_path", "")), paper2_root=paper2_root
        )
        if not artifact_path.is_file() or record.get("sha256") != file_sha256(artifact_path):
            raise ValueError(f"QA evidence artifact does not verify: {product_id}/{name}")
        if int(record.get("bytes", -1)) != artifact_path.stat().st_size:
            raise ValueError(f"QA evidence byte count differs: {product_id}/{name}")
        resolved[name] = artifact_path
    if artifacts["edr"]["sha256"] != edr_sha256:
        raise ValueError(f"QA evidence EDR hash differs for {product_id}")
    if artifacts["isis_run_manifest"]["sha256"] != isis_run_manifest_sha256:
        raise ValueError(f"QA evidence ISIS run hash differs for {product_id}")
    if artifacts["mola_source_lock"]["sha256"] != mola_source_lock_sha256:
        raise ValueError(f"QA evidence MOLA source lock differs for {product_id}")
    image_mask, dem_mask, errors, problem_contract = _measure_qa_inputs(
        resolved,
        product_id=product_id,
        paper2_root=paper2_root,
    )
    qa = image_qa(image_mask, dem_mask, errors.reshape(-1))
    expected_counts = {
        "mask_shape": list(image_mask.shape),
        "mask_pixel_count": int(image_mask.size),
        "coregistration_error_count": int(errors.size),
    }
    if payload.get("measurement_counts") != expected_counts:
        raise ValueError(f"QA evidence measurement counts differ for {product_id}")
    if payload.get("problem_contract") != problem_contract:
        raise ValueError(f"QA evidence problem contract differs for {product_id}")
    coarse_lineage = _verify_coarse_dem_lineage(
        resolved["coarse_dem_lineage_manifest"],
        product_id=product_id,
        problem=resolved["problem"],
        coarse_dem=resolved["coarse_dem"],
        dem_mask=resolved["dem_mask"],
        mola_source_lock=resolved["mola_source_lock"],
        paper2_root=paper2_root,
    )
    if payload.get("coarse_dem_lineage_content_sha256") != coarse_lineage.get("content_sha256"):
        raise ValueError(f"QA evidence coarse DEM lineage differs for {product_id}")
    coregistration_lineage = _verify_coregistration_lineage(
        resolved["coregistration_lineage_manifest"],
        product_id=product_id,
        source=str(payload["coregistration_source"]),
        problem=resolved["problem"],
        isis_run_manifest=resolved["isis_run_manifest"],
        image_unregistered=resolved["image_unregistered"],
        image_unregistered_mask=resolved["image_unregistered_mask"],
        image=resolved["image"],
        image_mask=resolved["image_mask"],
        errors=resolved["coregistration_errors"],
        paper2_root=paper2_root,
    )
    if payload.get("coregistration_lineage_content_sha256") != coregistration_lineage.get(
        "content_sha256"
    ):
        raise ValueError(f"QA evidence coregistration lineage differs for {product_id}")
    if payload.get("coreg_aux_influenced_primary_pixels") is not coregistration_lineage.get(
        "coreg_aux_influenced_primary_pixels"
    ):
        raise ValueError(f"QA evidence geometry-use flag differs for {product_id}")
    if payload.get("qa") != _qa_dict(qa):
        raise ValueError(f"QA evidence metrics do not reproduce for {product_id}")
    return payload, qa, resolved


def _load_ctx_qa_rejection_evidence(
    path: Path,
    *,
    product_id: str,
    observation_id: str,
    edr_sha256: str,
    isis_run_manifest_sha256: str,
    mola_source_lock_sha256: str,
    paper2_root: Path,
) -> tuple[dict[str, Any], AssetQA, dict[str, Path]]:
    evidence_path, payload = _read_object(path, paper2_root=paper2_root)
    if (
        payload.get("schema") != QA_REJECTION_EVIDENCE_SCHEMA
        or payload.get("state") != "qa_rejected"
        or payload.get("failure_stage") != "coregistration_control_audit"
    ):
        raise ValueError(f"unsupported CTX QA rejection evidence: {evidence_path}")
    if (
        payload.get("product_id") != product_id
        or payload.get("primary_observation_id") != observation_id
        or payload.get("coregistration_source") != COREGISTRATION_SOURCE
        or payload.get("reference_dem_accessed") is not False
        or not isinstance(payload.get("coreg_aux_influenced_primary_pixels"), bool)
        or payload.get("coreg_aux_pixel_values_copied") is not False
        or payload.get("registration_transform_family")
        != COREGISTRATION_TRANSFORM_FAMILY
        or payload.get("primary_intensity_source") != "ctx_edr_isis_cam2map_only"
        or payload.get("rotation_scale_shear_or_local_warp_applied") is not False
    ):
        raise ValueError(f"QA rejection evidence identity/lineage differs for {product_id}")
    declared_evidence_hash = payload.get("evidence_sha256")
    unhashed = dict(payload)
    unhashed.pop("evidence_sha256", None)
    if declared_evidence_hash != canonical_sha256(unhashed):
        raise ValueError(f"QA rejection evidence content hash differs for {product_id}")
    required = {
        "edr",
        "isis_run_manifest",
        "problem",
        "image_unregistered",
        "image_unregistered_mask",
        "image",
        "image_mask",
        "coarse_dem",
        "dem_mask",
        "coregistration_errors",
        "coregistration_tiepoints",
        "mola_source_lock",
        "coarse_dem_lineage_manifest",
        "coregistration_lineage_manifest",
        "raster_preparation_manifest",
    }
    artifacts = dict(payload.get("artifacts", {}))
    if set(artifacts) != required:
        raise ValueError(f"QA rejection artifact set differs for {product_id}")
    resolved: dict[str, Path] = {}
    for name in sorted(required):
        record = dict(artifacts[name])
        artifact = guard_paper2_path(
            str(record.get("relative_path", "")), paper2_root=paper2_root
        )
        if (
            not artifact.is_file()
            or record.get("sha256") != file_sha256(artifact)
            or int(record.get("bytes", -1)) != artifact.stat().st_size
        ):
            raise ValueError(f"QA rejection artifact does not verify: {product_id}/{name}")
        resolved[name] = artifact
    if artifacts["edr"]["sha256"] != edr_sha256:
        raise ValueError(f"QA rejection EDR hash differs for {product_id}")
    if artifacts["isis_run_manifest"]["sha256"] != isis_run_manifest_sha256:
        raise ValueError(f"QA rejection ISIS hash differs for {product_id}")
    if artifacts["mola_source_lock"]["sha256"] != mola_source_lock_sha256:
        raise ValueError(f"QA rejection MOLA hash differs for {product_id}")
    image_mask, dem_mask, errors, problem_contract = _measure_qa_inputs(
        resolved,
        product_id=product_id,
        paper2_root=paper2_root,
        minimum_coregistration_error_count=0,
    )
    if errors.size >= MIN_COREGISTRATION_ERROR_COUNT:
        raise ValueError(f"QA rejection has sufficient controls for {product_id}")
    controls = read_json(resolved["coregistration_tiepoints"])
    expected_counts = {
        "mask_shape": list(image_mask.shape),
        "mask_pixel_count": int(image_mask.size),
        "coregistration_error_count": int(errors.size),
        "rejected_control_count": len(dict(controls).get("residual_rejections", [])),
    }
    if payload.get("measurement_counts") != expected_counts:
        raise ValueError(f"QA rejection measurement counts differ for {product_id}")
    if payload.get("problem_contract") != problem_contract:
        raise ValueError(f"QA rejection problem contract differs for {product_id}")
    coarse_lineage = _verify_coarse_dem_lineage(
        resolved["coarse_dem_lineage_manifest"],
        product_id=product_id,
        problem=resolved["problem"],
        coarse_dem=resolved["coarse_dem"],
        dem_mask=resolved["dem_mask"],
        mola_source_lock=resolved["mola_source_lock"],
        paper2_root=paper2_root,
    )
    if payload.get("coarse_dem_lineage_content_sha256") != coarse_lineage.get(
        "content_sha256"
    ):
        raise ValueError(f"QA rejection coarse lineage differs for {product_id}")
    coregistration_lineage = _verify_coregistration_lineage(
        resolved["coregistration_lineage_manifest"],
        product_id=product_id,
        source=COREGISTRATION_SOURCE,
        problem=resolved["problem"],
        isis_run_manifest=resolved["isis_run_manifest"],
        image_unregistered=resolved["image_unregistered"],
        image_unregistered_mask=resolved["image_unregistered_mask"],
        image=resolved["image"],
        image_mask=resolved["image_mask"],
        errors=resolved["coregistration_errors"],
        paper2_root=paper2_root,
        minimum_control_count=0,
    )
    registration = dict(coregistration_lineage.get("registration", {}))
    if (
        registration.get("control_count_sufficient") is not False
        or registration.get("rejection_reasons") != ["coregistration_error_count<4"]
        or int(registration.get("residual_control_count", -1)) != int(errors.size)
    ):
        raise ValueError(f"QA rejection coregistration lineage differs for {product_id}")
    if payload.get("coregistration_lineage_content_sha256") != coregistration_lineage.get(
        "content_sha256"
    ):
        raise ValueError(f"QA rejection coregistration hash differs for {product_id}")
    if payload.get("coreg_aux_influenced_primary_pixels") is not coregistration_lineage.get(
        "coreg_aux_influenced_primary_pixels"
    ):
        raise ValueError(f"QA rejection geometry-use flag differs for {product_id}")
    _verify_raster_rejection_manifest(
        resolved["raster_preparation_manifest"],
        product_id=product_id,
        resolved=resolved,
        image_mask=image_mask,
        dem_mask=dem_mask,
        errors=errors,
        paper2_root=paper2_root,
    )
    qa = _insufficient_control_qa(image_mask, dem_mask, errors)
    if payload.get("qa") != qa.to_dict():
        raise ValueError(f"QA rejection metrics do not reproduce for {product_id}")
    return payload, qa, resolved


def _verified_isis_manifest(
    manifest_path: Path,
    *,
    edr: Path,
    edr_sha256: str,
    config: CTXPipelineConfig,
    paper2_root: Path,
) -> dict[str, Any]:
    _, payload = _read_object(manifest_path, paper2_root=paper2_root)
    if payload.get("schema_version") != 1:
        raise ValueError("ISIS run manifest schema differs")
    if payload.get("isis_version") != config.isis_version:
        raise ValueError("ISIS run manifest version differs from the frozen config")
    if payload.get("reported_isis_version") != config.isis_version:
        raise ValueError("ISIS runtime version differs from the frozen config")
    expected_versions = {
        "isis_build": config.isis_build,
        "ale_version": config.ale_version,
        "reported_ale_version": config.ale_version,
        "ale_build": config.ale_build,
        "csm_version": config.csm_version,
        "csm_build": config.csm_build,
        "usgscsm_version": config.usgscsm_version,
        "usgscsm_build": config.usgscsm_build,
    }
    if any(payload.get(name) != value for name, value in expected_versions.items()):
        raise ValueError("ISIS run manifest ALE/CSM package contract differs")
    if payload.get("csm_environment") != verify_csm_environment(config):
        raise ValueError("ISIS run manifest CSM environment bytes differ")
    if payload.get("edr_sha256") != edr_sha256:
        raise ValueError("ISIS run manifest is bound to a different EDR")
    if payload.get("reference_dem_accessed") is not False:
        raise ValueError("ISIS run manifest accessed a forbidden reference DEM")
    if payload.get("projection_terrain") != "MOLA_MEGR_32PPD_PLANETARY_RADIUS_DEMPREP":
        raise ValueError("ISIS run manifest does not use the frozen MOLA shape")
    if payload.get("mola_radius_prepared_sha256") != file_sha256(config.mola_radius_prepared_cube):
        raise ValueError("ISIS run manifest MOLA cube hash differs")
    verify_mola_preparation(config)
    if payload.get("mola_preparation_manifest_sha256") != file_sha256(
        mola_preparation_manifest_path(config)
    ):
        raise ValueError("ISIS run manifest MOLA preparation hash differs")
    verify_isis_base_kernel_lock(paper2_root=paper2_root)
    if payload.get("isis_base_kernel_lock_sha256") != file_sha256(
        paper2_root / ISIS_BASE_KERNEL_LOCK
    ):
        raise ValueError("ISIS run manifest base-kernel lock hash differs")
    if payload.get("isis_preferences_sha256") != file_sha256(config.isis_preferences):
        raise ValueError("ISIS run manifest preferences hash differs")
    if payload.get("map_template_sha256") != file_sha256(config.map_template):
        raise ValueError("ISIS run manifest map template hash differs")
    if (
        payload.get("spice_backend") != config.spice_backend
        or payload.get("spiceql_rest_url") != config.spiceql_rest_url
    ):
        raise ValueError("ISIS run manifest SPICE backend differs")
    if payload.get("ale_spiceql_compatibility") != verify_ale_spiceql_compatibility(
        paper2_root
    ):
        raise ValueError("ISIS run manifest ALE/SpiceQL compatibility binding differs")
    spatial_summing = payload.get("spatial_summing")
    if isinstance(spatial_summing, bool) or not isinstance(spatial_summing, int):
        raise ValueError("ISIS run manifest lacks an integer SpatialSumming")
    if spatial_summing < 1:
        raise ValueError("ISIS run manifest has invalid SpatialSumming")
    if payload.get("ctxevenodd_applied") is not (spatial_summing != 2):
        raise ValueError("ISIS run manifest ctxevenodd decision differs")
    command_records: list[tuple[str, ...]] = []
    for raw in payload.get("commands", []):
        record = dict(raw)
        argv = record.get("argv")
        if not isinstance(argv, list) or not argv:
            raise ValueError("ISIS run manifest has an invalid command record")
        if int(record.get("returncode", -1)) != 0:
            raise ValueError("ISIS run manifest contains a failed or unaudited command")
        command_records.append(tuple(str(value) for value in argv))
    expected_ctx = ctx_isis_commands(
        edr=edr,
        output_directory=manifest_path.parent,
        config=config,
        spatial_summing=spatial_summing,
    )
    expected_getkey = (
        "getkey",
        f"from={manifest_path.parent / 'ctx_raw.cub'}",
        "grp=Instrument",
        "keyword=SpatialSumming",
    )
    expected_order = [expected_ctx[0], expected_getkey, *expected_ctx[1:]]
    expected_retryable = {
        command[0]: command
        for command in expected_ctx
        if command[0] in SPICEQL_RETRYABLE_STAGES
    }
    if len(expected_retryable) != len(
        [command for command in expected_ctx if command[0] in SPICEQL_RETRYABLE_STAGES]
    ):
        raise ValueError("frozen CTX retryable stage names are not unique")
    raw_transient_failures = payload.get("transient_command_failures", [])
    if not isinstance(raw_transient_failures, list) or len(raw_transient_failures) > (
        len(expected_retryable) * (ISD_GENERATE_MAX_ATTEMPTS - 1)
    ):
        raise ValueError("ISIS run manifest has an invalid transient failure sequence")
    attempts_by_stage: dict[str, int] = {}
    closed_stages: set[str] = set()
    previous_stage: str | None = None
    for raw in raw_transient_failures:
        if not isinstance(raw, dict):
            raise ValueError("ISIS run manifest has an invalid transient failure record")
        record = dict(raw)
        argv = record.get("argv")
        stage = str(record.get("stage", ""))
        if (
            not isinstance(argv, list)
            or stage not in expected_retryable
            or tuple(str(value) for value in argv) != expected_retryable[stage]
        ):
            raise ValueError("ISIS transient failure does not match a frozen CTX stage argv")
        if previous_stage is not None and stage != previous_stage:
            closed_stages.add(previous_stage)
        if stage in closed_stages:
            raise ValueError("ISIS transient failures revisit an already completed stage")
        expected_attempt = attempts_by_stage.get(stage, 0) + 1
        attempts_by_stage[stage] = expected_attempt
        previous_stage = stage
        if (
            record.get("attempt") != expected_attempt
            or expected_attempt >= ISD_GENERATE_MAX_ATTEMPTS
            or record.get("retry_reason") != "bounded_spiceql_transport_failure"
            or not is_transient_spiceql_failure(record)
        ):
            raise ValueError("ISIS transient failure audit fields differ")
        returncode = record.get("returncode")
        if not (
            (
                isinstance(returncode, int)
                and not isinstance(returncode, bool)
                and returncode != 0
            )
            or (returncode is None and record.get("timed_out") is True)
        ):
            raise ValueError("ISIS transient failure is not an audited failed attempt")
        for field in ("stdout", "stderr"):
            if not isinstance(record.get(field), str) or len(record[field]) > 4000:
                raise ValueError("ISIS transient failure output is invalid")
    ctx_stage_names = {command[0] for command in expected_order}
    metadata_commands = {("isisversion",), ("isd_generate", "--version")}
    actual_ctx = [
        command
        for command in command_records
        if command
        and command[0] in ctx_stage_names
        and command not in metadata_commands
    ]
    if actual_ctx != expected_order:
        raise ValueError("ISIS run manifest does not contain the exact frozen CTX argv")
    allowed_stage_names = ctx_stage_names | {"isisversion", "pds2isis", "demprep"}
    extras = [command[0] for command in command_records if command[0] not in allowed_stage_names]
    if extras:
        raise ValueError(f"ISIS run manifest contains unexpected stages: {extras}")
    version_records = [command for command in command_records if command[0] == "isisversion"]
    if version_records != [("isisversion",)]:
        raise ValueError("ISIS run manifest lacks exactly one runtime version query")
    ale_version_records = [
        command for command in command_records if command == ("isd_generate", "--version")
    ]
    if ale_version_records != [("isd_generate", "--version")]:
        raise ValueError("ISIS run manifest lacks exactly one ALE runtime version query")
    if any(command[0] == "spiceinit" for command in command_records):
        raise ValueError("ISIS run manifest uses forbidden release-incompatible spiceinit")
    artifacts = dict(payload.get("artifacts", {}))
    projected = artifacts.get("ctx_cam2map.cub")
    if not isinstance(projected, dict):
        raise ValueError("ISIS run manifest lacks ctx_cam2map.cub")
    required_artifacts = {
        "ctx_raw.cub",
        "ctx_cal.cub",
        "ctx_isd.json",
        "ctx_cam2map.cub",
    }
    if spatial_summing != 2:
        required_artifacts.add("ctx_evenodd.cub")
    if set(artifacts) != required_artifacts:
        raise ValueError("ISIS run manifest lacks required CTX artifacts")
    expected_csm_cube = "ctx_evenodd.cub" if spatial_summing != 2 else "ctx_cal.cub"
    if payload.get("csm_attached_cube") != expected_csm_cube:
        raise ValueError("ISIS run manifest CSM-attached cube differs")
    for record in artifacts.values():
        path = guard_paper2_path(str(record.get("relative_path", "")), paper2_root=paper2_root)
        if not path.is_file() or record.get("sha256") != file_sha256(path):
            raise ValueError(f"ISIS artifact does not verify: {path}")
    isd_path = guard_paper2_path(
        str(artifacts["ctx_isd.json"].get("relative_path", "")), paper2_root=paper2_root
    )
    if not isinstance(read_json(isd_path), dict):
        raise ValueError("ISIS run manifest ISD is not a JSON object")
    return payload


def _product_state_path(root: Path, product_id: str) -> Path:
    return (
        root
        / "data/g0-processing/products"
        / _safe_identifier(product_id, name="product_id")
        / "state.json"
    )


def _qa_evidence_path(root: Path, product_id: str, qa_evidence_directory: Path) -> Path:
    return qa_evidence_directory / f"{_safe_identifier(product_id, name='product_id')}.json"


def _load_existing_state(
    path: Path, *, product_bindings_sha256: str, root: Path
) -> dict[str, Any]:
    if not path.exists():
        return {}
    _, state = _read_object(path, paper2_root=root)
    if state.get("schema") != PRODUCT_STATE_SCHEMA:
        raise ValueError(f"unsupported product processing state: {path}")
    if state.get("product_bindings_sha256") != product_bindings_sha256:
        raise ValueError(f"stale product processing state: {path}")
    return state


def _source_product_semantic_sha256(
    source_preparation: Mapping[str, Any], *, root: Path
) -> str:
    manifest_path = guard_paper2_path(
        str(source_preparation.get("manifest_relative_path", "")), paper2_root=root
    )
    if (
        not manifest_path.is_file()
        or source_preparation.get("manifest_sha256") != file_sha256(manifest_path)
    ):
        raise ValueError("source preparation manifest path/hash differs")
    manifest = verify_source_preparation_manifest(manifest_path, paper2_root=root)
    artifacts = {
        name: {
            "sha256": str(record["sha256"]),
            "byte_count": int(record["byte_count"]),
            "access_scope": str(record["access_scope"]),
        }
        for name, record in dict(manifest["artifacts"]).items()
        if name != "source_preparation_audit"
    }
    plans = {
        budget: {
            "sha256": str(record["sha256"]),
            "plan_semantic_sha256": str(record["plan_semantic_sha256"]),
        }
        for budget, record in dict(manifest["plans"]).items()
    }
    return canonical_sha256(
        {
            "product_id": manifest["product_id"],
            "protocol": manifest["protocol"],
            "paper1": manifest["paper1"],
            "paper1_code_lock_content_sha256": manifest[
                "paper1_code_lock_content_sha256"
            ],
            "problem_contract": manifest["problem_contract"],
            "artifacts": artifacts,
            "plans": plans,
            "coreg_aux_influenced_primary_pixels": manifest[
                "coreg_aux_influenced_primary_pixels"
            ],
        }
    )


def _processed_record(
    *,
    group_id: str,
    role: str,
    product_order: int,
    product_id: str,
    observation_id: str,
    selected_url: str,
    edr_integrity: Mapping[str, Any],
    seen_by_paper1: bool,
    qa: ImageQA | AssetQA,
    evidence_path: Path,
    evidence: Mapping[str, Any],
    resolved: Mapping[str, Path],
    edr_path: Path,
    download_receipt_path: Path,
    isis_manifest_path: Path,
    isis_payload: Mapping[str, Any],
    prepared_manifest_path: Path | None,
    source_preparation_manifest_path: Path | None,
    paper2_root: Path,
) -> dict[str, Any]:
    files = [
        _relative_file_record(edr_path, kind="ctx_edr", paper2_root=paper2_root),
        _relative_file_record(
            download_receipt_path,
            kind="ctx_edr_download_receipt",
            paper2_root=paper2_root,
        ),
        _relative_file_record(
            isis_manifest_path, kind="isis_run_manifest", paper2_root=paper2_root
        ),
        _relative_file_record(evidence_path, kind="qa_evidence", paper2_root=paper2_root),
        _relative_file_record(
            paper2_root / ISIS_BASE_KERNEL_LOCK,
            kind="isis_base_kernel_lock",
            paper2_root=paper2_root,
        ),
    ]
    for name in (
        "problem",
        "image_unregistered",
        "image_unregistered_mask",
        "image",
        "image_mask",
        "coarse_dem",
        "dem_mask",
        "coregistration_errors",
        "mola_source_lock",
        "coarse_dem_lineage_manifest",
        "coregistration_lineage_manifest",
    ):
        files.append(_relative_file_record(resolved[name], kind=name, paper2_root=paper2_root))
    if source_preparation_manifest_path is not None:
        files.append(
            _relative_file_record(
                source_preparation_manifest_path,
                kind="source_preparation_manifest",
                paper2_root=paper2_root,
            )
        )
    record: dict[str, Any] = {
        "geographic_group_id": group_id,
        "role": role,
        "product_order": product_order,
        "product_id": product_id,
        "primary_observation_id": observation_id,
        "urls": [selected_url],
        "edr_integrity": dict(edr_integrity),
        "files": files,
        "imagery_lineage": [
            "CTX EDR",
            "mroctx2isis",
            "ctxcal",
            *(["ctxevenodd"] if isis_payload.get("ctxevenodd_applied") else []),
            "ALE isd_generate --use_web_spice --only_naif_spice",
            "USGSCSM csminit shapemodel=MOLA MEGR",
            "cam2map",
        ],
        "dem_lineage": ["MOLA MEGR shape", "Paper2 coarse DEM"],
        "coreg_aux_influenced_primary_pixels": evidence[
            "coreg_aux_influenced_primary_pixels"
        ],
        "coreg_aux_pixel_values_copied": False,
        "registration_transform_family": COREGISTRATION_TRANSFORM_FAMILY,
        "primary_intensity_source": "ctx_edr_isis_cam2map_only",
        "rotation_scale_shear_or_local_warp_applied": False,
        "source_preparation_manifest_sha256": (
            None
            if source_preparation_manifest_path is None
            else file_sha256(source_preparation_manifest_path)
        ),
        "seen_by_paper1": seen_by_paper1,
        "qa": _qa_dict(qa) if isinstance(qa, ImageQA) else qa.to_dict(),
        "qa_evidence_sha256": file_sha256(evidence_path),
        "qa_evidence_content_sha256": evidence["evidence_sha256"],
    }
    if qa.passed:
        if prepared_manifest_path is None:
            raise ValueError("QA-passed product lacks a prepared manifest")
        record.update(
            {
                "prepared_manifest_sha256": file_sha256(prepared_manifest_path),
                "problem_sha256": file_sha256(resolved["problem"]),
            }
        )
        files.extend(
            [
                _relative_file_record(
                    prepared_manifest_path,
                    kind="prepared_manifest",
                    paper2_root=paper2_root,
                ),
            ]
        )
    return record


def _process_product_unlocked(
    *,
    group_id: str,
    role: str,
    product_id: str,
    product_order: int,
    edr_resolution: Mapping[str, Any],
    bindings_sha256: str,
    product_bindings_sha256: str,
    mola_source_lock_sha256: str,
    paper1_seen_products: set[str],
    root: Path,
    qa_evidence_directory: Path,
    config: CTXPipelineConfig,
    download_fn: Callable[..., Mapping[str, Any]],
    isis_fn: Callable[..., Mapping[str, Any]],
    source_preparation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state_path = _product_state_path(root, product_id)
    state = _load_existing_state(
        state_path, product_bindings_sha256=product_bindings_sha256, root=root
    )
    base = {
        "schema": PRODUCT_STATE_SCHEMA,
        "bindings_sha256": bindings_sha256,
        "product_bindings_sha256": product_bindings_sha256,
        "group_id": group_id,
        "role": role,
        "product_id": product_id,
        "product_order": product_order,
        "edr_resolution": dict(edr_resolution),
    }
    source_manifest_path: Path | None = None
    if source_preparation is not None:
        if source_preparation.get("status") != "complete":
            raise ValueError("only complete source preparation may enter CTX processing")
        source_manifest_path = guard_paper2_path(
            str(source_preparation.get("manifest_relative_path", "")),
            paper2_root=root,
        )
        if (
            not source_manifest_path.is_file()
            or source_preparation.get("manifest_sha256") != file_sha256(source_manifest_path)
        ):
            raise ValueError("source preparation manifest path/hash differs")
        base["source_preparation_manifest_path"] = source_manifest_path.relative_to(
            root
        ).as_posix()
        base["source_preparation_manifest_sha256"] = file_sha256(source_manifest_path)
    if edr_resolution.get("status") != "resolved":
        result = {**base, "status": str(edr_resolution.get("status"))}
        _write_recoverable(state_path, result, paper2_root=root)
        return result

    observation_id = str(edr_resolution["observation_id"])
    selected_url = str(edr_resolution["selected_url"])
    edr_directory = (
        root / "data/assets/ctx-edr" / _safe_identifier(observation_id, name="observation_id")
    )
    edr_path = edr_directory / f"{observation_id}.IMG"
    receipt_path = edr_directory / "download-receipt.json"
    try:
        receipt = dict(
            download_fn(
                url=selected_url,
                output=edr_path,
                receipt=receipt_path,
                paper2_root=root,
                max_attempts=4,
                validate_attached_pds3=True,
                repair_invalid_existing=True,
                resume_partial=True,
            )
        )
        observed_integrity = (
            validate_attached_pds3_fixed_length(edr_path, expected_filename=edr_path.name)
            if download_fn is download_official_asset
            else dict(receipt.get("integrity", {}))
        )
        if not observed_integrity or receipt.get("integrity") != observed_integrity:
            raise ValueError("CTX EDR receipt lacks matching attached-PDS3 integrity evidence")
    except Exception as exc:
        result = {
            **base,
            "status": "download_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
        }
        _write_recoverable(state_path, result, paper2_root=root)
        return result
    edr_sha256 = str(receipt["sha256"])

    isis_manifest_path: Path | None = None
    isis_payload: dict[str, Any] | None = None
    previous_manifest = state.get("isis_run_manifest_path")
    if previous_manifest:
        candidate = guard_paper2_path(str(previous_manifest), paper2_root=root)
        try:
            isis_payload = _verified_isis_manifest(
                candidate,
                edr=edr_path,
                edr_sha256=edr_sha256,
                config=config,
                paper2_root=root,
            )
            isis_manifest_path = candidate
        except (ValueError, FileNotFoundError):
            isis_payload = None
            isis_manifest_path = None
    isis_attempts = int(state.get("isis_attempts", 0))
    if isis_payload is None:
        attempt = isis_attempts + 1
        isis_attempts = attempt
        attempt_directory = state_path.parent / "isis" / f"{observation_id}-attempt-{attempt:03d}"
        try:
            isis_payload = dict(
                isis_fn(
                    edr=edr_path,
                    output_directory=attempt_directory,
                    config=config,
                )
            )
            isis_manifest_path = attempt_directory / "isis-run-manifest.json"
            write_canonical_json(isis_manifest_path, isis_payload, paper2_root=root)
            _verified_isis_manifest(
                isis_manifest_path,
                edr=edr_path,
                edr_sha256=edr_sha256,
                config=config,
                paper2_root=root,
            )
        except Exception as exc:
            result = {
                **base,
                "status": "isis_failed",
                "isis_attempts": attempt,
                "attempt_directory": attempt_directory.relative_to(root).as_posix(),
                "edr_relative_path": edr_path.relative_to(root).as_posix(),
                "edr_sha256": edr_sha256,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
            _write_recoverable(state_path, result, paper2_root=root)
            return result
    assert isis_manifest_path is not None and isis_payload is not None
    evidence_path = _qa_evidence_path(root, product_id, qa_evidence_directory)
    common = {
        **base,
        "isis_attempts": max(1, isis_attempts),
        "edr_relative_path": edr_path.relative_to(root).as_posix(),
        "edr_sha256": edr_sha256,
        "edr_integrity": observed_integrity,
        "download_receipt_path": receipt_path.relative_to(root).as_posix(),
        "isis_run_manifest_path": isis_manifest_path.relative_to(root).as_posix(),
        "isis_run_manifest_sha256": file_sha256(isis_manifest_path),
    }
    if not evidence_path.is_file():
        result = {
            **common,
            "status": "awaiting_qa_evidence",
            "expected_qa_evidence_path": evidence_path.relative_to(root).as_posix(),
        }
        _write_recoverable(state_path, result, paper2_root=root)
        return result
    try:
        evidence_header = read_json(evidence_path)
        if (
            isinstance(evidence_header, dict)
            and evidence_header.get("schema") == QA_REJECTION_EVIDENCE_SCHEMA
        ):
            evidence, rejected_qa, rejected_resolved = _load_ctx_qa_rejection_evidence(
                evidence_path,
                product_id=product_id,
                observation_id=observation_id,
                edr_sha256=edr_sha256,
                isis_run_manifest_sha256=file_sha256(isis_manifest_path),
                mola_source_lock_sha256=mola_source_lock_sha256,
                paper2_root=root,
            )
            record = _processed_record(
                group_id=group_id,
                role=role,
                product_order=product_order,
                product_id=product_id,
                observation_id=observation_id,
                selected_url=selected_url,
                edr_integrity=observed_integrity,
                seen_by_paper1=product_id in paper1_seen_products,
                qa=rejected_qa,
                evidence_path=evidence_path,
                evidence=evidence,
                resolved=rejected_resolved,
                edr_path=edr_path,
                download_receipt_path=receipt_path,
                isis_manifest_path=isis_manifest_path,
                isis_payload=isis_payload,
                prepared_manifest_path=None,
                source_preparation_manifest_path=source_manifest_path,
                paper2_root=root,
            )
            result = {
                **common,
                "status": "qa_failed",
                "qa": rejected_qa.to_dict(),
                "qa_evidence_path": evidence_path.relative_to(root).as_posix(),
                "qa_evidence_sha256": file_sha256(evidence_path),
                "qa_rejection_evidence_content_sha256": evidence["evidence_sha256"],
                "prepared_manifest_path": None,
            }
            _write_recoverable(state_path, result, paper2_root=root)
            record["processing_manifest_relative_path"] = state_path.relative_to(
                root
            ).as_posix()
            record["processing_manifest_sha256"] = file_sha256(state_path)
            result["processed_record"] = record
            return result
        evidence, qa, resolved = _load_ctx_qa_evidence(
            evidence_path,
            product_id=product_id,
            observation_id=observation_id,
            edr_sha256=edr_sha256,
            isis_run_manifest_sha256=file_sha256(isis_manifest_path),
            mola_source_lock_sha256=mola_source_lock_sha256,
            paper2_root=root,
        )
        prepared_manifest_path: Path | None = None
        if qa.passed:
            prepared_manifest_path = state_path.parent / "prepared-manifest.json"
            write_prepared_manifest(
                prepared_manifest_path,
                paper2_root=root,
                tile_id=product_id,
                role=role,
                primary_observation=observation_id,
                primary_reason=str(edr_resolution["reason"]),
                problem_sha256=file_sha256(resolved["problem"]),
                image=resolved["image"],
                image_mask=resolved["image_mask"],
                coarse_dem=resolved["coarse_dem"],
                qa=qa,
                lineage={"isis": isis_payload, "qa_evidence_sha256": file_sha256(evidence_path)},
            )
        record = _processed_record(
            group_id=group_id,
            role=role,
            product_order=product_order,
            product_id=product_id,
            observation_id=observation_id,
            selected_url=selected_url,
            edr_integrity=observed_integrity,
            seen_by_paper1=product_id in paper1_seen_products,
            qa=qa,
            evidence_path=evidence_path,
            evidence=evidence,
            resolved=resolved,
            edr_path=edr_path,
            download_receipt_path=receipt_path,
            isis_manifest_path=isis_manifest_path,
            isis_payload=isis_payload,
            prepared_manifest_path=prepared_manifest_path,
            source_preparation_manifest_path=source_manifest_path,
            paper2_root=root,
        )
        result = {
            **common,
            "status": "qa_passed" if qa.passed else "qa_failed",
            "qa": _qa_dict(qa),
            "qa_evidence_path": evidence_path.relative_to(root).as_posix(),
            "qa_evidence_sha256": file_sha256(evidence_path),
            "prepared_manifest_path": (
                prepared_manifest_path.relative_to(root).as_posix()
                if prepared_manifest_path is not None
                else None
            ),
        }
        _write_recoverable(state_path, result, paper2_root=root)
        record["processing_manifest_relative_path"] = state_path.relative_to(root).as_posix()
        record["processing_manifest_sha256"] = file_sha256(state_path)
        result["processed_record"] = record
        return result
    except Exception as exc:
        result = {
            **common,
            "status": "qa_evidence_invalid",
            "qa_evidence_path": evidence_path.relative_to(root).as_posix(),
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
        }
        _write_recoverable(state_path, result, paper2_root=root)
        return result


def _process_product(
    *,
    product_id: str,
    root: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Serialize one product across concurrent/restarted orchestrator processes."""
    state_path = _product_state_path(root, product_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.parent / ".processing.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        return _process_product_unlocked(
            product_id=product_id,
            root=root,
            **kwargs,
        )


def _process_group(
    *,
    assignment: Mapping[str, Any],
    edr_entries: Mapping[str, Mapping[str, Any]],
    bindings_sha256: str,
    product_binding_context_sha256: str,
    mola_source_lock_sha256: str,
    paper1_seen_products: set[str],
    root: Path,
    qa_evidence_directory: Path,
    config: CTXPipelineConfig,
    download_fn: Callable[..., Mapping[str, Any]],
    isis_fn: Callable[..., Mapping[str, Any]],
    source_preparation: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    group_id = str(assignment["group_id"])
    role = str(assignment["role"])
    products = [
        str(assignment["representative_product_id"]),
        *(str(value) for value in assignment.get("fallback_product_ids", [])),
    ]
    attempts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    selected_product_id: str | None = None
    for order, product_id in enumerate(products):
        source_record: Mapping[str, Any] | None = None
        if source_preparation is not None:
            source_record = source_preparation.get(product_id)
            if source_record is None:
                attempts.append(
                    {
                        "group_id": group_id,
                        "role": role,
                        "product_id": product_id,
                        "product_order": order,
                        "status": "awaiting_source_preparation",
                    }
                )
                break
            if source_record.get("status") == "source_rejected":
                attempts.append(
                    {
                        "group_id": group_id,
                        "role": role,
                        "product_id": product_id,
                        "product_order": order,
                        "status": "source_rejected",
                        "source_failure_relative_path": source_record.get(
                            "failure_relative_path"
                        ),
                        "source_failure_sha256": source_record.get("failure_sha256"),
                    }
                )
                continue
            if source_record.get("status") != "complete":
                raise ValueError(f"unsupported source preparation state for {product_id}")
        resolution = select_primary_edr(product_id, edr_entries)
        source_semantic_sha256 = (
            None
            if source_record is None
            else _source_product_semantic_sha256(source_record, root=root)
        )
        product_bindings_sha256 = canonical_sha256(
            {
                "context_sha256": product_binding_context_sha256,
                "group_id": group_id,
                "role": role,
                "product_id": product_id,
                "product_order": order,
                "edr_resolution": resolution,
                "source_semantic_sha256": source_semantic_sha256,
            }
        )
        result = _process_product(
            group_id=group_id,
            role=role,
            product_id=product_id,
            product_order=order,
            edr_resolution=resolution,
            bindings_sha256=bindings_sha256,
            product_bindings_sha256=product_bindings_sha256,
            mola_source_lock_sha256=mola_source_lock_sha256,
            paper1_seen_products=paper1_seen_products,
            root=root,
            qa_evidence_directory=qa_evidence_directory,
            config=config,
            download_fn=download_fn,
            isis_fn=isis_fn,
            source_preparation=source_record,
        )
        record = result.pop("processed_record", None)
        attempts.append(result)
        if record is not None:
            records.append(record)
        status = result["status"]
        if status == "qa_passed":
            selected_product_id = product_id
            break
        if status in {"qa_failed", "terminal_unavailable"}:
            continue
        break
    state = "complete" if selected_product_id is not None else "incomplete"
    result = {
        "schema": GROUP_STATE_SCHEMA,
        "bindings_sha256": bindings_sha256,
        "group_id": group_id,
        "role": role,
        "state": state,
        "selected_product_id": selected_product_id,
        "attempts": attempts,
        "processed_records": records,
    }
    group_path = (
        root
        / "data/g0-processing/groups"
        / _safe_identifier(group_id, name="group_id")
        / "state.json"
    )
    _write_recoverable(group_path, result, paper2_root=root)
    result["group_state_path"] = group_path.relative_to(root).as_posix()
    result["group_state_sha256"] = file_sha256(group_path)
    return result


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = CTXPipelineConfig(
        **{
            key: Path(value)
            if key
            not in {
                "isis_version",
                "isis_build",
                "ale_version",
                "ale_build",
                "csm_version",
                "csm_build",
                "usgscsm_version",
                "usgscsm_build",
                "spice_backend",
                "spiceql_rest_url",
                "spiceql_https_proxy",
            }
            else value
            for key, value in dict(payload["config"]).items()
        }
    )
    return _process_group(
        assignment=payload["assignment"],
        edr_entries=payload["edr_entries"],
        bindings_sha256=str(payload["bindings_sha256"]),
        product_binding_context_sha256=str(payload["product_binding_context_sha256"]),
        mola_source_lock_sha256=str(payload["mola_source_lock_sha256"]),
        paper1_seen_products=set(payload["paper1_seen_products"]),
        root=Path(payload["root"]),
        qa_evidence_directory=Path(payload["qa_evidence_directory"]),
        config=config,
        download_fn=download_official_asset,
        isis_fn=run_ctx_isis,
        source_preparation=payload.get("source_preparation"),
    )


def run_g0_ctx_processing(
    *,
    conflict_groups_path: str | Path,
    assignments_path: str | Path,
    edr_index_path: str | Path,
    mola_source_lock_path: str | Path,
    paper1_seen_lock_path: str | Path,
    processing_index_output: str | Path,
    processed_lock_output: str | Path,
    qa_evidence_directory: str | Path,
    source_preparation_index_path: str | Path | None = None,
    paper2_root: str | Path,
    config: CTXPipelineConfig,
    workers: int = MAX_PROCESS_WORKERS,
    require_formal_counts: bool = True,
    download_fn: Callable[..., Mapping[str, Any]] = download_official_asset,
    isis_fn: Callable[..., Mapping[str, Any]] = run_ctx_isis,
) -> dict[str, Any]:
    """Resume fixed-order group processing and emit processed-lock only after group QA closure."""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("CTX processing workers must be an integer between 1 and 4")
    if require_formal_counts and source_preparation_index_path is None:
        raise ValueError("formal G0 requires a verified source-preparation index")
    root = Path(paper2_root).resolve()
    if config.paper2_root.resolve() != root:
        raise ValueError("CTX pipeline config uses a different Paper2 root")
    config.validate()
    conflicts_file, conflicts = _read_object(conflict_groups_path, paper2_root=root)
    assignments_file, assignments_payload = _read_object(assignments_path, paper2_root=root)
    edr_file, edr_payload = _read_object(edr_index_path, paper2_root=root)
    mola_file, mola_payload = _read_object(mola_source_lock_path, paper2_root=root)
    seen_file, seen_payload = _read_object(paper1_seen_lock_path, paper2_root=root)
    conflict_hash = _conflict_groups_hash(conflicts)
    assignments, _ = _fixed_assignments(
        conflicts, assignments_payload, require_formal_counts=require_formal_counts
    )
    source_groups: dict[str, dict[str, Any]] | None = None
    source_index_payload: dict[str, Any] | None = None
    if source_preparation_index_path is not None:
        _, source_index_payload = _read_object(
            source_preparation_index_path,
            paper2_root=root,
        )
        source_schedule_bindings = {
            "conflict_groups_sha256": conflict_hash,
            "conflict_groups_file_sha256": file_sha256(conflicts_file),
            "assignments_file_sha256": file_sha256(assignments_file),
            "role_assignment_sha256": canonical_sha256(assignments),
            "catalog_file_sha256": str(conflicts["catalog_sha256"]),
        }
        source_groups = validate_source_preparation_index(
            source_index_payload,
            assignments=assignments,
            schedule_bindings=source_schedule_bindings,
            paper2_root=root,
        )
    edr_entries = validate_edr_index(
        edr_payload,
        catalog_sha256=str(conflicts["catalog_sha256"]),
        conflict_groups_sha256=conflict_hash,
    )
    if edr_payload.get("conflict_groups_file_sha256") != file_sha256(conflicts_file):
        raise ValueError("EDR index is not rebound to the current conflict-groups file bytes")
    if edr_payload.get("assignments_file_sha256") != file_sha256(assignments_file):
        raise ValueError(
            "EDR index is not rebound to the current/latest assignment draft file bytes"
        )
    if edr_payload.get("role_assignment_sha256") != canonical_sha256(assignments):
        raise ValueError("EDR index differs from fixed processing assignments")
    if not validate_mola_source_lock(mola_payload):
        raise ValueError("MOLA source lock is incomplete")
    mola_shape_assets = _verify_mola_shape_sources(mola_payload, config=config, paper2_root=root)
    verify_isis_base_kernel_lock(paper2_root=root)
    base_kernel_lock_sha256 = file_sha256(root / ISIS_BASE_KERNEL_LOCK)
    paper1_seen_products = set(
        validate_paper1_seen_lock(seen_payload, catalog_sha256=str(conflicts["catalog_sha256"]))
    )
    bindings = {
        "conflict_groups_sha256": conflict_hash,
        "conflict_groups_file_sha256": file_sha256(conflicts_file),
        "assignments_file_sha256": file_sha256(assignments_file),
        "role_assignment_sha256": canonical_sha256(assignments),
        "edr_index_sha256": file_sha256(edr_file),
        "mola_source_lock_sha256": file_sha256(mola_file),
        "mola_shape_assets_sha256": canonical_sha256(mola_shape_assets),
        "isis_base_kernel_lock_sha256": base_kernel_lock_sha256,
        "paper1_seen_lock_sha256": file_sha256(seen_file),
        "paper1_seen_source_sha256": str(seen_payload["source_file_sha256"]),
        "isis_config_sha256": canonical_sha256(
            {key: str(value) for key, value in asdict(config).items()}
        ),
    }
    if source_index_payload is not None:
        bindings["source_preparation_contract_sha256"] = str(
            source_index_payload["contract_sha256"]
        )
    bindings_sha256 = canonical_sha256(bindings)
    product_binding_context_sha256 = canonical_sha256(
        {
            "conflict_groups_sha256": conflict_hash,
            "mola_source_lock_sha256": file_sha256(mola_file),
            "mola_shape_assets_sha256": canonical_sha256(mola_shape_assets),
            "isis_base_kernel_lock_sha256": base_kernel_lock_sha256,
            "paper1_seen_source_sha256": str(seen_payload["source_file_sha256"]),
            "isis_config": {
                key: str(value)
                for key, value in asdict(config).items()
                if key != "spiceql_https_proxy"
            },
        }
    )
    qa_directory = guard_paper2_path(qa_evidence_directory, paper2_root=root)
    index_output = guard_paper2_path(processing_index_output, paper2_root=root)
    processed_output = guard_paper2_path(processed_lock_output, paper2_root=root)

    use_injected = download_fn is not download_official_asset or isis_fn is not run_ctx_isis
    if use_injected and workers != 1:
        raise ValueError("injected processing functions require workers=1")
    results: list[dict[str, Any]] = []
    pending_assignments = list(assignments)
    bootstrap_serial_group_count = 0
    if workers == 1:
        for assignment in pending_assignments:
            results.append(
                _process_group(
                    assignment=assignment,
                    edr_entries=edr_entries,
                    bindings_sha256=bindings_sha256,
                    product_binding_context_sha256=product_binding_context_sha256,
                    mola_source_lock_sha256=file_sha256(mola_file),
                    paper1_seen_products=paper1_seen_products,
                    root=root,
                    qa_evidence_directory=qa_directory,
                    config=config,
                    download_fn=download_fn,
                    isis_fn=isis_fn,
                    source_preparation=(
                        None
                        if source_groups is None
                        else source_groups[str(assignment["group_id"])]["attempts_by_product"]
                    ),
                )
            )
    else:
        # MOLA is a single shared derived artifact. Bootstrap it under one process
        # before starting the pool so independent CTX workers cannot race while
        # pds2isis/demprep are writing the same cubes. The serial group states are
        # normal recoverable checkpoints and are not re-submitted to the pool.
        while pending_assignments and not mola_preparation_ready(config):
            assignment = pending_assignments.pop(0)
            results.append(
                _process_group(
                    assignment=assignment,
                    edr_entries=edr_entries,
                    bindings_sha256=bindings_sha256,
                    product_binding_context_sha256=product_binding_context_sha256,
                    mola_source_lock_sha256=file_sha256(mola_file),
                    paper1_seen_products=paper1_seen_products,
                    root=root,
                    qa_evidence_directory=qa_directory,
                    config=config,
                    download_fn=download_fn,
                    isis_fn=isis_fn,
                    source_preparation=(
                        None
                        if source_groups is None
                        else source_groups[str(assignment["group_id"])]["attempts_by_product"]
                    ),
                )
            )
            bootstrap_serial_group_count += 1
        serialized_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        }
        if pending_assignments:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _worker,
                        {
                            "assignment": assignment,
                            "edr_entries": edr_entries,
                            "bindings_sha256": bindings_sha256,
                            "product_binding_context_sha256": product_binding_context_sha256,
                            "mola_source_lock_sha256": file_sha256(mola_file),
                            "paper1_seen_products": sorted(paper1_seen_products),
                            "root": str(root),
                            "qa_evidence_directory": str(qa_directory),
                            "config": serialized_config,
                            "source_preparation": (
                                None
                                if source_groups is None
                                else source_groups[str(assignment["group_id"])][
                                    "attempts_by_product"
                                ]
                            ),
                        },
                    ): str(assignment["group_id"])
                    for assignment in pending_assignments
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(
                            {
                                "schema": GROUP_STATE_SCHEMA,
                                "bindings_sha256": bindings_sha256,
                                "group_id": futures[future],
                                "state": "incomplete",
                                "selected_product_id": None,
                                "attempts": [],
                                "processed_records": [],
                                "worker_error_type": type(exc).__name__,
                                "worker_error_message": str(exc)[:500],
                            }
                        )
    results.sort(key=lambda item: str(item["group_id"]))
    complete_groups = [item for item in results if item.get("state") == "complete"]
    blockers = [
        f"group {item['group_id']} has not reached a QA-passed fixed product"
        for item in results
        if item.get("state") != "complete"
    ]
    processing_index = {
        "schema": PROCESSING_INDEX_SCHEMA,
        "state": "complete" if len(complete_groups) == len(assignments) else "incomplete",
        **bindings,
        "input_bindings_sha256": bindings_sha256,
        "worker_contract": {
            "executor": "sequential" if workers == 1 else "process",
            "requested": workers,
            "maximum": 4,
            "bootstrap_serial_group_count": bootstrap_serial_group_count,
            "shared_mola_prepared_before_pool": mola_preparation_ready(config),
        },
        "expected_group_count": len(assignments),
        "complete_group_count": len(complete_groups),
        "groups": results,
        "blockers": blockers,
    }
    _write_recoverable(index_output, processing_index, paper2_root=root)
    if blockers:
        if processed_output.exists():
            raise ValueError("incomplete processing cannot coexist with an existing processed-lock")
        return processing_index

    records = [dict(record) for group in results for record in group.get("processed_records", [])]
    products_by_id = {str(record["product_id"]): record for record in records}
    if len(products_by_id) != len(records):
        raise ValueError("processed product records are not unique")
    assignments_by_group = {str(assignment["group_id"]): assignment for assignment in assignments}
    group_selections = []
    for group in results:
        group_id = str(group["group_id"])
        assignment = assignments_by_group[group_id]
        group_selections.append(
            {
                "group_id": group_id,
                "role": str(assignment["role"]),
                "fixed_product_ids": [
                    str(assignment["representative_product_id"]),
                    *(str(value) for value in assignment.get("fallback_product_ids", [])),
                ],
                "selected_product_id": str(group["selected_product_id"]),
                "attempts": [
                    {
                        "product_id": str(attempt["product_id"]),
                        "status": str(attempt["status"]),
                    }
                    for attempt in group["attempts"]
                ],
                "group_state_relative_path": str(group["group_state_path"]),
                "group_state_sha256": str(group["group_state_sha256"]),
            }
        )
    processed_payload = {
        "schema": SCHEMAS["processed"],
        "state": "complete",
        "conflict_groups_sha256": conflict_hash,
        "edr_index_sha256": file_sha256(edr_file),
        "mola_source_lock_sha256": file_sha256(mola_file),
        "assignments_file_sha256": file_sha256(assignments_file),
        "role_assignment_sha256": canonical_sha256(assignments),
        "processing_input_bindings_sha256": bindings_sha256,
        "paper1_seen_source_sha256": str(seen_payload["source_file_sha256"]),
        "paper1_seen_product_ids": sorted(paper1_seen_products),
        "expected_group_count": len(assignments),
        "qa_passed_group_count": len(complete_groups),
        "products": [products_by_id[key] for key in sorted(products_by_id)],
        "group_selections": sorted(group_selections, key=lambda selection: selection["group_id"]),
    }
    if source_index_payload is not None:
        processed_payload["source_preparation_contract_sha256"] = str(
            source_index_payload["contract_sha256"]
        )
    validate_processed_lock(
        processed_payload,
        conflict_groups_sha256=conflict_hash,
        edr_index_sha256=file_sha256(edr_file),
        mola_source_lock_sha256=file_sha256(mola_file),
        paper2_root=root,
        require_source_preparation=require_formal_counts,
    )
    write_canonical_json(processed_output, processed_payload, paper2_root=root)
    return {**processing_index, "processed_lock_sha256": file_sha256(processed_output)}
