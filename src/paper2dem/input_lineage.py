from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import AssetLock, AssetRecord, PlanContract, ReconstructionProblem
from .e10_mola import array_sha256, read_e10_tile_manifest
from .g0_raster_preparation import RASTER_PREPARATION_SCHEMA
from .hashing import (
    canonical_sha256,
    file_sha256 as _uncached_file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)


CTX_INPUT_LINEAGE_SCHEMA = "paper2.ctx-reconstruction-input-lineage/v1"
CTX_SOURCE_EXPERIMENTS = frozenset({"E3", "E4", "E6", "E7", "E8", "E10"})
PAPER1_IMPORT_EXPERIMENTS = frozenset({"E3b", "E5", "E9"})

_EXPERIMENT_ROLE = {
    "E3": "paper2_locked_test",
    "E4": "paper2_locked_test",
    "E6": "paper2_locked_test",
    "E7": "bridge_val",
    "E8": "bridge_val",
    "E10": "paper2_locked_test",
    "E3b": "paper1_bridge_test",
    "E5": "paper1_bridge_test",
    "E9": "locked_external",
}
FileSignature = tuple[int, int, int, int, int]


def _file_signature(path: Path) -> FileSignature:
    status = path.stat()
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


@lru_cache(maxsize=16_384)
def _cached_file_sha256(path_text: str, signature: FileSignature) -> str:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("input-lineage artifact changed before hashing")
    return _uncached_file_sha256(path)


def file_sha256(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return _cached_file_sha256(str(resolved), _file_signature(resolved))


@lru_cache(maxsize=8)
def _cached_asset_lock(path_text: str, signature: FileSignature) -> AssetLock:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("asset lock changed before input-lineage load")
    result = AssetLock.read(path)
    if _file_signature(path) != signature:
        raise RuntimeError("asset lock changed during input-lineage load")
    return result


@lru_cache(maxsize=256)
def _cached_problem(path_text: str, signature: FileSignature) -> ReconstructionProblem:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("lineage problem changed before cached load")
    result = ReconstructionProblem.read(path)
    if _file_signature(path) != signature:
        raise RuntimeError("lineage problem changed during cached load")
    return result


@lru_cache(maxsize=1_024)
def _cached_plan(path_text: str, signature: FileSignature) -> PlanContract:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("lineage plan changed before cached load")
    result = PlanContract.read(path)
    if _file_signature(path) != signature:
        raise RuntimeError("lineage plan changed during cached load")
    return result


@lru_cache(maxsize=128)
def _cached_source_arrays(
    training_problem_text: str,
    training_problem_signature: FileSignature,
    image_text: str,
    image_signature: FileSignature,
    tile_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    training_problem = Path(training_problem_text)
    image_path = Path(image_text)
    if (
        _file_signature(training_problem) != training_problem_signature
        or _file_signature(image_path) != image_signature
    ):
        raise RuntimeError("CTX source arrays changed before cached load")
    with np.load(training_problem, allow_pickle=False) as archive:
        if "coarse_dem" not in archive.files:
            raise ValueError("controlled-degradation source problem lacks coarse_dem")
        z0 = np.array(archive["coarse_dem"], dtype=np.float32, order="C", copy=True)
        source_tile = str(archive["tile_id"].item())
    if source_tile != tile_id:
        raise ValueError("controlled-degradation source problem tile differs")
    image = np.array(
        np.load(image_path, allow_pickle=False),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    if (
        _file_signature(training_problem) != training_problem_signature
        or _file_signature(image_path) != image_signature
    ):
        raise RuntimeError("CTX source arrays changed during cached load")
    z0.setflags(write=False)
    image.setflags(write=False)
    return z0, image


def _content_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = guard_paper2_path(path, paper2_root=root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "relative_path": resolved.relative_to(root).as_posix(),
        "sha256": file_sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _verified_record(record: Mapping[str, Any], *, root: Path) -> Path:
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("input-lineage artifact path is invalid")
    path = guard_paper2_path(relative, paper2_root=root)
    if (
        not path.is_file()
        or file_sha256(path) != record.get("sha256")
        or path.stat().st_size != int(record.get("byte_count", -1))
    ):
        raise ValueError(f"input-lineage artifact bytes differ: {relative}")
    return path


def _asset_for_tile(asset_lock: AssetLock, tile_id: str) -> AssetRecord:
    matches = [asset for asset in asset_lock.assets if asset.product_id == tile_id]
    if len(matches) != 1:
        raise ValueError(f"formal asset lock does not identify exactly one tile: {tile_id}")
    return matches[0]


def _asset_files(asset: AssetRecord) -> dict[str, Any]:
    return {item.kind: item for item in asset.files if item.status == "present"}


def _read_prepared_manifest(path: Path, *, root: Path, tile_id: str) -> dict[str, Any]:
    payload = read_json(path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != RASTER_PREPARATION_SCHEMA
        or payload.get("state") != "complete"
        or payload.get("product_id") != tile_id
        or payload.get("reference_dem_accessed") is not False
        or payload.get("content_sha256") != _content_hash(payload)
    ):
        raise ValueError("CTX raster-preparation manifest identity/state/hash differs")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("CTX raster-preparation manifest lacks outputs")
    for name in ("image", "image_mask"):
        record = outputs.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"CTX raster-preparation manifest lacks {name}")
        _verified_record(record, root=root)
    return dict(payload)


def _paper1_dict(value: Any) -> dict[str, str]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("Paper1 source lock is malformed")
    return {str(key): str(item) for key, item in value.items()}


def build_ctx_input_lineage(
    *,
    source_preparation_manifest_path: str | Path,
    prepared_manifest_path: str | Path,
    asset_lock_path: str | Path,
    output_path: str | Path,
    paper2_root: str | Path,
    role_manifest_path: str | Path | None = None,
    effective_role: str | None = None,
) -> Path:
    """Publish a truth-free projection of source-preparation provenance.

    This builder is a data-preparation/audit operation.  It may verify the full
    source-preparation manifest.  The emitted manifest deliberately contains no
    evaluator-reference path, so training and inference consume only this
    projection and never parse the sealed reference locator.
    """

    root = Path(paper2_root).resolve()
    source_path = guard_paper2_path(source_preparation_manifest_path, paper2_root=root)
    prepared_path = guard_paper2_path(prepared_manifest_path, paper2_root=root)
    asset_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    for path in (source_path, prepared_path, asset_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Full verification is intentionally confined to this preprocessing command.
    from .source_preparation import verify_source_preparation_manifest

    source = verify_source_preparation_manifest(source_path, paper2_root=root)
    tile_id = str(source["product_id"])
    prepared = _read_prepared_manifest(prepared_path, root=root, tile_id=tile_id)
    asset_lock = _cached_asset_lock(
        str(asset_path.resolve()), _file_signature(asset_path)
    )
    asset = _asset_for_tile(asset_lock, tile_id)
    if not asset.qa.passed:
        raise ValueError("CTX input lineage requires a QA-passed formal asset")
    if source.get("role") != asset.role:
        raise ValueError("source-preparation and asset-lock roles differ")
    if _paper1_dict(source.get("paper1")) != asset_lock.paper1.to_dict():
        raise ValueError("source-preparation Paper1 lock differs from the asset lock")

    artifacts = source.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("source-preparation manifest lacks artifacts")
    training_record = artifacts.get("training_problem")
    if not isinstance(training_record, Mapping):
        raise ValueError("source-preparation manifest lacks its training problem")
    training_problem = _verified_record(training_record, root=root)
    if (
        asset.problem_sha256 != file_sha256(training_problem)
        or prepared.get("input_bindings", {}).get("problem_sha256")
        != file_sha256(training_problem)
        or asset.prepared_manifest_sha256 != file_sha256(prepared_path)
    ):
        raise ValueError("source/prepared/asset problem lineage differs")

    files = _asset_files(asset)
    for name, digest in (
        ("problem", file_sha256(training_problem)),
        ("prepared_manifest", file_sha256(prepared_path)),
        ("source_preparation_manifest", file_sha256(source_path)),
    ):
        item = files.get(name)
        if item is None or item.sha256 != digest:
            raise ValueError(f"asset lock does not bind the {name} bytes")

    image_record = dict(prepared["outputs"]["image"])
    image_mask_record = dict(prepared["outputs"]["image_mask"])
    role = asset.role if effective_role is None else str(effective_role)
    role_record: dict[str, Any] | None = None
    power_report_sha256: str | None = None
    if role_manifest_path is not None:
        role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
        role_manifest = read_json(role_path)
        if (
            not isinstance(role_manifest, Mapping)
            or role_manifest.get("schema") != "paper2.role-manifest/v1"
            or role_manifest.get("asset_lock_sha256") != file_sha256(asset_path)
            or tile_id not in role_manifest.get("roles", {}).get(role, ())
        ):
            raise ValueError("CTX input lineage role manifest differs")
        role_record = _record(role_path, root=root)
        power_report_sha256 = role_manifest.get("power_report_sha256")
    elif role != asset.role:
        raise ValueError("a promoted reserve lineage requires its final role manifest")
    if role != asset.role and (
        asset.role != "sealed_reserve"
        or role != "paper2_locked_test"
        or not is_sha256(str(power_report_sha256 or ""))
    ):
        raise ValueError("CTX input lineage contains an invalid reserve promotion")
    payload: dict[str, Any] = {
        "schema": CTX_INPUT_LINEAGE_SCHEMA,
        "status": "complete",
        "tile_id": tile_id,
        "role": role,
        "asset_role": asset.role,
        "power_report_sha256": power_report_sha256,
        "truth_paths_exported": False,
        "evaluator_reference_opened_by_training_or_inference": False,
        "asset_lock_sha256": file_sha256(asset_path),
        "source_preparation_manifest_sha256": file_sha256(source_path),
        "source_preparation_content_sha256": str(source["content_sha256"]),
        "paper1": asset_lock.paper1.to_dict(),
        "artifacts": {
            "training_problem": _record(training_problem, root=root),
            "prepared_manifest": _record(prepared_path, root=root),
            "image": image_record,
            "image_mask": image_mask_record,
        },
    }
    if role_record is not None:
        payload["artifacts"]["role_manifest"] = role_record
    payload["content_sha256"] = _content_hash(payload)
    return write_canonical_json(output_path, payload, paper2_root=root)


def read_ctx_input_lineage(
    path: str | Path,
    *,
    asset_lock_path: str | Path,
    paper2_root: str | Path,
) -> dict[str, Any]:
    """Read the truth-free lineage projection without touching a reference path."""

    root = Path(paper2_root).resolve()
    lineage_path = guard_paper2_path(path, paper2_root=root)
    asset_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    payload = read_json(lineage_path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CTX_INPUT_LINEAGE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("truth_paths_exported") is not False
        or payload.get("evaluator_reference_opened_by_training_or_inference") is not False
        or payload.get("content_sha256") != _content_hash(payload)
    ):
        raise ValueError("CTX reconstruction input lineage identity/state/hash differs")
    if not asset_path.is_file() or payload.get("asset_lock_sha256") != file_sha256(asset_path):
        raise ValueError("CTX reconstruction input lineage asset lock differs")
    asset_lock = _cached_asset_lock(
        str(asset_path.resolve()), _file_signature(asset_path)
    )
    tile_id = str(payload.get("tile_id", ""))
    asset = _asset_for_tile(asset_lock, tile_id)
    artifacts = payload.get("artifacts")
    expected_artifacts = {
        "training_problem",
        "prepared_manifest",
        "image",
        "image_mask",
    }
    if isinstance(artifacts, Mapping) and "role_manifest" in artifacts:
        expected_artifacts.add("role_manifest")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise ValueError("CTX reconstruction input lineage artifact set differs")
    verified = {
        name: _verified_record(record, root=root)
        for name, record in artifacts.items()
        if isinstance(record, Mapping)
    }
    if set(verified) != set(artifacts):
        raise ValueError("CTX reconstruction input lineage artifact record is malformed")
    role = str(payload.get("role", ""))
    asset_role = str(payload.get("asset_role", asset.role))
    if (
        asset_role != asset.role
        or not asset.qa.passed
        or _paper1_dict(payload.get("paper1")) != asset_lock.paper1.to_dict()
    ):
        raise ValueError("CTX reconstruction input lineage asset role/QA/Paper1 differs")
    role_manifest_path = verified.get("role_manifest")
    role_manifest = (
        read_json(role_manifest_path) if role_manifest_path is not None else None
    )
    if role != asset.role:
        if (
            asset.role != "sealed_reserve"
            or role != "paper2_locked_test"
            or not isinstance(role_manifest, Mapping)
            or role_manifest.get("schema") != "paper2.role-manifest/v1"
            or role_manifest.get("asset_lock_sha256") != file_sha256(asset_path)
            or tile_id not in role_manifest.get("roles", {}).get(role, ())
            or payload.get("power_report_sha256")
            != role_manifest.get("power_report_sha256")
            or not is_sha256(str(payload.get("power_report_sha256", "")))
        ):
            raise ValueError("CTX promoted reserve lineage/final role manifest differs")
    elif role_manifest is not None and (
        role_manifest.get("schema") != "paper2.role-manifest/v1"
        or role_manifest.get("asset_lock_sha256") != file_sha256(asset_path)
        or tile_id not in role_manifest.get("roles", {}).get(role, ())
    ):
        raise ValueError("CTX input lineage role manifest differs")
    prepared = _read_prepared_manifest(
        verified["prepared_manifest"], root=root, tile_id=tile_id
    )
    if (
        file_sha256(verified["training_problem"]) != asset.problem_sha256
        or file_sha256(verified["prepared_manifest"])
        != asset.prepared_manifest_sha256
        or prepared.get("input_bindings", {}).get("problem_sha256")
        != asset.problem_sha256
        or dict(prepared["outputs"]["image"]) != dict(artifacts["image"])
        or dict(prepared["outputs"]["image_mask"]) != dict(artifacts["image_mask"])
    ):
        raise ValueError("CTX reconstruction input lineage artifact parents differ")
    files = _asset_files(asset)
    source_file = files.get("source_preparation_manifest")
    if (
        source_file is None
        or source_file.sha256 != payload.get("source_preparation_manifest_sha256")
    ):
        raise ValueError("CTX lineage source-preparation manifest differs from asset lock")
    payload = dict(payload)
    payload["_verified_paths"] = verified
    return payload


def _load_source_arrays(lineage: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    verified = lineage.get("_verified_paths")
    if not isinstance(verified, Mapping):
        raise ValueError("CTX lineage has not been path-verified")
    training_problem = Path(verified["training_problem"])
    image_path = Path(verified["image"])
    return _cached_source_arrays(
        str(training_problem.resolve()),
        _file_signature(training_problem),
        str(image_path.resolve()),
        _file_signature(image_path),
        str(lineage.get("tile_id", "")),
    )


def _validate_base_problem_arrays(
    problem: ReconstructionProblem,
    *,
    source_z0: np.ndarray,
    source_image: np.ndarray,
    allow_e8_derivation: bool,
) -> None:
    condition = str(problem.metadata.get("experimental_condition", ""))
    if allow_e8_derivation and condition.startswith("degradation-scale="):
        if problem.metadata.get("base_z0_array_sha256") != array_sha256(source_z0):
            raise ValueError("E8 degradation problem does not bind its controlled base Z0")
    elif not np.array_equal(problem.z0, source_z0, equal_nan=True):
        raise ValueError("reconstruction Z0 differs from the controlled-degradation source")

    if allow_e8_derivation and condition == "image=missing":
        finite = problem.image[np.isfinite(problem.image)]
        if finite.size and not np.all(finite == finite.reshape(-1)[0]):
            raise ValueError("E8 missing-image problem retains spatial image information")
        if problem.metadata.get("base_image_array_sha256") != array_sha256(source_image):
            raise ValueError("E8 missing-image problem does not bind its base image")
    elif not np.array_equal(problem.image, source_image, equal_nan=True):
        raise ValueError("reconstruction image differs from the primary CTX prepared image")
    if np.any(~np.isfinite(problem.z0[problem.valid_mask])):
        raise ValueError("reconstruction valid support extends outside finite Z0")


def validate_ctx_reconstruction_lineage(
    *,
    experiment: str,
    problem_path: str | Path,
    plan_path: str | Path,
    prepared_manifest_path: str | Path,
    ctx_input_lineage_path: str | Path,
    asset_lock_path: str | Path,
    paper2_root: str | Path,
    e10_manifest_path: str | Path | None = None,
) -> dict[str, str]:
    if experiment not in CTX_SOURCE_EXPERIMENTS:
        raise ValueError(f"{experiment} is not a formal CTX-source experiment")
    root = Path(paper2_root).resolve()
    problem_file = guard_paper2_path(problem_path, paper2_root=root)
    plan_file = guard_paper2_path(plan_path, paper2_root=root)
    prepared_file = guard_paper2_path(prepared_manifest_path, paper2_root=root)
    lineage_file = guard_paper2_path(ctx_input_lineage_path, paper2_root=root)
    asset_file = guard_paper2_path(asset_lock_path, paper2_root=root)
    problem = _cached_problem(
        str(problem_file.resolve()), _file_signature(problem_file)
    )
    plan = _cached_plan(str(plan_file.resolve()), _file_signature(plan_file))
    lineage = read_ctx_input_lineage(
        lineage_file, asset_lock_path=asset_file, paper2_root=root
    )
    if (
        problem.tile_id != lineage.get("tile_id")
        or plan.tile_id != problem.tile_id
        or lineage.get("role") != _EXPERIMENT_ROLE[experiment]
        or problem.metadata.get("prepared_manifest_sha256") != file_sha256(prepared_file)
        or file_sha256(prepared_file)
        != file_sha256(lineage["_verified_paths"]["prepared_manifest"])
        or plan.problem_sha256
        != file_sha256(lineage["_verified_paths"]["training_problem"])
        or plan.paper1.to_dict() != _paper1_dict(lineage.get("paper1"))
    ):
        raise ValueError("formal CTX problem/plan/prepared/source lineage differs")
    source_z0, source_image = _load_source_arrays(lineage)

    e10_hash: str | None = None
    if experiment == "E10":
        if e10_manifest_path is None:
            raise ValueError("E10 requires its locked MOLA problem manifest")
        e10_file = guard_paper2_path(e10_manifest_path, paper2_root=root)
        e10 = read_e10_tile_manifest(e10_file, paper2_root=root)
        if (
            e10.get("tile_id") != problem.tile_id
            or e10.get("e10_problem_sha256") != file_sha256(problem_file)
        ):
            raise ValueError("E10 inference problem differs from its MOLA manifest")
        e3_path = guard_paper2_path(
            str(e10.get("e3_problem_relative_path", "")), paper2_root=root
        )
        e3_problem = _cached_problem(
            str(e3_path.resolve()), _file_signature(e3_path)
        )
        if (
            e10.get("e3_problem_sha256") != file_sha256(e3_path)
            or e3_problem.plan_sha256 != problem.plan_sha256
            or e3_problem.observations_sha256 != problem.observations_sha256
            or e3_problem.metadata.get("prepared_manifest_sha256")
            != file_sha256(prepared_file)
        ):
            raise ValueError("E10 manifest E3 source problem lineage differs")
        _validate_base_problem_arrays(
            e3_problem,
            source_z0=source_z0,
            source_image=source_image,
            allow_e8_derivation=False,
        )
        if not np.array_equal(problem.image, source_image, equal_nan=True):
            raise ValueError("E10 problem does not preserve the primary CTX image")
        e10_hash = file_sha256(e10_file)
    else:
        _validate_base_problem_arrays(
            problem,
            source_z0=source_z0,
            source_image=source_image,
            allow_e8_derivation=experiment == "E8",
        )

    result = {
        "ctx_input_lineage_manifest_sha256": file_sha256(lineage_file),
        "source_preparation_manifest_sha256": str(
            lineage["source_preparation_manifest_sha256"]
        ),
    }
    if e10_hash is not None:
        result["e10_mola_problem_manifest_sha256"] = e10_hash
    return result


def validate_paper1_import_lineage(
    *,
    experiment: str,
    problem_path: str | Path,
    plan_path: str | Path,
    prepared_manifest_path: str | Path,
    paper1_import_manifest_path: str | Path,
    asset_lock_path: str | Path,
    paper2_root: str | Path,
) -> dict[str, str]:
    if experiment not in PAPER1_IMPORT_EXPERIMENTS:
        raise ValueError(f"{experiment} is not a Paper1-import experiment")
    root = Path(paper2_root).resolve()
    problem_file = guard_paper2_path(problem_path, paper2_root=root)
    plan_file = guard_paper2_path(plan_path, paper2_root=root)
    prepared_file = guard_paper2_path(prepared_manifest_path, paper2_root=root)
    import_file = guard_paper2_path(paper1_import_manifest_path, paper2_root=root)
    asset_file = guard_paper2_path(asset_lock_path, paper2_root=root)
    problem = _cached_problem(
        str(problem_file.resolve()), _file_signature(problem_file)
    )
    plan = _cached_plan(str(plan_file.resolve()), _file_signature(plan_file))
    imported = read_json(import_file)
    asset_lock = _cached_asset_lock(
        str(asset_file.resolve()), _file_signature(asset_file)
    )
    if (
        not isinstance(imported, Mapping)
        or imported.get("schema") != "paper2.paper1-import/v1"
        or _paper1_dict(imported.get("paper1")) != asset_lock.paper1.to_dict()
        or plan.paper1.to_dict() != asset_lock.paper1.to_dict()
    ):
        raise ValueError("Paper1 import manifest/source lock differs")
    records = [
        record
        for record in imported.get("tiles", [])
        if isinstance(record, Mapping) and str(record.get("tile_id", "")) == problem.tile_id
    ]
    if len(records) != 1:
        raise ValueError("Paper1 import manifest does not identify exactly one tile")
    record = records[0]
    expected_role = _EXPERIMENT_ROLE[experiment]
    plan_hashes = record.get("plan_sha256")
    common_mismatch = (
        record.get("paper2_role") != expected_role
        or plan.tile_id != problem.tile_id
        or record.get("problem_sha256") != plan.problem_sha256
        or record.get("prepared_manifest_sha256") != file_sha256(prepared_file)
        or problem.metadata.get("prepared_manifest_sha256") != file_sha256(prepared_file)
        or not isinstance(plan_hashes, Mapping)
    )
    # E5 deliberately compares thirteen plan families generated from the same
    # imported candidate geometry.  The separate selector manifest proves each
    # generated plan; only the FAOPT member is byte-identical to Paper1.  E3b/E9
    # continue to require the exact imported Paper1 plan and selection source.
    imported_plan_mismatch = experiment != "E5" and (
        plan_hashes.get(str(plan.budget)) != file_sha256(plan_file)
        or record.get("selection_kmax_sha256") != plan.source_selection_sha256
    )
    if common_mismatch or imported_plan_mismatch:
        raise ValueError("Paper1 imported problem/plan/prepared lineage differs")
    return {"paper1_import_manifest_sha256": file_sha256(import_file)}
