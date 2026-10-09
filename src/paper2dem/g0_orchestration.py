from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .g0_processing import (
    COREGISTRATION_SOURCE,
    build_ctx_qa_evidence,
    build_ctx_qa_rejection_evidence,
)
from .g0_raster_preparation import (
    COARSE_DEM_LINEAGE_SCHEMA,
    COREGISTRATION_LINEAGE_SCHEMA,
    RASTER_PREPARATION_SCHEMA,
    prepare_g0_rasters,
)
from .hashing import canonical_sha256, file_sha256, guard_paper2_path, read_json
from .source_preparation import verify_source_preparation_manifest


def _source_manifests(
    payload: Mapping[str, Any], *, paper2_root: Path
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for group in payload.get("groups", []):
        for attempt in group.get("attempts", []):
            if attempt.get("status") != "complete":
                continue
            product_id = str(attempt.get("product_id", ""))
            if not product_id or product_id in records:
                raise ValueError("source-preparation index has duplicate complete products")
            path = guard_paper2_path(
                str(attempt.get("manifest_relative_path", "")), paper2_root=paper2_root
            )
            if not path.is_file() or attempt.get("manifest_sha256") != file_sha256(path):
                raise ValueError(f"source-preparation index has stale bytes: {product_id}")
            records[product_id] = verify_source_preparation_manifest(
                path, paper2_root=paper2_root
            )
    return records


def _resume_raster_preparation(
    directory: Path, *, product_id: str, paper2_root: Path
) -> dict[str, Any] | None:
    """Verify a terminal raster manifest before resuming sidecar publication."""
    manifest_path = directory / "raster-preparation-manifest.json"
    if not manifest_path.is_file():
        return None
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("raster preparation manifest must be one JSON object")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("resumed raster preparation content hash differs")
    if (
        payload.get("schema") != RASTER_PREPARATION_SCHEMA
        or payload.get("state") not in {"complete", "qa_rejected"}
        or payload.get("product_id") != product_id
        or payload.get("reference_dem_accessed") is not False
        or payload.get("coregistration_source") != COREGISTRATION_SOURCE
        or payload.get("registration_transform_family") != "translation_2d_only"
        or not isinstance(payload.get("coreg_aux_influenced_primary_pixels"), bool)
        or payload.get("coreg_aux_pixel_values_copied") is not False
        or payload.get("rotation_scale_shear_or_local_warp_applied") is not False
    ):
        raise ValueError("resumed raster preparation identity/state differs")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("resumed raster preparation lacks outputs")
    for name, raw in outputs.items():
        record = dict(raw)
        path = guard_paper2_path(
            str(record.get("relative_path", "")), paper2_root=paper2_root
        )
        if (
            not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or int(record.get("byte_count", -1)) != path.stat().st_size
        ):
            raise ValueError(f"resumed raster output differs: {name}")
    return payload


def _verified_lineage_artifact_alias(
    lineage_path: Path,
    *,
    artifact_name: str,
    current_path: Path,
    product_id: str,
    paper2_root: Path,
) -> Path:
    """Reuse an immutable lineage path only for an exact byte-identical alias."""
    current = guard_paper2_path(current_path, paper2_root=paper2_root)
    if not current.is_file() or current.stat().st_size <= 0:
        raise FileNotFoundError(current)
    if not lineage_path.is_file():
        return current
    payload = read_json(lineage_path)
    if not isinstance(payload, dict):
        raise ValueError("partial raster lineage must be one JSON object")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("partial raster lineage content hash differs")
    if (
        payload.get("schema")
        not in {COARSE_DEM_LINEAGE_SCHEMA, COREGISTRATION_LINEAGE_SCHEMA}
        or payload.get("state") != "complete"
        or payload.get("product_id") != product_id
    ):
        raise ValueError("partial raster lineage identity/state differs")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or artifact_name not in artifacts:
        raise ValueError(f"partial raster lineage lacks artifact: {artifact_name}")
    record = dict(artifacts[artifact_name])
    prior = guard_paper2_path(
        str(record.get("relative_path", "")), paper2_root=paper2_root
    )
    expected_bytes = int(record.get("byte_count", -1))
    expected_sha256 = str(record.get("sha256", ""))
    if (
        not prior.is_file()
        or expected_bytes <= 0
        or prior.stat().st_size != expected_bytes
        or file_sha256(prior) != expected_sha256
    ):
        raise ValueError(f"partial raster lineage artifact differs: {artifact_name}")
    if prior == current:
        return current
    if current.stat().st_size != expected_bytes or file_sha256(current) != expected_sha256:
        raise FileExistsError(
            f"refusing divergent lineage artifact alias: {lineage_path}/{artifact_name}"
        )
    return prior


def _resume_lineage_inputs(
    directory: Path,
    *,
    product_id: str,
    problem: Path,
    coreg_aux: Path,
    paper2_root: Path,
) -> tuple[Path, Path]:
    """Resolve exact-content aliases already frozen into partial lineages."""
    coarse_lineage = directory / "coarse-dem-lineage.json"
    coregistration_lineage = directory / "coregistration-lineage.json"
    resumed_problem = _verified_lineage_artifact_alias(
        coarse_lineage,
        artifact_name="problem",
        current_path=problem,
        product_id=product_id,
        paper2_root=paper2_root,
    )
    if coregistration_lineage.is_file():
        coregistration_problem = _verified_lineage_artifact_alias(
            coregistration_lineage,
            artifact_name="problem",
            current_path=problem,
            product_id=product_id,
            paper2_root=paper2_root,
        )
        if coarse_lineage.is_file() and coregistration_problem != resumed_problem:
            raise FileExistsError("partial raster lineages bind different problem paths")
        resumed_problem = coregistration_problem
    resumed_coreg_aux = _verified_lineage_artifact_alias(
        coregistration_lineage,
        artifact_name="coreg_aux",
        current_path=coreg_aux,
        product_id=product_id,
        paper2_root=paper2_root,
    )
    return resumed_problem, resumed_coreg_aux


def pending_qa_tasks(
    *,
    source_preparation_index: str | Path,
    g0_processing_index: str | Path,
    mola_source_lock: str | Path,
    mola_coarse_cube: str | Path,
    mola_coarse_cube_manifest: str | Path,
    isis_prefix: str | Path,
    paper2_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Resolve only measured-QA work whose source and ISIS states are both frozen."""
    root = Path(paper2_root).resolve()
    source_path = guard_paper2_path(source_preparation_index, paper2_root=root)
    processing_path = guard_paper2_path(g0_processing_index, paper2_root=root)
    source_payload = read_json(source_path)
    processing_payload = read_json(processing_path)
    source_by_product = _source_manifests(source_payload, paper2_root=root)
    shared = {
        "mola_source_lock": guard_paper2_path(mola_source_lock, paper2_root=root),
        "mola_coarse_cube": guard_paper2_path(mola_coarse_cube, paper2_root=root),
        "mola_coarse_cube_manifest": guard_paper2_path(
            mola_coarse_cube_manifest, paper2_root=root
        ),
        "isis_prefix": guard_paper2_path(isis_prefix, paper2_root=root),
    }
    tasks: list[dict[str, Any]] = []
    for group in processing_payload.get("groups", []):
        for attempt in group.get("attempts", []):
            if attempt.get("status") != "awaiting_qa_evidence":
                continue
            product_id = str(attempt.get("product_id", ""))
            source = source_by_product.get(product_id)
            if source is None:
                raise ValueError(f"G0 reached ISIS without a source manifest: {product_id}")
            if source.get("coreg_aux_influenced_primary_pixels") is not False:
                raise ValueError(f"source manifest permits coreg_aux pixel influence: {product_id}")
            artifacts = dict(source["artifacts"])
            product_directory = (
                root / "data/g0-processing/products" / product_id
            )
            raster_directory = product_directory / "raster-preparation"
            evidence = guard_paper2_path(
                str(attempt["expected_qa_evidence_path"]), paper2_root=root
            )
            tasks.append(
                {
                    "product_id": product_id,
                    "primary_observation_id": str(
                        dict(attempt["edr_resolution"])["observation_id"]
                    ),
                    "edr": guard_paper2_path(
                        str(attempt["edr_relative_path"]), paper2_root=root
                    ),
                    "isis_run_manifest": guard_paper2_path(
                        str(attempt["isis_run_manifest_path"]), paper2_root=root
                    ),
                    "problem": guard_paper2_path(
                        str(artifacts["training_problem"]["relative_path"]),
                        paper2_root=root,
                    ),
                    "coreg_aux": guard_paper2_path(
                        str(artifacts["coreg_aux"]["relative_path"]), paper2_root=root
                    ),
                    "raster_directory": raster_directory,
                    "qa_evidence": evidence,
                    **shared,
                }
            )
    return tuple(sorted(tasks, key=lambda value: value["product_id"]))


def advance_pending_qa_sidecars(
    *,
    source_preparation_index: str | Path,
    g0_processing_index: str | Path,
    mola_source_lock: str | Path,
    mola_coarse_cube: str | Path,
    mola_coarse_cube_manifest: str | Path,
    isis_prefix: str | Path,
    paper2_root: str | Path,
) -> dict[str, Any]:
    """Prepare real target-grid rasters and QA evidence; it never reads reference.npz."""
    root = Path(paper2_root).resolve()
    tasks = pending_qa_tasks(
        source_preparation_index=source_preparation_index,
        g0_processing_index=g0_processing_index,
        mola_source_lock=mola_source_lock,
        mola_coarse_cube=mola_coarse_cube,
        mola_coarse_cube_manifest=mola_coarse_cube_manifest,
        isis_prefix=isis_prefix,
        paper2_root=root,
    )
    completed: list[dict[str, Any]] = []
    for task in tasks:
        evidence = Path(task["qa_evidence"])
        if evidence.is_file():
            completed.append(
                {
                    "product_id": task["product_id"],
                    "status": "already_complete",
                    "qa_evidence_relative_path": evidence.relative_to(root).as_posix(),
                    "qa_evidence_sha256": file_sha256(evidence),
                }
            )
            continue
        raster = Path(task["raster_directory"])
        resume_problem, resume_coreg_aux = _resume_lineage_inputs(
            raster,
            product_id=str(task["product_id"]),
            problem=Path(task["problem"]),
            coreg_aux=Path(task["coreg_aux"]),
            paper2_root=root,
        )
        preparation = _resume_raster_preparation(
            raster,
            product_id=str(task["product_id"]),
            paper2_root=root,
        )
        if preparation is None:
            preparation = prepare_g0_rasters(
                paper2_root=root,
                product_id=task["product_id"],
                problem=resume_problem,
                isis_run_manifest=task["isis_run_manifest"],
                mola_source_lock=task["mola_source_lock"],
                mola_coarse_cube=task["mola_coarse_cube"],
                mola_coarse_cube_manifest=task["mola_coarse_cube_manifest"],
                coreg_aux=resume_coreg_aux,
                output_directory=task["raster_directory"],
                isis_prefix=task["isis_prefix"],
            )
        common = {
            "output": evidence,
            "paper2_root": root,
            "product_id": task["product_id"],
            "primary_observation_id": task["primary_observation_id"],
            "edr": task["edr"],
            "isis_run_manifest": task["isis_run_manifest"],
            "problem": resume_problem,
            "image_unregistered": raster / "image-unregistered.npy",
            "image_unregistered_mask": raster / "image-unregistered-mask.npy",
            "image": raster / "image.npy",
            "image_mask": raster / "image-mask.npy",
            "coarse_dem": raster / "coarse-dem.npy",
            "dem_mask": raster / "dem-mask.npy",
            "coregistration_errors": raster / "coregistration-errors.npy",
            "mola_source_lock": task["mola_source_lock"],
            "coarse_dem_lineage_manifest": raster / "coarse-dem-lineage.json",
            "coregistration_lineage_manifest": raster / "coregistration-lineage.json",
            "coregistration_source": COREGISTRATION_SOURCE,
        }
        if preparation.get("state") == "qa_rejected":
            build_ctx_qa_rejection_evidence(
                **common,
                coregistration_tiepoints=raster / "coregistration-tiepoints.json",
                raster_preparation_manifest=raster / "raster-preparation-manifest.json",
            )
        elif preparation.get("state") == "complete":
            build_ctx_qa_evidence(**common)
        else:
            raise ValueError("raster preparation returned an unsupported terminal state")
        completed.append(
            {
                "product_id": task["product_id"],
                "status": "generated",
                "qa_evidence_relative_path": evidence.relative_to(root).as_posix(),
                "qa_evidence_sha256": file_sha256(evidence),
            }
        )
    return {
        "schema": "paper2.g0.qa-sidecar-advance/v1",
        "reference_dem_accessed": False,
        "primary_intensity_source": "ctx_edr_isis_cam2map_only",
        "registration_transform_family": "translation_2d_only",
        "coreg_aux_pixel_values_copied": False,
        "pending_task_count": len(tasks),
        "generated_count": sum(item["status"] == "generated" for item in completed),
        "completed": completed,
    }
