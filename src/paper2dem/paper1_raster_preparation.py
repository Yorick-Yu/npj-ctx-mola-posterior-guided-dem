"""Prepare truth-free imagery and independent MOLA Z0 for fixed Paper1 tiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import os
import time

import numpy as np

from .g0_raster_preparation import (
    _default_command_runner,
    _export_geotiff,
    _rasterio_reader,
    _rasterio_reprojector,
    _write_array,
    ensure_mola_coarse_cube,
    load_problem_grid,
    prepare_g0_rasters,
)
from .hashing import canonical_sha256, file_sha256, guard_paper2_path, read_json, write_canonical_json
from .official_assets import ODEDiscoveryError, discover_ctx_edr, download_official_asset
from .paper1_inference_preparation import PAPER1_FORMAL_ROLES, PAPER1_RASTER_INDEX_SCHEMA, _copy_verified
from .preparation import CTXPipelineConfig, run_ctx_isis


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    value = guard_paper2_path(path, paper2_root=root)
    if not value.is_file() or value.stat().st_size <= 0:
        raise FileNotFoundError(value)
    return {
        "relative_path": value.relative_to(root).as_posix(),
        "sha256": file_sha256(value),
        "byte_count": value.stat().st_size,
    }


def _replace_checkpoint(path: Path, payload: Mapping[str, Any], *, root: Path) -> Path:
    """Atomically replace only the recoverable raster-index checkpoint."""

    import json
    import tempfile

    output = guard_paper2_path(path, paper2_root=root)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if output.is_file() and output.read_bytes() == encoded:
        return output
    if output.exists() and not output.is_file():
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".checkpoint", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _archive_invalid_lineage(path: Path, *, root: Path) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise FileExistsError(path)
    digest = file_sha256(path)
    archive = path.parent / "invalid-lineages" / f"{digest}.{time.time_ns()}.json"
    guard_paper2_path(archive, paper2_root=root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    path.replace(archive)


def _valid_finished_lineage(path: Path, *, root: Path, tile_id: str, role: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    unhashed = dict(payload) if isinstance(payload, Mapping) else {}
    unhashed.pop("content_sha256", None)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "paper2.paper1-raster-lineage/v1"
        or payload.get("status") != "complete"
        or payload.get("tile_id") != tile_id
        or payload.get("role") != role
        or payload.get("target_high_resolution_dem_accessed") is not False
        or payload.get("content_sha256") != canonical_sha256(unhashed)
    ):
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    required = {"image", "image_mask", "coarse_dem", "dem_mask", "lineage_manifest"}
    if set(artifacts) != required:
        return None
    for name, record in artifacts.items():
        if name == "lineage_manifest":
            continue
        if not isinstance(record, Mapping):
            return None
        relative = Path(str(record.get("relative_path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            return None
        artifact = root / relative
        if (
            not artifact.is_file()
            or file_sha256(artifact) != record.get("sha256")
            or artifact.stat().st_size != int(record.get("byte_count", -1))
        ):
            return None
    return dict(payload)


def _publish_lineage(path: Path, payload: dict[str, Any], *, root: Path) -> Path:
    # The lineage must include a self-record in the raster index, but cannot hash
    # itself recursively.  Its own artifact set therefore uses a stable locator;
    # the outer index supplies the immutable hash/byte count.
    payload["artifacts"]["lineage_manifest"] = {
        "relative_path": path.relative_to(root).as_posix(),
        "self_recorded_by_outer_index": True,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return write_canonical_json(path, payload, paper2_root=root)


def _ctx_observation_and_discovery(
    tile_id: str, *, tile_root: Path, root: Path
) -> tuple[str, str, dict[str, Any]]:
    observations = tuple(value for value in tile_id.split("__") if value)
    if len(observations) < 2:
        raise ValueError("Paper1 CTX tile does not contain an observation pair")
    for index, observation in enumerate(observations[:2]):
        discovery_root = tile_root / "discovery" / observation
        try:
            discovery = discover_ctx_edr(
                observation,
                raw_cache=discovery_root / "ode-response.raw.json",
                output=discovery_root / "discovery.json",
                paper2_root=root,
                max_attempts=16,
                backoff_initial_seconds=1.0,
                backoff_max_seconds=30.0,
            )
            return (
                observation,
                "first_observation" if index == 0 else "first_missing_used_second",
                discovery,
            )
        except ODEDiscoveryError as exc:
            if index == 0 and exc.status == "missing":
                continue
            raise
    raise FileNotFoundError(f"neither Paper1 CTX observation exists for {tile_id}")


def _prepare_ctx(
    *,
    tile_id: str,
    role: str,
    source: Path,
    imported: Mapping[str, Any],
    tile_root: Path,
    root: Path,
    config: CTXPipelineConfig,
    mola_source_lock: Path,
    mola_coarse_cube: Path,
    mola_coarse_manifest: Path,
) -> dict[str, Any]:
    prepared_source = source / "prepared-manifest.json"
    if file_sha256(prepared_source) != imported.get("prepared_manifest_sha256"):
        raise ValueError("Paper1 CTX prepared manifest differs from import lock")
    prepared = read_json(prepared_source)
    ortho_url = str(dict(prepared.get("assets", {})).get("orthoimage", ""))
    if not ortho_url.startswith("https://"):
        raise ValueError("Paper1 CTX prepared manifest lacks the official ORTHO URL")
    problem = _copy_verified(
        source / "problem-v2.npz",
        tile_root / "grid/problem-v2.npz",
        expected_sha256=str(imported["problem_sha256"]),
    )
    observation, reason, discovery = _ctx_observation_and_discovery(
        tile_id, tile_root=tile_root, root=root
    )
    edr = tile_root / "raw" / f"{observation}.IMG"
    edr_receipt = tile_root / "raw" / f"{observation}.IMG.receipt.json"
    download_official_asset(
        url=str(discovery["url"]),
        output=edr,
        receipt=edr_receipt,
        paper2_root=root,
        max_attempts=32,
        validate_attached_pds3=True,
        repair_invalid_existing=True,
        resume_partial=True,
        backoff_initial_seconds=1.0,
        backoff_max_seconds=30.0,
    )
    coreg_aux = tile_root / "raw/coreg-aux-ortho.tif"
    coreg_receipt = tile_root / "raw/coreg-aux-ortho.receipt.json"
    download_official_asset(
        url=ortho_url,
        output=coreg_aux,
        receipt=coreg_receipt,
        paper2_root=root,
        max_attempts=32,
        repair_invalid_existing=True,
        resume_partial=True,
        backoff_initial_seconds=1.0,
        backoff_max_seconds=30.0,
    )
    attempt = tile_root / "attempts" / f"ctx-{time.time_ns()}-{os.getpid()}"
    isis_directory = attempt / "isis"
    isis = run_ctx_isis(edr=edr, output_directory=isis_directory, config=config)
    isis_manifest = write_canonical_json(
        attempt / "isis-run-manifest.json", isis, paper2_root=root
    )
    raster = prepare_g0_rasters(
        paper2_root=root,
        product_id=tile_id,
        problem=problem,
        isis_run_manifest=isis_manifest,
        mola_source_lock=mola_source_lock,
        mola_coarse_cube=mola_coarse_cube,
        mola_coarse_cube_manifest=mola_coarse_manifest,
        coreg_aux=coreg_aux,
        output_directory=attempt / "rasters",
        isis_prefix=config.isis_prefix,
    )
    qa = dict(raster["qa_measurements"])
    passed = (
        float(qa["image_coverage_fraction"]) >= 0.95
        and float(qa["finite_dem_fraction"]) >= 0.99
        and float(qa["alignment_median_pixels"]) <= 1.0
        and float(qa["alignment_p95_pixels"]) <= 2.0
    )
    if not passed:
        raise ValueError(f"fixed Paper1 CTX bridge tile failed locked image QA: {tile_id}: {qa}")
    outputs = dict(raster["outputs"])
    lineage_path = tile_root / "paper1-raster-lineage.json"
    artifacts = {
        name: dict(outputs[source_name])
        for name, source_name in {
            "image": "image",
            "image_mask": "image_mask",
            "coarse_dem": "coarse_dem",
            "dem_mask": "dem_mask",
        }.items()
    }
    payload: dict[str, Any] = {
        "schema": "paper2.paper1-raster-lineage/v1",
        "status": "complete",
        "tile_id": tile_id,
        "role": role,
        "mission": "CTX",
        "target_high_resolution_dem_accessed": False,
        "legacy_problem_used_for_grid_geometry_only": True,
        "primary_image_source": "official_raw_CTX_EDR_ISIS10_MOLA_projection",
        "primary_observation": observation,
        "primary_observation_reason": reason,
        "coreg_aux_role": "audit_only_no_pixel_adjustment",
        "mola_z0_independent_of_target_dtm": True,
        "qa": {**qa, "passed": True},
        "parents": {
            "paper1_problem_grid": _record(problem, root=root),
            "edr_discovery": _record(
                tile_root / "discovery" / observation / "discovery.json", root=root
            ),
            "edr": _record(edr, root=root),
            "edr_receipt": _record(edr_receipt, root=root),
            "coreg_aux": _record(coreg_aux, root=root),
            "coreg_aux_receipt": _record(coreg_receipt, root=root),
            "isis_run_manifest": _record(isis_manifest, root=root),
            "raster_preparation_manifest": _record(
                attempt / "rasters/raster-preparation-manifest.json", root=root
            ),
        },
        "artifacts": artifacts,
    }
    _publish_lineage(lineage_path, payload, root=root)
    return read_json(lineage_path)


def _prepare_hirise(
    *,
    tile_id: str,
    role: str,
    source: Path,
    imported: Mapping[str, Any],
    tile_root: Path,
    root: Path,
    isis_prefix: Path,
    mola_source_lock: Path,
    mola_coarse_cube: Path,
    mola_coarse_manifest: Path,
) -> dict[str, Any]:
    problem = _copy_verified(
        source / "problem-v2.npz",
        tile_root / "grid/problem-v2.npz",
        expected_sha256=str(imported["problem_sha256"]),
    )
    prepared_source = source / "prepared-manifest.json"
    if file_sha256(prepared_source) != imported.get("prepared_manifest_sha256"):
        raise ValueError("Paper1 HiRISE prepared manifest differs from import lock")
    prepared = read_json(prepared_source)
    ortho_source = source / "raw/orthoimage.tif"
    ortho = tile_root / "raw/hirise-orthoimage.tif"
    if ortho_source.is_file():
        _copy_verified(ortho_source, ortho, expected_sha256=None)
        ortho_receipt: Path | None = None
        ortho_origin = "paper1_exact_official_ortho_bytes"
    else:
        url = str(dict(prepared.get("assets", {})).get("orthoimage", ""))
        if not url.startswith("https://"):
            raise FileNotFoundError("Paper1 HiRISE ORTHO is absent and has no official URL")
        ortho_receipt = tile_root / "raw/hirise-orthoimage.receipt.json"
        download_official_asset(
            url=url,
            output=ortho,
            receipt=ortho_receipt,
            paper2_root=root,
            max_attempts=32,
            repair_invalid_existing=True,
            resume_partial=True,
            backoff_initial_seconds=1.0,
            backoff_max_seconds=30.0,
        )
        ortho_origin = "official_prepared_manifest_url"
    grid = load_problem_grid(problem, product_id=tile_id)
    attempt = tile_root / "attempts" / f"hirise-{time.time_ns()}-{os.getpid()}"
    attempt.mkdir(parents=True)
    mola_export = attempt / "mola-megt-export.tif"
    command = _export_geotiff(
        mola_coarse_cube,
        mola_export,
        gdal_translate=isis_prefix / "bin/gdal_translate",
        command_runner=_default_command_runner,
    )
    image, image_mask = _rasterio_reprojector(_rasterio_reader(ortho), grid, "bilinear")
    z0, dem_mask = _rasterio_reprojector(_rasterio_reader(mola_export), grid, "bilinear")
    image = np.asarray(image, dtype=np.float32)
    image_mask = np.asarray(image_mask, dtype=bool)
    z0 = np.asarray(z0, dtype=np.float32)
    dem_mask = np.asarray(dem_mask, dtype=bool)
    image[~image_mask] = np.nan
    z0[~dem_mask] = np.nan
    coverage = float(np.mean(image_mask))
    finite_dem = float(np.mean(dem_mask))
    if coverage < 0.95 or finite_dem < 0.99:
        raise ValueError(
            f"Paper1 HiRISE raster coverage failed: {tile_id}: image={coverage}, MOLA={finite_dem}"
        )
    artifacts = {}
    for name, values in (
        ("image", image),
        ("image_mask", image_mask),
        ("coarse_dem", z0),
        ("dem_mask", dem_mask),
    ):
        output = attempt / f"{name.replace('_', '-')}.npy"
        _write_array(output, values, paper2_root=root)
        artifacts[name] = _record(output, root=root)
    lineage_path = tile_root / "paper1-raster-lineage.json"
    parents: dict[str, Any] = {
        "paper1_problem_grid": _record(problem, root=root),
        "official_ortho": _record(ortho, root=root),
        "mola_source_lock": _record(mola_source_lock, root=root),
        "mola_coarse_cube": _record(mola_coarse_cube, root=root),
        "mola_coarse_manifest": _record(mola_coarse_manifest, root=root),
        "mola_export": _record(mola_export, root=root),
    }
    if ortho_receipt is not None:
        parents["official_ortho_receipt"] = _record(ortho_receipt, root=root)
    payload: dict[str, Any] = {
        "schema": "paper2.paper1-raster-lineage/v1",
        "status": "complete",
        "tile_id": tile_id,
        "role": role,
        "mission": "HiRISE",
        "target_high_resolution_dem_accessed": False,
        "legacy_problem_used_for_grid_geometry_only": True,
        "primary_image_source": "official_HiRISE_controlled_ortho",
        "official_ortho_origin": ortho_origin,
        "mola_z0_independent_of_target_dtm": True,
        "registration_contract": "provider_controlled_ortho_no_paper2_pixel_adjustment",
        "qa": {
            "image_coverage_fraction": coverage,
            "finite_dem_fraction": finite_dem,
            "paper2_alignment_measurement": None,
            "passed": True,
        },
        "commands": [command],
        "parents": parents,
        "artifacts": artifacts,
    }
    _publish_lineage(lineage_path, payload, root=root)
    return read_json(lineage_path)


def build_paper1_raster_index(
    *,
    paper2_root: str | Path,
    paper1_data_root: str | Path,
    paper1_import_manifest_path: str | Path,
    role_manifest_path: str | Path,
    output_root: str | Path,
    output_index: str | Path,
    config: CTXPipelineConfig,
    mola_source_lock: str | Path,
    mola_coarse_cube: str | Path,
    mola_coarse_manifest: str | Path,
    roles: Sequence[str] = ("paper1_bridge_test", "locked_external"),
    workers: int = 2,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    source_root = Path(paper1_data_root).resolve()
    import_path = guard_paper2_path(paper1_import_manifest_path, paper2_root=root)
    role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
    output = guard_paper2_path(output_root, paper2_root=root, allow_root=True)
    index_path = guard_paper2_path(output_index, paper2_root=root)
    mola_lock = guard_paper2_path(mola_source_lock, paper2_root=root)
    coarse_cube = guard_paper2_path(mola_coarse_cube, paper2_root=root)
    coarse_manifest = guard_paper2_path(mola_coarse_manifest, paper2_root=root)
    requested_roles = tuple(str(value) for value in roles)
    if (
        not requested_roles
        or len(requested_roles) != len(set(requested_roles))
        or not set(requested_roles).issubset(PAPER1_FORMAL_ROLES)
        or workers not in {1, 2}
    ):
        raise ValueError("Paper1 raster roles/workers differ from the frozen contract")
    imported = read_json(import_path)
    role_manifest = read_json(role_path)
    if imported.get("schema") != "paper2.paper1-import/v1" or role_manifest.get(
        "schema"
    ) != "paper2.role-manifest/v1":
        raise ValueError("Paper1 import/role manifest schema differs")
    imported_by_tile = {
        str(raw["tile_id"]): dict(raw)
        for raw in imported.get("tiles", ())
        if raw.get("paper2_role") in requested_roles
    }
    expected = {
        str(tile): role
        for role in requested_roles
        for tile in role_manifest.get("roles", {}).get(role, ())
    }
    if set(imported_by_tile) != set(expected):
        raise ValueError("Paper1 raster import/role tile sets differ")
    ensure_mola_coarse_cube(
        source_lock_path=mola_lock,
        product_id="MEGT90N000FB",
        coarse_cube=coarse_cube,
        manifest_path=coarse_manifest,
        isis_prefix=config.isis_prefix,
        paper2_root=root,
    )

    def prepare(tile_id: str) -> dict[str, Any]:
        role = expected[tile_id]
        tile_root = output / role / tile_id
        lineage_path = tile_root / "paper1-raster-lineage.json"
        existing = _valid_finished_lineage(
            lineage_path, root=root, tile_id=tile_id, role=role
        )
        if existing is not None:
            return existing
        _archive_invalid_lineage(lineage_path, root=root)
        source = source_root / tile_id
        if role == "paper1_bridge_test":
            return _prepare_ctx(
                tile_id=tile_id,
                role=role,
                source=source,
                imported=imported_by_tile[tile_id],
                tile_root=tile_root,
                root=root,
                config=config,
                mola_source_lock=mola_lock,
                mola_coarse_cube=coarse_cube,
                mola_coarse_manifest=coarse_manifest,
            )
        return _prepare_hirise(
            tile_id=tile_id,
            role=role,
            source=source,
            imported=imported_by_tile[tile_id],
            tile_root=tile_root,
            root=root,
            isis_prefix=config.isis_prefix,
            mola_source_lock=mola_lock,
            mola_coarse_cube=coarse_cube,
            mola_coarse_manifest=coarse_manifest,
        )

    records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(prepare, tile): tile for tile in sorted(expected)}
        for future in as_completed(futures):
            tile = futures[future]
            try:
                records[tile] = future.result()
            except Exception as exc:
                failures.append(
                    {
                        "tile_id": tile,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                    }
                )
    tiles = []
    for tile_id, lineage in sorted(records.items()):
        lineage_path = output / expected[tile_id] / tile_id / "paper1-raster-lineage.json"
        artifacts = dict(lineage["artifacts"])
        artifacts["lineage_manifest"] = _record(lineage_path, root=root)
        tiles.append(
            {"tile_id": tile_id, "role": expected[tile_id], "artifacts": artifacts}
        )
    payload: dict[str, Any] = {
        "schema": PAPER1_RASTER_INDEX_SCHEMA,
        "status": "complete" if not failures and set(records) == set(expected) else "incomplete",
        "target_high_resolution_dem_accessed": False,
        "paper1_source_read_only": True,
        "paper1_import_manifest_sha256": file_sha256(import_path),
        "role_manifest_sha256": file_sha256(role_path),
        "mola_source_lock_sha256": file_sha256(mola_lock),
        "mola_coarse_manifest_sha256": file_sha256(coarse_manifest),
        "expected_tile_count": len(expected),
        "complete_tile_count": len(records),
        "failures": sorted(failures, key=lambda row: row["tile_id"]),
        "tiles": tiles,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    _replace_checkpoint(index_path, payload, root=root)
    return payload
