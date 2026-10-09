"""Prepare Paper1 bridge/external inputs with new imagery and controlled Z0."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import os
import shutil

import numpy as np

from .audit_footprints import write_audit_footprint_plan
from .contracts import LaserObservations, PlanContract, ReconstructionProblem, reconstruction_from_observations
from .hashing import canonical_sha256, file_sha256, guard_paper2_path, read_json, write_canonical_json
from .inference_preparation import SEALED_REFERENCE_INPUT_SCHEMA
from .observations import simulate_and_write_laser_observations
from .posterior_preparation import FrozenGPPrior, write_gp_posterior


PAPER1_RASTER_INDEX_SCHEMA = "paper2.paper1-raster-index/v1"
PAPER1_FORMAL_INPUT_INDEX_SCHEMA = "paper2.paper1-formal-inference-input-index/v1"
PAPER1_FORMAL_ROLES = frozenset({"paper1_bridge_test", "locked_external"})
ROLE_BUDGETS = {
    "paper1_bridge_test": (0, 5, 10, 20, 50, 100),
    "locked_external": (0, 5, 10, 20),
}
ROLE_MAX_BUDGET = {"paper1_bridge_test": 100, "locked_external": 20}


def _relative(path: Path, *, root: Path) -> str:
    return guard_paper2_path(path, paper2_root=root).relative_to(root).as_posix()


def _copy_verified(source: Path, destination: Path, *, expected_sha256: str | None) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = file_sha256(source)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"Paper1 source bytes differ: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and file_sha256(destination) == observed:
            return destination
        raise FileExistsError(f"immutable Paper2 import destination differs: {destination}")
    temporary = destination.with_name(f".{destination.name}.copy-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copyfile(source, temporary)
        if file_sha256(temporary) != observed:
            raise OSError("Paper1 copy hash differs before publication")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _verified_artifact(record: Mapping[str, Any], *, root: Path, name: str) -> Path:
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Paper1 raster {name} path is invalid")
    path = guard_paper2_path(root / relative, paper2_root=root)
    if (
        not path.is_file()
        or file_sha256(path) != record.get("sha256")
        or path.stat().st_size != int(record.get("byte_count", -1))
    ):
        raise ValueError(f"Paper1 raster {name} bytes differ")
    return path


def _load_raster_index(path: Path, *, root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    unhashed = dict(payload) if isinstance(payload, Mapping) else {}
    unhashed.pop("content_sha256", None)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != PAPER1_RASTER_INDEX_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("target_high_resolution_dem_accessed") is not False
        or payload.get("content_sha256") != canonical_sha256(unhashed)
    ):
        raise ValueError("Paper1 raster index identity/state/boundary differs")
    result: dict[str, dict[str, Any]] = {}
    for raw in payload.get("tiles", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("Paper1 raster index tile is malformed")
        tile_id = str(raw.get("tile_id", ""))
        role = str(raw.get("role", ""))
        if not tile_id or tile_id in result or role not in PAPER1_FORMAL_ROLES:
            raise ValueError("Paper1 raster index tile identity/role differs")
        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "image",
            "image_mask",
            "coarse_dem",
            "dem_mask",
            "lineage_manifest",
        }:
            raise ValueError("Paper1 raster index artifact set differs")
        paths = {
            name: _verified_artifact(record, root=root, name=name)
            for name, record in artifacts.items()
            if isinstance(record, Mapping)
        }
        if set(paths) != set(artifacts):
            raise ValueError("Paper1 raster index artifact record is malformed")
        lineage = read_json(paths["lineage_manifest"])
        if (
            not isinstance(lineage, Mapping)
            or lineage.get("schema") != "paper2.paper1-raster-lineage/v1"
            or lineage.get("status") != "complete"
            or lineage.get("tile_id") != tile_id
            or lineage.get("role") != role
            or lineage.get("target_high_resolution_dem_accessed") is not False
        ):
            raise ValueError("Paper1 raster lineage identity/boundary differs")
        result[tile_id] = {**dict(raw), "_paths": paths}
    if not result:
        raise ValueError("Paper1 raster index contains no tiles")
    return result


def _load_npy(path: Path, *, dtype: Any) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError("Paper1 formal rasters must use narrow .npy artifacts")
    return np.asarray(np.load(path, allow_pickle=False), dtype=dtype)


def prepare_paper1_formal_inference_inputs(
    *,
    paper2_root: str | Path,
    paper1_data_root: str | Path,
    import_manifest_path: str | Path,
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    raster_index_path: str | Path,
    normalization_path: str | Path,
    gp_prior_path: str | Path,
    roles: Sequence[str],
    replicates: Sequence[int] = tuple(range(5)),
    output_root: str | Path = "data/formal-inference-inputs/paper1",
    observation_seed: int = 20260721,
    build_audit_plans: bool = True,
) -> dict[str, Any]:
    """Materialize Paper1 evaluation tiles without reusing its target-derived Z0/image."""

    root = Path(paper2_root).resolve()
    source_root = Path(paper1_data_root).resolve()
    import_path = guard_paper2_path(import_manifest_path, paper2_root=root)
    role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
    asset_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    raster_path = guard_paper2_path(raster_index_path, paper2_root=root)
    normalization = guard_paper2_path(normalization_path, paper2_root=root)
    prior_path = guard_paper2_path(gp_prior_path, paper2_root=root)
    for path in (import_path, role_path, asset_path, raster_path, normalization, prior_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    requested_roles = tuple(str(value) for value in roles)
    if (
        not requested_roles
        or len(requested_roles) != len(set(requested_roles))
        or not set(requested_roles).issubset(PAPER1_FORMAL_ROLES)
    ):
        raise ValueError("Paper1 formal roles must be unique bridge/external roles")
    repeats = tuple(sorted({int(value) for value in replicates}))
    if not repeats or any(value < 0 for value in repeats):
        raise ValueError("Paper1 formal replicates must be nonnegative")

    imported = read_json(import_path)
    role_manifest = read_json(role_path)
    if imported.get("schema") != "paper2.paper1-import/v1":
        raise ValueError("Paper1 import manifest schema differs")
    if role_manifest.get("schema") != "paper2.role-manifest/v1":
        raise ValueError("role manifest schema differs")
    requested_tiles = {
        str(tile)
        for role in requested_roles
        for tile in role_manifest.get("roles", {}).get(role, ())
    }
    expected_count = sum(
        len(role_manifest.get("roles", {}).get(role, ())) for role in requested_roles
    )
    if len(requested_tiles) != expected_count:
        raise ValueError("Paper1 formal roles overlap or are empty")
    imported_by_tile = {
        str(raw["tile_id"]): dict(raw)
        for raw in imported.get("tiles", ())
        if raw.get("paper2_role") in requested_roles
    }
    if set(imported_by_tile) != requested_tiles:
        raise ValueError("Paper1 import and role manifests differ")
    raster_by_tile = _load_raster_index(raster_path, root=root)
    if set(raster_by_tile) != requested_tiles:
        raise ValueError("Paper1 raster index does not exactly cover requested roles")

    output = guard_paper2_path(output_root, paper2_root=root, allow_root=True)
    if output == root:
        raise ValueError("Paper1 formal output must be below Paper2")
    prior = FrozenGPPrior.from_mapping(read_json(prior_path))
    frozen_hashes = {
        "asset_lock": file_sha256(asset_path),
        "gp_prior": file_sha256(prior_path),
        "normalization": file_sha256(normalization),
        "paper1_import": file_sha256(import_path),
        "paper1_raster_index": file_sha256(raster_path),
    }
    cells: list[dict[str, Any]] = []
    tiles: list[dict[str, Any]] = []
    sealed_entries: list[dict[str, str]] = []
    for tile_id in sorted(requested_tiles):
        imported_record = imported_by_tile[tile_id]
        role = str(imported_record["paper2_role"])
        raster = raster_by_tile[tile_id]
        if raster.get("role") != role:
            raise ValueError("Paper1 raster/import roles differ")
        source = source_root / tile_id
        tile_root = output / role / tile_id
        legacy_root = tile_root / "paper1-source"
        legacy_problem = _copy_verified(
            source / "problem-v2.npz",
            legacy_root / "problem-v2.npz",
            expected_sha256=str(imported_record["problem_sha256"]),
        )
        prepared = _copy_verified(
            source / "prepared-manifest.json",
            legacy_root / "prepared-manifest.json",
            expected_sha256=str(imported_record["prepared_manifest_sha256"]),
        )
        sealed = _copy_verified(
            source / "reference.npz",
            legacy_root / "reference.npz",
            expected_sha256=None,
        )
        with np.load(legacy_problem, allow_pickle=False) as archive:
            legacy_tile = str(archive["tile_id"].item())
            legacy_z0 = np.asarray(archive["coarse_dem"], dtype=np.float32)
            legacy_valid = np.asarray(archive["valid_mask"], dtype=bool)
            legacy_shape = legacy_z0.shape
            legacy_transform = tuple(float(value) for value in archive["transform"])
            legacy_crs = str(archive["crs_wkt"].item())
        with np.load(sealed, allow_pickle=False) as archive:
            sealed_tile = str(archive["tile_id"].item())
            sealed_shape = np.asarray(archive["dem"]).shape
        if legacy_tile != tile_id or sealed_tile != tile_id or sealed_shape != legacy_shape:
            raise ValueError("Paper1 copied problem/reference tile grids differ")
        paths = raster["_paths"]
        image = _load_npy(paths["image"], dtype=np.float32)
        image_mask = _load_npy(paths["image_mask"], dtype=bool)
        mola_z0 = _load_npy(paths["coarse_dem"], dtype=np.float32)
        mola_mask = _load_npy(paths["dem_mask"], dtype=bool)
        if any(
            value.shape != legacy_shape
            for value in (image, image_mask, legacy_z0, legacy_valid, mola_z0, mola_mask)
        ):
            raise ValueError("Paper1 new image/MOLA rasters differ from legacy plan grid")
        if not np.array_equal(image_mask, np.isfinite(image)):
            raise ValueError("Paper1 raster image mask differs from finite support")
        if not np.array_equal(mola_mask, np.isfinite(mola_z0)) or np.mean(mola_mask) < 0.99:
            raise ValueError("Paper1 raster MOLA finite support is below 99 percent")
        if np.any(~np.isfinite(legacy_z0[legacy_valid])):
            raise ValueError("Paper1 controlled Z0 is non-finite on its valid mask")

        max_budget = ROLE_MAX_BUDGET[role]
        max_plan_path: Path | None = None
        max_problem: ReconstructionProblem | None = None
        for budget in ROLE_BUDGETS[role]:
            plan_path = guard_paper2_path(
                root / "data/plans/paper1-faopt" / tile_id / f"k{budget:03d}" / "plan.json",
                paper2_root=root,
            )
            plan = PlanContract.read(plan_path)
            expected_plan_hash = str(imported_record["plan_sha256"][str(budget)])
            if (
                file_sha256(plan_path) != expected_plan_hash
                or plan.tile_id != tile_id
                or plan.budget != budget
                or plan.problem_sha256 != file_sha256(legacy_problem)
                or plan.transform != legacy_transform
                or plan.crs_wkt != legacy_crs
            ):
                raise ValueError("Paper1 imported plan/grid/hash differs")
            for replicate in repeats:
                cell_root = tile_root / f"k{budget:03d}" / f"r{replicate:02d}"
                observations_path = cell_root / "laser-observations.npz"
                simulate_and_write_laser_observations(
                    plan,
                    sealed,
                    observations_path,
                    replicate=replicate,
                    seed=observation_seed,
                    paper2_root=root,
                )
                observations = LaserObservations.read(observations_path)
                problem = reconstruction_from_observations(
                    tile_id=tile_id,
                    z0=legacy_z0,
                    image=image,
                    valid_mask=legacy_valid,
                    observations=observations,
                    observations_path=observations_path,
                    frozen_model_hashes=frozen_hashes,
                    metadata={
                        "prepared_manifest_sha256": file_sha256(prepared),
                        "paper1_input_lineage_sha256": file_sha256(
                            paths["lineage_manifest"]
                        ),
                        "legacy_problem_sha256": file_sha256(legacy_problem),
                        "z0_source": "paper1-controlled-degradation-factor-24",
                        "degradation_factor": 24,
                    },
                    paper2_root=root,
                )
                problem_path = problem.write(
                    cell_root / "reconstruction-problem.npz", paper2_root=root
                )
                posterior = cell_root / "gp-posterior.npz"
                posterior_manifest = cell_root / "gp-posterior-manifest.json"
                write_gp_posterior(
                    problem=problem,
                    plan=plan,
                    prior=prior,
                    problem_sha256=file_sha256(problem_path),
                    plan_sha256=file_sha256(plan_path),
                    prior_sha256=file_sha256(prior_path),
                    posterior_output=posterior,
                    manifest_output=posterior_manifest,
                    paper2_root=root,
                )
                cells.append(
                    {
                        "tile_id": tile_id,
                        "role": role,
                        "budget": budget,
                        "noise_replicate": replicate,
                        "problem": _relative(problem_path, root=root),
                        "problem_sha256": file_sha256(problem_path),
                        "plan": _relative(plan_path, root=root),
                        "plan_sha256": file_sha256(plan_path),
                        "observations": _relative(observations_path, root=root),
                        "observations_sha256": file_sha256(observations_path),
                        "gp_posterior": _relative(posterior, root=root),
                        "gp_posterior_sha256": file_sha256(posterior),
                        "gp_manifest": _relative(posterior_manifest, root=root),
                        "gp_manifest_sha256": file_sha256(posterior_manifest),
                    }
                )
                if budget == max_budget and replicate == 0:
                    max_plan_path = plan_path
                    max_problem = problem
        if max_plan_path is None or max_problem is None:
            raise AssertionError("Paper1 formal preparation lost its maximum plan")
        audit_path: Path | None = None
        if build_audit_plans:
            audit_path = tile_root / "audit-footprint-plan.json"
            write_audit_footprint_plan(
                audit_path,
                problem=max_problem,
                selection_plan=PlanContract.read(max_plan_path),
                selection_plan_path=max_plan_path,
                asset_lock_path=asset_path,
                paper2_root=root,
                required_selection_budget=max_budget,
            )
        sealed_entries.append(
            {
                "tile_id": tile_id,
                "reference_relative_path": _relative(sealed, root=root),
                "grid_plan_relative_path": _relative(max_plan_path, root=root),
            }
        )
        tiles.append(
            {
                "tile_id": tile_id,
                "role": role,
                "prepared_manifest": _relative(prepared, root=root),
                "prepared_manifest_sha256": file_sha256(prepared),
                "candidate_problem": _relative(legacy_problem, root=root),
                "candidate_problem_sha256": file_sha256(legacy_problem),
                "paper1_import_manifest": _relative(import_path, root=root),
                "paper1_import_manifest_sha256": file_sha256(import_path),
                "paper1_input_lineage_manifest": _relative(
                    paths["lineage_manifest"], root=root
                ),
                "paper1_input_lineage_manifest_sha256": file_sha256(
                    paths["lineage_manifest"]
                ),
                "audit_footprint_plan": (
                    _relative(audit_path, root=root) if audit_path is not None else None
                ),
                "audit_footprint_plan_sha256": (
                    file_sha256(audit_path) if audit_path is not None else None
                ),
            }
        )

    cells.sort(key=lambda row: (row["role"], row["tile_id"], row["budget"], row["noise_replicate"]))
    tiles.sort(key=lambda row: (row["role"], row["tile_id"]))
    index: dict[str, Any] = {
        "schema": PAPER1_FORMAL_INPUT_INDEX_SCHEMA,
        "status": "complete",
        "truth_paths_exported_to_inference_workers": False,
        "paper1_source_read_only": True,
        "asset_lock_sha256": file_sha256(asset_path),
        "role_manifest_sha256": file_sha256(role_path),
        "paper1_import_manifest_sha256": file_sha256(import_path),
        "paper1_raster_index_sha256": file_sha256(raster_path),
        "normalization_relative_path": _relative(normalization, root=root),
        "normalization_sha256": file_sha256(normalization),
        "gp_prior_relative_path": _relative(prior_path, root=root),
        "gp_prior_sha256": file_sha256(prior_path),
        "observation_seed": observation_seed,
        "roles": list(requested_roles),
        "replicates": list(repeats),
        "tile_counts": dict(sorted(Counter(row["role"] for row in tiles).items())),
        "tiles": tiles,
        "cells": cells,
    }
    index["content_sha256"] = canonical_sha256(index)
    index_path = write_canonical_json(
        output / "formal-inference-input-index.json", index, paper2_root=root
    )
    sealed_input = {
        "schema": SEALED_REFERENCE_INPUT_SCHEMA,
        "entries": sorted(sealed_entries, key=lambda row: row["tile_id"]),
    }
    sealed_path = write_canonical_json(
        output / "sealed-reference-input.json", sealed_input, paper2_root=root
    )
    return {
        **index,
        "index_relative_path": _relative(index_path, root=root),
        "index_file_sha256": file_sha256(index_path),
        "sealed_reference_input_relative_path": _relative(sealed_path, root=root),
        "sealed_reference_input_sha256": file_sha256(sealed_path),
    }
