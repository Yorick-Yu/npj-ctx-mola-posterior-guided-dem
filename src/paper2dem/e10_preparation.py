"""Prepare every physical E10 cell from an E3 cell and independent MOLA Z0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import PlanContract, ReconstructionProblem
from .e10_mola import build_e10_mola_problem, write_e10_mola_audit
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)
from .inference_preparation import FORMAL_INPUT_INDEX_SCHEMA
from .posterior_preparation import FrozenGPPrior, write_gp_posterior


E10_INPUT_INDEX_SCHEMA = "paper2.e10-formal-input-index/v1"
E10_BUDGET_REPLICATES = {
    0: (0,),
    20: tuple(range(5)),
    100: tuple(range(5)),
}
E10_MANIFESTS_PER_TILE = sum(len(value) for value in E10_BUDGET_REPLICATES.values())


def _relative(path: Path, *, root: Path) -> str:
    return guard_paper2_path(path, paper2_root=root).relative_to(root).as_posix()


def _verified_path(
    value: str | Path,
    *,
    root: Path,
    expected_sha256: str | None = None,
) -> Path:
    path = guard_paper2_path(value, paper2_root=root)
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"E10 source artifact bytes differ: {path}")
    return path


def _verified_record(record: Mapping[str, Any], *, root: Path, name: str) -> Path:
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"E10 {name} record has an invalid relative path")
    path = _verified_path(
        relative,
        root=root,
        expected_sha256=str(record.get("sha256", "")),
    )
    if path.stat().st_size != int(record.get("byte_count", -1)):
        raise ValueError(f"E10 {name} record byte count differs")
    return path


def _read_standard_index(path: Path, *, root: Path) -> Mapping[str, Any]:
    payload = read_json(path)
    unhashed = dict(payload) if isinstance(payload, Mapping) else {}
    unhashed.pop("content_sha256", None)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != FORMAL_INPUT_INDEX_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("truth_paths_exported_to_inference_workers") is not False
        or payload.get("content_sha256") != canonical_sha256(unhashed)
    ):
        raise ValueError("E10 requires a complete truth-isolated formal input index")
    return payload


def _merge_standard_indexes(
    paths: Sequence[Path], *, root: Path
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    if not paths:
        raise ValueError("E10 requires at least one E3 input index")
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    tiles: dict[str, dict[str, Any]] = {}
    shared: dict[str, Any] | None = None
    shared_fields = (
        "asset_lock_sha256",
        "role_manifest_sha256",
        "normalization_relative_path",
        "normalization_sha256",
        "gp_prior_relative_path",
        "gp_prior_sha256",
        "observation_seed",
    )
    for path in paths:
        payload = _read_standard_index(path, root=root)
        current = {field: payload.get(field) for field in shared_fields}
        if shared is None:
            shared = current
        elif current != shared:
            raise ValueError("E10 source indexes do not share one frozen input contract")
        for raw in payload.get("cells", ()):
            if not isinstance(raw, Mapping):
                raise ValueError("E10 source index cell is malformed")
            record = dict(raw)
            key = (
                str(record.get("tile_id", "")),
                int(record.get("budget", -1)),
                int(record.get("noise_replicate", -1)),
            )
            for field, hash_field in (
                ("problem", "problem_sha256"),
                ("plan", "plan_sha256"),
                ("observations", "observations_sha256"),
                ("gp_posterior", "gp_posterior_sha256"),
                ("gp_manifest", "gp_manifest_sha256"),
            ):
                _verified_path(
                    str(record.get(field, "")),
                    root=root,
                    expected_sha256=str(record.get(hash_field, "")),
                )
            previous = cells.setdefault(key, record)
            if previous != record:
                raise ValueError(f"E10 source indexes conflict for cell {key}")
        for raw in payload.get("tiles", ()):
            if not isinstance(raw, Mapping):
                raise ValueError("E10 source index tile is malformed")
            record = dict(raw)
            tile_id = str(record.get("tile_id", ""))
            previous = tiles.get(tile_id)
            if previous is None:
                tiles[tile_id] = record
                continue
            for field in (
                "role",
                "asset_role",
                "geographic_group_id",
                "prepared_manifest",
                "prepared_manifest_sha256",
                "source_preparation_manifest",
                "source_preparation_manifest_sha256",
                "ctx_input_lineage_manifest",
                "ctx_input_lineage_manifest_sha256",
            ):
                if previous.get(field) != record.get(field):
                    raise ValueError(f"E10 source tile metadata conflicts: {tile_id}/{field}")
            if previous.get("audit_footprint_plan") is None and record.get(
                "audit_footprint_plan"
            ) is not None:
                tiles[tile_id] = record
    if shared is None:
        raise AssertionError("E10 source index merge lost its shared contract")
    return cells, tiles, shared


def _mola_inputs(
    tile: Mapping[str, Any], *, root: Path
) -> dict[str, Path]:
    prepared_path = _verified_path(
        str(tile.get("prepared_manifest", "")),
        root=root,
        expected_sha256=str(tile.get("prepared_manifest_sha256", "")),
    )
    prepared = read_json(prepared_path)
    outputs = prepared.get("outputs") if isinstance(prepared, Mapping) else None
    if not isinstance(outputs, Mapping):
        raise ValueError("E10 prepared manifest lacks MOLA output records")
    coarse = _verified_record(dict(outputs.get("coarse_dem", {})), root=root, name="coarse_dem")
    mask = _verified_record(dict(outputs.get("dem_mask", {})), root=root, name="dem_mask")
    lineage = _verified_record(
        dict(outputs.get("coarse_lineage", {})), root=root, name="coarse_lineage"
    )
    lineage_payload = read_json(lineage)
    artifacts = lineage_payload.get("artifacts") if isinstance(lineage_payload, Mapping) else None
    if not isinstance(artifacts, Mapping):
        raise ValueError("E10 MOLA coarse lineage lacks artifact records")
    source_lock = _verified_record(
        dict(artifacts.get("mola_source_lock", {})),
        root=root,
        name="mola_source_lock",
    )
    return {
        "coarse_dem_path": coarse,
        "dem_mask_path": mask,
        "coarse_lineage_path": lineage,
        "mola_source_lock_path": source_lock,
    }


def prepare_e10_formal_inputs(
    *,
    paper2_root: str | Path,
    source_index_paths: Sequence[str | Path],
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    gp_prior_path: str | Path,
    output_root: str | Path = "data/formal-inference-inputs/E10",
) -> dict[str, Any]:
    """Build the exact K={0,20,100} E10 physical cells before G5 unlock."""

    root = Path(paper2_root).resolve()
    role_path = _verified_path(role_manifest_path, root=root)
    asset_path = _verified_path(asset_lock_path, root=root)
    prior_path = _verified_path(gp_prior_path, root=root)
    index_paths = tuple(_verified_path(path, root=root) for path in source_index_paths)
    roles = read_json(role_path)
    if (
        not isinstance(roles, Mapping)
        or roles.get("schema") != "paper2.role-manifest/v1"
        or roles.get("asset_lock_sha256") != file_sha256(asset_path)
    ):
        raise ValueError("E10 role manifest differs from the formal asset lock")
    locked_tiles = tuple(
        sorted(str(value) for value in roles.get("roles", {}).get("paper2_locked_test", ()))
    )
    if len(locked_tiles) not in {24, 30, 36} or len(set(locked_tiles)) != len(locked_tiles):
        raise ValueError("E10 requires the frozen 24/30/36 locked-test tile set")
    source_cells, source_tiles, shared = _merge_standard_indexes(index_paths, root=root)
    if (
        shared["asset_lock_sha256"] != file_sha256(asset_path)
        or shared["role_manifest_sha256"] != file_sha256(role_path)
        or shared["gp_prior_sha256"] != file_sha256(prior_path)
    ):
        raise ValueError("E10 source indexes differ from their requested locks/prior")
    prior = FrozenGPPrior.from_mapping(read_json(prior_path))
    output = guard_paper2_path(output_root, paper2_root=root, allow_root=True)
    if output == root:
        raise ValueError("E10 output must be below Paper2")

    required_keys = {
        (tile_id, budget, replicate)
        for tile_id in locked_tiles
        for budget, replicates in E10_BUDGET_REPLICATES.items()
        for replicate in replicates
    }
    if not required_keys.issubset(source_cells):
        raise ValueError(
            f"E10 source inputs are incomplete: missing={len(required_keys - set(source_cells))}"
        )
    if not set(locked_tiles).issubset(source_tiles):
        raise ValueError("E10 source indexes lack locked-test tile metadata")

    cells: list[dict[str, Any]] = []
    tile_records: list[dict[str, Any]] = []
    cell_manifests: list[Path] = []
    for tile_id in locked_tiles:
        tile = source_tiles[tile_id]
        if tile.get("role") != "paper2_locked_test":
            raise ValueError("E10 source tile is not in the effective locked-test role")
        audit_path = _verified_path(
            str(tile.get("audit_footprint_plan", "")),
            root=root,
            expected_sha256=str(tile.get("audit_footprint_plan_sha256", "")),
        )
        mola = _mola_inputs(tile, root=root)
        for budget, replicates in E10_BUDGET_REPLICATES.items():
            for replicate in replicates:
                base = source_cells[(tile_id, budget, replicate)]
                base_problem = _verified_path(
                    str(base["problem"]),
                    root=root,
                    expected_sha256=str(base["problem_sha256"]),
                )
                plan = _verified_path(
                    str(base["plan"]),
                    root=root,
                    expected_sha256=str(base["plan_sha256"]),
                )
                observations = _verified_path(
                    str(base["observations"]),
                    root=root,
                    expected_sha256=str(base["observations_sha256"]),
                )
                cell_root = output / tile_id / f"k{budget:03d}" / f"r{replicate:02d}"
                problem_path = cell_root / "reconstruction-problem.npz"
                manifest_path = cell_root / "e10-mola-problem-manifest.json"
                build_e10_mola_problem(
                    e3_problem_path=base_problem,
                    output_path=problem_path,
                    manifest_path=manifest_path,
                    paper2_root=root,
                    **mola,
                )
                problem = ReconstructionProblem.read(problem_path)
                posterior_path = cell_root / "gp-posterior.npz"
                posterior_manifest = cell_root / "gp-posterior-manifest.json"
                write_gp_posterior(
                    problem=problem,
                    plan=PlanContract.read(plan),
                    prior=prior,
                    problem_sha256=file_sha256(problem_path),
                    plan_sha256=file_sha256(plan),
                    prior_sha256=file_sha256(prior_path),
                    posterior_output=posterior_path,
                    manifest_output=posterior_manifest,
                    paper2_root=root,
                )
                cell_manifests.append(manifest_path)
                cells.append(
                    {
                        "experiment": "E10",
                        "condition": "main",
                        "tile_id": tile_id,
                        "role": "paper2_locked_test",
                        "budget": budget,
                        "noise_replicate": replicate,
                        "problem": _relative(problem_path, root=root),
                        "problem_sha256": file_sha256(problem_path),
                        "plan": _relative(plan, root=root),
                        "plan_sha256": file_sha256(plan),
                        "observations": _relative(observations, root=root),
                        "observations_sha256": file_sha256(observations),
                        "gp_posterior": _relative(posterior_path, root=root),
                        "gp_posterior_sha256": file_sha256(posterior_path),
                        "gp_manifest": _relative(posterior_manifest, root=root),
                        "gp_manifest_sha256": file_sha256(posterior_manifest),
                        "e10_mola_problem_manifest": _relative(
                            manifest_path, root=root
                        ),
                        "e10_mola_problem_manifest_sha256": file_sha256(
                            manifest_path
                        ),
                    }
                )
        tile_records.append(
            {
                **{
                    field: tile[field]
                    for field in (
                        "tile_id",
                        "role",
                        "asset_role",
                        "geographic_group_id",
                        "prepared_manifest",
                        "prepared_manifest_sha256",
                        "source_preparation_manifest",
                        "source_preparation_manifest_sha256",
                        "ctx_input_lineage_manifest",
                        "ctx_input_lineage_manifest_sha256",
                        "audit_footprint_plan",
                        "audit_footprint_plan_sha256",
                    )
                },
                "mola_coarse_dem": _relative(mola["coarse_dem_path"], root=root),
                "mola_coarse_lineage": _relative(
                    mola["coarse_lineage_path"], root=root
                ),
            }
        )

    audit_path = output / "e10-mola-audit.json"
    write_e10_mola_audit(
        audit_path,
        tile_manifest_paths=cell_manifests,
        role_manifest_path=role_path,
        asset_lock_path=asset_path,
        paper2_root=root,
        expected_tile_count=len(locked_tiles),
        expected_manifests_per_tile=E10_MANIFESTS_PER_TILE,
    )
    cells.sort(key=lambda row: (row["tile_id"], row["budget"], row["noise_replicate"]))
    payload: dict[str, Any] = {
        "schema": E10_INPUT_INDEX_SCHEMA,
        "status": "complete",
        "truth_paths_exported_to_inference_workers": False,
        "experiment": "E10",
        "z0_source": "locked-independent-mola-megt",
        "asset_lock_sha256": file_sha256(asset_path),
        "role_manifest_sha256": file_sha256(role_path),
        "normalization_relative_path": str(shared["normalization_relative_path"]),
        "normalization_sha256": str(shared["normalization_sha256"]),
        "gp_prior_relative_path": _relative(prior_path, root=root),
        "gp_prior_sha256": file_sha256(prior_path),
        "source_index_sha256": canonical_sha256(
            [file_sha256(path) for path in index_paths]
        ),
        "e10_mola_audit": _relative(audit_path, root=root),
        "e10_mola_audit_sha256": file_sha256(audit_path),
        "budgets": list(E10_BUDGET_REPLICATES),
        "replicates_by_budget": {
            str(budget): list(replicates)
            for budget, replicates in E10_BUDGET_REPLICATES.items()
        },
        "tile_count": len(tile_records),
        "cell_count": len(cells),
        "tiles": tile_records,
        "cells": cells,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    index_path = write_canonical_json(
        output / "formal-inference-input-index.json", payload, paper2_root=root
    )
    return {
        **payload,
        "index_relative_path": _relative(index_path, root=root),
        "index_file_sha256": file_sha256(index_path),
    }
