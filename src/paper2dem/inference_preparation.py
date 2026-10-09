"""Truth-isolated preparation of reusable formal inference inputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .audit_footprints import write_audit_footprint_plan
from .contracts import (
    AssetLock,
    LaserObservations,
    PlanContract,
    ReconstructionProblem,
    reconstruction_from_observations,
)
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)
from .input_lineage import build_ctx_input_lineage
from .observations import simulate_and_write_laser_observations
from .posterior_preparation import FrozenGPPrior, write_gp_posterior
from .training_preparation import (
    _controlled_dem_from_problem,
    _load_npy,
    _present_file,
    _source_plan_and_reference,
)


FORMAL_INPUT_INDEX_SCHEMA = "paper2.formal-inference-input-index/v1"
SEALED_REFERENCE_INPUT_SCHEMA = "paper2.sealed-reference-input/v1"
CTX_FORMAL_ROLES = frozenset(
    {
        "bridge_val",
        "uncertainty_calibration",
        "paper2_locked_test",
        "sealed_reserve",
    }
)


def _relative(path: Path, *, root: Path) -> str:
    return guard_paper2_path(path, paper2_root=root).relative_to(root).as_posix()


def _role_tiles(role_manifest: Mapping[str, Any], roles: Sequence[str]) -> dict[str, tuple[str, ...]]:
    if role_manifest.get("schema") != "paper2.role-manifest/v1":
        raise ValueError("formal inference preparation requires paper2.role-manifest/v1")
    requested = tuple(str(role) for role in roles)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("formal inference roles must be unique and nonempty")
    if not set(requested).issubset(CTX_FORMAL_ROLES):
        raise ValueError("formal CTX inference preparation received an unsupported role")
    role_values = role_manifest.get("roles")
    if not isinstance(role_values, Mapping):
        raise ValueError("role manifest lacks role tile bindings")
    result: dict[str, tuple[str, ...]] = {}
    for role in requested:
        values = tuple(str(value) for value in role_values.get(role, ()))
        if not values or len(values) != len(set(values)):
            raise ValueError(f"role {role} has no unique frozen tile set")
        result[role] = values
    return result


def prepare_formal_inference_inputs(
    *,
    paper2_root: str | Path,
    asset_lock_path: str | Path,
    role_manifest_path: str | Path,
    normalization_path: str | Path,
    gp_prior_path: str | Path,
    roles: Sequence[str],
    budgets: Sequence[int],
    replicates: Sequence[int],
    output_root: str | Path = "data/formal-inference-inputs",
    observation_seed: int = 20260721,
    build_audit_plans: bool = True,
) -> dict[str, Any]:
    """Prepare problem/GP inputs while keeping evaluator reference paths separate.

    The returned inference index is safe for inference workers.  Reference paths are
    emitted only to a separate sealed-reference *input* manifest consumed by the
    evaluator-side lock builder.
    """

    root = Path(paper2_root).resolve()
    asset_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
    normalization_file = guard_paper2_path(normalization_path, paper2_root=root)
    prior_file = guard_paper2_path(gp_prior_path, paper2_root=root)
    for path in (asset_path, role_path, normalization_file, prior_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    asset_lock = AssetLock.read(asset_path)
    role_manifest = read_json(role_path)
    selected_role_tiles = _role_tiles(role_manifest, roles)
    requested_tiles = {
        tile_id for values in selected_role_tiles.values() for tile_id in values
    }
    if len(requested_tiles) != sum(len(values) for values in selected_role_tiles.values()):
        raise ValueError("formal inference roles reuse a tile")
    effective_role_by_tile = {
        tile_id: role
        for role, values in selected_role_tiles.items()
        for tile_id in values
    }
    budget_values = tuple(sorted({int(value) for value in budgets}))
    replicate_values = tuple(sorted({int(value) for value in replicates}))
    if (
        not budget_values
        or any(value not in {0, 5, 10, 20, 50, 100} for value in budget_values)
        or not replicate_values
        or any(value < 0 for value in replicate_values)
    ):
        raise ValueError("formal inference budgets/replicates are invalid")
    if build_audit_plans and (100 not in budget_values or 0 not in replicate_values):
        raise ValueError("audit allocation requires the K=100 replicate-0 input")
    prior = FrozenGPPrior.from_mapping(read_json(prior_file))
    output = guard_paper2_path(output_root, paper2_root=root, allow_root=True)
    if output == root:
        raise ValueError("formal inference output must be below the Paper2 root")

    assets = sorted(
        (asset for asset in asset_lock.assets if asset.product_id in requested_tiles),
        key=lambda asset: (effective_role_by_tile[asset.product_id], asset.product_id),
    )
    if {asset.product_id for asset in assets} != requested_tiles:
        missing = sorted(requested_tiles.difference(asset.product_id for asset in assets))
        raise ValueError(f"asset lock lacks requested formal inference tiles: {missing}")
    for asset in assets:
        effective_role = effective_role_by_tile[asset.product_id]
        promoted_reserve = (
            asset.role == "sealed_reserve" and effective_role == "paper2_locked_test"
        )
        if asset.role != effective_role and not promoted_reserve:
            raise ValueError("role manifest and asset lock differ for a formal inference tile")
        if not asset.qa.passed or asset.prepared_manifest_sha256 is None:
            raise ValueError(f"formal inference asset is not QA/prepared: {asset.product_id}")

    asset_hash = file_sha256(asset_path)
    role_hash = file_sha256(role_path)
    normalization_hash = file_sha256(normalization_file)
    prior_hash = file_sha256(prior_file)
    records: list[dict[str, Any]] = []
    reference_entries: list[dict[str, str]] = []
    tile_records: list[dict[str, Any]] = []
    for asset in assets:
        effective_role = effective_role_by_tile[asset.product_id]
        tile_root = output / effective_role / asset.product_id
        source_manifest_path = _present_file(
            asset, "source_preparation_manifest", root=root
        )
        prepared_manifest_path = _present_file(asset, "prepared_manifest", root=root)
        if file_sha256(prepared_manifest_path) != asset.prepared_manifest_sha256:
            raise ValueError("prepared manifest differs from the formal asset lock")
        lineage_path = tile_root / "ctx-input-lineage.json"
        build_ctx_input_lineage(
            source_preparation_manifest_path=source_manifest_path,
            prepared_manifest_path=prepared_manifest_path,
            asset_lock_path=asset_path,
            output_path=lineage_path,
            paper2_root=root,
            role_manifest_path=role_path,
            effective_role=effective_role,
        )
        image_path = _present_file(asset, "image", root=root)
        image = np.asarray(_load_npy(image_path, name="image"), dtype=np.float32)
        _, controlled_z0, controlled_mask = _controlled_dem_from_problem(
            asset, root=root
        )
        z0 = np.asarray(controlled_z0, dtype=np.float32)
        valid_mask = np.asarray(controlled_mask, dtype=bool)
        if image.shape != z0.shape or valid_mask.shape != z0.shape:
            raise ValueError(
                "formal inference image/controlled-Z0/mask arrays do not align"
            )

        reference_for_tile: Path | None = None
        kmax_plan_path: Path | None = None
        kmax_problem_path: Path | None = None
        for budget in budget_values:
            plan_path, reference_path, _ = _source_plan_and_reference(
                asset, root=root, budget=budget
            )
            plan = PlanContract.read(plan_path)
            if (
                plan.tile_id != asset.product_id
                or plan.budget != budget
                or plan.problem_sha256 != asset.problem_sha256
            ):
                raise ValueError("formal inference plan differs from the frozen G0 asset")
            if reference_for_tile is None:
                reference_for_tile = reference_path
            elif reference_for_tile != reference_path:
                raise ValueError("one tile exposes multiple evaluator references")
            for replicate in replicate_values:
                cell_root = tile_root / f"k{budget:03d}" / f"r{replicate:02d}"
                observations_path = cell_root / "laser-observations.npz"
                simulate_and_write_laser_observations(
                    plan,
                    reference_path,
                    observations_path,
                    replicate=replicate,
                    seed=observation_seed,
                    paper2_root=root,
                )
                observations = LaserObservations.read(observations_path)
                problem = reconstruction_from_observations(
                    tile_id=asset.product_id,
                    z0=z0,
                    image=image,
                    valid_mask=valid_mask,
                    observations=observations,
                    observations_path=observations_path,
                    frozen_model_hashes={
                        "asset_lock": asset_hash,
                        "gp_prior": prior_hash,
                        "normalization": normalization_hash,
                        "role_manifest": role_hash,
                    },
                    metadata={
                        "prepared_manifest_sha256": file_sha256(
                            prepared_manifest_path
                        ),
                        "z0_source": "paper1-controlled-degradation-factor-24",
                        "degradation_factor": 24,
                    },
                    paper2_root=root,
                )
                problem_path = problem.write(
                    cell_root / "reconstruction-problem.npz", paper2_root=root
                )
                posterior_path = cell_root / "gp-posterior.npz"
                gp_manifest_path = cell_root / "gp-posterior-manifest.json"
                write_gp_posterior(
                    problem=problem,
                    plan=plan,
                    prior=prior,
                    problem_sha256=file_sha256(problem_path),
                    plan_sha256=file_sha256(plan_path),
                    prior_sha256=prior_hash,
                    posterior_output=posterior_path,
                    manifest_output=gp_manifest_path,
                    paper2_root=root,
                )
                records.append(
                    {
                        "tile_id": asset.product_id,
                        "role": effective_role,
                        "budget": budget,
                        "noise_replicate": replicate,
                        "problem": _relative(problem_path, root=root),
                        "problem_sha256": file_sha256(problem_path),
                        "plan": _relative(plan_path, root=root),
                        "plan_sha256": file_sha256(plan_path),
                        "observations": _relative(observations_path, root=root),
                        "observations_sha256": file_sha256(observations_path),
                        "gp_posterior": _relative(posterior_path, root=root),
                        "gp_posterior_sha256": file_sha256(posterior_path),
                        "gp_manifest": _relative(gp_manifest_path, root=root),
                        "gp_manifest_sha256": file_sha256(gp_manifest_path),
                    }
                )
                if budget == 100 and replicate == 0:
                    kmax_plan_path = plan_path
                    kmax_problem_path = problem_path
        if reference_for_tile is None or kmax_plan_path is None or kmax_problem_path is None:
            raise AssertionError("formal inference preparation lost its Kmax/reference inputs")
        audit_path: Path | None = None
        if build_audit_plans:
            audit_path = tile_root / "audit-footprint-plan.json"
            write_audit_footprint_plan(
                audit_path,
                problem=ReconstructionProblem.read(kmax_problem_path),
                selection_plan=PlanContract.read(kmax_plan_path),
                selection_plan_path=kmax_plan_path,
                asset_lock_path=asset_path,
                paper2_root=root,
            )
        reference_entries.append(
            {
                "tile_id": asset.product_id,
                "reference_relative_path": _relative(reference_for_tile, root=root),
                "grid_plan_relative_path": _relative(kmax_plan_path, root=root),
            }
        )
        tile_records.append(
            {
                "tile_id": asset.product_id,
                "role": effective_role,
                "asset_role": asset.role,
                "geographic_group_id": asset.geographic_group_id,
                "prepared_manifest": _relative(prepared_manifest_path, root=root),
                "prepared_manifest_sha256": file_sha256(prepared_manifest_path),
                "source_preparation_manifest": _relative(
                    source_manifest_path, root=root
                ),
                "source_preparation_manifest_sha256": file_sha256(
                    source_manifest_path
                ),
                "ctx_input_lineage_manifest": _relative(lineage_path, root=root),
                "ctx_input_lineage_manifest_sha256": file_sha256(lineage_path),
                "audit_footprint_plan": (
                    _relative(audit_path, root=root) if audit_path is not None else None
                ),
                "audit_footprint_plan_sha256": (
                    file_sha256(audit_path) if audit_path is not None else None
                ),
            }
        )

    records.sort(
        key=lambda item: (
            str(item["role"]),
            str(item["tile_id"]),
            int(item["budget"]),
            int(item["noise_replicate"]),
        )
    )
    tile_records.sort(key=lambda item: (str(item["role"]), str(item["tile_id"])))
    index: dict[str, Any] = {
        "schema": FORMAL_INPUT_INDEX_SCHEMA,
        "status": "complete",
        "truth_paths_exported_to_inference_workers": False,
        "asset_lock_sha256": asset_hash,
        "role_manifest_sha256": role_hash,
        "normalization_relative_path": _relative(normalization_file, root=root),
        "normalization_sha256": normalization_hash,
        "gp_prior_relative_path": _relative(prior_file, root=root),
        "gp_prior_sha256": prior_hash,
        "observation_seed": int(observation_seed),
        "roles": list(roles),
        "budgets": list(budget_values),
        "replicates": list(replicate_values),
        "tile_counts": dict(
            sorted(
                Counter(
                    effective_role_by_tile[asset.product_id] for asset in assets
                ).items()
            )
        ),
        "tiles": tile_records,
        "cells": records,
    }
    index["content_sha256"] = canonical_sha256(index)
    index_path = write_canonical_json(
        output / "formal-inference-input-index.json", index, paper2_root=root
    )
    reference_input = {
        "schema": SEALED_REFERENCE_INPUT_SCHEMA,
        "entries": sorted(reference_entries, key=lambda item: item["tile_id"]),
    }
    reference_input_path = write_canonical_json(
        output / "sealed-reference-input.json", reference_input, paper2_root=root
    )
    return {
        **index,
        "index_relative_path": _relative(index_path, root=root),
        "index_file_sha256": file_sha256(index_path),
        "sealed_reference_input_relative_path": _relative(
            reference_input_path, root=root
        ),
        "sealed_reference_input_sha256": file_sha256(reference_input_path),
    }
