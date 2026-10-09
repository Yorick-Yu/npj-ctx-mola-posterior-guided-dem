"""Build and audit the preregistered validation-only E8 stress inputs.

The builder is the only E8 component allowed to open the sealed simulator input.
Published reconstruction problems and stress manifests contain hashes, never the
simulator path.  Formal inference therefore receives a condition-specific problem
without gaining a locator for the clean DEM.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .audit_footprints import (
    array_sha256,
    read_audit_footprint_plan,
    validate_audit_plan_binding,
    write_audit_footprint_plan,
)
from .contracts import (
    FootprintContract,
    LaserObservations,
    PlanContract,
    PlanPoint,
    ReconstructionProblem,
    reconstruction_from_observations,
)
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)
from .observations import simulate_and_write_laser_observations
from .posterior_preparation import FrozenGPPrior, write_gp_posterior
from .sealed_reference import read_sealed_reference_lock


E8_STRESS_SCHEMA = "paper2.e8-stress-manifest/v1"
E8_INPUT_INDEX_SCHEMA = "paper2.e8-formal-input-index/v1"
E8_STRESS_CONDITIONS = (
    "degradation-scale=0.5",
    "degradation-scale=2.0",
    "noise-scale=0.5",
    "noise-scale=2.0",
    "footprint=point",
    "footprint-diameter-m=50",
    "footprint-diameter-m=200",
    "track-spacing-scale=0.5",
    "track-spacing-scale=2.0",
    "image=missing",
    "stitch-window=uniform",
    "stitch-window=tukey-0.5",
)
E8_BUDGETS = (20, 100)
E8_REPLICATES = tuple(range(5))
BASE_DEGRADATION_FACTOR = 24


def _content_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )


def _relative(path: Path, *, root: Path) -> str:
    return guard_paper2_path(path, paper2_root=root).relative_to(root).as_posix()


def _existing(path: str | Path, *, root: Path) -> Path:
    result = guard_paper2_path(path, paper2_root=root)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _verified_relative(record: Mapping[str, Any], *, root: Path) -> Path:
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("E8 artifact path is invalid")
    path = _existing(root / relative, root=root)
    if (
        file_sha256(path) != record.get("sha256")
        or path.stat().st_size != int(record.get("byte_count", -1))
    ):
        raise ValueError(f"E8 artifact bytes differ: {relative}")
    return path


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    path = _existing(path, root=root)
    return {
        "relative_path": _relative(path, root=root),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def controlled_degrade_dem(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    *,
    factor: int,
) -> np.ndarray:
    """Exact Paper1 block-mean, nearest-fill, cubic-upsample degradation."""

    values = np.asarray(dem, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or valid.shape != values.shape or not np.any(valid):
        raise ValueError("controlled degradation requires a valid two-dimensional DEM")
    if factor <= 1:
        result = np.array(values, copy=True)
        if not np.all(valid):
            nearest = ndimage.distance_transform_edt(
                ~valid, return_distances=False, return_indices=True
            )
            result = result[tuple(nearest)]
        return result
    height, width = values.shape
    coarse_height = math.ceil(height / factor)
    coarse_width = math.ceil(width / factor)
    padded = np.full(
        (coarse_height * factor, coarse_width * factor), np.nan, dtype=np.float64
    )
    padded[:height, :width] = np.where(valid, values, np.nan)
    blocks = padded.reshape(coarse_height, factor, coarse_width, factor)
    with np.errstate(invalid="ignore"):
        coarse = np.nanmean(blocks, axis=(1, 3))
    coarse_valid = np.isfinite(coarse)
    if not np.any(coarse_valid):
        raise ValueError("controlled degradation produced no finite coarse cells")
    nearest = ndimage.distance_transform_edt(
        ~coarse_valid, return_distances=False, return_indices=True
    )
    coarse_filled = np.nan_to_num(coarse)[tuple(nearest)]
    zoom = (height / coarse_height, width / coarse_width)
    result = ndimage.zoom(
        coarse_filled, zoom, order=3, mode="nearest", prefilter=True
    )
    result = result[:height, :width]
    if result.shape != (height, width):
        result = np.pad(
            result,
            ((0, height - result.shape[0]), (0, width - result.shape[1])),
            mode="edge",
        )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("controlled degradation produced non-finite values")
    return np.asarray(result, dtype=np.float32)


def _pixel_centres(
    rows: np.ndarray,
    columns: np.ndarray,
    transform: Sequence[float],
) -> np.ndarray:
    a, b, c, d, e, f = (float(value) for value in transform)
    col = np.asarray(columns, dtype=np.float64) + 0.5
    row = np.asarray(rows, dtype=np.float64) + 0.5
    return np.column_stack((a * col + b * row + c, d * col + e * row + f))


def _full_support_mask(
    valid_mask: np.ndarray,
    transform: Sequence[float],
    *,
    diameter_m: float,
) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)
    if diameter_m <= 0.0:
        return valid
    a, b, _, d, e, _ = (float(value) for value in transform)
    basis = np.asarray(((a, b), (d, e)), dtype=np.float64)
    singular = np.linalg.svd(basis, compute_uv=False)
    minimum_pixel = float(np.min(singular))
    maximum_pixel = float(np.max(singular))
    if minimum_pixel <= 0.0:
        raise ValueError("E8 track-spacing grid transform is singular")
    # Include a half-pixel diagonal so snapped centres remain conservatively
    # inside the full-support domain used by the footprint quadrature.
    radius = diameter_m / 2.0 + maximum_pixel / math.sqrt(2.0)
    extent = int(math.ceil(radius / minimum_pixel))
    row_offsets, column_offsets = np.indices(
        (2 * extent + 1, 2 * extent + 1), dtype=np.float64
    )
    row_offsets -= extent
    column_offsets -= extent
    offsets = np.stack((column_offsets, row_offsets), axis=-1)
    distances = np.linalg.norm(offsets @ basis.T, axis=-1)
    structure = distances <= radius + 1.0e-9
    return ndimage.binary_erosion(valid, structure=structure, border_value=0)


def _along_cross_basis(points: Sequence[PlanPoint]) -> tuple[np.ndarray, np.ndarray]:
    by_track: dict[str, list[np.ndarray]] = {}
    for point in points:
        by_track.setdefault(point.track_id, []).append(
            np.asarray((point.x, point.y), dtype=np.float64)
        )
    if len(by_track) < 2:
        raise ValueError("E8 track-spacing stress requires at least two selected tracks")
    within = []
    for values in by_track.values():
        stack = np.stack(values)
        within.append(stack - np.mean(stack, axis=0, keepdims=True))
    residuals = np.concatenate(within, axis=0)
    covariance = residuals.T @ residuals
    if float(np.trace(covariance)) <= 1.0e-12:
        coordinates = np.asarray([(point.x, point.y) for point in points])
        centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
        covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    along = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=np.float64)
    if abs(along[0]) > 1.0e-12:
        along *= 1.0 if along[0] > 0.0 else -1.0
    elif along[1] < 0.0:
        along *= -1.0
    cross = np.asarray((-along[1], along[0]), dtype=np.float64)
    return along, cross


def _track_spacing(points: Sequence[PlanPoint], cross: np.ndarray) -> float:
    by_track: dict[str, list[np.ndarray]] = {}
    for point in points:
        by_track.setdefault(point.track_id, []).append(
            np.asarray((point.x, point.y), dtype=np.float64)
        )
    projected = sorted(
        float(np.mean(np.stack(values), axis=0) @ cross)
        for values in by_track.values()
    )
    differences = np.diff(projected)
    positive = differences[differences > 1.0e-9]
    if positive.size == 0:
        raise ValueError("E8 selected tracks have no measurable cross-track spacing")
    return float(np.median(positive))


def derive_track_spacing_plan(
    base_plan: PlanContract,
    problem: ReconstructionProblem,
    *,
    scale: float,
) -> tuple[PlanContract, dict[str, Any]]:
    """Scale cross-track offsets and deterministically snap to full-support pixels."""

    base_plan.validate()
    problem.validate()
    if base_plan.budget != 100 or base_plan.tile_id != problem.tile_id:
        raise ValueError("E8 track-spacing derivation requires the tile Kmax=100 plan")
    if scale not in {0.5, 2.0}:
        raise ValueError("E8 track-spacing scale must be 0.5 or 2.0")
    _, cross = _along_cross_basis(base_plan.points)
    coordinates = np.asarray([(point.x, point.y) for point in base_plan.points])
    centre = np.mean(coordinates, axis=0)
    by_track: dict[str, list[int]] = {}
    for index, point in enumerate(base_plan.points):
        by_track.setdefault(point.track_id, []).append(index)
    desired = np.array(coordinates, copy=True)
    for indices in by_track.values():
        centroid = np.mean(coordinates[indices], axis=0)
        offset = float((centroid - centre) @ cross)
        desired[indices] += (scale - 1.0) * offset * cross

    support = _full_support_mask(
        problem.valid_mask,
        base_plan.transform,
        diameter_m=base_plan.footprint.diameter_m,
    )
    rows, columns = np.nonzero(support)
    if len(rows) < base_plan.budget:
        raise ValueError("E8 track-spacing support has fewer than 100 valid centres")
    centres = _pixel_centres(rows, columns, base_plan.transform)
    tree = cKDTree(centres)
    snapped = np.empty_like(desired)
    used: set[int] = set()
    query_count = min(len(centres), max(256, base_plan.budget * 4))
    for point_index, target in enumerate(desired):
        _, candidate_indices = tree.query(target, k=query_count)
        candidates = np.atleast_1d(candidate_indices).astype(int).tolist()
        chosen = next((value for value in candidates if value not in used), None)
        if chosen is None:
            distances = np.sum((centres - target) ** 2, axis=1)
            order = np.argsort(distances, kind="stable")
            chosen = next((int(value) for value in order if int(value) not in used), None)
        if chosen is None:  # pragma: no cover - guarded by support cardinality
            raise RuntimeError("E8 track-spacing snapping exhausted valid centres")
        used.add(chosen)
        snapped[point_index] = centres[chosen]

    derived_points = tuple(
        PlanPoint(
            rank=point.rank,
            candidate_id=point.candidate_id,
            track_id=point.track_id,
            x=float(snapped[index, 0]),
            y=float(snapped[index, 1]),
        )
        for index, point in enumerate(base_plan.points)
    )
    original_spacing = _track_spacing(base_plan.points, cross)
    achieved_spacing = _track_spacing(derived_points, cross)
    achieved_scale = achieved_spacing / original_spacing
    lower, upper = ((0.30, 0.80) if scale == 0.5 else (1.40, 2.60))
    if not lower <= achieved_scale <= upper:
        raise ValueError(
            f"E8 track-spacing snapping missed the stress regime: {achieved_scale:.3f}"
        )
    geometry_hash = canonical_sha256(
        [
            {
                "candidate_id": point.candidate_id,
                "track_id": point.track_id,
                "x_hex": point.x.hex(),
                "y_hex": point.y.hex(),
            }
            for point in derived_points
        ]
    )
    provenance = {
        "schema": "paper2.e8-track-spacing-derivation/v1",
        "scale": scale,
        "basis_rule": "within-track-principal-axis-then-orthogonal-cross-axis",
        "snap_rule": "nearest-unique-full-support-pixel-centre",
        "base_plan_contract_sha256": base_plan.sha256,
        "base_candidate_geometry_sha256": base_plan.candidate_geometry_sha256,
        "derived_candidate_geometry_sha256": geometry_hash,
        "original_median_track_spacing_m": original_spacing,
        "achieved_median_track_spacing_m": achieved_spacing,
        "achieved_scale": achieved_scale,
    }
    source_hash = canonical_sha256(provenance)
    plan = PlanContract(
        tile_id=base_plan.tile_id,
        scenario_id=base_plan.scenario_id,
        method="e8-track-spacing",
        budget=base_plan.budget,
        crs_wkt=base_plan.crs_wkt,
        transform=base_plan.transform,
        footprint=base_plan.footprint,
        noise_std_m=base_plan.noise_std_m,
        candidate_geometry_sha256=geometry_hash,
        problem_sha256=base_plan.problem_sha256,
        paper1=base_plan.paper1,
        source_selection_sha256=source_hash,
        points=derived_points,
    )
    plan.validate()
    return plan, provenance


def _modified_plan(
    base: PlanContract,
    *,
    condition: str,
    footprint: FootprintContract | None = None,
    noise_std_m: float | None = None,
) -> tuple[PlanContract, dict[str, Any]]:
    protocol = {
        "schema": "paper2.e8-plan-parameter-stress/v1",
        "condition": condition,
        "base_plan_contract_sha256": base.sha256,
        "base_candidate_geometry_sha256": base.candidate_geometry_sha256,
        "footprint": (base.footprint if footprint is None else footprint).to_dict(),
        "noise_std_m": base.noise_std_m if noise_std_m is None else noise_std_m,
        "points_unchanged": True,
    }
    result = PlanContract(
        tile_id=base.tile_id,
        scenario_id=base.scenario_id,
        method="e8-parameter-stress",
        budget=base.budget,
        crs_wkt=base.crs_wkt,
        transform=base.transform,
        footprint=base.footprint if footprint is None else footprint,
        noise_std_m=base.noise_std_m if noise_std_m is None else noise_std_m,
        candidate_geometry_sha256=base.candidate_geometry_sha256,
        problem_sha256=base.problem_sha256,
        paper1=base.paper1,
        source_selection_sha256=base.source_selection_sha256,
        points=base.points,
    )
    result.validate()
    return result, protocol


def _write_plan_family(
    plan: PlanContract,
    output: Path,
    *,
    root: Path,
) -> dict[int, Path]:
    if plan.budget != 100:
        raise ValueError("E8 derived plan family requires a Kmax=100 plan")
    maximum = plan.write(output / "k100" / "plan.json", paper2_root=root)
    prefix = plan.prefix(20).write(output / "k020" / "plan.json", paper2_root=root)
    return {20: prefix, 100: maximum}


def _condition_plan_family(
    *,
    condition: str,
    base_plan: PlanContract,
    base_paths: Mapping[int, Path],
    base_problem: ReconstructionProblem,
    output: Path,
    root: Path,
) -> tuple[dict[int, Path], dict[str, Any], bool]:
    if condition.startswith("noise-scale="):
        scale = float(condition.split("=", 1)[1])
        plan, provenance = _modified_plan(
            base_plan,
            condition=condition,
            noise_std_m=base_plan.noise_std_m * scale,
        )
    elif condition == "footprint=point":
        plan, provenance = _modified_plan(
            base_plan,
            condition=condition,
            footprint=FootprintContract("point", 0.0, 1, True),
        )
    elif condition.startswith("footprint-diameter-m="):
        diameter = float(condition.split("=", 1)[1])
        plan, provenance = _modified_plan(
            base_plan,
            condition=condition,
            footprint=FootprintContract(
                "disk", diameter, base_plan.footprint.quadrature_points, True
            ),
        )
    elif condition.startswith("track-spacing-scale="):
        scale = float(condition.split("=", 1)[1])
        plan, provenance = derive_track_spacing_plan(
            base_plan, base_problem, scale=scale
        )
    else:
        return dict(base_paths), {
            "schema": "paper2.e8-plan-reuse/v1",
            "condition": condition,
            "base_plan_contract_sha256": base_plan.sha256,
        }, False
    return _write_plan_family(plan, output / "plans", root=root), provenance, True


def _reference_arrays(path: Path, *, tile_id: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not {"dem", "valid_mask"}.issubset(archive.files):
            raise ValueError("E8 sealed simulator archive lacks DEM/mask")
        if "tile_id" in archive.files and str(archive["tile_id"].item()) != tile_id:
            raise ValueError("E8 sealed simulator tile differs")
        dem = np.asarray(archive["dem"], dtype=np.float64)
        valid = np.asarray(archive["valid_mask"], dtype=bool)
    if dem.ndim != 2 or valid.shape != dem.shape or not np.any(valid):
        raise ValueError("E8 sealed simulator arrays are invalid")
    return dem, valid & np.isfinite(dem)


def _base_maps(index: Mapping[str, Any], *, root: Path) -> tuple[dict[Any, Any], dict[Any, Any]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in index.get("cells", ()):
        cell = dict(raw)
        key = (
            str(cell.get("tile_id", "")),
            int(cell.get("budget", -1)),
            int(cell.get("noise_replicate", -1)),
        )
        if key in cells:
            raise ValueError(f"E8 base index contains duplicate cell {key}")
        for field in ("problem", "plan", "observations", "gp_posterior", "gp_manifest"):
            path = _existing(root / str(cell.get(field, "")), root=root)
            if file_sha256(path) != cell.get(f"{field}_sha256"):
                raise ValueError(f"E8 base index {field} bytes differ")
        cells[key] = cell
    tiles: dict[str, dict[str, Any]] = {}
    for raw in index.get("tiles", ()):
        tile = dict(raw)
        tile_id = str(tile.get("tile_id", ""))
        if not tile_id or tile_id in tiles:
            raise ValueError("E8 base index tile records are invalid")
        for field in (
            "prepared_manifest",
            "source_preparation_manifest",
            "ctx_input_lineage_manifest",
            "audit_footprint_plan",
        ):
            path = _existing(root / str(tile.get(field, "")), root=root)
            if file_sha256(path) != tile.get(f"{field}_sha256"):
                raise ValueError(f"E8 base tile {field} bytes differ")
        tiles[tile_id] = tile
    return cells, tiles


def _stress_manifest(
    *,
    tile_id: str,
    condition: str,
    base_index_path: Path,
    asset_lock_path: Path,
    role_manifest_path: Path,
    sealed_lock_path: Path,
    simulator_entry: Mapping[str, Any],
    plan_provenance: Mapping[str, Any],
    base_plan_paths: Mapping[int, Path],
    plan_paths: Mapping[int, Path],
    audit_path: Path,
    cells: Sequence[Mapping[str, Any]],
    output_path: Path,
    root: Path,
) -> Path:
    payload: dict[str, Any] = {
        "schema": E8_STRESS_SCHEMA,
        "status": "complete",
        "role": "bridge_val",
        "tile_id": tile_id,
        "condition": condition,
        "validation_only": True,
        "confirmatory_claim_allowed": False,
        "simulator_path_exported": False,
        "base_input_index_sha256": file_sha256(base_index_path),
        "asset_lock_sha256": file_sha256(asset_lock_path),
        "role_manifest_sha256": file_sha256(role_manifest_path),
        "sealed_simulator_lock_sha256": file_sha256(sealed_lock_path),
        "simulator_file_sha256": str(simulator_entry["reference_sha256"]),
        "simulator_grid_sha256": str(simulator_entry["grid_sha256"]),
        "plan_provenance": dict(plan_provenance),
        "base_plans": {
            str(budget): _artifact(path, root=root)
            for budget, path in sorted(base_plan_paths.items())
        },
        "condition_plans": {
            str(budget): _artifact(path, root=root)
            for budget, path in sorted(plan_paths.items())
        },
        "audit_footprint_plan": _artifact(audit_path, root=root),
        "cells": sorted(
            (dict(value) for value in cells),
            key=lambda value: (int(value["budget"]), int(value["noise_replicate"])),
        ),
    }
    payload["content_sha256"] = _content_sha256(payload)
    return write_canonical_json(output_path, payload, paper2_root=root)


def prepare_e8_formal_inputs(
    *,
    paper2_root: str | Path,
    base_input_index_path: str | Path,
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    sealed_reference_lock_path: str | Path,
    gp_prior_path: str | Path,
    output_root: str | Path,
    observation_seed: int = 20260721,
) -> dict[str, Any]:
    """Materialize all 12 x 12 x 2 x 5 E8 physical input cells."""

    root = Path(paper2_root).resolve()
    base_path = _existing(base_input_index_path, root=root)
    role_path = _existing(role_manifest_path, root=root)
    asset_path = _existing(asset_lock_path, root=root)
    sealed_path = _existing(sealed_reference_lock_path, root=root)
    prior_path = _existing(gp_prior_path, root=root)
    output = guard_paper2_path(output_root, paper2_root=root, allow_root=True)
    if output == root:
        raise ValueError("E8 output must be below the Paper2 root")
    base = read_json(base_path)
    roles = read_json(role_path)
    if (
        not isinstance(base, Mapping)
        or base.get("schema") != "paper2.formal-inference-input-index/v1"
        or base.get("status") != "complete"
        or base.get("truth_paths_exported_to_inference_workers") is not False
        or base.get("asset_lock_sha256") != file_sha256(asset_path)
        or base.get("role_manifest_sha256") != file_sha256(role_path)
    ):
        raise ValueError("E8 base inference index identity/locks differ")
    tile_ids = tuple(
        str(value) for value in roles.get("roles", {}).get("bridge_val", ())
    )
    if len(tile_ids) != 12 or len(set(tile_ids)) != 12:
        raise ValueError("formal E8 requires exactly 12 bridge_val tiles")
    if base.get("tile_counts") != {"bridge_val": 12}:
        raise ValueError("E8 base index must exactly cover bridge_val")
    if not set(E8_BUDGETS).issubset(set(int(value) for value in base.get("budgets", ()))):
        raise ValueError("E8 base index lacks K=20/100")
    if not set(E8_REPLICATES).issubset(
        set(int(value) for value in base.get("replicates", ()))
    ):
        raise ValueError("E8 base index lacks five paired noise replicates")
    cells_by_key, tiles_by_id = _base_maps(base, root=root)
    expected_base_keys = {
        (tile_id, budget, replicate)
        for tile_id in tile_ids
        for budget in E8_BUDGETS
        for replicate in E8_REPLICATES
    }
    if not expected_base_keys.issubset(cells_by_key):
        raise ValueError("E8 base index does not cover every required physical cell")
    if set(tiles_by_id) != set(tile_ids):
        raise ValueError("E8 base tile records differ from bridge_val")

    sealed = read_sealed_reference_lock(
        sealed_path, paper2_root=root, verify_files=True
    )
    simulator_by_tile = {
        str(entry["tile_id"]): dict(entry)
        for entry in sealed["entries"]
        if str(entry["tile_id"]) in tile_ids
    }
    if set(simulator_by_tile) != set(tile_ids):
        raise ValueError("E8 sealed simulator lock lacks bridge_val")
    prior = FrozenGPPrior.from_mapping(read_json(prior_path))
    if file_sha256(prior_path) != base.get("gp_prior_sha256"):
        raise ValueError("E8 GP prior differs from the base input index")

    manifest_records: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    for tile_id in sorted(tile_ids):
        tile = tiles_by_id[tile_id]
        simulator_entry = simulator_by_tile[tile_id]
        simulator_path = _existing(
            root / str(simulator_entry["reference_relative_path"]), root=root
        )
        simulator_dem, simulator_valid = _reference_arrays(
            simulator_path, tile_id=tile_id
        )
        base_plan_paths = {
            budget: _existing(
                root / str(cells_by_key[(tile_id, budget, 0)]["plan"]), root=root
            )
            for budget in E8_BUDGETS
        }
        base_plan = PlanContract.read(base_plan_paths[100])
        base_problem_path = _existing(
            root / str(cells_by_key[(tile_id, 100, 0)]["problem"]), root=root
        )
        base_problem = ReconstructionProblem.read(base_problem_path)
        if (
            base_plan.tile_id != tile_id
            or base_problem.tile_id != tile_id
            or simulator_dem.shape != base_problem.z0.shape
            or tuple(simulator_entry["transform"]) != base_plan.transform
            or str(simulator_entry["crs_wkt"]) != base_plan.crs_wkt
        ):
            raise ValueError("E8 base plan/problem/simulator grids differ")

        for condition in E8_STRESS_CONDITIONS:
            condition_root = output / tile_id / condition.replace("=", "-")
            plan_paths, plan_provenance, plan_changed = _condition_plan_family(
                condition=condition,
                base_plan=base_plan,
                base_paths=base_plan_paths,
                base_problem=base_problem,
                output=condition_root,
                root=root,
            )
            condition_cells: list[dict[str, Any]] = []
            condition_protocol = canonical_sha256(
                {
                    "schema": E8_STRESS_SCHEMA,
                    "tile_id": tile_id,
                    "condition": condition,
                    "plan_provenance": plan_provenance,
                    "base_input_index_sha256": file_sha256(base_path),
                }
            )
            for budget in E8_BUDGETS:
                plan_path = plan_paths[budget]
                plan = PlanContract.read(plan_path)
                for replicate in E8_REPLICATES:
                    base_cell = cells_by_key[(tile_id, budget, replicate)]
                    base_problem_cell = ReconstructionProblem.read(
                        _existing(root / str(base_cell["problem"]), root=root)
                    )
                    if plan_changed:
                        observations_path = condition_root / f"k{budget:03d}" / (
                            f"r{replicate:02d}/laser-observations.npz"
                        )
                        simulate_and_write_laser_observations(
                            plan,
                            simulator_path,
                            observations_path,
                            replicate=replicate,
                            seed=observation_seed,
                            paper2_root=root,
                        )
                    else:
                        observations_path = _existing(
                            root / str(base_cell["observations"]), root=root
                        )
                    observations = LaserObservations.read(observations_path)
                    z0 = np.asarray(base_problem_cell.z0, dtype=np.float32)
                    image = np.asarray(base_problem_cell.image, dtype=np.float32)
                    metadata = dict(base_problem_cell.metadata)
                    metadata.update(
                        {
                            "experimental_condition": condition,
                            "e8_stress_protocol_sha256": condition_protocol,
                            "base_problem_sha256": file_sha256(
                                _existing(root / str(base_cell["problem"]), root=root)
                            ),
                        }
                    )
                    gp_changed = plan_changed
                    if condition.startswith("degradation-scale="):
                        scale = float(condition.split("=", 1)[1])
                        factor = int(round(BASE_DEGRADATION_FACTOR * scale))
                        z0 = controlled_degrade_dem(
                            simulator_dem,
                            simulator_valid & base_problem_cell.valid_mask,
                            factor=factor,
                        )
                        metadata.update(
                            {
                                "z0_source": f"controlled-degradation-factor-{factor}",
                                "degradation_factor": factor,
                                "base_z0_array_sha256": array_sha256(
                                    base_problem_cell.z0
                                ),
                            }
                        )
                        gp_changed = True
                    elif condition == "image=missing":
                        image = np.full(base_problem_cell.image.shape, np.nan, dtype=np.float32)
                        metadata["base_image_array_sha256"] = array_sha256(
                            base_problem_cell.image
                        )
                    frozen_hashes = dict(base_problem_cell.frozen_model_hashes)
                    frozen_hashes["e8_stress_protocol"] = condition_protocol
                    problem = reconstruction_from_observations(
                        tile_id=tile_id,
                        z0=z0,
                        image=image,
                        valid_mask=base_problem_cell.valid_mask,
                        observations=observations,
                        observations_path=observations_path,
                        frozen_model_hashes=frozen_hashes,
                        metadata=metadata,
                        paper2_root=root,
                    )
                    cell_root = condition_root / f"k{budget:03d}" / f"r{replicate:02d}"
                    problem_path = problem.write(
                        cell_root / "reconstruction-problem.npz", paper2_root=root
                    )
                    if gp_changed:
                        posterior_path = cell_root / "gp-posterior.npz"
                        gp_manifest_path = cell_root / "gp-posterior-manifest.json"
                        write_gp_posterior(
                            problem=problem,
                            plan=plan,
                            prior=prior,
                            problem_sha256=file_sha256(problem_path),
                            plan_sha256=file_sha256(plan_path),
                            prior_sha256=file_sha256(prior_path),
                            posterior_output=posterior_path,
                            manifest_output=gp_manifest_path,
                            paper2_root=root,
                        )
                    else:
                        posterior_path = _existing(
                            root / str(base_cell["gp_posterior"]), root=root
                        )
                        gp_manifest_path = _existing(
                            root / str(base_cell["gp_manifest"]), root=root
                        )
                    record = {
                        "tile_id": tile_id,
                        "role": "bridge_val",
                        "condition": condition,
                        "budget": budget,
                        "noise_replicate": replicate,
                        "problem": _artifact(problem_path, root=root),
                        "plan": _artifact(plan_path, root=root),
                        "observations": _artifact(observations_path, root=root),
                        "gp_posterior": _artifact(posterior_path, root=root),
                        "gp_manifest": _artifact(gp_manifest_path, root=root),
                    }
                    condition_cells.append(record)
                    all_cells.append(record)
            audit_problem_record = next(
                value
                for value in condition_cells
                if value["budget"] == 100 and value["noise_replicate"] == 0
            )
            audit_path = condition_root / "audit-footprint-plan.json"
            minimum_separation = (
                150.0 if condition == "footprint-diameter-m=200" else 100.0
            )
            write_audit_footprint_plan(
                audit_path,
                problem=ReconstructionProblem.read(
                    root / audit_problem_record["problem"]["relative_path"]
                ),
                selection_plan=PlanContract.read(plan_paths[100]),
                selection_plan_path=plan_paths[100],
                asset_lock_path=asset_path,
                paper2_root=root,
                minimum_selection_separation_m=minimum_separation,
            )
            manifest_path = _stress_manifest(
                tile_id=tile_id,
                condition=condition,
                base_index_path=base_path,
                asset_lock_path=asset_path,
                role_manifest_path=role_path,
                sealed_lock_path=sealed_path,
                simulator_entry=simulator_entry,
                plan_provenance=plan_provenance,
                base_plan_paths=base_plan_paths,
                plan_paths=plan_paths,
                audit_path=audit_path,
                cells=condition_cells,
                output_path=condition_root / "e8-stress-manifest.json",
                root=root,
            )
            manifest_records.append(
                {
                    "tile_id": tile_id,
                    "condition": condition,
                    "manifest": _artifact(manifest_path, root=root),
                    "audit_footprint_plan": _artifact(audit_path, root=root),
                    "prepared_manifest": _artifact(
                        root / str(tile["prepared_manifest"]), root=root
                    ),
                    "source_preparation_manifest": _artifact(
                        root / str(tile["source_preparation_manifest"]), root=root
                    ),
                    "ctx_input_lineage_manifest": _artifact(
                        root / str(tile["ctx_input_lineage_manifest"]), root=root
                    ),
                }
            )

    expected_count = (
        len(tile_ids)
        * len(E8_STRESS_CONDITIONS)
        * len(E8_BUDGETS)
        * len(E8_REPLICATES)
    )
    if len(all_cells) != expected_count or len(manifest_records) != 12 * 12:
        raise AssertionError("E8 physical input closure differs from the preregistration")
    payload: dict[str, Any] = {
        "schema": E8_INPUT_INDEX_SCHEMA,
        "status": "complete",
        "role": "bridge_val",
        "validation_only": True,
        "confirmatory_claim_allowed": False,
        "simulator_paths_exported_to_inference_workers": False,
        "asset_lock_sha256": file_sha256(asset_path),
        "role_manifest_sha256": file_sha256(role_path),
        "sealed_simulator_lock_sha256": file_sha256(sealed_path),
        "base_input_index_sha256": file_sha256(base_path),
        "gp_prior_relative_path": _relative(prior_path, root=root),
        "gp_prior_sha256": file_sha256(prior_path),
        "normalization_relative_path": str(base["normalization_relative_path"]),
        "normalization_sha256": str(base["normalization_sha256"]),
        "conditions": list(E8_STRESS_CONDITIONS),
        "budgets": list(E8_BUDGETS),
        "replicates": list(E8_REPLICATES),
        "tile_ids": sorted(tile_ids),
        "manifests": sorted(
            manifest_records,
            key=lambda value: (value["tile_id"], value["condition"]),
        ),
        "cells": sorted(
            all_cells,
            key=lambda value: (
                value["tile_id"],
                value["condition"],
                value["budget"],
                value["noise_replicate"],
            ),
        ),
    }
    payload["content_sha256"] = _content_sha256(payload)
    index_path = write_canonical_json(
        output / "formal-inference-input-index.json", payload, paper2_root=root
    )
    return {
        **payload,
        "index_relative_path": _relative(index_path, root=root),
        "index_file_sha256": file_sha256(index_path),
    }


@lru_cache(maxsize=None)
def _audited_e8_manifest(
    manifest_path_text: str,
    paper2_root_text: str,
    manifest_file_sha256: str,
) -> tuple[str, str, dict[tuple[int, int], dict[str, str]]]:
    root = Path(paper2_root_text)
    manifest_file = Path(manifest_path_text)
    if file_sha256(manifest_file) != manifest_file_sha256:
        raise ValueError("E8 stress manifest changed before binding")
    payload = read_json(manifest_file)
    expected_keys = {
        "schema",
        "status",
        "role",
        "tile_id",
        "condition",
        "validation_only",
        "confirmatory_claim_allowed",
        "simulator_path_exported",
        "base_input_index_sha256",
        "asset_lock_sha256",
        "role_manifest_sha256",
        "sealed_simulator_lock_sha256",
        "simulator_file_sha256",
        "simulator_grid_sha256",
        "plan_provenance",
        "base_plans",
        "condition_plans",
        "audit_footprint_plan",
        "cells",
        "content_sha256",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_keys
        or payload.get("schema") != E8_STRESS_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("role") != "bridge_val"
        or payload.get("validation_only") is not True
        or payload.get("confirmatory_claim_allowed") is not False
        or payload.get("simulator_path_exported") is not False
        or payload.get("content_sha256") != _content_sha256(payload)
        or payload.get("condition") not in E8_STRESS_CONDITIONS
    ):
        raise ValueError("E8 stress manifest identity/boundary/hash differs")
    for name in (
        "base_input_index_sha256",
        "asset_lock_sha256",
        "role_manifest_sha256",
        "sealed_simulator_lock_sha256",
        "simulator_file_sha256",
        "simulator_grid_sha256",
    ):
        if not is_sha256(str(payload.get(name, ""))):
            raise ValueError(f"E8 stress manifest {name} is invalid")
    base_plans = payload.get("base_plans")
    condition_plans = payload.get("condition_plans")
    if (
        not isinstance(base_plans, Mapping)
        or not isinstance(condition_plans, Mapping)
        or set(base_plans) != {"20", "100"}
        or set(condition_plans) != {"20", "100"}
    ):
        raise ValueError("E8 stress manifest plan-family closure differs")
    for record in (*base_plans.values(), *condition_plans.values()):
        if not isinstance(record, Mapping):
            raise ValueError("E8 stress manifest plan record is malformed")
        _verified_relative(record, root=root)
    condition_plan_paths = {
        int(budget): _verified_relative(record, root=root)
        for budget, record in condition_plans.items()
    }
    audit_record = payload.get("audit_footprint_plan")
    if not isinstance(audit_record, Mapping):
        raise ValueError("E8 stress manifest lacks its audit footprint plan")
    audit_path = _verified_relative(audit_record, root=root)
    audit_plan = read_audit_footprint_plan(audit_path)
    if (
        audit_plan.tile_id != payload.get("tile_id")
        or audit_plan.asset_lock_sha256 != payload.get("asset_lock_sha256")
    ):
        raise ValueError("E8 audit footprint plan differs from the stress manifest")

    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != len(E8_BUDGETS) * len(E8_REPLICATES):
        raise ValueError("E8 stress manifest physical-cell count differs")
    index: dict[tuple[int, int], dict[str, str]] = {}
    expected_cell_keys = {
        "tile_id",
        "role",
        "condition",
        "budget",
        "noise_replicate",
        "problem",
        "plan",
        "observations",
        "gp_posterior",
        "gp_manifest",
    }
    for raw in cells:
        if not isinstance(raw, Mapping) or set(raw) != expected_cell_keys:
            raise ValueError("E8 stress manifest cell keys differ")
        budget = int(raw.get("budget", -1))
        replicate = int(raw.get("noise_replicate", -1))
        key = (budget, replicate)
        if (
            budget not in E8_BUDGETS
            or replicate not in E8_REPLICATES
            or key in index
            or raw.get("tile_id") != payload.get("tile_id")
            or raw.get("role") != "bridge_val"
            or raw.get("condition") != payload.get("condition")
        ):
            raise ValueError("E8 stress manifest cell identity differs")
        paths: dict[str, str] = {}
        for name in ("problem", "plan", "observations", "gp_posterior", "gp_manifest"):
            record = raw.get(name)
            if not isinstance(record, Mapping):
                raise ValueError(f"E8 stress manifest {name} record is malformed")
            paths[name] = str(_verified_relative(record, root=root))
        if Path(paths["plan"]) != condition_plan_paths[budget]:
            raise ValueError("E8 cell plan differs from its condition plan family")
        problem = ReconstructionProblem.read(paths["problem"])
        plan = PlanContract.read(paths["plan"])
        observations = LaserObservations.read(paths["observations"])
        if (
            problem.tile_id != payload.get("tile_id")
            or plan.tile_id != problem.tile_id
            or observations.tile_id != problem.tile_id
            or plan.budget != budget
            or observations.replicate != replicate
            or problem.plan_sha256 != plan.sha256
            or observations.plan_sha256 != plan.sha256
            or problem.observations_sha256 != file_sha256(Path(paths["observations"]))
            or problem.metadata.get("experimental_condition") != payload.get("condition")
        ):
            raise ValueError("E8 problem/plan/observations condition binding differs")
        index[key] = paths
    if set(index) != {
        (budget, replicate)
        for budget in E8_BUDGETS
        for replicate in E8_REPLICATES
    }:
        raise ValueError("E8 stress manifest cell key closure differs")
    for budget in E8_BUDGETS:
        record = index[(budget, 0)]
        validate_audit_plan_binding(
            audit_plan,
            problem=ReconstructionProblem.read(record["problem"]),
            measurement_plan=PlanContract.read(record["plan"]),
        )
    return str(payload["tile_id"]), str(payload["condition"]), index


def validate_e8_stress_binding(
    *,
    manifest_path: str | Path,
    condition: str,
    budget: int,
    noise_replicate: int,
    problem_path: str | Path,
    plan_path: str | Path,
    observations_path: str | Path | None,
    paper2_root: str | Path,
) -> str:
    """Bind one formal E8 cell to its exact condition-specific physical inputs."""

    root = Path(paper2_root).resolve()
    manifest_file = _existing(manifest_path, root=root)
    manifest_hash = file_sha256(manifest_file)
    manifest_tile, manifest_condition, index = _audited_e8_manifest(
        str(manifest_file), str(root), manifest_hash
    )
    if manifest_condition != condition or condition not in E8_STRESS_CONDITIONS:
        raise ValueError("E8 stress manifest condition differs")
    record = index.get((budget, noise_replicate))
    if record is None:
        raise ValueError("E8 stress manifest does not identify the physical cell")
    requested = {
        "problem": _existing(problem_path, root=root),
        "plan": _existing(plan_path, root=root),
    }
    for name, path in requested.items():
        if path != Path(record[name]):
            raise ValueError(f"E8 {name} path differs from the stress manifest")
    bound_observations = Path(record["observations"])
    requested_observations = (
        bound_observations
        if observations_path is None
        else _existing(observations_path, root=root)
    )
    if requested_observations != bound_observations:
        raise ValueError("E8 observations path differs from the stress manifest")
    requested["observations"] = requested_observations
    if not manifest_tile:
        raise AssertionError("audited E8 manifest lost its tile identity")
    return manifest_hash
