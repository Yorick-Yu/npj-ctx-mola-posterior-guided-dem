"""Immutable held-out footprint plans, observations, and path projections.

The audit footprints are deliberately outside the reconstruction contract.  Their
coordinates may be read at inference only to project completed DEM paths; their
returns and clean targets remain evaluator-only and are never inputs to selection,
GP assimilation, training, or terminal data consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json

import numpy as np

from .contracts import FootprintContract, PlanContract, ReconstructionProblem
from .footprints import build_footprint_operator_with_metadata
from .hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    is_sha256,
    read_json,
    write_canonical_json,
    write_immutable_bytes,
)
from .sealed_reference import reference_grid_sha256


AUDIT_FOOTPRINT_COUNT = 128
AUDIT_FOOTPRINT = FootprintContract(
    shape="disk",
    diameter_m=100.0,
    quadrature_points=32,
    require_full_support=True,
)
AUDIT_USAGE_PROHIBITIONS = (
    "candidate_selection",
    "gp_assimilation",
    "model_training",
    "terminal_data_consistency",
)


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class AuditFootprintPoint:
    audit_id: str
    rank: int
    x: float
    y: float

    def validate(self) -> None:
        if not self.audit_id or self.rank <= 0:
            raise ValueError("audit footprint ID/rank is invalid")
        if not np.isfinite(self.x) or not np.isfinite(self.y):
            raise ValueError("audit footprint coordinates must be finite")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"audit_id": self.audit_id, "rank": self.rank, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AuditFootprintPoint":
        return cls(
            audit_id=str(value["audit_id"]),
            rank=int(value["rank"]),
            x=float(value["x"]),
            y=float(value["y"]),
        )


@dataclass(frozen=True)
class AuditFootprintPlan:
    SCHEMA = "paper2.audit-footprint-plan/v1"

    tile_id: str
    grid_shape: tuple[int, int]
    crs_wkt: str
    transform: tuple[float, float, float, float, float, float]
    footprint: FootprintContract
    points: tuple[AuditFootprintPoint, ...]
    valid_mask_sha256: str
    prepared_manifest_sha256: str
    asset_lock_sha256: str
    selection_plan_file_sha256: str
    selection_plan_contract_sha256: str
    selection_source_sha256: str
    selection_points: tuple[tuple[str, float, float], ...]
    allocation_seed_sha256: str
    heldout_noise_seed_sha256: str
    minimum_selection_separation_m: float
    minimum_pair_separation_m: float
    selection_max_budget: int = 100
    usage_prohibitions: tuple[str, ...] = AUDIT_USAGE_PROHIBITIONS

    def validate(self) -> None:
        if not self.tile_id or len(self.grid_shape) != 2 or any(value <= 0 for value in self.grid_shape):
            raise ValueError("audit footprint tile/grid is invalid")
        if not self.crs_wkt.strip() or len(self.transform) != 6 or not all(
            np.isfinite(value) for value in self.transform
        ):
            raise ValueError("audit footprint CRS/transform is invalid")
        self.footprint.validate()
        if self.footprint != AUDIT_FOOTPRINT:
            raise ValueError("formal audit footprints must be 100 m disks with 32-point full support")
        if len(self.points) != AUDIT_FOOTPRINT_COUNT:
            raise ValueError("formal audit plan must contain exactly 128 footprints")
        ranks = [point.rank for point in self.points]
        identifiers = [point.audit_id for point in self.points]
        if ranks != list(range(1, AUDIT_FOOTPRINT_COUNT + 1)) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ValueError("audit footprint ranks/IDs are not exact and unique")
        for point in self.points:
            point.validate()
        for name, digest in (
            ("valid_mask", self.valid_mask_sha256),
            ("prepared_manifest", self.prepared_manifest_sha256),
            ("asset_lock", self.asset_lock_sha256),
            ("selection_plan_file", self.selection_plan_file_sha256),
            ("selection_plan_contract", self.selection_plan_contract_sha256),
            ("selection_source", self.selection_source_sha256),
            ("allocation_seed", self.allocation_seed_sha256),
            ("heldout_noise_seed", self.heldout_noise_seed_sha256),
        ):
            if not is_sha256(digest):
                raise ValueError(f"audit footprint {name} hash is invalid")
        if not self.selection_points:
            raise ValueError("audit plan must bind the complete frozen selection plan")
        if self.selection_max_budget not in {20, 100}:
            raise ValueError("formal audit selection maximum must be 20 or 100")
        if len(self.selection_points) != self.selection_max_budget:
            raise ValueError("audit plan selection-point count differs from its maximum budget")
        selection_ids = [item[0] for item in self.selection_points]
        if len(selection_ids) != len(set(selection_ids)):
            raise ValueError("bound selection points are not unique")
        if any(not item[0] or not np.isfinite(item[1]) or not np.isfinite(item[2]) for item in self.selection_points):
            raise ValueError("bound selection points are malformed")
        if self.minimum_selection_separation_m < self.footprint.diameter_m:
            raise ValueError("audit footprints must be separated from every selected footprint")
        if self.minimum_pair_separation_m <= 0:
            raise ValueError("audit footprint pair separation must be positive")
        if tuple(self.usage_prohibitions) != AUDIT_USAGE_PROHIBITIONS:
            raise ValueError("audit footprint usage prohibitions differ from the frozen boundary")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "tile_id": self.tile_id,
            "count": len(self.points),
            "grid_shape": list(self.grid_shape),
            "crs_wkt": self.crs_wkt,
            "transform": list(self.transform),
            "grid_sha256": reference_grid_sha256(
                tile_id=self.tile_id,
                shape=self.grid_shape,
                crs_wkt=self.crs_wkt,
                transform=self.transform,
            ),
            "footprint": self.footprint.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "valid_mask_sha256": self.valid_mask_sha256,
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "asset_lock_sha256": self.asset_lock_sha256,
            "selection_plan_file_sha256": self.selection_plan_file_sha256,
            "selection_plan_contract_sha256": self.selection_plan_contract_sha256,
            "selection_source_sha256": self.selection_source_sha256,
            "selection_points": [
                {"candidate_id": candidate_id, "x": x, "y": y}
                for candidate_id, x, y in self.selection_points
            ],
            "selection_max_budget": self.selection_max_budget,
            "allocation_protocol": "paper2-heldout-full-support-pixel-permutation/v1",
            "allocation_seed_sha256": self.allocation_seed_sha256,
            "heldout_noise_protocol": "paper2-audit-candidate-paired-noise/v1",
            "heldout_noise_seed_sha256": self.heldout_noise_seed_sha256,
            "minimum_selection_separation_m": self.minimum_selection_separation_m,
            "minimum_pair_separation_m": self.minimum_pair_separation_m,
            "usage_prohibitions": list(self.usage_prohibitions),
            "created_after_selection": True,
            "reference_used_for_allocation": False,
        }
        if include_hash:
            payload["audit_plan_sha256"] = canonical_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditFootprintPlan":
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported audit footprint plan schema")
        expected = canonical_sha256(
            {name: value for name, value in payload.items() if name != "audit_plan_sha256"}
        )
        if payload.get("audit_plan_sha256") != expected:
            raise ValueError("audit footprint plan semantic hash mismatch")
        result = cls(
            tile_id=str(payload["tile_id"]),
            grid_shape=tuple(int(value) for value in payload["grid_shape"]),
            crs_wkt=str(payload["crs_wkt"]),
            transform=tuple(float(value) for value in payload["transform"]),
            footprint=FootprintContract.from_dict(dict(payload["footprint"])),
            points=tuple(AuditFootprintPoint.from_dict(item) for item in payload["points"]),
            valid_mask_sha256=str(payload["valid_mask_sha256"]),
            prepared_manifest_sha256=str(payload["prepared_manifest_sha256"]),
            asset_lock_sha256=str(payload["asset_lock_sha256"]),
            selection_plan_file_sha256=str(payload["selection_plan_file_sha256"]),
            selection_plan_contract_sha256=str(payload["selection_plan_contract_sha256"]),
            selection_source_sha256=str(payload["selection_source_sha256"]),
            selection_points=tuple(
                (str(item["candidate_id"]), float(item["x"]), float(item["y"]))
                for item in payload["selection_points"]
            ),
            allocation_seed_sha256=str(payload["allocation_seed_sha256"]),
            heldout_noise_seed_sha256=str(payload["heldout_noise_seed_sha256"]),
            minimum_selection_separation_m=float(payload["minimum_selection_separation_m"]),
            minimum_pair_separation_m=float(payload["minimum_pair_separation_m"]),
            selection_max_budget=int(payload.get("selection_max_budget", 100)),
            usage_prohibitions=tuple(str(value) for value in payload["usage_prohibitions"]),
        )
        result.validate()
        if payload.get("count") != AUDIT_FOOTPRINT_COUNT:
            raise ValueError("audit footprint count field differs")
        observed_grid = reference_grid_sha256(
            tile_id=result.tile_id,
            shape=result.grid_shape,
            crs_wkt=result.crs_wkt,
            transform=result.transform,
        )
        if payload.get("grid_sha256") != observed_grid:
            raise ValueError("audit footprint grid hash differs")
        if payload.get("created_after_selection") is not True or payload.get(
            "reference_used_for_allocation"
        ) is not False:
            raise ValueError("audit footprint allocation boundary differs")
        return result


def read_audit_footprint_plan(path: str | Path) -> AuditFootprintPlan:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("audit footprint plan must be a JSON object")
    return AuditFootprintPlan.from_dict(payload)


def _pixel_center(
    row: int, column: int, transform: Sequence[float]
) -> tuple[float, float]:
    a, b, c, d, e, f = (float(value) for value in transform)
    col_center = column + 0.5
    row_center = row + 0.5
    return (
        a * col_center + b * row_center + c,
        d * col_center + e * row_center + f,
    )


def build_audit_footprint_plan(
    *,
    problem: ReconstructionProblem,
    selection_plan: PlanContract,
    selection_plan_path: str | Path,
    asset_lock_path: str | Path,
    minimum_selection_separation_m: float = 100.0,
    minimum_pair_separation_m: float = 50.0,
    required_selection_budget: int = 100,
) -> AuditFootprintPlan:
    """Allocate exactly 128 deterministic locations without reading a reference DEM."""

    problem.validate()
    selection_plan.validate()
    if selection_plan.tile_id != problem.tile_id:
        raise ValueError("selection plan and audit grid tile differ")
    if required_selection_budget not in {20, 100}:
        raise ValueError("audit allocation selection maximum must be 20 or 100")
    if selection_plan.budget != required_selection_budget:
        raise ValueError(
            f"audit allocation requires the exact Kmax={required_selection_budget} plan"
        )
    prepared_hash = str(problem.metadata.get("prepared_manifest_sha256", ""))
    if not is_sha256(prepared_hash):
        raise ValueError("audit allocation requires a prepared-manifest hash")
    selection_path = Path(selection_plan_path)
    asset_path = Path(asset_lock_path)
    if selection_path.resolve() == asset_path.resolve():
        raise ValueError("selection plan and asset lock inputs cannot be the same file")
    valid_hash = array_sha256(np.asarray(problem.valid_mask, dtype=bool))
    seed_hash = canonical_sha256(
        {
            "protocol": "paper2-heldout-full-support-pixel-permutation/v1",
            "tile_id": problem.tile_id,
            "prepared_manifest_sha256": prepared_hash,
            "valid_mask_sha256": valid_hash,
            "selection_plan_file_sha256": file_sha256(selection_path),
            "selection_plan_contract_sha256": selection_plan.sha256,
            "asset_lock_sha256": file_sha256(asset_path),
        }
    )
    noise_seed_hash = canonical_sha256(
        {"protocol": "paper2-audit-candidate-paired-noise/v1", "allocation": seed_hash}
    )
    rows, columns = np.nonzero(np.asarray(problem.valid_mask, dtype=bool))
    if len(rows) < AUDIT_FOOTPRINT_COUNT:
        raise ValueError("valid mask has fewer pixels than required audit footprints")
    generator = np.random.Generator(np.random.PCG64(int(seed_hash[:16], 16)))
    ordering = generator.permutation(len(rows))
    selected_xy = np.asarray(
        [(point.x, point.y) for point in selection_plan.points], dtype=np.float64
    )
    accepted: list[tuple[float, float]] = []
    for index in ordering.tolist():
        center = np.asarray(_pixel_center(int(rows[index]), int(columns[index]), selection_plan.transform))
        if np.any(np.linalg.norm(selected_xy - center, axis=1) < minimum_selection_separation_m):
            continue
        if accepted and np.any(
            np.linalg.norm(np.asarray(accepted, dtype=np.float64) - center, axis=1)
            < minimum_pair_separation_m
        ):
            continue
        support = build_footprint_operator_with_metadata(
            center[None, :],
            grid_shape=problem.z0.shape,
            transform=selection_plan.transform,
            diameter_m=AUDIT_FOOTPRINT.diameter_m,
            quadrature_points=AUDIT_FOOTPRINT.quadrature_points,
            valid_mask=problem.valid_mask,
            require_full_support=True,
            drop_invalid=True,
        )
        if len(support.kept_indices) != 1:
            continue
        accepted.append((float(center[0]), float(center[1])))
        if len(accepted) == AUDIT_FOOTPRINT_COUNT:
            break
    if len(accepted) != AUDIT_FOOTPRINT_COUNT:
        raise ValueError(
            f"could allocate only {len(accepted)}/128 held-out full-support footprints"
        )
    result = AuditFootprintPlan(
        tile_id=problem.tile_id,
        grid_shape=problem.z0.shape,
        crs_wkt=selection_plan.crs_wkt,
        transform=selection_plan.transform,
        footprint=AUDIT_FOOTPRINT,
        points=tuple(
            AuditFootprintPoint(
                audit_id=f"audit-{problem.tile_id}-{rank:03d}", rank=rank, x=x, y=y
            )
            for rank, (x, y) in enumerate(accepted, start=1)
        ),
        valid_mask_sha256=valid_hash,
        prepared_manifest_sha256=prepared_hash,
        asset_lock_sha256=file_sha256(asset_path),
        selection_plan_file_sha256=file_sha256(selection_path),
        selection_plan_contract_sha256=selection_plan.sha256,
        selection_source_sha256=selection_plan.source_selection_sha256,
        selection_points=tuple(
            (point.candidate_id, point.x, point.y) for point in selection_plan.points
        ),
        allocation_seed_sha256=seed_hash,
        heldout_noise_seed_sha256=noise_seed_hash,
        minimum_selection_separation_m=float(minimum_selection_separation_m),
        minimum_pair_separation_m=float(minimum_pair_separation_m),
        selection_max_budget=required_selection_budget,
    )
    result.validate()
    return result


def write_audit_footprint_plan(
    path: str | Path,
    *,
    problem: ReconstructionProblem,
    selection_plan: PlanContract,
    selection_plan_path: str | Path,
    asset_lock_path: str | Path,
    paper2_root: str | Path,
    required_selection_budget: int = 100,
) -> Path:
    result = build_audit_footprint_plan(
        problem=problem,
        selection_plan=selection_plan,
        selection_plan_path=selection_plan_path,
        asset_lock_path=asset_lock_path,
        required_selection_budget=required_selection_budget,
    )
    return write_canonical_json(path, result.to_dict(), paper2_root=paper2_root)


def validate_audit_plan_binding(
    audit_plan: AuditFootprintPlan,
    *,
    problem: ReconstructionProblem,
    measurement_plan: PlanContract,
) -> Any:
    """Rebuild the operator and prove the audit set stayed outside reconstruction."""

    audit_plan.validate()
    problem.validate()
    measurement_plan.validate()
    if audit_plan.tile_id != problem.tile_id or audit_plan.tile_id != measurement_plan.tile_id:
        raise ValueError("audit plan, problem, and measurement plan tile differ")
    if audit_plan.grid_shape != problem.z0.shape:
        raise ValueError("audit plan grid shape differs from reconstruction")
    if audit_plan.crs_wkt != measurement_plan.crs_wkt or audit_plan.transform != measurement_plan.transform:
        raise ValueError("audit plan grid CRS/transform differs from reconstruction plan")
    if audit_plan.valid_mask_sha256 != array_sha256(np.asarray(problem.valid_mask, dtype=bool)):
        raise ValueError("audit plan valid mask differs from reconstruction")
    if audit_plan.prepared_manifest_sha256 != problem.metadata.get("prepared_manifest_sha256"):
        raise ValueError("audit plan prepared manifest differs from reconstruction")
    if audit_plan.selection_source_sha256 != measurement_plan.source_selection_sha256:
        raise ValueError("audit plan was allocated against a different selection source")
    expected_prefix = audit_plan.selection_points[: measurement_plan.budget]
    observed_prefix = tuple(
        (point.candidate_id, point.x, point.y) for point in measurement_plan.points
    )
    if observed_prefix != expected_prefix:
        raise ValueError("measurement plan is not the bound Kmax selection prefix")
    centers = np.asarray([(point.x, point.y) for point in audit_plan.points], dtype=np.float64)
    selected = np.asarray([(item[1], item[2]) for item in audit_plan.selection_points])
    if np.any(
        np.linalg.norm(centers[:, None, :] - selected[None, :, :], axis=2)
        < audit_plan.minimum_selection_separation_m
    ):
        raise ValueError("audit footprint overlaps the frozen selection exclusion zone")
    result = build_footprint_operator_with_metadata(
        centers,
        grid_shape=problem.z0.shape,
        transform=measurement_plan.transform,
        diameter_m=audit_plan.footprint.diameter_m,
        quadrature_points=audit_plan.footprint.quadrature_points,
        valid_mask=problem.valid_mask,
        require_full_support=True,
        drop_invalid=False,
    )
    if result.operator.shape[0] != AUDIT_FOOTPRINT_COUNT:
        raise ValueError("audit footprint operator is not exact")
    return result.operator


@dataclass(frozen=True)
class AuditFootprintObservations:
    tile_id: str
    audit_plan_file_sha256: str
    audit_plan_contract_sha256: str
    replicate: int
    audit_ids: np.ndarray
    return_values_m: np.ndarray
    noise_std_m: float

    SCHEMA = "paper2.audit-footprint-observations/v1"

    def validate(self) -> None:
        if not self.tile_id or self.replicate < 0:
            raise ValueError("audit observation tile/replicate is invalid")
        if not is_sha256(self.audit_plan_file_sha256) or not is_sha256(
            self.audit_plan_contract_sha256
        ):
            raise ValueError("audit observation plan hashes are invalid")
        ids = np.asarray(self.audit_ids, dtype=str)
        values = np.asarray(self.return_values_m, dtype=np.float64)
        if ids.shape != (AUDIT_FOOTPRINT_COUNT,) or values.shape != (AUDIT_FOOTPRINT_COUNT,):
            raise ValueError("audit observations must contain exactly 128 returns")
        if len(set(ids.tolist())) != AUDIT_FOOTPRINT_COUNT or not np.all(np.isfinite(values)):
            raise ValueError("audit observation IDs/returns are invalid")
        if not np.isfinite(self.noise_std_m) or self.noise_std_m <= 0:
            raise ValueError("audit observation noise standard deviation must be positive")

    @property
    def metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.SCHEMA,
            "tile_id": self.tile_id,
            "audit_plan_file_sha256": self.audit_plan_file_sha256,
            "audit_plan_contract_sha256": self.audit_plan_contract_sha256,
            "replicate": self.replicate,
            "noise_std_m": self.noise_std_m,
            "clean_targets_included": False,
            "reference_identifier_included": False,
            "instrument_noise_is_evaluator_only": True,
        }

    def write(self, path: str | Path, *, paper2_root: str | Path) -> Path:
        self.validate()
        stream = BytesIO()
        np.savez_compressed(
            stream,
            metadata=np.asarray(canonical_json_bytes(self.metadata).decode("utf-8")),
            audit_ids=np.asarray(self.audit_ids, dtype=str),
            return_values_m=np.asarray(self.return_values_m, dtype=np.float64),
        )
        return write_immutable_bytes(path, stream.getvalue(), paper2_root=paper2_root)

    @classmethod
    def read(cls, path: str | Path) -> "AuditFootprintObservations":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("schema") != cls.SCHEMA:
                raise ValueError("unsupported audit observation schema")
            if metadata.get("clean_targets_included") is not False or metadata.get(
                "reference_identifier_included"
            ) is not False:
                raise ValueError("audit observation evaluator boundary differs")
            result = cls(
                tile_id=str(metadata["tile_id"]),
                audit_plan_file_sha256=str(metadata["audit_plan_file_sha256"]),
                audit_plan_contract_sha256=str(metadata["audit_plan_contract_sha256"]),
                replicate=int(metadata["replicate"]),
                audit_ids=np.asarray(archive["audit_ids"], dtype=str),
                return_values_m=np.asarray(archive["return_values_m"], dtype=np.float64),
                noise_std_m=float(metadata["noise_std_m"]),
            )
        result.validate()
        return result


def _supported_projection(operator: Any, surfaces: np.ndarray) -> np.ndarray:
    paths = np.asarray(surfaces, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("full DEM path stack must have shape (M, rows, columns)")
    matrix = operator.tocsr()
    flattened = paths.reshape(paths.shape[0], -1)
    used = np.unique(matrix.indices)
    if not np.all(np.isfinite(flattened[:, used])):
        raise ValueError("full DEM paths are non-finite on audit footprint support")
    safe = np.where(np.isfinite(flattened), flattened, 0.0)
    return np.asarray((matrix @ safe.T).T, dtype=np.float64)


def build_audit_projection_payload(
    *,
    audit_plan_path: str | Path,
    audit_plan: AuditFootprintPlan,
    operator: Any,
    full_dem_paths: np.ndarray,
    matrix_key_identifier: str,
    generative: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    projections = _supported_projection(operator, full_dem_paths)
    if projections.shape[1] != AUDIT_FOOTPRINT_COUNT:
        raise ValueError("audit path projection count differs")
    if generative and projections.shape[0] < 2:
        raise ValueError("generative audit projection requires at least two complete paths")
    mean = projections.mean(axis=0)
    variance = projections.var(axis=0, ddof=1) if projections.shape[0] >= 2 else np.zeros(128)
    metadata: dict[str, Any] = {
        "schema": "paper2.audit-footprint-path-projections/v1",
        "tile_id": audit_plan.tile_id,
        "matrix_key_identifier": matrix_key_identifier,
        "audit_plan_file_sha256": file_sha256(audit_plan_path),
        "audit_plan_contract_sha256": audit_plan.sha256,
        "path_count": int(projections.shape[0]),
        "footprint_count": AUDIT_FOOTPRINT_COUNT,
        "projection_order": "after_global_stitch_and_terminal_dc",
        "variance_source": (
            "complete_dem_path_ensemble_spatial_covariance"
            if generative
            else "single_complete_dem_point_prediction"
        ),
        "diagonal_pixel_variance_projection": False,
        "audit_returns_read": False,
        "clean_reference_read": False,
        "path_projection_array_sha256": array_sha256(projections.astype(np.float64)),
        "mean_array_sha256": array_sha256(mean.astype(np.float64)),
        "sample_variance_array_sha256": array_sha256(variance.astype(np.float64)),
    }
    metadata["projection_contract_sha256"] = canonical_sha256(metadata)
    return metadata, projections, mean, variance


def write_audit_path_projections(
    path: str | Path,
    *,
    audit_plan_path: str | Path,
    audit_plan: AuditFootprintPlan,
    operator: Any,
    full_dem_paths: np.ndarray,
    matrix_key_identifier: str,
    generative: bool,
    paper2_root: str | Path,
) -> Path:
    metadata, projections, mean, variance = build_audit_projection_payload(
        audit_plan_path=audit_plan_path,
        audit_plan=audit_plan,
        operator=operator,
        full_dem_paths=full_dem_paths,
        matrix_key_identifier=matrix_key_identifier,
        generative=generative,
    )
    stream = BytesIO()
    np.savez_compressed(
        stream,
        metadata=np.asarray(canonical_json_bytes(metadata).decode("utf-8")),
        path_projections_m=projections.astype(np.float64),
        mean_m=mean.astype(np.float64),
        sample_variance_m2=variance.astype(np.float64),
    )
    return write_immutable_bytes(path, stream.getvalue(), paper2_root=paper2_root)


def write_audit_projection_prefix(
    path: str | Path,
    *,
    source_projection_path: str | Path,
    prefix_count: int,
    matrix_key_identifier: str,
    paper2_root: str | Path,
) -> Path:
    """Derive an immutable ordered M-prefix from an E7 M=32 projection stack."""

    source_metadata, source_values, _, _ = read_audit_path_projections(
        source_projection_path
    )
    if len(source_values) != 32 or prefix_count not in {2, 4, 8, 16}:
        raise ValueError("audit projection prefix requires M in {2,4,8,16} from M=32")
    projections = np.asarray(source_values[:prefix_count], dtype=np.float64)
    mean = projections.mean(axis=0)
    variance = projections.var(axis=0, ddof=1)
    metadata: dict[str, Any] = {
        key: value
        for key, value in source_metadata.items()
        if key
        not in {
            "projection_contract_sha256",
            "matrix_key_identifier",
            "path_count",
            "path_projection_array_sha256",
            "mean_array_sha256",
            "sample_variance_array_sha256",
        }
    }
    metadata.update(
        {
            "matrix_key_identifier": matrix_key_identifier,
            "path_count": prefix_count,
            "path_projection_array_sha256": array_sha256(projections),
            "mean_array_sha256": array_sha256(mean),
            "sample_variance_array_sha256": array_sha256(variance),
            "strict_ordered_prefix": True,
            "source_projection_sha256": file_sha256(source_projection_path),
            "source_path_count": 32,
        }
    )
    metadata["projection_contract_sha256"] = canonical_sha256(metadata)
    stream = BytesIO()
    np.savez_compressed(
        stream,
        metadata=np.asarray(canonical_json_bytes(metadata).decode("utf-8")),
        path_projections_m=projections,
        mean_m=mean,
        sample_variance_m2=variance,
    )
    return write_immutable_bytes(path, stream.getvalue(), paper2_root=paper2_root)


def read_audit_path_projections(path: str | Path) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        projections = np.asarray(archive["path_projections_m"], dtype=np.float64)
        mean = np.asarray(archive["mean_m"], dtype=np.float64)
        variance = np.asarray(archive["sample_variance_m2"], dtype=np.float64)
    expected = canonical_sha256(
        {name: value for name, value in metadata.items() if name != "projection_contract_sha256"}
    )
    if (
        metadata.get("schema") != "paper2.audit-footprint-path-projections/v1"
        or metadata.get("projection_contract_sha256") != expected
        or metadata.get("projection_order") != "after_global_stitch_and_terminal_dc"
        or metadata.get("diagonal_pixel_variance_projection") is not False
        or metadata.get("audit_returns_read") is not False
        or metadata.get("clean_reference_read") is not False
    ):
        raise ValueError("audit path projection contract differs")
    if projections.ndim != 2 or projections.shape[1] != AUDIT_FOOTPRINT_COUNT:
        raise ValueError("audit path projection array shape differs")
    if mean.shape != (AUDIT_FOOTPRINT_COUNT,) or variance.shape != (AUDIT_FOOTPRINT_COUNT,):
        raise ValueError("audit path projection summary shape differs")
    if not np.all(np.isfinite(projections)) or np.any(variance < 0):
        raise ValueError("audit path projections are non-finite or negative")
    if not np.allclose(mean, projections.mean(axis=0), rtol=0.0, atol=1e-12):
        raise ValueError("audit projection mean does not reproduce from paths")
    expected_variance = projections.var(axis=0, ddof=1) if len(projections) >= 2 else np.zeros(128)
    if not np.allclose(variance, expected_variance, rtol=0.0, atol=1e-12):
        raise ValueError("audit projection variance does not reproduce from paths")
    for field, value in (
        ("path_projection_array_sha256", projections),
        ("mean_array_sha256", mean),
        ("sample_variance_array_sha256", variance),
    ):
        if metadata.get(field) != array_sha256(value.astype(np.float64)):
            raise ValueError(f"audit projection {field} differs")
    if int(metadata.get("path_count", -1)) != len(projections):
        raise ValueError("audit projection path count differs")
    if metadata.get("strict_ordered_prefix") is True:
        if (
            int(metadata.get("source_path_count", -1)) != 32
            or not is_sha256(str(metadata.get("source_projection_sha256", "")))
            or len(projections) not in {2, 4, 8, 16}
        ):
            raise ValueError("audit projection prefix provenance differs")
    return metadata, projections, mean, variance


def simulate_audit_footprint_observations(
    *,
    audit_plan_path: str | Path,
    reference_path: str | Path,
    replicate: int,
    noise_std_m: float,
) -> AuditFootprintObservations:
    audit_plan = read_audit_footprint_plan(audit_plan_path)
    if replicate < 0 or not np.isfinite(noise_std_m) or noise_std_m <= 0:
        raise ValueError("audit replicate/noise is invalid")
    with np.load(reference_path, allow_pickle=False) as archive:
        dem = np.asarray(archive["dem"], dtype=np.float64)
        valid = np.asarray(archive["valid_mask"], dtype=bool)
        if "tile_id" in archive.files and str(archive["tile_id"].item()) != audit_plan.tile_id:
            raise ValueError("audit plan and reference tile differ")
    if dem.shape != audit_plan.grid_shape or valid.shape != audit_plan.grid_shape:
        raise ValueError("audit plan and reference grid shape differ")
    if not np.all(np.isfinite(dem[valid])):
        raise ValueError("reference is non-finite on its valid mask")
    centers = np.asarray([(point.x, point.y) for point in audit_plan.points])
    operator = build_footprint_operator_with_metadata(
        centers,
        grid_shape=dem.shape,
        transform=audit_plan.transform,
        diameter_m=audit_plan.footprint.diameter_m,
        quadrature_points=audit_plan.footprint.quadrature_points,
        valid_mask=valid,
        require_full_support=True,
        drop_invalid=False,
    ).operator
    clean = np.asarray(operator @ np.where(valid, dem, 0.0).ravel(), dtype=np.float64)
    noisy = np.empty(AUDIT_FOOTPRINT_COUNT, dtype=np.float64)
    for index, point in enumerate(audit_plan.points):
        digest = canonical_sha256(
            {
                "protocol": "paper2-audit-candidate-paired-noise/v1",
                "seed_sha256": audit_plan.heldout_noise_seed_sha256,
                "replicate": replicate,
                "tile_id": audit_plan.tile_id,
                "audit_id": point.audit_id,
            }
        )
        generator = np.random.Generator(np.random.PCG64(int(digest[:16], 16)))
        noisy[index] = clean[index] + generator.normal(0.0, noise_std_m)
    result = AuditFootprintObservations(
        tile_id=audit_plan.tile_id,
        audit_plan_file_sha256=file_sha256(audit_plan_path),
        audit_plan_contract_sha256=audit_plan.sha256,
        replicate=replicate,
        audit_ids=np.asarray([point.audit_id for point in audit_plan.points]),
        return_values_m=noisy,
        noise_std_m=float(noise_std_m),
    )
    result.validate()
    return result
