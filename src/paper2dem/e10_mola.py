from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import hashlib

import numpy as np

from .contracts import ReconstructionProblem
from .g0 import validate_mola_source_lock
from .g0_raster_preparation import (
    COARSE_DEM_ALGORITHM,
    COARSE_DEM_LINEAGE_SCHEMA,
)
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)


E10_PROBLEM_METADATA_SCHEMA = "paper2.e10-mola-problem-metadata/v1"
E10_TILE_MANIFEST_SCHEMA = "paper2.e10-mola-problem/v1"
E10_AUDIT_SCHEMA = "paper2.e10-mola-audit/v1"
MOLA_PRODUCT_ID = "MEGT90N000FB"


def array_sha256(values: np.ndarray) -> str:
    """Hash an array without depending on an NPZ container's compression bytes."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _content_hash(payload: Mapping[str, Any], field: str = "content_sha256") -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != field})


def _verified_record(
    record: Mapping[str, Any], *, root: Path, expected_path: Path | None = None
) -> Path:
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("MOLA lineage contains an invalid relative artifact path")
    path = guard_paper2_path(relative, paper2_root=root)
    if expected_path is not None and path != expected_path:
        raise ValueError("MOLA lineage artifact path differs from the requested input")
    if (
        not path.is_file()
        or not is_sha256(str(record.get("sha256", "")))
        or file_sha256(path) != record.get("sha256")
        or int(record.get("byte_count", -1)) != path.stat().st_size
    ):
        raise ValueError(f"MOLA lineage artifact bytes differ: {path}")
    return path


def _load_npy(path: Path, *, dtype: Any) -> np.ndarray:
    with path.open("rb") as stream:
        values = np.load(stream, allow_pickle=False)
    if isinstance(values, np.lib.npyio.NpzFile):  # pragma: no cover - defensive
        values.close()
        raise ValueError("E10 G0 arrays must be single-array NPY artifacts")
    return np.asarray(values, dtype=dtype)


def verify_mola_tile_inputs(
    *,
    tile_id: str,
    coarse_dem_path: str | Path,
    dem_mask_path: str | Path,
    coarse_lineage_path: str | Path,
    mola_source_lock_path: str | Path,
    paper2_root: str | Path,
) -> dict[str, Any]:
    """Verify that one target-grid Z0 is derived only from the locked MOLA MEGT."""
    root = Path(paper2_root).resolve()
    coarse_path = guard_paper2_path(coarse_dem_path, paper2_root=root)
    mask_path = guard_paper2_path(dem_mask_path, paper2_root=root)
    lineage_path = guard_paper2_path(coarse_lineage_path, paper2_root=root)
    source_lock_path = guard_paper2_path(mola_source_lock_path, paper2_root=root)
    for path in (coarse_path, mask_path, lineage_path, source_lock_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_lock = read_json(source_lock_path)
    if not isinstance(source_lock, Mapping) or not validate_mola_source_lock(source_lock):
        raise ValueError("E10 requires the fully downloaded and byte-locked MOLA source lock")
    lineage = read_json(lineage_path)
    if not isinstance(lineage, Mapping):
        raise ValueError("MOLA coarse lineage must be a JSON object")
    if (
        lineage.get("schema") != COARSE_DEM_LINEAGE_SCHEMA
        or lineage.get("state") != "complete"
        or lineage.get("algorithm") != COARSE_DEM_ALGORITHM
        or lineage.get("product_id") != tile_id
        or lineage.get("reference_dem_accessed") is not False
        or lineage.get("content_sha256") != _content_hash(lineage)
    ):
        raise ValueError("MOLA coarse lineage identity/state/hash differs")
    mola_source = lineage.get("mola_source")
    if not isinstance(mola_source, Mapping) or mola_source.get("product_id") != MOLA_PRODUCT_ID:
        raise ValueError("E10 coarse lineage does not use the frozen MOLA MEGT product")
    products = {
        str(product.get("product_id")): product for product in source_lock.get("products", [])
    }
    source_product = products.get(MOLA_PRODUCT_ID)
    if not isinstance(source_product, Mapping) or (
        mola_source.get("image_sha256") != source_product.get("image_sha256")
        or mola_source.get("label_sha256") != source_product.get("label_sha256")
    ):
        raise ValueError("MOLA coarse lineage source hashes differ from the source lock")
    artifacts = lineage.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("MOLA coarse lineage lacks artifact bindings")
    for name in ("mola_source_lock", "coarse_dem", "dem_mask", "problem"):
        if not isinstance(artifacts.get(name), Mapping):
            raise ValueError(f"MOLA coarse lineage lacks {name}")
    _verified_record(
        artifacts["mola_source_lock"], root=root, expected_path=source_lock_path
    )
    _verified_record(artifacts["coarse_dem"], root=root, expected_path=coarse_path)
    _verified_record(artifacts["dem_mask"], root=root, expected_path=mask_path)
    _verified_record(artifacts["problem"], root=root)

    coarse = _load_npy(coarse_path, dtype=np.float32)
    mask = _load_npy(mask_path, dtype=bool)
    target_grid = lineage.get("target_grid")
    if not isinstance(target_grid, Mapping):
        raise ValueError("MOLA coarse lineage lacks its target-grid contract")
    expected_shape = tuple(int(value) for value in target_grid.get("shape", []))
    if (
        coarse.ndim != 2
        or mask.shape != coarse.shape
        or expected_shape != coarse.shape
        or not np.any(mask)
        or not np.array_equal(mask, np.isfinite(coarse))
    ):
        raise ValueError("MOLA coarse DEM/mask/target-grid shapes or finite support differ")
    if float(np.mean(mask)) < 0.99:
        raise ValueError("E10 MOLA coarse DEM fails the frozen 99% finite-support threshold")
    return {
        "coarse_dem_relative_path": coarse_path.relative_to(root).as_posix(),
        "dem_mask_relative_path": mask_path.relative_to(root).as_posix(),
        "coarse_lineage_relative_path": lineage_path.relative_to(root).as_posix(),
        "mola_source_lock_relative_path": source_lock_path.relative_to(root).as_posix(),
        "coarse_dem": coarse,
        "dem_mask": mask,
        "coarse_dem_sha256": file_sha256(coarse_path),
        "dem_mask_sha256": file_sha256(mask_path),
        "coarse_dem_array_sha256": array_sha256(coarse),
        "dem_mask_array_sha256": array_sha256(mask),
        "coarse_lineage_sha256": file_sha256(lineage_path),
        "coarse_lineage_content_sha256": str(lineage["content_sha256"]),
        "mola_source_lock_sha256": file_sha256(source_lock_path),
        "mola_image_sha256": str(source_product["image_sha256"]),
        "mola_label_sha256": str(source_product["label_sha256"]),
        "target_grid_sha256": canonical_sha256(target_grid),
        "shape": list(coarse.shape),
        "finite_fraction": float(np.mean(mask)),
    }


def build_e10_mola_problem(
    *,
    e3_problem_path: str | Path,
    coarse_dem_path: str | Path,
    dem_mask_path: str | Path,
    coarse_lineage_path: str | Path,
    mola_source_lock_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    paper2_root: str | Path,
) -> dict[str, Any]:
    """Create a truth-blind E10 problem while preserving observations and imagery."""
    root = Path(paper2_root).resolve()
    source_path = guard_paper2_path(e3_problem_path, paper2_root=root)
    output = guard_paper2_path(output_path, paper2_root=root)
    manifest_file = guard_paper2_path(manifest_path, paper2_root=root)
    if output == source_path:
        raise ValueError("E10 must use a different problem file from E3")
    source = ReconstructionProblem.read(source_path)
    verified = verify_mola_tile_inputs(
        tile_id=source.tile_id,
        coarse_dem_path=coarse_dem_path,
        dem_mask_path=dem_mask_path,
        coarse_lineage_path=coarse_lineage_path,
        mola_source_lock_path=mola_source_lock_path,
        paper2_root=root,
    )
    coarse = np.asarray(verified.pop("coarse_dem"), dtype=np.float32)
    mola_mask = np.asarray(verified.pop("dem_mask"), dtype=bool)
    if coarse.shape != source.z0.shape:
        raise ValueError("E10 MOLA Z0 shape differs from the E3 reconstruction grid")
    e3_z0_hash = array_sha256(source.z0)
    e10_z0_hash = array_sha256(coarse)
    if e3_z0_hash == e10_z0_hash or np.array_equal(source.z0, coarse, equal_nan=True):
        raise ValueError("E3 and E10 use the same Z0; independent-MOLA E10 is not identified")
    valid = source.valid_mask & mola_mask & np.isfinite(coarse)
    if not np.any(valid):
        raise ValueError("E10 source and MOLA masks have no common valid support")
    metadata = {
        **dict(source.metadata),
        "e10_contract_schema": E10_PROBLEM_METADATA_SCHEMA,
        "experiment": "E10",
        "z0_source": "locked-independent-mola-megt",
        "e3_source_problem_sha256": file_sha256(source_path),
        "e3_z0_array_sha256": e3_z0_hash,
        "e10_z0_array_sha256": e10_z0_hash,
        "z0_distinct_from_e3": True,
        "valid_mask_rule": "e3-valid-and-mola-finite",
        **verified,
    }
    frozen = dict(source.frozen_model_hashes)
    frozen.update(
        {
            "e10_mola_source_lock": str(verified["mola_source_lock_sha256"]),
            "e10_mola_coarse_lineage": str(verified["coarse_lineage_sha256"]),
            "e10_mola_coarse_array": str(verified["coarse_dem_array_sha256"]),
            "e10_mola_mask_array": str(verified["dem_mask_array_sha256"]),
        }
    )
    problem = ReconstructionProblem(
        tile_id=source.tile_id,
        z0=coarse,
        image=source.image,
        valid_mask=valid,
        observation_candidate_ids=source.observation_candidate_ids,
        observation_coordinates=source.observation_coordinates,
        observation_values_m=source.observation_values_m,
        observation_quality_flags=source.observation_quality_flags,
        plan_sha256=source.plan_sha256,
        observations_sha256=source.observations_sha256,
        frozen_model_hashes=tuple(sorted(frozen.items())),
        metadata=metadata,
    )
    written = problem.write(output, paper2_root=root)
    if file_sha256(written) == file_sha256(source_path):
        raise AssertionError("E10 and E3 problem container hashes unexpectedly match")
    payload: dict[str, Any] = {
        "schema": E10_TILE_MANIFEST_SCHEMA,
        "status": "complete",
        "tile_id": source.tile_id,
        "truth_data_opened": False,
        "e3_problem_relative_path": source_path.relative_to(root).as_posix(),
        "e3_problem_sha256": file_sha256(source_path),
        "e10_problem_relative_path": written.relative_to(root).as_posix(),
        "e10_problem_sha256": file_sha256(written),
        "e3_z0_array_sha256": e3_z0_hash,
        "e10_z0_array_sha256": e10_z0_hash,
        "z0_distinct_from_e3": True,
        "mola": verified,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    write_canonical_json(manifest_file, payload, paper2_root=root)
    return read_e10_tile_manifest(manifest_file, paper2_root=root)


def read_e10_tile_manifest(
    path: str | Path, *, paper2_root: str | Path, verify_problem: bool = True
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    manifest_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(manifest_path)
    if not isinstance(payload, Mapping):
        raise ValueError("E10 tile manifest must be a JSON object")
    if (
        payload.get("schema") != E10_TILE_MANIFEST_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("truth_data_opened") is not False
        or payload.get("z0_distinct_from_e3") is not True
        or payload.get("content_sha256") != _content_hash(payload)
    ):
        raise ValueError("E10 tile manifest identity/state/hash differs")
    source_path = guard_paper2_path(
        str(payload.get("e3_problem_relative_path", "")), paper2_root=root
    )
    e10_path = guard_paper2_path(
        str(payload.get("e10_problem_relative_path", "")), paper2_root=root
    )
    if source_path == e10_path:
        raise ValueError("E10 and E3 problem paths are identical")
    if (
        not source_path.is_file()
        or file_sha256(source_path) != payload.get("e3_problem_sha256")
        or not e10_path.is_file()
        or file_sha256(e10_path) != payload.get("e10_problem_sha256")
        or payload.get("e3_problem_sha256") == payload.get("e10_problem_sha256")
        or payload.get("e3_z0_array_sha256") == payload.get("e10_z0_array_sha256")
    ):
        raise ValueError("E10/E3 problem bytes or Z0 identity differ from the manifest")
    if verify_problem:
        source = ReconstructionProblem.read(source_path)
        e10 = ReconstructionProblem.read(e10_path)
        mola = payload.get("mola")
        if not isinstance(mola, Mapping):
            raise ValueError("E10 tile manifest lacks MOLA input lineage")
        reverified = verify_mola_tile_inputs(
            tile_id=str(payload["tile_id"]),
            coarse_dem_path=str(mola.get("coarse_dem_relative_path", "")),
            dem_mask_path=str(mola.get("dem_mask_relative_path", "")),
            coarse_lineage_path=str(mola.get("coarse_lineage_relative_path", "")),
            mola_source_lock_path=str(mola.get("mola_source_lock_relative_path", "")),
            paper2_root=root,
        )
        reverified_coarse = np.asarray(reverified.pop("coarse_dem"), dtype=np.float32)
        reverified.pop("dem_mask")
        if dict(mola) != reverified:
            raise ValueError("E10 tile manifest MOLA lineage does not reverify")
        if source.tile_id != e10.tile_id or source.tile_id != payload.get("tile_id"):
            raise ValueError("E10/E3 tile identities differ")
        if (
            array_sha256(source.z0) != payload.get("e3_z0_array_sha256")
            or array_sha256(e10.z0) != payload.get("e10_z0_array_sha256")
            or e10.metadata.get("z0_source") != "locked-independent-mola-megt"
            or e10.metadata.get("mola_source_lock_sha256")
            != dict(payload.get("mola", {})).get("mola_source_lock_sha256")
            or e10.metadata.get("coarse_lineage_sha256")
            != dict(payload.get("mola", {})).get("coarse_lineage_sha256")
            or not np.array_equal(e10.z0, reverified_coarse, equal_nan=True)
        ):
            raise ValueError("E10 problem contents do not reproduce the MOLA manifest")
    return dict(payload)


def build_e10_mola_audit(
    *,
    tile_manifest_paths: Sequence[str | Path],
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    paper2_root: str | Path,
    expected_tile_count: int = 24,
    expected_manifests_per_tile: int = 1,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
    asset_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    roles = read_json(role_path)
    if not isinstance(roles, Mapping) or roles.get("schema") != "paper2.role-manifest/v1":
        raise ValueError("E10 requires the formal role manifest")
    expected_tiles = tuple(sorted(str(value) for value in roles.get("roles", {}).get(
        "paper2_locked_test", []
    )))
    if len(expected_tiles) != expected_tile_count or len(set(expected_tiles)) != len(
        expected_tiles
    ):
        raise ValueError(f"E10 requires exactly {expected_tile_count} locked CTX tiles")
    if expected_manifests_per_tile <= 0:
        raise ValueError("E10 expected manifests per tile must be positive")
    if roles.get("asset_lock_sha256") != file_sha256(asset_path):
        raise ValueError("E10 role manifest binds a different asset lock")
    records: list[dict[str, Any]] = []
    observed_tiles: list[str] = []
    observed_paths: set[Path] = set()
    for raw_path in tile_manifest_paths:
        path = guard_paper2_path(raw_path, paper2_root=root)
        if path in observed_paths:
            raise ValueError("E10 audit contains a duplicate cell manifest path")
        observed_paths.add(path)
        manifest = read_e10_tile_manifest(path, paper2_root=root)
        tile_id = str(manifest["tile_id"])
        observed_tiles.append(tile_id)
        records.append(
            {
                "tile_id": tile_id,
                "tile_manifest_relative_path": path.relative_to(root).as_posix(),
                "tile_manifest_sha256": file_sha256(path),
                "e3_problem_sha256": manifest["e3_problem_sha256"],
                "e10_problem_sha256": manifest["e10_problem_sha256"],
                "e3_z0_array_sha256": manifest["e3_z0_array_sha256"],
                "e10_z0_array_sha256": manifest["e10_z0_array_sha256"],
                "mola_source_lock_sha256": manifest["mola"]["mola_source_lock_sha256"],
                "mola_coarse_lineage_sha256": manifest["mola"][
                    "coarse_lineage_sha256"
                ],
                "shape": manifest["mola"]["shape"],
                "finite_fraction": manifest["mola"]["finite_fraction"],
            }
        )
    counts = {
        tile_id: observed_tiles.count(tile_id) for tile_id in sorted(set(observed_tiles))
    }
    if set(counts) != set(expected_tiles) or any(
        count != expected_manifests_per_tile for count in counts.values()
    ):
        raise ValueError(
            "E10 cell manifests do not exactly cover every locked CTX tile"
        )
    source_locks = {record["mola_source_lock_sha256"] for record in records}
    if len(source_locks) != 1:
        raise ValueError("E10 tiles do not share one locked MOLA source release")
    for tile_id in expected_tiles:
        tile_records = [record for record in records if record["tile_id"] == tile_id]
        for field in (
            "mola_source_lock_sha256",
            "mola_coarse_lineage_sha256",
            "e10_z0_array_sha256",
            "shape",
            "finite_fraction",
        ):
            if len({canonical_sha256(record[field]) for record in tile_records}) != 1:
                raise ValueError(f"E10 tile {tile_id} does not share one MOLA Z0: {field}")
    payload: dict[str, Any] = {
        "schema": E10_AUDIT_SCHEMA,
        "status": "complete",
        "experiment": "E10",
        "expected_tile_count": expected_tile_count,
        "expected_manifests_per_tile": expected_manifests_per_tile,
        "audited_tile_count": len(counts),
        "audited_manifest_count": len(records),
        "role_manifest_sha256": file_sha256(role_path),
        "asset_lock_sha256": file_sha256(asset_path),
        "role_manifest": {
            "relative_path": role_path.relative_to(root).as_posix(),
            "sha256": file_sha256(role_path),
            "byte_count": role_path.stat().st_size,
        },
        "asset_lock": {
            "relative_path": asset_path.relative_to(root).as_posix(),
            "sha256": file_sha256(asset_path),
            "byte_count": asset_path.stat().st_size,
        },
        "mola_source_lock_sha256": next(iter(source_locks)),
        "all_z0_from_locked_mola": True,
        "all_e3_e10_problem_and_z0_distinct": True,
        "truth_data_opened": False,
        "tiles": sorted(records, key=lambda value: value["tile_id"]),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def write_e10_mola_audit(
    path: str | Path,
    *,
    tile_manifest_paths: Sequence[str | Path],
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    paper2_root: str | Path,
    expected_tile_count: int = 24,
    expected_manifests_per_tile: int = 1,
) -> Path:
    return write_canonical_json(
        path,
        build_e10_mola_audit(
            tile_manifest_paths=tile_manifest_paths,
            role_manifest_path=role_manifest_path,
            asset_lock_path=asset_lock_path,
            paper2_root=paper2_root,
            expected_tile_count=expected_tile_count,
            expected_manifests_per_tile=expected_manifests_per_tile,
        ),
        paper2_root=paper2_root,
    )


def read_e10_mola_audit(
    path: str | Path, *, paper2_root: str | Path, verify_tiles: bool = True
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    audit_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(audit_path)
    if not isinstance(payload, Mapping):
        raise ValueError("E10 audit must be a JSON object")
    if (
        payload.get("schema") != E10_AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("all_z0_from_locked_mola") is not True
        or payload.get("all_e3_e10_problem_and_z0_distinct") is not True
        or payload.get("truth_data_opened") is not False
        or payload.get("content_sha256") != _content_hash(payload)
    ):
        raise ValueError("E10 audit identity/state/hash differs")
    records = payload.get("tiles")
    expected_count = int(payload.get("expected_tile_count", -1))
    expected_per_tile = int(payload.get("expected_manifests_per_tile", 1))
    if (
        not isinstance(records, list)
        or len(records) != expected_count * expected_per_tile
        or int(payload.get("audited_tile_count", -1)) != expected_count
        or int(payload.get("audited_manifest_count", len(records))) != len(records)
    ):
        raise ValueError("E10 audit tile/cell-manifest counts do not close")
    tile_ids = [str(record.get("tile_id", "")) for record in records]
    counts = {tile_id: tile_ids.count(tile_id) for tile_id in set(tile_ids)}
    if not all(tile_ids) or any(value != expected_per_tile for value in counts.values()):
        raise ValueError("E10 audit per-tile cell-manifest multiplicity differs")
    role_path = _verified_record(dict(payload.get("role_manifest", {})), root=root)
    asset_path = _verified_record(dict(payload.get("asset_lock", {})), root=root)
    roles = read_json(role_path)
    if (
        not isinstance(roles, Mapping)
        or roles.get("schema") != "paper2.role-manifest/v1"
        or roles.get("asset_lock_sha256") != file_sha256(asset_path)
        or payload.get("role_manifest_sha256") != file_sha256(role_path)
        or payload.get("asset_lock_sha256") != file_sha256(asset_path)
    ):
        raise ValueError("E10 audit role/asset parent binding differs")
    locked_tiles = sorted(
        str(value) for value in roles.get("roles", {}).get("paper2_locked_test", [])
    )
    if locked_tiles != sorted(counts) or len(locked_tiles) != expected_count:
        raise ValueError("E10 audit does not exactly cover the frozen locked-test role")
    if verify_tiles:
        for record in records:
            manifest_path = guard_paper2_path(
                str(record.get("tile_manifest_relative_path", "")), paper2_root=root
            )
            if (
                not manifest_path.is_file()
                or file_sha256(manifest_path) != record.get("tile_manifest_sha256")
            ):
                raise ValueError("E10 audit tile-manifest bytes differ")
            tile = read_e10_tile_manifest(manifest_path, paper2_root=root)
            for field in (
                "e3_problem_sha256",
                "e10_problem_sha256",
                "e3_z0_array_sha256",
                "e10_z0_array_sha256",
            ):
                if record.get(field) != tile.get(field):
                    raise ValueError(f"E10 audit tile {field} differs")
    return dict(payload)
