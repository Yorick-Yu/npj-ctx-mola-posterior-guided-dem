from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import hashlib

import numpy as np

from .contracts import PlanContract
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)


def reference_grid_sha256(
    *,
    tile_id: str,
    shape: Sequence[int],
    crs_wkt: str,
    transform: Sequence[float],
) -> str:
    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError("sealed reference shape must be two positive dimensions")
    if len(transform) != 6 or not all(np.isfinite(float(value)) for value in transform):
        raise ValueError("sealed reference transform must contain six finite values")
    if not tile_id or not crs_wkt.strip():
        raise ValueError("sealed reference tile/CRS cannot be empty")
    return canonical_sha256(
        {
            "tile_id": tile_id,
            "shape": [int(value) for value in shape],
            "crs_wkt": crs_wkt,
            "transform_hex": [float(value).hex() for value in transform],
        }
    )


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_sealed_reference_lock(
    entries: Sequence[Mapping[str, Any]], *, paper2_root: str | Path
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    records: list[dict[str, Any]] = []
    for raw in entries:
        tile_id = str(raw["tile_id"])
        reference_path = guard_paper2_path(raw["reference"], paper2_root=root)
        plan_path = guard_paper2_path(raw["plan"], paper2_root=root)
        if not reference_path.is_file() or not plan_path.is_file():
            raise FileNotFoundError(reference_path if not reference_path.is_file() else plan_path)
        plan = PlanContract.read(plan_path)
        if plan.tile_id != tile_id:
            raise ValueError("sealed reference entry tile differs from its grid plan")
        with np.load(reference_path, allow_pickle=False) as archive:
            if "dem" not in archive.files or "valid_mask" not in archive.files:
                raise ValueError("sealed reference requires dem and valid_mask arrays")
            dem = np.asarray(archive["dem"], dtype=np.float64)
            valid = np.asarray(archive["valid_mask"], dtype=bool)
        if dem.ndim != 2 or valid.shape != dem.shape or not np.any(valid):
            raise ValueError("sealed reference DEM/mask grid is invalid")
        if not np.all(np.isfinite(dem[valid])):
            raise ValueError("sealed reference DEM is non-finite over valid pixels")
        records.append(
            {
                "tile_id": tile_id,
                "reference_relative_path": reference_path.relative_to(root).as_posix(),
                "reference_sha256": file_sha256(reference_path),
                "dem_array_sha256": _array_sha256(dem),
                "valid_mask_array_sha256": _array_sha256(valid),
                "valid_fraction": float(np.mean(valid)),
                "shape": [int(value) for value in dem.shape],
                "crs_wkt": plan.crs_wkt,
                "transform": [float(value) for value in plan.transform],
                "grid_sha256": reference_grid_sha256(
                    tile_id=tile_id,
                    shape=dem.shape,
                    crs_wkt=plan.crs_wkt,
                    transform=plan.transform,
                ),
                "grid_plan_relative_path": plan_path.relative_to(root).as_posix(),
                "grid_plan_sha256": file_sha256(plan_path),
                "problem_sha256": plan.problem_sha256,
            }
        )
    records.sort(key=lambda value: value["tile_id"])
    tile_ids = [record["tile_id"] for record in records]
    if not records or len(tile_ids) != len(set(tile_ids)):
        raise ValueError("sealed reference lock requires unique nonempty tile entries")
    payload: dict[str, Any] = {
        "schema": "paper2.sealed-reference-lock/v1",
        "evaluator_only": True,
        "reference_paths_forbidden_in_problem_plan_inference": True,
        "entries": records,
    }
    payload["sealed_reference_lock_sha256"] = canonical_sha256(payload)
    return payload


def read_sealed_reference_lock(
    path: str | Path,
    *,
    paper2_root: str | Path,
    verify_files: bool = True,
) -> Mapping[str, Any]:
    root = Path(paper2_root).resolve()
    lock_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(lock_path)
    if not isinstance(payload, Mapping):
        raise ValueError("sealed reference lock must be a JSON object")
    expected = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "sealed_reference_lock_sha256"
        }
    )
    if (
        payload.get("schema") != "paper2.sealed-reference-lock/v1"
        or payload.get("sealed_reference_lock_sha256") != expected
        or payload.get("evaluator_only") is not True
        or payload.get("reference_paths_forbidden_in_problem_plan_inference") is not True
    ):
        raise ValueError("sealed reference lock identity/hash/boundary differs")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("sealed reference lock has no entries")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("sealed reference entry is malformed")
        tile_id = str(entry.get("tile_id", ""))
        if not tile_id or tile_id in seen:
            raise ValueError("sealed reference tile IDs must be unique")
        seen.add(tile_id)
        for name in (
            "reference_sha256",
            "dem_array_sha256",
            "valid_mask_array_sha256",
            "grid_sha256",
            "grid_plan_sha256",
            "problem_sha256",
        ):
            if not is_sha256(str(entry.get(name, ""))):
                raise ValueError(f"sealed reference {name} is invalid")
        observed_grid = reference_grid_sha256(
            tile_id=tile_id,
            shape=entry["shape"],
            crs_wkt=str(entry["crs_wkt"]),
            transform=entry["transform"],
        )
        if observed_grid != entry["grid_sha256"]:
            raise ValueError("sealed reference grid contract hash differs")
        if verify_files:
            reference_path = guard_paper2_path(
                root / str(entry["reference_relative_path"]), paper2_root=root
            )
            plan_path = guard_paper2_path(
                root / str(entry["grid_plan_relative_path"]), paper2_root=root
            )
            if not reference_path.is_file() or file_sha256(reference_path) != entry[
                "reference_sha256"
            ]:
                raise ValueError("sealed reference bytes differ from lock")
            if not plan_path.is_file() or file_sha256(plan_path) != entry["grid_plan_sha256"]:
                raise ValueError("sealed reference grid-plan bytes differ from lock")
    return payload


def reference_hashes_by_tile(lock: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {
        str(entry["tile_id"]): {
            "sealed_reference_sha256": str(entry["reference_sha256"]),
            "sealed_reference_grid_sha256": str(entry["grid_sha256"]),
        }
        for entry in lock["entries"]
    }


def write_sealed_reference_lock(
    path: str | Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    paper2_root: str | Path,
) -> Path:
    return write_canonical_json(
        path,
        build_sealed_reference_lock(entries, paper2_root=paper2_root),
        paper2_root=paper2_root,
    )
