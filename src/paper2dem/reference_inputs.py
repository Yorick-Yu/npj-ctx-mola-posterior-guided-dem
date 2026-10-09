"""Composition and verification of evaluator-only sealed-reference inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import PlanContract
from .hashing import canonical_sha256, file_sha256, guard_paper2_path, read_json
from .inference_preparation import SEALED_REFERENCE_INPUT_SCHEMA


def _relative_file(value: Any, *, root: Path, field: str) -> tuple[str, Path]:
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"sealed reference {field} must be a relative Paper2 path")
    path = guard_paper2_path(root / relative, paper2_root=root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.relative_to(root).as_posix(), path


def merge_sealed_reference_inputs(
    manifests: Sequence[str | Path], *, paper2_root: str | Path
) -> dict[str, Any]:
    """Merge independently prepared role inputs without exposing reference paths.

    A tile may occur in more than one source manifest only when the reference bytes,
    grid-plan bytes, and plan grid contract are exactly identical.  The output is
    suitable for the one-shot global sealed-reference-lock builder used by both
    calibration and G5.
    """

    root = Path(paper2_root).resolve()
    if not manifests:
        raise ValueError("at least one sealed-reference input manifest is required")
    source_records: list[dict[str, Any]] = []
    by_tile: dict[str, dict[str, Any]] = {}
    for raw_manifest in manifests:
        manifest_path = guard_paper2_path(raw_manifest, paper2_root=root)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        payload = read_json(manifest_path)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != SEALED_REFERENCE_INPUT_SCHEMA
            or not isinstance(payload.get("entries"), list)
            or not payload["entries"]
        ):
            raise ValueError("sealed-reference input schema/entries differ")
        source_records.append(
            {
                "relative_path": manifest_path.relative_to(root).as_posix(),
                "sha256": file_sha256(manifest_path),
                "entry_count": len(payload["entries"]),
            }
        )
        local_seen: set[str] = set()
        for raw in payload["entries"]:
            if not isinstance(raw, Mapping):
                raise ValueError("sealed-reference input entry is malformed")
            tile_id = str(raw.get("tile_id", ""))
            if not tile_id or tile_id in local_seen:
                raise ValueError("one sealed-reference input repeats a tile")
            local_seen.add(tile_id)
            reference_relative, reference = _relative_file(
                raw.get("reference_relative_path"), root=root, field="reference"
            )
            plan_relative, plan_path = _relative_file(
                raw.get("grid_plan_relative_path"), root=root, field="grid plan"
            )
            plan = PlanContract.read(plan_path)
            if plan.tile_id != tile_id:
                raise ValueError("sealed-reference tile differs from its grid plan")
            record = {
                "tile_id": tile_id,
                "reference_relative_path": reference_relative,
                "reference_sha256": file_sha256(reference),
                "grid_plan_relative_path": plan_relative,
                "grid_plan_sha256": file_sha256(plan_path),
                "grid_contract_sha256": canonical_sha256(
                    {
                        "tile_id": plan.tile_id,
                        "crs_wkt": plan.crs_wkt,
                        "transform": [float(value).hex() for value in plan.transform],
                    }
                ),
            }
            previous = by_tile.setdefault(tile_id, record)
            if previous != record:
                raise ValueError(
                    f"sealed-reference inputs disagree for duplicate tile {tile_id}"
                )

    entries = [
        {
            "tile_id": record["tile_id"],
            "reference_relative_path": record["reference_relative_path"],
            "grid_plan_relative_path": record["grid_plan_relative_path"],
        }
        for record in sorted(by_tile.values(), key=lambda value: value["tile_id"])
    ]
    if not entries:
        raise ValueError("merged sealed-reference input has no tiles")
    payload: dict[str, Any] = {
        "schema": SEALED_REFERENCE_INPUT_SCHEMA,
        "composition_schema": "paper2.sealed-reference-input-composition/v1",
        "evaluator_only": True,
        "reference_paths_exported_to_inference_workers": False,
        "sources": sorted(source_records, key=lambda value: value["relative_path"]),
        "entries": entries,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload
