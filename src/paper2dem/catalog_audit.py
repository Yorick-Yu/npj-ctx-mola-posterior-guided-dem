from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import numpy as np

from .assets import buffered_bbox_overlap
from .hashing import canonical_sha256, file_sha256, read_json, write_canonical_json


def _rows(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of records: {path}")
    return value


def _is_ctx(row: Mapping[str, Any]) -> bool:
    return "ctx" in str(row.get("collection", "")).lower()


def _observations(row: Mapping[str, Any]) -> tuple[str, ...]:
    tile_id = str(row["tile_id"])
    if _is_ctx(row):
        values = tuple(value for value in tile_id.split("__") if value)
        if len(values) < 2:
            raise ValueError(f"CTX product lacks its source observation pair: {tile_id}")
        return values
    return (tile_id,)


def _bbox(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in row["bbox"])
    if len(values) != 4 or not all(np.isfinite(values)):
        raise ValueError(f"invalid bbox for {row.get('tile_id')}")
    if values[0] > values[2] or values[1] > values[3]:
        raise ValueError(f"wrapped bbox must be normalized before audit: {row.get('tile_id')}")
    return values


def merge_formal_with_ctx_extension(
    formal_rows: Iterable[Mapping[str, Any]],
    extension_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Build the pre-outcome Paper2 catalog from Paper1 plus CTX-only extension rows.

    Paper1 records are retained verbatim (including the external HiRISE members),
    while a secondary server catalog contributes only valid CTX products.  Duplicate
    product IDs are rejected instead of silently selecting one source.
    """
    formal = [dict(row) for row in formal_rows]
    extension_ctx = [dict(row) for row in extension_rows if _is_ctx(row)]
    merged = [*formal, *extension_ctx]
    identifiers = [str(row.get("tile_id", "")) for row in merged]
    if any(not value for value in identifiers):
        raise ValueError("catalog records require nonempty tile_id values")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("formal and extension catalogs contain duplicate product IDs")
    for row in merged:
        if _is_ctx(row):
            _observations(row)
            _bbox(row)
    return tuple(sorted(merged, key=lambda row: str(row["tile_id"])))


def filter_ctx_without_observations(
    rows: Iterable[Mapping[str, Any]], excluded_observation_ids: Iterable[str]
) -> tuple[dict[str, Any], ...]:
    """Derive CTX extension candidates that share no frozen Paper1 observation."""
    excluded = {str(value) for value in excluded_observation_ids}
    if any(not value for value in excluded):
        raise ValueError("excluded observation IDs must be nonempty")
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if not _is_ctx(row):
            continue
        observations = _observations(row)
        _bbox(row)
        if excluded.isdisjoint(observations):
            result.append(row)
    identifiers = [str(row["tile_id"]) for row in result]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("filtered CTX extension contains duplicate product IDs")
    return tuple(sorted(result, key=lambda row: str(row["tile_id"])))


def _conflicts(
    left: Mapping[str, Any], right: Mapping[str, Any], *, buffer_degrees: float
) -> bool:
    return bool(
        set(_observations(left)).intersection(_observations(right))
        or buffered_bbox_overlap(_bbox(left), _bbox(right), buffer_degrees=buffer_degrees)
    )


def build_catalog_conflict_groups(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed_material: str,
    buffer_degrees: float = 0.25,
) -> tuple[dict[str, Any], ...]:
    """Build the complete transitive conflict graph before selecting candidates.

    The fixed product order is part of the contract: the first product is the
    representative and the remaining products are its deterministic fallbacks.
    """
    candidates = sorted(list(rows), key=lambda row: str(row["tile_id"]))
    identifiers = [str(row["tile_id"]) for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("catalog product IDs must be unique before grouping")
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if _conflicts(candidates[left], candidates[right], buffer_degrees=buffer_degrees):
                union(left, right)
    components: dict[int, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(candidates):
        components.setdefault(find(index), []).append(row)

    groups: list[dict[str, Any]] = []
    for component in components.values():
        product_ids = tuple(sorted(str(row["tile_id"]) for row in component))
        observation_ids = tuple(
            sorted({observation for row in component for observation in _observations(row)})
        )
        boxes = np.asarray([_bbox(row) for row in component], dtype=np.float64)
        bbox = [
            float(np.min(boxes[:, 0])),
            float(np.min(boxes[:, 1])),
            float(np.max(boxes[:, 2])),
            float(np.max(boxes[:, 3])),
        ]
        group_id = "geo-" + canonical_sha256(
            {
                "product_ids": product_ids,
                "observation_ids": observation_ids,
                "buffer_degrees": buffer_degrees,
            }
        )[:20]
        fixed_order = sorted(
            product_ids,
            key=lambda product_id: (
                canonical_sha256(
                    {
                        "seed_material": seed_material,
                        "group_id": group_id,
                        "product_id": product_id,
                    }
                ),
                product_id,
            ),
        )
        groups.append(
            {
                "group_id": group_id,
                "product_ids": list(product_ids),
                "observation_ids": list(observation_ids),
                "bbox_lonlat": bbox,
                "center_latitude": float((bbox[1] + bbox[3]) / 2.0),
                "representative_product_id": fixed_order[0],
                "fallback_product_ids": fixed_order[1:],
            }
        )
    return tuple(sorted(groups, key=lambda group: str(group["group_id"])))


def greedy_independent_products(
    rows: Iterable[Mapping[str, Any]], *, seed_material: str, buffer_degrees: float = 0.25
) -> tuple[str, ...]:
    groups = build_catalog_conflict_groups(
        rows, seed_material=seed_material, buffer_degrees=buffer_degrees
    )
    ordered = sorted(
        groups,
        key=lambda group: (
            canonical_sha256(
                {
                    "seed_material": seed_material,
                    "group_id": group["group_id"],
                    "representative_product_id": group["representative_product_id"],
                }
            ),
            str(group["group_id"]),
        ),
    )
    return tuple(str(group["representative_product_id"]) for group in ordered)


def audit_catalog_prelock(
    catalog_path: str | Path,
    benchmark_lock_path: str | Path,
    *,
    expected_catalog_sha256: str,
    expected_benchmark_sha256: str,
    seed_material: str,
    buffer_degrees: float = 0.25,
    catalog_source_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    catalog_path = Path(catalog_path)
    benchmark_lock_path = Path(benchmark_lock_path)
    actual_catalog_hash = file_sha256(catalog_path)
    actual_benchmark_hash = file_sha256(benchmark_lock_path)
    catalog = _rows(catalog_path)
    formal = _rows(benchmark_lock_path)
    formal_ids = {str(row["tile_id"]) for row in formal}
    formal_ctx = [row for row in formal if _is_ctx(row)]
    ctx = [row for row in catalog if _is_ctx(row)]
    nonformal = [row for row in ctx if str(row["tile_id"]) not in formal_ids]
    # The graph is deliberately formed before any formal-product exclusion.  If
    # A conflicts with a locked product and A connects transitively to B, the
    # entire A--B component is excluded.  Filtering rows first would leak B into
    # the candidate pool and overcount both products and independent groups.
    all_groups = build_catalog_conflict_groups(
        ctx, seed_material=seed_material, buffer_degrees=buffer_degrees
    )
    rows_by_id = {str(row["tile_id"]): row for row in ctx}
    eligible_groups: list[dict[str, Any]] = []
    excluded_groups: list[dict[str, Any]] = []
    for group in all_groups:
        members = [rows_by_id[product_id] for product_id in group["product_ids"]]
        direct_locked_ids = sorted(set(group["product_ids"]).intersection(formal_ids))
        formal_conflict_ids = sorted(
            {
                str(locked["tile_id"])
                for member in members
                for locked in formal_ctx
                if _conflicts(member, locked, buffer_degrees=buffer_degrees)
            }
        )
        enriched = dict(group)
        enriched["excluded"] = bool(direct_locked_ids or formal_conflict_ids)
        enriched["locked_product_ids"] = direct_locked_ids
        enriched["formal_conflict_product_ids"] = formal_conflict_ids
        if enriched["excluded"]:
            excluded_groups.append(enriched)
        else:
            eligible_groups.append(enriched)
    eligible_ids = {
        product_id for group in eligible_groups for product_id in group["product_ids"]
    }
    eligible = [row for row in ctx if str(row["tile_id"]) in eligible_ids]
    independent = tuple(
        str(group["representative_product_id"])
        for group in sorted(eligible_groups, key=lambda group: str(group["group_id"]))
    )
    terrain_ready = [
        row
        for row in eligible
        if np.isfinite(float(dict(row.get("qa", {})).get("terrain_complexity", np.nan)))
    ]
    edr_ready = [
        row
        for row in eligible
        if any("edr" in str(key).lower() for key in dict(row.get("assets", {})))
    ]
    errors: list[str] = []
    if actual_catalog_hash != expected_catalog_sha256:
        errors.append("catalog SHA-256 differs from the frozen catalog source")
    if actual_benchmark_hash != expected_benchmark_sha256:
        errors.append("benchmark-lock SHA-256 differs from the frozen Paper1 source")
    source_lock_hash: str | None = None
    if catalog_source_lock_path is None:
        if len(catalog) != 892:
            errors.append(f"catalog count is {len(catalog)}, expected 892")
        if len(ctx) != 676:
            errors.append(f"CTX catalog count is {len(ctx)}, expected 676")
        if len(eligible) != 662:
            errors.append(f"eligible CTX count is {len(eligible)}, expected 662")
        if len(independent) != 407:
            errors.append(f"independent conflict-group count is {len(independent)}, expected 407")
    else:
        source_lock_path = Path(catalog_source_lock_path)
        source_lock = read_json(source_lock_path)
        source_lock_hash = file_sha256(source_lock_path)
        if not isinstance(source_lock, dict):
            errors.append("catalog source lock must be a JSON object")
        else:
            if source_lock.get("schema") != "paper2.catalog-source-lock/v1":
                errors.append("catalog source lock schema is invalid")
            if source_lock.get("catalog_sha256") != actual_catalog_hash:
                errors.append("catalog source lock does not bind the catalog bytes")
            if int(source_lock.get("catalog_total", -1)) != len(catalog):
                errors.append("catalog source lock total count does not close")
            if int(source_lock.get("catalog_ctx", -1)) != len(ctx):
                errors.append("catalog source lock CTX count does not close")
    blockers: list[str] = []
    if len(independent) < 108:
        blockers.append("fewer than 108 independent eligible CTX products")
    if len(terrain_ready) < 108:
        blockers.append("coarse-terrain scores are not available for 108 eligible products")
    if len(edr_ready) < 108:
        blockers.append("catalog does not yet bind raw CTX EDR URLs for 108 eligible products")
    payload: dict[str, Any] = {
        "schema": "paper2.catalog-prelock-audit/v1",
        "inputs": {
            "catalog": {"sha256": actual_catalog_hash, "expected": expected_catalog_sha256},
            "benchmark_lock": {
                "sha256": actual_benchmark_hash,
                "expected": expected_benchmark_sha256,
            },
            "catalog_source_lock": (
                None
                if source_lock_hash is None
                else {"sha256": source_lock_hash, "path_recorded": False}
            ),
        },
        "buffer_degrees": buffer_degrees,
        "seed_material": seed_material,
        "counts": {
            "catalog_total": len(catalog),
            "catalog_ctx": len(ctx),
            "formal_total": len(formal),
            "formal_ctx": len(formal_ctx),
            "nonformal_ctx": len(nonformal),
            "conservative_eligible_ctx": len(eligible),
            "independent_conflict_groups": len(independent),
            "greedy_independent_ctx": len(independent),
            "eligible_with_coarse_terrain_score": len(terrain_ready),
            "eligible_with_raw_edr_url": len(edr_ready),
        },
        "independent_product_ids": list(independent),
        "conflict_groups": {
            "schema": "paper2.g0.conflict-groups/v1",
            "catalog_sha256": actual_catalog_hash,
            "benchmark_lock_sha256": actual_benchmark_hash,
            "buffer_degrees": buffer_degrees,
            "grouping_algorithm": "full-ctx-transitive-shared-observation-or-buffered-overlap-v1",
            "eligible_groups": eligible_groups,
            "excluded_groups": excluded_groups,
        },
        "errors": errors,
        "g0_blockers": blockers,
        "hash_gate_passed": not errors,
        "role_assignment_ready": not errors and not blockers,
        "formal_test_unlocked": False,
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    return payload


def write_catalog_prelock_audit(
    output: str | Path,
    catalog_path: str | Path,
    benchmark_lock_path: str | Path,
    *,
    expected_catalog_sha256: str,
    expected_benchmark_sha256: str,
    seed_material: str,
    paper2_root: str | Path,
    catalog_source_lock_path: str | Path | None = None,
) -> Path:
    payload = audit_catalog_prelock(
        catalog_path,
        benchmark_lock_path,
        expected_catalog_sha256=expected_catalog_sha256,
        expected_benchmark_sha256=expected_benchmark_sha256,
        seed_material=seed_material,
        catalog_source_lock_path=catalog_source_lock_path,
    )
    return write_canonical_json(output, payload, paper2_root=paper2_root)
