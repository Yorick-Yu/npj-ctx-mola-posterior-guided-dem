from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .contracts import AssetLock
from .hashing import canonical_sha256


FORMAL_QUOTAS = {
    "bridge_train": 4,
    "bridge_val": 1,
    "uncertainty_calibration": 1,
    "paper2_locked_test": 2,
    "sealed_reserve": 1,
}

LATITUDE_BANDS = ((-65.0, -20.0), (-20.0, 20.0), (20.0, 65.0))


@dataclass(frozen=True)
class CandidateAsset:
    product_id: str
    observation_ids: tuple[str, ...]
    bbox_lonlat: tuple[float, float, float, float]
    center_latitude: float
    terrain_score: float
    seen_by_paper1: bool = False

    def validate(self) -> None:
        if not self.product_id:
            raise ValueError("candidate product_id cannot be empty")
        if not self.observation_ids:
            raise ValueError("candidate observation_ids cannot be empty")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("candidate observation_ids must be unique")
        if len(self.bbox_lonlat) != 4 or not all(np.isfinite(item) for item in self.bbox_lonlat):
            raise ValueError("candidate bbox must contain four finite values")
        west, south, east, north = self.bbox_lonlat
        if west > east or south > north:
            raise ValueError("candidate bbox must be normalized and cannot wrap")
        if not np.isfinite(self.center_latitude) or not np.isfinite(self.terrain_score):
            raise ValueError("candidate latitude and terrain score must be finite")


@dataclass(frozen=True)
class GeographicGroup:
    group_id: str
    product_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    bbox_lonlat: tuple[float, float, float, float]
    center_latitude: float
    terrain_score: float
    seen_by_paper1: bool
    latitude_band: int | None = None
    terrain_quartile: int | None = None

    @property
    def stratum(self) -> tuple[int, int]:
        if self.latitude_band is None or self.terrain_quartile is None:
            raise ValueError("group has not been stratified")
        return self.latitude_band, self.terrain_quartile


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def buffered_bbox_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    buffer_degrees: float,
) -> bool:
    if not np.isfinite(buffer_degrees) or buffer_degrees < 0:
        raise ValueError("buffer_degrees must be finite and nonnegative")
    left_west, left_south, left_east, left_north = left
    right_west, right_south, right_east, right_north = right
    return not (
        left_east + buffer_degrees < right_west - buffer_degrees
        or right_east + buffer_degrees < left_west - buffer_degrees
        or left_north + buffer_degrees < right_south - buffer_degrees
        or right_north + buffer_degrees < left_south - buffer_degrees
    )


def build_geographic_groups(
    candidates: tuple[CandidateAsset, ...] | list[CandidateAsset],
    *,
    buffer_degrees: float = 0.25,
) -> tuple[GeographicGroup, ...]:
    """Compute connected components from shared observations or buffered overlap."""
    ordered = sorted(candidates, key=lambda item: item.product_id)
    if not ordered:
        return ()
    product_ids = [item.product_id for item in ordered]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("candidate product IDs must be unique")
    for candidate in ordered:
        candidate.validate()
    components = _UnionFind(len(ordered))
    observation_owner: dict[str, int] = {}
    for index, candidate in enumerate(ordered):
        for observation_id in candidate.observation_ids:
            owner = observation_owner.setdefault(observation_id, index)
            components.union(index, owner)
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            if buffered_bbox_overlap(
                ordered[left].bbox_lonlat,
                ordered[right].bbox_lonlat,
                buffer_degrees=buffer_degrees,
            ):
                components.union(left, right)
    members: dict[int, list[CandidateAsset]] = {}
    for index, candidate in enumerate(ordered):
        members.setdefault(components.find(index), []).append(candidate)
    groups: list[GeographicGroup] = []
    for component in members.values():
        component = sorted(component, key=lambda item: item.product_id)
        component_products = tuple(item.product_id for item in component)
        observations = tuple(
            sorted({observation for item in component for observation in item.observation_ids})
        )
        boxes = np.asarray([item.bbox_lonlat for item in component], dtype=np.float64)
        bbox = (
            float(np.min(boxes[:, 0])),
            float(np.min(boxes[:, 1])),
            float(np.max(boxes[:, 2])),
            float(np.max(boxes[:, 3])),
        )
        group_hash = canonical_sha256(
            {
                "products": component_products,
                "observations": observations,
                "buffer_degrees": buffer_degrees,
            }
        )
        groups.append(
            GeographicGroup(
                group_id=f"geo-{group_hash[:20]}",
                product_ids=component_products,
                observation_ids=observations,
                bbox_lonlat=bbox,
                center_latitude=float(np.mean([item.center_latitude for item in component])),
                terrain_score=float(np.mean([item.terrain_score for item in component])),
                seen_by_paper1=any(item.seen_by_paper1 for item in component),
            )
        )
    return tuple(sorted(groups, key=lambda item: item.group_id))


def _balanced_rank_bins(
    groups: list[GeographicGroup],
    *,
    bins: int,
    value,
) -> dict[str, int]:
    ordered = sorted(groups, key=lambda item: (value(item), item.group_id))
    return {
        group.group_id: min(bins - 1, index * bins // len(ordered))
        for index, group in enumerate(ordered)
    }


def stratify_groups(
    groups: tuple[GeographicGroup, ...] | list[GeographicGroup],
) -> tuple[GeographicGroup, ...]:
    """Apply fixed latitude bands and terrain ranks defined by the unseen pool.

    Paper1-seen groups may only enter bridge_train, so allowing them to define
    confirmation-set quartiles can leave a nominal quartile with too few eligible
    unseen groups. The pre-outcome reference distribution is therefore the unseen
    candidate population; seen groups are mapped onto those frozen ranks afterward.
    """
    values = list(groups)
    if len(values) < 12:
        raise ValueError("at least 12 geographic groups are required for 3 x 4 strata")
    latitude: dict[str, int] = {}
    for group in values:
        value = group.center_latitude
        if -65.0 <= value < -20.0:
            latitude[group.group_id] = 0
        elif -20.0 <= value < 20.0:
            latitude[group.group_id] = 1
        elif 20.0 <= value <= 65.0:
            latitude[group.group_id] = 2
        else:
            raise ValueError(
                f"group {group.group_id} latitude {value} is outside locked [-65, 65] range"
            )
    terrain: dict[str, int] = {}
    for band in range(3):
        band_groups = [item for item in values if latitude[item.group_id] == band]
        unseen_groups = [item for item in band_groups if not item.seen_by_paper1]
        if len(unseen_groups) < 4:
            raise ValueError(
                f"latitude band {band} has fewer than four unseen geographic groups"
            )
        unseen_bins = _balanced_rank_bins(
            unseen_groups,
            bins=4,
            value=lambda item: item.terrain_score,
        )
        terrain.update(unseen_bins)
        unseen_order = sorted(
            (item.terrain_score, item.group_id) for item in unseen_groups
        )
        remaining_seen = [item for item in band_groups if item.group_id not in unseen_bins]
        minimum_total_per_stratum = sum(FORMAL_QUOTAS.values())
        for terrain_bin in range(4):
            unseen_in_bin = [
                item for item in unseen_groups if unseen_bins[item.group_id] == terrain_bin
            ]
            deficit = max(0, minimum_total_per_stratum - len(unseen_in_bin))
            target = float(np.median([item.terrain_score for item in unseen_in_bin]))
            for _ in range(deficit):
                if not remaining_seen:
                    break
                selected = min(
                    remaining_seen,
                    key=lambda item: (abs(item.terrain_score - target), item.group_id),
                )
                terrain[selected.group_id] = terrain_bin
                remaining_seen.remove(selected)
        for group in remaining_seen:
            rank = bisect_right(unseen_order, (group.terrain_score, group.group_id))
            terrain[group.group_id] = min(3, rank * 4 // len(unseen_order))
    return tuple(
        sorted(
            (
                replace(
                    group,
                    latitude_band=latitude[group.group_id],
                    terrain_quartile=terrain[group.group_id],
                )
                for group in values
            ),
            key=lambda item: item.group_id,
        )
    )


def assign_stratified_roles(
    groups: tuple[GeographicGroup, ...] | list[GeographicGroup],
    *,
    seed_material: str,
    quotas: dict[str, int] | None = None,
) -> dict[str, str]:
    """Select and assign exactly 108 independent groups without outcome inspection."""
    quotas = dict(FORMAL_QUOTAS if quotas is None else quotas)
    if quotas != FORMAL_QUOTAS:
        raise ValueError(f"formal role quotas are locked and must equal {FORMAL_QUOTAS}")
    if not seed_material:
        raise ValueError("seed_material cannot be empty")
    values = list(groups)
    group_ids = [item.group_id for item in values]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("group IDs must be unique")
    strata: dict[tuple[int, int], list[GeographicGroup]] = {
        (latitude, terrain): [] for latitude in range(3) for terrain in range(4)
    }
    for group in values:
        strata[group.stratum].append(group)
    assignments: dict[str, str] = {}
    per_stratum = sum(quotas.values())
    for stratum, members in sorted(strata.items()):
        eligible_new = [item for item in members if not item.seen_by_paper1]
        seen = [item for item in members if item.seen_by_paper1]

        def score(group: GeographicGroup) -> str:
            return canonical_sha256(
                {
                    "seed_material": seed_material,
                    "stratum": stratum,
                    "group_id": group.group_id,
                    "products": group.product_ids,
                }
            )

        seen.sort(key=lambda item: (score(item), item.group_id))
        eligible_new.sort(key=lambda item: (score(item), item.group_id))
        train_seen = seen[: quotas["bridge_train"]]
        train_needed = quotas["bridge_train"] - len(train_seen)
        if len(eligible_new) < per_stratum - len(train_seen):
            raise ValueError(
                f"stratum {stratum} lacks unseen groups for locked roles: "
                f"need {per_stratum - len(train_seen)}, found {len(eligible_new)}"
            )
        cursor = 0
        for group in train_seen:
            assignments[group.group_id] = "bridge_train"
        for group in eligible_new[cursor : cursor + train_needed]:
            assignments[group.group_id] = "bridge_train"
        cursor += train_needed
        for role in (
            "bridge_val",
            "uncertainty_calibration",
            "paper2_locked_test",
            "sealed_reserve",
        ):
            stop = cursor + quotas[role]
            for group in eligible_new[cursor:stop]:
                assignments[group.group_id] = role
            cursor = stop
    expected = {role: count * 12 for role, count in quotas.items()}
    observed = {role: list(assignments.values()).count(role) for role in quotas}
    if observed != expected:
        raise AssertionError(f"assignment count invariant failed: {observed} != {expected}")
    return assignments


@dataclass(frozen=True)
class AssetLeakageAudit:
    ok: bool
    errors: tuple[str, ...]
    group_role_counts: tuple[tuple[str, int], ...]
    stratum_role_counts: tuple[tuple[str, int, int, int], ...]
    failed_qa_products: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "paper2.asset-leakage-audit/v1",
            "ok": self.ok,
            "errors": list(self.errors),
            "group_role_counts": dict(self.group_role_counts),
            "stratum_role_counts": [
                {
                    "role": role,
                    "latitude_band": latitude,
                    "terrain_quartile": terrain,
                    "groups": count,
                }
                for role, latitude, terrain, count in self.stratum_role_counts
            ],
            "failed_qa_products": list(self.failed_qa_products),
        }


def audit_asset_lock_geography(asset_lock: AssetLock) -> AssetLeakageAudit:
    """Audit group, raw-observation, buffered-overlap, quotas, and QA gates."""
    errors: list[str] = []
    try:
        asset_lock.validate()
    except ValueError as exc:
        errors.append(str(exc))
    assets = list(asset_lock.assets)
    for left_index, left in enumerate(assets):
        for right in assets[left_index + 1 :]:
            shared = sorted(set(left.observation_ids).intersection(right.observation_ids))
            different_groups = left.geographic_group_id != right.geographic_group_id
            if shared and different_groups:
                errors.append(
                    f"different-group raw observation conflict {left.product_id}/{right.product_id}: "
                    f"roles={left.role}/{right.role}, observations={shared}"
                )
            if left.geographic_group_id == right.geographic_group_id and left.role != right.role:
                errors.append(
                    f"cross-role geographic group leakage {left.product_id}/{right.product_id}"
                )
            if different_groups and buffered_bbox_overlap(
                left.bbox_lonlat,
                right.bbox_lonlat,
                buffer_degrees=asset_lock.grouping_buffer_degrees,
            ):
                errors.append(
                    f"different-group buffered geographic overlap {left.product_id}/{right.product_id}: "
                    f"roles={left.role}/{right.role}"
                )
    group_roles = {
        (asset.geographic_group_id, asset.role) for asset in assets
    }
    role_counts: dict[str, int] = {}
    for _, role in group_roles:
        role_counts[role] = role_counts.get(role, 0) + 1
    expected_counts = {
        "bridge_train": 48,
        "bridge_val": 12,
        "uncertainty_calibration": 12,
        "paper2_locked_test": 24,
        "sealed_reserve": 12,
    }
    for role, expected in expected_counts.items():
        observed = role_counts.get(role, 0)
        if observed != expected:
            errors.append(f"role {role} has {observed} groups; expected {expected}")
    strata_counts: dict[tuple[str, int, int], int] = {}
    seen_groups: set[tuple[str, str]] = set()
    for asset in assets:
        identity = (asset.geographic_group_id, asset.role)
        if identity in seen_groups:
            continue
        seen_groups.add(identity)
        if asset.latitude_band is not None and asset.terrain_quartile is not None:
            key = (asset.role, asset.latitude_band, asset.terrain_quartile)
            strata_counts[key] = strata_counts.get(key, 0) + 1
    for (latitude, terrain) in [(i, j) for i in range(3) for j in range(4)]:
        for role, quota in FORMAL_QUOTAS.items():
            observed = strata_counts.get((role, latitude, terrain), 0)
            if observed != quota:
                errors.append(
                    f"stratum ({latitude},{terrain}) role {role} has {observed}; expected {quota}"
                )
    failed_qa = tuple(
        sorted(
            asset.product_id
            for asset in assets
            if asset.role in expected_counts and not asset.qa.passed
        )
    )
    if failed_qa:
        errors.append(f"formal products failing QA: {len(failed_qa)}")
    return AssetLeakageAudit(
        ok=not errors,
        errors=tuple(sorted(set(errors))),
        group_role_counts=tuple(sorted(role_counts.items())),
        stratum_role_counts=tuple(
            (role, latitude, terrain, count)
            for (role, latitude, terrain), count in sorted(strata_counts.items())
        ),
        failed_qa_products=failed_qa,
    )
