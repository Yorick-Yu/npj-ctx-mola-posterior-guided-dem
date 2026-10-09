"""Frozen selection of the CTX observation actually consumed by formal G0."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .hashing import canonical_sha256


FORMAL_G0_GROUP_COUNT = 108


def locked_assignments(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    assignments = [
        {
            "group_id": str(raw["group_id"]),
            "role": str(raw["role"]),
            "representative_product_id": str(raw["representative_product_id"]),
            "fallback_product_ids": [str(value) for value in raw["fallback_product_ids"]],
        }
        for raw in payload.get("assignments", [])
    ]
    if len(assignments) != FORMAL_G0_GROUP_COUNT:
        raise ValueError(
            f"formal G0 requires {FORMAL_G0_GROUP_COUNT} assignments, found {len(assignments)}"
        )
    group_ids = [entry["group_id"] for entry in assignments]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("formal G0 assignments contain duplicate groups")
    return assignments


def source_selected_products(
    source_index: Mapping[str, Any], assignments: list[dict[str, Any]]
) -> dict[str, str]:
    """Return the one source-ready product frozen for each geographic group."""

    if (
        source_index.get("state") != "frontier_ready"
        or int(source_index.get("expected_group_count", -1)) != FORMAL_G0_GROUP_COUNT
        or int(source_index.get("source_ready_group_count", -1))
        != FORMAL_G0_GROUP_COUNT
        or source_index.get("role_assignment_sha256") != canonical_sha256(assignments)
    ):
        raise ValueError("source preparation is not the complete assignment-bound frontier")
    groups = source_index.get("groups")
    if not isinstance(groups, list) or len(groups) != FORMAL_G0_GROUP_COUNT:
        raise ValueError("source preparation does not contain 108 groups")
    assignment_by_group = {entry["group_id"]: entry for entry in assignments}
    selected: dict[str, str] = {}
    for raw in groups:
        group = dict(raw)
        group_id = str(group.get("group_id", ""))
        if group_id in selected or group_id not in assignment_by_group:
            raise ValueError("source preparation contains an unknown or duplicate group")
        assignment = assignment_by_group[group_id]
        fixed = [
            assignment["representative_product_id"],
            *assignment["fallback_product_ids"],
        ]
        if (
            group.get("role") != assignment["role"]
            or group.get("state") != "source_ready"
            or list(group.get("fixed_product_ids", [])) != fixed
        ):
            raise ValueError("source-preparation group differs from its frozen assignment")
        attempts = group.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError("source-preparation group lacks attempts")
        complete = [dict(item) for item in attempts if dict(item).get("status") == "complete"]
        if len(complete) != 1:
            raise ValueError("source-preparation group must contain one complete attempt")
        product_id = str(complete[0].get("product_id", ""))
        if (
            product_id not in fixed
            or group.get("source_ready_product_id") != product_id
            or int(complete[0].get("product_order", -1)) != fixed.index(product_id)
        ):
            raise ValueError("source-ready product differs from the frozen fallback order")
        selected[group_id] = product_id
    if set(selected) != set(assignment_by_group):
        raise ValueError("source preparation does not cover every frozen group")
    return selected


def formal_primary_tasks(
    *,
    assignments: list[dict[str, Any]],
    edr_index: Mapping[str, Any],
    source_index: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Bind each group to the first CTX observation of its source-ready product."""

    if (
        edr_index.get("schema") != "paper2.g0.edr-index/v1"
        or edr_index.get("state") != "complete"
        or int(edr_index.get("failed_observation_count", -1)) != 0
        or edr_index.get("role_assignment_sha256") != canonical_sha256(assignments)
    ):
        raise ValueError("EDR index is not complete and bound to the frozen assignments")
    entries = {
        str(raw["observation_id"]): dict(raw) for raw in edr_index.get("entries", [])
    }
    selected = source_selected_products(source_index, assignments)
    tasks: list[dict[str, str]] = []
    assignment_by_group = {entry["group_id"]: entry for entry in assignments}
    for group_id, product_id in selected.items():
        observations = tuple(value for value in product_id.split("__") if value)
        if len(observations) != 2:
            raise ValueError("source-ready product is not an exact CTX stereo pair")
        primary = entries.get(observations[0])
        if primary is None or primary.get("status") != "resolved":
            raise ValueError("source-ready first CTX observation is not resolved")
        tasks.append(
            {
                "group_id": group_id,
                "role": str(assignment_by_group[group_id]["role"]),
                "product_id": product_id,
                "observation_id": observations[0],
                "url": str(primary["selected_url"]),
            }
        )
    observation_ids = [task["observation_id"] for task in tasks]
    if len(tasks) != FORMAL_G0_GROUP_COUNT or len(observation_ids) != len(
        set(observation_ids)
    ):
        raise ValueError("formal source-selected primary observations must be 108 unique IDs")
    return sorted(tasks, key=lambda task: task["group_id"])
