from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .assets import FORMAL_QUOTAS, GeographicGroup
from .g0_sidecars import FORMAL_ROLE_COUNTS
from .hashing import canonical_sha256, file_sha256, is_sha256, read_json


QA_REPLACEMENT_SCHEMA = "paper2.g0.role-assignments/v1"
QA_REPLACEMENT_PROTOCOL = "same-stratum-fixed-seed-source-qa-replacement-v1"
ROOT_ASSIGNMENT_DRAFT_FILE_SHA256 = (
    "ba2162ce5f7caff941c54419e69aa7ae2d85dea032c7d739db73476b01a3a7dd"
)


def _assignment_score(group: GeographicGroup, *, seed_material: str) -> str:
    return canonical_sha256(
        {
            "seed_material": seed_material,
            "stratum": group.stratum,
            "group_id": group.group_id,
            "products": group.product_ids,
        }
    )


def _fixed_products(group: GeographicGroup, group_records: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    raw = group_records[group.group_id]
    products = (
        str(raw["representative_product_id"]),
        *(str(value) for value in raw.get("fallback_product_ids", [])),
    )
    if set(products) != set(group.product_ids) or len(products) != len(set(products)):
        raise ValueError(f"fixed product order differs from group membership: {group.group_id}")
    return products


def select_qa_replacements(
    *,
    stratified_groups: Sequence[GeographicGroup],
    group_records: Mapping[str, Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    exhausted_group_ids: Sequence[str],
    seed_material: str,
    previous_history: Sequence[Mapping[str, Any]] = (),
    failure_evidence: Mapping[str, Sequence[str]] | None = None,
    require_formal_counts: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace only terminal source-QA failures while preserving role and stratum.

    Candidate order is fixed by the original assignment seed.  Paper1-seen groups remain
    bridge-train-only.  No model metric, reference DEM value, or test outcome is an input.
    """
    if not seed_material:
        raise ValueError("replacement selection requires the frozen assignment seed")
    groups = {group.group_id: group for group in stratified_groups}
    if len(groups) != len(stratified_groups):
        raise ValueError("stratified replacement groups must be unique")
    current = {str(raw["group_id"]): dict(raw) for raw in assignments}
    if len(current) != len(assignments):
        raise ValueError("current assignments must be unique")
    if require_formal_counts:
        roles = Counter(str(raw.get("role", "")) for raw in assignments)
        if dict(roles) != FORMAL_ROLE_COUNTS:
            raise ValueError(f"formal assignment counts differ: {dict(roles)}")
    exhausted = tuple(sorted(str(value) for value in exhausted_group_ids))
    if not exhausted or any(value not in current for value in exhausted):
        raise ValueError("every exhausted group must belong to the current assignment")
    if len(exhausted) != len(set(exhausted)):
        raise ValueError("exhausted group identities must be unique")

    removed_before = {
        str(item.get("removed_group_id", "")) for item in previous_history
    }
    selected = set(current)
    unavailable = selected | removed_before
    evidence = {} if failure_evidence is None else failure_evidence
    role_order = {role: index for index, role in enumerate(FORMAL_QUOTAS)}
    ordered_exhausted = sorted(
        exhausted,
        key=lambda group_id: (role_order[str(current[group_id]["role"])], group_id),
    )
    replacements: dict[str, dict[str, Any]] = {}
    new_history = [dict(item) for item in previous_history]
    cycle = 1 + max((int(item.get("cycle", 0)) for item in previous_history), default=0)
    for removed_id in ordered_exhausted:
        removed = current[removed_id]
        if removed_id not in groups:
            raise ValueError(f"assigned group is absent from stratified pool: {removed_id}")
        removed_group = groups[removed_id]
        role = str(removed["role"])
        candidates = [
            group
            for group in groups.values()
            if group.group_id not in unavailable
            and group.stratum == removed_group.stratum
            and (role == "bridge_train" or not group.seen_by_paper1)
        ]
        candidates.sort(
            key=lambda group: (
                0 if role == "bridge_train" and group.seen_by_paper1 else 1,
                _assignment_score(group, seed_material=seed_material),
                group.group_id,
            )
        )
        if not candidates:
            raise ValueError(
                f"no unused same-stratum candidate can replace {removed_id}/{role}"
            )
        replacement = candidates[0]
        products = _fixed_products(replacement, group_records)
        replacements[removed_id] = {
            "group_id": replacement.group_id,
            "role": role,
            "latitude_band": replacement.stratum[0],
            "terrain_quartile": replacement.stratum[1],
            "representative_product_id": products[0],
            "fallback_product_ids": list(products[1:]),
        }
        unavailable.add(replacement.group_id)
        failure_hashes = sorted(str(value) for value in evidence.get(removed_id, ()))
        if any(not is_sha256(value) for value in failure_hashes):
            raise ValueError(f"replacement failure evidence is not SHA-256: {removed_id}")
        new_history.append(
            {
                "cycle": cycle,
                "removed_group_id": removed_id,
                "replacement_group_id": replacement.group_id,
                "role": role,
                "latitude_band": replacement.stratum[0],
                "terrain_quartile": replacement.stratum[1],
                "replacement_seen_by_paper1": replacement.seen_by_paper1,
                "candidate_rank": 0,
                "candidate_score_sha256": _assignment_score(
                    replacement, seed_material=seed_material
                ),
                "qa_failure_evidence_sha256s": failure_hashes,
                "model_outcomes_accessed": False,
                "reference_dem_accessed": False,
            }
        )

    result = [
        replacements.get(group_id, dict(raw)) for group_id, raw in current.items()
    ]
    result.sort(key=lambda item: str(item["group_id"]))
    result_ids = [str(item["group_id"]) for item in result]
    if len(result_ids) != len(set(result_ids)) or set(result_ids).intersection(exhausted):
        raise AssertionError("QA replacement did not produce unique, removed assignments")
    if Counter(str(item["role"]) for item in result) != Counter(
        str(item["role"]) for item in assignments
    ):
        raise AssertionError("QA replacement changed formal role counts")
    return result, new_history


def validate_formal_assignment_parent(
    path: str | Path,
    *,
    expected_file_sha256: str,
    root_assignment_file_sha256: str,
) -> dict[str, Any]:
    """Accept the initial parent or a hash-chained pre-unlock QA replacement parent."""
    assignment_path = Path(path)
    if not is_sha256(expected_file_sha256) or file_sha256(assignment_path) != expected_file_sha256:
        raise ValueError("assignment bytes differ from the explicitly supplied parent hash")
    payload = read_json(assignment_path)
    if expected_file_sha256 == root_assignment_file_sha256:
        return payload
    if payload.get("schema") != QA_REPLACEMENT_SCHEMA:
        raise ValueError("non-root formal assignment must be a QA replacement manifest")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("QA replacement assignment content hash differs")
    if (
        payload.get("state") != "qa_replacement_frozen"
        or payload.get("formal_test_unlocked") is not False
        or payload.get("root_assignments_file_sha256") != root_assignment_file_sha256
        or payload.get("selection_protocol") != QA_REPLACEMENT_PROTOCOL
        or not payload.get("qa_replacement_history")
    ):
        raise ValueError("QA replacement assignment lacks its frozen pre-unlock chain")
    return payload
