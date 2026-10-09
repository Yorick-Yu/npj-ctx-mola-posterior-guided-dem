from __future__ import annotations

from pathlib import Path
from typing import Any

from .g0 import SCHEMAS, validate_edr_index
from .g0_sidecars import _conflict_groups_hash, _fixed_assignments
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)


def rebind_edr_index_to_assignment_bytes(
    *,
    source_edr_index: str | Path,
    conflict_groups: str | Path,
    assignments: str | Path,
    output: str | Path,
    paper2_root: str | Path,
    require_formal_counts: bool = True,
) -> dict[str, Any]:
    """Rebind verified ODE responses to a byte-new assignment parent without rediscovery."""
    root = Path(paper2_root).resolve()
    source_path = guard_paper2_path(source_edr_index, paper2_root=root)
    conflicts_path = guard_paper2_path(conflict_groups, paper2_root=root)
    assignments_path = guard_paper2_path(assignments, paper2_root=root)
    source = read_json(source_path)
    conflicts_payload = read_json(conflicts_path)
    assignments_payload = read_json(assignments_path)
    conflict_hash = _conflict_groups_hash(conflicts_payload)
    normalized, specifications = _fixed_assignments(
        conflicts_payload,
        assignments_payload,
        require_formal_counts=require_formal_counts,
    )
    entries = validate_edr_index(
        source,
        catalog_sha256=str(conflicts_payload["catalog_sha256"]),
        conflict_groups_sha256=conflict_hash,
    )
    role_hash = canonical_sha256(normalized)
    if source.get("role_assignment_sha256") != role_hash:
        raise ValueError("source EDR index has different assignment semantics; rediscovery required")
    required_observations = [str(item["observation_id"]) for item in specifications]
    if set(entries) != set(required_observations):
        raise ValueError("source EDR index observation set differs; rediscovery required")
    required_hash = canonical_sha256(specifications)
    bindings = {
        "catalog_sha256": str(conflicts_payload["catalog_sha256"]),
        "conflict_groups_sha256": conflict_hash,
        "conflict_groups_file_sha256": file_sha256(conflicts_path),
        "assignments_file_sha256": file_sha256(assignments_path),
        "role_assignment_sha256": role_hash,
        "required_observations_sha256": required_hash,
    }
    ordered_entries = [entries[observation_id] for observation_id in required_observations]
    payload: dict[str, Any] = {
        "schema": SCHEMAS["edr"],
        "state": "complete",
        **bindings,
        "input_bindings_sha256": canonical_sha256(bindings),
        "requested_observation_count": len(ordered_entries),
        "terminal_observation_count": len(ordered_entries),
        "failed_observation_count": 0,
        "entries": ordered_entries,
        "rebound_without_network_access": True,
        "source_edr_index_sha256": file_sha256(source_path),
        "source_ode_response_set_sha256": canonical_sha256(
            [
                {
                    "observation_id": entry["observation_id"],
                    "ode_response_sha256": entry["ode_response_sha256"],
                }
                for entry in ordered_entries
            ]
        ),
    }
    validate_edr_index(
        payload,
        catalog_sha256=str(conflicts_payload["catalog_sha256"]),
        conflict_groups_sha256=conflict_hash,
    )
    written = write_canonical_json(output, payload, paper2_root=root)
    return read_json(written)
