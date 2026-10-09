"""Conservative disk-capacity preflight for a fresh or resumed G0 CTX/ISIS run."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
import math
import shutil

from .hashing import canonical_sha256, file_sha256, guard_paper2_path, read_json
from .hashing import write_canonical_json


G0_CAPACITY_SCHEMA = "paper2.g0.disk-capacity-preflight/v3"
GIB = 1024**3
DEFAULT_EXPANSION_FACTOR = 20.5
DEFAULT_MISSING_EDR_BOUND_BYTES = 300 * 1024**2
DEFAULT_OPERATIONAL_OVERHEAD_BYTES = 10 * GIB
DEFAULT_MINIMUM_FREE_AFTER_G0_BYTES = 100 * GIB


def _object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _complete_source_products(source: Mapping[str, Any], expected_count: int) -> list[dict[str, str]]:
    if source.get("state") != "frontier_ready":
        raise ValueError("G0 capacity preflight requires frontier_ready source preparation")
    groups = source.get("groups")
    if not isinstance(groups, list) or len(groups) != expected_count:
        raise ValueError(f"source preparation must contain exactly {expected_count} groups")
    selected: list[dict[str, str]] = []
    for raw_group in groups:
        group = dict(raw_group)
        attempts = group.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError("source-preparation group lacks attempts")
        selected_product_id = str(group.get("source_ready_product_id", ""))
        selected_attempts = [
            dict(item)
            for item in attempts
            if dict(item).get("status") == "complete"
            and str(dict(item).get("product_id", "")) == selected_product_id
        ]
        attempt_product_ids = [str(dict(item).get("product_id", "")) for item in attempts]
        if (
            group.get("state") != "source_ready"
            or not selected_product_id
            or len(selected_attempts) != 1
            or len(attempt_product_ids) != len(set(attempt_product_ids))
            or "" in attempt_product_ids
        ):
            raise ValueError(
                "each source-preparation group must identify one complete active product"
            )
        selected.append(
            {
                "group_id": str(group.get("group_id", "")),
                "role": str(group.get("role", "")),
                "product_id": selected_product_id,
            }
        )
    if any(not item["group_id"] or not item["role"] or not item["product_id"] for item in selected):
        raise ValueError("source-preparation identity fields are incomplete")
    return selected


def _verified_existing_isis_artifacts(
    *,
    processing_index: Path | None,
    selected_observations: list[dict[str, Any]],
    source_contract_sha256: str | None,
    edr_index_sha256: str,
    paper2_root: Path,
    allow_cross_generation_resume_credit: bool,
) -> tuple[int, list[dict[str, Any]], str]:
    """Credit only hash-bound ISIS artifacts reusable by the selected frontier.

    A QA replacement changes the aggregate assignment/source/EDR indexes even for
    groups that already passed.  Their product-level bindings deliberately remain
    stable, so a later generation may credit that verified subset.  New/replaced
    groups receive zero credit.  The opt-in keeps the original fail-closed behavior
    for callers that require one exact aggregate generation.
    """
    if processing_index is None or not processing_index.is_file():
        return 0, [], "not_available"
    processing = _object(processing_index)
    if processing.get("schema") != "paper2.g0.ctx-processing-index/v1":
        raise ValueError("unsupported G0 processing-index schema")
    if int(processing.get("expected_group_count", -1)) != len(selected_observations):
        raise ValueError("G0 processing index count differs from the current source frontier")
    same_edr_generation = processing.get("edr_index_sha256") == edr_index_sha256
    same_source_generation = (
        source_contract_sha256 is None
        or processing.get("source_preparation_contract_sha256")
        == source_contract_sha256
    )
    same_generation = same_edr_generation and same_source_generation
    if not same_edr_generation and not allow_cross_generation_resume_credit:
        raise ValueError("G0 processing index is not bound to the current EDR index")
    if (
        not same_source_generation
        and not allow_cross_generation_resume_credit
    ):
        raise ValueError("G0 processing index is not bound to the current source contract")
    raw_groups = processing.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("G0 processing index lacks groups")
    groups = {str(dict(raw).get("group_id", "")): dict(raw) for raw in raw_groups}
    if len(groups) != len(raw_groups) or "" in groups:
        raise ValueError("G0 processing index has duplicate/empty group identities")

    seen_inodes: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    measured_bytes = 0
    for selected in selected_observations:
        group = groups.get(selected["group_id"])
        if group is None:
            records.append(
                {
                    **selected,
                    "status": "not_in_previous_processing_index",
                    "credited_bytes": 0,
                }
            )
            continue
        if group.get("role") != selected["role"]:
            raise ValueError("G0 processing group identity/role differs from source frontier")
        state_path = guard_paper2_path(
            str(group.get("group_state_path", "")), paper2_root=paper2_root
        )
        if (
            not state_path.is_file()
            or group.get("group_state_sha256") != file_sha256(state_path)
        ):
            raise ValueError("G0 processing group state bytes are missing or stale")
        matches = [
            dict(raw)
            for raw in group.get("attempts", [])
            if dict(raw).get("product_id") == selected["product_id"]
        ]
        if not matches and allow_cross_generation_resume_credit:
            records.append(
                {
                    **selected,
                    "status": "selected_product_not_in_previous_processing_index",
                    "credited_bytes": 0,
                }
            )
            continue
        if len(matches) != 1:
            raise ValueError("G0 processing group lacks its current source-selected product")
        attempt = matches[0]
        status = str(attempt.get("status", ""))
        if status not in {"awaiting_qa_evidence", "qa_passed"}:
            records.append({**selected, "status": status, "credited_bytes": 0})
            continue
        manifest_path = guard_paper2_path(
            str(attempt.get("isis_run_manifest_path", "")), paper2_root=paper2_root
        )
        if (
            not manifest_path.is_file()
            or attempt.get("isis_run_manifest_sha256") != file_sha256(manifest_path)
        ):
            raise ValueError("G0 selected product has a missing/stale ISIS manifest")
        manifest = _object(manifest_path)
        if manifest.get("reference_dem_accessed") is not False:
            raise ValueError("G0 capacity credit cannot include reference-accessing ISIS output")
        edr_path = guard_paper2_path(
            str(selected.get("edr_relative_path", "")), paper2_root=paper2_root
        )
        if (
            not edr_path.is_file()
            or manifest.get("edr_sha256") != file_sha256(edr_path)
        ):
            raise ValueError("G0 capacity credit ISIS output differs from selected EDR bytes")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("G0 ISIS manifest lacks declared artifacts")
        product_bytes = 0
        declared_paths = [manifest_path]
        for raw in artifacts.values():
            record = dict(raw)
            unresolved = paper2_root / str(record.get("relative_path", ""))
            if unresolved.is_symlink():
                raise ValueError("G0 capacity credit refuses symlinked ISIS artifacts")
            artifact = guard_paper2_path(unresolved, paper2_root=paper2_root)
            if (
                not artifact.is_file()
                or artifact.stat().st_size <= 0
                or record.get("sha256") != file_sha256(artifact)
            ):
                raise ValueError("G0 capacity credit artifact is missing/empty")
            if manifest_path.parent not in artifact.parents:
                raise ValueError("G0 ISIS manifest declares an artifact outside its attempt")
            declared_paths.append(artifact)
        for path in declared_paths:
            stat = path.stat()
            identity = (int(stat.st_dev), int(stat.st_ino))
            if identity in seen_inodes:
                continue
            seen_inodes.add(identity)
            product_bytes += int(stat.st_size)
        measured_bytes += product_bytes
        records.append({**selected, "status": status, "credited_bytes": product_bytes})
    return (
        measured_bytes,
        records,
        (
            "verified_current_processing_index"
            if same_generation
            else "verified_cross_generation_processing_subset"
        ),
    )


def audit_g0_disk_capacity(
    *,
    paper2_root: str | Path,
    source_index: str | Path,
    edr_index: str | Path,
    output: str | Path,
    processing_index: str | Path | None = None,
    expected_group_count: int = 108,
    expansion_factor: float = DEFAULT_EXPANSION_FACTOR,
    missing_edr_bound_bytes: int = DEFAULT_MISSING_EDR_BOUND_BYTES,
    operational_overhead_bytes: int = DEFAULT_OPERATIONAL_OVERHEAD_BYTES,
    minimum_free_after_g0_bytes: int = DEFAULT_MINIMUM_FREE_AFTER_G0_BYTES,
    allow_cross_generation_resume_credit: bool = False,
    disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    source_path = guard_paper2_path(source_index, paper2_root=root)
    edr_path = guard_paper2_path(edr_index, paper2_root=root)
    processing_path = (
        None
        if processing_index is None
        else guard_paper2_path(processing_index, paper2_root=root)
    )
    output_path = guard_paper2_path(output, paper2_root=root)
    if not source_path.is_file() or not edr_path.is_file():
        raise FileNotFoundError("G0 capacity source or EDR index is missing")
    if expected_group_count < 1 or expansion_factor < 1:
        raise ValueError("G0 capacity count and expansion factor are invalid")
    if min(
        missing_edr_bound_bytes,
        operational_overhead_bytes,
        minimum_free_after_g0_bytes,
    ) < 0:
        raise ValueError("G0 capacity byte bounds cannot be negative")

    source = _object(source_path)
    edr = _object(edr_path)
    if (
        edr.get("state") != "complete"
        or int(edr.get("failed_observation_count", -1)) != 0
        or int(edr.get("terminal_observation_count", -1))
        != int(edr.get("requested_observation_count", -2))
    ):
        raise ValueError("G0 capacity preflight requires a complete failure-free EDR index")
    resolved = {
        str(item.get("observation_id", ""))
        for item in edr.get("entries", [])
        if dict(item).get("status") == "resolved"
    }
    selected_products = _complete_source_products(source, expected_group_count)
    observations: list[dict[str, Any]] = []
    for selected in selected_products:
        product_observations = tuple(
            value for value in selected["product_id"].split("__") if value
        )
        if len(product_observations) != 2:
            raise ValueError("G0 source product is not a two-observation CTX stereo ID")
        observation = next((value for value in product_observations if value in resolved), None)
        if observation is None:
            raise ValueError("G0 source product has no resolved EDR observation")
        candidate = root / "data/assets/ctx-edr" / observation / f"{observation}.IMG"
        if candidate.is_file() and candidate.stat().st_size > 0:
            actual_bytes: int | None = candidate.stat().st_size
            estimated_bytes = actual_bytes
            size_basis = "existing_validated_candidate_bytes"
        else:
            actual_bytes = None
            estimated_bytes = missing_edr_bound_bytes
            size_basis = "frozen_missing_edr_upper_bound"
        observations.append(
            {
                **selected,
                "observation_id": observation,
                "edr_relative_path": candidate.relative_to(root).as_posix(),
                "actual_edr_bytes": actual_bytes,
                "estimated_edr_bytes": estimated_bytes,
                "size_basis": size_basis,
            }
        )
    observation_ids = [item["observation_id"] for item in observations]
    if len(set(observation_ids)) != expected_group_count:
        raise ValueError("G0 selected primary observations are not unique")

    estimated_edr_bytes = sum(int(item["estimated_edr_bytes"]) for item in observations)
    estimated_isis_bytes = int(math.ceil(estimated_edr_bytes * expansion_factor))
    measured_existing_isis_bytes, resume_records, resume_state = (
        _verified_existing_isis_artifacts(
            processing_index=processing_path,
            selected_observations=observations,
            source_contract_sha256=(
                str(source["contract_sha256"]) if "contract_sha256" in source else None
            ),
            edr_index_sha256=file_sha256(edr_path),
            paper2_root=root,
            allow_cross_generation_resume_credit=allow_cross_generation_resume_credit,
        )
    )
    credited_existing_isis_bytes = min(measured_existing_isis_bytes, estimated_isis_bytes)
    estimated_remaining_isis_bytes = max(
        0, estimated_isis_bytes - credited_existing_isis_bytes
    )
    estimated_additional_bytes = estimated_remaining_isis_bytes + operational_overhead_bytes
    usage = disk_usage(root)
    required_available_bytes = estimated_additional_bytes + minimum_free_after_g0_bytes
    sufficient = int(usage.free) >= required_available_bytes
    payload: dict[str, Any] = {
        "schema": G0_CAPACITY_SCHEMA,
        "state": "pass" if sufficient else "insufficient",
        "formal_g0_launch_authorized": sufficient,
        "source_index_relative_path": source_path.relative_to(root).as_posix(),
        "source_index_sha256": file_sha256(source_path),
        "edr_index_relative_path": edr_path.relative_to(root).as_posix(),
        "edr_index_sha256": file_sha256(edr_path),
        "processing_index_relative_path": (
            processing_path.relative_to(root).as_posix()
            if processing_path is not None and processing_path.is_file()
            else None
        ),
        "processing_index_sha256": (
            file_sha256(processing_path)
            if processing_path is not None and processing_path.is_file()
            else None
        ),
        "expected_group_count": expected_group_count,
        "selected_observations": observations,
        "model": {
            "expansion_factor_over_selected_edr_bytes": expansion_factor,
            "missing_edr_bound_bytes": missing_edr_bound_bytes,
            "operational_overhead_bytes": operational_overhead_bytes,
            "minimum_free_after_g0_bytes": minimum_free_after_g0_bytes,
            "conservative_double_count_existing_g0_outputs": processing_path is None
            or not processing_path.is_file(),
            "cross_generation_resume_credit_authorized": bool(
                allow_cross_generation_resume_credit
            ),
            "existing_artifact_credit_basis": (
                "none_fresh_launch"
                if resume_state == "not_available"
                else "current group-state and ISIS-manifest hashes plus declared regular files"
            ),
            "calibration_note": (
                "20.5 exceeds the observed 19.902 full-chain artifact/EDR ratio of the "
                "P08 spatial-summing-1 pilot"
            ),
        },
        "estimated_selected_edr_bytes": estimated_edr_bytes,
        "estimated_isis_artifact_bytes": estimated_isis_bytes,
        "resume_state": resume_state,
        "resume_products": resume_records,
        "measured_existing_isis_artifact_bytes": measured_existing_isis_bytes,
        "credited_existing_isis_artifact_bytes": credited_existing_isis_bytes,
        "estimated_remaining_isis_artifact_bytes": estimated_remaining_isis_bytes,
        "estimated_additional_bytes": estimated_additional_bytes,
        "filesystem_total_bytes": int(usage.total),
        "filesystem_used_bytes": int(usage.used),
        "filesystem_available_bytes": int(usage.free),
        "required_available_bytes": required_available_bytes,
        "available_margin_bytes": int(usage.free) - required_available_bytes,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    write_canonical_json(output_path, payload, paper2_root=root)
    return payload
