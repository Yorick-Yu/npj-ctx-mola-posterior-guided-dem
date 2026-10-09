"""Recoverable pruning of ISIS cubes from superseded pre-formal G0 groups."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO, Mapping
import fcntl
import os

from .g0_processing import (
    QA_EVIDENCE_SCHEMA,
    QA_REJECTION_EVIDENCE_SCHEMA,
    _load_ctx_qa_evidence,
    _load_ctx_qa_rejection_evidence,
    processing_exhausted_group_ids,
)
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)
from .source_preparation import source_exhausted_group_ids


PLAN_SCHEMA = "paper2.g0.superseded-replacement-cube-prune-plan/v1"
COMPLETION_SCHEMA = "paper2.g0.superseded-replacement-cube-prune-completion/v1"
ASSIGNMENT_SCHEMA = "paper2.g0.role-assignments/v1"
PROCESSING_SCHEMA = "paper2.g0.ctx-processing-index/v1"
SOURCE_INDEX_SCHEMA = "paper2.g0.source-preparation-index/v1"
PRUNABLE_CUBE_NAMES = {
    "ctx_raw.cub",
    "ctx_cal.cub",
    "ctx_evenodd.cub",
    "ctx_cam2map.cub",
}


def _object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _verify_content_hash(payload: Mapping[str, Any], *, field: str) -> None:
    declared = payload.get(field)
    unhashed = dict(payload)
    unhashed.pop(field, None)
    if declared != canonical_sha256(unhashed):
        raise ValueError(f"{field} does not match canonical content")


def _signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "byte_count": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _assignment_products(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("replacement pruning requires non-empty assignments")
    result: dict[str, set[str]] = {}
    for raw in assignments:
        assignment = dict(raw)
        group_id = str(assignment.get("group_id", ""))
        products = {
            str(assignment.get("representative_product_id", "")),
            *(str(value) for value in assignment.get("fallback_product_ids", [])),
        }
        if not group_id or "" in products or group_id in result:
            raise ValueError("assignment group/product identities are invalid")
        result[group_id] = products
    return result


def _lock_products(
    root: Path, product_ids: set[str], stack: ExitStack
) -> list[BinaryIO]:
    streams: list[BinaryIO] = []
    for product_id in sorted(product_ids):
        unresolved_product = root / "data/g0-processing/products" / product_id
        if unresolved_product.is_symlink():
            raise ValueError(f"replacement pruning refuses a symlinked product: {product_id}")
        product = guard_paper2_path(unresolved_product, paper2_root=root)
        if not product.exists():
            continue
        lock = product / ".processing.lock"
        stream = stack.enter_context(lock.open("a+b"))
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"replacement pruning refused an active product: {product_id}"
            ) from exc
        streams.append(stream)
    return streams


def _verify_failed_evidence(
    *, attempt: Mapping[str, Any], root: Path, mola_source_lock_sha256: str
) -> dict[str, Any] | None:
    if attempt.get("status") != "qa_failed":
        return None
    product_id = str(attempt.get("product_id", ""))
    evidence_path = guard_paper2_path(
        str(attempt.get("qa_evidence_path", "")), paper2_root=root
    )
    if (
        not evidence_path.is_file()
        or attempt.get("qa_evidence_sha256") != file_sha256(evidence_path)
    ):
        raise ValueError(f"superseded QA evidence differs: {product_id}")
    header = _object(evidence_path)
    common = {
        "path": evidence_path,
        "product_id": product_id,
        "observation_id": str(dict(attempt.get("edr_resolution", {})).get("observation_id", "")),
        "edr_sha256": str(attempt.get("edr_sha256", "")),
        "isis_run_manifest_sha256": str(attempt.get("isis_run_manifest_sha256", "")),
        "mola_source_lock_sha256": mola_source_lock_sha256,
        "paper2_root": root,
    }
    if header.get("schema") == QA_REJECTION_EVIDENCE_SCHEMA:
        _, qa, _ = _load_ctx_qa_rejection_evidence(**common)
    elif header.get("schema") == QA_EVIDENCE_SCHEMA:
        _, qa, _ = _load_ctx_qa_evidence(**common)
    else:
        raise ValueError(f"superseded QA evidence schema differs: {product_id}")
    if qa.passed:
        raise ValueError(f"replacement pruning refuses QA-passed evidence: {product_id}")
    return {
        "product_id": product_id,
        "relative_path": evidence_path.relative_to(root).as_posix(),
        "sha256": file_sha256(evidence_path),
        "content_sha256": header["evidence_sha256"],
        "schema": header["schema"],
    }


def _candidate_records(root: Path, product_ids: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for product_id in sorted(product_ids):
        product = guard_paper2_path(
            root / "data/g0-processing/products" / product_id,
            paper2_root=root,
        )
        if not product.exists():
            continue
        for candidate in sorted((product / "isis").glob("*/*.cub")):
            relative = candidate.relative_to(root)
            if candidate.name not in PRUNABLE_CUBE_NAMES:
                raise ValueError(f"unexpected replacement cube target: {relative}")
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"replacement cube target is not a regular file: {relative}")
            guard_paper2_path(candidate, paper2_root=root)
            manifest_path = candidate.parent / "isis-run-manifest.json"
            manifest_sha256: str | None = None
            declared_sha256: str | None = None
            if manifest_path.is_file():
                manifest = _object(manifest_path)
                artifact = dict(dict(manifest.get("artifacts", {})).get(candidate.name, {}))
                if artifact.get("relative_path") != relative.as_posix():
                    raise ValueError("replacement cube manifest path differs")
                declared_sha256 = str(artifact.get("sha256", ""))
                manifest_sha256 = file_sha256(manifest_path)
            before = _signature(candidate)
            digest = file_sha256(candidate)
            after = _signature(candidate)
            if before != after:
                raise RuntimeError(f"replacement cube changed while hashing: {relative}")
            if declared_sha256 is not None and declared_sha256 != digest:
                raise ValueError(f"replacement cube manifest hash differs: {relative}")
            records.append(
                {
                    "product_id": product_id,
                    "relative_path": relative.as_posix(),
                    "sha256": digest,
                    "signature": after,
                    "isis_run_manifest_sha256": manifest_sha256,
                    "reconstructable_from_preserved_edr_and_frozen_commands": True,
                }
            )
    return records


def _validate_resume_plan(
    plan: Mapping[str, Any], *, expected: Mapping[str, str]
) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("state") != "authorized":
        raise ValueError("replacement prune resume plan identity differs")
    _verify_content_hash(plan, field="content_sha256")
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("replacement prune resume plan parents differ")


def prune_superseded_replacement_cubes(
    *,
    paper2_root: str | Path,
    previous_assignments: str | Path,
    next_assignments: str | Path,
    previous_source_index: str | Path,
    previous_processing_index: str | Path,
    plan_output: str | Path,
    completion_output: str | Path,
) -> dict[str, Any]:
    """Hash, receipt, and remove only cubes from groups replaced before G0 freeze."""

    root = Path(paper2_root).resolve()
    previous_path = guard_paper2_path(previous_assignments, paper2_root=root)
    next_path = guard_paper2_path(next_assignments, paper2_root=root)
    source_path = guard_paper2_path(previous_source_index, paper2_root=root)
    processing_path = guard_paper2_path(previous_processing_index, paper2_root=root)
    plan_path = guard_paper2_path(plan_output, paper2_root=root)
    completion_path = guard_paper2_path(completion_output, paper2_root=root)
    for required in (previous_path, next_path, source_path, processing_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    formal_locks = (
        root / "data/assets/processed-lock.json",
        root / "data/locks/asset-lock.json",
        root / "data/assets/asset-lock.json",
    )
    if any(path.exists() for path in formal_locks):
        raise RuntimeError("replacement pruning refused after a formal G0 lock")

    previous = _object(previous_path)
    next_payload = _object(next_path)
    source = _object(source_path)
    processing = _object(processing_path)
    for payload in (previous, next_payload):
        if payload.get("schema") != ASSIGNMENT_SCHEMA:
            raise ValueError("replacement pruning assignment schema differs")
        _verify_content_hash(payload, field="content_sha256")
    if processing.get("schema") != PROCESSING_SCHEMA:
        raise ValueError("replacement pruning processing schema differs")
    if source.get("schema") != SOURCE_INDEX_SCHEMA:
        raise ValueError("replacement pruning source-index schema differs")
    _verify_content_hash(source, field="content_sha256")
    if (
        next_payload.get("parent_assignments_file_sha256") != file_sha256(previous_path)
        or next_payload.get("source_preparation_index_file_sha256")
        != file_sha256(source_path)
        or next_payload.get("processing_index_file_sha256") != file_sha256(processing_path)
        or next_payload.get("formal_test_unlocked") is not False
        or next_payload.get("reference_dem_accessed") is not False
        or next_payload.get("model_outcomes_accessed") is not False
    ):
        raise ValueError("replacement pruning parent/blind contract differs")
    previous_groups = _assignment_products(previous)
    next_groups = _assignment_products(next_payload)
    previous_assignments_list = [dict(item) for item in previous["assignments"]]
    if (
        source.get("assignments_file_sha256") != file_sha256(previous_path)
        or int(source.get("expected_group_count", -1)) != len(previous_groups)
    ):
        raise ValueError("replacement pruning source index differs from previous assignments")
    removed_group_ids = set(previous_groups).difference(next_groups)
    added_group_ids = set(next_groups).difference(previous_groups)
    if (
        not removed_group_ids
        or len(previous_groups) != len(next_groups)
        or len(removed_group_ids) != len(added_group_ids)
    ):
        raise ValueError("replacement pruning group delta is not one-for-one")
    history = next_payload.get("qa_replacement_history")
    if not isinstance(history, list) or not history:
        raise ValueError("replacement pruning lacks replacement history")
    latest_cycle = max(int(dict(item).get("cycle", -1)) for item in history)
    latest = [dict(item) for item in history if int(dict(item).get("cycle", -1)) == latest_cycle]
    if (
        {str(item.get("removed_group_id", "")) for item in latest} != removed_group_ids
        or {str(item.get("replacement_group_id", "")) for item in latest} != added_group_ids
        or any(
            item.get("reference_dem_accessed") is not False
            or item.get("model_outcomes_accessed") is not False
            or not item.get("qa_failure_evidence_sha256s")
            for item in latest
        )
    ):
        raise ValueError("replacement pruning latest-cycle evidence differs")

    next_product_ids = set().union(*next_groups.values())
    removed_product_ids = set().union(
        *(previous_groups[group_id] for group_id in removed_group_ids)
    )
    if removed_product_ids.intersection(next_product_ids):
        raise ValueError("replacement pruning refuses a product retained by the next generation")

    expected_parents = {
        "previous_assignments_file_sha256": file_sha256(previous_path),
        "next_assignments_file_sha256": file_sha256(next_path),
        "previous_source_index_file_sha256": file_sha256(source_path),
        "previous_processing_index_file_sha256": file_sha256(processing_path),
    }
    if completion_path.is_file():
        completion = _object(completion_path)
        if completion.get("schema") != COMPLETION_SCHEMA:
            raise ValueError("replacement prune completion schema differs")
        _verify_content_hash(completion, field="content_sha256")
        if any(completion.get(key) != value for key, value in expected_parents.items()):
            raise ValueError("replacement prune completion parents differ")
        for item in completion["removed"]:
            unresolved = root / str(item["relative_path"])
            if unresolved.is_symlink():
                raise ValueError("replacement prune completion target became a symlink")
            candidate = guard_paper2_path(unresolved, paper2_root=root)
            if candidate.exists():
                raise ValueError("replacement prune completion has a surviving target")
        return {**completion, "resumed": True}

    processing_groups = {
        str(dict(group).get("group_id", "")): dict(group)
        for group in processing.get("groups", [])
    }
    processing_generation_matches_previous = (
        processing.get("assignments_file_sha256") == file_sha256(previous_path)
    )
    if (
        not processing_generation_matches_previous
        and removed_group_ids.intersection(processing_groups)
    ):
        raise ValueError(
            "replacement pruning found removed groups in a differently bound processing index"
        )
    source_exhausted = set(source_exhausted_group_ids(source))
    processing_exhausted = (
        set(processing_exhausted_group_ids(processing, previous_assignments_list))
        if processing_generation_matches_previous
        else set()
    )
    if not removed_group_ids.issubset(source_exhausted | processing_exhausted):
        raise ValueError("replacement pruning group lacks independently verified exhaustion")
    latest_by_removed = {str(item["removed_group_id"]): item for item in latest}
    source_groups = {
        str(dict(group).get("group_id", "")): dict(group)
        for group in source.get("groups", [])
    }
    if len(source_groups) != len(source.get("groups", [])):
        raise ValueError("replacement pruning source groups are duplicated")
    evidence_records: list[dict[str, Any]] = []
    processing_product_ids: set[str] = set()
    for group_id in sorted(removed_group_ids):
        group = processing_groups.get(group_id)
        expected_failure_hashes: set[str] = set()
        if group_id in processing_exhausted:
            if group is None:
                raise ValueError("replacement pruning lacks an exhausted processing group")
            if group.get("state") == "complete" or group.get("selected_product_id") is not None:
                raise ValueError("replacement pruning refuses a completed/selected group")
            group_state = guard_paper2_path(
                str(group.get("group_state_path", "")), paper2_root=root
            )
            if (
                not group_state.is_file()
                or group.get("group_state_sha256") != file_sha256(group_state)
            ):
                raise ValueError("replacement pruning group-state bytes differ")
            expected_failure_hashes.add(str(group["group_state_sha256"]))
            for raw_attempt in group.get("attempts", []):
                attempt = dict(raw_attempt)
                product_id = str(attempt.get("product_id", ""))
                if product_id:
                    processing_product_ids.add(product_id)
                if attempt.get("qa_evidence_sha256") is not None:
                    expected_failure_hashes.add(str(attempt["qa_evidence_sha256"]))
                if not plan_path.is_file():
                    evidence = _verify_failed_evidence(
                        attempt=attempt,
                        root=root,
                        mola_source_lock_sha256=str(
                            processing.get("mola_source_lock_sha256", "")
                        ),
                    )
                    if evidence is not None:
                        evidence_records.append(evidence)
        else:
            source_group = source_groups.get(group_id)
            if source_group is None:
                raise ValueError("replacement pruning lacks an exhausted source group")
            for raw_attempt in source_group.get("attempts", []):
                attempt = dict(raw_attempt)
                if attempt.get("status") != "source_rejected":
                    continue
                failure_path = guard_paper2_path(
                    str(attempt.get("failure_relative_path", "")), paper2_root=root
                )
                if (
                    not failure_path.is_file()
                    or attempt.get("failure_sha256") != file_sha256(failure_path)
                ):
                    raise ValueError("replacement pruning source-failure bytes differ")
                failure = _object(failure_path)
                _verify_content_hash(failure, field="content_sha256")
                expected_failure_hashes.add(str(attempt["failure_sha256"]))
        if sorted(expected_failure_hashes) != sorted(
            str(value)
            for value in latest_by_removed[group_id]["qa_failure_evidence_sha256s"]
        ):
            raise ValueError("replacement pruning history failure hashes differ")
    if processing_product_ids.difference(removed_product_ids):
        raise ValueError("replacement pruning processing products escape removed assignments")

    with ExitStack() as stack:
        _lock_products(root, removed_product_ids, stack)
        if plan_path.is_file():
            plan = _object(plan_path)
            _validate_resume_plan(plan, expected=expected_parents)
        else:
            candidates = _candidate_records(root, removed_product_ids)
            if candidates and not processing_generation_matches_previous:
                raise ValueError(
                    "replacement pruning refuses cubes without a matching processing generation"
                )
            plan = {
                "schema": PLAN_SCHEMA,
                "state": "authorized",
                **expected_parents,
                "latest_replacement_cycle": latest_cycle,
                "removed_group_ids": sorted(removed_group_ids),
                "added_group_ids": sorted(added_group_ids),
                "removed_product_ids": sorted(removed_product_ids),
                "verified_failed_qa_evidence": sorted(
                    evidence_records, key=lambda item: item["product_id"]
                ),
                "candidate_count": len(candidates),
                "candidate_bytes": sum(
                    int(item["signature"]["byte_count"]) for item in candidates
                ),
                "candidates": candidates,
                "processing_generation_matches_previous_assignments": (
                    processing_generation_matches_previous
                ),
                "preserved": {
                    "ctx_edr": True,
                    "source_assets": True,
                    "isis_run_manifests": True,
                    "ctx_isd": True,
                    "qa_evidence": True,
                    "registered_and_unregistered_target_grid_arrays": True,
                    "lineage_and_tiepoints": True,
                    "processing_and_assignment_indexes": True,
                },
                "reference_dem_accessed": False,
                "model_outcomes_accessed": False,
                "formal_test_unlocked": False,
            }
            plan["content_sha256"] = canonical_sha256(plan)
            write_canonical_json(plan_path, plan, paper2_root=root)

        removed: list[dict[str, Any]] = []
        for raw in plan.get("candidates", []):
            record = dict(raw)
            unresolved = root / str(record.get("relative_path", ""))
            if unresolved.is_symlink():
                raise ValueError("replacement prune resume target became a symlink")
            candidate = guard_paper2_path(unresolved, paper2_root=root)
            if candidate.exists():
                if candidate.is_symlink() or not candidate.is_file():
                    raise ValueError("replacement prune resume target is not a regular file")
                if _signature(candidate) != record.get("signature"):
                    raise RuntimeError(f"replacement cube changed after plan: {candidate}")
                candidate.unlink()
            if candidate.exists():
                raise RuntimeError(f"replacement cube remains after unlink: {candidate}")
            removed.append(
                {
                    "relative_path": str(record["relative_path"]),
                    "sha256": str(record["sha256"]),
                    "byte_count": int(dict(record["signature"])["byte_count"]),
                }
            )
        for directory in sorted({(root / item["relative_path"]).parent for item in removed}):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        completion: dict[str, Any] = {
            "schema": COMPLETION_SCHEMA,
            "state": "complete",
            **expected_parents,
            "plan_relative_path": plan_path.relative_to(root).as_posix(),
            "plan_file_sha256": file_sha256(plan_path),
            "plan_content_sha256": plan["content_sha256"],
            "removed_count": len(removed),
            "removed_bytes": sum(item["byte_count"] for item in removed),
            "removed": removed,
            "source_edr_removed": False,
            "source_asset_removed": False,
            "qa_or_lineage_evidence_removed": False,
            "isis_manifest_or_isd_removed": False,
            "reconstruction_required_only_for_superseded_preformal_products": True,
        }
        completion["content_sha256"] = canonical_sha256(completion)
        write_canonical_json(completion_path, completion, paper2_root=root)
        return completion
