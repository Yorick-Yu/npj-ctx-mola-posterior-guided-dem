"""Audited pruning of pre-formal, reproducible stale ISIS cube intermediates."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO
import fcntl
import os

from .hashing import canonical_sha256, file_sha256, guard_paper2_path, read_json
from .hashing import write_canonical_json
from .preparation import verify_ale_spiceql_compatibility


G0_PRUNE_PLAN_SCHEMA = "paper2.g0.stale-isis-prune-plan/v1"
G0_PRUNE_COMPLETION_SCHEMA = "paper2.g0.stale-isis-prune-completion/v1"
PRUNABLE_CUBE_NAMES = {
    "ctx_raw.cub",
    "ctx_cal.cub",
    "ctx_evenodd.cub",
    "ctx_cam2map.cub",
}
PRUNABLE_PRODUCT_STATES = {
    "awaiting_qa_evidence",
    "download_failed",
    "isis_failed",
}


def _object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "byte_count": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _lock_streams(root: Path, stack: ExitStack) -> tuple[BinaryIO, BinaryIO]:
    streams: list[BinaryIO] = []
    for relative in (
        "reports/gates/G0/formal-g0-lifecycle.lock",
        "reports/gates/G0/formal-g0-server.lock",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = stack.enter_context(path.open("a+b"))
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("G0 pruning refused while a lifecycle process holds its lock") from exc
        streams.append(stream)
    return streams[0], streams[1]


def prune_stale_g0_isis_cubes(
    *,
    paper2_root: str | Path,
    plan_output: str | Path,
    completion_output: str | Path,
    current_compatibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    plan_path = guard_paper2_path(plan_output, paper2_root=root)
    completion_path = guard_paper2_path(completion_output, paper2_root=root)
    if plan_path.exists() or completion_path.exists():
        raise FileExistsError("G0 prune plan and completion paths must both be new")
    formal_locks = (
        root / "data/assets/processed-lock.json",
        root / "data/locks/asset-lock.json",
        root / "data/assets/asset-lock.json",
    )
    present_locks = [path for path in formal_locks if path.exists()]
    if present_locks:
        raise RuntimeError(f"G0 pruning refused after a formal lock exists: {present_locks}")
    compatibility = dict(
        current_compatibility or verify_ale_spiceql_compatibility(root)
    )
    current_module_sha256 = str(compatibility.get("module_sha256", ""))
    if len(current_module_sha256) != 64:
        raise ValueError("current ALE/SpiceQL compatibility binding is invalid")

    with ExitStack() as stack:
        _lock_streams(root, stack)
        candidates: list[dict[str, Any]] = []
        preserved_current_manifest_cubes: list[dict[str, Any]] = []
        pattern = root / "data/g0-processing/products"
        for candidate in sorted(pattern.glob("*/isis/*/*.cub")):
            relative = candidate.relative_to(root)
            if (
                len(relative.parts) != 7
                or relative.parts[:3] != ("data", "g0-processing", "products")
                or relative.parts[4] != "isis"
                or candidate.name not in PRUNABLE_CUBE_NAMES
            ):
                raise ValueError(f"unexpected G0 cube pruning target: {relative}")
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"G0 cube pruning target is not a regular file: {relative}")
            guard_paper2_path(candidate, paper2_root=root)
            product_directory = candidate.parents[2]
            attempt_directory = candidate.parent
            state_path = product_directory / "state.json"
            state = _object(state_path)
            if state.get("status") not in PRUNABLE_PRODUCT_STATES:
                raise RuntimeError(
                    f"G0 cube belongs to a non-prunable product state: {state.get('status')}"
                )
            manifest_path = attempt_directory / "isis-run-manifest.json"
            manifest_sha256: str | None = None
            stale_reason = "failed_attempt_without_run_manifest"
            if manifest_path.is_file():
                manifest = _object(manifest_path)
                old_compatibility = manifest.get("ale_spiceql_compatibility")
                old_module_sha256 = (
                    str(old_compatibility.get("module_sha256", ""))
                    if isinstance(old_compatibility, Mapping)
                    else ""
                )
                if old_module_sha256 == current_module_sha256:
                    artifacts = manifest.get("artifacts")
                    artifact = (
                        artifacts.get(candidate.name)
                        if isinstance(artifacts, Mapping)
                        else None
                    )
                    if (
                        not isinstance(artifact, Mapping)
                        or artifact.get("relative_path") != relative.as_posix()
                        or artifact.get("sha256") != file_sha256(candidate)
                    ):
                        raise RuntimeError(
                            "current-transport G0 cube is not bound by its run manifest"
                        )
                    preserved_current_manifest_cubes.append(
                        {
                            "relative_path": relative.as_posix(),
                            "sha256": str(artifact["sha256"]),
                            "signature": _signature(candidate),
                            "product_state": str(state["status"]),
                            "isis_run_manifest_sha256": file_sha256(manifest_path),
                            "preserve_reason": "run_manifest_bound_to_current_spiceql_transport",
                        }
                    )
                    continue
                manifest_sha256 = file_sha256(manifest_path)
                stale_reason = "run_manifest_bound_to_superseded_spiceql_transport"
            before = _signature(candidate)
            digest = file_sha256(candidate)
            after = _signature(candidate)
            if before != after:
                raise RuntimeError(f"G0 cube changed while hashing: {relative}")
            candidates.append(
                {
                    "relative_path": relative.as_posix(),
                    "sha256": digest,
                    "signature": after,
                    "product_state": str(state["status"]),
                    "product_state_relative_path": state_path.relative_to(root).as_posix(),
                    "product_state_sha256": file_sha256(state_path),
                    "isis_run_manifest_sha256": manifest_sha256,
                    "stale_reason": stale_reason,
                    "reconstructable_from_locked_edr_and_frozen_commands": True,
                }
            )
        if not candidates:
            raise RuntimeError("G0 stale ISIS prune found no whitelisted cube files")

        plan: dict[str, Any] = {
            "schema": G0_PRUNE_PLAN_SCHEMA,
            "state": "authorized",
            "formal_locks_absent": True,
            "lifecycle_locks_held": True,
            "current_ale_spiceql_compatibility": compatibility,
            "whitelisted_filenames": sorted(PRUNABLE_CUBE_NAMES),
            "candidate_count": len(candidates),
            "candidate_bytes": sum(item["signature"]["byte_count"] for item in candidates),
            "candidates": candidates,
            "preserved_current_manifest_cube_count": len(
                preserved_current_manifest_cubes
            ),
            "preserved_current_manifest_cube_bytes": sum(
                item["signature"]["byte_count"]
                for item in preserved_current_manifest_cubes
            ),
            "preserved_current_manifest_cubes": preserved_current_manifest_cubes,
            "preserved_patterns": [
                "*.IMG",
                "ctx_isd.json",
                "isis-run-manifest.json",
                "state.json",
                "print.prt",
            ],
        }
        plan["content_sha256"] = canonical_sha256(plan)
        write_canonical_json(plan_path, plan, paper2_root=root)
        plan_file_sha256 = file_sha256(plan_path)

        removed: list[dict[str, Any]] = []
        for record in candidates:
            candidate = guard_paper2_path(record["relative_path"], paper2_root=root)
            if _signature(candidate) != record["signature"]:
                raise RuntimeError(f"G0 cube changed after prune plan publication: {candidate}")
            candidate.unlink()
            if candidate.exists():
                raise RuntimeError(f"G0 cube remains after unlink: {candidate}")
            removed.append(
                {
                    "relative_path": record["relative_path"],
                    "sha256": record["sha256"],
                    "byte_count": record["signature"]["byte_count"],
                }
            )
        for directory in sorted({(root / item["relative_path"]).parent for item in removed}):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        completion: dict[str, Any] = {
            "schema": G0_PRUNE_COMPLETION_SCHEMA,
            "state": "complete",
            "plan_relative_path": plan_path.relative_to(root).as_posix(),
            "plan_file_sha256": plan_file_sha256,
            "plan_content_sha256": plan["content_sha256"],
            "removed_count": len(removed),
            "removed_bytes": sum(item["byte_count"] for item in removed),
            "removed": removed,
            "source_edr_removed": False,
            "isd_or_manifest_removed": False,
            "preserved_current_manifest_cube_count": len(
                preserved_current_manifest_cubes
            ),
            "reconstruction_required_for_formal_g0": True,
        }
        completion["content_sha256"] = canonical_sha256(completion)
        write_canonical_json(completion_path, completion, paper2_root=root)
        return completion
