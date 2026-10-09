"""Production input validation and torch runner for the G2/G3 gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from datetime import timedelta
from importlib import metadata as importlib_metadata
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlparse

import numpy as np
import yaml

from .assets import audit_asset_lock_geography
from .contracts import AssetLock, PlanContract, ReconstructionProblem
from .hashing import canonical_sha256, file_sha256, guard_paper2_path, is_sha256
from .models import TORCH_AVAILABLE, UNetConfig, build_model
from .posterior_preparation import (
    REQUIRED_TRAINING_SOURCE_HASHES,
    verify_gp_posterior_manifest,
    verify_training_patch_derivation,
)
from .training import configure_determinism, train_model
from .training_gates import (
    REAL_DATA_SCOPE,
    GateEvidence,
    GateProtocol,
    GateRunReceipt,
    GateRunner,
    GateStageSpec,
    PRODUCTION_EXECUTION_SCOPE,
    _execute_training_gates,
    _validate_completed_prefix,
    gate_stages,
    training_config_for_stage,
)


MAXIMUM_CONFIG_SCHEMA = "paper2.g3-maximum-input/v1"
TRAINING_SOURCE_SCHEMA = "paper2.training-source-manifest/v1"
DERIVATION_RECEIPT_SCHEMA = "paper2.training-derivation-receipt/v1"
ENVIRONMENT_LOCK_SCHEMA = "paper2.training-environment-lock/v1"
LOCAL_DISTRIBUTED_ENVIRONMENT = {
    # This server's configured external DNS resolver is not reachable from the
    # training process.  c10d performs reverse lookups even for a loopback
    # TCPStore, so bound the process-local resolver instead of changing the
    # machine's /etc/resolv.conf.
    "RES_OPTIONS": "attempts:1 timeout:1",
    "GLOO_SOCKET_IFNAME": "lo",
    "NCCL_SOCKET_IFNAME": "lo",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    # Required for deterministic CUDA >= 10.2 matrix multiplications.
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
REQUEST_SCHEMA = "paper2.training-gate-request/v1"


def _filestore_process_group_kwargs(
    *, paper2_root: Path, rank: int, world_size: int
) -> dict[str, Any]:
    """Validate the optional, Paper2-contained FileStore rendezvous contract."""

    value = os.environ.get("PAPER2_DDP_INIT_METHOD")
    if value is None:
        return {}
    parsed = urlparse(value)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("PAPER2_DDP_INIT_METHOD must be a local file URI")
    rendezvous = guard_paper2_path(
        Path(unquote(parsed.path)), paper2_root=paper2_root, allow_root=True
    )
    if rendezvous == paper2_root:
        raise ValueError("FileStore rendezvous must be below the Paper2 root")
    return {
        "init_method": rendezvous.as_uri(),
        "rank": rank,
        "world_size": world_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_input(path: str | Path, *, root: Path) -> Path:
    result = guard_paper2_path(path, paper2_root=root)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _relative(path: Path, root: Path) -> str:
    return guard_paper2_path(path, paper2_root=root).relative_to(root).as_posix()


def paper2_training_code_sha256(root: Path) -> str:
    records = []
    for directory in (root / "src", root / "scripts"):
        for path in sorted(directory.rglob("*.py")):
            records.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                }
            )
    if not records:
        raise ValueError("Paper2 training code tree contains no Python source files")
    return canonical_sha256(records)


def collect_training_environment(*, paper2_root: str | Path) -> dict[str, Any]:
    """Collect the live interpreter, package, torch, CUDA and physical-GPU identity."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("formal training environment collection requires PyTorch")
    import torch

    root = Path(paper2_root).resolve()
    executable = Path(sys.executable).resolve()
    try:
        executable_relative = executable.relative_to(root).as_posix()
    except ValueError as exc:
        raise PermissionError("formal training interpreter must be inside paper2/") from exc
    packages = sorted(
        {
            (
                str(distribution.metadata.get("Name") or distribution.name).lower(),
                str(distribution.version),
            )
            for distribution in importlib_metadata.distributions()
        }
    )
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            device_uuid = str(getattr(properties, "uuid", ""))
            if not device_uuid:
                raise RuntimeError("CUDA runtime does not expose a physical GPU UUID")
            devices.append(
                {
                    "index": index,
                    "uuid": device_uuid,
                    "name": str(properties.name),
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                }
            )
    snapshot: dict[str, Any] = {
        "python_executable": executable_relative,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": (
            None if torch.backends.cudnn.version() is None else int(torch.backends.cudnn.version())
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_devices": devices,
        "local_distributed_environment": dict(LOCAL_DISTRIBUTED_ENVIRONMENT),
        "package_inventory_sha256": canonical_sha256(packages),
        "package_count": len(packages),
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    return snapshot


def build_training_environment_lock(
    *, paper2_root: str | Path, output: str | Path
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    output_path = guard_paper2_path(output, paper2_root=root)
    payload: dict[str, Any] = {
        "schema": ENVIRONMENT_LOCK_SCHEMA,
        "state": "ready",
        "required_world_size": 2,
        "runtime": collect_training_environment(paper2_root=root),
    }
    if not payload["runtime"]["cuda_available"] or payload["runtime"]["gpu_count"] != 2:
        raise RuntimeError("formal training environment requires exactly two visible CUDA GPUs")
    payload["lock_sha256"] = canonical_sha256(payload)
    if output_path.exists():
        if _read_json(output_path) != payload:
            raise FileExistsError("refusing to replace a different training environment lock")
    else:
        _atomic_json(output_path, payload)
    return payload


def _validate_live_environment_lock(*, root: Path, path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    declared = dict(payload)
    declared_hash = str(declared.pop("lock_sha256", ""))
    if (
        payload.get("schema") != ENVIRONMENT_LOCK_SCHEMA
        or payload.get("state") != "ready"
        or payload.get("required_world_size") != 2
        or declared_hash != canonical_sha256(declared)
    ):
        raise ValueError("training environment lock is not a valid ready lock")
    locked_runtime = payload.get("runtime")
    if not isinstance(locked_runtime, Mapping):
        raise ValueError("training environment lock lacks a runtime snapshot")
    runtime_content = dict(locked_runtime)
    runtime_hash = str(runtime_content.pop("snapshot_sha256", ""))
    if runtime_hash != canonical_sha256(runtime_content):
        raise ValueError("training environment snapshot content hash mismatch")
    live = collect_training_environment(paper2_root=root)
    if payload.get("runtime") != live:
        raise RuntimeError("live Python/torch/CUDA/GPU/package environment differs from its lock")
    return payload


def gate_protocol_from_yaml(path: str | Path, *, paper2_root: str | Path) -> GateProtocol:
    root = Path(paper2_root).resolve()
    config_path = _existing_input(path, root=root)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("gates"), dict):
        raise ValueError("protocol config lacks a gates mapping")
    gates = payload["gates"]
    protocol = GateProtocol(
        feasibility_updates=tuple(int(item) for item in gates["feasibility_updates"]),
        stability_steps=int(gates["stability_steps"]),
        overfit_sample_count=int(gates.get("overfit_sample_count", 32)),
        overfit_validation_ratio_max=float(
            gates.get("overfit_validation_ratio_max", 0.25)
        ),
        max_allocated_fraction=float(gates["max_allocated_fraction"]),
        max_reserved_fraction=float(gates["max_reserved_fraction"]),
        max_consecutive_loss_scale_backoffs=int(
            gates.get("max_consecutive_loss_scale_backoffs", 3)
        ),
    )
    protocol.validate()
    return protocol


def _source_hashes_by_tile(index_payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    raw_tiles = index_payload.get("tiles")
    if not isinstance(raw_tiles, list) or not raw_tiles:
        raise ValueError("training index has no tile contracts")
    tile_ids = {str(item.get("tile_id", "")) for item in raw_tiles}
    if "source_hashes_by_tile" in index_payload:
        raw = index_payload["source_hashes_by_tile"]
        if not isinstance(raw, Mapping):
            raise ValueError("source_hashes_by_tile must be an object")
        sources = {
            str(tile_id): {str(name): str(value) for name, value in dict(values).items()}
            for tile_id, values in raw.items()
        }
    else:
        raw = index_payload.get("source_hashes")
        if not isinstance(raw, Mapping) or len(tile_ids) != 1:
            raise ValueError("training index does not close source hashes over its tiles")
        sources = {
            next(iter(tile_ids)): {str(name): str(value) for name, value in raw.items()}
        }
    if set(sources) != tile_ids:
        raise ValueError("training index source hashes do not exactly cover its tile set")
    for tile_id, values in sources.items():
        missing = sorted(REQUIRED_TRAINING_SOURCE_HASHES.difference(values))
        if missing:
            raise ValueError(f"training tile {tile_id} lacks prepared-source hashes: {missing}")
        invalid = sorted(name for name, value in values.items() if not is_sha256(value))
        if invalid:
            raise ValueError(f"training tile {tile_id} has invalid source hashes: {invalid}")
    return sources


def _validate_role_authorization(
    *,
    asset_lock: AssetLock,
    role_payload: Mapping[str, Any],
    train_batches: Any,
    val_batches: Any,
) -> None:
    if role_payload.get("schema") != "paper2.role-manifest/v1":
        raise ValueError("role manifest schema is invalid")
    roles = role_payload.get("roles")
    groups = role_payload.get("geographic_groups")
    if not isinstance(roles, Mapping) or not isinstance(groups, Mapping):
        raise ValueError("role manifest lacks role and geographic-group maps")
    required_group_counts = {"bridge_train": 48, "bridge_val": 12}
    for indexed, expected_role in (
        (train_batches, "bridge_train"),
        (val_batches, "bridge_val"),
    ):
        role_tiles_raw = [str(item) for item in roles.get(expected_role, [])]
        role_groups_raw = [str(item) for item in groups.get(expected_role, [])]
        if len(role_tiles_raw) != len(set(role_tiles_raw)) or len(role_groups_raw) != len(
            set(role_groups_raw)
        ):
            raise ValueError(f"role manifest contains duplicate {expected_role} members")
        allowed_tiles = set(role_tiles_raw)
        allowed_groups = set(role_groups_raw)
        asset_tiles = {
            asset.product_id for asset in asset_lock.assets if asset.role == expected_role
        }
        asset_groups = {
            asset.geographic_group_id
            for asset in asset_lock.assets
            if asset.role == expected_role
        }
        indexed_tiles = {sample.tile_id for sample in indexed.all_samples}
        indexed_groups = {sample.geographic_group_id for sample in indexed.all_samples}
        if not (
            indexed_tiles == allowed_tiles == asset_tiles
            and indexed_groups == allowed_groups == asset_groups
        ):
            raise PermissionError(
                f"{expected_role} training index is not the exact frozen role/asset set"
            )
        if len(indexed_groups) != required_group_counts[expected_role]:
            raise ValueError(
                f"{expected_role} must cover exactly {required_group_counts[expected_role]} "
                "geographic groups"
            )
        for sample in indexed.all_samples:
            if sample.role != expected_role:
                raise PermissionError(f"training gate refuses role {sample.role}")
            if sample.tile_id not in allowed_tiles or sample.geographic_group_id not in allowed_groups:
                raise PermissionError("training patch is absent from the frozen role manifest")


def _validate_g0_parent_chain(
    *,
    asset_lock: AssetLock,
    ready_draft_path: Path,
    processed_lock_path: Path,
) -> None:
    draft = _read_json(ready_draft_path)
    draft_without_hash = {key: value for key, value in draft.items() if key != "draft_sha256"}
    if (
        draft.get("schema") != "paper2.g0.asset-lock-draft/v1"
        or draft.get("state") != "ready_for_formal_freeze"
        or draft.get("formal_test_unlocked") is not False
        or draft.get("blockers") != []
        or int(draft.get("selected_qa_passed_product_count", -1)) != 108
        or draft.get("draft_sha256") != canonical_sha256(draft_without_hash)
    ):
        raise ValueError("G0 parent draft is not a hash-valid ready-for-freeze artifact")
    bindings = draft.get("bindings")
    if not isinstance(bindings, Mapping) or bindings.get(
        "processed_lock_file_sha256"
    ) != file_sha256(processed_lock_path):
        raise ValueError("G0 ready draft does not bind the supplied processed lock")
    for name, digest in bindings.items():
        if name.endswith("_sha256") and not is_sha256(str(digest or "")):
            raise ValueError(f"G0 ready draft has an incomplete sidecar binding: {name}")
    processed = _read_json(processed_lock_path)
    if (
        processed.get("schema") != "paper2.g0.processed-lock/v1"
        or processed.get("state") != "complete"
        or int(processed.get("expected_group_count", -1)) != 108
        or int(processed.get("qa_passed_group_count", -1)) != 108
    ):
        raise ValueError("G0 processed lock is not complete over all 108 groups")
    expected_role_counts = {
        "bridge_train": 48,
        "bridge_val": 12,
        "uncertainty_calibration": 12,
        "paper2_locked_test": 24,
        "sealed_reserve": 12,
    }
    assets_by_group = {asset.geographic_group_id: asset for asset in asset_lock.assets}
    if len(assets_by_group) != 108:
        raise ValueError("formal G0 asset lock must select exactly one asset in each of 108 groups")
    observed_role_counts = {
        role: sum(asset.role == role for asset in asset_lock.assets)
        for role in expected_role_counts
    }
    if observed_role_counts != expected_role_counts:
        raise ValueError("formal G0 asset-lock role counts differ from the frozen 48/12/12/24/12")
    assignments = draft.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != 108:
        raise ValueError("G0 ready draft does not contain exactly 108 assignments")
    assignment_roles = {
        str(item.get("group_id", "")): str(item.get("role", "")) for item in assignments
    }
    if len(assignment_roles) != 108 or assignment_roles != {
        group_id: asset.role for group_id, asset in assets_by_group.items()
    }:
        raise ValueError("formal asset-lock roles differ from the G0 ready draft")
    selections = processed.get("group_selections")
    if not isinstance(selections, list) or len(selections) != 108:
        raise ValueError("G0 processed lock does not contain 108 group selections")
    selected = {
        str(item.get("group_id", "")): (
            str(item.get("role", "")),
            str(item.get("selected_product_id", "")),
        )
        for item in selections
    }
    if len(selected) != 108 or selected != {
        group_id: (asset.role, asset.product_id) for group_id, asset in assets_by_group.items()
    }:
        raise ValueError("formal asset lock differs from the complete G0 group selections")
    products = {
        str(item.get("product_id", "")): item for item in processed.get("products", [])
    }
    for asset in asset_lock.assets:
        record = products.get(asset.product_id)
        if (
            not isinstance(record, Mapping)
            or record.get("prepared_manifest_sha256") != asset.prepared_manifest_sha256
            or record.get("problem_sha256") != asset.problem_sha256
            or not isinstance(record.get("qa"), Mapping)
            or record["qa"].get("passed") is not True
            or record.get("coreg_aux_influenced_primary_pixels") is not True
            or record.get("coreg_aux_pixel_values_copied") is not False
            or record.get("registration_transform_family") != "translation_2d_only"
            or record.get("primary_intensity_source") != "ctx_edr_isis_cam2map_only"
            or record.get("rotation_scale_shear_or_local_warp_applied") is not False
        ):
            raise ValueError(
                f"formal asset {asset.product_id} differs from rigid-translation G0 evidence"
            )


def _validate_maximum_config(
    *,
    root: Path,
    path: Path,
    payload: Mapping[str, Any],
    train_index_path: Path,
    val_index_path: Path,
    asset_lock_path: Path,
    train_sources: Mapping[str, Mapping[str, str]],
    val_sources: Mapping[str, Mapping[str, str]],
    asset_lock: AssetLock,
) -> dict[str, Any]:
    if payload.get("schema") != MAXIMUM_CONFIG_SCHEMA:
        raise ValueError(f"maximum-input manifest must use {MAXIMUM_CONFIG_SCHEMA}")
    expected_fields = {
        "patch_size": 256,
        "image_count": 1,
        "laser_budget": 100,
        "actual_imagery": True,
        "full_training_validation_sampling_loop": True,
        "training_index_sha256": file_sha256(train_index_path),
        "validation_index_sha256": file_sha256(val_index_path),
        "asset_lock_sha256": file_sha256(asset_lock_path),
    }
    mismatches = [name for name, value in expected_fields.items() if payload.get(name) != value]
    if mismatches:
        raise ValueError(f"maximum-input manifest has mismatched fields: {mismatches}")
    sampling = payload.get("validation_sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("maximum-input manifest lacks validation_sampling")
    path_count = int(sampling.get("path_count", 0))
    step_count = int(sampling.get("step_count", 0))
    if (path_count, step_count) != (32, 40):
        raise ValueError("formal G3 validation sampling is locked to M=32 and N=40")
    records = payload.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError("maximum-input manifest has no bound tile contracts")
    combined_sources = {**train_sources, **val_sources}
    if len(combined_sources) != len(train_sources) + len(val_sources):
        raise ValueError("training and validation indices share a tile identity")
    assets = {asset.product_id: asset for asset in asset_lock.assets}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("maximum-input tile records must be objects")
        tile_id = str(record.get("tile_id", ""))
        if not tile_id or tile_id in seen or tile_id not in combined_sources:
            raise ValueError(f"maximum-input manifest contains an invalid tile: {tile_id}")
        seen.add(tile_id)
        sources = combined_sources[tile_id]
        plan_path = _existing_input(str(record.get("plan", "")), root=root)
        problem_path = _existing_input(str(record.get("problem", "")), root=root)
        plan_file_hash = file_sha256(plan_path)
        problem_file_hash = file_sha256(problem_path)
        if plan_file_hash != sources["plan_sha256"] or plan_file_hash != record.get(
            "plan_sha256"
        ):
            raise ValueError(f"maximum-input plan hash mismatch for tile {tile_id}")
        if problem_file_hash != sources["problem_sha256"] or problem_file_hash != record.get(
            "problem_sha256"
        ):
            raise ValueError(f"maximum-input problem hash mismatch for tile {tile_id}")
        plan = PlanContract.read(plan_path)
        problem = ReconstructionProblem.read(problem_path)
        if plan.tile_id != tile_id or problem.tile_id != tile_id:
            raise ValueError("maximum-input plan/problem tile identity mismatch")
        if plan.budget != 100 or len(plan.points) != 100:
            raise ValueError("G3 maximum-input plans must contain exactly K=100 footprints")
        if problem.plan_sha256 != plan.sha256:
            raise ValueError("maximum-input reconstruction problem binds a different plan")
        if problem.image.ndim != 2 or problem.image.shape != problem.z0.shape:
            raise ValueError("G3 maximum input requires one aligned image plane")
        if not np.any(np.isfinite(problem.image[problem.valid_mask])):
            raise ValueError("G3 maximum input contains no finite actual-image pixels")
        asset = assets.get(tile_id)
        if asset is None or not asset.qa.passed or not asset.imagery_lineage:
            raise ValueError("G3 maximum input is not backed by a QA-passed image asset")
        prepared_hash = problem.metadata.get("prepared_manifest_sha256")
        if not prepared_hash or prepared_hash != asset.prepared_manifest_sha256:
            raise ValueError("G3 problem differs from the asset's prepared-image manifest")
    if seen != set(combined_sources):
        raise ValueError("maximum-input tile bindings do not close over train and validation")
    return {
        "schema": MAXIMUM_CONFIG_SCHEMA,
        **expected_fields,
        "validation_sampling": {
            "path_count": path_count,
            "step_count": step_count,
        },
        "tile_count": len(seen),
        "manifest_file_sha256": file_sha256(path),
    }


def _normalization(path: Path) -> dict[str, float]:
    payload = _read_json(path)
    required = {"elevation_center", "elevation_scale", "image_center", "image_scale"}
    if set(payload) != required:
        raise ValueError("training normalization fields differ from the frozen contract")
    values = {name: float(value) for name, value in payload.items()}
    if (
        not all(np.isfinite(value) for value in values.values())
        or values["elevation_scale"] <= 0
        or values["image_scale"] <= 0
    ):
        raise ValueError("training normalization contains invalid values")
    return values


def _validate_training_sources(
    *,
    root: Path,
    path: Path,
    payload: Mapping[str, Any],
    train_index_path: Path,
    val_index_path: Path,
    asset_lock_path: Path,
    train_payload: Mapping[str, Any],
    val_payload: Mapping[str, Any],
    train_sources: Mapping[str, Mapping[str, str]],
    val_sources: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], str]:
    if payload.get("schema") != TRAINING_SOURCE_SCHEMA:
        raise ValueError(f"training source manifest must use {TRAINING_SOURCE_SCHEMA}")
    expected_bindings = {
        "training_index_sha256": file_sha256(train_index_path),
        "validation_index_sha256": file_sha256(val_index_path),
        "asset_lock_sha256": file_sha256(asset_lock_path),
    }
    mismatches = [name for name, value in expected_bindings.items() if payload.get(name) != value]
    if mismatches:
        raise ValueError(f"training source manifest has mismatched bindings: {mismatches}")
    combined_sources = {**train_sources, **val_sources}
    if len(combined_sources) != len(train_sources) + len(val_sources):
        raise ValueError("training and validation source manifests share a tile")
    tile_contracts: dict[str, tuple[str, str]] = {}
    for index_payload in (train_payload, val_payload):
        role = str(index_payload["role"])
        for tile in index_payload["tiles"]:
            tile_contracts[str(tile["tile_id"])] = (
                role,
                str(tile["geographic_group_id"]),
            )
    records = payload.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError("training source manifest has no tile records")
    proofs: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_names = {
        "problem": "problem_sha256",
        "plan": "plan_sha256",
        "gp_manifest": "gp_manifest_sha256",
        "gp_posterior": "gp_posterior_sha256",
        "target_label": "target_label_sha256",
        "normalization": "normalization_sha256",
    }
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("training source tile records must be objects")
        record = dict(raw_record)
        tile_id = str(record.get("tile_id", ""))
        if not tile_id or tile_id in seen or tile_id not in combined_sources:
            raise ValueError(f"training source manifest contains invalid tile {tile_id}")
        seen.add(tile_id)
        role, group_id = tile_contracts[tile_id]
        if record.get("role") != role or record.get("geographic_group_id") != group_id:
            raise ValueError("training source tile role/group differs from the training index")
        sources = combined_sources[tile_id]
        paths: dict[str, Path] = {}
        for path_name, hash_name in source_names.items():
            source_path = _existing_input(str(record.get(path_name, "")), root=root)
            observed_hash = file_sha256(source_path)
            if observed_hash != sources[hash_name] or observed_hash != record.get(hash_name):
                raise ValueError(f"training source {path_name} hash mismatch for {tile_id}")
            paths[path_name] = source_path
        problem = ReconstructionProblem.read(paths["problem"])
        plan = PlanContract.read(paths["plan"])
        if problem.tile_id != tile_id or plan.tile_id != tile_id or problem.plan_sha256 != plan.sha256:
            raise ValueError("training source problem/plan identity mismatch")
        gp_payload = _read_json(paths["gp_manifest"])
        verify_gp_posterior_manifest(
            gp_payload,
            manifest_file_sha256=file_sha256(paths["gp_manifest"]),
            problem_sha256=file_sha256(paths["problem"]),
            plan_sha256=file_sha256(paths["plan"]),
            posterior_sha256=file_sha256(paths["gp_posterior"]),
        )
        dem_key = str(record.get("target_dem_key", "dem"))
        mask_key = str(record.get("target_mask_key", "valid_mask"))
        with np.load(paths["target_label"], allow_pickle=False) as archive:
            required = {dem_key, mask_key, "tile_id", "transform", "crs_wkt"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f"training target {tile_id} lacks fields: {missing}")
            target = np.asarray(archive[dem_key], dtype=np.float64)
            target_mask = np.asarray(archive[mask_key], dtype=bool)
            target_tile = str(np.asarray(archive["tile_id"]).item())
            target_transform = tuple(float(item) for item in archive["transform"])
            target_crs = str(np.asarray(archive["crs_wkt"]).item())
        if (
            target_tile != tile_id
            or target_transform != plan.transform
            or target_crs != plan.crs_wkt
        ):
            raise ValueError("training target grid identity differs from its plan")
        with np.load(paths["gp_posterior"], allow_pickle=False) as archive:
            if set(archive.files) != {"zf", "omega", "variance"}:
                raise ValueError("GP posterior fields differ from the frozen contract")
            zf = np.asarray(archive["zf"], dtype=np.float64)
            omega = np.asarray(archive["omega"], dtype=np.float64)
        index_payload = train_payload if role == "bridge_train" else val_payload
        proof = verify_training_patch_derivation(
            problem=problem,
            plan=plan,
            target=target,
            target_valid_mask=target_mask,
            zf=zf,
            omega=omega,
            normalization=_normalization(paths["normalization"]),
            role=role,
            geographic_group_id=group_id,
            index_payload=index_payload,
            paper2_root=root,
        )
        proof["source_hashes"] = dict(sorted(sources.items()))
        proofs.append(proof)
    if seen != set(combined_sources):
        raise ValueError("training source manifest does not close over train and validation tiles")
    proofs.sort(key=lambda item: str(item["tile_id"]))
    derivation_sha256 = canonical_sha256(
        {
            "schema": "paper2.training-source-derivation/v1",
            **expected_bindings,
            "tiles": proofs,
        }
    )
    return (
        {
            "schema": TRAINING_SOURCE_SCHEMA,
            **expected_bindings,
            "tile_count": len(proofs),
            "manifest_file_sha256": file_sha256(path),
            "derivation_sha256": derivation_sha256,
        },
        derivation_sha256,
    )


def _receipt_content_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )


def build_training_derivation_receipt(
    *,
    paper2_root: str | Path,
    training_index: str | Path,
    validation_index: str | Path,
    environment_lock: str | Path,
    asset_lock: str | Path,
    training_source_manifest: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Re-derive prepared patches in a truth-authorized preparation process.

    The resulting receipt deliberately contains hashes but no source paths.  A
    distributed worker can therefore verify the prepared patch/index closure
    without receiving, resolving, or opening a reference/target DEM path.
    """

    from .cli import _IndexedBatches

    root = Path(paper2_root).resolve()
    train_path = _existing_input(training_index, root=root)
    val_path = _existing_input(validation_index, root=root)
    environment_path = _existing_input(environment_lock, root=root)
    _validate_live_environment_lock(root=root, path=environment_path)
    asset_path = _existing_input(asset_lock, root=root)
    source_path = _existing_input(training_source_manifest, root=root)
    output_path = guard_paper2_path(output, paper2_root=root)
    train_payload = _read_json(train_path)
    val_payload = _read_json(val_path)
    train_batches = _IndexedBatches(
        train_path, root=root, device=None, role="bridge_train"
    )
    val_batches = _IndexedBatches(val_path, root=root, device=None, role="bridge_val")
    train_sources = _source_hashes_by_tile(train_payload)
    val_sources = _source_hashes_by_tile(val_payload)
    _, derivation_sha256 = _validate_training_sources(
        root=root,
        path=source_path,
        payload=_read_json(source_path),
        train_index_path=train_path,
        val_index_path=val_path,
        asset_lock_path=asset_path,
        train_payload=train_payload,
        val_payload=val_payload,
        train_sources=train_sources,
        val_sources=val_sources,
    )
    combined_sources = {**train_sources, **val_sources}
    payload: dict[str, Any] = {
        "schema": DERIVATION_RECEIPT_SCHEMA,
        "training_index_sha256": file_sha256(train_path),
        "validation_index_sha256": file_sha256(val_path),
        "asset_lock_sha256": file_sha256(asset_path),
        "environment_lock_sha256": file_sha256(environment_path),
        "training_source_manifest_sha256": file_sha256(source_path),
        "verifier_code_sha256": paper2_training_code_sha256(root),
        "source_derivation_sha256": derivation_sha256,
        "training_sample_set_sha256": train_batches.sample_set_sha256,
        "validation_sample_set_sha256": val_batches.sample_set_sha256,
        "source_hashes_by_tile": {
            tile_id: dict(sorted(values.items()))
            for tile_id, values in sorted(combined_sources.items())
        },
        "tile_count": len(combined_sources),
        "truth_isolation": "reference_paths_consumed_only_by_preparation_verifier",
    }
    payload["receipt_sha256"] = _receipt_content_hash(payload)
    if output_path.exists():
        if _read_json(output_path) != payload:
            raise FileExistsError("refusing to replace a different derivation receipt")
    else:
        _atomic_json(output_path, payload)
    return payload


def _validate_derivation_receipt(
    *,
    root: Path,
    payload: Mapping[str, Any],
    train_index_path: Path,
    val_index_path: Path,
    environment_lock_path: Path,
    asset_lock_path: Path,
    train_batches: Any,
    val_batches: Any,
    train_sources: Mapping[str, Mapping[str, str]],
    val_sources: Mapping[str, Mapping[str, str]],
) -> str:
    if payload.get("schema") != DERIVATION_RECEIPT_SCHEMA:
        raise ValueError(f"derivation receipt must use {DERIVATION_RECEIPT_SCHEMA}")
    if payload.get("receipt_sha256") != _receipt_content_hash(payload):
        raise ValueError("training derivation receipt content hash mismatch")
    combined_sources = {**train_sources, **val_sources}
    if len(combined_sources) != len(train_sources) + len(val_sources):
        raise ValueError("training and validation derivation sources share a tile")
    expected = {
        "training_index_sha256": file_sha256(train_index_path),
        "validation_index_sha256": file_sha256(val_index_path),
        "asset_lock_sha256": file_sha256(asset_lock_path),
        "environment_lock_sha256": file_sha256(environment_lock_path),
        "verifier_code_sha256": paper2_training_code_sha256(root),
        "training_sample_set_sha256": train_batches.sample_set_sha256,
        "validation_sample_set_sha256": val_batches.sample_set_sha256,
        "source_hashes_by_tile": {
            tile_id: dict(sorted(values.items()))
            for tile_id, values in sorted(combined_sources.items())
        },
        "tile_count": len(combined_sources),
        "truth_isolation": "reference_paths_consumed_only_by_preparation_verifier",
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatches:
        raise ValueError(f"training derivation receipt differs from prepared inputs: {mismatches}")
    _require_source_hash = str(payload.get("training_source_manifest_sha256", ""))
    if not is_sha256(_require_source_hash):
        raise ValueError("derivation receipt lacks a valid source-manifest hash")
    derivation_sha256 = str(payload.get("source_derivation_sha256", ""))
    if not is_sha256(derivation_sha256):
        raise ValueError("derivation receipt lacks a valid source-derivation hash")
    return derivation_sha256


@dataclass(frozen=True)
class PreparedGateInputs:
    root: Path
    training_index: Path
    validation_index: Path
    environment_lock: Path
    g1_report: Path
    asset_lock: Path
    role_manifest: Path
    maximum_config_manifest: Path
    training_derivation_receipt: Path
    protocol_config: Path
    g0_ready_draft: Path
    processed_lock: Path
    evidence: GateEvidence

    def request_inputs(self) -> dict[str, str]:
        return dict(self.evidence.runtime_inputs)


def load_real_gate_inputs(
    *,
    paper2_root: str | Path,
    training_index: str | Path,
    validation_index: str | Path,
    environment_lock: str | Path,
    g1_report: str | Path,
    asset_lock: str | Path,
    role_manifest: str | Path,
    maximum_config_manifest: str | Path,
    training_derivation_receipt: str | Path,
    protocol_config: str | Path,
    g0_ready_draft: str | Path,
    processed_lock: str | Path,
    overfit_sample_count: int = 32,
) -> PreparedGateInputs:
    """Verify that G2/G3 inputs are real, prepared, role-safe Paper2 artifacts."""

    from .cli import _IndexedBatches

    root = Path(paper2_root).resolve()
    train_path = _existing_input(training_index, root=root)
    val_path = _existing_input(validation_index, root=root)
    environment_path = _existing_input(environment_lock, root=root)
    _validate_live_environment_lock(root=root, path=environment_path)
    g1_path = _existing_input(g1_report, root=root)
    asset_path = _existing_input(asset_lock, root=root)
    role_path = _existing_input(role_manifest, root=root)
    maximum_path = _existing_input(maximum_config_manifest, root=root)
    derivation_receipt_path = _existing_input(training_derivation_receipt, root=root)
    protocol_config_path = _existing_input(protocol_config, root=root)
    g0_ready_draft_path = _existing_input(g0_ready_draft, root=root)
    processed_lock_path = _existing_input(processed_lock, root=root)
    g1_payload = _read_json(g1_path)
    if g1_payload.get("schema") != "paper2.gate/G1-v1" or g1_payload.get("status") != "PASS":
        raise ValueError("G2/G3 requires a passed paper2.gate/G1-v1 report")
    if float(g1_payload.get("analytic_tolerance", float("nan"))) != 1.0e-5:
        raise ValueError("G1 report uses a different analytic tolerance")
    current_g1_sources = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted((root / "src" / "paper2dem").glob("*.py"))
    }
    if g1_payload.get("source_sha256") != current_g1_sources:
        raise ValueError("G1 report is stale relative to the current Paper2 source tree")
    train_batches = _IndexedBatches(
        train_path,
        root=root,
        device=None,
        role="bridge_train",
    )
    val_batches = _IndexedBatches(val_path, root=root, device=None, role="bridge_val")
    if train_batches.asset_lock_sha256 != val_batches.asset_lock_sha256:
        raise ValueError("training and validation indices bind different asset locks")
    if file_sha256(asset_path) != train_batches.asset_lock_sha256:
        raise ValueError("training indices do not bind the supplied formal asset lock")
    if len(train_batches.all_samples) < overfit_sample_count:
        raise ValueError("bridge_train contains fewer than the required 32 overfit patches")
    train_payload = _read_json(train_path)
    val_payload = _read_json(val_path)
    train_sources = _source_hashes_by_tile(train_payload)
    val_sources = _source_hashes_by_tile(val_payload)
    asset_contract = AssetLock.read(asset_path)
    geography = audit_asset_lock_geography(asset_contract)
    if not geography.ok:
        raise ValueError("formal asset lock fails geography/QA isolation")
    _validate_g0_parent_chain(
        asset_lock=asset_contract,
        ready_draft_path=g0_ready_draft_path,
        processed_lock_path=processed_lock_path,
    )
    role_payload = _read_json(role_path)
    if role_payload.get("asset_lock_sha256") != file_sha256(asset_path):
        raise ValueError("role manifest binds a different formal asset lock")
    _validate_role_authorization(
        asset_lock=asset_contract,
        role_payload=role_payload,
        train_batches=train_batches,
        val_batches=val_batches,
    )
    source_derivation_sha256 = _validate_derivation_receipt(
        root=root,
        payload=_read_json(derivation_receipt_path),
        train_index_path=train_path,
        val_index_path=val_path,
        environment_lock_path=environment_path,
        asset_lock_path=asset_path,
        train_batches=train_batches,
        val_batches=val_batches,
        train_sources=train_sources,
        val_sources=val_sources,
    )
    maximum_config = _validate_maximum_config(
        root=root,
        path=maximum_path,
        payload=_read_json(maximum_path),
        train_index_path=train_path,
        val_index_path=val_path,
        asset_lock_path=asset_path,
        train_sources=train_sources,
        val_sources=val_sources,
        asset_lock=asset_contract,
    )
    selected = [sample.contract for sample in train_batches.all_samples[:overfit_sample_count]]
    overfit_selection_sha256 = canonical_sha256(
        {
            "protocol": "paper2-g2-first-32-prepared-patches/v1",
            "parent_sample_set_sha256": train_batches.sample_set_sha256,
            "samples": selected,
        }
    )
    train_file_hash = file_sha256(train_path)
    val_file_hash = file_sha256(val_path)
    asset_file_hash = file_sha256(asset_path)
    environment_hash = file_sha256(environment_path)
    input_hashes = {
        "asset_lock_sha256": asset_file_hash,
        "role_manifest_sha256": file_sha256(role_path),
        "training_index_sha256": train_file_hash,
        "validation_index_sha256": val_file_hash,
        "environment_lock_sha256": environment_hash,
        "g1_report_sha256": file_sha256(g1_path),
        "maximum_config_manifest_sha256": file_sha256(maximum_path),
        "training_derivation_receipt_sha256": file_sha256(derivation_receipt_path),
        "protocol_config_sha256": file_sha256(protocol_config_path),
        "g0_ready_draft_sha256": file_sha256(g0_ready_draft_path),
        "processed_lock_sha256": file_sha256(processed_lock_path),
    }
    runtime_inputs = {
        "training_index": _relative(train_path, root),
        "validation_index": _relative(val_path, root),
        "environment_lock": _relative(environment_path, root),
        "g1_report": _relative(g1_path, root),
        "asset_lock": _relative(asset_path, root),
        "role_manifest": _relative(role_path, root),
        "maximum_config_manifest": _relative(maximum_path, root),
        "training_derivation_receipt": _relative(derivation_receipt_path, root),
        "protocol_config": _relative(protocol_config_path, root),
        "g0_ready_draft": _relative(g0_ready_draft_path, root),
        "processed_lock": _relative(processed_lock_path, root),
    }
    evidence = GateEvidence(
        input_hashes=input_hashes,
        code_sha256=paper2_training_code_sha256(root),
        environment_sha256=environment_hash,
        data_scope=REAL_DATA_SCOPE,
        synthetic=False,
        consumed_roles=("bridge_train", "bridge_val"),
        overfit_sample_count=overfit_sample_count,
        overfit_selection_sha256=overfit_selection_sha256,
        overfit_split_hashes={
            "asset_lock": asset_file_hash,
            "bridge_train_index": train_file_hash,
            "g2_train_selection": overfit_selection_sha256,
            "g2_validation_selection": overfit_selection_sha256,
        },
        full_split_hashes={
            "asset_lock": asset_file_hash,
            "bridge_train": train_batches.sample_set_sha256,
            "bridge_train_index": train_file_hash,
            "bridge_val": val_batches.sample_set_sha256,
            "bridge_val_index": val_file_hash,
        },
        maximum_config=maximum_config,
        verification_scope="prepared_derivation_receipt",
        source_derivation_sha256=source_derivation_sha256,
        runtime_inputs=runtime_inputs,
    )
    evidence.validate(require_formal=True)
    return PreparedGateInputs(
        root=root,
        training_index=train_path,
        validation_index=val_path,
        environment_lock=environment_path,
        g1_report=g1_path,
        asset_lock=asset_path,
        role_manifest=role_path,
        maximum_config_manifest=maximum_path,
        training_derivation_receipt=derivation_receipt_path,
        protocol_config=protocol_config_path,
        g0_ready_draft=g0_ready_draft_path,
        processed_lock=processed_lock_path,
        evidence=evidence,
    )


class SubprocessTorchGateRunner(GateRunner):
    """Launch each stage with an isolated local FileStore spawn and preserve its log."""

    def __init__(
        self,
        *,
        inputs: PreparedGateInputs,
        protocol: GateProtocol,
        seed: int,
        world_size: int,
        timeout_seconds: int = 7 * 24 * 60 * 60,
    ) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.inputs = inputs
        self.protocol = protocol
        self.seed = seed
        self.world_size = world_size
        if timeout_seconds <= 0:
            raise ValueError("stage timeout must be positive")
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        stage: GateStageSpec,
        *,
        attempt: int,
        resume_checkpoint: Path | None,
        resume_manifest: Path | None,
        output_directory: Path,
    ) -> GateRunReceipt:
        root = self.inputs.root
        attempt_directory = guard_paper2_path(
            output_directory / f"attempt-{attempt:03d}", paper2_root=root, allow_root=True
        )
        attempt_directory.mkdir(parents=True, exist_ok=True)
        request = {
            "schema": REQUEST_SCHEMA,
            "stage": asdict(stage),
            "stage_sha256": stage.sha256,
            "protocol": asdict(self.protocol),
            "protocol_sha256": self.protocol.sha256,
            "evidence_sha256": self.inputs.evidence.sha256,
            "seed": self.seed,
            "world_size": self.world_size,
            "inputs": self.inputs.request_inputs(),
            "resume_checkpoint": (
                None if resume_checkpoint is None else _relative(resume_checkpoint, root)
            ),
            "resume_manifest": None if resume_manifest is None else _relative(resume_manifest, root),
            "checkpoint_output": _relative(attempt_directory / "checkpoints", root),
        }
        request_path = attempt_directory / "request.json"
        if request_path.exists():
            if _read_json(request_path) != request:
                raise FileExistsError("refusing to replace a different stage request")
        else:
            _atomic_json(request_path, request)
        worker = root / "scripts" / "run_training_gate_stage_spawn.py"
        if not worker.is_file():
            raise FileNotFoundError(worker)
        command = [
            sys.executable,
            str(worker),
            "--paper2-root",
            str(root),
            "--request",
            str(request_path),
            "--world-size",
            str(self.world_size),
        ]
        environment = dict(os.environ)
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(root / "src") + (
            "" if not existing_pythonpath else os.pathsep + existing_pythonpath
        )
        environment.update(LOCAL_DISTRIBUTED_ENVIRONMENT)
        log_path = attempt_directory / "runner.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError(
                    f"training gate stage exceeded {self.timeout_seconds}s; log={log_path}"
                ) from None
        if returncode != 0:
            raise RuntimeError(
                f"training gate worker failed with exit {returncode}; log={log_path}"
            )
        manifest = attempt_directory / "checkpoints" / "checkpoint-manifest.json"
        if not manifest.is_file():
            raise RuntimeError("training gate worker exited without a checkpoint manifest")
        return GateRunReceipt(
            checkpoint_manifest_path=manifest,
            request_path=request_path,
            execution_scope=PRODUCTION_EXECUTION_SCOPE,
        )


def execute_formal_training_gates(
    *,
    paper2_root: str | Path,
    training_index: str | Path,
    validation_index: str | Path,
    environment_lock: str | Path,
    g1_report: str | Path,
    asset_lock: str | Path,
    role_manifest: str | Path,
    maximum_config_manifest: str | Path,
    training_derivation_receipt: str | Path,
    protocol_config: str | Path,
    g0_ready_draft: str | Path,
    processed_lock: str | Path,
    seed: int,
    world_size: int,
    output_directory: str | Path,
    timeout_seconds: int = 7 * 24 * 60 * 60,
) -> dict[str, Any]:
    """Run formal gates without accepting caller-injected evidence or a runner."""

    root = Path(paper2_root).resolve()
    protocol = gate_protocol_from_yaml(protocol_config, paper2_root=root)
    inputs = load_real_gate_inputs(
        paper2_root=root,
        training_index=training_index,
        validation_index=validation_index,
        environment_lock=environment_lock,
        g1_report=g1_report,
        asset_lock=asset_lock,
        role_manifest=role_manifest,
        maximum_config_manifest=maximum_config_manifest,
        training_derivation_receipt=training_derivation_receipt,
        protocol_config=protocol_config,
        g0_ready_draft=g0_ready_draft,
        processed_lock=processed_lock,
        overfit_sample_count=protocol.overfit_sample_count,
    )
    output = guard_paper2_path(output_directory, paper2_root=root, allow_root=True)
    runner = SubprocessTorchGateRunner(
        inputs=inputs,
        protocol=protocol,
        seed=seed,
        world_size=world_size,
        timeout_seconds=timeout_seconds,
    )
    return _execute_training_gates(
        paper2_root=root,
        output_directory=output,
        report_output=output / "training-gate-report.json",
        evidence=inputs.evidence,
        protocol=protocol,
        runner=runner,
        seed=seed,
        world_size=world_size,
        require_formal_evidence=True,
        allow_formal_output=True,
    )


def validate_training_environment_lock(
    *, paper2_root: str | Path, environment_lock: str | Path
) -> dict[str, Any]:
    """Public fail-closed validation of the live formal CUDA environment."""

    root = Path(paper2_root).resolve()
    path = _existing_input(environment_lock, root=root)
    return _validate_live_environment_lock(root=root, path=path)


def _gate_evidence_from_payload(raw: Mapping[str, Any]) -> GateEvidence:
    values = dict(raw)
    values["consumed_roles"] = tuple(str(value) for value in values["consumed_roles"])
    return GateEvidence(**values)


def validate_formal_training_gate_report(
    *,
    paper2_root: str | Path,
    report_path: str | Path,
    training_index: str | Path,
    validation_index: str | Path,
    environment_lock: str | Path,
    asset_lock: str | Path,
    role_manifest: str | Path,
    expected_world_size: int,
) -> dict[str, Any]:
    """Revalidate all six production G2/G3 stages before formal main training.

    The aggregate's embedded runtime paths are not trusted by themselves.  They
    are resolved below ``paper2/``, the complete real-data evidence is rebuilt,
    every request/stage/checkpoint/continuation file is replayed, and the GPU
    UUID set is compared with the live environment lock.
    """

    root = Path(paper2_root).resolve()
    report_file = _existing_input(report_path, root=root)
    explicit = {
        "training_index": _existing_input(training_index, root=root),
        "validation_index": _existing_input(validation_index, root=root),
        "environment_lock": _existing_input(environment_lock, root=root),
        "asset_lock": _existing_input(asset_lock, root=root),
        "role_manifest": _existing_input(role_manifest, root=root),
    }
    report = _read_json(report_file)
    semantic = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    if (
        report.get("schema") != "paper2.training-gate-report/v1"
        or report.get("gate") != "G2/G3"
        or report.get("status") != "complete"
        or report.get("formal_gate") is not True
        or report.get("synthetic") is not False
        or report.get("data_scope") != REAL_DATA_SCOPE
        or int(report.get("world_size", -1)) != 2
        or int(report.get("seed", -1)) != 20260721
        or report.get("report_sha256") != semantic
    ):
        raise ValueError("formal G2/G3 aggregate identity/state/semantic hash differs")
    if expected_world_size != 2:
        raise ValueError("formal main training must use the two-GPU G2/G3 world size")

    protocol = _protocol_from_json(dict(report.get("protocol", {})))
    protocol.validate()
    if report.get("protocol_sha256") != protocol.sha256:
        raise ValueError("formal G2/G3 protocol hash differs")
    raw_evidence = report.get("evidence")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError("formal G2/G3 report lacks evidence")
    evidence = _gate_evidence_from_payload(raw_evidence)
    evidence.validate(require_formal=True)
    if (
        report.get("evidence_sha256") != evidence.sha256
        or report.get("input_hashes") != evidence.to_dict()["input_hashes"]
        or report.get("code_sha256") != evidence.code_sha256
        or report.get("environment_sha256") != evidence.environment_sha256
    ):
        raise ValueError("formal G2/G3 aggregate evidence bindings differ")

    runtime_paths = {
        name: _existing_input(root / relative, root=root)
        for name, relative in evidence.runtime_inputs.items()
    }
    for name, path in explicit.items():
        if runtime_paths[name] != path:
            raise ValueError(f"formal training {name} differs from the G2/G3 runtime input")
    rebuilt = load_real_gate_inputs(
        paper2_root=root,
        training_index=runtime_paths["training_index"],
        validation_index=runtime_paths["validation_index"],
        environment_lock=runtime_paths["environment_lock"],
        g1_report=runtime_paths["g1_report"],
        asset_lock=runtime_paths["asset_lock"],
        role_manifest=runtime_paths["role_manifest"],
        maximum_config_manifest=runtime_paths["maximum_config_manifest"],
        training_derivation_receipt=runtime_paths["training_derivation_receipt"],
        protocol_config=runtime_paths["protocol_config"],
        g0_ready_draft=runtime_paths["g0_ready_draft"],
        processed_lock=runtime_paths["processed_lock"],
        overfit_sample_count=protocol.overfit_sample_count,
    )
    if rebuilt.evidence.to_dict() != evidence.to_dict():
        raise ValueError("formal G2/G3 evidence is stale relative to current input bytes")
    yaml_protocol = gate_protocol_from_yaml(
        runtime_paths["protocol_config"], paper2_root=root
    )
    if yaml_protocol != protocol:
        raise ValueError("formal G2/G3 report protocol differs from its current YAML input")

    stages = gate_stages(protocol)
    records = report.get("stages")
    if (
        not isinstance(records, list)
        or len(records) != len(stages)
        or [str(record.get("stage_id", "")) for record in records]
        != [stage.stage_id for stage in stages]
        or report.get("stage_order") != [stage.stage_id for stage in stages]
        or any(record.get("status") != "PASS" for record in records)
        or any(record.get("execution_scope") != PRODUCTION_EXECUTION_SCOPE for record in records)
    ):
        raise ValueError("formal G2/G3 six-stage order/status/execution scope differs")
    expected_identity = canonical_sha256(
        {
            "schema": "paper2.training-gate-report/v1",
            "protocol_sha256": protocol.sha256,
            "evidence_sha256": evidence.sha256,
            "seed": 20260721,
            "world_size": 2,
            "stage_sha256": [stage.sha256 for stage in stages],
        }
    )
    if (
        report.get("identity_sha256") != expected_identity
        or report.get("gates")
        != {
            "G2": {"status": "PASS", "stage_count": 4},
            "G3": {"status": "PASS", "stage_count": 2},
        }
        or report.get("resume_policy") != "exact_completed_prefix_only_no_stage_skip"
        or report.get("limitations") != []
    ):
        raise ValueError("formal G2/G3 aggregate contract differs")
    _validate_completed_prefix(
        {"completed": records},
        root=root,
        stages=stages,
        protocol=protocol,
        evidence=evidence,
        seed=20260721,
        world_size=2,
    )

    environment = validate_training_environment_lock(
        paper2_root=root, environment_lock=explicit["environment_lock"]
    )
    locked_uuids = {
        str(device["uuid"]) for device in environment["runtime"]["gpu_devices"]
    }
    if len(locked_uuids) != 2:
        raise ValueError("formal environment lock does not contain two unique GPU UUIDs")
    for record in records:
        manifest = _read_json(
            _existing_input(record["checkpoint_manifest_path"], root=root)
        )
        stage_uuids = {
            str(rank["device_uuid"])
            for rank in manifest.get("telemetry", {}).get("ranks", [])
        }
        if stage_uuids != locked_uuids:
            raise ValueError("formal G2/G3 stage GPU UUIDs differ from the live environment lock")
    return report


class _SelectedBatches:
    def __init__(self, source: Iterable[Mapping[str, Any]], indices: tuple[int, ...]) -> None:
        if not indices or tuple(sorted(set(indices))) != indices:
            raise ValueError("selected training indices must be sorted and unique")
        self.source = source
        self.indices = indices

    def __iter__(self):
        wanted = set(self.indices)
        final = self.indices[-1]
        for index, batch in enumerate(self.source):
            if index in wanted:
                yield batch
            if index >= final:
                break


def _protocol_from_json(value: Mapping[str, Any]) -> GateProtocol:
    converted = dict(value)
    converted["feasibility_updates"] = tuple(int(item) for item in converted["feasibility_updates"])
    return GateProtocol(**converted)


def _stage_from_json(value: Mapping[str, Any]) -> GateStageSpec:
    return GateStageSpec(**dict(value))


def execute_torch_gate_request(*, paper2_root: str | Path, request_path: str | Path) -> None:
    """Execute one validated gate request inside a FileStore-spawn worker."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is unavailable in this runtime")
    import torch

    from .cli import _IndexedBatches, _validation_metric
    from .objectives import bridge_training_loss, reverse_bridge_path

    root = Path(paper2_root).resolve()
    request_file = _existing_input(request_path, root=root)
    request = _read_json(request_file)
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("training gate request schema is invalid")
    protocol = _protocol_from_json(request["protocol"])
    protocol.validate()
    if request.get("protocol_sha256") != protocol.sha256:
        raise ValueError("training gate request protocol hash mismatch")
    stage = _stage_from_json(request["stage"])
    locked_stages = {item.stage_id: item for item in gate_stages(protocol)}
    if locked_stages.get(stage.stage_id) != stage or request.get("stage_sha256") != stage.sha256:
        raise ValueError("training gate request is not one locked stage")
    inputs = dict(request["inputs"])
    prepared = load_real_gate_inputs(
        paper2_root=root,
        training_index=inputs["training_index"],
        validation_index=inputs["validation_index"],
        environment_lock=inputs["environment_lock"],
        g1_report=inputs["g1_report"],
        asset_lock=inputs["asset_lock"],
        role_manifest=inputs["role_manifest"],
        maximum_config_manifest=inputs["maximum_config_manifest"],
        training_derivation_receipt=inputs["training_derivation_receipt"],
        protocol_config=inputs["protocol_config"],
        g0_ready_draft=inputs["g0_ready_draft"],
        processed_lock=inputs["processed_lock"],
        overfit_sample_count=protocol.overfit_sample_count,
    )
    if request.get("evidence_sha256") != prepared.evidence.sha256:
        raise ValueError("training gate inputs changed after request creation")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != int(request["world_size"]):
        raise ValueError("spawned world size differs from the gate request")
    if not torch.cuda.is_available():
        raise RuntimeError("formal G2/G3 execution requires CUDA telemetry")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        torch.distributed.init_process_group(
            "nccl",
            device_id=device,
            timeout=timedelta(minutes=30),
            **_filestore_process_group_kwargs(
                paper2_root=root, rank=rank, world_size=world_size
            ),
        )
    try:
        seed = int(request["seed"])
        config = training_config_for_stage(stage, protocol=protocol, seed=seed)
        config.validate(world_size=world_size)
        configure_determinism(seed)
        model_config = UNetConfig()
        model_config.validate(formal=True)
        model = build_model(model_config).to(device)
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        if stage.selection == "overfit32":
            base_train = _IndexedBatches(
                prepared.training_index,
                root=root,
                device=device,
                role="bridge_train",
            )
            all_indices = tuple(range(protocol.overfit_sample_count))
            local_indices = tuple(index for index in all_indices if index % world_size == rank)
            train_batches: Iterable[Mapping[str, Any]] = _SelectedBatches(
                base_train, local_indices
            )
            validation_batches: Any = _SelectedBatches(base_train, all_indices)
        else:
            train_batches = _IndexedBatches(
                prepared.training_index,
                root=root,
                device=device,
                role="bridge_train",
                rank=rank,
                world_size=world_size,
                shard=world_size > 1,
            )
            validation_batches = _IndexedBatches(
                prepared.validation_index,
                root=root,
                device=device,
                role="bridge_val",
            )
        replay_telemetry: dict[str, Any] | None = None
        if config.amp_forward_replay:
            replay_telemetry = {
                "protocol": "same-sample-amp-forward-replay/v1",
                "enabled": True,
                "accumulation_steps": config.accumulation_steps(world_size),
                "objective_calls": 0,
                "attempt_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "first_success_objective_call": None,
                "first_success_update": None,
            }

        def loss(model_value: Any, batch: Mapping[str, Any]):
            return bridge_training_loss(
                model_value,
                batch,
                method=stage.method,
                g_min=config.g_min,
                alpha=config.alpha,
                endpoint_weight=config.endpoint_weight,
                gradient_weight=config.gradient_weight,
                amp_forward_replay=config.amp_forward_replay,
                replay_telemetry=replay_telemetry,
            )

        sampling_calls = 0

        def validation(model_value: Any, _: int) -> float:
            nonlocal sampling_calls
            metric = _validation_metric(
                model_value,
                validation_batches,
                stage.method,
                g_min=config.g_min,
                alpha=config.alpha,
                ddpm_cosine_offset=config.ddpm_cosine_offset,
            )
            if stage.selection == "full_k100":
                batch = next(iter(validation_batches))
                generator = torch.Generator(device=device)
                sampling = prepared.evidence.maximum_config["validation_sampling"]
                model_value.eval()
                for _ in range(int(sampling["path_count"])):
                    generator.manual_seed(seed + sampling_calls)
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.float16),
                    ):
                        prediction = reverse_bridge_path(
                            model_value,
                            zf=batch["zf"],
                            condition=batch["condition"],
                            omega=batch["omega"],
                            method=stage.method,
                            g_min=config.g_min,
                            alpha=config.alpha,
                            steps=int(sampling["step_count"]),
                            generator=generator,
                        )
                    if not bool(torch.all(torch.isfinite(prediction))):
                        raise FloatingPointError("non-finite validation sampling path")
                    sampling_calls += 1
                model_value.train()
            return metric

        resume_checkpoint = request.get("resume_checkpoint")
        resume_manifest = request.get("resume_manifest")
        checkpoint_path = (
            None if resume_checkpoint is None else _existing_input(resume_checkpoint, root=root)
        )
        manifest_path = (
            None if resume_manifest is None else _existing_input(resume_manifest, root=root)
        )
        previous_manifest = None if manifest_path is None else _read_json(manifest_path)
        output = guard_paper2_path(
            str(request["checkpoint_output"]), paper2_root=root, allow_root=True
        )
        manifest = train_model(
            model=model,
            train_batches=train_batches,
            validation=validation,
            loss_function=loss,
            output=output,
            config=config,
            model_config=model_config,
            split_hashes=prepared.evidence.split_hashes(stage),
            code_hash=prepared.evidence.code_sha256,
            environment_hash=prepared.evidence.environment_sha256,
            formal=False,
            stop_update=stage.stop_update,
            resume_checkpoint=checkpoint_path,
            resume_manifest=previous_manifest,
            evaluate_at_start=stage.selection == "overfit32" and stage.start_update == 0,
            strict_finite_gradients=stage.require_stability,
            objective_telemetry=replay_telemetry,
        )
        if rank == 0:
            manifest["gate_validation_sampling_calls"] = sampling_calls
            manifest["gate_validation_sampling_nfe"] = sampling_calls * int(
                prepared.evidence.maximum_config["validation_sampling"]["step_count"]
            )
            manifest["gate_maximum_config_sha256"] = canonical_sha256(
                prepared.evidence.maximum_config
            )
            manifest["gate_data_scope"] = REAL_DATA_SCOPE
            manifest["gate_evidence_sha256"] = prepared.evidence.sha256
            manifest["gate_stage_sha256"] = stage.sha256
            manifest["gate_request_file_sha256"] = file_sha256(request_file)
            manifest["synthetic"] = False
            manifest["formal_gate"] = False
            manifest["manifest_sha256"] = canonical_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            _atomic_json(output / "checkpoint-manifest.json", manifest)
    finally:
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
