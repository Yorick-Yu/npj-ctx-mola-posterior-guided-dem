"""Pre-outcome real-data selection of the two-GPU formal training strategy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import math

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)
from .models import canonical_model_identity
from .training import FORMAL_SEEDS, TrainingConfig, distributed_stochastic_seed


REAL_STRATEGY_SCHEMA = "paper2.real-data-training-strategy/v1"
REAL_STRATEGY_BENCHMARK_UPDATES = 2_000
REAL_STRATEGY_VALIDATION_INTERVAL = 2_000
REAL_STRATEGY_METHOD = "OT-SB-full"
REAL_STRATEGY_BINDINGS = {
    "training_index",
    "validation_index",
    "environment_lock",
    "asset_lock",
    "role_manifest",
    "g2_g3_report",
    "training_selection",
}


def _semantic(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != field}


def _src_tree_sha256(root: Path) -> str:
    records = [
        (path.relative_to(root / "src").as_posix(), file_sha256(path))
        for path in sorted((root / "src").rglob("*.py"))
    ]
    if not records:
        raise ValueError("real-data strategy benchmark found no Paper2 source files")
    return canonical_sha256(records)


def _artifact_record(path: Path, *, root: Path) -> dict[str, str]:
    guarded = guard_paper2_path(path, paper2_root=root)
    if not guarded.is_file():
        raise FileNotFoundError(guarded)
    return {
        "relative_path": guarded.relative_to(root).as_posix(),
        "sha256": file_sha256(guarded),
    }


def _validate_selection(path: Path) -> Mapping[str, Any]:
    payload = read_json(path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "paper2.training-hyperparameter-selection/v1"
        or payload.get("status") != "complete"
        or payload.get("selection_role") != "bridge_val"
        or payload.get("test_results_inspected") is not False
        or payload.get("selection_sha256")
        != canonical_sha256(_semantic(payload, "selection_sha256"))
    ):
        raise ValueError("real-data strategy requires the blind completed training selection")
    selected = payload.get("selected")
    required = {
        "g_min",
        "alpha",
        "endpoint_weight",
        "gradient_weight",
        "ddpm_cosine_offset",
    }
    if not isinstance(selected, Mapping) or set(selected) != required:
        raise ValueError("real-data strategy selection hyperparameters differ")
    return payload


def _validate_benchmark_manifest(
    path: Path,
    *,
    root: Path,
    expected_world_size: int,
    expected_seed: int,
    selected: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(manifest_path)
    if not isinstance(payload, Mapping) or payload.get("manifest_sha256") != canonical_sha256(
        _semantic(payload, "manifest_sha256")
    ):
        raise ValueError("training-strategy checkpoint manifest hash differs")
    if (
        payload.get("status") != "complete"
        or payload.get("method") != REAL_STRATEGY_METHOD
        or payload.get("model_identity") != canonical_model_identity(REAL_STRATEGY_METHOD)
        or int(payload.get("seed", -1)) != expected_seed
        or int(payload.get("completed_updates", -1))
        != REAL_STRATEGY_BENCHMARK_UPDATES
        or int(payload.get("target_updates", -1)) != REAL_STRATEGY_BENCHMARK_UPDATES
    ):
        raise ValueError("training-strategy checkpoint identity/update contract differs")
    raw_config = payload.get("training_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("training-strategy checkpoint lacks its training config")
    config = TrainingConfig(**dict(raw_config))
    config.validate(world_size=expected_world_size)
    expected_config = {
        "updates": REAL_STRATEGY_BENCHMARK_UPDATES,
        "validation_interval": REAL_STRATEGY_VALIDATION_INTERVAL,
        "micro_batch_per_gpu": 1,
        "effective_batch": 8,
        "network_precision": "fp16",
        "data_loader_workers": 0,
        "early_stopping": False,
    }
    mismatches = [
        name for name, value in expected_config.items() if getattr(config, name) != value
    ]
    for name in (
        "g_min",
        "alpha",
        "endpoint_weight",
        "gradient_weight",
        "ddpm_cosine_offset",
    ):
        if float(getattr(config, name)) != float(selected[name]):
            mismatches.append(name)
    if mismatches:
        raise ValueError(f"training-strategy config differs: {sorted(set(mismatches))}")
    distributed = payload.get("distributed")
    expected_distributed = {
        "world_size": expected_world_size,
        "stochastic_seed_protocol": "paper2-ddp-rank-stochastic-seed/v1",
        "rank_stochastic_seeds": [
            distributed_stochastic_seed(expected_seed, rank)
            for rank in range(expected_world_size)
        ],
    }
    if distributed != expected_distributed:
        raise ValueError("training-strategy distributed seed contract differs")
    history = payload.get("validation_history")
    if (
        not isinstance(history, list)
        or len(history) != 1
        or int(history[0].get("update", -1)) != REAL_STRATEGY_BENCHMARK_UPDATES
        or not math.isfinite(float(history[0].get("metric", float("nan"))))
    ):
        raise ValueError("training-strategy validation schedule differs")
    stage = payload.get("stage")
    if (
        not isinstance(stage, Mapping)
        or int(stage.get("start_update", -1)) != 0
        or int(stage.get("stop_update", -1)) != REAL_STRATEGY_BENCHMARK_UPDATES
        or int(stage.get("continuous_updates_completed", -1))
        != REAL_STRATEGY_BENCHMARK_UPDATES
    ):
        raise ValueError("training-strategy run is not one continuous 2000-update stage")
    telemetry = payload.get("telemetry")
    ranks = telemetry.get("ranks") if isinstance(telemetry, Mapping) else None
    if (
        not isinstance(ranks, list)
        or len(ranks) != expected_world_size
        or telemetry.get("oom_detected") is not False
        or telemetry.get("nonfinite_detected") is not False
    ):
        raise ValueError("training-strategy telemetry is incomplete")
    observed_ranks = sorted(int(item.get("rank", -1)) for item in ranks)
    device_uuids = [str(item.get("device_uuid", "")) for item in ranks]
    if observed_ranks != list(range(expected_world_size)) or len(set(device_uuids)) != len(
        device_uuids
    ):
        raise ValueError("training-strategy rank/GPU identities differ")
    for item in ranks:
        allocated = int(item.get("peak_allocated_bytes", -1))
        reserved = int(item.get("peak_reserved_bytes", -1))
        total = int(item.get("total_memory_bytes", 0))
        if (
            total <= 0
            or allocated < 0
            or reserved < 0
            or allocated / total >= 0.80
            or reserved / total >= 0.90
            or item.get("oom_detected") is not False
            or item.get("nonfinite_detected") is not False
        ):
            raise ValueError("training-strategy memory/finite gate failed")
    wall = float(payload.get("wall_seconds", float("nan")))
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError("training-strategy wall time is invalid")
    for name in ("best.pt", "last.pt"):
        candidate = manifest_path.parent / name
        expected_hash = (
            payload.get("checkpoint_sha256")
            if name == "best.pt"
            else payload.get("last_checkpoint_sha256")
        )
        if not candidate.is_file() or file_sha256(candidate) != expected_hash:
            raise ValueError(f"training-strategy {name} bytes differ")
    return payload


def _environment_gpu_uuids(path: Path) -> tuple[str, str]:
    payload = read_json(path)
    runtime = payload.get("runtime") if isinstance(payload, Mapping) else None
    devices = runtime.get("gpu_devices") if isinstance(runtime, Mapping) else None
    if not isinstance(devices, list) or len(devices) != 2:
        raise ValueError("training-strategy environment must bind exactly two GPUs")
    uuids = tuple(str(item.get("uuid", "")) for item in devices)
    if any(not value for value in uuids) or len(set(uuids)) != 2:
        raise ValueError("training-strategy environment GPU UUIDs are invalid")
    return uuids


def build_real_training_strategy_receipt(
    *,
    paper2_root: str | Path,
    binding_paths: Mapping[str, Path],
    ddp_manifest_path: Path,
    independent_manifest_paths: Sequence[Path],
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    if set(binding_paths) != REAL_STRATEGY_BINDINGS:
        raise ValueError("training-strategy binding set differs")
    bindings = {
        name: _artifact_record(Path(path), root=root)
        for name, path in sorted(binding_paths.items())
    }
    selection_path = root / bindings["training_selection"]["relative_path"]
    selection = _validate_selection(selection_path)
    selected = dict(selection["selected"])
    g2_g3 = read_json(root / bindings["g2_g3_report"]["relative_path"])
    if (
        g2_g3.get("schema") != "paper2.training-gate-report/v1"
        or g2_g3.get("status") != "complete"
        or g2_g3.get("formal_gate") is not True
        or g2_g3.get("world_size") != 2
    ):
        raise ValueError("training-strategy receipt requires the passed two-GPU G2/G3 gate")
    if len(independent_manifest_paths) != 2:
        raise ValueError("independent strategy requires exactly two one-GPU manifests")
    ddp = _validate_benchmark_manifest(
        Path(ddp_manifest_path),
        root=root,
        expected_world_size=2,
        expected_seed=FORMAL_SEEDS[0],
        selected=selected,
    )
    independent = [
        _validate_benchmark_manifest(
            Path(path),
            root=root,
            expected_world_size=1,
            expected_seed=seed,
            selected=selected,
        )
        for path, seed in zip(independent_manifest_paths, FORMAL_SEEDS[:2], strict=True)
    ]
    manifests = [ddp, *independent]
    source_hash = _src_tree_sha256(root)
    if {str(item.get("code_sha256")) for item in manifests} != {source_hash}:
        raise ValueError("training-strategy checkpoint source hashes are stale")
    environment_hash = bindings["environment_lock"]["sha256"]
    if {str(item.get("environment_sha256")) for item in manifests} != {
        environment_hash
    }:
        raise ValueError("training-strategy checkpoints bind another environment")
    split_hashes = {canonical_sha256(dict(item["split_hashes"])) for item in manifests}
    if len(split_hashes) != 1:
        raise ValueError("training-strategy modes use different train/validation splits")
    if (
        ddp["split_hashes"]["bridge_train_index"]
        != bindings["training_index"]["sha256"]
        or ddp["split_hashes"]["bridge_val_index"]
        != bindings["validation_index"]["sha256"]
    ):
        raise ValueError("training-strategy manifests bind different data indices")
    expected_gpu_uuids = set(
        _environment_gpu_uuids(root / bindings["environment_lock"]["relative_path"])
    )
    ddp_gpu_uuids = {
        str(item["device_uuid"]) for item in ddp["telemetry"]["ranks"]
    }
    independent_gpu_uuids = {
        str(item["device_uuid"])
        for manifest in independent
        for item in manifest["telemetry"]["ranks"]
    }
    if ddp_gpu_uuids != expected_gpu_uuids or independent_gpu_uuids != expected_gpu_uuids:
        raise ValueError("training-strategy modes did not each use both locked GPUs")

    ddp_wall = float(ddp["wall_seconds"])
    independent_wall = max(float(item["wall_seconds"]) for item in independent)
    records = {
        "ddp": {
            "per_model_world_size": 2,
            "concurrent_model_jobs": 1,
            "completed_formal_model_updates": REAL_STRATEGY_BENCHMARK_UPDATES,
            "wall_seconds": ddp_wall,
            "formal_model_updates_per_hour": (
                REAL_STRATEGY_BENCHMARK_UPDATES / ddp_wall * 3600.0
            ),
            "checkpoint_manifests": [
                _artifact_record(Path(ddp_manifest_path), root=root)
            ],
            "gpu_uuids": sorted(ddp_gpu_uuids),
            "passed": True,
        },
        "independent": {
            "per_model_world_size": 1,
            "concurrent_model_jobs": 2,
            "completed_formal_model_updates": 2 * REAL_STRATEGY_BENCHMARK_UPDATES,
            "wall_seconds": independent_wall,
            "formal_model_updates_per_hour": (
                2 * REAL_STRATEGY_BENCHMARK_UPDATES / independent_wall * 3600.0
            ),
            "checkpoint_manifests": [
                _artifact_record(Path(path), root=root)
                for path in independent_manifest_paths
            ],
            "gpu_uuids": sorted(independent_gpu_uuids),
            "passed": True,
        },
    }
    # Ties retain synchronized DDP because it has the smaller scheduler surface.
    chosen_mode = max(
        ("ddp", "independent"),
        key=lambda name: (
            float(records[name]["formal_model_updates_per_hour"]),
            name == "ddp",
        ),
    )
    chosen = records[chosen_mode]
    payload: dict[str, Any] = {
        "schema": REAL_STRATEGY_SCHEMA,
        "status": "complete",
        "formal_gate": True,
        "selection_role": "bridge_train+bridge_val",
        "test_results_inspected": False,
        "benchmark_method": REAL_STRATEGY_METHOD,
        "updates_per_model": REAL_STRATEGY_BENCHMARK_UPDATES,
        "validation_interval": REAL_STRATEGY_VALIDATION_INTERVAL,
        "effective_batch_per_model": 8,
        "criterion": "maximum_completed_formal_model_updates_per_wall_hour",
        "tie_break": "ddp",
        "records": records,
        "chosen_mode": chosen_mode,
        "world_size": int(chosen["per_model_world_size"]),
        "concurrent_model_jobs": int(chosen["concurrent_model_jobs"]),
        "bindings": bindings,
        "source_code_sha256": source_hash,
        "split_contract_sha256": next(iter(split_hashes)),
        "gpu_uuids": sorted(expected_gpu_uuids),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_real_training_strategy_receipt(
    path: str | Path,
    *,
    paper2_root: str | Path,
    binding_paths: Mapping[str, Path],
    ddp_manifest_path: Path,
    independent_manifest_paths: Sequence[Path],
) -> Path:
    payload = build_real_training_strategy_receipt(
        paper2_root=paper2_root,
        binding_paths=binding_paths,
        ddp_manifest_path=ddp_manifest_path,
        independent_manifest_paths=independent_manifest_paths,
    )
    return write_canonical_json(path, payload, paper2_root=paper2_root)


def read_real_training_strategy_receipt(
    path: str | Path, *, paper2_root: str | Path
) -> Mapping[str, Any]:
    root = Path(paper2_root).resolve()
    receipt_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(receipt_path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != REAL_STRATEGY_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("formal_gate") is not True
        or payload.get("test_results_inspected") is not False
        or payload.get("receipt_sha256")
        != canonical_sha256(_semantic(payload, "receipt_sha256"))
    ):
        raise ValueError("real-data training-strategy receipt contract differs")
    raw_bindings = payload.get("bindings")
    if not isinstance(raw_bindings, Mapping) or set(raw_bindings) != REAL_STRATEGY_BINDINGS:
        raise ValueError("real-data training-strategy bindings differ")
    binding_paths: dict[str, Path] = {}
    for name, record in raw_bindings.items():
        if not isinstance(record, Mapping):
            raise ValueError("real-data training-strategy binding is malformed")
        bound = guard_paper2_path(
            root / str(record.get("relative_path", "")), paper2_root=root
        )
        if not bound.is_file() or file_sha256(bound) != record.get("sha256"):
            raise ValueError(f"real-data training-strategy binding bytes differ: {name}")
        binding_paths[str(name)] = bound
    records = payload.get("records")
    if not isinstance(records, Mapping) or set(records) != {"ddp", "independent"}:
        raise ValueError("real-data training-strategy mode records differ")
    ddp_records = records["ddp"].get("checkpoint_manifests")
    independent_records = records["independent"].get("checkpoint_manifests")
    if not isinstance(ddp_records, list) or len(ddp_records) != 1:
        raise ValueError("real-data DDP strategy manifest set differs")
    if not isinstance(independent_records, list) or len(independent_records) != 2:
        raise ValueError("real-data independent strategy manifest set differs")

    def resolve_manifest(record: Mapping[str, Any]) -> Path:
        candidate = guard_paper2_path(
            root / str(record.get("relative_path", "")), paper2_root=root
        )
        if not candidate.is_file() or file_sha256(candidate) != record.get("sha256"):
            raise ValueError("real-data strategy benchmark manifest bytes differ")
        return candidate

    rebuilt = build_real_training_strategy_receipt(
        paper2_root=root,
        binding_paths=binding_paths,
        ddp_manifest_path=resolve_manifest(ddp_records[0]),
        independent_manifest_paths=[resolve_manifest(item) for item in independent_records],
    )
    if dict(payload) != rebuilt:
        raise ValueError("real-data training-strategy receipt does not reproduce")
    if payload.get("chosen_mode") not in {"ddp", "independent"}:
        raise ValueError("real-data training-strategy choice is invalid")
    if int(payload.get("world_size", 0)) not in {1, 2}:
        raise ValueError("real-data training-strategy world size is invalid")
    if not is_sha256(str(payload.get("source_code_sha256", ""))):
        raise ValueError("real-data training-strategy source hash is invalid")
    return payload
