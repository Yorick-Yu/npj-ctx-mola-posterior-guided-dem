from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import math

from .assets import audit_asset_lock_geography
from .calibration import read_calibration_manifest
from .contracts import AssetLock
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
)
from .models import UNetConfig, canonical_model_identity, formal_model_identities, method_spec
from .training import (
    FORMAL_SEEDS,
    TrainingConfig,
    distributed_stochastic_seed,
)


FORMAL_SPLIT_HASH_FIELDS = {
    "bridge_train",
    "bridge_train_index",
    "bridge_val",
    "bridge_val_index",
}


def validate_role_bindings(
    role_tiles: Mapping[str, Sequence[str]],
    asset_lock: AssetLock,
    *,
    test_group_count: int,
    role_groups: Mapping[str, Sequence[str]] | None = None,
) -> None:
    audit = audit_asset_lock_geography(asset_lock)
    if not audit.ok:
        raise ValueError(f"asset-lock geography/QA audit failed: {audit.errors[:3]}")
    required = {
        "bridge_train": 48,
        "bridge_val": 12,
        "uncertainty_calibration": 12,
        "paper2_locked_test": test_group_count,
        "sealed_reserve": 36 - test_group_count if test_group_count > 24 else 12,
        "paper1_bridge_test": 12,
        "locked_external": 6,
    }
    role_groups = role_tiles if role_groups is None else role_groups
    all_ids: dict[str, str] = {}
    all_groups: dict[str, str] = {}
    for role, count in required.items():
        values = [str(value) for value in role_tiles.get(role, ())]
        groups = [str(value) for value in role_groups.get(role, ())]
        if len(values) != count or len(values) != len(set(values)):
            raise ValueError(f"role {role} must contain exactly {count} unique tile IDs")
        if len(groups) != count or len(groups) != len(set(groups)):
            raise ValueError(f"role {role} must contain exactly {count} unique geographic groups")
        for value in values:
            previous = all_ids.setdefault(value, role)
            if previous != role:
                raise ValueError(f"tile {value} is reused by {previous} and {role}")
        for value in groups:
            previous = all_groups.setdefault(value, role)
            if previous != role:
                raise ValueError(f"geographic group {value} is reused by {previous} and {role}")
    locked_roles: dict[str, set[str]] = {}
    for asset in asset_lock.assets:
        locked_roles.setdefault(asset.role, set()).add(asset.geographic_group_id)
    for role in (
        "bridge_train",
        "bridge_val",
        "uncertainty_calibration",
        "paper2_locked_test",
        "sealed_reserve",
    ):
        expected = set(str(value) for value in role_groups[role])
        # When reserve groups are promoted, the immutable G0 asset-lock retains
        # their original role; G5 accepts only the declared 6-group prefix.
        if role == "paper2_locked_test" and test_group_count > 24:
            promoted = expected.difference(locked_roles.get(role, set()))
            if not promoted.issubset(locked_roles.get("sealed_reserve", set())):
                raise ValueError("expanded test groups are not drawn from the sealed reserve")
            expected = expected.difference(promoted)
        if role == "sealed_reserve" and test_group_count > 24:
            # The remaining reserve must match the G0 reserve after removing the
            # promoted prefix; the exact prefix is locked by the power report.
            if not expected.issubset(locked_roles.get("sealed_reserve", set())):
                raise ValueError("remaining reserve differs from the G0 asset-lock")
        elif expected != locked_roles.get(role, set()):
            raise ValueError(f"roles.json differs from asset-lock for {role}")


def validate_formal_checkpoint_manifest(
    path: Path,
) -> tuple[tuple[str, int], Mapping[str, Any]]:
    payload = read_json(path)
    semantic = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    if payload.get("manifest_sha256") != semantic:
        raise ValueError(f"checkpoint manifest hash mismatch: {path}")
    if payload.get("status") != "complete":
        raise ValueError(f"checkpoint is not complete: {path}")
    training_raw = payload.get("training_config")
    model_raw = payload.get("model_config")
    if not isinstance(training_raw, Mapping) or not isinstance(model_raw, Mapping):
        raise ValueError(f"checkpoint lacks formal training/model config: {path}")
    try:
        config = TrainingConfig(**dict(training_raw))
        model_values = dict(model_raw)
        if "channel_multipliers" in model_values:
            model_values["channel_multipliers"] = tuple(model_values["channel_multipliers"])
        model_config = UNetConfig(**model_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint config cannot be reconstructed: {path}") from exc
    distributed = payload.get("distributed")
    if not isinstance(distributed, Mapping):
        raise ValueError(f"checkpoint lacks a distributed contract: {path}")
    world_size = int(distributed.get("world_size", 0))
    if world_size not in {1, 2}:
        raise ValueError("formal checkpoint world_size must be one or two GPUs")
    config.validate(formal=True, world_size=world_size)
    model_config.validate(formal=True)
    if payload.get("training_config_sha256") != canonical_sha256(dict(training_raw)):
        raise ValueError(f"checkpoint training-config hash mismatch: {path}")
    if payload.get("model_config_sha256") != canonical_sha256(dict(model_raw)):
        raise ValueError(f"checkpoint model-config hash mismatch: {path}")
    method = str(payload.get("method", ""))
    identity = str(payload.get("model_identity", ""))
    seed = int(payload.get("seed", -1))
    if method != config.method or seed != config.seed:
        raise ValueError(f"checkpoint config identity/seed mismatch: {path}")
    if identity != canonical_model_identity(method) or identity not in formal_model_identities():
        raise ValueError(f"checkpoint method is not a canonical formal identity: {path}")
    expected_distributed = {
        "world_size": world_size,
        "stochastic_seed_protocol": "paper2-ddp-rank-stochastic-seed/v1",
        "rank_stochastic_seeds": [
            distributed_stochastic_seed(seed, rank) for rank in range(world_size)
        ],
    }
    if dict(distributed) != expected_distributed:
        raise ValueError(f"checkpoint distributed seed contract differs: {path}")
    if int(payload.get("completed_updates", -1)) != 120_000:
        raise ValueError(f"checkpoint did not complete 120000 updates: {(identity, seed)}")
    if int(payload.get("target_updates", -1)) != 120_000:
        raise ValueError(f"checkpoint target update count differs: {(identity, seed)}")
    history = payload.get("validation_history")
    if not isinstance(history, list) or len(history) != 60:
        raise ValueError(f"checkpoint must record all 60 validation evaluations: {path}")
    expected_updates = list(range(2_000, 120_001, 2_000))
    observed_updates: list[int] = []
    metrics: list[float] = []
    for item in history:
        if not isinstance(item, Mapping):
            raise ValueError(f"checkpoint validation history is malformed: {path}")
        observed_updates.append(int(item.get("update", -1)))
        metrics.append(float(item.get("metric", float("nan"))))
    if observed_updates != expected_updates or not all(math.isfinite(value) for value in metrics):
        raise ValueError(f"checkpoint validation schedule/metrics differ: {path}")
    minimum = min(metrics)
    selected_index = metrics.index(minimum)
    if (
        payload.get("selection_rule") != "minimum_bridge_val_metric_earliest_tie"
        or int(payload.get("selected_update", -1)) != expected_updates[selected_index]
        or int(payload.get("update", -1)) != expected_updates[selected_index]
        or float(payload.get("validation_metric", float("nan"))) != minimum
    ):
        raise ValueError(f"checkpoint is not the preregistered best validation update: {path}")
    split_hashes = payload.get("split_hashes")
    if not isinstance(split_hashes, Mapping) or set(split_hashes) != FORMAL_SPLIT_HASH_FIELDS:
        raise ValueError(f"checkpoint split hashes are incomplete: {path}")
    if any(not is_sha256(str(value)) for value in split_hashes.values()):
        raise ValueError(f"checkpoint split hashes are invalid: {path}")
    if not is_sha256(str(payload.get("code_sha256", ""))) or not is_sha256(
        str(payload.get("environment_sha256", ""))
    ):
        raise ValueError(f"checkpoint code/environment hashes are invalid: {path}")
    checkpoint_path = path.parent / str(payload.get("checkpoint_file", ""))
    last_path = path.parent / "last.pt"
    if checkpoint_path.name != "best.pt":
        raise ValueError(f"formal selected checkpoint must be named best.pt: {path}")
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != payload.get(
        "checkpoint_sha256"
    ):
        raise ValueError(f"selected checkpoint file/hash mismatch: {(identity, seed)}")
    if not last_path.is_file() or file_sha256(last_path) != payload.get(
        "last_checkpoint_sha256"
    ):
        raise ValueError(f"last checkpoint file/hash mismatch: {(identity, seed)}")
    return (identity, seed), payload


def validate_checkpoint_set(
    paths: Sequence[Path], *, expected_world_size: int | None = None
) -> dict[tuple[str, int], Mapping[str, Any]]:
    expected = {
        (identity, seed) for identity in formal_model_identities() for seed in FORMAL_SEEDS
    }
    observed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for path in paths:
        key, payload = validate_formal_checkpoint_manifest(path)
        if key in observed:
            raise ValueError(f"duplicate checkpoint identity/seed: {key}")
        observed[key] = payload
    if set(observed) != expected:
        missing = sorted(expected.difference(observed))
        unexpected = sorted(set(observed).difference(expected))
        raise ValueError(f"checkpoint set is not exact; missing={missing}, unexpected={unexpected}")
    split_contracts = {
        canonical_sha256(dict(payload["split_hashes"])) for payload in observed.values()
    }
    code_hashes = {str(payload["code_sha256"]) for payload in observed.values()}
    environment_hashes = {str(payload["environment_sha256"]) for payload in observed.values()}
    if len(split_contracts) != 1 or len(code_hashes) != 1 or len(environment_hashes) != 1:
        raise ValueError("formal checkpoints do not share one split/code/environment contract")
    world_sizes = {
        int(payload["distributed"]["world_size"]) for payload in observed.values()
    }
    if len(world_sizes) != 1:
        raise ValueError("formal checkpoints do not share one distributed world size")
    if expected_world_size is not None and world_sizes != {int(expected_world_size)}:
        raise ValueError("formal checkpoint world size differs from the frozen strategy")
    return observed


def validate_calibration_set(
    paths: Sequence[Path],
    checkpoints: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    asset_lock_sha256: str,
    role_manifest_sha256: str | None = None,
    sealed_reference_lock_sha256: str | None = None,
    calibration_tile_ids: Sequence[str] | None = None,
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    generative_identities = {
        identity for identity in formal_model_identities() if method_spec(identity).generative
    }
    expected = {
        (identity, seed, target)
        for identity in generative_identities
        for seed in FORMAL_SEEDS
        for target in ("pixel", "footprint")
    }
    checkpoint_owner = {
        str(payload["checkpoint_sha256"]): key for key, payload in checkpoints.items()
    }
    observed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for path in paths:
        calibrator, payload = read_calibration_manifest(path)
        owner = checkpoint_owner.get(str(payload.get("checkpoint_sha256", "")))
        if owner is None:
            raise ValueError(f"calibration binds an unknown formal checkpoint: {path}")
        identity, seed = owner
        if str(payload.get("method")) not in {identity, str(checkpoints[owner].get("method"))}:
            raise ValueError(f"calibration method differs from checkpoint identity: {path}")
        if payload.get("calibration_asset_lock_sha256") != asset_lock_sha256:
            raise ValueError(f"calibration uses a different asset lock: {path}")
        if int(payload.get("training_seed", -1)) != seed:
            raise ValueError(f"calibration seed differs from checkpoint owner: {path}")
        if role_manifest_sha256 is not None and payload.get(
            "role_manifest_sha256"
        ) != role_manifest_sha256:
            raise ValueError(f"calibration uses a different role manifest: {path}")
        if sealed_reference_lock_sha256 is not None and payload.get(
            "sealed_reference_lock_sha256"
        ) != sealed_reference_lock_sha256:
            raise ValueError(f"calibration uses a different sealed reference lock: {path}")
        if calibration_tile_ids is not None and set(str(value) for value in payload.get(
            "tile_ids", ()
        )) != set(str(value) for value in calibration_tile_ids):
            raise ValueError(f"calibration tile set differs from the frozen role: {path}")
        if payload.get("role") != "uncertainty_calibration":
            raise ValueError(f"calibration role is not uncertainty_calibration: {path}")
        if int(payload.get("sample_count", 0)) <= 0:
            raise ValueError(f"calibration sample count is not positive: {path}")
        if calibrator.target == "footprint" and int(payload.get("sample_count", -1)) != 12 * 128:
            raise ValueError(f"footprint calibration sample count must be exactly 1536: {path}")
        key = (identity, seed, calibrator.target)
        if key in observed:
            raise ValueError(f"duplicate calibration identity/seed/target: {key}")
        observed[key] = payload
    if set(observed) != expected:
        missing = sorted(expected.difference(observed))
        unexpected = sorted(set(observed).difference(expected))
        raise ValueError(f"calibration set is not exact; missing={missing}, unexpected={unexpected}")
    for identity in sorted(generative_identities):
        for seed in FORMAL_SEEDS:
            pixel = observed[(identity, seed, "pixel")]
            footprint = observed[(identity, seed, "footprint")]
            for field in (
                "fitting_input_sha256",
                "tile_ids",
                "inference_manifests",
                "role_manifest_sha256",
                "sealed_reference_lock_sha256",
                "calibration_asset_lock_sha256",
            ):
                if pixel.get(field) != footprint.get(field):
                    raise ValueError(
                        f"pixel/footprint calibration pair differs for {(identity, seed)}: {field}"
                    )
    return observed


def validate_power_report(
    path: Path, freeze: Mapping[str, Any]
) -> Mapping[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != "paper2.power-analysis/v1" or payload.get("status") != "complete":
        raise ValueError("power report must be a completed paper2.power-analysis/v1 artifact")
    if payload.get("test_results_inspected") is not False:
        raise ValueError("power analysis must certify that test results were not inspected")
    observed_hash = file_sha256(path)
    if freeze.get("power_report_sha256") != observed_hash:
        raise ValueError("freeze does not bind the power-report file hash")
    for name in ("test_group_count", "power"):
        if float(payload[name]) != float(freeze[name]):
            raise ValueError(f"freeze {name} differs from power report")
    if int(payload["test_group_count"]) not in {24, 30, 36} or float(payload["power"]) < 0.8:
        raise ValueError("power report does not meet the preregistered group-count/power gate")
    reserve_prefix = payload.get("promoted_reserve_group_ids", [])
    expected_promoted = int(payload["test_group_count"]) - 24
    if len(reserve_prefix) != expected_promoted or len(reserve_prefix) != len(set(reserve_prefix)):
        raise ValueError("power report has an invalid 6-group reserve promotion prefix")
    return payload


def read_trace_locks(path: Path) -> Mapping[str, Mapping[str, str]]:
    payload = read_json(path)
    if payload.get("schema") != "paper2.expected-trace-locks/v1":
        raise ValueError("trace locks use an unsupported schema")
    values = payload.get("trace_locks")
    if not isinstance(values, Mapping):
        raise ValueError("trace locks must be a mapping keyed by matrix identifier")
    result: dict[str, dict[str, str]] = {}
    for identifier, fields in values.items():
        if not is_sha256(str(identifier)) or not isinstance(fields, Mapping):
            raise ValueError("trace-lock keys/values are malformed")
        converted = {str(name): str(digest) for name, digest in fields.items()}
        if any(not is_sha256(digest) for digest in converted.values()):
            raise ValueError(f"trace lock {identifier} contains an invalid digest")
        result[str(identifier)] = converted
    return result


def validate_g5_unlock(
    path: str | Path,
    *,
    matrix_spec_path: str | Path,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    g4_freeze_path: str | Path,
    paper2_root: str | Path,
) -> Mapping[str, Any]:
    """Reconstruct the complete G5 parent chain for formal inference/finalization."""

    root = Path(paper2_root).resolve()
    paths = {
        "g5": guard_paper2_path(path, paper2_root=root),
        "matrix_spec": guard_paper2_path(matrix_spec_path, paper2_root=root),
        "expected_matrix": guard_paper2_path(expected_matrix_path, paper2_root=root),
        "analysis_lock": guard_paper2_path(analysis_lock_path, paper2_root=root),
        "role_manifest": guard_paper2_path(role_manifest_path, paper2_root=root),
        "asset_lock": guard_paper2_path(asset_lock_path, paper2_root=root),
        "g4_freeze": guard_paper2_path(g4_freeze_path, paper2_root=root),
    }
    for value in paths.values():
        if not value.is_file():
            raise FileNotFoundError(value)

    payload = read_json(paths["g5"])
    if not isinstance(payload, Mapping):
        raise ValueError("G5 unlock must be a JSON object")
    semantic = canonical_sha256(
        {key: value for key, value in payload.items() if key != "g5_unlock_sha256"}
    )
    required_file_hashes = {
        "expected_matrix_spec_sha256": file_sha256(paths["matrix_spec"]),
        "expected_matrix_sha256": file_sha256(paths["expected_matrix"]),
        "analysis_lock_sha256": file_sha256(paths["analysis_lock"]),
        "role_manifest_sha256": file_sha256(paths["role_manifest"]),
        "asset_lock_sha256": file_sha256(paths["asset_lock"]),
        "g4_freeze_sha256": file_sha256(paths["g4_freeze"]),
    }
    if (
        payload.get("schema") != "paper2.g5-unlock/v1"
        or payload.get("status") != "complete"
        or payload.get("formal_test_unlocked") is not True
        or payload.get("g5_unlock_sha256") != semantic
        or any(payload.get(name) != digest for name, digest in required_file_hashes.items())
    ):
        raise ValueError("G5 unlock identity, semantic hash, or parent-file binding differs")

    from .analysis import read_analysis_lock
    from .g4 import read_g4_freeze
    from .matrix_spec import validate_formal_matrix_derivation

    analysis = read_analysis_lock(paths["analysis_lock"])
    if analysis.get("schema") != "paper2.analysis-lock/v1":
        raise ValueError("G5 analysis lock schema differs")
    freeze = read_g4_freeze(paths["g4_freeze"], paper2_root=root)
    roles = read_json(paths["role_manifest"])
    if not isinstance(roles, Mapping) or roles.get("schema") != "paper2.role-manifest/v1":
        raise ValueError("G5 role manifest schema differs")
    if roles.get("asset_lock_sha256") != file_sha256(paths["asset_lock"]):
        raise ValueError("G5 role manifest binds a different asset lock")
    role_tiles = roles.get("roles")
    role_groups = roles.get("geographic_groups")
    if not isinstance(role_tiles, Mapping) or not isinstance(role_groups, Mapping):
        raise ValueError("G5 role manifest lacks role/group maps")
    asset_lock = AssetLock.read(paths["asset_lock"])
    test_group_count = int(analysis.get("test_group_count", -1))
    validate_role_bindings(
        role_tiles,
        asset_lock,
        test_group_count=test_group_count,
        role_groups=role_groups,
    )
    spec, expected = validate_formal_matrix_derivation(
        matrix_spec_path=paths["matrix_spec"],
        expected_matrix_path=paths["expected_matrix"],
        analysis_lock_path=paths["analysis_lock"],
        role_tiles=role_tiles,
        paper2_root=root,
    )
    semantic_bindings = {
        "asset_lock_semantic_sha256": asset_lock.sha256,
        "role_manifest_semantic_sha256": canonical_sha256(roles),
        "g4_freeze_semantic_sha256": str(freeze["g4_freeze_sha256"]),
        "matrix_spec_semantic_sha256": str(spec["matrix_spec_sha256"]),
        "analysis_lock_semantic_sha256": str(analysis["analysis_lock_sha256"]),
        "expected_matrix_semantic_sha256": expected.sha256,
    }
    if any(payload.get(name) != digest for name, digest in semantic_bindings.items()):
        raise ValueError("G5 unlock semantic parent binding differs")
    if int(payload.get("cells", -1)) != len(expected.cells):
        raise ValueError("G5 unlock cell count differs from the formal matrix")
    if (
        analysis.get("expected_matrix_spec_sha256") != file_sha256(paths["matrix_spec"])
        or analysis.get("asset_lock_sha256") != file_sha256(paths["asset_lock"])
        or analysis.get("g4_freeze_sha256") != file_sha256(paths["g4_freeze"])
        or freeze.get("asset_lock_sha256") != file_sha256(paths["asset_lock"])
        or freeze.get("role_manifest_sha256") != file_sha256(paths["role_manifest"])
    ):
        raise ValueError("G4/analysis/G5 parent chain does not close")
    return payload
