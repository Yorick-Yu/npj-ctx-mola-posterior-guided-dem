"""Frozen, resumable orchestration contract for the 33 formal checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import math

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)
from .models import UNetConfig, formal_model_identities
from .training import FORMAL_SEEDS, TrainingConfig
from .training_strategy import read_real_training_strategy_receipt


SELECTION_SCHEMA = "paper2.training-hyperparameter-selection/v1"
FORMAL_TRAINING_CONFIG_SCHEMA = "paper2.formal-training-config/v2"
FORMAL_TRAINING_UPDATES = 120_000
FORMAL_VALIDATION_INTERVAL = 2_000
FORMAL_TRAINING_CHUNK_UPDATES = 10_000
FORMAL_WORLD_SIZE = 2
VALIDATION_G_MIN_GRID = (0.02, 0.05, 0.10)
VALIDATION_ALPHA_GRID = (0.5, 1.0, 2.0)
VALIDATION_CANDIDATE_TARGET_UPDATES = 10_000
VALIDATION_FINALIST_TARGET_UPDATES = 30_000
VALIDATION_FINALIST_COUNT = 2
SELECTED_HYPERPARAMETERS = {
    "g_min",
    "alpha",
    "endpoint_weight",
    "gradient_weight",
    "ddpm_cosine_offset",
}


def _semantic_payload(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != field}


def read_training_selection(
    path: str | Path, *, paper2_root: str | Path
) -> Mapping[str, Any]:
    root = Path(paper2_root).resolve()
    selection_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(selection_path)
    if not isinstance(payload, Mapping):
        raise ValueError("training hyperparameter selection must be a JSON object")
    if (
        payload.get("schema") != SELECTION_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("selection_role") != "bridge_val"
        or payload.get("test_results_inspected") is not False
    ):
        raise ValueError("training selection identity, role, or blind state differs")
    if payload.get("selection_sha256") != canonical_sha256(
        _semantic_payload(payload, "selection_sha256")
    ):
        raise ValueError("training selection semantic hash mismatch")
    selected = payload.get("selected")
    if not isinstance(selected, Mapping) or set(selected) != SELECTED_HYPERPARAMETERS:
        raise ValueError("training selection hyperparameter set differs")
    values = {name: float(selected[name]) for name in SELECTED_HYPERPARAMETERS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("training selection contains non-finite values")
    if not 0.0 < values["g_min"] <= 1.0 or values["alpha"] <= 0.0:
        raise ValueError("training selection g_min/alpha is invalid")
    if values["endpoint_weight"] < 0.0 or values["gradient_weight"] < 0.0:
        raise ValueError("training selection loss weights must be nonnegative")
    if not 0.0 <= values["ddpm_cosine_offset"] < 1.0:
        raise ValueError("training selection DDPM cosine offset is invalid")
    tile_ids = payload.get("tile_ids")
    if not isinstance(tile_ids, list) or len(tile_ids) != 12 or len(set(tile_ids)) != 12:
        raise ValueError("training selection must contain exactly 12 bridge_val tiles")
    expected_jobs = [
        (g_min, alpha)
        for g_min in VALIDATION_G_MIN_GRID
        for alpha in VALIDATION_ALPHA_GRID
    ]
    grid = payload.get("candidate_grid")
    if grid != {
        "g_min": list(VALIDATION_G_MIN_GRID),
        "alpha": list(VALIDATION_ALPHA_GRID),
        "phase1_updates_per_candidate": VALIDATION_CANDIDATE_TARGET_UPDATES,
        "finalist_count": VALIDATION_FINALIST_COUNT,
        "finalist_updates": VALIDATION_FINALIST_TARGET_UPDATES,
        "seed": FORMAL_SEEDS[0],
    } or payload.get("selection_rule") != (
        "phase1_top2_then_minimum_30k_best_bridge_val_metric_then_frozen_order"
    ):
        raise ValueError("training selection successive-halving contract differs")

    def validate_records(name: str, expected_count: int, expected_updates: int):
        records = payload.get(name)
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(f"training selection {name} count differs")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"training selection {name} record is malformed")
            if int(record.get("completed_updates", -1)) != expected_updates or not math.isfinite(
                float(record.get("validation_metric", float("nan")))
            ):
                raise ValueError(f"training selection {name} metric/update differs")
            evidence = record.get("checkpoint_manifest")
            if not isinstance(evidence, Mapping):
                raise ValueError(f"training selection {name} lacks checkpoint evidence")
            evidence_path = guard_paper2_path(
                root / str(evidence.get("relative_path", "")), paper2_root=root
            )
            if not evidence_path.is_file() or file_sha256(evidence_path) != evidence.get(
                "sha256"
            ):
                raise ValueError(f"training selection {name} checkpoint bytes differ")
        return records

    phase1 = validate_records(
        "phase1_candidates", len(expected_jobs), VALIDATION_CANDIDATE_TARGET_UPDATES
    )
    observed_phase1 = [
        (float(record["g_min"]), float(record["alpha"])) for record in phase1
    ]
    if observed_phase1 != expected_jobs:
        raise ValueError("training selection phase1 grid order differs")
    ranked = sorted(
        phase1,
        key=lambda record: (
            float(record["validation_metric"]),
            expected_jobs.index((float(record["g_min"]), float(record["alpha"]))),
        ),
    )
    finalists = validate_records(
        "finalists", VALIDATION_FINALIST_COUNT, VALIDATION_FINALIST_TARGET_UPDATES
    )
    expected_finalists = [
        (float(record["g_min"]), float(record["alpha"]))
        for record in ranked[:VALIDATION_FINALIST_COUNT]
    ]
    observed_finalists = [
        (float(record["g_min"]), float(record["alpha"])) for record in finalists
    ]
    if observed_finalists != expected_finalists:
        raise ValueError("training selection finalists differ from the blind phase1 top two")
    winner = min(
        finalists,
        key=lambda record: (
            float(record["validation_metric"]),
            observed_finalists.index(
                (float(record["g_min"]), float(record["alpha"]))
            ),
        ),
    )
    if (
        values["g_min"] != float(winner["g_min"])
        or values["alpha"] != float(winner["alpha"])
        or values["endpoint_weight"] != 0.1
        or values["gradient_weight"] != 0.1
        or values["ddpm_cosine_offset"] != 0.008
    ):
        raise ValueError("training selection does not match the blind finalist winner")
    return payload


def build_formal_training_config(
    *,
    paper2_root: str | Path,
    selection_path: str | Path,
    strategy_receipt_path: str | Path,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    selected_path = guard_paper2_path(selection_path, paper2_root=root)
    selection = read_training_selection(selected_path, paper2_root=root)
    strategy_path = guard_paper2_path(strategy_receipt_path, paper2_root=root)
    strategy = read_real_training_strategy_receipt(strategy_path, paper2_root=root)
    selected = dict(selection["selected"])
    payload: dict[str, Any] = {
        "schema": FORMAL_TRAINING_CONFIG_SCHEMA,
        "status": "frozen",
        "test_results_inspected": False,
        "selection_role": "bridge_val",
        "selection_evidence": {
            "relative_path": selected_path.relative_to(root).as_posix(),
            "sha256": file_sha256(selected_path),
        },
        "training_strategy_evidence": {
            "relative_path": strategy_path.relative_to(root).as_posix(),
            "sha256": file_sha256(strategy_path),
            "receipt_sha256": strategy["receipt_sha256"],
        },
        "methods": list(formal_model_identities()),
        "seeds": list(FORMAL_SEEDS),
        "updates": FORMAL_TRAINING_UPDATES,
        "validation_interval": FORMAL_VALIDATION_INTERVAL,
        "chunk_updates": FORMAL_TRAINING_CHUNK_UPDATES,
        "execution_strategy": strategy["chosen_mode"],
        "world_size": int(strategy["world_size"]),
        "concurrent_model_jobs": int(strategy["concurrent_model_jobs"]),
        "micro_batch_per_gpu": 1,
        "effective_batch": 8,
        "optimizer": "adamw",
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-2,
        "network_precision": "fp16",
        "data_loader_workers": 0,
        "early_stopping": False,
        **selected,
    }
    payload["formal_training_config_sha256"] = canonical_sha256(payload)
    return payload


def write_formal_training_config(
    path: str | Path,
    *,
    paper2_root: str | Path,
    selection_path: str | Path,
    strategy_receipt_path: str | Path,
) -> Path:
    payload = build_formal_training_config(
        paper2_root=paper2_root,
        selection_path=selection_path,
        strategy_receipt_path=strategy_receipt_path,
    )
    return write_canonical_json(path, payload, paper2_root=paper2_root)


def read_formal_training_config(
    path: str | Path, *, paper2_root: str | Path
) -> Mapping[str, Any]:
    root = Path(paper2_root).resolve()
    config_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(config_path)
    if not isinstance(payload, Mapping):
        raise ValueError("formal training config must be a JSON object")
    if payload.get("formal_training_config_sha256") != canonical_sha256(
        _semantic_payload(payload, "formal_training_config_sha256")
    ):
        raise ValueError("formal training config semantic hash mismatch")
    expected = {
        "schema": FORMAL_TRAINING_CONFIG_SCHEMA,
        "status": "frozen",
        "test_results_inspected": False,
        "selection_role": "bridge_val",
        "methods": list(formal_model_identities()),
        "seeds": list(FORMAL_SEEDS),
        "updates": FORMAL_TRAINING_UPDATES,
        "validation_interval": FORMAL_VALIDATION_INTERVAL,
        "chunk_updates": FORMAL_TRAINING_CHUNK_UPDATES,
        "micro_batch_per_gpu": 1,
        "effective_batch": 8,
        "optimizer": "adamw",
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-2,
        "network_precision": "fp16",
        "data_loader_workers": 0,
        "early_stopping": False,
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatches:
        raise ValueError(f"formal training config differs: {mismatches}")
    strategy_evidence = payload.get("training_strategy_evidence")
    if not isinstance(strategy_evidence, Mapping):
        raise ValueError("formal training config lacks real-data strategy evidence")
    strategy_path = guard_paper2_path(
        root / str(strategy_evidence.get("relative_path", "")), paper2_root=root
    )
    if not strategy_path.is_file() or file_sha256(strategy_path) != strategy_evidence.get(
        "sha256"
    ):
        raise ValueError("formal training strategy evidence bytes differ")
    strategy = read_real_training_strategy_receipt(strategy_path, paper2_root=root)
    strategy_expected = {
        "execution_strategy": strategy["chosen_mode"],
        "world_size": int(strategy["world_size"]),
        "concurrent_model_jobs": int(strategy["concurrent_model_jobs"]),
    }
    if strategy_evidence.get("receipt_sha256") != strategy["receipt_sha256"] or any(
        payload.get(name) != value for name, value in strategy_expected.items()
    ):
        raise ValueError("formal training strategy choice differs from its receipt")
    evidence = payload.get("selection_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("formal training config lacks selection evidence")
    evidence_path = guard_paper2_path(
        root / str(evidence.get("relative_path", "")), paper2_root=root
    )
    if not evidence_path.is_file() or file_sha256(evidence_path) != evidence.get("sha256"):
        raise ValueError("formal training selection evidence bytes differ")
    selection = read_training_selection(evidence_path, paper2_root=root)
    for name in SELECTED_HYPERPARAMETERS:
        if float(payload.get(name, float("nan"))) != float(selection["selected"][name]):
            raise ValueError(f"formal training config {name} differs from selection")
    return payload


def training_config_for_identity(
    manifest: Mapping[str, Any], *, method: str, seed: int
) -> TrainingConfig:
    if method not in formal_model_identities() or seed not in FORMAL_SEEDS:
        raise ValueError("formal training job identity is not in the exact 33-model matrix")
    config = TrainingConfig(
        method=method,
        seed=seed,
        updates=int(manifest["updates"]),
        micro_batch_per_gpu=int(manifest["micro_batch_per_gpu"]),
        effective_batch=int(manifest["effective_batch"]),
        optimizer=str(manifest["optimizer"]),
        learning_rate=float(manifest["learning_rate"]),
        weight_decay=float(manifest["weight_decay"]),
        validation_interval=int(manifest["validation_interval"]),
        network_precision=str(manifest["network_precision"]),
        data_loader_workers=int(manifest["data_loader_workers"]),
        early_stopping=bool(manifest["early_stopping"]),
        g_min=float(manifest["g_min"]),
        alpha=float(manifest["alpha"]),
        endpoint_weight=float(manifest["endpoint_weight"]),
        gradient_weight=float(manifest["gradient_weight"]),
        ddpm_cosine_offset=float(manifest["ddpm_cosine_offset"]),
    )
    config.validate(formal=True, world_size=int(manifest["world_size"]))
    UNetConfig().validate(formal=True)
    return config


def formal_training_job_schedule(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, int, int | None], ...]:
    """Freeze every model identity to a stable device lane for all resume chunks."""

    strategy = str(manifest.get("execution_strategy", ""))
    world_size = int(manifest.get("world_size", 0))
    concurrency = int(manifest.get("concurrent_model_jobs", 0))
    if {
        "ddp": (2, 1),
        "independent": (1, 2),
    }.get(strategy) != (world_size, concurrency):
        raise ValueError("formal training scheduler differs from the frozen strategy")
    jobs = [
        (method, seed)
        for method in formal_model_identities()
        for seed in FORMAL_SEEDS
    ]
    return tuple(
        (
            method,
            seed,
            index % 2 if strategy == "independent" else None,
        )
        for index, (method, seed) in enumerate(jobs)
    )


def next_training_stop(completed_updates: int) -> int:
    completed = int(completed_updates)
    if completed < 0 or completed > FORMAL_TRAINING_UPDATES:
        raise ValueError("completed formal updates lie outside [0,120000]")
    if completed % FORMAL_VALIDATION_INTERVAL:
        raise ValueError("formal resume prefix must end on a validation boundary")
    if completed == FORMAL_TRAINING_UPDATES:
        return completed
    return min(
        FORMAL_TRAINING_UPDATES,
        ((completed // FORMAL_TRAINING_CHUNK_UPDATES) + 1)
        * FORMAL_TRAINING_CHUNK_UPDATES,
    )


def next_validation_candidate_stop(
    completed_updates: int, *, target_updates: int
) -> int:
    completed = int(completed_updates)
    target = int(target_updates)
    if target not in {
        VALIDATION_CANDIDATE_TARGET_UPDATES,
        VALIDATION_FINALIST_TARGET_UPDATES,
    }:
        raise ValueError("validation candidate target must be 10000 or 30000")
    if completed < 0 or completed > target:
        raise ValueError("validation candidate updates lie outside its target prefix")
    if completed % FORMAL_VALIDATION_INTERVAL:
        raise ValueError("validation candidate prefix must end on a validation boundary")
    if completed == target:
        return completed
    return min(
        target,
        completed + FORMAL_VALIDATION_INTERVAL,
    )
