from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
import math

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)
from .matrix import (
    EvaluationManifest,
    ExpectedMatrix,
    InferenceManifest,
    audit_expected_matrix,
)
from .models import UNetConfig, build_model, method_spec
from .training import expected_formal_checkpoint_count


E11_SCHEMA = "paper2.e11-efficiency/v1"
RUNTIME_SCHEMA = "paper2.inference-runtime/v1"
PHASE_TIMING_FIELDS = (
    "data_loading",
    "forward",
    "backward",
    "optimizer",
    "validation",
)
REPORTED_METRICS = (
    "parameter_count",
    "training_updates",
    "training_stage_wall_seconds",
    "data_loading_seconds",
    "forward_seconds",
    "backward_seconds",
    "optimizer_seconds",
    "validation_seconds",
    "gpu_hours",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "nfe",
    "seconds_per_km2",
)

ParameterCounter = Callable[[Path, Mapping[str, Any]], Mapping[str, int]]
_PARAMETER_CONTRACT_CACHE: dict[
    str, tuple[dict[str, tuple[int, ...]], dict[str, int]]
] = {}


def _semantic_manifest_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )


def _finite_nonnegative(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a measured numeric value")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a measured numeric value") from exc
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return number


def _relative_record(path: Path, *, root: Path) -> dict[str, Any]:
    guarded = guard_paper2_path(path, paper2_root=root)
    if not guarded.is_file() or guarded.stat().st_size <= 0:
        raise FileNotFoundError(guarded)
    return {
        "relative_path": guarded.relative_to(root).as_posix(),
        "sha256": file_sha256(guarded),
        "size_bytes": guarded.stat().st_size,
    }


def _artifact_path(
    manifest: InferenceManifest | EvaluationManifest,
    name: str,
    *,
    root: Path,
) -> Path:
    matches = [artifact for artifact in manifest.artifacts if artifact.name == name]
    if len(matches) != 1:
        raise ValueError(f"manifest requires exactly one {name} artifact")
    artifact = matches[0]
    path = guard_paper2_path(artifact.relative_path, paper2_root=root)
    if not path.is_file() or file_sha256(path) != artifact.sha256:
        raise ValueError(f"bound {name} artifact bytes differ")
    return path


def _default_parameter_counter(
    checkpoint_path: Path, manifest: Mapping[str, Any]
) -> Mapping[str, int]:
    """Count model parameters and verify every checkpoint tensor against its config."""
    try:
        import torch
    except (ImportError, OSError) as exc:  # pragma: no cover - server dependency
        raise RuntimeError("E11 checkpoint parameter audit requires PyTorch") from exc
    raw_config = manifest.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint manifest lacks model_config")
    config = UNetConfig(
        patch_size=int(raw_config["patch_size"]),
        sample_channels=int(raw_config["sample_channels"]),
        condition_channels=int(raw_config["condition_channels"]),
        output_channels=int(raw_config["output_channels"]),
        base_channels=int(raw_config["base_channels"]),
        channel_multipliers=tuple(int(value) for value in raw_config["channel_multipliers"]),
        residual_blocks_per_level=int(raw_config["residual_blocks_per_level"]),
        attention_max_resolution=int(raw_config["attention_max_resolution"]),
        time_embedding_channels=int(raw_config["time_embedding_channels"]),
    )
    config.validate(formal=False)
    config_hash = canonical_sha256(config.to_dict())
    cached = _PARAMETER_CONTRACT_CACHE.get(config_hash)
    if cached is None:
        model = build_model(config)
        expected_shapes = {
            name: tuple(value.shape) for name, value in model.state_dict().items()
        }
        parameter_counts = {
            "trainable_parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
            ),
            "state_tensor_element_count": int(
                sum(value.numel() for value in model.state_dict().values())
            ),
        }
        _PARAMETER_CONTRACT_CACHE[config_hash] = (expected_shapes, parameter_counts)
    else:
        expected_shapes, parameter_counts = cached
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or not isinstance(loaded.get("model"), Mapping):
        raise ValueError("checkpoint lacks a weights-only model state dictionary")
    state = loaded["model"]
    if set(state) != set(expected_shapes):
        raise ValueError("checkpoint model keys differ from the manifest architecture")
    for name, tensor in state.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(f"checkpoint tensor shape differs: {name}")
    result = dict(parameter_counts)
    if min(result.values()) <= 0:
        raise ValueError("checkpoint architecture has no measured parameters")
    return result


def _checkpoint_record(
    path: Path,
    *,
    root: Path,
    parameter_counter: ParameterCounter,
    count_parameters: bool,
    formal: bool,
) -> dict[str, Any]:
    manifest_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(manifest_path)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint manifest must be a JSON object")
    if (
        payload.get("schema_version") != 1
        or payload.get("manifest_sha256") != _semantic_manifest_hash(payload)
        or payload.get("status") not in {"running", "complete"}
    ):
        raise ValueError("checkpoint manifest identity/state/hash differs")
    identity = str(payload.get("model_identity", ""))
    seed = int(payload.get("seed", -1))
    if not identity or seed < 0:
        raise ValueError("checkpoint manifest lacks model identity or seed")
    stage = payload.get("stage")
    if not isinstance(stage, Mapping):
        raise ValueError("checkpoint manifest lacks stage timing boundaries")
    start_update = int(stage.get("start_update", -1))
    stop_update = int(stage.get("stop_update", -1))
    completed_updates = int(payload.get("completed_updates", -1))
    target_updates = int(payload.get("target_updates", -1))
    if not (0 <= start_update < stop_update == completed_updates <= target_updates):
        raise ValueError("checkpoint stage/update boundaries do not reproduce")
    if formal and target_updates != 120_000:
        raise ValueError("formal E11 checkpoint target is not 120000 updates")
    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping) or int(
        training_config.get("updates", -1)
    ) != target_updates:
        raise ValueError("checkpoint training configuration update count differs")

    cumulative_phase = payload.get("phase_timing_seconds")
    stage_phase = payload.get("stage_phase_timing_seconds")
    if (
        not isinstance(cumulative_phase, Mapping)
        or set(cumulative_phase) != set(PHASE_TIMING_FIELDS)
        or not isinstance(stage_phase, Mapping)
        or set(stage_phase) != set(PHASE_TIMING_FIELDS)
    ):
        raise ValueError(
            "checkpoint lacks exact data/forward/backward/optimizer/validation timings"
        )
    phase_seconds = {
        name: _finite_nonnegative(f"checkpoint stage {name} time", stage_phase[name])
        for name in PHASE_TIMING_FIELDS
    }
    cumulative_phase_seconds = {
        name: _finite_nonnegative(
            f"checkpoint cumulative {name} time", cumulative_phase[name]
        )
        for name in PHASE_TIMING_FIELDS
    }
    if any(
        cumulative_phase_seconds[name] + 1.0e-12 < phase_seconds[name]
        for name in PHASE_TIMING_FIELDS
    ):
        raise ValueError("checkpoint cumulative phase time is below its current stage")
    telemetry = payload.get("telemetry")
    ranks = telemetry.get("ranks") if isinstance(telemetry, Mapping) else None
    if not isinstance(ranks, list) or not ranks:
        raise ValueError("checkpoint manifest lacks per-rank telemetry")
    rank_records: list[dict[str, Any]] = []
    for rank in ranks:
        if not isinstance(rank, Mapping):
            raise ValueError("checkpoint rank telemetry is malformed")
        wall = _finite_nonnegative("checkpoint rank wall time", rank.get("wall_seconds"), positive=True)
        allocated = int(rank.get("peak_allocated_bytes", -1))
        reserved = int(rank.get("peak_reserved_bytes", -1))
        total = int(rank.get("total_memory_bytes", -1))
        device_uuid = str(rank.get("device_uuid") or "")
        if allocated < 0 or reserved < allocated or total <= 0 or reserved > total:
            raise ValueError("checkpoint rank memory telemetry is invalid")
        if formal and not device_uuid:
            raise ValueError("formal checkpoint rank lacks a GPU UUID")
        rank_records.append(
            {
                "rank": int(rank.get("rank", -1)),
                "device_uuid": device_uuid,
                "wall_seconds": wall,
                "peak_memory_allocated_bytes": allocated,
                "peak_memory_reserved_bytes": reserved,
                "total_memory_bytes": total,
            }
        )
    manifest_wall = _finite_nonnegative(
        "checkpoint stage wall time", payload.get("wall_seconds"), positive=True
    )
    if not math.isclose(
        manifest_wall,
        max(record["wall_seconds"] for record in rank_records),
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    ):
        raise ValueError("checkpoint wall time differs from per-rank telemetry")
    checkpoint_name = str(payload.get("checkpoint_file", ""))
    if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_file must be one filename beside its manifest")
    checkpoint_path = guard_paper2_path(
        manifest_path.parent / checkpoint_name, paper2_root=root
    )
    if (
        not checkpoint_path.is_file()
        or file_sha256(checkpoint_path) != payload.get("checkpoint_sha256")
    ):
        raise ValueError("checkpoint bytes differ from the checkpoint manifest")
    parameter_counts: Mapping[str, int] | None = None
    if count_parameters:
        parameter_counts = parameter_counter(checkpoint_path, payload)
        required_counts = {"trainable_parameter_count", "state_tensor_element_count"}
        if set(parameter_counts) != required_counts or any(
            isinstance(value, bool) or int(value) <= 0 for value in parameter_counts.values()
        ):
            raise ValueError("checkpoint parameter counter returned an invalid contract")
    return {
        "manifest": _relative_record(manifest_path, root=root),
        "manifest_semantic_sha256": str(payload["manifest_sha256"]),
        "checkpoint": _relative_record(checkpoint_path, root=root),
        "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        "model_identity": identity,
        "seed": seed,
        "status": str(payload["status"]),
        "start_update": start_update,
        "stop_update": stop_update,
        "target_updates": target_updates,
        "selected_update": int(payload.get("selected_update", payload.get("update", -1))),
        "patch_size": int(dict(payload.get("model_config", {})).get("patch_size", -1)),
        "stage_wall_seconds": manifest_wall,
        "phase_timing_seconds": phase_seconds,
        "cumulative_phase_timing_seconds": cumulative_phase_seconds,
        "rank_telemetry": rank_records,
        "training_gpu_hours": float(
            sum(record["wall_seconds"] for record in rank_records) / 3600.0
        ),
        "peak_memory_allocated_bytes": max(
            record["peak_memory_allocated_bytes"] for record in rank_records
        ),
        "peak_memory_reserved_bytes": max(
            record["peak_memory_reserved_bytes"] for record in rank_records
        ),
        "parameter_counts": None if parameter_counts is None else dict(parameter_counts),
    }


def _runtime_record(
    manifest: InferenceManifest, *, root: Path, formal: bool
) -> dict[str, Any]:
    runtime_path = _artifact_path(manifest, "runtime", root=root)
    payload = read_json(runtime_path)
    if not isinstance(payload, Mapping):
        raise ValueError("inference runtime must be a JSON object")
    if payload.get("schema") != RUNTIME_SCHEMA or payload.get("status") != "complete":
        raise ValueError("inference runtime lacks the registered complete schema")
    wall = _finite_nonnegative("inference wall time", payload.get("wall_seconds"), positive=True)
    area = _finite_nonnegative("inference area", payload.get("area_km2"), positive=True)
    seconds_per_km2 = _finite_nonnegative(
        "inference seconds/km2", payload.get("seconds_per_km2"), positive=True
    )
    if not math.isclose(seconds_per_km2, wall / area, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError("inference seconds/km2 does not reproduce from wall time and area")
    nfe = payload.get("nfe")
    if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe < 0:
        raise ValueError("inference NFE must be a measured nonnegative integer")
    learned = method_spec(manifest.key.method).learned
    allocated_raw = payload.get("peak_gpu_memory_allocated_bytes")
    reserved_raw = payload.get("peak_gpu_memory_reserved_bytes")
    device_uuid = payload.get("device_uuid")
    allocated = None if allocated_raw is None else int(allocated_raw)
    reserved = None if reserved_raw is None else int(reserved_raw)
    if learned and formal:
        if allocated is None or reserved is None or allocated < 0 or reserved < allocated:
            raise ValueError("formal learned inference lacks valid allocated/reserved GPU memory")
        if not isinstance(device_uuid, str) or not device_uuid.strip():
            raise ValueError("formal learned inference lacks a GPU UUID")
    elif (allocated is None) != (reserved is None):
        raise ValueError("inference allocated/reserved memory must be present together")
    if allocated is not None and (allocated < 0 or reserved is None or reserved < allocated):
        raise ValueError("inference allocated/reserved memory is invalid")
    return {
        "key_sha256": manifest.key.identifier,
        "key": manifest.key.to_dict(),
        "runtime": _relative_record(runtime_path, root=root),
        "wall_seconds": wall,
        "area_km2": area,
        "seconds_per_km2": seconds_per_km2,
        "nfe": nfe,
        "peak_memory_allocated_bytes": allocated,
        "peak_memory_reserved_bytes": reserved,
        "device_uuid": device_uuid,
        "inference_gpu_hours": wall / 3600.0 if device_uuid else 0.0,
    }


def _metric_record(manifest: EvaluationManifest, *, root: Path) -> dict[str, float]:
    metric_path = _artifact_path(manifest, "metrics", root=root)
    payload = read_json(metric_path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "paper2.evaluation/v1"
        or payload.get("key") != manifest.key.to_dict()
    ):
        raise ValueError("evaluation metric artifact identity differs")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("evaluation metric artifact lacks metrics")
    rmse = _finite_nonnegative("evaluation RMSE", metrics.get("rmse"))
    return {"rmse": rmse}


def _require_stage_coverage(records: Sequence[Mapping[str, Any]], *, formal: bool) -> None:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["model_identity"]), int(record["seed"]))].append(record)
    for identity_seed, stages in grouped.items():
        ordered = sorted(stages, key=lambda record: int(record["start_update"]))
        target_updates = {int(record["target_updates"]) for record in ordered}
        if len(target_updates) != 1:
            raise ValueError(f"checkpoint stages change target updates: {identity_seed}")
        cursor = 0
        for record in ordered:
            start = int(record["start_update"])
            stop = int(record["stop_update"])
            if start != cursor:
                raise ValueError(f"checkpoint stages have a gap/overlap: {identity_seed}")
            cursor = stop
        target = next(iter(target_updates))
        if cursor != target or ordered[-1]["status"] != "complete":
            raise ValueError(f"checkpoint stages do not close training: {identity_seed}")
        summed = {
            name: float(
                sum(float(record["phase_timing_seconds"][name]) for record in ordered)
            )
            for name in PHASE_TIMING_FIELDS
        }
        cumulative = ordered[-1]["cumulative_phase_timing_seconds"]
        if any(
            not math.isclose(
                summed[name],
                float(cumulative[name]),
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
            for name in PHASE_TIMING_FIELDS
        ):
            raise ValueError(f"checkpoint cumulative phase timings do not close: {identity_seed}")
        if formal and target != 120_000:
            raise ValueError("formal checkpoint stage coverage does not reach 120000")


def build_e11_efficiency(
    *,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    inference_manifest_paths: Sequence[str | Path],
    evaluation_manifest_paths: Sequence[str | Path],
    checkpoint_manifest_paths: Sequence[str | Path],
    paper2_root: str | Path,
    formal: bool = True,
    parameter_counter: ParameterCounter = _default_parameter_counter,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    expected = ExpectedMatrix.read(expected_path)
    if expected.analysis_lock_sha256 != file_sha256(analysis_path):
        raise ValueError("E11 expected matrix binds a different analysis lock")
    inference_paths = [
        guard_paper2_path(path, paper2_root=root) for path in inference_manifest_paths
    ]
    evaluation_paths = [
        guard_paper2_path(path, paper2_root=root) for path in evaluation_manifest_paths
    ]
    if len(inference_paths) != len(set(inference_paths)) or len(evaluation_paths) != len(
        set(evaluation_paths)
    ):
        raise ValueError("E11 manifest paths contain duplicates")
    inferences = [InferenceManifest.read(path) for path in inference_paths]
    evaluations = [EvaluationManifest.read(path) for path in evaluation_paths]
    audit = audit_expected_matrix(expected, inferences, evaluations, paper2_root=root)
    if not audit.ok:
        raise ValueError("E11 cannot aggregate an open expected matrix")

    final_checkpoint_hashes = {
        dict(manifest.trace_hashes)["checkpoint_manifest_sha256"]
        for manifest in inferences
        if method_spec(manifest.key.method).learned
    }
    checkpoint_paths = [
        guard_paper2_path(path, paper2_root=root) for path in checkpoint_manifest_paths
    ]
    if not checkpoint_paths or len(checkpoint_paths) != len(set(checkpoint_paths)):
        raise ValueError("E11 requires unique checkpoint manifest paths")
    supplied_hashes = {file_sha256(path) for path in checkpoint_paths}
    if not final_checkpoint_hashes.issubset(supplied_hashes):
        raise ValueError("E11 lacks a checkpoint manifest referenced by inference")
    checkpoint_records = [
        _checkpoint_record(
            path,
            root=root,
            parameter_counter=parameter_counter,
            count_parameters=file_sha256(path) in final_checkpoint_hashes,
            formal=formal,
        )
        for path in checkpoint_paths
    ]
    _require_stage_coverage(checkpoint_records, formal=formal)
    final_records = [
        record
        for record in checkpoint_records
        if record["manifest"]["sha256"] in final_checkpoint_hashes
    ]
    if len(final_records) != len(final_checkpoint_hashes):
        raise ValueError("E11 final checkpoint manifests are duplicated")
    if formal and len(final_records) != expected_formal_checkpoint_count():
        raise ValueError("formal E11 does not contain the exact 33 learned checkpoints")
    final_by_manifest_hash = {
        str(record["manifest"]["sha256"]): record for record in final_records
    }
    for inference in inferences:
        if not method_spec(inference.key.method).learned:
            continue
        trace = dict(inference.trace_hashes)
        checkpoint = final_by_manifest_hash.get(trace["checkpoint_manifest_sha256"])
        if checkpoint is None or checkpoint["checkpoint_sha256"] != trace["checkpoint_sha256"]:
            raise ValueError("inference checkpoint bytes differ from E11 checkpoint evidence")

    runtimes = [_runtime_record(manifest, root=root, formal=formal) for manifest in inferences]
    evaluation_by_id = {manifest.key.identifier: manifest for manifest in evaluations}
    pareto_rows = []
    for runtime in runtimes:
        evaluation = evaluation_by_id[runtime["key_sha256"]]
        pareto_rows.append(
            {
                "key_sha256": runtime["key_sha256"],
                "experiment": runtime["key"]["experiment"],
                "tile_id": runtime["key"]["tile_id"],
                "method": runtime["key"]["method"],
                "budget": runtime["key"]["budget"],
                "rmse": _metric_record(evaluation, root=root)["rmse"],
                "wall_seconds": runtime["wall_seconds"],
                "nfe": runtime["nfe"],
            }
        )

    training_gpu_hours = float(
        sum(float(record["training_gpu_hours"]) for record in checkpoint_records)
    )
    inference_gpu_hours = float(sum(float(record["inference_gpu_hours"]) for record in runtimes))
    memory_allocated = [
        int(value)
        for value in (
            [record["peak_memory_allocated_bytes"] for record in checkpoint_records]
            + [record["peak_memory_allocated_bytes"] for record in runtimes]
        )
        if value is not None
    ]
    memory_reserved = [
        int(value)
        for value in (
            [record["peak_memory_reserved_bytes"] for record in checkpoint_records]
            + [record["peak_memory_reserved_bytes"] for record in runtimes]
        )
        if value is not None
    ]
    phase_totals = {
        name: float(
            sum(float(record["phase_timing_seconds"][name]) for record in checkpoint_records)
        )
        for name in PHASE_TIMING_FIELDS
    }
    payload: dict[str, Any] = {
        "schema": E11_SCHEMA,
        "status": "complete",
        "formal": formal,
        "reported_metrics": list(REPORTED_METRICS),
        "expected_matrix": _relative_record(expected_path, root=root),
        "analysis_lock": _relative_record(analysis_path, root=root),
        "matrix_audit": audit.to_dict(),
        "matrix_audit_sha256": canonical_sha256(audit.to_dict()),
        "source_manifests": {
            "inference": [_relative_record(path, root=root) for path in sorted(inference_paths)],
            "evaluation": [_relative_record(path, root=root) for path in sorted(evaluation_paths)],
            "checkpoint": [_relative_record(path, root=root) for path in sorted(checkpoint_paths)],
        },
        "counts": {
            "expected_cells": len(expected.cells),
            "inference_cells": len(inferences),
            "evaluation_cells": len(evaluations),
            "training_stage_manifests": len(checkpoint_records),
            "final_checkpoint_manifests": len(final_records),
        },
        "summary": {
            "training_gpu_hours": training_gpu_hours,
            "inference_gpu_hours": inference_gpu_hours,
            "gpu_hours": training_gpu_hours + inference_gpu_hours,
            "peak_memory_allocated_bytes": max(memory_allocated),
            "peak_memory_reserved_bytes": max(memory_reserved),
            "nfe": int(sum(int(record["nfe"]) for record in runtimes)),
            "wall_seconds": float(sum(float(record["wall_seconds"]) for record in runtimes)),
            "area_km2": float(sum(float(record["area_km2"]) for record in runtimes)),
            "seconds_per_km2": float(
                sum(float(record["wall_seconds"]) for record in runtimes)
                / sum(float(record["area_km2"]) for record in runtimes)
            ),
            "training_stage_wall_seconds": float(
                sum(float(record["stage_wall_seconds"]) for record in checkpoint_records)
            ),
            "phase_timing_seconds": phase_totals,
        },
        "checkpoints": checkpoint_records,
        "inference_runtime": runtimes,
        "accuracy_cost_rows": sorted(
            pareto_rows, key=lambda record: str(record["key_sha256"])
        ),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def write_e11_efficiency(
    path: str | Path,
    **kwargs: Any,
) -> Path:
    root = Path(kwargs["paper2_root"]).resolve()
    return write_canonical_json(
        path, build_e11_efficiency(**kwargs), paper2_root=root
    )


def read_e11_efficiency(
    path: str | Path, *, paper2_root: str | Path, verify_sources: bool = True
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    report_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(report_path)
    if not isinstance(payload, Mapping):
        raise ValueError("E11 efficiency report must be a JSON object")
    if (
        payload.get("schema") != E11_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("formal") is not True
        or tuple(payload.get("reported_metrics", [])) != REPORTED_METRICS
        or payload.get("content_sha256")
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "content_sha256"}
        )
    ):
        raise ValueError("E11 report identity/state/metric/hash differs")
    audit = payload.get("matrix_audit")
    if not isinstance(audit, Mapping) or audit.get("ok") is not True or payload.get(
        "matrix_audit_sha256"
    ) != canonical_sha256(audit):
        raise ValueError("E11 report does not bind a closed matrix audit")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("E11 report lacks its summary")
    for name in (
        "gpu_hours",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "nfe",
        "seconds_per_km2",
    ):
        _finite_nonnegative(f"E11 {name}", summary.get(name))
    if verify_sources:
        for section in ("expected_matrix", "analysis_lock"):
            record = payload.get(section)
            if not isinstance(record, Mapping):
                raise ValueError(f"E11 lacks {section} binding")
            source = guard_paper2_path(
                str(record.get("relative_path", "")), paper2_root=root
            )
            if (
                not source.is_file()
                or file_sha256(source) != record.get("sha256")
                or source.stat().st_size != int(record.get("size_bytes", -1))
            ):
                raise ValueError(f"E11 {section} bytes differ")
        source_sets = payload.get("source_manifests")
        if not isinstance(source_sets, Mapping):
            raise ValueError("E11 source manifest set is malformed")
        for records in source_sets.values():
            if not isinstance(records, list):
                raise ValueError("E11 source manifest records are malformed")
            for record in records:
                source = guard_paper2_path(
                    str(record.get("relative_path", "")), paper2_root=root
                )
                if (
                    not source.is_file()
                    or file_sha256(source) != record.get("sha256")
                    or source.stat().st_size != int(record.get("size_bytes", -1))
                ):
                    raise ValueError("E11 source manifest bytes differ")
    return dict(payload)
