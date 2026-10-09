"""Recoverable execution of the G5-locked physical experiment matrix.

The coordinator never exposes a sealed reference path to inference.  Each unit
runs inference first, then creates evaluator-only held-out footprint returns and
evaluates the immutable inference artifacts.  E6 and E7 are grouped so their
shared-path invariants cannot be broken by scheduling or resume behavior.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
import contextlib
import fcntl
import io
import json
import math
import os
import shutil
import time

from .audit_footprints import (
    AuditFootprintObservations,
    read_audit_footprint_plan,
    simulate_audit_footprint_observations,
)
from .cell_bindings import CELL_BINDING_SCHEMA
from .contracts import PlanContract
from .e7_prefix import E7_PATH_COUNTS, derive_e7_prefixes
from .g5 import validate_g5_unlock
from .hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256 as _uncached_file_sha256,
    guard_paper2_path,
    read_json,
)
from .matrix import (
    EvaluationManifest,
    ExpectedCell,
    ExpectedMatrix,
    InferenceManifest,
    audit_expected_matrix,
)
from .models import canonical_model_identity, method_spec
from .phase_gates import G6_DELAYED_EXPERIMENTS, validate_g6_unlock
from .sealed_reference import read_sealed_reference_lock


WORKLIST_SCHEMA = "paper2.formal-execution-worklist/v1"
PROGRESS_SCHEMA = "paper2.formal-execution-worker-progress/v1"
CAPACITY_PREFLIGHT_SCHEMA = "paper2.formal-execution-capacity-preflight/v1"
EXPERIMENTS = ("E3", "E3b", "E4", "E5", "E6", "E7", "E8", "E9", "E10")
GIB = 1024**3
DEFAULT_CAPACITY_SAFETY_FACTOR = 1.25
DEFAULT_MINIMUM_FREE_BYTES = 100 * GIB
DEFAULT_PER_CELL_METADATA_BYTES = 64 * 1024
FileSignature = tuple[int, int, int, int, int]


def _file_signature(path: Path) -> FileSignature:
    status = path.stat()
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


@lru_cache(maxsize=16_384)
def _cached_file_sha256(path_text: str, signature: FileSignature) -> str:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("formal execution input changed before hashing")
    return _uncached_file_sha256(path)


def file_sha256(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return _cached_file_sha256(str(resolved), _file_signature(resolved))


@lru_cache(maxsize=512)
def _cached_plan_contract(path_text: str, signature: FileSignature) -> PlanContract:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("formal execution plan changed before cached load")
    return PlanContract.read(path)


@lru_cache(maxsize=128)
def _cached_audit_footprint_plan(path_text: str, signature: FileSignature) -> Any:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("formal audit plan changed before cached load")
    return read_audit_footprint_plan(path)


@lru_cache(maxsize=1_024)
def _cached_audit_footprint_observations(
    path_text: str, signature: FileSignature
) -> AuditFootprintObservations:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("formal audit observations changed before cached load")
    observations = AuditFootprintObservations.read(path)
    observations.audit_ids.setflags(write=False)
    observations.return_values_m.setflags(write=False)
    return observations


def _semantic_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_sha256({name: value for name, value in payload.items() if name != field})


def _source_tree_sha256(root: Path) -> str:
    return canonical_sha256(
        [
            (path.relative_to(root).as_posix(), file_sha256(path))
            for path in sorted(root.rglob("*.py"))
            if path.is_file()
        ]
    )


def _inside_file(path: str | Path, *, root: Path) -> Path:
    result = guard_paper2_path(path, paper2_root=root)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _inside_directory(path: str | Path, *, root: Path) -> Path:
    result = guard_paper2_path(path, paper2_root=root, allow_root=True)
    if result == root:
        raise ValueError("formal execution output must be below the Paper2 root")
    return result


@dataclass(frozen=True)
class FormalExecutionContract:
    root: Path
    expected_path: Path
    matrix_spec_path: Path
    analysis_lock_path: Path
    g5_unlock_path: Path
    g4_freeze_path: Path
    role_manifest_path: Path
    asset_lock_path: Path
    environment_lock_path: Path
    sealed_reference_lock_path: Path
    trace_locks_path: Path
    cell_bindings_path: Path
    g6_unlock_path: Path | None
    expected: ExpectedMatrix
    expected_by_id: Mapping[str, ExpectedCell]
    bindings: Mapping[str, Mapping[str, Any]]
    references: Mapping[str, Mapping[str, Any]]
    source_code_sha256: str

    def binding_path(self, cell_id: str, name: str) -> Path:
        value = self.bindings[cell_id].get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"formal cell {cell_id} lacks binding {name}")
        return _inside_file(self.root / value, root=self.root)


def load_formal_execution_contract(
    *,
    paper2_root: str | Path,
    expected_matrix_path: str | Path,
    matrix_spec_path: str | Path,
    analysis_lock_path: str | Path,
    g5_unlock_path: str | Path,
    g4_freeze_path: str | Path,
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    environment_lock_path: str | Path,
    sealed_reference_lock_path: str | Path,
    trace_locks_path: str | Path,
    cell_bindings_path: str | Path,
    g6_unlock_path: str | Path | None = None,
    require_g6: bool = False,
) -> FormalExecutionContract:
    root = Path(paper2_root).resolve()
    expected_path = _inside_file(expected_matrix_path, root=root)
    matrix_path = _inside_file(matrix_spec_path, root=root)
    analysis_path = _inside_file(analysis_lock_path, root=root)
    g5_path = _inside_file(g5_unlock_path, root=root)
    g4_path = _inside_file(g4_freeze_path, root=root)
    role_path = _inside_file(role_manifest_path, root=root)
    asset_path = _inside_file(asset_lock_path, root=root)
    environment_path = _inside_file(environment_lock_path, root=root)
    sealed_path = _inside_file(sealed_reference_lock_path, root=root)
    trace_path = _inside_file(trace_locks_path, root=root)
    binding_path = _inside_file(cell_bindings_path, root=root)
    g6_path = (
        None if g6_unlock_path is None else _inside_file(g6_unlock_path, root=root)
    )
    validate_g5_unlock(
        g5_path,
        matrix_spec_path=matrix_path,
        expected_matrix_path=expected_path,
        analysis_lock_path=analysis_path,
        role_manifest_path=role_path,
        asset_lock_path=asset_path,
        g4_freeze_path=g4_path,
        paper2_root=root,
    )
    if require_g6:
        if g6_path is None:
            raise ValueError("delayed formal experiments require the completed G6 unlock")
        validate_g6_unlock(
            g6_path,
            g5_unlock_path=g5_path,
            expected_matrix_path=expected_path,
            analysis_lock_path=analysis_path,
            paper2_root=root,
        )

    expected = ExpectedMatrix.read(expected_path)
    if expected.analysis_lock_sha256 != file_sha256(analysis_path):
        raise ValueError("formal execution expected matrix differs from the analysis lock")
    binding_payload = read_json(binding_path)
    if not isinstance(binding_payload, Mapping):
        raise ValueError("formal cell bindings must be a JSON object")
    if (
        binding_payload.get("schema") != CELL_BINDING_SCHEMA
        or binding_payload.get("status") != "complete"
        or binding_payload.get("content_sha256")
        != _semantic_hash(binding_payload, "content_sha256")
        or binding_payload.get("asset_lock_sha256") != file_sha256(asset_path)
        or binding_payload.get("environment_lock_sha256")
        != file_sha256(environment_path)
        or binding_payload.get("sealed_reference_lock_sha256")
        != file_sha256(sealed_path)
        or binding_payload.get("g4_freeze_sha256") != file_sha256(g4_path)
    ):
        raise ValueError("formal cell bindings identity/status/parent hashes differ")
    raw_bindings = binding_payload.get("cells")
    if not isinstance(raw_bindings, Mapping):
        raise ValueError("formal cell bindings lack a keyed cell map")
    expected_by_id = {cell.key.identifier: cell for cell in expected.cells}
    if set(raw_bindings) != set(expected_by_id):
        raise ValueError("formal cell bindings do not exactly cover the expected matrix")

    trace_payload = read_json(trace_path)
    if (
        not isinstance(trace_payload, Mapping)
        or trace_payload.get("schema") != "paper2.expected-trace-locks/v1"
        or trace_payload.get("cell_input_bindings_sha256") != file_sha256(binding_path)
        or trace_payload.get("role_manifest_sha256") != file_sha256(role_path)
    ):
        raise ValueError("formal trace locks do not bind the selected cell-input map")
    analysis = read_json(analysis_path)
    if analysis.get("expected_trace_locks_sha256") != file_sha256(trace_path):
        raise ValueError("analysis lock does not bind the selected trace-lock file")
    trace_cells = trace_payload.get("trace_locks")
    if not isinstance(trace_cells, Mapping) or set(trace_cells) != set(expected_by_id):
        raise ValueError("formal trace-lock cells differ from the expected matrix")
    locked_code_hashes: set[str] = set()
    for identifier, cell in expected_by_id.items():
        if dict(cell.locked_trace_hashes) != dict(trace_cells[identifier]):
            raise ValueError(f"expected matrix trace lock differs for {identifier}")
        code_hash = dict(cell.locked_trace_hashes).get("code_sha256")
        if code_hash is None:
            raise ValueError(f"formal trace lock lacks code_sha256 for {identifier}")
        locked_code_hashes.add(code_hash)
    if len(locked_code_hashes) != 1:
        raise ValueError("formal cells do not share one frozen source-tree hash")
    source_code_sha256 = _source_tree_sha256(root / "src")
    if locked_code_hashes != {source_code_sha256}:
        raise ValueError("formal worker source tree differs from the G5 trace lock")

    sealed = read_sealed_reference_lock(sealed_path, paper2_root=root, verify_files=True)
    references = {str(entry["tile_id"]): dict(entry) for entry in sealed["entries"]}
    for cell in expected.cells:
        reference = references.get(cell.key.tile_id)
        if reference is None:
            raise ValueError(f"sealed reference lock lacks {cell.key.tile_id}")
        if (
            cell.sealed_reference_sha256 != reference["reference_sha256"]
            or cell.sealed_reference_grid_sha256 != reference["grid_sha256"]
        ):
            raise ValueError(f"expected reference lock differs for {cell.key.tile_id}")
    return FormalExecutionContract(
        root=root,
        expected_path=expected_path,
        matrix_spec_path=matrix_path,
        analysis_lock_path=analysis_path,
        g5_unlock_path=g5_path,
        g4_freeze_path=g4_path,
        role_manifest_path=role_path,
        asset_lock_path=asset_path,
        environment_lock_path=environment_path,
        sealed_reference_lock_path=sealed_path,
        trace_locks_path=trace_path,
        cell_bindings_path=binding_path,
        g6_unlock_path=g6_path,
        expected=expected,
        expected_by_id=expected_by_id,
        bindings={str(name): dict(value) for name, value in raw_bindings.items()},
        references=references,
        source_code_sha256=source_code_sha256,
    )


@dataclass(frozen=True)
class ExecutionUnit:
    unit_id: str
    experiment: str
    cell_ids: tuple[str, ...]
    model_identity: str
    training_seed: int
    estimated_cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "experiment": self.experiment,
            "cell_ids": list(self.cell_ids),
            "model_identity": self.model_identity,
            "training_seed": self.training_seed,
            "estimated_cost": self.estimated_cost,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionUnit":
        result = cls(
            unit_id=str(value["unit_id"]),
            experiment=str(value["experiment"]),
            cell_ids=tuple(str(item) for item in value["cell_ids"]),
            model_identity=str(value["model_identity"]),
            training_seed=int(value["training_seed"]),
            estimated_cost=float(value["estimated_cost"]),
        )
        if (
            not result.unit_id
            or result.experiment not in EXPERIMENTS
            or not result.cell_ids
            or not math.isfinite(result.estimated_cost)
            or result.estimated_cost <= 0
        ):
            raise ValueError("formal execution unit is malformed")
        return result


def _unit_cost(cells: Sequence[ExpectedCell]) -> float:
    cost = 0.0
    for cell in cells:
        key = cell.key
        spec = method_spec(key.method)
        if spec.generative:
            cost += float(key.path_count * key.step_count)
        elif spec.learned:
            cost += 1.0
        else:
            cost += 0.05
    return max(cost, 0.05)


def estimate_formal_storage_bytes(
    *,
    expected_by_id: Mapping[str, ExpectedCell],
    references: Mapping[str, Mapping[str, Any]],
    units: Sequence[ExecutionUnit],
    per_cell_metadata_bytes: int = DEFAULT_PER_CELL_METADATA_BYTES,
) -> dict[str, Any]:
    """Estimate persistent bytes without opening any sealed reference values.

    Formal arrays use compressed NPZ files, but this deliberately grants no
    compression credit.  It counts the two float32 prediction grids, float64
    held-out footprint path projections, per-cell JSON/NPZ overhead, each E6
    shared D0 path stack, and each E7 M=32 prefix-source path stack.
    """

    if per_cell_metadata_bytes < 0:
        raise ValueError("formal per-cell metadata estimate cannot be negative")
    byte_fields = (
        "prediction_bytes",
        "audit_projection_bytes",
        "shared_path_stack_bytes",
        "metadata_bytes",
    )
    totals = {name: 0 for name in byte_fields}
    by_experiment: dict[str, dict[str, int]] = defaultdict(
        lambda: {"units": 0, "cells": 0, **{name: 0 for name in byte_fields}}
    )
    seen_cells: set[str] = set()
    for unit in units:
        experiment = by_experiment[unit.experiment]
        experiment["units"] += 1
        unit_cells: list[ExpectedCell] = []
        for cell_id in unit.cell_ids:
            if cell_id in seen_cells:
                raise ValueError("formal capacity units contain a duplicate cell")
            seen_cells.add(cell_id)
            try:
                cell = expected_by_id[cell_id]
                raw_shape = references[cell.key.tile_id]["shape"]
            except KeyError as exc:
                raise ValueError(
                    f"formal capacity estimate lacks cell/reference {cell_id}"
                ) from exc
            if (
                not isinstance(raw_shape, Sequence)
                or isinstance(raw_shape, (str, bytes))
                or len(raw_shape) != 2
                or any(int(value) <= 0 for value in raw_shape)
            ):
                raise ValueError("formal capacity reference shape is invalid")
            pixels = int(raw_shape[0]) * int(raw_shape[1])
            contributions = {
                "prediction_bytes": 2 * 4 * pixels,
                # The projection NPZ contains M paths plus mean and variance.
                "audit_projection_bytes": (int(cell.key.path_count) + 2) * 128 * 8,
                "metadata_bytes": int(per_cell_metadata_bytes),
            }
            experiment["cells"] += 1
            for name, value in contributions.items():
                experiment[name] += value
                totals[name] += value
            unit_cells.append(cell)

        stack_bytes = 0
        if unit.experiment == "E6":
            source = unit_cells[0]
            shape = references[source.key.tile_id]["shape"]
            pixels = int(shape[0]) * int(shape[1])
            stack_bytes = 4 * int(source.key.path_count) * pixels
        elif unit.experiment == "E7":
            source = max(unit_cells, key=lambda value: value.key.path_count)
            if int(source.key.path_count) != 32:
                raise ValueError("formal E7 capacity unit lacks its M=32 source")
            shape = references[source.key.tile_id]["shape"]
            pixels = int(shape[0]) * int(shape[1])
            stack_bytes = 4 * 32 * pixels
        experiment["shared_path_stack_bytes"] += stack_bytes
        totals["shared_path_stack_bytes"] += stack_bytes

    raw_bytes = sum(totals.values())
    per_experiment = {}
    for name, values in sorted(by_experiment.items()):
        per_experiment[name] = {
            **values,
            "raw_estimated_bytes": sum(values[field] for field in byte_fields),
        }
    return {
        "units": len(units),
        "cells": len(seen_cells),
        **totals,
        "raw_estimated_bytes": raw_bytes,
        "by_experiment": per_experiment,
    }


def run_capacity_preflight(
    contract: FormalExecutionContract,
    *,
    output_root: str | Path,
    report_path: str | Path,
    units: Sequence[ExecutionUnit],
    safety_factor: float = DEFAULT_CAPACITY_SAFETY_FACTOR,
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
) -> Mapping[str, Any]:
    """Write an operational capacity receipt and fail before workers launch."""

    if not math.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("formal capacity safety factor must be finite and >= 1")
    if minimum_free_bytes < 0:
        raise ValueError("formal minimum free bytes cannot be negative")
    target = _inside_directory(output_root, root=contract.root)
    target.mkdir(parents=True, exist_ok=True)
    estimate = estimate_formal_storage_bytes(
        expected_by_id=contract.expected_by_id,
        references=contract.references,
        units=units,
    )
    usage = shutil.disk_usage(target)
    raw_bytes = int(estimate["raw_estimated_bytes"])
    protected_estimate = int(math.ceil(raw_bytes * safety_factor))
    required_available = protected_estimate + int(minimum_free_bytes)
    sufficient = not units or int(usage.free) >= required_available
    payload = {
        "schema": CAPACITY_PREFLIGHT_SCHEMA,
        "status": "pass" if sufficient else "insufficient",
        "estimation_basis": (
            "uncompressed_float_arrays_plus_cell_metadata_then_safety_factor"
        ),
        "compression_credit_bytes": 0,
        "safety_factor": float(safety_factor),
        "minimum_free_bytes_after_stage": int(minimum_free_bytes),
        "filesystem_total_bytes": int(usage.total),
        "filesystem_used_bytes": int(usage.used),
        "filesystem_available_bytes": int(usage.free),
        "protected_estimated_bytes": protected_estimate,
        "required_available_bytes": required_available,
        "available_margin_bytes": int(usage.free) - required_available,
        "expected_matrix_sha256": file_sha256(contract.expected_path),
        "cell_bindings_sha256": file_sha256(contract.cell_bindings_path),
        "pending": estimate,
    }
    write_mutable_progress(report_path, payload, paper2_root=contract.root)
    if not sufficient:
        raise RuntimeError(
            "formal execution disk capacity is insufficient: "
            f"available={usage.free / GIB:.1f} GiB, "
            f"estimated={protected_estimate / GIB:.1f} GiB, "
            f"reserved={minimum_free_bytes / GIB:.1f} GiB"
        )
    return payload


def build_execution_units(
    expected: ExpectedMatrix, experiments: Sequence[str]
) -> tuple[ExecutionUnit, ...]:
    selected = tuple(dict.fromkeys(str(value) for value in experiments))
    if not selected or any(value not in EXPERIMENTS for value in selected):
        raise ValueError(f"formal experiments must be selected from {EXPERIMENTS}")
    cells = [cell for cell in expected.cells if cell.key.experiment in selected]
    if not cells:
        raise ValueError("selected formal experiment set has no expected cells")
    grouped: dict[tuple[Any, ...], list[ExpectedCell]] = defaultdict(list)
    singles: list[list[ExpectedCell]] = []
    for cell in cells:
        key = cell.key
        if key.experiment == "E6":
            group = (
                "E6",
                key.tile_id,
                key.method,
                key.budget,
                key.noise_replicate,
                key.training_seed,
                key.model_identity,
                key.path_seed_set,
                key.path_count,
                key.step_count,
            )
            grouped[group].append(cell)
        elif key.experiment == "E7":
            group = (
                "E7",
                key.tile_id,
                key.method,
                key.budget,
                key.noise_replicate,
                key.training_seed,
                key.model_identity,
                key.path_seed_set,
                key.step_count,
                key.dc_mode,
            )
            grouped[group].append(cell)
        else:
            singles.append([cell])
    bundles = singles + list(grouped.values())
    units: list[ExecutionUnit] = []
    for bundle in bundles:
        experiment = bundle[0].key.experiment
        if experiment == "E6":
            by_mode = {cell.key.dc_mode: cell for cell in bundle}
            if set(by_mode) != {"D0", "D1", "D3"}:
                raise ValueError("each E6 execution unit must contain D0/D1/D3")
            ordered = [by_mode[name] for name in ("D0", "D1", "D3")]
            cost = _unit_cost(ordered[:1]) + 0.2 * _unit_cost(ordered[1:])
        elif experiment == "E7":
            by_count = {cell.key.path_count: cell for cell in bundle}
            if set(by_count) != set(E7_PATH_COUNTS):
                raise ValueError("each E7 execution unit must contain the complete M prefix grid")
            ordered = [by_count[32], *(by_count[value] for value in (2, 4, 8, 16))]
            cost = _unit_cost(ordered[:1]) + 0.01 * _unit_cost(ordered[1:])
        else:
            ordered = bundle
            cost = _unit_cost(ordered)
        first = ordered[0].key
        unit_id = canonical_sha256(
            {
                "experiment": experiment,
                "cells": [cell.key.identifier for cell in ordered],
            }
        )
        units.append(
            ExecutionUnit(
                unit_id=unit_id,
                experiment=experiment,
                cell_ids=tuple(cell.key.identifier for cell in ordered),
                model_identity=canonical_model_identity(first.method),
                training_seed=first.training_seed,
                estimated_cost=cost,
            )
        )
    return tuple(
        sorted(
            units,
            key=lambda value: (
                value.model_identity,
                value.training_seed,
                value.experiment,
                value.unit_id,
            ),
        )
    )


def partition_execution_units(
    units: Sequence[ExecutionUnit], worker_count: int
) -> tuple[tuple[ExecutionUnit, ...], ...]:
    if worker_count <= 0:
        raise ValueError("formal execution worker count must be positive")
    groups: dict[tuple[str, int], list[ExecutionUnit]] = defaultdict(list)
    for unit in units:
        groups[(unit.model_identity, unit.training_seed)].append(unit)
    partitions: list[list[ExecutionUnit]] = [[] for _ in range(worker_count)]
    loads = [0.0] * worker_count
    ordered_groups = sorted(
        groups.values(),
        key=lambda values: (-sum(item.estimated_cost for item in values), values[0].unit_id),
    )
    for values in ordered_groups:
        index = min(range(worker_count), key=lambda item: (loads[item], item))
        partitions[index].extend(values)
        loads[index] += sum(item.estimated_cost for item in values)
    for values in partitions:
        values.sort(
            key=lambda item: (
                item.model_identity,
                item.training_seed,
                item.experiment,
                item.unit_id,
            )
        )
    return tuple(tuple(values) for values in partitions)


def cell_output_directory(
    contract: FormalExecutionContract, output_root: str | Path, cell_id: str
) -> Path:
    root = _inside_directory(output_root, root=contract.root)
    cell = contract.expected_by_id[cell_id]
    return root / cell.key.experiment / cell_id[:2] / cell_id


def _artifact_files_ok(manifest: InferenceManifest | EvaluationManifest, *, root: Path) -> bool:
    for artifact in manifest.artifacts:
        try:
            path = _inside_file(root / artifact.relative_path, root=root)
        except (FileNotFoundError, ValueError):
            return False
        if file_sha256(path) != artifact.sha256:
            return False
    return True


def cell_output_is_complete(
    contract: FormalExecutionContract, output_root: str | Path, cell_id: str
) -> bool:
    cell = contract.expected_by_id[cell_id]
    directory = cell_output_directory(contract, output_root, cell_id)
    inference_path = directory / "inference-manifest.json"
    evaluation_path = directory / "evaluation-manifest.json"
    if not inference_path.is_file() or not evaluation_path.is_file():
        return False
    try:
        inference = InferenceManifest.read(inference_path)
        evaluation = EvaluationManifest.read(evaluation_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    trace = dict(inference.trace_hashes)
    if (
        inference.status != "complete"
        or inference.key != cell.key
        or any(name not in trace for name in cell.required_trace_hashes)
        or any(trace.get(name) != digest for name, digest in cell.locked_trace_hashes)
        or trace.get("analysis_lock_sha256") != file_sha256(contract.analysis_lock_path)
        or not _artifact_files_ok(inference, root=contract.root)
    ):
        return False
    if (
        evaluation.status != "complete"
        or evaluation.key != cell.key
        or evaluation.inference_manifest_sha256 != inference.sha256
        or evaluation.sealed_reference_sha256 != cell.sealed_reference_sha256
        or evaluation.sealed_reference_grid_sha256 != cell.sealed_reference_grid_sha256
        or not _artifact_files_ok(evaluation, root=contract.root)
    ):
        return False
    metrics_artifacts = [item for item in evaluation.artifacts if item.name == "metrics"]
    if len(metrics_artifacts) != 1:
        return False
    metrics = read_json(contract.root / metrics_artifacts[0].relative_path)
    audit = metrics.get("metrics", {}).get("audit_footprints")
    return bool(
        isinstance(audit, Mapping)
        and audit.get("count") == 128
        and audit.get("diagonal_pixel_variance_projection") is False
    )


def unit_is_complete(
    contract: FormalExecutionContract, output_root: str | Path, unit: ExecutionUnit
) -> bool:
    complete = all(
        cell_output_is_complete(contract, output_root, cell_id)
        for cell_id in unit.cell_ids
    )
    if not complete:
        return False
    try:
        _validate_unit_outputs(contract, output_root=output_root, unit=unit)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError):
        return False
    return True


def _quarantine(path: Path, *, contract: FormalExecutionContract, label: str) -> None:
    if not path.exists():
        return
    quarantine = contract.root / "runs" / "quarantine" / "formal-execution"
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / f"{label}-{time.time_ns()}-{path.name}"
    os.replace(path, destination)


def _invoke_cli(
    arguments: list[str], *, formal_contract: FormalExecutionContract | None = None
) -> None:
    from .cli import main, register_formal_execution_context

    if formal_contract is not None:
        register_formal_execution_context(
            paper2_root=formal_contract.root,
            expected_matrix_path=formal_contract.expected_path,
            matrix_spec_path=formal_contract.matrix_spec_path,
            analysis_lock_path=formal_contract.analysis_lock_path,
            g5_unlock_path=formal_contract.g5_unlock_path,
            g4_freeze_path=formal_contract.g4_freeze_path,
            role_manifest_path=formal_contract.role_manifest_path,
            asset_lock_path=formal_contract.asset_lock_path,
            environment_lock_path=formal_contract.environment_lock_path,
            sealed_reference_lock_path=formal_contract.sealed_reference_lock_path,
            expected_by_id=formal_contract.expected_by_id,
            sealed_references=formal_contract.references,
            source_code_sha256=formal_contract.source_code_sha256,
        )

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        returncode = main(arguments)
    if returncode != 0:
        diagnostic = captured.getvalue()[-8_000:]
        raise RuntimeError(
            f"Paper2 CLI command failed: {' '.join(arguments[:6])}\n{diagnostic}"
        )


def _inference_arguments(
    contract: FormalExecutionContract,
    *,
    cell_id: str,
    output: Path,
    device: str,
    e6_uncorrected_path: Path | None,
) -> list[str]:
    cell = contract.expected_by_id[cell_id]
    key = cell.key
    binding = contract.bindings[cell_id]
    settings = binding.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError(f"formal cell {cell_id} lacks frozen inference settings")
    arguments = [
        "--paper2-root",
        str(contract.root),
        "infer",
        "--problem",
        str(contract.binding_path(cell_id, "problem")),
        "--plan",
        str(contract.binding_path(cell_id, "plan")),
        "--zf",
        str(contract.binding_path(cell_id, "zf")),
        "--zf-key",
        "zf",
        "--omega",
        str(contract.binding_path(cell_id, "omega")),
        "--omega-key",
        "omega",
        "--normalization",
        str(contract.binding_path(cell_id, "normalization")),
        "--prepared-manifest",
        str(contract.binding_path(cell_id, "prepared_manifest")),
        "--audit-footprint-plan",
        str(contract.binding_path(cell_id, "audit_footprint_plan")),
        "--asset-lock",
        str(contract.asset_lock_path),
        "--analysis-lock",
        str(contract.analysis_lock_path),
        "--environment-lock",
        str(contract.environment_lock_path),
        "--g5-unlock",
        str(contract.g5_unlock_path),
        "--g4-freeze",
        str(contract.g4_freeze_path),
        "--role-manifest",
        str(contract.role_manifest_path),
        "--expected-matrix",
        str(contract.expected_path),
        "--matrix-spec",
        str(contract.matrix_spec_path),
        "--method",
        key.method,
        "--experiment",
        key.experiment,
        "--condition",
        key.condition,
        "--noise-replicate",
        str(key.noise_replicate),
        "--training-seed",
        str(key.training_seed),
        "--path-seed-set",
        key.path_seed_set,
        "--paths",
        str(key.path_count),
        "--steps",
        str(key.step_count),
        "--g-min",
        str(settings["g_min"]),
        "--alpha",
        str(settings["alpha"]),
        "--ddpm-cosine-offset",
        str(settings["ddpm_cosine_offset"]),
        "--ddpm-eta",
        str(settings["ddpm_eta"]),
        "--dc-mode",
        key.dc_mode,
        "--tau-dc",
        str(settings["tau_dc"]),
        "--dc-delta",
        str(settings["dc_delta"]),
        "--lambda-dc",
        str(settings["lambda_dc"]),
        "--device",
        device,
        "--formal",
        "--output",
        str(output),
    ]
    optional = {
        "ctx_input_lineage_manifest": "--ctx-input-lineage-manifest",
        "paper1_import_manifest": "--paper1-import-manifest",
        "e10_mola_problem_manifest": "--e10-mola-problem-manifest",
        "e8_stress_manifest": "--e8-stress-manifest",
        "selector_manifest": "--selector-manifest",
        "checkpoint": "--checkpoint",
        "checkpoint_manifest": "--checkpoint-manifest",
        "calibration_manifest": "--calibration-manifest",
        "footprint_calibration_manifest": "--footprint-calibration-manifest",
    }
    for name, flag in optional.items():
        if name in binding:
            arguments.extend((flag, str(contract.binding_path(cell_id, name))))
    if key.experiment in G6_DELAYED_EXPERIMENTS:
        if contract.g6_unlock_path is None:
            raise ValueError(f"formal {key.experiment} cannot run before G6")
        arguments.extend(("--g6-unlock", str(contract.g6_unlock_path)))
    if e6_uncorrected_path is not None:
        arguments.extend(("--e6-uncorrected-path-stack", str(e6_uncorrected_path)))
    return arguments


def _audit_observation_path(
    contract: FormalExecutionContract, *, cell_id: str
) -> Path:
    cell = contract.expected_by_id[cell_id]
    plan_path = contract.binding_path(cell_id, "plan")
    plan = _cached_plan_contract(
        str(plan_path.resolve()), _file_signature(plan_path)
    )
    audit_plan_path = contract.binding_path(cell_id, "audit_footprint_plan")
    reference = contract.references[cell.key.tile_id]
    digest = canonical_sha256(
        {
            "audit_plan_sha256": file_sha256(audit_plan_path),
            "sealed_reference_sha256": reference["reference_sha256"],
            "replicate": cell.key.noise_replicate,
            "noise_std_m_hex": float(plan.noise_std_m).hex(),
        }
    )
    return (
        contract.root
        / "data/evaluator/audit-observations"
        / digest[:2]
        / f"{digest}.npz"
    )


def _audit_observations_match(
    output: Path,
    *,
    tile_id: str,
    replicate: int,
    audit_plan_file_sha256: str,
    audit_plan_contract_sha256: str,
    noise_std_m: float,
) -> bool:
    if not output.is_file():
        return False
    try:
        observed = _cached_audit_footprint_observations(
            str(output.resolve()), _file_signature(output)
        )
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    return bool(
        observed.tile_id == tile_id
        and observed.replicate == replicate
        and observed.audit_plan_file_sha256 == audit_plan_file_sha256
        and observed.audit_plan_contract_sha256 == audit_plan_contract_sha256
        and observed.noise_std_m == noise_std_m
    )


def _materialize_audit_observations(
    contract: FormalExecutionContract, *, cell_id: str
) -> Path:
    cell = contract.expected_by_id[cell_id]
    plan_path = contract.binding_path(cell_id, "plan")
    plan = _cached_plan_contract(
        str(plan_path.resolve()), _file_signature(plan_path)
    )
    audit_plan_path = contract.binding_path(cell_id, "audit_footprint_plan")
    audit_plan = _cached_audit_footprint_plan(
        str(audit_plan_path.resolve()), _file_signature(audit_plan_path)
    )
    audit_plan_file_sha256 = file_sha256(audit_plan_path)
    audit_plan_contract_sha256 = audit_plan.sha256
    reference = contract.references[cell.key.tile_id]
    reference_path = _inside_file(
        contract.root / str(reference["reference_relative_path"]), root=contract.root
    )
    output = _audit_observation_path(contract, cell_id=cell_id)
    lock_path = (
        contract.root
        / "data/evaluator/audit-observations/locks"
        / f"{output.stem}.lock"
    )
    match_arguments = {
        "tile_id": cell.key.tile_id,
        "replicate": cell.key.noise_replicate,
        "audit_plan_file_sha256": audit_plan_file_sha256,
        "audit_plan_contract_sha256": audit_plan_contract_sha256,
        "noise_std_m": plan.noise_std_m,
    }
    if _audit_observations_match(output, **match_arguments):
        return output
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if _audit_observations_match(output, **match_arguments):
            return output
        if output.is_file():
            _quarantine(output, contract=contract, label="audit-observations")
        observations = simulate_audit_footprint_observations(
            audit_plan_path=audit_plan_path,
            reference_path=reference_path,
            replicate=cell.key.noise_replicate,
            noise_std_m=plan.noise_std_m,
        )
        return observations.write(output, paper2_root=contract.root)


def _evaluate_cell(
    contract: FormalExecutionContract, *, cell_id: str, directory: Path
) -> None:
    cell = contract.expected_by_id[cell_id]
    binding = contract.bindings[cell_id]
    plan_path = contract.binding_path(cell_id, "plan")
    plan = _cached_plan_contract(
        str(plan_path.resolve()), _file_signature(plan_path)
    )
    determinant = plan.transform[0] * plan.transform[4] - plan.transform[1] * plan.transform[3]
    pixel_size_m = math.sqrt(abs(float(determinant)))
    if not math.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("formal evaluation pixel size is invalid")
    reference = contract.references[cell.key.tile_id]
    arguments = [
        "--paper2-root",
        str(contract.root),
        "evaluate",
        "--prediction",
        str(directory / "prediction.npz"),
        "--inference-manifest",
        str(directory / "inference-manifest.json"),
        "--problem",
        str(contract.binding_path(cell_id, "problem")),
        "--plan",
        str(plan_path),
        "--audit-footprint-plan",
        str(contract.binding_path(cell_id, "audit_footprint_plan")),
        "--audit-footprint-observations",
        str(_materialize_audit_observations(contract, cell_id=cell_id)),
        "--sealed-reference",
        str(
            _inside_file(
                contract.root / str(reference["reference_relative_path"]),
                root=contract.root,
            )
        ),
        "--reference-lock",
        str(contract.sealed_reference_lock_path),
        "--formal",
        "--pixel-size-m",
        repr(pixel_size_m),
        "--output",
        str(directory / "metrics.json"),
    ]
    if "footprint_calibration_manifest" in binding:
        arguments.extend(
            (
                "--footprint-calibration-manifest",
                str(contract.binding_path(cell_id, "footprint_calibration_manifest")),
            )
        )
    _invoke_cli(arguments)


def _execute_direct_cell(
    contract: FormalExecutionContract,
    *,
    output_root: str | Path,
    cell_id: str,
    device: str,
    e6_uncorrected_path: Path | None = None,
) -> str:
    if cell_output_is_complete(contract, output_root, cell_id):
        return "skipped"
    directory = cell_output_directory(contract, output_root, cell_id)
    if directory.exists():
        _quarantine(directory, contract=contract, label=cell_id)
    directory.mkdir(parents=True, exist_ok=True)
    _invoke_cli(
        _inference_arguments(
            contract,
            cell_id=cell_id,
            output=directory / "prediction.npz",
            device=device,
            e6_uncorrected_path=e6_uncorrected_path,
        ),
        formal_contract=contract,
    )
    _evaluate_cell(contract, cell_id=cell_id, directory=directory)
    if not cell_output_is_complete(contract, output_root, cell_id):
        raise RuntimeError(f"formal cell did not close after execution: {cell_id}")
    return "executed"


def _validate_unit_outputs(
    contract: FormalExecutionContract, *, output_root: str | Path, unit: ExecutionUnit
) -> None:
    cells = tuple(contract.expected_by_id[cell_id] for cell_id in unit.cell_ids)
    inferences = []
    evaluations = []
    for cell_id in unit.cell_ids:
        directory = cell_output_directory(contract, output_root, cell_id)
        inferences.append(InferenceManifest.read(directory / "inference-manifest.json"))
        evaluations.append(EvaluationManifest.read(directory / "evaluation-manifest.json"))
    subset = ExpectedMatrix(
        analysis_lock_sha256=contract.expected.analysis_lock_sha256,
        cells=cells,
    )
    audit = audit_expected_matrix(
        subset, inferences, evaluations, paper2_root=contract.root
    )
    if not audit.ok:
        raise RuntimeError(
            f"formal unit audit failed {unit.unit_id}: "
            f"hash={len(audit.hash_errors)}, artifacts={len(audit.artifact_errors)}, "
            f"evaluations={len(audit.evaluation_errors)}"
        )


def _execute_e6_unit(
    contract: FormalExecutionContract,
    *,
    output_root: str | Path,
    unit: ExecutionUnit,
    device: str,
) -> tuple[int, int]:
    cells = [contract.expected_by_id[cell_id] for cell_id in unit.cell_ids]
    by_mode = {cell.key.dc_mode: cell.key.identifier for cell in cells}
    shared_directory = contract.root / "runs/formal-shared/E6" / unit.unit_id[:2] / unit.unit_id
    shared_path = shared_directory / "e6-uncorrected-path-stack.npz"
    d0_id = by_mode["D0"]
    if not cell_output_is_complete(contract, output_root, d0_id):
        for cell_id in unit.cell_ids:
            directory = cell_output_directory(contract, output_root, cell_id)
            if directory.exists():
                _quarantine(directory, contract=contract, label=cell_id)
        if shared_directory.exists():
            _quarantine(shared_directory, contract=contract, label=f"E6-{unit.unit_id}")
    shared_directory.mkdir(parents=True, exist_ok=True)
    executed = 0
    skipped = 0
    for mode in ("D0", "D1", "D3"):
        result = _execute_direct_cell(
            contract,
            output_root=output_root,
            cell_id=by_mode[mode],
            device=device,
            e6_uncorrected_path=shared_path,
        )
        executed += int(result == "executed")
        skipped += int(result == "skipped")
    _validate_unit_outputs(contract, output_root=output_root, unit=unit)
    return executed, skipped


def _execute_e7_unit(
    contract: FormalExecutionContract,
    *,
    output_root: str | Path,
    unit: ExecutionUnit,
    device: str,
) -> tuple[int, int]:
    cells = [contract.expected_by_id[cell_id] for cell_id in unit.cell_ids]
    by_count = {cell.key.path_count: cell.key.identifier for cell in cells}
    source_id = by_count[32]
    if not cell_output_is_complete(contract, output_root, source_id):
        for cell_id in unit.cell_ids:
            directory = cell_output_directory(contract, output_root, cell_id)
            if directory.exists():
                _quarantine(directory, contract=contract, label=cell_id)
    result = _execute_direct_cell(
        contract,
        output_root=output_root,
        cell_id=source_id,
        device=device,
    )
    executed = int(result == "executed")
    skipped = int(result == "skipped")
    source_directory = cell_output_directory(contract, output_root, source_id)
    missing_outputs: dict[int, Path] = {}
    for count in (2, 4, 8, 16):
        cell_id = by_count[count]
        if cell_output_is_complete(contract, output_root, cell_id):
            skipped += 1
            continue
        directory = cell_output_directory(contract, output_root, cell_id)
        if directory.exists():
            _quarantine(directory, contract=contract, label=cell_id)
        directory.mkdir(parents=True, exist_ok=True)
        missing_outputs[count] = directory / "prediction.npz"
    if missing_outputs:
        derive_e7_prefixes(
            source_inference_manifest_path=source_directory
            / "inference-manifest.json",
            outputs=missing_outputs,
            paper2_root=contract.root,
            expected_by_id=contract.expected_by_id,
            expected_analysis_lock_sha256=contract.expected.analysis_lock_sha256,
            source_code_sha256=contract.source_code_sha256,
            formal=True,
        )
    for count in missing_outputs:
        cell_id = by_count[count]
        directory = cell_output_directory(contract, output_root, cell_id)
        _evaluate_cell(contract, cell_id=cell_id, directory=directory)
        if not cell_output_is_complete(contract, output_root, cell_id):
            raise RuntimeError(f"derived E7 cell did not close: {cell_id}")
        executed += 1
    _validate_unit_outputs(contract, output_root=output_root, unit=unit)
    return executed, skipped


def execute_unit(
    contract: FormalExecutionContract,
    *,
    output_root: str | Path,
    unit: ExecutionUnit,
    device: str,
) -> dict[str, Any]:
    if any(cell_id not in contract.expected_by_id for cell_id in unit.cell_ids):
        raise ValueError("execution unit contains a cell outside the expected matrix")
    if unit.experiment == "E6":
        executed, skipped = _execute_e6_unit(
            contract, output_root=output_root, unit=unit, device=device
        )
    elif unit.experiment == "E7":
        executed, skipped = _execute_e7_unit(
            contract, output_root=output_root, unit=unit, device=device
        )
    else:
        if len(unit.cell_ids) != 1:
            raise ValueError("ordinary formal execution units must contain one cell")
        result = _execute_direct_cell(
            contract,
            output_root=output_root,
            cell_id=unit.cell_ids[0],
            device=device,
        )
        executed = int(result == "executed")
        skipped = int(result == "skipped")
        _validate_unit_outputs(contract, output_root=output_root, unit=unit)
    return {
        "unit_id": unit.unit_id,
        "experiment": unit.experiment,
        "cells": len(unit.cell_ids),
        "executed": executed,
        "skipped": skipped,
    }


def write_mutable_progress(
    path: str | Path, payload: Mapping[str, Any], *, paper2_root: str | Path
) -> None:
    """Atomically update non-scientific worker progress state."""

    target = guard_paper2_path(path, paper2_root=paper2_root)
    # Use an atomic same-directory replacement because progress is intentionally
    # mutable and is not part of the scientific lock chain.
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
    with temporary.open("wb") as stream:
        stream.write(canonical_json_bytes(dict(payload)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
