"""Recoverable, hash-bound G2/G3 training gates.

The aggregate gate is intentionally separate from the synthetic GPU strategy
benchmark.  A report can set ``formal_gate=true`` only when its evidence was
constructed from hash-verified Paper2 training indices with the bridge_train and
bridge_val roles.  Stage execution is prefix-only: recovery may continue after
the last passed stage, but there is no API for starting at an arbitrary stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol, Sequence
import json
import math
import os
import tempfile

from .hashing import canonical_sha256, file_sha256, guard_paper2_path, is_sha256
from .models import UNetConfig
from .training import (
    PHASE_TIMING_KEYS,
    TrainingConfig,
    combine_phase_timings,
    distributed_stochastic_seed,
)


GATE_REPORT_SCHEMA = "paper2.training-gate-report/v1"
GATE_STAGE_SCHEMA = "paper2.training-gate-stage/v1"
GATE_STATE_SCHEMA = "paper2.training-gate-state/v1"
REAL_DATA_SCOPE = "prepared_training_patches"
ALLOWED_GATE_ROLES = frozenset({"bridge_train", "bridge_val"})
PRODUCTION_EXECUTION_SCOPE = "production_filestore_spawn"


def _require_sha256(name: str, value: str) -> None:
    if not is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_finite(name: str, value: Any) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _relative(path: Path, root: Path) -> str:
    resolved = guard_paper2_path(path, paper2_root=root)
    return resolved.relative_to(root).as_posix()


@dataclass(frozen=True)
class GateProtocol:
    feasibility_updates: tuple[int, int, int] = (200, 2_000, 10_000)
    stability_steps: int = 500
    overfit_sample_count: int = 32
    overfit_validation_ratio_max: float = 0.25
    max_allocated_fraction: float = 0.80
    max_reserved_fraction: float = 0.90
    max_consecutive_loss_scale_backoffs: int = 3
    g_min: float = 0.05
    alpha: float = 1.0
    endpoint_weight: float = 0.1
    gradient_weight: float = 0.1
    ddpm_cosine_offset: float = 0.008

    def validate(self) -> None:
        if self.feasibility_updates != (200, 2_000, 10_000):
            raise ValueError("G2/G3 feasibility updates are fixed at 200, 2000, and 10000")
        if self.stability_steps != 500:
            raise ValueError("G3 stability is fixed at 500 continuous updates")
        if self.overfit_sample_count != 32:
            raise ValueError("G2 overfit selection is fixed at 32 prepared patches")
        if not 0.0 < self.overfit_validation_ratio_max < 1.0:
            raise ValueError("overfit validation ratio threshold must lie in (0,1)")
        if not 0.0 < self.max_allocated_fraction < 1.0:
            raise ValueError("allocated-memory fraction must lie in (0,1)")
        if not 0.0 < self.max_reserved_fraction < 1.0:
            raise ValueError("reserved-memory fraction must lie in (0,1)")
        if self.max_allocated_fraction >= self.max_reserved_fraction:
            raise ValueError("allocated-memory threshold must be below reserved-memory threshold")
        if self.max_consecutive_loss_scale_backoffs < 0:
            raise ValueError("loss-scale backoff threshold cannot be negative")
        for name, value in (
            ("g_min", self.g_min),
            ("alpha", self.alpha),
            ("endpoint_weight", self.endpoint_weight),
            ("gradient_weight", self.gradient_weight),
            ("ddpm_cosine_offset", self.ddpm_cosine_offset),
        ):
            _require_finite(name, value)

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class GateStageSpec:
    stage_id: str
    gate: str
    method: str
    selection: str
    target_updates: int
    stop_update: int
    start_update: int
    resume_from: str | None
    require_overfit: bool = False
    require_stability: bool = False

    @property
    def continuous_updates(self) -> int:
        return self.stop_update - self.start_update

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


def gate_stages(protocol: GateProtocol) -> tuple[GateStageSpec, ...]:
    protocol.validate()
    smoke, overfit, feasibility = protocol.feasibility_updates
    return (
        GateStageSpec(
            "g2-f00-overfit-200",
            "G2",
            "F00",
            "overfit32",
            overfit,
            smoke,
            0,
            None,
        ),
        GateStageSpec(
            "g2-f00-overfit-2000",
            "G2",
            "F00",
            "overfit32",
            overfit,
            overfit,
            smoke,
            "g2-f00-overfit-200",
            require_overfit=True,
        ),
        GateStageSpec(
            "g2-f11-overfit-200",
            "G2",
            "F11",
            "overfit32",
            overfit,
            smoke,
            0,
            None,
        ),
        GateStageSpec(
            "g2-f11-overfit-2000",
            "G2",
            "F11",
            "overfit32",
            overfit,
            overfit,
            smoke,
            "g2-f11-overfit-200",
            require_overfit=True,
        ),
        GateStageSpec(
            "g3-full-stability-500",
            "G3",
            "OT-SB-full",
            "full_k100",
            feasibility,
            protocol.stability_steps,
            0,
            None,
            require_stability=True,
        ),
        GateStageSpec(
            "g3-full-feasibility-10000",
            "G3",
            "OT-SB-full",
            "full_k100",
            feasibility,
            feasibility,
            protocol.stability_steps,
            "g3-full-stability-500",
        ),
    )


def training_config_for_stage(
    stage: GateStageSpec,
    *,
    protocol: GateProtocol,
    seed: int,
) -> TrainingConfig:
    return TrainingConfig(
        method=stage.method,
        seed=seed,
        updates=stage.target_updates,
        validation_interval=200 if stage.selection == "overfit32" else 2_000,
        data_loader_workers=0,
        g_min=protocol.g_min,
        alpha=protocol.alpha,
        endpoint_weight=protocol.endpoint_weight,
        gradient_weight=protocol.gradient_weight,
        ddpm_schedule="cosine",
        ddpm_cosine_offset=protocol.ddpm_cosine_offset,
    )


@dataclass(frozen=True)
class GateEvidence:
    input_hashes: Mapping[str, str]
    code_sha256: str
    environment_sha256: str
    data_scope: str
    synthetic: bool
    consumed_roles: tuple[str, ...]
    overfit_sample_count: int
    overfit_selection_sha256: str
    overfit_split_hashes: Mapping[str, str]
    full_split_hashes: Mapping[str, str]
    maximum_config: Mapping[str, Any]
    verification_scope: str
    source_derivation_sha256: str
    runtime_inputs: Mapping[str, str]

    def validate(self, *, require_formal: bool = True) -> None:
        if not self.input_hashes:
            raise ValueError("gate evidence requires input hashes")
        for name, digest in self.input_hashes.items():
            _require_sha256(f"input hash {name}", str(digest))
        _require_sha256("code hash", self.code_sha256)
        _require_sha256("environment hash", self.environment_sha256)
        _require_sha256("overfit selection hash", self.overfit_selection_sha256)
        _require_sha256("source derivation hash", self.source_derivation_sha256)
        if self.verification_scope not in {"prepared_derivation_receipt", "test_fixture"}:
            raise ValueError("unsupported gate evidence verification scope")
        required_runtime_inputs = {
            "training_index",
            "validation_index",
            "environment_lock",
            "g1_report",
            "asset_lock",
            "role_manifest",
            "maximum_config_manifest",
            "training_derivation_receipt",
            "protocol_config",
            "g0_ready_draft",
            "processed_lock",
        }
        if set(self.runtime_inputs) != required_runtime_inputs:
            raise ValueError("gate runtime inputs differ from the path-safe request contract")
        for name, value in self.runtime_inputs.items():
            relative = PurePosixPath(str(value))
            if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
                raise ValueError(f"gate runtime input {name} must be a safe relative path")
        for label, values in (
            ("overfit split", self.overfit_split_hashes),
            ("full split", self.full_split_hashes),
        ):
            if not values:
                raise ValueError(f"{label} hashes cannot be empty")
            for name, digest in values.items():
                _require_sha256(f"{label} hash {name}", str(digest))
        if self.overfit_sample_count != 32:
            raise ValueError("formal G2 evidence must select exactly 32 prepared patches")
        if not self.consumed_roles:
            raise ValueError("gate evidence must identify consumed roles")
        forbidden = sorted(set(self.consumed_roles).difference(ALLOWED_GATE_ROLES))
        if forbidden:
            raise PermissionError(f"training gates refuse locked/test roles: {forbidden}")
        required_maximum = {
            "patch_size": 256,
            "image_count": 1,
            "laser_budget": 100,
            "actual_imagery": True,
            "full_training_validation_sampling_loop": True,
        }
        mismatches = [
            key for key, value in required_maximum.items() if self.maximum_config.get(key) != value
        ]
        if mismatches:
            raise ValueError(f"maximum-input contract differs from the formal configuration: {mismatches}")
        _require_sha256(
            "maximum-input manifest hash",
            str(self.maximum_config.get("manifest_file_sha256", "")),
        )
        if require_formal:
            if self.synthetic or self.data_scope != REAL_DATA_SCOPE:
                raise PermissionError(
                    "formal G2/G3 requires real prepared training patches; synthetic evidence is ineligible"
                )
            if self.verification_scope != "prepared_derivation_receipt":
                raise PermissionError("formal G2/G3 requires a preparation-time derivation receipt")

    @property
    def formal_eligible(self) -> bool:
        try:
            self.validate(require_formal=True)
        except (ValueError, PermissionError):
            return False
        return True

    @property
    def sha256(self) -> str:
        self.validate(require_formal=False)
        return canonical_sha256(self.to_dict())

    def split_hashes(self, stage: GateStageSpec) -> dict[str, str]:
        values = self.overfit_split_hashes if stage.selection == "overfit32" else self.full_split_hashes
        return dict(sorted((str(key), str(value)) for key, value in values.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_hashes": dict(sorted(self.input_hashes.items())),
            "code_sha256": self.code_sha256,
            "environment_sha256": self.environment_sha256,
            "data_scope": self.data_scope,
            "synthetic": self.synthetic,
            "consumed_roles": list(self.consumed_roles),
            "overfit_sample_count": self.overfit_sample_count,
            "overfit_selection_sha256": self.overfit_selection_sha256,
            "overfit_split_hashes": dict(sorted(self.overfit_split_hashes.items())),
            "full_split_hashes": dict(sorted(self.full_split_hashes.items())),
            "maximum_config": dict(self.maximum_config),
            "verification_scope": self.verification_scope,
            "source_derivation_sha256": self.source_derivation_sha256,
            "runtime_inputs": dict(sorted(self.runtime_inputs.items())),
        }


@dataclass(frozen=True)
class GateRunReceipt:
    checkpoint_manifest_path: Path
    request_path: Path | None
    execution_scope: str


class GateRunner(Protocol):
    def run(
        self,
        stage: GateStageSpec,
        *,
        attempt: int,
        resume_checkpoint: Path | None,
        resume_manifest: Path | None,
        output_directory: Path,
    ) -> GateRunReceipt:
        """Run one stage and return its checkpoint and launcher receipt."""


def _manifest_content_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != "manifest_sha256"})


def _validate_checkpoint_manifest(
    path: Path,
    *,
    root: Path,
    stage: GateStageSpec,
    protocol: GateProtocol,
    evidence: GateEvidence,
    seed: int,
    world_size: int,
    predecessor: Mapping[str, Any] | None,
    execution_scope: str,
    request_file_sha256: str | None,
) -> dict[str, Any]:
    manifest_path = guard_paper2_path(path, paper2_root=root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = _read_json(manifest_path)
    if payload.get("manifest_sha256") != _manifest_content_hash(payload):
        raise ValueError("checkpoint manifest content hash mismatch")
    config = training_config_for_stage(stage, protocol=protocol, seed=seed)
    config.validate(world_size=world_size)
    expected = {
        "method": stage.method,
        "model_identity": config.identity,
        "seed": seed,
        "training_config_sha256": canonical_sha256(asdict(config)),
        "model_config_sha256": canonical_sha256(UNetConfig().to_dict()),
        "split_hashes": evidence.split_hashes(stage),
        "code_sha256": evidence.code_sha256,
        "environment_sha256": evidence.environment_sha256,
        "completed_updates": stage.stop_update,
        "target_updates": stage.target_updates,
        "status": "complete" if stage.stop_update == stage.target_updates else "running",
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatches:
        raise ValueError(f"stage checkpoint contract mismatch: {mismatches}")
    if canonical_sha256(payload.get("training_config")) != canonical_sha256(asdict(config)):
        raise ValueError("checkpoint training config content differs from its locked hash")
    if canonical_sha256(payload.get("model_config")) != canonical_sha256(
        UNetConfig().to_dict()
    ):
        raise ValueError("checkpoint model config content differs from its locked hash")
    distributed = payload.get("distributed")
    expected_distributed = {
        "world_size": world_size,
        "stochastic_seed_protocol": "paper2-ddp-rank-stochastic-seed/v1",
        "rank_stochastic_seeds": [
            distributed_stochastic_seed(seed, rank) for rank in range(world_size)
        ],
    }
    if distributed != expected_distributed:
        raise ValueError("stage checkpoint distributed world size mismatch")
    stage_record = payload.get("stage")
    if not isinstance(stage_record, dict) or any(
        stage_record.get(name) != value
        for name, value in {
            "start_update": stage.start_update,
            "stop_update": stage.stop_update,
            "continuous_updates_completed": stage.continuous_updates,
            "strict_finite_gradients": stage.require_stability,
        }.items()
    ):
        raise ValueError("stage checkpoint does not prove the expected uninterrupted update range")
    expected_parent = None if predecessor is None else predecessor["last_checkpoint_sha256"]
    if payload.get("parent_checkpoint_sha256") != expected_parent:
        raise ValueError("stage checkpoint does not bind the required predecessor")
    checkpoint = manifest_path.parent / str(payload.get("checkpoint_file", ""))
    if not checkpoint.is_file() or file_sha256(checkpoint) != payload.get("checkpoint_sha256"):
        raise ValueError("selected checkpoint bytes differ from the checkpoint manifest")
    last_checkpoint = manifest_path.parent / "last.pt"
    if not last_checkpoint.is_file() or file_sha256(last_checkpoint) != payload.get(
        "last_checkpoint_sha256"
    ):
        raise ValueError("last checkpoint bytes differ from the checkpoint manifest")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("stage checkpoint lacks metrics")
    for name in ("last_validation_metric", "first_update_loss", "last_update_loss"):
        _require_finite(f"stage metric {name}", metrics.get(name))
    history = payload.get("validation_history")
    if not isinstance(history, list) or not history:
        raise ValueError("stage checkpoint lacks validation history")
    if int(history[-1].get("update", -1)) != stage.stop_update:
        raise ValueError("stage validation history does not end at the stage boundary")
    updates = [int(item.get("update", -1)) for item in history]
    values = [_require_finite("validation history metric", item.get("metric")) for item in history]
    if updates != sorted(set(updates)) or any(
        update <= stage.start_update or update > stage.stop_update for update in updates
    ):
        raise ValueError("stage validation history updates are not strictly ordered/in-range")
    best_metric = min(values)
    best_update = min(update for update, value in zip(updates, values) if value == best_metric)
    if (
        int(payload.get("selected_update", -1)) != best_update
        or int(payload.get("update", -1)) != best_update
        or float(payload.get("validation_metric", float("nan"))) != best_metric
    ):
        raise ValueError("selected checkpoint is not the earliest minimum validation metric")
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("oom_detected") is not False or telemetry.get(
        "nonfinite_detected"
    ) is not False:
        raise ValueError("stage checkpoint reports OOM or non-finite execution")
    ranks = telemetry.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise ValueError("stage telemetry must contain exactly one record per rank")
    observed_ranks = []
    observed_devices = []
    rank_phase_timings: list[Mapping[str, Any]] = []
    for rank_record in ranks:
        if not isinstance(rank_record, dict):
            raise ValueError("rank telemetry records must be objects")
        rank_value = int(rank_record.get("rank", -1))
        observed_ranks.append(rank_value)
        device_uuid = str(rank_record.get("device_uuid", ""))
        observed_devices.append(device_uuid)
        if not device_uuid:
            raise ValueError("rank telemetry lacks a GPU UUID")
        rank_timing = rank_record.get("phase_timing_seconds")
        if not isinstance(rank_timing, Mapping):
            raise ValueError("rank telemetry lacks measured phase timing")
        combine_phase_timings(rank_timing)
        rank_phase_timings.append(rank_timing)
        if rank_record.get("continuous_updates_completed") != stage.continuous_updates:
            raise ValueError("rank telemetry does not prove the continuous stage length")
        if rank_record.get("oom_detected") is not False or rank_record.get(
            "nonfinite_detected"
        ) is not False:
            raise ValueError("rank telemetry reports OOM or non-finite execution")
        if int(rank_record.get("loss_scale_backoff_count", -1)) != 0:
            raise ValueError("training gate encountered an AMP overflow/backoff")
        consecutive_backoffs = int(rank_record.get("max_consecutive_loss_scale_backoffs", -1))
        if consecutive_backoffs < 0 or consecutive_backoffs > (
            protocol.max_consecutive_loss_scale_backoffs
        ):
            raise ValueError("rank telemetry reports persistent loss-scale backoff")
        if stage.require_stability:
            total = int(rank_record.get("total_memory_bytes") or 0)
            allocated = int(rank_record.get("peak_allocated_bytes") or -1)
            reserved = int(rank_record.get("peak_reserved_bytes") or -1)
            if total <= 0 or allocated < 0 or reserved < 0:
                raise ValueError("G3 stability requires measured CUDA memory on every rank")
            if allocated / total >= protocol.max_allocated_fraction:
                raise ValueError("G3 allocated-memory fraction reached the locked 80% limit")
            if reserved / total >= protocol.max_reserved_fraction:
                raise ValueError("G3 reserved-memory fraction reached the locked 90% limit")
    if sorted(observed_ranks) != list(range(world_size)) or len(set(observed_devices)) != world_size:
        raise ValueError("rank telemetry does not close over two unique GPUs")
    stage_timing = payload.get("stage_phase_timing_seconds")
    cumulative_timing = payload.get("phase_timing_seconds")
    if not isinstance(stage_timing, Mapping) or not isinstance(cumulative_timing, Mapping):
        raise ValueError("checkpoint lacks measured cumulative phase timing")
    expected_stage_timing = {
        key: max(float(value[key]) for value in rank_phase_timings)
        for key in PHASE_TIMING_KEYS
    }
    if any(
        not math.isclose(
            float(stage_timing.get(key, -1.0)), expected_stage_timing[key], rel_tol=1e-12, abs_tol=1e-12
        )
        for key in PHASE_TIMING_KEYS
    ):
        raise ValueError("checkpoint stage timing is not derived from per-rank measurements")
    parent_timing = None
    if predecessor is not None:
        parent_manifest = _read_json(
            guard_paper2_path(str(predecessor["checkpoint_manifest_path"]), paper2_root=root)
        )
        parent_timing = parent_manifest.get("phase_timing_seconds")
        if not isinstance(parent_timing, Mapping):
            raise ValueError("predecessor checkpoint lacks cumulative phase timing")
    expected_cumulative_timing = combine_phase_timings(expected_stage_timing, parent_timing)
    if any(
        not math.isclose(
            float(cumulative_timing.get(key, -1.0)),
            expected_cumulative_timing[key],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for key in PHASE_TIMING_KEYS
    ) or payload.get("phase_timing_clock") != "host_perf_counter_cumulative_measured/v1":
        raise ValueError("checkpoint cumulative phase timing contract does not close")
    if stage.selection == "full_k100":
        sampling = evidence.maximum_config["validation_sampling"]
        expected_paths = int(sampling["path_count"]) * len(history)
        if int(payload.get("gate_validation_sampling_calls", -1)) != expected_paths:
            raise ValueError("G3 validation sampling path count does not close")
        if int(payload.get("gate_validation_sampling_nfe", -1)) != expected_paths * int(
            sampling["step_count"]
        ):
            raise ValueError("G3 validation sampling NFE does not close")
        if payload.get("gate_maximum_config_sha256") != canonical_sha256(
            evidence.maximum_config
        ):
            raise ValueError("G3 checkpoint binds a different maximum-input contract")
    if payload.get("gate_data_scope") != evidence.data_scope or payload.get("synthetic") is not False:
        raise ValueError("stage checkpoint is not marked as real prepared-data execution")
    if payload.get("formal_gate") is not False:
        raise ValueError("individual stage manifests cannot unlock the aggregate formal gate")
    if payload.get("gate_evidence_sha256") != evidence.sha256:
        raise ValueError("stage checkpoint binds different gate evidence")
    if payload.get("gate_stage_sha256") != stage.sha256:
        raise ValueError("stage checkpoint binds a different stage contract")
    expected_request_hash = (
        request_file_sha256 if execution_scope == PRODUCTION_EXECUTION_SCOPE else None
    )
    if payload.get("gate_request_file_sha256") != expected_request_hash:
        raise ValueError("stage checkpoint does not bind its exact launcher request")
    if execution_scope == PRODUCTION_EXECUTION_SCOPE:
        try:
            import torch
        except (ImportError, OSError) as exc:  # pragma: no cover - production runtime only
            raise RuntimeError("production checkpoint verification requires PyTorch") from exc
        best_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        last_state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        for name, state, expected_update in (
            ("selected", best_state, best_update),
            ("last", last_state, stage.stop_update),
        ):
            if not isinstance(state, Mapping) or int(state.get("update", -1)) != expected_update:
                raise ValueError(f"{name} checkpoint contains the wrong update")
            if not isinstance(state.get("model"), Mapping) or not state["model"]:
                raise ValueError(f"{name} checkpoint lacks model state")
            if not isinstance(state.get("optimizer"), Mapping) or not state["optimizer"]:
                raise ValueError(f"{name} checkpoint lacks optimizer state")
            if not isinstance(state.get("scaler"), Mapping):
                raise ValueError(f"{name} checkpoint lacks AMP scaler state")
        rng_states = last_state.get("rank_rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != world_size:
            raise ValueError("last checkpoint lacks one RNG state per rank")
        if sorted(int(item.get("rank", -1)) for item in rng_states) != list(range(world_size)):
            raise ValueError("last checkpoint RNG ranks do not close")
    return payload


def _validate_request_receipt(
    *,
    root: Path,
    request_path: Path | None,
    execution_scope: str,
    stage: GateStageSpec,
    protocol: GateProtocol,
    evidence: GateEvidence,
    seed: int,
    world_size: int,
    predecessor: Mapping[str, Any] | None,
    manifest_path: Path,
) -> tuple[str | None, str | None]:
    if execution_scope != PRODUCTION_EXECUTION_SCOPE:
        return None, None
    if request_path is None:
        raise ValueError("production stages require a FileStore-spawn request receipt")
    path = guard_paper2_path(request_path, paper2_root=root)
    if not path.is_file():
        raise FileNotFoundError(path)
    request = _read_json(path)
    expected_resume_checkpoint = (
        None if predecessor is None else predecessor["last_checkpoint_path"]
    )
    expected_resume_manifest = (
        None if predecessor is None else predecessor["checkpoint_manifest_path"]
    )
    expected = {
        "schema": "paper2.training-gate-request/v1",
        "stage": asdict(stage),
        "stage_sha256": stage.sha256,
        "protocol": asdict(protocol),
        "protocol_sha256": protocol.sha256,
        "evidence_sha256": evidence.sha256,
        "seed": seed,
        "world_size": world_size,
        "inputs": dict(evidence.runtime_inputs),
        "resume_checkpoint": expected_resume_checkpoint,
        "resume_manifest": expected_resume_manifest,
        "checkpoint_output": manifest_path.parent.relative_to(root).as_posix(),
    }
    mismatches = [name for name, value in expected.items() if request.get(name) != value]
    if mismatches:
        raise ValueError(
            f"FileStore-spawn request receipt differs from the locked stage: {mismatches}"
        )
    return path.relative_to(root).as_posix(), file_sha256(path)


def _stage_report(
    *,
    root: Path,
    stage: GateStageSpec,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    evidence: GateEvidence,
    protocol: GateProtocol,
    request_path: str | None,
    request_sha256: str | None,
    execution_scope: str,
) -> dict[str, Any]:
    checkpoint_path = manifest_path.parent / str(manifest["checkpoint_file"])
    last_path = manifest_path.parent / "last.pt"
    payload: dict[str, Any] = {
        "schema": GATE_STAGE_SCHEMA,
        "stage_id": stage.stage_id,
        "gate": stage.gate,
        "status": "PASS",
        "formal_gate": False,
        "synthetic": False,
        "data_scope": evidence.data_scope,
        "stage_contract": asdict(stage),
        "stage_contract_sha256": stage.sha256,
        "protocol_sha256": protocol.sha256,
        "evidence_sha256": evidence.sha256,
        "input_hashes": dict(sorted(evidence.input_hashes.items())),
        "code_sha256": evidence.code_sha256,
        "environment_sha256": evidence.environment_sha256,
        "execution_scope": execution_scope,
        "production_execution": execution_scope == PRODUCTION_EXECUTION_SCOPE,
        "request": (
            None
            if request_path is None
            else {"relative_path": request_path, "sha256": request_sha256}
        ),
        "checkpoint_manifest": {
            "relative_path": _relative(manifest_path, root),
            "sha256": file_sha256(manifest_path),
        },
        "selected_checkpoint": {
            "relative_path": _relative(checkpoint_path, root),
            "sha256": file_sha256(checkpoint_path),
        },
        "last_checkpoint": {
            "relative_path": _relative(last_path, root),
            "sha256": file_sha256(last_path),
        },
        "training_config_sha256": manifest["training_config_sha256"],
        "model_config_sha256": manifest["model_config_sha256"],
        "split_hashes": manifest["split_hashes"],
        "metrics": manifest["metrics"],
        "validation_history": manifest["validation_history"],
        "telemetry": manifest["telemetry"],
        "stage_phase_timing_seconds": manifest["stage_phase_timing_seconds"],
        "phase_timing_seconds": manifest["phase_timing_seconds"],
        "phase_timing_clock": manifest["phase_timing_clock"],
    }
    payload["stage_report_sha256"] = canonical_sha256(payload)
    return payload


def _load_state(path: Path, *, identity_sha256: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": GATE_STATE_SCHEMA,
            "identity_sha256": identity_sha256,
            "completed": [],
            "failures": [],
            "active_attempt": None,
        }
    state = _read_json(path)
    if state.get("schema") != GATE_STATE_SCHEMA or state.get("identity_sha256") != identity_sha256:
        raise ValueError("existing G2/G3 state belongs to a different evidence/configuration identity")
    if (
        not isinstance(state.get("completed"), list)
        or not isinstance(state.get("failures"), list)
        or (state.get("active_attempt") is not None and not isinstance(state["active_attempt"], dict))
    ):
        raise ValueError("G2/G3 state is malformed")
    return state


def _validate_completed_prefix(
    state: Mapping[str, Any],
    *,
    root: Path,
    stages: Sequence[GateStageSpec],
    protocol: GateProtocol,
    evidence: GateEvidence,
    seed: int,
    world_size: int,
) -> None:
    completed = list(state["completed"])
    observed = [str(item.get("stage_id", "")) for item in completed]
    expected = [stage.stage_id for stage in stages[: len(completed)]]
    if observed != expected:
        raise ValueError("resume state is not an exact prefix of the locked stage order")
    completed_by_id: dict[str, Mapping[str, Any]] = {}
    for stage, record in zip(stages, completed):
        predecessor = (
            None if stage.resume_from is None else completed_by_id.get(stage.resume_from)
        )
        if stage.resume_from is not None and predecessor is None:
            raise ValueError("completed stage chain lacks its locked predecessor")
        report_path = guard_paper2_path(str(record["stage_report_path"]), paper2_root=root)
        if not report_path.is_file() or file_sha256(report_path) != record.get(
            "stage_report_file_sha256"
        ):
            raise ValueError("a completed stage report is missing or changed")
        report = _read_json(report_path)
        if (
            report.get("schema") != GATE_STAGE_SCHEMA
            or report.get("status") != "PASS"
            or report.get("stage_report_sha256")
            != canonical_sha256(
                {key: value for key, value in report.items() if key != "stage_report_sha256"}
            )
        ):
            raise ValueError("a completed stage report is not a passed stage artifact")
        last_path = guard_paper2_path(str(record["last_checkpoint_path"]), paper2_root=root)
        if not last_path.is_file() or file_sha256(last_path) != record.get(
            "last_checkpoint_sha256"
        ):
            raise ValueError("a completed stage's continuation checkpoint changed")
        manifest_path = guard_paper2_path(
            str(record["checkpoint_manifest_path"]), paper2_root=root
        )
        if not manifest_path.is_file() or file_sha256(manifest_path) != record.get(
            "checkpoint_manifest_file_sha256"
        ):
            raise ValueError("a completed stage's checkpoint manifest changed")
        execution_scope = str(record.get("execution_scope", ""))
        request_value = record.get("request_path")
        request_path = (
            None
            if request_value is None
            else guard_paper2_path(str(request_value), paper2_root=root)
        )
        request_relative, request_sha256 = _validate_request_receipt(
            root=root,
            request_path=request_path,
            execution_scope=execution_scope,
            stage=stage,
            protocol=protocol,
            evidence=evidence,
            seed=seed,
            world_size=world_size,
            predecessor=predecessor,
            manifest_path=manifest_path,
        )
        if request_sha256 != record.get("request_file_sha256"):
            raise ValueError("a completed stage request receipt changed")
        manifest = _validate_checkpoint_manifest(
            manifest_path,
            root=root,
            stage=stage,
            protocol=protocol,
            evidence=evidence,
            seed=seed,
            world_size=world_size,
            predecessor=predecessor,
            execution_scope=execution_scope,
            request_file_sha256=request_sha256,
        )
        derived_report = _stage_report(
            root=root,
            stage=stage,
            manifest_path=manifest_path,
            manifest=manifest,
            evidence=evidence,
            protocol=protocol,
            request_path=request_relative,
            request_sha256=request_sha256,
            execution_scope=execution_scope,
        )
        if report != derived_report:
            raise ValueError("a completed stage report is not derivable from its locked artifacts")
        if stage.require_overfit:
            if predecessor is None:
                raise ValueError("overfit continuation lacks its initial stage")
            initial_report = _read_json(
                guard_paper2_path(str(predecessor["stage_report_path"]), paper2_root=root)
            )
            initial = _require_finite(
                "initial overfit validation metric",
                initial_report["metrics"].get("initial_validation_metric"),
            )
            final = _require_finite(
                "final overfit validation metric",
                manifest["metrics"].get("last_validation_metric"),
            )
            if initial <= 0.0 or final / initial > protocol.overfit_validation_ratio_max:
                raise ValueError("completed overfit chain no longer satisfies the locked ratio")
        completed_by_id[stage.stage_id] = record


def _execute_training_gates(
    *,
    paper2_root: str | Path,
    output_directory: str | Path,
    report_output: str | Path,
    evidence: GateEvidence,
    protocol: GateProtocol,
    runner: GateRunner,
    seed: int,
    world_size: int,
    require_formal_evidence: bool = True,
    allow_formal_output: bool = False,
) -> dict[str, Any]:
    """Run or resume the fixed G2/G3 prefix and return the aggregate report."""

    root = Path(paper2_root).resolve()
    output = guard_paper2_path(output_directory, paper2_root=root, allow_root=True)
    if output == root:
        raise ValueError("G2/G3 output directory must be below the Paper2 root")
    report_path = guard_paper2_path(report_output, paper2_root=root)
    if report_path.parent != output:
        raise ValueError("aggregate gate report must be written directly in the gate output directory")
    if seed < 0 or world_size <= 0:
        raise ValueError("seed must be nonnegative and world_size must be positive")
    protocol.validate()
    if allow_formal_output is not require_formal_evidence:
        raise ValueError("formal evidence validation and formal-output authority must agree")
    evidence.validate(require_formal=require_formal_evidence)
    if require_formal_evidence and (seed != 20260721 or world_size != 2):
        raise ValueError("formal G2/G3 is locked to seed=20260721 and world_size=2")
    stages = gate_stages(protocol)
    identity = canonical_sha256(
        {
            "schema": GATE_REPORT_SCHEMA,
            "protocol_sha256": protocol.sha256,
            "evidence_sha256": evidence.sha256,
            "seed": seed,
            "world_size": world_size,
            "stage_sha256": [stage.sha256 for stage in stages],
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    state = _load_state(state_path, identity_sha256=identity)
    if state.get("active_attempt") is not None:
        interrupted = dict(state["active_attempt"])
        state["failures"].append(
            {
                "stage_id": interrupted.get("stage_id"),
                "attempt": interrupted.get("attempt"),
                "error_type": "InterruptedAttempt",
                "error_message": "previous process ended before atomically closing the attempt",
            }
        )
        state["active_attempt"] = None
        _atomic_json(state_path, state)
    _validate_completed_prefix(
        state,
        root=root,
        stages=stages,
        protocol=protocol,
        evidence=evidence,
        seed=seed,
        world_size=world_size,
    )
    completed_by_id = {str(item["stage_id"]): item for item in state["completed"]}
    for stage in stages[len(state["completed"]) :]:
        predecessor = None
        resume_checkpoint = None
        resume_manifest = None
        if stage.resume_from is not None:
            predecessor = completed_by_id.get(stage.resume_from)
            if predecessor is None:
                raise ValueError("resume cannot skip the required predecessor stage")
            resume_checkpoint = guard_paper2_path(
                str(predecessor["last_checkpoint_path"]), paper2_root=root
            )
            resume_manifest = guard_paper2_path(
                str(predecessor["checkpoint_manifest_path"]), paper2_root=root
            )
        prior_failures = [
            item for item in state["failures"] if item.get("stage_id") == stage.stage_id
        ]
        attempt = len(prior_failures) + 1
        stage_directory = output / "stages" / stage.stage_id
        state["active_attempt"] = {"stage_id": stage.stage_id, "attempt": attempt}
        _atomic_json(state_path, state)
        try:
            receipt = runner.run(
                stage,
                attempt=attempt,
                resume_checkpoint=resume_checkpoint,
                resume_manifest=resume_manifest,
                output_directory=stage_directory,
            )
            manifest_path = guard_paper2_path(
                receipt.checkpoint_manifest_path, paper2_root=root
            )
            request_relative, request_sha256 = _validate_request_receipt(
                root=root,
                request_path=receipt.request_path,
                execution_scope=receipt.execution_scope,
                stage=stage,
                protocol=protocol,
                evidence=evidence,
                seed=seed,
                world_size=world_size,
                predecessor=predecessor,
                manifest_path=manifest_path,
            )
            manifest = _validate_checkpoint_manifest(
                manifest_path,
                root=root,
                stage=stage,
                protocol=protocol,
                evidence=evidence,
                seed=seed,
                world_size=world_size,
                predecessor=predecessor,
                execution_scope=receipt.execution_scope,
                request_file_sha256=request_sha256,
            )
            if stage.require_overfit:
                initial_stage = completed_by_id[str(stage.resume_from)]
                initial_report = _read_json(
                    guard_paper2_path(
                        str(initial_stage["stage_report_path"]), paper2_root=root
                    )
                )
                initial = _require_finite(
                    "initial overfit validation metric",
                    initial_report["metrics"].get("initial_validation_metric"),
                )
                final = _require_finite(
                    "final overfit validation metric",
                    manifest["metrics"].get("last_validation_metric"),
                )
                if initial <= 0.0 or final / initial > protocol.overfit_validation_ratio_max:
                    raise ValueError(
                        "32-patch overfit did not reach the locked validation-metric ratio"
                    )
            report = _stage_report(
                root=root,
                stage=stage,
                manifest_path=manifest_path,
                manifest=manifest,
                evidence=evidence,
                protocol=protocol,
                request_path=request_relative,
                request_sha256=request_sha256,
                execution_scope=receipt.execution_scope,
            )
            # Keep the immutable report beside this attempt's checkpoint manifest.
            # If the orchestrator crashes before updating state.json, a new attempt
            # can proceed without overwriting evidence from the orphaned attempt.
            stage_report_path = manifest_path.parent / "stage-report.json"
            if stage_report_path.exists():
                existing = _read_json(stage_report_path)
                if existing != report:
                    raise FileExistsError("refusing to replace a different passed stage report")
            else:
                _atomic_json(stage_report_path, report)
            last_path = manifest_path.parent / "last.pt"
            record = {
                "stage_id": stage.stage_id,
                "stage_report_path": _relative(stage_report_path, root),
                "stage_report_file_sha256": file_sha256(stage_report_path),
                "checkpoint_manifest_path": _relative(manifest_path, root),
                "checkpoint_manifest_file_sha256": file_sha256(manifest_path),
                "last_checkpoint_path": _relative(last_path, root),
                "last_checkpoint_sha256": file_sha256(last_path),
                "request_path": request_relative,
                "request_file_sha256": request_sha256,
                "execution_scope": receipt.execution_scope,
            }
            state["completed"].append(record)
            completed_by_id[stage.stage_id] = record
            state["active_attempt"] = None
            _atomic_json(state_path, state)
        except Exception as exc:
            state["failures"].append(
                {
                    "stage_id": stage.stage_id,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1_000],
                }
            )
            state["active_attempt"] = None
            _atomic_json(state_path, state)
            raise
    _validate_completed_prefix(
        state,
        root=root,
        stages=stages,
        protocol=protocol,
        evidence=evidence,
        seed=seed,
        world_size=world_size,
    )
    production_closed = all(
        record.get("execution_scope") == PRODUCTION_EXECUTION_SCOPE
        for record in state["completed"]
    )
    formal_gate = (
        allow_formal_output
        and evidence.formal_eligible
        and not evidence.synthetic
        and production_closed
    )
    stage_records = []
    for record in state["completed"]:
        stage_report = _read_json(
            guard_paper2_path(str(record["stage_report_path"]), paper2_root=root)
        )
        stage_records.append(
            {
                "stage_id": record["stage_id"],
                "stage_report_path": record["stage_report_path"],
                "stage_report_file_sha256": record["stage_report_file_sha256"],
                "checkpoint_manifest_path": record["checkpoint_manifest_path"],
                "checkpoint_manifest_file_sha256": record[
                    "checkpoint_manifest_file_sha256"
                ],
                "last_checkpoint_path": record["last_checkpoint_path"],
                "last_checkpoint_sha256": record["last_checkpoint_sha256"],
                "status": stage_report["status"],
                "execution_scope": record["execution_scope"],
                "request_path": record["request_path"],
                "request_file_sha256": record["request_file_sha256"],
            }
        )
    gate_status = "PASS" if formal_gate else "INELIGIBLE"
    aggregate: dict[str, Any] = {
        "schema": GATE_REPORT_SCHEMA,
        "gate": "G2/G3",
        "status": "complete" if formal_gate else "ineligible",
        "formal_gate": formal_gate,
        "synthetic": evidence.synthetic,
        "data_scope": evidence.data_scope,
        "gates": {
            "G2": {"status": gate_status, "stage_count": 4},
            "G3": {"status": gate_status, "stage_count": 2},
        },
        "identity_sha256": identity,
        "protocol": asdict(protocol),
        "protocol_sha256": protocol.sha256,
        "evidence": evidence.to_dict(),
        "evidence_sha256": evidence.sha256,
        "input_hashes": dict(sorted(evidence.input_hashes.items())),
        "code_sha256": evidence.code_sha256,
        "environment_sha256": evidence.environment_sha256,
        "seed": seed,
        "world_size": world_size,
        "stage_order": [stage.stage_id for stage in stages],
        "stages": stage_records,
        "resume_policy": "exact_completed_prefix_only_no_stage_skip",
        "limitations": [] if formal_gate else ["synthetic_or_nonprepared_evidence_cannot_pass"],
    }
    aggregate["report_sha256"] = canonical_sha256(aggregate)
    if report_path.exists():
        existing = _read_json(report_path)
        if existing != aggregate:
            raise FileExistsError("refusing to replace a different aggregate G2/G3 report")
    else:
        _atomic_json(report_path, aggregate)
    return aggregate


def execute_training_gates(
    *,
    paper2_root: str | Path,
    output_directory: str | Path,
    report_output: str | Path,
    evidence: GateEvidence,
    protocol: GateProtocol,
    runner: GateRunner,
    seed: int,
    world_size: int,
) -> dict[str, Any]:
    """Run diagnostic gates without authority to emit ``formal_gate=true``.

    Formal execution is exposed only by ``execute_formal_training_gates`` in
    the runtime module, which internally constructs the production FileStore-spawn
    runner and its evidence rather than accepting caller-injected identities.
    """

    return _execute_training_gates(
        paper2_root=paper2_root,
        output_directory=output_directory,
        report_output=report_output,
        evidence=evidence,
        protocol=protocol,
        runner=runner,
        seed=seed,
        world_size=world_size,
        require_formal_evidence=False,
        allow_formal_output=False,
    )
