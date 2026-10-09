from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time

import numpy as np

from .models import TORCH_AVAILABLE, UNetConfig, canonical_model_identity, method_spec


FORMAL_SEEDS = (20260721, 20260722, 20260723)
PHASE_TIMING_KEYS = (
    "data_loading",
    "forward",
    "backward",
    "optimizer",
    "validation",
)


def combine_phase_timings(
    stage: Mapping[str, Any], parent: Mapping[str, Any] | None = None
) -> dict[str, float]:
    """Validate measured phase seconds and add an optional resume prefix."""

    def validated(name: str, value: Mapping[str, Any]) -> dict[str, float]:
        if set(value) != set(PHASE_TIMING_KEYS):
            raise ValueError(f"{name} phase timing keys differ from the frozen contract")
        converted: dict[str, float] = {}
        for key in PHASE_TIMING_KEYS:
            raw = value[key]
            if isinstance(raw, bool):
                raise ValueError(f"{name} phase timing {key} must be measured seconds")
            seconds = float(raw)
            if not math.isfinite(seconds) or seconds < 0.0:
                raise ValueError(f"{name} phase timing {key} must be finite and nonnegative")
            converted[key] = seconds
        return converted

    current = validated("stage", stage)
    prefix = (
        {key: 0.0 for key in PHASE_TIMING_KEYS}
        if parent is None
        else validated("parent", parent)
    )
    return {key: prefix[key] + current[key] for key in PHASE_TIMING_KEYS}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    seed: int
    updates: int = 120_000
    micro_batch_per_gpu: int = 1
    effective_batch: int = 8
    optimizer: str = "adamw"
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-2
    learning_rate_schedule: str = "constant"
    warmup_updates: int = 0
    cosine_decay_updates: int | None = None
    minimum_learning_rate_ratio: float = 0.0
    validation_interval: int = 2_000
    selection_metric: str = "bridge_val_normalized_mse"
    network_precision: str = "fp16"
    data_loader_workers: int = 4
    early_stopping: bool = False
    g_min: float = 0.05
    alpha: float = 1.0
    endpoint_weight: float = 0.1
    gradient_weight: float = 0.1
    bridge_t_min: float = 1.0e-3
    gradient_clip_norm: float | None = None
    amp_forward_replay: bool = False
    ddpm_schedule: str = "cosine"
    ddpm_cosine_offset: float = 0.008

    def validate(self, *, formal: bool = False, world_size: int = 1) -> None:
        spec = method_spec(self.method)
        if not spec.learned:
            raise ValueError(f"{self.method} has no trainable checkpoint")
        if self.seed < 0 or self.updates <= 0:
            raise ValueError("seed and updates must be non-negative/positive")
        if self.micro_batch_per_gpu != 1:
            raise ValueError("Paper2 protocol fixes micro_batch_per_gpu=1")
        if self.effective_batch != 8:
            raise ValueError("Paper2 protocol fixes effective_batch=8")
        if self.optimizer.lower() != "adamw":
            raise ValueError("Paper2 protocol fixes the optimizer to AdamW")
        if self.effective_batch % (self.micro_batch_per_gpu * world_size):
            raise ValueError("effective batch must be divisible by micro-batch times world size")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if self.early_stopping:
            raise ValueError("early stopping is forbidden; all methods receive equal updates")
        if not 0 <= self.data_loader_workers <= 8:
            raise ValueError("total DataLoader workers must not exceed 8")
        if self.network_precision not in {"fp16", "fp32"}:
            raise ValueError("network_precision must be fp16 or fp32")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.learning_rate_schedule not in {"constant", "warmup_cosine"}:
            raise ValueError("learning_rate_schedule must be constant or warmup_cosine")
        if self.warmup_updates < 0 or self.warmup_updates >= self.updates:
            raise ValueError("warmup_updates must lie in [0, updates)")
        if self.learning_rate_schedule == "constant" and self.warmup_updates != 0:
            raise ValueError("constant learning rate cannot use warmup updates")
        if self.learning_rate_schedule == "constant" and self.cosine_decay_updates is not None:
            raise ValueError("constant learning rate cannot set cosine_decay_updates")
        if self.learning_rate_schedule == "warmup_cosine":
            decay_update = (
                self.updates
                if self.cosine_decay_updates is None
                else self.cosine_decay_updates
            )
            if not self.warmup_updates < decay_update <= self.updates:
                raise ValueError(
                    "cosine_decay_updates must lie in (warmup_updates, updates]"
                )
        if (
            not np.isfinite(self.minimum_learning_rate_ratio)
            or not 0.0 <= self.minimum_learning_rate_ratio <= 1.0
        ):
            raise ValueError("minimum_learning_rate_ratio must lie in [0,1]")
        if (
            self.learning_rate_schedule == "constant"
            and self.minimum_learning_rate_ratio != 0.0
        ):
            raise ValueError("constant learning rate cannot use a minimum ratio")
        if not np.isfinite(self.g_min) or not 0.0 < self.g_min <= 1.0:
            raise ValueError("g_min must be finite and lie in (0,1]")
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        for name, value in (
            ("endpoint_weight", self.endpoint_weight),
            ("gradient_weight", self.gradient_weight),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.bridge_t_min) or not 0.0 < self.bridge_t_min < 0.5:
            raise ValueError("bridge_t_min must be finite and lie in (0,0.5)")
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be null or finite and positive")
        if not isinstance(self.amp_forward_replay, bool):
            raise ValueError("amp_forward_replay must be boolean")
        if self.ddpm_schedule != "cosine":
            raise ValueError("the registered DDPM control requires the cosine schedule")
        if not np.isfinite(self.ddpm_cosine_offset) or not 0.0 <= self.ddpm_cosine_offset < 1.0:
            raise ValueError("ddpm_cosine_offset must be finite and lie in [0,1)")
        if formal:
            if self.method == "F11":
                raise ValueError(
                    "formal F11 must reuse the OT-SB-full checkpoint and cannot be retrained"
                )
            if self.updates != 120_000 or self.validation_interval != 2_000:
                raise ValueError("formal runs require 120000 updates and validation every 2000")
            if self.seed not in FORMAL_SEEDS:
                raise ValueError("formal run seed is not one of the three frozen shared seeds")
            if self.network_precision != "fp16":
                raise ValueError("formal network training requires FP16")
            if self.selection_metric != "bridge_val_normalized_mse":
                raise ValueError("formal checkpoint selection metric differs")

    @property
    def identity(self) -> str:
        return canonical_model_identity(self.method)

    def accumulation_steps(self, world_size: int) -> int:
        self.validate(world_size=world_size)
        return self.effective_batch // (self.micro_batch_per_gpu * world_size)


def configure_determinism(seed: int) -> None:
    # CPython's process-start hash seed has the same unsigned 32-bit bound as
    # NumPy.  A 63-bit rank seed in the inherited environment can make any
    # later helper interpreter abort before startup.
    projected_seed = seed % (2**32)
    os.environ.setdefault("PYTHONHASHSEED", str(projected_seed))
    random.seed(seed)
    # NumPy's legacy RandomState only accepts an unsigned 32-bit seed, while
    # the hash-derived per-rank Paper2 seed intentionally occupies 63 bits.
    # Keep the full seed for Python/PyTorch and deterministically project only
    # the NumPy stream into its documented domain.
    np.random.seed(projected_seed)
    if TORCH_AVAILABLE:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def distributed_stochastic_seed(seed: int, rank: int) -> int:
    """Derive deterministic, non-overlapping random streams for DDP ranks."""

    if seed < 0 or rank < 0:
        raise ValueError("seed and rank must be nonnegative")
    if rank == 0:
        return seed
    payload = _canonical_bytes(
        {
            "protocol": "paper2-ddp-rank-stochastic-seed/v1",
            "base_seed": seed,
            "rank": rank,
        }
    )
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**63 - 1)


def deterministic_rank_indices(count: int, *, rank: int, world_size: int) -> tuple[int, ...]:
    """Return a stable, disjoint strided shard of an indexed training set."""

    if count <= 0:
        raise ValueError("indexed training set must be nonempty")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must lie in [0, world_size)")
    indices = tuple(range(rank, count, world_size))
    if not indices:
        raise ValueError("training index has fewer samples than DDP ranks")
    return indices


def gradient_sync_context(
    model: Any,
    *,
    micro_step: int,
    accumulation_steps: int,
    world_size: int,
) -> Any:
    """Suppress DDP gradient reductions until the last accumulated micro-step."""

    if accumulation_steps <= 0 or not 0 <= micro_step < accumulation_steps:
        raise ValueError("micro_step must index the positive accumulation window")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if world_size == 1 or micro_step == accumulation_steps - 1:
        return nullcontext()
    no_sync = getattr(model, "no_sync", None)
    if no_sync is None:
        raise RuntimeError("multi-rank accumulated training requires a DDP no_sync context")
    return no_sync()


def expected_formal_checkpoint_count() -> int:
    from .models import formal_model_identities

    count = len(formal_model_identities()) * len(FORMAL_SEEDS)
    if count != 33:
        raise AssertionError("formal protocol requires exactly 33 learned model identities")
    return count


def checkpoint_manifest(
    *,
    config: TrainingConfig,
    model_config: UNetConfig,
    split_hashes: Mapping[str, str],
    code_hash: str,
    environment_hash: str,
    update: int,
    validation_metric: float | None,
    validation_history: Iterable[Mapping[str, Any]] | None = None,
    checkpoint_path: Path,
    completed_updates: int | None = None,
    parent_checkpoint_sha256: str | None = None,
    status: str = "complete",
    world_size: int = 1,
) -> dict[str, Any]:
    if status not in {"running", "complete", "failed"}:
        raise ValueError("invalid checkpoint status")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    selected_update = int(update)
    completed = selected_update if completed_updates is None else int(completed_updates)
    if selected_update < 0 or completed < selected_update or completed > config.updates:
        raise ValueError(
            "checkpoint updates must satisfy 0 <= selected_update <= completed_updates "
            "<= target_updates"
        )
    if status == "complete" and completed != config.updates:
        raise ValueError("complete checkpoint manifests require all target updates")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": config.method,
        "model_identity": config.identity,
        "seed": config.seed,
        # ``update`` is retained as a backwards-compatible alias for the update
        # whose weights are named by ``checkpoint_sha256``.
        "update": selected_update,
        "selected_update": selected_update,
        "completed_updates": completed,
        "target_updates": config.updates,
        "validation_metric_name": config.selection_metric,
        "validation_metric": validation_metric,
        "validation_history": (
            []
            if validation_history is None
            else [
                {"update": int(item["update"]), "metric": float(item["metric"])}
                for item in validation_history
            ]
        ),
        "selection_rule": "minimum_bridge_val_metric_earliest_tie",
        "training_config": asdict(config),
        "training_config_sha256": _payload_sha256(asdict(config)),
        "model_config": model_config.to_dict(),
        "model_config_sha256": _payload_sha256(model_config.to_dict()),
        "split_hashes": dict(sorted(split_hashes.items())),
        "code_sha256": code_hash,
        "environment_sha256": environment_hash,
        "distributed": {
            "world_size": world_size,
            "stochastic_seed_protocol": "paper2-ddp-rank-stochastic-seed/v1",
            "rank_stochastic_seeds": [
                distributed_stochastic_seed(config.seed, rank) for rank in range(world_size)
            ],
        },
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "status": status,
    }
    payload["manifest_sha256"] = _payload_sha256(payload)
    return payload


def validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    config: TrainingConfig,
    model_config: UNetConfig,
    split_hashes: Mapping[str, str],
    code_hash: str,
    environment_hash: str,
    world_size: int = 1,
) -> None:
    if manifest.get("manifest_sha256") != _payload_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("resume checkpoint manifest semantic hash mismatch")
    expected = {
        "method": config.method,
        "model_identity": config.identity,
        "seed": config.seed,
        "training_config_sha256": _payload_sha256(asdict(config)),
        "model_config_sha256": _payload_sha256(model_config.to_dict()),
        "split_hashes": dict(sorted(split_hashes.items())),
        "code_sha256": code_hash,
        "environment_sha256": environment_hash,
        "distributed": {
            "world_size": world_size,
            "stochastic_seed_protocol": "paper2-ddp-rank-stochastic-seed/v1",
            "rank_stochastic_seeds": [
                distributed_stochastic_seed(config.seed, rank) for rank in range(world_size)
            ],
        },
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"resume checkpoint contract mismatch: {', '.join(mismatches)}")


def checkpoint_model_state(model: Any) -> Mapping[str, Any]:
    """Return inference-loadable weights even when training uses DDP."""

    owner = getattr(model, "module", model)
    state = owner.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise ValueError("model checkpoint state must be a nonempty mapping")
    return state


def load_checkpoint_model_state(model: Any, state: Mapping[str, Any]) -> None:
    """Load an inference-form checkpoint into a bare model or a DDP wrapper."""

    owner = getattr(model, "module", model)
    owner.load_state_dict(state)


def _cycle(source: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    while True:
        emitted = False
        for batch in source:
            emitted = True
            yield batch
        if not emitted:
            raise ValueError("training iterable is empty")


def _cycle_from_offset(
    source: Iterable[Mapping[str, Any]], offset: int
) -> tuple[Iterator[Mapping[str, Any]], str]:
    """Resume the exact cyclic batch order, using an indexed seek when available."""

    position = int(offset)
    if position < 0 or position != offset:
        raise ValueError("training batch offset must be a nonnegative integer")
    seek = getattr(source, "iter_from", None)
    if callable(seek):
        return iter(seek(position)), "verified-indexed-cyclic-offset/v1"
    batches = _cycle(source)
    for _ in range(position):
        next(batches)
    return batches, "verified-replay/v1"


def train_model(
    *,
    model: Any,
    train_batches: Iterable[Mapping[str, Any]],
    validation: Callable[[Any, int], float],
    loss_function: Callable[[Any, Mapping[str, Any]], Any],
    output: Path,
    config: TrainingConfig,
    model_config: UNetConfig,
    split_hashes: Mapping[str, str],
    code_hash: str,
    environment_hash: str,
    formal: bool = False,
    stop_update: int | None = None,
    resume_checkpoint: Path | None = None,
    resume_manifest: Mapping[str, Any] | None = None,
    evaluate_at_start: bool = False,
    strict_finite_gradients: bool = False,
    preserve_validation_history_on_resume: bool = False,
    objective_telemetry: Mapping[str, Any]
    | Callable[[], Mapping[str, Any]]
    | None = None,
    initialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an equal-update training job and write hash-locked best/last checkpoints.

    ``loss_function`` owns bridge-time/noise sampling. It must return a scalar tensor.
    This loop deliberately has no early-stop path; validation only selects the best
    checkpoint among runs that all complete the same update count.
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training")
    import torch

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if not 0 <= rank < world_size:
        raise ValueError("RANK must lie in [0, WORLD_SIZE)")
    config.validate(formal=formal, world_size=world_size)
    model_config.validate(formal=formal)
    stage_stop = config.updates if stop_update is None else int(stop_update)
    if not 0 < stage_stop <= config.updates:
        raise ValueError("stop_update must lie in [1, config.updates]")
    if (resume_checkpoint is None) != (resume_manifest is None):
        raise ValueError("resume_checkpoint and resume_manifest must be supplied together")
    configure_determinism(distributed_stochastic_seed(config.seed, rank))
    accumulation = config.accumulation_steps(world_size)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def scheduled_learning_rate(update: int) -> float:
        if config.learning_rate_schedule == "constant":
            return float(config.learning_rate)
        if config.warmup_updates and update <= config.warmup_updates:
            return float(config.learning_rate) * float(update) / float(
                config.warmup_updates
            )
        decay_end = (
            config.updates
            if config.cosine_decay_updates is None
            else config.cosine_decay_updates
        )
        decay_updates = decay_end - config.warmup_updates
        progress = (update - config.warmup_updates) / max(decay_updates, 1)
        progress = min(max(float(progress), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = config.minimum_learning_rate_ratio + (
            1.0 - config.minimum_learning_rate_ratio
        ) * cosine
        return float(config.learning_rate) * ratio

    use_amp = config.network_precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output.mkdir(parents=True, exist_ok=True)
    start_update = 0
    parent_checkpoint_sha256: str | None = None
    best_metric = float("inf")
    best_update: int | None = None
    best_path = output / "best.pt"
    validation_history: list[dict[str, float | int]] = []
    parent_wall_seconds = 0.0
    if resume_checkpoint is not None and resume_manifest is not None:
        validate_resume_manifest(
            resume_manifest,
            config=config,
            model_config=model_config,
            split_hashes=split_hashes,
            code_hash=code_hash,
            environment_hash=environment_hash,
            world_size=world_size,
        )
        parent_timing = resume_manifest.get("phase_timing_seconds")
        if not isinstance(parent_timing, Mapping):
            raise ValueError("resume manifest lacks cumulative measured phase timing")
        combine_phase_timings(
            {key: 0.0 for key in PHASE_TIMING_KEYS},
            parent_timing,
        )
        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        parent_checkpoint_sha256 = _file_sha256(resume_checkpoint)
        if resume_manifest.get("last_checkpoint_sha256") != parent_checkpoint_sha256:
            raise ValueError("resume checkpoint differs from the prior manifest's last checkpoint")
        state = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        start_update = int(state.get("update", -1))
        if start_update != int(resume_manifest.get("completed_updates", -2)):
            raise ValueError("resume checkpoint update differs from the prior manifest")
        if not 0 < start_update < stage_stop:
            raise ValueError("resume update must be positive and precede stop_update")
        model_state = state.get("model")
        if not isinstance(model_state, Mapping) or not model_state:
            raise ValueError("resume checkpoint lacks model state")
        load_checkpoint_model_state(model, model_state)
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        if preserve_validation_history_on_resume:
            raw_history = resume_manifest.get("validation_history")
            if not isinstance(raw_history, list) or not raw_history:
                raise ValueError("formal resume manifest lacks validation history")
            validation_history = [
                {"update": int(item["update"]), "metric": float(item["metric"])}
                for item in raw_history
                if isinstance(item, Mapping)
            ]
            if len(validation_history) != len(raw_history):
                raise ValueError("formal resume validation history is malformed")
            updates = [int(item["update"]) for item in validation_history]
            metrics = [float(item["metric"]) for item in validation_history]
            if (
                updates != sorted(set(updates))
                or updates[-1] != start_update
                or any(update <= 0 or update > start_update for update in updates)
                or not all(math.isfinite(metric) for metric in metrics)
            ):
                raise ValueError("formal resume validation history does not close its prefix")
            best_metric = min(metrics)
            best_update = min(
                update
                for update, metric in zip(updates, metrics, strict=True)
                if metric == best_metric
            )
            if (
                int(resume_manifest.get("selected_update", -1)) != best_update
                or float(resume_manifest.get("validation_metric", float("nan")))
                != best_metric
            ):
                raise ValueError("formal resume selected checkpoint differs from its history")
            selected_source = resume_checkpoint.parent / str(
                resume_manifest.get("checkpoint_file", "")
            )
            if (
                not selected_source.is_file()
                or _file_sha256(selected_source)
                != resume_manifest.get("checkpoint_sha256")
            ):
                raise ValueError("formal resume selected checkpoint bytes differ")
            if selected_source.resolve() != best_path.resolve():
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".best.pt.resume.", dir=output
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    shutil.copyfile(selected_source, temporary)
                    temporary.replace(best_path)
                finally:
                    temporary.unlink(missing_ok=True)
            parent_wall_seconds = float(resume_manifest.get("wall_seconds", 0.0))
            if not math.isfinite(parent_wall_seconds) or parent_wall_seconds < 0.0:
                raise ValueError("formal resume wall time must be finite and nonnegative")
    batch_offset = start_update * accumulation
    batches, batch_resume_protocol = _cycle_from_offset(train_batches, batch_offset)
    if start_update:
        rank_rng_states = state.get("rank_rng_states")
        if not isinstance(rank_rng_states, list) or len(rank_rng_states) != world_size:
            raise ValueError("resume checkpoint lacks one RNG state per distributed rank")
        rng_state = rank_rng_states[rank]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        # ``torch.load(..., map_location=device)`` also maps serialized RNG
        # byte tensors.  PyTorch's RNG setters require their state tensors on
        # CPU even when restoring a CUDA generator.
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng_state["torch_cuda"].cpu(), device)
    initial_validation_metric: float | None = None
    first_update_loss: float | None = None
    last_update_loss: float | None = None
    loss_scale_backoffs = 0
    consecutive_loss_scale_backoffs = 0
    max_consecutive_loss_scale_backoffs = 0
    gradient_clip_applications = 0
    max_preclip_gradient_norm = 0.0
    validation_calls = 0
    first_learning_rate: float | None = None
    last_learning_rate: float | None = None
    stage_phase_timing = {key: 0.0 for key in PHASE_TIMING_KEYS}
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    if evaluate_at_start and start_update == 0:
        phase_start = time.perf_counter()
        initial_validation_metric = float(validation(model, 0))
        stage_phase_timing["validation"] += time.perf_counter() - phase_start
        validation_calls += 1
        if not np.isfinite(initial_validation_metric):
            raise FloatingPointError("non-finite initial validation metric")
    for update in range(start_update + 1, stage_stop + 1):
        phase_start = time.perf_counter()
        current_learning_rate = scheduled_learning_rate(update)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        if first_learning_rate is None:
            first_learning_rate = current_learning_rate
        last_learning_rate = current_learning_rate
        optimizer.zero_grad(set_to_none=True)
        stage_phase_timing["optimizer"] += time.perf_counter() - phase_start
        update_loss = 0.0
        for micro_step in range(accumulation):
            phase_start = time.perf_counter()
            batch = next(batches)
            stage_phase_timing["data_loading"] += time.perf_counter() - phase_start
            with gradient_sync_context(
                model,
                micro_step=micro_step,
                accumulation_steps=accumulation,
                world_size=world_size,
            ):
                phase_start = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    raw_loss = loss_function(model, batch)
                    loss = raw_loss / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at update {update}")
                update_loss += float(raw_loss.detach().cpu()) / accumulation
                stage_phase_timing["forward"] += time.perf_counter() - phase_start
                phase_start = time.perf_counter()
                scaler.scale(loss).backward()
                stage_phase_timing["backward"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        if strict_finite_gradients or config.gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    float("inf")
                    if config.gradient_clip_norm is None
                    else config.gradient_clip_norm
                ),
                error_if_nonfinite=False,
            )
            if bool(torch.isfinite(gradient_norm)):
                gradient_norm_value = float(gradient_norm.detach().cpu())
                max_preclip_gradient_norm = max(
                    max_preclip_gradient_norm, gradient_norm_value
                )
                if (
                    config.gradient_clip_norm is not None
                    and gradient_norm_value > config.gradient_clip_norm
                ):
                    gradient_clip_applications += 1
            if not bool(torch.isfinite(gradient_norm)):
                if strict_finite_gradients:
                    raise FloatingPointError(f"non-finite gradient at update {update}")
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if scale_after < scale_before:
            loss_scale_backoffs += 1
            consecutive_loss_scale_backoffs += 1
            max_consecutive_loss_scale_backoffs = max(
                max_consecutive_loss_scale_backoffs,
                consecutive_loss_scale_backoffs,
            )
        else:
            consecutive_loss_scale_backoffs = 0
        if strict_finite_gradients:
            parameter_norms = [
                torch.linalg.vector_norm(parameter.detach().float()) for parameter in model.parameters()
            ]
            if not bool(torch.all(torch.isfinite(torch.stack(parameter_norms)))):
                raise FloatingPointError(f"non-finite model parameter at update {update}")
        if not np.isfinite(update_loss):
            raise FloatingPointError(f"non-finite aggregate loss at update {update}")
        stage_phase_timing["optimizer"] += time.perf_counter() - phase_start
        if first_update_loss is None:
            first_update_loss = update_loss
        last_update_loss = update_loss
        if update % config.validation_interval == 0 or update == stage_stop:
            phase_start = time.perf_counter()
            metric = float(validation(model, update))
            stage_phase_timing["validation"] += time.perf_counter() - phase_start
            validation_calls += 1
            if not np.isfinite(metric):
                raise FloatingPointError(f"non-finite validation metric at update {update}")
            validation_history.append({"update": update, "metric": metric})
            if metric < best_metric:
                best_metric = metric
                best_update = update
                if rank == 0:
                    torch.save(
                        {
                            "model": checkpoint_model_state(model),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "update": update,
                        },
                        best_path,
                    )
            if rank == 0:
                _atomic_json(
                    output / "progress.json",
                    {
                        "schema": "paper2.training-progress/v1",
                        "state": "running",
                        "current_update": update,
                        "current_validation_metric": metric,
                        "best_update": best_update,
                        "best_validation_metric": best_metric,
                        "current_learning_rate": current_learning_rate,
                        "gradient_clip_applications": gradient_clip_applications,
                        "max_preclip_gradient_norm": max_preclip_gradient_norm,
                        "wall_seconds": time.monotonic() - start,
                    },
                )
    last_path = output / "last.pt"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        device_properties = torch.cuda.get_device_properties(device)
        raw_device_uuid = getattr(device_properties, "uuid", None)
        if not raw_device_uuid:
            raise RuntimeError("CUDA runtime does not expose the GPU UUID required by G3")
        device_uuid: str | None = str(raw_device_uuid)
        total_memory_bytes: int | None = int(device_properties.total_memory)
        peak_allocated_bytes: int | None = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes: int | None = int(torch.cuda.max_memory_reserved(device))
    else:
        device_uuid = None
        total_memory_bytes = None
        peak_allocated_bytes = None
        peak_reserved_bytes = None
    objective_telemetry_record = None
    if objective_telemetry is not None:
        raw_objective_telemetry = (
            objective_telemetry() if callable(objective_telemetry) else objective_telemetry
        )
        if not isinstance(raw_objective_telemetry, Mapping):
            raise ValueError("objective telemetry provider must return a mapping")
        # Canonical JSON round-trip both snapshots mutable mappings and rejects
        # non-finite or non-serializable telemetry before checkpoint publication.
        objective_telemetry_record = json.loads(
            _canonical_bytes(dict(raw_objective_telemetry)).decode("utf-8")
        )
    local_telemetry = {
        "rank": rank,
        "device_uuid": device_uuid,
        "wall_seconds": time.monotonic() - start,
        "continuous_updates_completed": stage_stop - start_update,
        "loss_scale_backoff_count": loss_scale_backoffs,
        "max_consecutive_loss_scale_backoffs": max_consecutive_loss_scale_backoffs,
        "gradient_clip_norm": config.gradient_clip_norm,
        "gradient_clip_applications": gradient_clip_applications,
        "max_preclip_gradient_norm": max_preclip_gradient_norm,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "total_memory_bytes": total_memory_bytes,
        "oom_detected": False,
        "nonfinite_detected": False,
        "phase_timing_seconds": dict(stage_phase_timing),
    }
    if objective_telemetry_record is not None:
        local_telemetry["objective_telemetry"] = objective_telemetry_record
    local_rng_state = {
        "rank": rank,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    rank_telemetry: list[dict[str, Any] | None] = [None] * world_size
    rank_rng_states: list[dict[str, Any] | None] = [None] * world_size
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.all_gather_object(rank_telemetry, local_telemetry)
        torch.distributed.all_gather_object(rank_rng_states, local_rng_state)
        if rank == 0:
            torch.save(
                {
                    "model": checkpoint_model_state(model),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "update": stage_stop,
                    "rank_rng_states": rank_rng_states,
                },
                last_path,
            )
        torch.distributed.barrier()
    else:
        rank_telemetry[0] = local_telemetry
        rank_rng_states[0] = local_rng_state
        torch.save(
            {
                "model": checkpoint_model_state(model),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "update": stage_stop,
                "rank_rng_states": rank_rng_states,
            },
            last_path,
        )
    if rank != 0:
        return {
            "schema_version": 1,
            "status": "worker_complete",
            "rank": rank,
            "start_update": start_update,
            "completed_updates": stage_stop,
        }
    if best_update is None:
        raise AssertionError("training completed without selecting a validation checkpoint")
    stage_phase_timing_critical = {
        key: max(
            float(item["phase_timing_seconds"][key])
            for item in rank_telemetry
            if item is not None
        )
        for key in PHASE_TIMING_KEYS
    }
    parent_phase_timing = (
        None if resume_manifest is None else resume_manifest.get("phase_timing_seconds")
    )
    cumulative_phase_timing = combine_phase_timings(
        stage_phase_timing_critical,
        None if parent_phase_timing is None else dict(parent_phase_timing),
    )
    manifest = checkpoint_manifest(
        config=config,
        model_config=model_config,
        split_hashes=split_hashes,
        code_hash=code_hash,
        environment_hash=environment_hash,
        update=best_update,
        completed_updates=stage_stop,
        validation_metric=best_metric,
        validation_history=validation_history,
        checkpoint_path=best_path,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        status="complete" if stage_stop == config.updates else "running",
        world_size=world_size,
    )
    manifest["last_checkpoint_sha256"] = _file_sha256(last_path)
    stage_wall_seconds = max(
        float(item["wall_seconds"]) for item in rank_telemetry if item is not None
    )
    manifest["wall_seconds"] = parent_wall_seconds + stage_wall_seconds
    manifest["stage_wall_seconds"] = stage_wall_seconds
    manifest["stage"] = {
        "start_update": start_update,
        "stop_update": stage_stop,
        "continuous_updates_completed": stage_stop - start_update,
        "evaluate_at_start": bool(evaluate_at_start),
        "strict_finite_gradients": bool(strict_finite_gradients),
        "batch_resume_protocol": batch_resume_protocol,
        "batch_resume_offset": batch_offset,
        "batch_cache_protocol": str(
            getattr(train_batches, "cache_protocol", "external-iterable/v1")
        ),
        "batch_cache_sample_count": int(
            getattr(train_batches, "cached_sample_count", 0)
        ),
        "batch_cache_bytes": int(getattr(train_batches, "cached_bytes", 0)),
    }
    manifest["resume_history_mode"] = (
        "cumulative" if preserve_validation_history_on_resume else "stage_local"
    )
    manifest["metrics"] = {
        "initial_validation_metric": initial_validation_metric,
        "last_validation_metric": (
            None if not validation_history else float(validation_history[-1]["metric"])
        ),
        "first_update_loss": first_update_loss,
        "last_update_loss": last_update_loss,
        "validation_calls": (
            len(validation_history)
            if preserve_validation_history_on_resume
            else validation_calls
        ),
        "first_learning_rate": first_learning_rate,
        "last_learning_rate": last_learning_rate,
    }
    if initialization is not None:
        manifest["initialization"] = json.loads(
            _canonical_bytes(dict(initialization)).decode("utf-8")
        )
    manifest["telemetry"] = {
        "oom_detected": False,
        "nonfinite_detected": False,
        "ranks": [item for item in rank_telemetry if item is not None],
    }
    manifest["stage_phase_timing_seconds"] = stage_phase_timing_critical
    manifest["phase_timing_seconds"] = cumulative_phase_timing
    manifest["phase_timing_clock"] = "host_perf_counter_cumulative_measured/v1"
    manifest["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _atomic_json(output / "checkpoint-manifest.json", manifest)
    _atomic_json(
        output / "progress.json",
        {
            "schema": "paper2.training-progress/v1",
            "state": "complete",
            "current_update": stage_stop,
            "current_validation_metric": float(validation_history[-1]["metric"]),
            "best_update": best_update,
            "best_validation_metric": best_metric,
            "current_learning_rate": last_learning_rate,
            "gradient_clip_applications": gradient_clip_applications,
            "max_preclip_gradient_norm": max_preclip_gradient_norm,
            "wall_seconds": manifest["wall_seconds"],
            "checkpoint_manifest_sha256": manifest["manifest_sha256"],
        },
    )
    return manifest
