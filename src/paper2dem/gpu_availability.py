"""Audited dual-GPU availability gate for the formal Paper2 training chain."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
import json
import subprocess
import time

from .hashing import canonical_sha256, file_sha256, guard_paper2_path, write_canonical_json


GPU_AVAILABILITY_SCHEMA = "paper2.dual-gpu-availability/v1"
DEFAULT_MAX_UTILIZATION_PERCENT = 10
DEFAULT_MAX_MEMORY_USED_MIB = 1024
DEFAULT_MAX_TEMPERATURE_C = 75
DEFAULT_REQUIRED_CONSECUTIVE_SAMPLES = 3
DEFAULT_POLL_SECONDS = 30.0
DEFAULT_MAXIMUM_WAIT_SECONDS = 7 * 24 * 60 * 60
NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.free,temperature.gpu",
    "--format=csv,noheader,nounits",
)


def _normalized_gpu_uuid(value: object) -> str:
    """Normalize the optional ``GPU-`` prefix used by NVIDIA's CLI.

    PyTorch exposes ``torch.cuda.get_device_properties(...).uuid`` without the
    prefix on some driver/runtime combinations, while ``nvidia-smi`` includes
    it for the same physical device.
    """

    text = str(value).strip()
    if text.lower().startswith("gpu-"):
        text = text[4:]
    text = text.lower()
    if not text:
        raise ValueError("GPU UUID is empty after normalization")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def expected_gpu_uuids(environment_lock: Path) -> tuple[str, str]:
    payload = _read_json(environment_lock)
    runtime = payload.get("runtime")
    devices = runtime.get("gpu_devices") if isinstance(runtime, Mapping) else None
    if (
        payload.get("schema") != "paper2.training-environment-lock/v1"
        or payload.get("state") != "ready"
        or payload.get("required_world_size") != 2
        or not isinstance(devices, list)
        or len(devices) != 2
    ):
        raise ValueError("training environment lock does not bind exactly two GPUs")
    ordered = sorted((dict(device) for device in devices), key=lambda item: int(item["index"]))
    if [int(item["index"]) for item in ordered] != [0, 1]:
        raise ValueError("training environment lock GPU indices differ from 0,1")
    uuids = tuple(str(item.get("uuid", "")) for item in ordered)
    if any(not value for value in uuids) or len(set(uuids)) != 2:
        raise ValueError("training environment lock GPU UUIDs are invalid")
    return uuids  # type: ignore[return-value]


def parse_nvidia_smi_snapshot(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 7:
            raise ValueError("nvidia-smi availability row must have seven fields")
        index, uuid, name, utilization, memory_used, memory_free, temperature = fields
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "utilization_gpu_percent": int(utilization),
                "memory_used_mib": int(memory_used),
                "memory_free_mib": int(memory_free),
                "temperature_c": int(temperature),
            }
        )
    rows.sort(key=lambda item: item["index"])
    if len(rows) != 2 or [item["index"] for item in rows] != [0, 1]:
        raise ValueError("formal availability gate requires exactly GPU indices 0 and 1")
    return rows[0], rows[1]


def sample_is_idle(
    devices: Sequence[Mapping[str, Any]],
    *,
    expected_uuids: Sequence[str],
    max_utilization_percent: int,
    max_memory_used_mib: int,
    max_temperature_c: int,
) -> bool:
    observed = tuple(_normalized_gpu_uuid(item["uuid"]) for item in devices)
    expected = tuple(_normalized_gpu_uuid(value) for value in expected_uuids)
    if len(devices) != 2 or observed != expected:
        raise ValueError("live GPU UUIDs differ from the training environment lock")
    return all(
        int(item["utilization_gpu_percent"]) <= max_utilization_percent
        and int(item["memory_used_mib"]) <= max_memory_used_mib
        and int(item["temperature_c"]) <= max_temperature_c
        for item in devices
    )


def wait_for_dual_gpu_idle(
    *,
    paper2_root: str | Path,
    environment_lock: str | Path,
    output: str | Path,
    max_utilization_percent: int = DEFAULT_MAX_UTILIZATION_PERCENT,
    max_memory_used_mib: int = DEFAULT_MAX_MEMORY_USED_MIB,
    max_temperature_c: int = DEFAULT_MAX_TEMPERATURE_C,
    required_consecutive_samples: int = DEFAULT_REQUIRED_CONSECUTIVE_SAMPLES,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    maximum_wait_seconds: float = DEFAULT_MAXIMUM_WAIT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    wall_time: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    environment_path = guard_paper2_path(environment_lock, paper2_root=root)
    output_path = guard_paper2_path(output, paper2_root=root)
    if output_path.exists():
        raise FileExistsError(f"GPU availability receipt must be new: {output_path}")
    if not environment_path.is_file():
        raise FileNotFoundError(f"training environment lock is missing: {environment_path}")
    if (
        max_utilization_percent < 0
        or max_memory_used_mib < 0
        or max_temperature_c <= 0
        or required_consecutive_samples < 1
        or poll_seconds <= 0
        or maximum_wait_seconds <= 0
    ):
        raise ValueError("GPU availability thresholds and intervals must be positive")

    uuids = expected_gpu_uuids(environment_path)
    thresholds = {
        "max_utilization_gpu_percent": int(max_utilization_percent),
        "max_memory_used_mib": int(max_memory_used_mib),
        "max_temperature_c": int(max_temperature_c),
        "required_consecutive_samples": int(required_consecutive_samples),
        "poll_seconds": float(poll_seconds),
        "maximum_wait_seconds": float(maximum_wait_seconds),
    }
    started_monotonic = monotonic()
    started_unix = wall_time()
    consecutive = 0
    samples: list[dict[str, Any]] = []
    state = "waiting"
    while True:
        sampled_unix = wall_time()
        try:
            completed = runner(
                NVIDIA_SMI_QUERY,
                check=False,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"nvidia-smi returned {completed.returncode}: {completed.stderr[-500:]}"
                )
            devices = parse_nvidia_smi_snapshot(completed.stdout)
            idle = sample_is_idle(
                devices,
                expected_uuids=uuids,
                max_utilization_percent=max_utilization_percent,
                max_memory_used_mib=max_memory_used_mib,
                max_temperature_c=max_temperature_c,
            )
            consecutive = consecutive + 1 if idle else 0
            sample: dict[str, Any] = {
                "sampled_unix_seconds": sampled_unix,
                "devices": [dict(item) for item in devices],
                "idle": idle,
                "consecutive_idle_samples": consecutive,
            }
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            consecutive = 0
            sample = {
                "sampled_unix_seconds": sampled_unix,
                "devices": [],
                "idle": False,
                "consecutive_idle_samples": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
            }
        samples.append(sample)
        elapsed = monotonic() - started_monotonic
        if consecutive >= required_consecutive_samples:
            state = "ready"
            break
        if elapsed >= maximum_wait_seconds:
            state = "timeout"
            break
        sleeper(min(poll_seconds, maximum_wait_seconds - elapsed))

    payload: dict[str, Any] = {
        "schema": GPU_AVAILABILITY_SCHEMA,
        "state": state,
        "training_authorized": state == "ready",
        "interfering_processes_terminated": False,
        "paper2_root": ".",
        "environment_lock_relative_path": environment_path.relative_to(root).as_posix(),
        "environment_lock_sha256": file_sha256(environment_path),
        "expected_gpu_uuids": list(uuids),
        "nvidia_smi_query": list(NVIDIA_SMI_QUERY),
        "thresholds": thresholds,
        "started_unix_seconds": started_unix,
        "finished_unix_seconds": wall_time(),
        "elapsed_seconds": monotonic() - started_monotonic,
        "samples": samples,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    write_canonical_json(output_path, payload, paper2_root=root)
    return payload
