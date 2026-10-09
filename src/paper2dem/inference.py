from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os
import tempfile
import time

import numpy as np


def patch_origins(length: int, patch_size: int = 256, overlap: float = 0.25) -> tuple[int, ...]:
    if length <= 0 or patch_size <= 0 or patch_size > length:
        raise ValueError("length and patch size must define at least one valid patch")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must lie in [0,1)")
    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    origins = list(range(0, max(length - patch_size, 0) + 1, stride))
    final = length - patch_size
    if not origins or origins[-1] != final:
        origins.append(final)
    return tuple(origins)


def patch_slices(
    shape: tuple[int, int], patch_size: int = 256, overlap: float = 0.25
) -> tuple[tuple[slice, slice], ...]:
    rows = patch_origins(shape[0], patch_size, overlap)
    columns = patch_origins(shape[1], patch_size, overlap)
    return tuple(
        (slice(row, row + patch_size), slice(column, column + patch_size))
        for row in rows
        for column in columns
    )


def blending_window(patch_size: int, *, kind: str = "hann") -> np.ndarray:
    if patch_size <= 1:
        return np.ones((patch_size, patch_size), dtype=np.float64)
    if kind == "hann":
        one_dimensional = np.hanning(patch_size)
    elif kind == "uniform":
        one_dimensional = np.ones(patch_size, dtype=np.float64)
    elif kind == "tukey-0.5":
        from scipy.signal.windows import tukey

        one_dimensional = tukey(patch_size, alpha=0.5)
    else:
        raise ValueError(f"unsupported locked stitching window: {kind}")
    # Do not leave global boundaries with zero support.
    one_dimensional = np.maximum(one_dimensional, 1.0e-3)
    return np.outer(one_dimensional, one_dimensional)


def stitch_patches(
    patches: Sequence[np.ndarray],
    slices: Sequence[tuple[slice, slice]],
    output_shape: tuple[int, int],
    *,
    window_kind: str = "hann",
) -> np.ndarray:
    if len(patches) != len(slices) or not patches:
        raise ValueError("patches and slices must be non-empty and one-to-one")
    first = np.asarray(patches[0], dtype=np.float64)
    patch_shape = first.shape[-2:]
    if patch_shape[0] != patch_shape[1]:
        raise ValueError("formal inference uses square patches")
    window = blending_window(patch_shape[0], kind=window_kind)
    leading = first.shape[:-2]
    numerator = np.zeros((*leading, *output_shape), dtype=np.float64)
    denominator = np.zeros(output_shape, dtype=np.float64)
    for patch, (rows, columns) in zip(patches, slices):
        array = np.asarray(patch, dtype=np.float64)
        if array.shape != first.shape:
            raise ValueError("all patches must share shape")
        numerator[..., rows, columns] += array * window
        denominator[rows, columns] += window
    if np.any(denominator <= 0):
        raise ValueError("patch layout did not cover every output pixel")
    return numerator / denominator


@dataclass
class OnlineMoments:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(array)
            self.m2 = np.zeros_like(array)
        if array.shape != self.mean.shape:
            raise ValueError("online samples must share shape")
        self.count += 1
        delta = array - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (array - self.mean)

    @property
    def variance(self) -> np.ndarray:
        if self.mean is None or self.m2 is None:
            raise ValueError("no samples accumulated")
        if self.count < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.count - 1)


@dataclass(frozen=True)
class PathEnsembleResult:
    mean: np.ndarray
    variance: np.ndarray
    path_stack: np.ndarray | None = None


def infer_path_ensemble(
    *,
    tile_shape: tuple[int, int],
    path_count: int,
    patch_predictor: Callable[[tuple[slice, slice], int], np.ndarray],
    terminal_dc: Callable[[np.ndarray], np.ndarray],
    patch_size: int = 256,
    overlap: float = 0.25,
    window_kind: str = "hann",
    retain_path_stack: bool = False,
) -> PathEnsembleResult:
    """Generate complete paths and optionally retain their ordered tile stack."""

    if path_count <= 0:
        raise ValueError("path_count must be positive")
    layout = patch_slices(tile_shape, patch_size, overlap)
    moments = OnlineMoments()
    retained: list[np.ndarray] | None = [] if retain_path_stack else None
    for path_index in range(path_count):
        patches = [patch_predictor(location, path_index) for location in layout]
        uncorrected_tile = stitch_patches(
            patches, layout, tile_shape, window_kind=window_kind
        )
        corrected_tile = terminal_dc(uncorrected_tile)
        if np.asarray(corrected_tile).shape[-2:] != tile_shape:
            raise ValueError("terminal DC changed tile shape")
        # Formal predictions are persisted as float32.  E7 accumulation uses the
        # same values that enter its immutable path stack so derived prefixes can
        # be reproduced bit-for-bit from that stack.
        sample = np.asarray(corrected_tile, dtype=np.float32)
        moments.update(sample)
        if retained is not None:
            retained.append(np.array(sample, copy=True))
    if moments.mean is None:
        raise AssertionError("path loop did not execute")
    stack = np.stack(retained, axis=0) if retained is not None else None
    return PathEnsembleResult(
        mean=np.asarray(moments.mean, dtype=np.float64),
        variance=np.asarray(moments.variance, dtype=np.float64),
        path_stack=stack,
    )


def infer_path_ensemble_batched(
    *,
    tile_shape: tuple[int, int],
    path_count: int,
    path_batch_size: int,
    patch_predictor: Callable[[tuple[slice, slice], tuple[int, ...]], np.ndarray],
    terminal_dc: Callable[[np.ndarray], np.ndarray],
    patch_size: int = 256,
    overlap: float = 0.25,
    window_kind: str = "hann",
    retain_path_stack: bool = False,
) -> PathEnsembleResult:
    """Generate independent paths in batches while preserving path order.

    ``patch_predictor`` receives the exact ordered global path indices.  Terminal
    DC remains a full-tile, one-path operation, so batching changes only neural
    network execution and not the physical reconstruction contract.
    """

    if path_count <= 0 or path_batch_size <= 0:
        raise ValueError("path count and path batch size must be positive")
    layout = patch_slices(tile_shape, patch_size, overlap)
    moments = OnlineMoments()
    retained: list[np.ndarray] | None = [] if retain_path_stack else None
    for start in range(0, path_count, path_batch_size):
        indices = tuple(range(start, min(start + path_batch_size, path_count)))
        patches = []
        for location in layout:
            patch = np.asarray(patch_predictor(location, indices))
            expected_shape = (len(indices), patch_size, patch_size)
            if patch.shape != expected_shape:
                raise ValueError(
                    "batched patch predictor shape differs: "
                    f"observed={patch.shape}, expected={expected_shape}"
                )
            patches.append(patch)
        uncorrected = stitch_patches(
            patches, layout, tile_shape, window_kind=window_kind
        )
        if uncorrected.shape != (len(indices), *tile_shape):
            raise AssertionError("batched stitching changed path/tile shape")
        for batch_index, _ in enumerate(indices):
            corrected = terminal_dc(uncorrected[batch_index])
            if np.asarray(corrected).shape[-2:] != tile_shape:
                raise ValueError("terminal DC changed tile shape")
            sample = np.asarray(corrected, dtype=np.float32)
            moments.update(sample)
            if retained is not None:
                retained.append(np.array(sample, copy=True))
    if moments.mean is None:
        raise AssertionError("batched path loop did not execute")
    stack = np.stack(retained, axis=0) if retained is not None else None
    return PathEnsembleResult(
        mean=np.asarray(moments.mean, dtype=np.float64),
        variance=np.asarray(moments.variance, dtype=np.float64),
        path_stack=stack,
    )


def infer_paths(
    *,
    tile_shape: tuple[int, int],
    path_count: int,
    patch_predictor: Callable[[tuple[slice, slice], int], np.ndarray],
    terminal_dc: Callable[[np.ndarray], np.ndarray],
    patch_size: int = 256,
    overlap: float = 0.25,
    window_kind: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate/stitch each complete tile, then apply terminal DC exactly once.

    The ordering is a deliberate physical contract: patch-level DC would truncate
    footprints crossing patch boundaries and is therefore not exposed here.
    """

    result = infer_path_ensemble(
        tile_shape=tile_shape,
        path_count=path_count,
        patch_predictor=patch_predictor,
        terminal_dc=terminal_dc,
        patch_size=patch_size,
        overlap=overlap,
        window_kind=window_kind,
        retain_path_stack=False,
    )
    return result.mean, result.variance


INFERENCE_KEY_FIELDS = (
    "experiment",
    "tile_id",
    "method",
    "model_identity",
    "training_seed",
    "budget",
    "observation_id",
    "noise_replicate",
    "path_seed_set",
    "path_count",
    "step_count",
    "dc_mode",
)


def inference_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    missing = [field for field in INFERENCE_KEY_FIELDS if field not in record]
    if missing:
        raise ValueError(f"inference record lacks key fields: {', '.join(missing)}")
    return tuple(record[field] for field in INFERENCE_KEY_FIELDS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_inference_manifest(
    output: Path,
    *,
    key: Mapping[str, Any],
    plan_sha256: str,
    observation_sha256: str,
    checkpoint_sha256: str | None,
    result_files: Iterable[Path],
    wall_seconds: float,
    peak_gpu_memory_bytes: int | None,
    nfe: int,
    area_km2: float,
    retry_count: int = 0,
    status: str = "complete",
) -> dict[str, Any]:
    inference_key(key)
    files = {path.name: file_sha256(path) for path in sorted(result_files)}
    if not files:
        raise ValueError("inference manifest must bind at least one result file")
    payload = {
        "schema_version": 1,
        **dict(key),
        "plan_sha256": plan_sha256,
        "observation_sha256": observation_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "result_sha256": files,
        "wall_seconds": float(wall_seconds),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "nfe": int(nfe),
        "area_km2": float(area_km2),
        "seconds_per_km2": float(wall_seconds / area_km2) if area_km2 > 0 else None,
        "retry_count": int(retry_count),
        "status": status,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


class Timer:
    def __enter__(self) -> "Timer":
        self.started = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        self.wall_seconds = time.monotonic() - self.started
